"""
Fine-tune MLX Whisper for IPA transcription.

This script trains only the decoder while keeping the encoder frozen,
as the encoder already understands audio features well.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_whisper.load_models import load_model
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from ipa_data_loader import create_data_loader
from evaluate_ipa import evaluate_batch
from mlx_whisper.decoding import DecodingOptions
from mlx.core import save_safetensors


def count_parameters(params_dict):
    """Recursively count parameters in nested dictionary."""
    total = 0
    for value in params_dict.values():
        if isinstance(value, dict):
            total += count_parameters(value)
        elif hasattr(value, 'size'):
            total += value.size
    return total


def flatten_params(params, prefix=""):
    """Flatten nested dictionary/list of parameters."""
    flat_params = {}
    if isinstance(params, dict):
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            flat_params.update(flatten_params(v, key))
    elif isinstance(params, list):
        for i, v in enumerate(params):
            key = f"{prefix}.{i}" if prefix else str(i)
            flat_params.update(flatten_params(v, key))
    else:
        if prefix:
            flat_params[prefix] = params
    return flat_params


def freeze_encoder(model):
    """Freeze the encoder parameters, only train decoder."""
    print("\nFreezing encoder parameters...")

    # Freeze encoder
    if hasattr(model, 'encoder'):
        model.encoder.freeze()
        print("  ✓ Encoder frozen")
    else:
        print("  ⚠ Warning: Could not find encoder to freeze")

    # Unfreeze decoder
    if hasattr(model, 'decoder'):
        model.decoder.unfreeze()
        print("  ✓ Decoder unfrozen (trainable)")
    else:
        print("  ⚠ Warning: Could not find decoder to unfreeze")

    # Count trainable parameters
    trainable_params = count_parameters(model.trainable_parameters())
    total_params = count_parameters(model.parameters())

    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")


def compute_loss(model, batch: Dict, tokenizer) -> mx.array:
    """
    Compute cross-entropy loss for the batch.

    Args:
        model: Whisper model
        batch: Dictionary with 'mel_features' and 'tokens'
        tokenizer: Whisper tokenizer (for EOT token ID)

    Returns:
        Loss value
    """
    mel_features = batch['mel_features']  # (batch_size, n_frames, n_mels)
    tokens = batch['tokens']              # (batch_size, seq_len)

    # First, encode audio
    audio_features = model.embed_audio(mel_features)  # Encoder forward pass

    # Prepare decoder input and target
    # Input: all tokens except last (for teacher forcing)
    # Target: all tokens except first (what we predict)
    decoder_input = tokens[:, :-1]  # Remove last token
    target_tokens = tokens[:, 1:]   # Remove first token (shift left)

    # Decode
    logits = model.logits(decoder_input, audio_features)  # Decoder forward pass

    # Compute cross-entropy loss
    # logits shape: (batch_size, seq_len-1, vocab_size)
    # target_tokens shape: (batch_size, seq_len-1)

    # Create mask to ignore padding tokens (EOT used for padding)
    # We want to compute loss only on non-padding tokens
    # CRITICAL FIX: We must include the FIRST EOT token so the model learns to stop.
    # Logic: Keep if (token != EOT) OR (it is the first EOT)
    is_eot = target_tokens == tokenizer.eot
    eot_cumsum = mx.cumsum(is_eot, axis=1)
    
    # Mask is True for tokens we want to keep (loss computed)
    # Keep if NOT EOT, or if it IS EOT but it's the first one (cumsum == 1)
    mask = (target_tokens != tokenizer.eot) | (eot_cumsum == 1)

    # Flatten for cross-entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    target_flat = target_tokens.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Compute loss only on non-padding tokens
    loss = nn.losses.cross_entropy(logits_flat, target_flat, reduction='none')
    masked_loss = mx.where(mask_flat, loss, mx.zeros_like(loss))

    # Average over non-padding tokens
    num_valid = mx.sum(mask_flat)
    total_loss = mx.sum(masked_loss) / mx.maximum(num_valid, 1)  # Avoid division by zero

    return total_loss


def train_step(model, optimizer, batch: Dict, tokenizer, max_grad_norm: float = 1.0) -> Tuple[mx.array, mx.array]:
    """
    Single training step.

    Args:
        model: Whisper model
        optimizer: MLX optimizer
        batch: Training batch
        tokenizer: Whisper tokenizer
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        (loss, gradients)
    """
    def loss_fn(model):
        return compute_loss(model, batch, tokenizer)

    # Compute loss and gradients
    loss, grads = nn.value_and_grad(model, loss_fn)(model)

    # Clip gradients to prevent explosion
    def clip_grad_dict(grad_dict, max_norm):
        """Recursively clip gradients in nested dictionary."""
        clipped = {}
        for key, value in grad_dict.items():
            if isinstance(value, dict):
                clipped[key] = clip_grad_dict(value, max_norm)
            elif hasattr(value, 'shape'):
                # Clip this gradient
                grad_norm = mx.sqrt(mx.sum(value * value))
                clip_coef = max_norm / (grad_norm + 1e-6)
                clip_coef = mx.minimum(clip_coef, 1.0)
                clipped[key] = value * clip_coef
            else:
                clipped[key] = value
        return clipped

    grads = clip_grad_dict(grads, max_grad_norm)

    # Update parameters
    optimizer.update(model, grads)

    # Force computation
    mx.eval(loss)

    return loss, grads


def validate(model, dataset, tokenizer, num_samples: int = 100) -> Dict:
    """
    Validate model on a subset of samples using PER and PFER.

    Args:
        model: Whisper model
        dataset: Test dataset
        tokenizer: Whisper tokenizer
        num_samples: Number of samples to validate on

    Returns:
        Dictionary with validation metrics
    """
    print(f"\nValidating on {num_samples} samples...")
    model.eval()  # Set to evaluation mode

    references = []
    hypotheses = []
    
    # Create a small batch size for validation decoding (decoding is memory intensive)
    val_batch_size = 4
    num_batches = (num_samples + val_batch_size - 1) // val_batch_size
    
    # Decoding options
    decode_options = DecodingOptions(
        language=None, # Multilingual
        without_timestamps=True,
        fp16=False, # Use float32 for safety
        length_penalty=1.0, # Penalize longer sequences slightly to discourage loops
    )

    for i in range(num_batches):
        indices = list(range(i * val_batch_size, min((i + 1) * val_batch_size, num_samples)))
        if not indices:
            break
            
        batch = dataset.get_batch(indices)
        mel = batch['mel_features']
        
        # Decode
        # model.decode expects mel features
        try:
            results = model.decode(mel, decode_options)
            
            # Handle single result vs list
            if not isinstance(results, list):
                results = [results]
                
            batch_hypotheses = [r.text.strip() for r in results]
            
            # Get references (we need to decode tokens back to text, or use the raw text if available)
            # The dataset loader might not expose raw text easily in get_batch, 
            # but we can decode the tokens.
            batch_tokens = batch['tokens']
            batch_references = []
            for j in range(len(indices)):
                # Decode tokens
                # We manually strip special tokens that look like <|...|>
                ref_text = tokenizer.decode(batch_tokens[j].tolist())
                
                # Simple cleanup of special tokens
                # Whisper special tokens are usually <|...|>
                # We can use a regex or just rely on the fact that they are usually at the start/end
                # But let's be robust.
                import re
                ref_text = re.sub(r'<\|.*?\|>', '', ref_text)
                batch_references.append(ref_text.strip())
            
            references.extend(batch_references)
            hypotheses.extend(batch_hypotheses)
            
            # Visual logging for the first batch
            if i == 0:
                print("\nSample Predictions:")
                for k in range(min(3, len(batch_references))):
                    print(f"  Ref:  [{batch_references[k]}]")
                    print(f"  Pred: [{batch_hypotheses[k]}]")
                    print("-" * 30)
                    
        except Exception as e:
            print(f"Error during validation decoding: {e}")
            import traceback
            traceback.print_exc()

    # Calculate metrics
    metrics = evaluate_batch(references, hypotheses)
    
    model.train()  # Back to training mode
    
    print(f"Validation Results:")
    print(f"  PER:  {metrics['per']:.2f}%")
    print(f"  PFER: {metrics['pfer']:.2f}%")
    
    return metrics


def save_checkpoint(model, optimizer, step: int, loss, output_dir: Path):
    """Save model checkpoint."""
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights using safetensors to avoid mx.savez limit (>1024 args)
    # We flatten the parameters dictionary for saving
    weights = flatten_params(model.parameters())
    save_safetensors(str(checkpoint_dir / "model.safetensors"), weights)

    # Save training state (basic info only, optimizer state is complex)
    state = {
        'step': step,
        'loss': float(loss.item()) if hasattr(loss, 'item') else float(loss),
    }

    with open(checkpoint_dir / "training_state.json", 'w') as f:
        json.dump(state, f, indent=2)

    print(f"  ✓ Saved checkpoint to {checkpoint_dir}")


def train(
    model_name: str,
    train_data_path: str,
    test_data_path: str,
    output_dir: str,
    num_steps: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    validate_every: int = 100,
    save_every: int = 500,
    test_run: bool = False,
):
    """
    Main training function.

    Args:
        model_name: MLX Whisper model name
        train_data_path: Path to training JSON
        test_data_path: Path to test JSON
        output_dir: Directory to save checkpoints
        num_steps: Number of training steps
        batch_size: Batch size
        learning_rate: Learning rate
        validate_every: Validate every N steps
        save_every: Save checkpoint every N steps
        test_run: If True, use only 100 samples for quick testing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Fine-tuning Whisper for IPA Transcription")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {model_name}")
    import time as time_module
    start = time_module.time()
    model = load_model(model_name)
    print(f"  ✓ Model loaded in {time_module.time() - start:.1f}s")

    # Convert to float32 to avoid numerical instability
    print("  Converting to float32...")
    start = time_module.time()
    model.set_dtype(mx.float32)
    print(f"  ✓ Converted in {time_module.time() - start:.1f}s")

    # Freeze encoder (decoder-only training)
    freeze_encoder(model)

    # Setup optimizer
    print(f"\nSetting up AdamW optimizer (lr={learning_rate})")
    optimizer = optim.AdamW(learning_rate=learning_rate)

    # Detect model size and set correct n_mels
    # whisper-large uses 128 mel bins, small/medium use 80
    n_mels = 128 if 'large' in model_name.lower() else 80
    print(f"  Using n_mels={n_mels} for model size")

    # Load datasets
    print(f"\nLoading training data: {train_data_path}")
    train_dataset = create_data_loader(train_data_path, multilingual=True, n_mels=n_mels)

    print(f"Loading test data: {test_data_path}")
    test_dataset = create_data_loader(test_data_path, multilingual=True, n_mels=n_mels)

    # Get tokenizer (we can use it from the dataset)
    tokenizer = train_dataset.tokenizer

    # Test run: limit to 100 samples
    if test_run:
        print("\n⚠ TEST RUN MODE: Using only 100 training samples")
        train_dataset.data = train_dataset.data[:100]
        num_steps = min(num_steps, 100)

    # Training loop
    print("\n" + "=" * 70)
    print(f"Starting training for {num_steps} steps")
    print("=" * 70)

    model.train()  # Set to training mode
    start_time = time.time()
    latest_loss = None

    for step in range(1, num_steps + 1):
        try:
            # Sample random batch
            batch_indices = np.random.choice(len(train_dataset), size=batch_size, replace=False)
            batch = train_dataset.get_batch(batch_indices.tolist())

            # Training step
            step_start = time.time()
            loss, grads = train_step(model, optimizer, batch, tokenizer)
            latest_loss = loss  # Save for final checkpoint
            step_time = time.time() - step_start

            # Log
            if step % 10 == 0 or step <= 5:  # Always log first 5 steps
                print(f"Step {step}/{num_steps} | Loss: {loss.item():.4f} | "
                      f"Time: {step_time:.3f}s | "
                      f"Samples/sec: {batch_size / step_time:.1f}")

            # Validate
            if step % validate_every == 0:
                metrics = validate(model, test_dataset, tokenizer, num_samples=100)
                val_per = metrics['per']
                val_pfer = metrics['pfer']

            # Save checkpoint
            if step % save_every == 0:
                save_checkpoint(model, optimizer, step, loss, output_dir)

        except Exception as e:
            print(f"\n✗ Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Final validation
    print("\n" + "=" * 70)
    print("Training complete! Running final validation...")
    print("=" * 70)
    metrics = validate(model, test_dataset, tokenizer, num_samples=min(500, len(test_dataset)))
    val_per = metrics['per']
    val_pfer = metrics['pfer']

    # Save final model
    if latest_loss is not None:
        print("\nSaving final model...")
        save_checkpoint(model, optimizer, num_steps, latest_loss, output_dir)

        total_time = time.time() - start_time
        print(f"\n✓ Training complete in {total_time / 60:.1f} minutes")
        print(f"  Final loss: {latest_loss.item():.4f}")
        print(f"  Final PER: {val_per:.2f}%")
        print(f"  Final PFER: {val_pfer:.2f}%")
        print(f"  Model saved to: {output_dir}")
    else:
        print("\n✗ Training failed - no loss computed")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for IPA transcription")

    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/whisper-small-mlx",
        help="MLX Whisper model name (default: whisper-small-mlx)"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/english_only_train_ipa.json",
        help="Path to training JSON file"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/english_only_test_ipa.json",
        help="Path to test JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/whisper-ipa",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Number of training steps (approx 10 epochs)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1000,
        help="Validate every N steps (approx 1 epoch)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (approx 1 epoch)"
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run with only 100 samples"
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        validate_every=args.validate_every,
        save_every=args.save_every,
        test_run=args.test_run,
    )


if __name__ == "__main__":
    main()
