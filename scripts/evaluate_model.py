"""
Evaluate trained Whisper model on test set with PER and PFER metrics.

Now that we have Python 3.13, both PER and PFER are fully supported!
"""

import sys
sys.path.insert(0, 'scripts')

import json
import mlx.core as mx
from pathlib import Path
from tqdm import tqdm
import mlx_whisper
from mlx_whisper.load_models import load_model

from evaluate_ipa import phone_error_rate, phone_feature_error_rate, evaluate_batch


def load_checkpoint_model(checkpoint_path: str, base_model: str = "mlx-community/whisper-small-mlx"):
    """
    Load a trained checkpoint by loading weights into base model.

    Args:
        checkpoint_path: Path to checkpoint directory containing model.npz
        base_model: Base model to load architecture from

    Returns:
        Loaded model with trained weights
    """
    from mlx.utils import tree_flatten, tree_unflatten

    print(f"Loading base model architecture: {base_model}")
    model = load_model(base_model)

    # Convert model to float32 to match training dtype
    model.set_dtype(mx.float32)

    # Load trained weights (only decoder - encoder was frozen during training)
    # Try safetensors first (new format), then npz (old format)
    weights_path = Path(checkpoint_path) / "model.safetensors"
    if weights_path.exists():
        print(f"Loading trained weights from: {weights_path}")
        trained_weights = mx.load(str(weights_path))
    else:
        weights_path = Path(checkpoint_path) / "model.npz"
        if weights_path.exists():
            print(f"Loading trained weights from: {weights_path}")
            trained_weights = mx.load(str(weights_path))
        else:
            print(f"WARNING: No weights found at {checkpoint_path}, using base model")
            return model

    if True: # Indent block to match previous logic structure or just proceed
        # Filter to only decoder weights (encoder was frozen)

        # Filter to only decoder weights (encoder was frozen)
        decoder_weights = {k: v for k, v in trained_weights.items() if k.startswith('decoder.')}
        print(f"Found {len(decoder_weights)} decoder parameters to load")

        # Get current model parameters
        current_params = tree_flatten(model.parameters())
        current_params_dict = dict(current_params)

        # Update only decoder parameters
        for k, v in decoder_weights.items():
            if k in current_params_dict:
                current_params_dict[k] = v

        # Reconstruct parameter tree
        updated_params = tree_unflatten(list(current_params_dict.items()))
        model.update(updated_params)
        mx.eval(model.parameters())

        print("✓ Decoder weights loaded successfully")
    else:
        print(f"WARNING: No weights found at {weights_path}, using base model")

    return model


def transcribe_with_model(model_path: str, audio_path: str, language: str = "en", is_checkpoint: bool = False) -> str:
    """
    Transcribe audio file to IPA using Whisper model.

    Args:
        model_path: Path to model checkpoint or HuggingFace model name
        audio_path: Path to audio file
        language: Language code (default: "en")
        is_checkpoint: If True, load as checkpoint; otherwise use as HF model name

    Returns:
        IPA transcription string
    """
    try:
        if is_checkpoint:
            # For checkpoints, we need to load the model and use mlx_whisper.transcribe
            # But mlx_whisper.transcribe doesn't accept a model object directly
            # We'll need to use a different approach
            model = load_checkpoint_model(model_path)

            # Use mlx_whisper's transcribe with the loaded model
            # Unfortunately mlx_whisper.transcribe doesn't accept model object
            # We need to use it differently
            import mlx_whisper.transcribe as transcribe_module
            result = transcribe_module.transcribe(
                audio_path,
                path_or_hf_repo=model_path.replace("/checkpoint-", "/base-"),  # Hack: won't work
                language=language,
                word_timestamps=False,
            )
        else:
            # Use mlx_whisper's transcribe function for HF models
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=model_path,
                language=language,
                word_timestamps=False,
            )
        text = result['text'].strip()
        return text
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""


def evaluate_model(model_path: str, test_data_path: str, num_samples: int = None,
                   model_name: str = "Model", is_checkpoint: bool = False, n_mels: int = 80,
                   base_model: str = "mlx-community/whisper-small-mlx"):
    """
    Evaluate a model on test data.

    Args:
        model_path: Path to model checkpoint directory or HuggingFace model name
        test_data_path: Path to test JSON file
        num_samples: Number of samples to evaluate (None = all)
        model_name: Name to display for this model
        is_checkpoint: Whether this is a checkpoint (needs special loading)

    Returns:
        Dictionary with evaluation results
    """
    print("=" * 70)
    print(f"Evaluating {model_name}")
    print("=" * 70)

    # Load test dataset
    print(f"\nLoading test data: {test_data_path}")
    with open(test_data_path) as f:
        test_data = json.load(f)

    if num_samples:
        test_data = test_data[:num_samples]
        print(f"Evaluating on {num_samples} samples")
    else:
        print(f"Evaluating on all {len(test_data)} samples")

    print(f"\nModel: {model_path}")

    # Load model if it's a checkpoint
    if is_checkpoint:
        print("\nLoading checkpoint...")
        model = load_checkpoint_model(model_path, base_model=base_model)

        # We need to use the model directly for inference
        # Let's use mlx_whisper's internal functions
        from mlx_whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
        from mlx_whisper.decoding import DecodingOptions, decode

        decode_options = DecodingOptions(
            language="en",
            without_timestamps=True,
        )

    references = []
    hypotheses = []

    print("\nTranscribing test samples...")
    for i, sample in enumerate(tqdm(test_data)):
        audio_path = sample['audio_path']
        reference_ipa = sample['ipa_transcription']

        # Transcribe
        if is_checkpoint:
            try:
                # Load and preprocess audio
                audio = load_audio(audio_path)
                audio = pad_or_trim(audio)
                mel = log_mel_spectrogram(audio, n_mels=n_mels)
                mel = mx.expand_dims(mel, 0)
                # Ensure mel is float32 to match our model
                mel = mel.astype(mx.float32)

                # Encode audio features
                # The encoder.encode() has a dtype check, so we need to bypass it
                # by directly calling the encoder without the check
                audio_features = model.encoder(mel)

                # Decode using decoder
                result = decode(model, audio_features, decode_options)
                hypothesis_ipa = result[0].text.strip()
            except Exception as e:
                print(f"\nError transcribing {audio_path}: {e}")
                hypothesis_ipa = ""
        else:
            hypothesis_ipa = transcribe_with_model(model_path, audio_path, is_checkpoint=False)

        references.append(reference_ipa)
        hypotheses.append(hypothesis_ipa)

        # Show first few examples
        if i < 3:
            per = phone_error_rate(reference_ipa, hypothesis_ipa)
            pfer = phone_feature_error_rate(reference_ipa, hypothesis_ipa)
            print(f"\nSample {i + 1}:")
            print(f"  Reference:  {reference_ipa}")
            print(f"  Hypothesis: {hypothesis_ipa}")
            print(f"  PER:  {per:.2f}%")
            print(f"  PFER: {pfer:.2f}%")

    # Calculate overall metrics
    print("\n" + "=" * 70)
    print(f"{model_name} - Overall Results")
    print("=" * 70)

    results = evaluate_batch(references, hypotheses)

    print(f"\nPER (Phone Error Rate):         {results['per']:.2f}% (±{results['per_std']:.2f}%)")
    print(f"PFER (Phone Feature Error Rate): {results['pfer']:.2f}% (±{results['pfer_std']:.2f}%)")
    print(f"Number of samples: {results['num_samples']}")

    return results


def compare_models(base_results: dict, trained_results: dict):
    """Print comparison between base and trained models."""
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("n" + "=" * 70)

    print(f"\n{'Metric':<30} {'Base Model':<15} {'Trained Model':<15} {'Improvement':<15}")
    print("-" * 70)

    per_diff = base_results['per'] - trained_results['per']
    pfer_diff = base_results['pfer'] - trained_results['pfer']

    print(f"{'PER (Phone Error Rate)':<30} {base_results['per']:>6.2f}%{'':<8} {trained_results['per']:>6.2f}%{'':<8} {per_diff:>+6.2f}%")
    print(f"{'PFER (Feature Error Rate)':<30} {base_results['pfer']:>6.2f}%{'':<8} {trained_results['pfer']:>6.2f}%{'':<8} {pfer_diff:>+6.2f}%")

    print("\n" + "=" * 70)
    print("Benchmark Comparison (from paper)")
    print("=" * 70)
    print("Target scores (zero-shot, unseen languages):")
    print("  - Best in paper (1k samples): 21.2% PFER")
    print("  - Wav2Vec2Phoneme: 22.4% PFER")
    print("  - Human IAA: 19.6% PFER")
    print("\nTarget scores (supervised, trained languages):")
    print("  - Overall: 5.7% PFER")
    print("  - Polish (best): 2.5% PFER")

    if trained_results['pfer'] < 50:
        print("\n✅ MINIMUM VIABLE: PFER < 50% achieved!")
    if trained_results['pfer'] < 30:
        print("✅ GOOD: PFER < 30% achieved!")
    if trained_results['pfer'] < 25:
        print("✅ EXCELLENT: PFER < 25% achieved!")
    if trained_results['pfer'] < 21.2:
        print("🎉 SOTA: Beat paper's best zero-shot result!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Whisper-IPA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/whisper-ipa-english/checkpoint-250",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="mlx-community/whisper-small-mlx",
        help="Base model for comparison"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/english_only_test_ipa.json",
        help="Path to test data JSON"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100, use 0 for all)"
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base model evaluation (only evaluate checkpoint)"
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel bins (80 for small/medium, 128 for large)"
    )

    args = parser.parse_args()

    num_samples = None if args.num_samples == 0 else args.num_samples

    # Evaluate base model
    if not args.skip_base:
        base_results = evaluate_model(
            args.base_model,
            args.test_data,
            num_samples,
            model_name="Base Whisper Model",
            is_checkpoint=False,
            n_mels=args.n_mels,
            base_model=args.base_model
        )
    else:
        base_results = None

    # Evaluate trained checkpoint
    trained_results = evaluate_model(
        args.checkpoint,
        args.test_data,
        num_samples,
        model_name="Trained Checkpoint",
        is_checkpoint=True,
        n_mels=args.n_mels,
        base_model=args.base_model
    )

    # Compare if we have both
    if base_results:
        compare_models(base_results, trained_results)

    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)
