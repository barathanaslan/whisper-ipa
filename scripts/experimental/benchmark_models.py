"""
Benchmark different Whisper model sizes for training feasibility.

Tests memory usage, training speed, and inference speed for:
- whisper-tiny
- whisper-small (current)
- whisper-medium
- whisper-large-v3 (target)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import mlx.core as mx
from mlx_whisper.load_models import load_model
from mlx.utils import tree_flatten
from ipa_data_loader import create_data_loader


def get_model_info(model):
    """Get model parameter counts and memory usage."""
    params = tree_flatten(model.parameters())
    total_params = sum(p.size for _, p in params)

    # Count encoder vs decoder params
    encoder_params = sum(p.size for k, p in params if k.startswith('encoder.'))
    decoder_params = sum(p.size for k, p in params if k.startswith('decoder.'))

    # Estimate memory (assuming float32 = 4 bytes per parameter)
    total_memory_mb = (total_params * 4) / (1024 * 1024)
    decoder_memory_mb = (decoder_params * 4) / (1024 * 1024)

    return {
        'total_params': total_params,
        'encoder_params': encoder_params,
        'decoder_params': decoder_params,
        'trainable_params': decoder_params,  # We only train decoder
        'total_memory_mb': total_memory_mb,
        'decoder_memory_mb': decoder_memory_mb,
    }


def benchmark_model(model_name: str, data_loader, batch_size: int = 4, num_steps: int = 10):
    """
    Benchmark a model for training.

    Args:
        model_name: HuggingFace model name
        data_loader: Data loader for training data
        batch_size: Batch size for training
        num_steps: Number of steps to benchmark

    Returns:
        Dictionary with benchmark results
    """
    print("=" * 70)
    print(f"Benchmarking: {model_name}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    start_time = time.time()
    model = load_model(model_name)
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")

    # Convert to float32 for training
    model.set_dtype(mx.float32)

    # Get model info
    info = get_model_info(model)
    print(f"\nModel Info:")
    print(f"  Total params:     {info['total_params']:,} ({info['total_memory_mb']:.1f} MB)")
    print(f"  Encoder params:   {info['encoder_params']:,}")
    print(f"  Decoder params:   {info['decoder_params']:,} ({info['decoder_memory_mb']:.1f} MB)")
    print(f"  Trainable params: {info['trainable_params']:,}")

    # Freeze encoder (decoder-only training)
    model.encoder.freeze()

    # Simple training loop benchmark
    print(f"\nBenchmarking training ({num_steps} steps, batch size {batch_size})...")

    from train_whisper_ipa import compute_loss

    step_times = []
    sample_times = []

    for step in range(num_steps):
        step_start = time.time()

        # Get batch
        indices = [(step * batch_size + i) % len(data_loader.data)
                   for i in range(batch_size)]
        batch = data_loader.get_batch(indices)

        # Forward pass only (no backward to save time)
        mel_features = mx.array(batch['mel_features'])
        tokens = mx.array(batch['tokens'])

        forward_start = time.time()
        loss = compute_loss(model, mel_features, tokens)
        mx.eval(loss)
        forward_time = time.time() - forward_start

        step_time = time.time() - step_start
        step_times.append(step_time)
        sample_times.append(batch_size / step_time)

        if step % 5 == 0:
            print(f"  Step {step}: {step_time:.3f}s, {sample_times[-1]:.2f} samples/sec")

    avg_step_time = sum(step_times) / len(step_times)
    avg_samples_per_sec = sum(sample_times) / len(sample_times)

    # Estimate full training time (7266 samples, 1000 steps)
    total_samples = 7266
    num_training_steps = 1000
    estimated_time_min = (num_training_steps * avg_step_time) / 60
    estimated_time_hours = estimated_time_min / 60

    print(f"\nBenchmark Results:")
    print(f"  Avg step time:    {avg_step_time:.3f}s")
    print(f"  Avg throughput:   {avg_samples_per_sec:.2f} samples/sec")
    print(f"  Est. 1000 steps:  {estimated_time_min:.1f} min ({estimated_time_hours:.2f} hours)")
    print(f"  Est. 10000 steps: {estimated_time_min*10:.1f} min ({estimated_time_hours*10:.2f} hours)")

    # Estimate memory for different batch sizes
    print(f"\nEstimated memory usage (decoder-only training, float32):")
    for bs in [1, 2, 4, 8]:
        # Model weights + optimizer state (2x for Adam) + gradients + batch activations
        base_mb = info['decoder_memory_mb']
        optimizer_mb = base_mb * 2  # Adam states
        gradient_mb = base_mb
        activation_mb_per_sample = 100  # Rough estimate
        total_mb = base_mb + optimizer_mb + gradient_mb + (activation_mb_per_sample * bs)
        print(f"  Batch size {bs}: ~{total_mb:.0f} MB ({total_mb/1024:.1f} GB)")

    result = {
        'model_name': model_name,
        'load_time': load_time,
        'avg_step_time': avg_step_time,
        'avg_samples_per_sec': avg_samples_per_sec,
        'est_1000_steps_hours': estimated_time_hours,
        'est_10000_steps_hours': estimated_time_hours * 10,
        **info
    }

    return result


if __name__ == "__main__":
    # Load a small sample of data for benchmarking
    print("Loading training data...")
    data_loader = create_data_loader("data/processed/english_only_train_ipa.json")
    print(f"✓ Loaded {len(data_loader.data)} training samples")

    # Models to benchmark
    models = [
        "mlx-community/whisper-tiny-mlx",
        "mlx-community/whisper-small-mlx",
        "mlx-community/whisper-medium-mlx",
        "mlx-community/whisper-large-v3-mlx",  # Target multilingual model
    ]

    results = []

    for model_name in models:
        try:
            result = benchmark_model(model_name, data_loader, batch_size=4, num_steps=10)
            results.append(result)
            print("\n")
        except Exception as e:
            print(f"\n❌ Error benchmarking {model_name}: {e}\n")
            continue

    # Summary table
    print("=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    print(f"\n{'Model':<30} {'Decoder Params':<15} {'Samples/sec':<15} {'1000 steps':<15} {'Est. Memory (BS=4)'}")
    print("-" * 100)

    for r in results:
        model_short = r['model_name'].split('/')[-1].replace('-mlx', '')
        decoder_params_m = r['decoder_params'] / 1_000_000
        samples_sec = r['avg_samples_per_sec']
        time_1000 = r['est_1000_steps_hours']

        # Estimate memory for batch size 4
        base_mb = r['decoder_memory_mb']
        mem_gb = (base_mb + base_mb*2 + base_mb + 400) / 1024

        print(f"{model_short:<30} {decoder_params_m:>6.1f}M{'':<8} {samples_sec:>6.2f}/s{'':<8} {time_1000:>6.2f}h{'':<8} ~{mem_gb:.0f} GB")

    print("\n" + "=" * 100)
    print(f"System: M3 Ultra with 96GB unified memory")
    print(f"Training: Decoder-only (encoder frozen), float32, batch size 4")
    print("=" * 100)
