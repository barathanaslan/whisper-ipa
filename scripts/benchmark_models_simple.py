"""
Simple benchmark to show model sizes and estimated training requirements.
"""

import mlx.core as mx
from mlx_whisper.load_models import load_model
from mlx.utils import tree_flatten


def analyze_model(model_name: str):
    """Analyze model size and estimate training requirements."""
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")

    # Load model
    print("Loading...")
    model = load_model(model_name)

    # Get parameters
    params = tree_flatten(model.parameters())
    total_params = sum(p.size for _, p in params)
    encoder_params = sum(p.size for k, p in params if k.startswith('encoder.'))
    decoder_params = sum(p.size for k, p in params if k.startswith('decoder.'))

    # Memory estimates (float32 = 4 bytes/param)
    total_mb = (total_params * 4) / (1024 ** 2)
    decoder_mb = (decoder_params * 4) / (1024 ** 2)

    # Training memory estimate (decoder-only, float32)
    # Model + Optimizer (Adam: 2x model) + Gradients + Activations
    for batch_size in [1, 2, 4, 8]:
        base = decoder_mb
        optimizer = base * 2  # Adam states (momentum + variance)
        gradients = base
        # Rough activation estimate: ~100MB per sample for small, scales with model
        activation_per_sample = 100 * (decoder_params / 150_000_000)
        activations = activation_per_sample * batch_size

        total = base + optimizer + gradients + activations
        print(f"\nBatch Size {batch_size}:")
        print(f"  Model weights:    {base:>8.0f} MB")
        print(f"  Optimizer state:  {optimizer:>8.0f} MB  (Adam)")
        print(f"  Gradients:        {gradients:>8.0f} MB")
        print(f"  Activations:      {activations:>8.0f} MB  (est.)")
        print(f"  {'─'*40}")
        print(f"  TOTAL:            {total:>8.0f} MB  ({total/1024:>6.1f} GB)")

    # Training speed estimates (based on whisper-small benchmark)
    # Small model: ~17 samples/sec on M3 Ultra
    # Speed scales roughly inversely with decoder params
    small_decoder_params = 153_580_800
    small_samples_per_sec = 17.0

    estimated_samples_per_sec = small_samples_per_sec * (small_decoder_params / decoder_params)

    # Time for 1000 and 10000 steps with batch size 4
    steps_per_epoch = 7266 / 4  # 7266 samples, batch size 4
    time_per_step_sec = 4 / estimated_samples_per_sec
    time_1000_steps_min = (1000 * time_per_step_sec) / 60
    time_10000_steps_min = (10000 * time_per_step_sec) / 60

    print(f"\n{'─'*80}")
    print(f"Training Speed Estimates (Batch Size 4):")
    print(f"  Est. throughput:  {estimated_samples_per_sec:>6.1f} samples/sec")
    print(f"  Time per step:    {time_per_step_sec:>6.2f} sec")
    print(f"  1,000 steps:      {time_1000_steps_min:>6.0f} min  ({time_1000_steps_min/60:>5.1f} hours)")
    print(f"  10,000 steps:     {time_10000_steps_min:>6.0f} min  ({time_10000_steps_min/60:>5.1f} hours)")

    print(f"\n{'─'*80}")
    print(f"Summary:")
    print(f"  Total Parameters:     {total_params/1_000_000:>8.1f}M  ({total_mb/1024:>6.2f} GB)")
    print(f"  Encoder Parameters:   {encoder_params/1_000_000:>8.1f}M  (frozen)")
    print(f"  Decoder Parameters:   {decoder_params/1_000_000:>8.1f}M  ({decoder_mb/1024:>6.2f} GB)")
    print(f"  Trainable:            {decoder_params/1_000_000:>8.1f}M  (decoder only)")

    return {
        'model_name': model_name,
        'total_params': total_params,
        'decoder_params': decoder_params,
        'decoder_mb': decoder_mb,
        'estimated_samples_per_sec': estimated_samples_per_sec,
        'time_1000_steps_hours': time_1000_steps_min / 60,
        'time_10000_steps_hours': time_10000_steps_min / 60,
    }


if __name__ == "__main__":
    models = [
        "mlx-community/whisper-tiny-mlx",
        "mlx-community/whisper-small-mlx",
        "mlx-community/whisper-medium-mlx",
        "mlx-community/whisper-large-v3-mlx",
    ]

    results = []
    for model_name in models:
        try:
            result = analyze_model(model_name)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

    # Final summary table
    print(f"\n\n{'='*100}")
    print(f"{'COMPLETE BENCHMARK SUMMARY':^100}")
    print(f"{'='*100}\n")

    print(f"{'Model':<20} {'Decoder':<12} {'Speed':<15} {'1K Steps':<12} {'10K Steps':<12} {'Memory (BS=4)':<15}")
    print(f"{'':20} {'Params':<12} {'(samp/sec)':<15} {'(hours)':<12} {'(hours)':<12} {'(GB)':<15}")
    print(f"{'─'*100}")

    for r in results:
        name = r['model_name'].split('/')[-1].replace('-mlx', '')
        dec_m = r['decoder_params'] / 1_000_000
        speed = r['estimated_samples_per_sec']
        t1k = r['time_1000_steps_hours']
        t10k = r['time_10000_steps_hours']

        # Memory for BS=4 (model + optimizer + gradients + activations)
        base = r['decoder_mb']
        mem_gb = (base + base*2 + base + (100 * r['decoder_params']/150_000_000 * 4)) / 1024

        print(f"{name:<20} {dec_m:>7.0f}M     {speed:>8.1f}       {t1k:>8.1f}       {t10k:>9.1f}       {mem_gb:>8.0f}")

    print(f"\n{'='*100}")
    print(f"System: Apple M3 Ultra (96GB Unified Memory)")
    print(f"Config: Decoder-only training (encoder frozen), float32, batch size 4")
    print(f"Note: Speed estimates based on whisper-small actual performance (~17 samples/sec)")
    print(f"      Memory includes: model weights + Adam optimizer + gradients + activations")
    print(f"{'='*100}")
