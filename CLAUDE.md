# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tuning OpenAI's Whisper model to output IPA (International Phonetic Alphabet) transcriptions instead of standard orthographic text. Uses MLX (Apple's ML framework) on Mac Studio M3 Ultra with 96GB unified memory.

## Environment Setup

```bash
source venv/bin/activate  # Python 3.13, always activate before running scripts
```

Key dependencies: mlx, mlx-whisper, torch, transformers, panphon, editdistance, piper-tts

## Common Commands

```bash
# Train a model (decoder-only fine-tuning, encoder is frozen)
python scripts/train_whisper_ipa.py \
  --model mlx-community/whisper-small-mlx \
  --train_data data/processed/combined_train_ipa.json \
  --test_data data/processed/combined_test_ipa.json \
  --output_dir checkpoints/whisper-ipa-english \
  --num_steps 20000 --batch_size 16 --learning_rate 1e-5

# Evaluate a checkpoint
python scripts/evaluate_model.py --checkpoint checkpoints/whisper-ipa-english/checkpoint-8000 \
  --test_data data/processed/timit_test_ipa.json

# Transcribe a single audio file
python scripts/transcribe_single.py

# Prepare datasets
python scripts/prepare_timit_dataset.py
python scripts/prepare_metu_turkish.py
python scripts/prepare_ogi_spelled.py

# Combine datasets
python scripts/combine_datasets.py

# Verify IPA normalization consistency
python scripts/verify_ipa_normalization.py

# Benchmark model variants
python scripts/benchmark_models.py

# Monitor training speed
python calculate_real_speed.py [PID] checkpoints/whisper-ipa [TOTAL_STEPS]
```

## Architecture

**Training pipeline** (`scripts/train_whisper_ipa.py`): Freezes encoder, trains decoder only with AdamW optimizer. Loss is cross-entropy with EOT padding mask (first EOT included so model learns to stop). Gradient clipping at max_norm=1.0. Validates every N steps using PER/PFER metrics.

**Data loading** (`scripts/ipa_data_loader.py`): Loads audio-IPA pairs from JSON, converts to mel spectrograms (80 bins for small/medium, 128 for large models). Tokenizes IPA with Whisper special tokens: `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>[IPA]<|endoftext|>`.

**Evaluation** (`scripts/evaluate_ipa.py`): Two metrics — PER (Phone Error Rate, edit distance at phone level) and PFER (Phone Feature Error Rate, weighted by phonetic features via PanPhon library). IPA tokenization is currently character-level.

**Checkpoint loading** (`scripts/evaluate_model.py`): Supports both .safetensors (current) and .npz (legacy) formats. Only decoder weights are saved/loaded since encoder is frozen.

**Dataset format** (JSON):
```json
{"audio_path": "/path/to/audio.wav", "ipa_transcription": "hɛloʊ wɜrld", "speaker_id": "SPEAKER001", "dataset_source": "timit"}
```

## Key Technical Details

- Whisper's GPT-2 BPE tokenizer already covers 100% of IPA characters — no tokenizer modification needed
- Small/medium models use 80 mel frequency bins; large models use 128
- All training uses float32 for numerical stability (`model.set_dtype(mx.float32)`)
- Checkpoints save decoder-only weights in safetensors format with `training_state.json` metadata
- Mac/MPS constraints: DataLoader num_workers must be 0; no ROI operations supported

## Datasets

Processed datasets live in `data/processed/` as JSON files. Source corpora: TIMIT (English), METU (Turkish), OGI (multilingual). Combined variants and english-only variants available. Raw archives in `data/`.
