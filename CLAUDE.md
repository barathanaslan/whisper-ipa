# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tuning OpenAI's Whisper model to output IPA (International Phonetic Alphabet) transcriptions instead of standard orthographic text. Uses MLX (Apple's ML framework) on Mac Studio M3 Ultra with 96GB unified memory. Target: beat Taguchi et al. ("multipa") baseline of 21.2% PFER zero-shot.

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
  --train-data data/processed/combined_train_ipa.json \
  --test-data data/processed/combined_test_ipa.json \
  --output-dir checkpoints/whisper-ipa-english \
  --steps 20000 --batch-size 16 --lr 1e-5

# Quick test run (100 samples, capped steps)
python scripts/train_whisper_ipa.py --test-run

# Evaluate a checkpoint against base model
python scripts/evaluate_model.py \
  --checkpoint checkpoints/whisper-ipa-english/checkpoint-8000 \
  --test-data data/processed/timit_test_ipa.json \
  --base-model mlx-community/whisper-small-mlx \
  --n-mels 80 --num-samples 100

# Evaluate checkpoint only (skip base model comparison)
python scripts/evaluate_model.py --checkpoint <path> --skip-base

# Transcribe a single audio file (edit script to change checkpoint/audio paths)
python scripts/transcribe_single.py

# Prepare datasets
python scripts/data_prep/prepare_timit_dataset.py
python scripts/data_prep/prepare_metu_turkish.py
python scripts/data_prep/prepare_ogi_spelled.py
python scripts/data_prep/combine_datasets.py

# Verify IPA normalization consistency
python scripts/data_prep/verify_ipa_normalization.py

# Parse zero-shot test annotations (A3)
python scripts/parse_zeroshot_test.py

# Compute inter-annotator agreement (A6)
python scripts/compute_iaa.py

# Prepare CommonVoice dataset (convert teammate's IPA output to pipeline format)
python scripts/data_prep/prepare_commonvoice_dataset.py \
  --input-dir data/v3_improved \
  --audio-root /path/to/commonvoice/audio \
  --output-dir data/processed \
  --languages ja pl mt hu fi el ta \
  --train-per-lang 1000 2000 \
  --val-per-lang 200 --test-per-lang 100

# Benchmark model variants
python scripts/experimental/benchmark_models.py

# Monitor training speed (reads PID runtime + checkpoint timestamps)
python calculate_real_speed.py [PID] checkpoints/whisper-ipa [TOTAL_STEPS]
```

## Architecture

**Training pipeline** (`scripts/train_whisper_ipa.py`): Freezes encoder via `model.encoder.freeze()`, trains decoder only with AdamW optimizer. Loss is cross-entropy with EOT padding mask — first EOT is included in loss so model learns to stop, subsequent EOTs (padding) are masked out via cumulative sum. Gradient clipping at max_norm=1.0. Validates every N steps by running decoder beam search and computing PER/PFER on predictions.

**Data loading** (`scripts/ipa_data_loader.py`): `IPADataset` class loads audio-IPA pairs from JSON. Audio is loaded via `mlx_whisper.audio.load_audio` (resamples to 16kHz), padded/trimmed to 30s, then converted to log mel spectrograms. IPA text is tokenized with Whisper's multilingual tokenizer using special token prefix: `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>[IPA]<|endoftext|>`. Batch padding uses EOT token.

**Evaluation** (`scripts/evaluate_ipa.py`): Two metrics:
- **PER** (Phone Error Rate): edit distance at phone level via `editdistance` library
- **PFER** (Phone Feature Error Rate): custom DP edit distance where substitution cost = feature_mismatches/24 using PanPhon's 24 phonetic features; insertion/deletion cost = 1

`tokenize_ipa()` uses panphon `ipa_segs()` with Unicode-category fallback to properly handle combining diacritics (m̩, n̩, l̩, ŋ̍, ɾ̃, ə̥). Also provides `PFERCalculatorCosine` (Taguchi's formula) and `normalize_ipa_for_comparison()` (NFC + strip + g→ɡ).

**Checkpoint system**: `save_checkpoint()` flattens nested model params and saves decoder-only weights in safetensors format. `load_checkpoint_model()` (in `evaluate_model.py` and `transcribe_single.py`) loads base model, then overlays decoder weights filtered by `decoder.` prefix using `tree_flatten`/`tree_unflatten`.

**Training logging** (A4): `TrainingLogger` writes two CSV files — `training_log.csv` (step, loss, lr, timing, peak memory every 10 steps) and `validation_log.csv` (PER, PFER at each validation). Also: `training_config.json` at start (hyperparameters + hardware), `training_summary.json` at end (wall-clock, final/best metrics), and `best-checkpoint/` directory tracking lowest PFER. Console output format unchanged for `calculate_real_speed.py` backward compat.

**Dataset format** (JSON arrays in `data/processed/`):
```json
{"audio_path": "/path/to/audio.wav", "ipa_transcription": "hɛloʊ wɜrld", "speaker_id": "SPEAKER001", "dataset_source": "timit"}
```

## Key Technical Details

- Whisper's GPT-2 BPE tokenizer already covers 100% of IPA characters — no tokenizer modification needed
- Small/medium models use 80 mel frequency bins; large models use 128 — controlled by `n_mels` parameter throughout
- All training uses float32 for numerical stability (`model.set_dtype(mx.float32)`)
- Checkpoints save decoder-only weights in safetensors format (overcomes `mx.savez` >1024 args limit) with `training_state.json` metadata
- Mac/MPS constraints: DataLoader num_workers must be 0; no ROI operations supported; no float64
- Scripts in `scripts/` import each other (e.g., `train_whisper_ipa.py` imports from `ipa_data_loader` and `evaluate_ipa`), so they must be run from the `scripts/` directory or with `scripts/` on `sys.path`
- `evaluate_model.py` uses `sys.path.insert(0, 'scripts')` to handle imports when run from project root
- Training uses random batch sampling (`np.random.choice`) rather than epoch-based iteration
- Gold standard for zero-shot evaluation: Hamanishi annotations (broad phonetic). Confirmed by IAA = 19.6% PFER (Hamanishi as reference, Ariga as hypothesis)
- PFER formula: Hamming distance (mismatches/24), validated against paper's reported IAA
- Zero-shot test data parsed to `data/processed/zeroshot_test.json` (98 usable IAA pairs)
- Scripts organized: core pipeline in `scripts/`, dataset prep in `scripts/data_prep/`, prototypes in `scripts/experimental/`

## Datasets

Processed datasets live in `data/processed/` as JSON files:
- **TIMIT** (English): 4,620 train / 1,680 test — ARPABET phonetic labels converted to IPA via mapping in `prepare_timit_dataset.py` (61 phones, closures dropped)
- **METU** (Turkish): Turkish speech corpus
- **OGI** (multilingual): spelled speech corpus
- Combined variants (`combined_*.json`) and English-only variants (`english_only_*.json`) available

Raw archives in `data/`.
