"""
CommonVoice Dataset Preparation Script

Converts teammate's preprocessed CommonVoice IPA data into our pipeline's
JSON format. Creates training size variants (1k/lang, 2k/lang, full),
plus validation and test splits.

Input format (per-language JSON from teammate's preprocessing):
  {"audio_path": "/Users/omerfaruk/.../clip.mp3",
   "sentence": "...", "ipa_transcription": "...",
   "locale": "ja", "path": "common_voice_ja_12345.mp3"}

Output format (our pipeline standard):
  {"audio_path": "/absolute/path/to/clip.mp3",
   "ipa_transcription": "NFC-normalized IPA",
   "speaker_id": "unknown", "dataset_source": "commonvoice",
   "language": "ja", "split": "train"}
"""

import json
import os
import argparse
import unicodedata
from pathlib import Path
from typing import List, Dict

import numpy as np


def load_language_data(input_dir: Path, locale: str) -> List[Dict]:
    """Load teammate's JSON for a single language.

    Auto-detects filename pattern: {locale}_train_ipa.json or {locale}.json
    """
    candidates = [
        input_dir / f"{locale}_train_ipa.json",
        input_dir / f"{locale}.json",
        input_dir / f"{locale}_train.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Loaded {len(data)} entries from {path.name}")
            return data
    raise FileNotFoundError(
        f"No data file found for locale '{locale}' in {input_dir}. "
        f"Tried: {[c.name for c in candidates]}"
    )


def remap_audio_path(entry: Dict, audio_root: Path, locale: str) -> str:
    """Remap non-portable audio path to local audio root.

    Uses the 'path' field (bare filename) from teammate's data.
    Falls back to extracting filename from 'audio_path'.
    """
    filename = entry.get("path", "")
    if not filename:
        # Extract filename from the full audio_path
        filename = Path(entry.get("audio_path", "")).name
    if not filename:
        return ""

    # CommonVoice layout: {audio_root}/{locale}/clips/{filename}
    local_path = audio_root / locale / "clips" / filename
    return str(local_path)


def process_language(
    entries: List[Dict],
    audio_root: Path,
    locale: str,
    check_audio: bool = True,
) -> List[Dict]:
    """Process entries for one language: remap paths, NFC normalize, filter."""
    processed = []
    skipped_audio = 0
    skipped_ipa = 0

    for entry in entries:
        ipa = entry.get("ipa_transcription", "")

        # NFC normalize IPA text
        ipa = unicodedata.normalize("NFC", ipa.strip())

        # Filter: empty, too short, too long
        if not ipa or len(ipa) < 2:
            skipped_ipa += 1
            continue
        if len(ipa) > 500:
            skipped_ipa += 1
            continue

        # Remap audio path
        audio_path = remap_audio_path(entry, audio_root, locale)
        if not audio_path:
            skipped_audio += 1
            continue

        # Check audio file exists (optional, can be slow)
        if check_audio and not os.path.isfile(audio_path):
            skipped_audio += 1
            continue

        processed.append({
            "audio_path": audio_path,
            "ipa_transcription": ipa,
            "speaker_id": "unknown",
            "dataset_source": "commonvoice",
            "language": locale,
        })

    if skipped_audio > 0:
        print(f"    Skipped {skipped_audio} entries (audio not found)")
    if skipped_ipa > 0:
        print(f"    Skipped {skipped_ipa} entries (IPA empty/too short/too long)")

    return processed


def split_data(
    data: List[Dict],
    locale: str,
    test_per_lang: int,
    val_per_lang: int,
    rng: np.random.Generator,
) -> tuple:
    """Split data into test, validation, and train pool.

    Test and val are extracted first (fixed sizes), remainder is train pool.
    """
    n = len(data)
    indices = rng.permutation(n)

    # Extract test first, then val, remainder is train
    test_n = min(test_per_lang, n)
    val_n = min(val_per_lang, n - test_n)
    train_n = n - test_n - val_n

    if test_n < test_per_lang:
        print(f"    WARNING: {locale} has only {n} samples, "
              f"test capped at {test_n} (wanted {test_per_lang})")
    if val_n < val_per_lang:
        print(f"    WARNING: {locale} val capped at {val_n} (wanted {val_per_lang})")

    test_idx = indices[:test_n]
    val_idx = indices[test_n:test_n + val_n]
    train_idx = indices[test_n + val_n:]

    test_data = [data[i] for i in test_idx]
    val_data = [data[i] for i in val_idx]
    train_data = [data[i] for i in train_idx]

    # Tag splits
    for entry in test_data:
        entry["split"] = "test"
    for entry in val_data:
        entry["split"] = "val"
    for entry in train_data:
        entry["split"] = "train"

    return train_data, val_data, test_data


def save_json(data: List[Dict], path: Path):
    """Save dataset to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data)} samples to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CommonVoice IPA data to pipeline JSON format"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory with teammate's per-language JSON files",
    )
    parser.add_argument(
        "--audio-root", type=str, required=True,
        help="Local root where CommonVoice audio files live",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--languages", nargs="+", default=["ja", "pl", "mt", "hu", "fi", "el", "ta"],
        help="Language codes to process (default: ja pl mt hu fi el ta)",
    )
    parser.add_argument(
        "--train-per-lang", nargs="+", type=int, default=[1000, 2000],
        help="Training sizes per variant (default: 1000 2000)",
    )
    parser.add_argument(
        "--val-per-lang", type=int, default=200,
        help="Validation samples per language (default: 200)",
    )
    parser.add_argument(
        "--test-per-lang", type=int, default=100,
        help="Test samples per language (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--no-check-audio", action="store_true",
        help="Skip checking if audio files exist locally",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    audio_root = Path(args.audio_root)
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("CommonVoice Dataset Preparation")
    print("=" * 70)
    print(f"  Input dir:   {input_dir}")
    print(f"  Audio root:  {audio_root}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Languages:   {args.languages}")
    print(f"  Train sizes: {args.train_per_lang}")
    print(f"  Val/lang:    {args.val_per_lang}")
    print(f"  Test/lang:   {args.test_per_lang}")
    print(f"  Seed:        {args.seed}")
    print()

    all_train = []
    all_val = []
    all_test = []

    for locale in args.languages:
        print(f"\nProcessing {locale}...")
        try:
            raw_data = load_language_data(input_dir, locale)
        except FileNotFoundError as e:
            print(f"  SKIPPING: {e}")
            continue

        processed = process_language(
            raw_data, audio_root, locale,
            check_audio=not args.no_check_audio,
        )
        print(f"  {len(processed)} entries after filtering")

        if len(processed) == 0:
            print(f"  SKIPPING {locale}: no valid entries")
            continue

        train_pool, val_data, test_data = split_data(
            processed, locale,
            args.test_per_lang, args.val_per_lang, rng,
        )
        print(f"  Split: train={len(train_pool)}, val={len(val_data)}, test={len(test_data)}")

        all_train.extend(train_pool)
        all_val.extend(val_data)
        all_test.extend(test_data)

    # Save test and validation sets
    print("\n" + "-" * 70)
    print("Saving datasets...")
    save_json(all_test, output_dir / "commonvoice_test.json")
    save_json(all_val, output_dir / "commonvoice_val.json")

    # Save full train
    save_json(all_train, output_dir / "commonvoice_train_full.json")

    # Save train variants (subsample per language)
    for size in args.train_per_lang:
        variant = []
        for locale in args.languages:
            lang_pool = [e for e in all_train if e["language"] == locale]
            if len(lang_pool) == 0:
                continue
            n_sample = min(size, len(lang_pool))
            if n_sample < size:
                print(f"  WARNING: {locale} train pool has only {len(lang_pool)} "
                      f"(wanted {size})")
            sampled_idx = rng.choice(len(lang_pool), size=n_sample, replace=False)
            variant.extend([lang_pool[i] for i in sampled_idx])
        save_json(variant, output_dir / f"commonvoice_train_{size}perlang.json")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Test:  {len(all_test)} samples")
    print(f"  Val:   {len(all_val)} samples")
    print(f"  Train (full): {len(all_train)} samples")
    for size in args.train_per_lang:
        path = output_dir / f"commonvoice_train_{size}perlang.json"
        if path.exists():
            data = json.load(open(path))
            print(f"  Train ({size}/lang): {len(data)} samples")
    print("\nDone!")


if __name__ == "__main__":
    main()
