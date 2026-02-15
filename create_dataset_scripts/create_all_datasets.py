#!/usr/bin/env python3
"""Create all 3 dataset versions (v1_raw, v2_filtered, v3_improved) with train/validation/test splits."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocess_local import main

LANGUAGES = ["fi", "el", "hu", "ja", "mt", "pl", "ta"]
NUM_SAMPLES = 1000


def run_all():
    all_counts = {}

    # Generate train sets for all 3 modes
    for mode in ("raw", "filtered", "improved"):
        print(f"\n{'#'*60}")
        print(f"  Creating dataset: {mode} (train)")
        print(f"{'#'*60}")
        counts = main(languages=LANGUAGES, num_samples=NUM_SAMPLES, mode=mode, split="train")
        all_counts[(mode, "train")] = counts

    # Generate validation and test sets (raw mode only, as per paper)
    for split, n_samples in [("validation", 200), ("test", 100)]:
        for mode in ("raw", "filtered", "improved"):
            print(f"\n{'#'*60}")
            print(f"  Creating dataset: {mode} ({split})")
            print(f"{'#'*60}")
            counts = main(languages=LANGUAGES, num_samples=n_samples, mode=mode, split=split)
            all_counts[(mode, split)] = counts

    # Print summary table
    print(f"\n\n{'='*70}")
    print("  SUMMARY: Samples per language per version per split")
    print(f"{'='*70}")

    for split in ("train", "validation", "test"):
        print(f"\n  --- {split.upper()} ---")
        header = f"  {'Lang':<6}"
        for mode in ("raw", "filtered", "improved"):
            header += f"{'v1_raw' if mode == 'raw' else 'v2_filtered' if mode == 'filtered' else 'v3_improved':>14}"
        print(header)
        print(f"  {'-'*48}")

        for lang in LANGUAGES:
            row = f"  {lang:<6}"
            for mode in ("raw", "filtered", "improved"):
                row += f"{all_counts.get((mode, split), {}).get(lang, 0):>14}"
            print(row)

        print(f"  {'-'*48}")
        row = f"  {'TOTAL':<6}"
        for mode in ("raw", "filtered", "improved"):
            row += f"{sum(all_counts.get((mode, split), {}).values()):>14}"
        print(row)

    print()


if __name__ == "__main__":
    run_all()
