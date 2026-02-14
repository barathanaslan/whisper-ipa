#!/usr/bin/env python3
"""Create all 3 dataset versions (v1_raw, v2_filtered, v3_improved)."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocess_local import main

LANGUAGES = ["fi", "el", "hu", "ja", "mt", "pl", "ta"]
NUM_SAMPLES = 1000


def run_all():
    all_counts = {}

    for mode in ("raw", "filtered", "improved"):
        print(f"\n{'#'*60}")
        print(f"  Creating dataset: {mode}")
        print(f"{'#'*60}")
        counts = main(languages=LANGUAGES, num_samples=NUM_SAMPLES, mode=mode)
        all_counts[mode] = counts

    # Print summary table
    print(f"\n\n{'='*60}")
    print("  SUMMARY: Samples per language per version")
    print(f"{'='*60}")
    header = f"  {'Lang':<6}"
    for mode in ("raw", "filtered", "improved"):
        header += f"{'v1_raw' if mode == 'raw' else 'v2_filtered' if mode == 'filtered' else 'v3_improved':>14}"
    print(header)
    print(f"  {'-'*48}")

    for lang in LANGUAGES:
        row = f"  {lang:<6}"
        for mode in ("raw", "filtered", "improved"):
            row += f"{all_counts[mode].get(lang, 0):>14}"
        print(row)

    # Totals
    print(f"  {'-'*48}")
    row = f"  {'TOTAL':<6}"
    for mode in ("raw", "filtered", "improved"):
        row += f"{sum(all_counts[mode].values()):>14}"
    print(row)
    print()


if __name__ == "__main__":
    run_all()
