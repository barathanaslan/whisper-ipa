#!/usr/bin/env python3
"""Analyze CommonVoice dataset quality across 7 languages."""

import os
import re
import pandas as pd

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "../../dataset")
LANGUAGES = {
    "finnish": "fi",
    "greek": "el",
    "hungarian": "hu",
    "japanese": "ja",
    "maltese": "mt",
    "polish": "pl",
    "tamil": "ta",
}


def analyze_language(lang_name, lang_code):
    base = os.path.join(DATASET_ROOT, lang_name, lang_code)
    tsv_path = os.path.join(base, "train.tsv")
    clips_dir = os.path.join(base, "clips")

    print(f"\n{'='*80}")
    print(f"  {lang_name.upper()} ({lang_code})")
    print(f"{'='*80}")

    if not os.path.exists(tsv_path):
        print(f"  [ERROR] train.tsv not found at {tsv_path}")
        return

    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    total = len(df)
    print(f"\n  1. Total rows: {total:,}")

    # 2. Empty/NaN sentences
    empty_sent = df["sentence"].isna().sum() + (df["sentence"].astype(str).str.strip() == "").sum()
    print(f"  2. Empty/NaN sentences: {empty_sent}")

    # Work with string sentences
    sents = df["sentence"].astype(str).str.strip()

    # 3. Duplicates
    dup_count = sents.duplicated(keep=False).sum()
    unique_dups = sents.duplicated(keep="first").sum()
    print(f"  3. Duplicate sentences: {unique_dups} duplicates ({dup_count} total rows involved)")

    # 4. Suspicious characters
    html_ent = sents.str.contains(r"&[a-zA-Z]+;|&#\d+;", regex=True, na=False).sum()
    urls = sents.str.contains(r"https?://|www\.", regex=True, na=False).sum()
    numbers_only = sents.str.match(r"^[\d\s\.\,\-\+]+$", na=False).sum()
    print(f"  4. Suspicious content:")
    print(f"     - HTML entities: {html_ent}")
    print(f"     - URLs: {urls}")
    print(f"     - Numbers-only: {numbers_only}")

    # 5. Very short / very long
    lengths = sents.str.len()
    very_short = (lengths < 3).sum()
    very_long = (lengths > 500).sum()
    print(f"  5. Very short (<3 chars): {very_short}  |  Very long (>500 chars): {very_long}")
    if very_short > 0:
        examples = sents[lengths < 3].head(5).tolist()
        print(f"     Short examples: {examples}")
    if very_long > 0:
        examples = sents[lengths > 500].head(3).apply(lambda x: x[:80] + "...").tolist()
        print(f"     Long examples: {examples}")

    # 6. Missing clip files
    clips_exist = os.path.isdir(clips_dir)
    if clips_exist:
        clip_files = set(os.listdir(clips_dir))
        paths = df["path"].astype(str).str.strip()
        missing_clips = paths.apply(lambda p: p not in clip_files).sum()
        print(f"  6. Rows where clip file not found in clips/: {missing_clips} / {total}")
    else:
        print(f"  6. [WARNING] clips/ directory not found at {clips_dir}")

    # 7. Missing/empty path values
    empty_paths = df["path"].isna().sum() + (df["path"].astype(str).str.strip() == "").sum()
    print(f"  7. Missing/empty 'path' values: {empty_paths}")

    # 8. Non-text content
    pure_symbols = sents.str.match(r"^[^\w\s]+$", na=False).sum()
    print(f"  8. Non-text content:")
    print(f"     - Pure numbers: {numbers_only}")
    print(f"     - Pure symbols/punctuation: {pure_symbols}")

    # 9. Down votes
    if "down_votes" in df.columns:
        down_gt0 = (df["down_votes"] > 0).sum()
        print(f"  9. Rows with down_votes > 0: {down_gt0} ({down_gt0/total*100:.1f}%)")
        if "up_votes" in df.columns:
            more_down = (df["down_votes"] > df["up_votes"]).sum()
            print(f"     Rows where down_votes > up_votes: {more_down}")
    else:
        print(f"  9. No 'down_votes' column found")

    # 10. Sample sentences
    print(f"  10. Sample sentences:")
    samples = sents.sample(min(5, total), random_state=42)
    for i, s in enumerate(samples):
        print(f"      [{i+1}] {s[:120]}")

    # Summary stats
    print(f"\n  Sentence length stats: mean={lengths.mean():.1f}, median={lengths.median():.1f}, "
          f"min={lengths.min()}, max={lengths.max()}")


def main():
    print("CommonVoice Dataset Quality Analysis")
    print(f"Dataset root: {os.path.abspath(DATASET_ROOT)}")

    for lang_name, lang_code in LANGUAGES.items():
        analyze_language(lang_name, lang_code)

    print(f"\n{'='*80}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
