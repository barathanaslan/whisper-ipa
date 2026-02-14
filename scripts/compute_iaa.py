"""
Compute Inter-Annotator Agreement (A6).

Loads parsed zero-shot test data, computes IAA between Ariga and Hamanishi
using PER, PFER-Hamming, and PFER-Cosine in both directions. Identifies
which direction + metric matches the paper's reported 19.6% PFER.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_ipa import (
    phone_error_rate,
    phone_feature_error_rate,
    phone_feature_error_rate_cosine,
    normalize_ipa_for_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "zeroshot_test.json"

PAPER_IAA = 19.6  # Taguchi et al. reported IAA


def load_usable_pairs(path):
    """Load entries usable for IAA computation."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [e for e in data if e['usable_for_iaa']]


def compute_metrics(ref_texts, hyp_texts):
    """Compute PER, PFER-Hamming, and PFER-Cosine for a list of pairs."""
    per_scores = []
    pfer_h_scores = []
    pfer_c_scores = []

    for ref, hyp in zip(ref_texts, hyp_texts):
        per_scores.append(phone_error_rate(ref, hyp))
        pfer_h_scores.append(phone_feature_error_rate(ref, hyp))
        pfer_c_scores.append(phone_feature_error_rate_cosine(ref, hyp))

    return {
        'per': np.mean(per_scores),
        'per_std': np.std(per_scores),
        'pfer_hamming': np.mean(pfer_h_scores),
        'pfer_hamming_std': np.std(pfer_h_scores),
        'pfer_cosine': np.mean(pfer_c_scores),
        'pfer_cosine_std': np.std(pfer_c_scores),
        'per_scores': per_scores,
        'pfer_h_scores': pfer_h_scores,
        'pfer_c_scores': pfer_c_scores,
    }


def main():
    print("=" * 70)
    print("Inter-Annotator Agreement (A6)")
    print("=" * 70)

    # Load data
    pairs = load_usable_pairs(INPUT_PATH)
    print(f"\nUsable pairs: {len(pairs)}")

    # Collect excluded entries for reporting
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    excluded = [e for e in all_data if e['has_both_annotators'] and not e['usable_for_iaa']]
    no_both = [e for e in all_data if not e['has_both_annotators']]
    if excluded:
        exc_ids = [e['id'] for e in excluded]
        print(f"Excluded (poor quality): IDs {exc_ids}")
    print(f"Single-annotator only: {len(no_both)} entries")

    # Normalize IPA
    ariga_texts = [normalize_ipa_for_comparison(e['ipa_ariga']) for e in pairs]
    hamanishi_texts = [normalize_ipa_for_comparison(e['ipa_hamanishi']) for e in pairs]
    ids = [e['id'] for e in pairs]

    # Sanity check: self-comparison
    print("\n--- Sanity Check: Self-Comparison ---")
    self_per = np.mean([phone_error_rate(a, a) for a in ariga_texts[:5]])
    self_pfer = np.mean([phone_feature_error_rate(a, a) for a in ariga_texts[:5]])
    print(f"Ariga vs Ariga (first 5): PER={self_per:.4f}%, PFER={self_pfer:.4f}%")
    assert self_per == 0.0 and self_pfer == 0.0, "Self-comparison should be 0!"
    print("PASSED: Self-comparison = 0%")

    # Direction A: Ariga as reference
    print("\n--- Direction: Ariga as reference, Hamanishi as hypothesis ---")
    dir_a = compute_metrics(ariga_texts, hamanishi_texts)
    print(f"  PER:           {dir_a['per']:.1f}% (\u00b1{dir_a['per_std']:.1f}%)")
    print(f"  PFER-Hamming:  {dir_a['pfer_hamming']:.1f}% (\u00b1{dir_a['pfer_hamming_std']:.1f}%)")
    print(f"  PFER-Cosine:   {dir_a['pfer_cosine']:.1f}% (\u00b1{dir_a['pfer_cosine_std']:.1f}%)")

    # Direction B: Hamanishi as reference
    print("\n--- Direction: Hamanishi as reference, Ariga as hypothesis ---")
    dir_b = compute_metrics(hamanishi_texts, ariga_texts)
    print(f"  PER:           {dir_b['per']:.1f}% (\u00b1{dir_b['per_std']:.1f}%)")
    print(f"  PFER-Hamming:  {dir_b['pfer_hamming']:.1f}% (\u00b1{dir_b['pfer_hamming_std']:.1f}%)")
    print(f"  PFER-Cosine:   {dir_b['pfer_cosine']:.1f}% (\u00b1{dir_b['pfer_cosine_std']:.1f}%)")

    # Find best match to paper's 19.6%
    print(f"\n--- Paper reports: {PAPER_IAA}% PFER ---")
    candidates = [
        ("Ariga-ref, PFER-Hamming", dir_a['pfer_hamming']),
        ("Ariga-ref, PFER-Cosine", dir_a['pfer_cosine']),
        ("Hamanishi-ref, PFER-Hamming", dir_b['pfer_hamming']),
        ("Hamanishi-ref, PFER-Cosine", dir_b['pfer_cosine']),
    ]
    best = min(candidates, key=lambda x: abs(x[1] - PAPER_IAA))
    print(f"Best match: {best[0]} = {best[1]:.1f}%")
    print(f"Difference from paper: {abs(best[1] - PAPER_IAA):.1f}%")

    if "Ariga-ref" in best[0]:
        print("=> Gold standard: ARIGA (reference in best-matching direction)")
    else:
        print("=> Gold standard: HAMANISHI (reference in best-matching direction)")

    if "Cosine" in best[0]:
        print("=> Metric: PFER-Cosine (Taguchi's formula)")
    else:
        print("=> Metric: PFER-Hamming (our formula)")

    # Per-sample distribution
    print("\n--- Per-Sample Score Distribution (best-matching metric) ---")
    if "Ariga-ref" in best[0]:
        scores = dir_a['pfer_c_scores'] if "Cosine" in best[0] else dir_a['pfer_h_scores']
    else:
        scores = dir_b['pfer_c_scores'] if "Cosine" in best[0] else dir_b['pfer_h_scores']

    scores_arr = np.array(scores)
    print(f"  Min:    {scores_arr.min():.1f}%")
    print(f"  Max:    {scores_arr.max():.1f}%")
    print(f"  Median: {np.median(scores_arr):.1f}%")
    print(f"  Mean:   {scores_arr.mean():.1f}%")

    # Top 5 highest-disagreement samples
    print("\n--- Top 5 Highest-Disagreement Samples ---")
    ranked = sorted(zip(ids, scores, ariga_texts, hamanishi_texts),
                    key=lambda x: x[1], reverse=True)
    for rank, (eid, score, a_ipa, h_ipa) in enumerate(ranked[:5], 1):
        print(f"  {rank}. ID {eid}: {score:.1f}%")
        print(f"     Ariga:     {a_ipa}")
        print(f"     Hamanishi: {h_ipa}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
