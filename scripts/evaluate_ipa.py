"""
Evaluation metrics for IPA transcription.

Implements:
- PER (Phone Error Rate): Edit distance at phone level
- PFER (Phone Feature Error Rate): Edit distance with phonetic features
"""

import unicodedata

import numpy as np
from typing import List, Tuple, Dict
import editdistance
import panphon

# Module-level cache for panphon FeatureTable (avoid re-init per call)
_ft = None

def _get_feature_table():
    """Get cached panphon FeatureTable instance."""
    global _ft
    if _ft is None:
        _ft = panphon.FeatureTable()
    return _ft


def tokenize_ipa(text: str) -> List[str]:
    """
    Tokenize IPA text into phones, properly handling combining diacritics.

    Uses panphon's ipa_segs() for primary segmentation. Falls back to
    Unicode-category-based grouping when panphon drops characters
    (e.g. ŋ̍ where panphon loses U+030D).

    Args:
        text: IPA transcription string

    Returns:
        List of phone strings
    """
    text = text.replace(' ', '')
    if not text:
        return []

    # Primary segmentation via panphon
    ft = _get_feature_table()
    phones = ft.ipa_segs(text)

    # If panphon preserved all characters, use its result
    if ''.join(phones) == text:
        return phones

    # Fallback for texts where panphon drops characters (e.g. ŋ̍):
    # Group combining marks (category M) and spacing modifier letters
    # (U+02B0-U+02FF, category Lm) with their preceding base character.
    segments = []
    for char in text:
        cat = unicodedata.category(char)
        is_modifier = (cat.startswith('M') or
                       (cat == 'Lm' and '\u02b0' <= char <= '\u02ff'))
        if segments and is_modifier:
            segments[-1] += char
        else:
            segments.append(char)
    return segments


def phone_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Phone Error Rate (PER).

    PER = (Substitutions + Insertions + Deletions) / Total_Phones_in_Reference × 100

    Args:
        reference: Ground truth IPA transcription
        hypothesis: Predicted IPA transcription

    Returns:
        PER as a percentage
    """
    ref_phones = tokenize_ipa(reference)
    hyp_phones = tokenize_ipa(hypothesis)

    if len(ref_phones) == 0:
        return 0.0 if len(hyp_phones) == 0 else 100.0

    # Calculate edit distance
    distance = editdistance.eval(ref_phones, hyp_phones)

    # PER = edit_distance / reference_length × 100
    per = (distance / len(ref_phones)) * 100.0

    return per


class PFERCalculator:
    """Calculator for Phone Feature Error Rate using PanPhon."""

    def __init__(self):
        """Initialize with PanPhon feature table."""
        self.ft = panphon.FeatureTable()
        self.num_features = 24  # PanPhon uses 24 phonetic features

    def get_phone_features(self, phone: str) -> np.ndarray:
        """
        Get feature vector for a phone.

        Args:
            phone: IPA phone character

        Returns:
            Numpy array of features (24 dimensions)
        """
        try:
            features = self.ft.word_to_vector_list(phone, numeric=True)
            if len(features) > 0:
                return np.array(features[0])
            else:
                # Unknown phone - return zero vector
                return np.zeros(self.num_features)
        except:
            # Fallback for unknown phones
            return np.zeros(self.num_features)

    def feature_distance(self, phone1: str, phone2: str) -> float:
        """
        Calculate feature-based distance between two phones.

        As per paper: feature mismatch costs 1/24, other ops cost 1

        Args:
            phone1: First phone
            phone2: Second phone

        Returns:
            Distance (0 to 1, where 1 = completely different)
        """
        if phone1 == phone2:
            return 0.0

        feat1 = self.get_phone_features(phone1)
        feat2 = self.get_phone_features(phone2)

        # Count feature mismatches
        mismatches = np.sum(feat1 != feat2)

        # Cost = mismatches / 24 (as per paper)
        cost = mismatches / self.num_features

        return cost

    def phone_feature_error_rate(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Phone Feature Error Rate (PFER).

        PFER uses edit distance with custom substitution costs based on
        phonetic features. Insertions and deletions cost 1, but substitutions
        cost based on feature similarity (0 to 1).

        Args:
            reference: Ground truth IPA transcription
            hypothesis: Predicted IPA transcription

        Returns:
            PFER as a percentage
        """
        ref_phones = tokenize_ipa(reference)
        hyp_phones = tokenize_ipa(hypothesis)

        if len(ref_phones) == 0:
            return 0.0 if len(hyp_phones) == 0 else 100.0

        # Dynamic programming for feature-weighted edit distance
        m, n = len(ref_phones), len(hyp_phones)
        dp = np.zeros((m + 1, n + 1))

        # Initialize: deletions and insertions
        for i in range(m + 1):
            dp[i][0] = i  # Cost of i deletions
        for j in range(n + 1):
            dp[0][j] = j  # Cost of j insertions

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                ref_phone = ref_phones[i - 1]
                hyp_phone = hyp_phones[j - 1]

                # Substitution cost based on feature distance
                sub_cost = self.feature_distance(ref_phone, hyp_phone)

                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,        # Deletion
                    dp[i][j - 1] + 1.0,        # Insertion
                    dp[i - 1][j - 1] + sub_cost  # Substitution
                )

        # PFER = feature_edit_distance / reference_length × 100
        pfer = (dp[m][n] / len(ref_phones)) * 100.0

        return pfer


# Global calculator instance
_pfer_calc = None

def get_pfer_calculator():
    """Get global PFER calculator instance."""
    global _pfer_calc
    if _pfer_calc is None:
        _pfer_calc = PFERCalculator()
    return _pfer_calc


def phone_feature_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Phone Feature Error Rate (PFER).

    Convenience function that uses global PFER calculator.

    Args:
        reference: Ground truth IPA transcription
        hypothesis: Predicted IPA transcription

    Returns:
        PFER as a percentage
    """
    calc = get_pfer_calculator()
    return calc.phone_feature_error_rate(reference, hypothesis)


def evaluate_batch(references: List[str], hypotheses: List[str]) -> Dict:
    """
    Evaluate a batch of predictions.

    Args:
        references: List of ground truth IPA transcriptions
        hypotheses: List of predicted IPA transcriptions

    Returns:
        Dictionary with average PER and PFER, plus standard deviations
    """
    assert len(references) == len(hypotheses), \
        "Reference and hypothesis lists must have same length"

    per_scores = []
    pfer_scores = []

    for ref, hyp in zip(references, hypotheses):
        per = phone_error_rate(ref, hyp)
        pfer = phone_feature_error_rate(ref, hyp)

        per_scores.append(per)
        pfer_scores.append(pfer)

    return {
        'per': np.mean(per_scores),
        'pfer': np.mean(pfer_scores),
        'per_std': np.std(per_scores),
        'pfer_std': np.std(pfer_scores),
        'num_samples': len(references),
        'per_scores': per_scores,
        'pfer_scores': pfer_scores,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Testing PER and PFER Metrics")
    print("=" * 70)

    # Test cases
    test_cases = [
        ("Perfect match", "kæt", "kæt"),
        ("Small difference (aspiration)", "kæt", "kʰæt"),
        ("Vowel difference", "kæt", "kɛt"),
        ("Complete difference", "kæt", "dɑg"),
        ("Length mismatch", "kæt", "kæti"),
        ("Deletion", "kæt", "kt"),
        # Combining diacritic tests (A1 fix)
        ("Syllabic consonant", "bʌtn̩", "bʌtn̩"),
        ("Nasalized flap vs plain", "ɾ̃æ", "ræ"),
        ("Devoiced schwa", "ə̥tʃ", "ətʃ"),
    ]

    print("\nIndividual test cases:")
    print("-" * 70)

    for name, ref, hyp in test_cases:
        per = phone_error_rate(ref, hyp)
        pfer = phone_feature_error_rate(ref, hyp)

        print(f"\n{name}:")
        print(f"  Reference:  {ref}")
        print(f"  Hypothesis: {hyp}")
        print(f"  PER:  {per:6.2f}%")
        print(f"  PFER: {pfer:6.2f}%")
        if per > 0:
            print(f"  PFER/PER ratio: {pfer/per:.3f} (PFER should be ≤ PER)")

    # Batch evaluation
    print("\n" + "=" * 70)
    print("Batch Evaluation")
    print("=" * 70)

    refs = [tc[1] for tc in test_cases]
    hyps = [tc[2] for tc in test_cases]

    results = evaluate_batch(refs, hyps)

    print(f"\nResults over {results['num_samples']} samples:")
    print(f"  Average PER:  {results['per']:.2f}% (±{results['per_std']:.2f}%)")
    print(f"  Average PFER: {results['pfer']:.2f}% (±{results['pfer_std']:.2f}%)")
    print(f"  PFER/PER ratio: {results['pfer']/results['per']:.3f}")

    # Direct tokenization assertions (A1 fix verification)
    print("\n" + "=" * 70)
    print("Tokenization Assertions")
    print("=" * 70)

    assert tokenize_ipa("n̩æp") == ["n̩", "æ", "p"], "syllabic n broken"
    assert tokenize_ipa("ɾ̃æ") == ["ɾ̃", "æ"], "nasalized flap broken"
    assert tokenize_ipa("ə̥tʃ") == ["ə̥", "t", "ʃ"], "devoiced schwa broken"
    assert tokenize_ipa("tʃ") == ["t", "ʃ"], "affricate should split"
    assert tokenize_ipa("ŋ̍") == ["ŋ̍"], "syllabic ng — post-pass should catch"
    assert tokenize_ipa("kæt") == ["k", "æ", "t"], "simple phones unchanged"
    assert tokenize_ipa("m̩") == ["m̩"], "syllabic m broken"
    assert tokenize_ipa("l̩") == ["l̩"], "syllabic l broken"
    assert tokenize_ipa("") == [], "empty string should return empty list"
    print("All tokenization assertions passed!")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
