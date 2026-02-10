"""
Verify and normalize IPA transcriptions in our datasets.
According to WHISPER_IPA_TOKEN_RESEARCH.md, NFC normalization is CRITICAL.
"""

import json
import unicodedata
from pathlib import Path
from collections import defaultdict

def check_normalization(text: str) -> bool:
    """Check if text is already NFC normalized."""
    return unicodedata.normalize('NFC', text) == text

def analyze_dataset(json_path: Path):
    """Analyze IPA normalization status of a dataset."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {json_path.name}")
    print(f"{'='*80}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    total_samples = len(data)
    normalized_count = 0
    needs_normalization = 0
    normalization_changes = []

    # Check each entry
    for i, entry in enumerate(data):
        ipa = entry.get('ipa_transcription', '')

        if check_normalization(ipa):
            normalized_count += 1
        else:
            needs_normalization += 1
            normalized_ipa = unicodedata.normalize('NFC', ipa)
            normalization_changes.append({
                'index': i,
                'original': ipa,
                'normalized': normalized_ipa,
                'utterance_id': entry.get('utterance_id', entry.get('speaker_id', 'unknown'))
            })

    # Report
    print(f"\nTotal samples: {total_samples}")
    print(f"Already normalized: {normalized_count} ({normalized_count/total_samples*100:.1f}%)")
    print(f"Needs normalization: {needs_normalization} ({needs_normalization/total_samples*100:.1f}%)")

    if normalization_changes:
        print(f"\n⚠️  WARNING: {needs_normalization} samples need normalization!")
        print("\nFirst 5 examples of normalization changes:")
        for change in normalization_changes[:5]:
            print(f"\n  Sample {change['index']} ({change['utterance_id']}):")
            print(f"    Original:   '{change['original']}'")
            print(f"    Normalized: '{change['normalized']}'")
            print(f"    Original bytes:   {change['original'].encode('utf-8')}")
            print(f"    Normalized bytes: {change['normalized'].encode('utf-8')}")
    else:
        print("\n✅ All IPA transcriptions are properly normalized!")

    return needs_normalization > 0, normalization_changes

def apply_normalization(json_path: Path, output_path: Path = None):
    """Apply NFC normalization to all IPA transcriptions."""
    if output_path is None:
        output_path = json_path.parent / f"{json_path.stem}_normalized.json"

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Normalize all IPA transcriptions
    for entry in data:
        if 'ipa_transcription' in entry:
            entry['ipa_transcription'] = unicodedata.normalize('NFC', entry['ipa_transcription'])

    # Save normalized data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Normalized dataset saved to: {output_path}")
    return output_path

def main():
    """Main function to check and normalize all datasets."""
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'data' / 'processed'

    print("=" * 80)
    print("IPA NORMALIZATION VERIFICATION")
    print("=" * 80)
    print("\nAccording to WHISPER_IPA_TOKEN_RESEARCH.md:")
    print("  - NFC Unicode normalization is CRITICAL for training")
    print("  - Ensures consistent tokenization across all samples")
    print("  - Prevents training confusion from different Unicode representations")
    print()

    datasets_to_check = [
        'timit_train_ipa.json',
        'timit_test_ipa.json',
        'metu_turkish_ipa.json',
        'ogi_spelled_ipa.json',
        'combined_train_ipa.json',
        'combined_test_ipa.json'
    ]

    needs_fix = {}

    for dataset_name in datasets_to_check:
        dataset_path = processed_dir / dataset_name
        if dataset_path.exists():
            needs_norm, changes = analyze_dataset(dataset_path)
            if needs_norm:
                needs_fix[dataset_name] = changes
        else:
            print(f"\n⚠️  Dataset not found: {dataset_name}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if needs_fix:
        print(f"\n⚠️  {len(needs_fix)} dataset(s) need normalization:")
        for name in needs_fix:
            print(f"  - {name}")

        print("\nShould we apply normalization? (y/n): ", end="")
        response = input().strip().lower()

        if response == 'y':
            print("\nApplying normalization...")
            for dataset_name in needs_fix:
                dataset_path = processed_dir / dataset_name
                # Overwrite original file
                apply_normalization(dataset_path, dataset_path)

            print("\n✅ All datasets normalized!")
        else:
            print("\nSkipped normalization. You can run this script again to normalize.")
    else:
        print("\n✅ All datasets are properly normalized!")
        print("Safe to proceed with training.")

if __name__ == '__main__':
    main()
