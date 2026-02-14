"""
Combine all processed datasets into unified training and test sets.
"""

import json
from pathlib import Path
from typing import List, Dict

def load_json(file_path: Path) -> List[Dict]:
    """Load JSON dataset file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: Path):
    """Save JSON dataset file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    """Main function to combine all datasets."""
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'data' / 'processed'

    print("=" * 80)
    print("Combining All Datasets for Voice-to-IPA Training")
    print("=" * 80)
    print()

    # Load all datasets
    print("Loading datasets...")
    timit_train = load_json(processed_dir / 'timit_train_ipa.json')
    timit_test = load_json(processed_dir / 'timit_test_ipa.json')
    metu_turkish = load_json(processed_dir / 'metu_turkish_ipa.json')
    ogi_spelled = load_json(processed_dir / 'ogi_spelled_ipa.json')

    print(f"  TIMIT Train: {len(timit_train):,} samples")
    print(f"  TIMIT Test:  {len(timit_test):,} samples")
    print(f"  METU Turkish: {len(metu_turkish):,} samples")
    print(f"  OGI Spelled: {len(ogi_spelled):,} samples")
    print()

    # Add dataset source to each entry
    for entry in timit_train:
        entry['dataset_source'] = 'timit'
        entry['split'] = 'train'

    for entry in timit_test:
        entry['dataset_source'] = 'timit'
        entry['split'] = 'test'

    for entry in metu_turkish:
        entry['dataset_source'] = 'metu_turkish'
        entry['split'] = 'train'  # We'll use METU for training

    for entry in ogi_spelled:
        entry['dataset_source'] = 'ogi_spelled'
        entry['split'] = 'train'  # We'll use OGI for training

    # Combine datasets
    # Training set: TIMIT train + METU Turkish + OGI Spelled
    combined_train = timit_train + metu_turkish + ogi_spelled

    # Test set: TIMIT test (keep as official test set)
    combined_test = timit_test

    # Save combined datasets
    print("Saving combined datasets...")
    train_output = processed_dir / 'combined_train_ipa.json'
    test_output = processed_dir / 'combined_test_ipa.json'

    save_json(combined_train, train_output)
    save_json(combined_test, test_output)

    print(f"  Training set: {len(combined_train):,} samples -> {train_output}")
    print(f"  Test set:     {len(combined_test):,} samples -> {test_output}")
    print()

    # Statistics
    print("Combined Dataset Statistics:")
    print("=" * 80)
    print(f"Training samples:  {len(combined_train):,}")
    print(f"  - TIMIT English:  {len(timit_train):,}")
    print(f"  - METU Turkish:   {len(metu_turkish):,}")
    print(f"  - OGI Spelled:    {len(ogi_spelled):,}")
    print()
    print(f"Test samples:      {len(combined_test):,}")
    print(f"  - TIMIT English:  {len(timit_test):,}")
    print()
    print(f"Total samples:     {len(combined_train) + len(combined_test):,}")
    print()

    # Estimate duration
    estimated_hours = (len(combined_train) + len(combined_test)) * 3 / 3600
    print(f"Estimated duration: ~{estimated_hours:.1f} hours")
    print()

    # Language breakdown
    english_count = len(timit_train) + len(timit_test) + len(ogi_spelled)
    turkish_count = len(metu_turkish)
    print(f"Language breakdown:")
    print(f"  English: {english_count:,} samples ({english_count/(english_count+turkish_count)*100:.1f}%)")
    print(f"  Turkish: {turkish_count:,} samples ({turkish_count/(english_count+turkish_count)*100:.1f}%)")
    print()
    print("=" * 80)
    print("Dataset combination complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
