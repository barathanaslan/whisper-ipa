"""
TIMIT Dataset Preparation Script
Converts TIMIT ARPABET phonetic transcriptions to IPA format
and creates a dataset ready for Whisper fine-tuning.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

# ARPABET to IPA mapping based on TIMIT phoneme set
# Reference: TIMIT PHONCODE.DOC
ARPABET_TO_IPA = {
    # Stops
    'b': 'b',
    'd': 'd',
    'g': 'ɡ',
    'p': 'p',
    't': 't',
    'k': 'k',
    'dx': 'ɾ',      # flap
    'q': 'ʔ',       # glottal stop

    # Closures (silence before stop release)
    'bcl': '',      # typically not transcribed in IPA
    'dcl': '',
    'gcl': '',
    'pcl': '',
    'tcl': '',
    'kcl': '',

    # Affricates
    'jh': 'dʒ',
    'ch': 'tʃ',

    # Fricatives
    's': 's',
    'sh': 'ʃ',
    'z': 'z',
    'zh': 'ʒ',
    'f': 'f',
    'th': 'θ',
    'v': 'v',
    'dh': 'ð',

    # Nasals
    'm': 'm',
    'n': 'n',
    'ng': 'ŋ',
    'em': 'm̩',     # syllabic m
    'en': 'n̩',     # syllabic n
    'eng': 'ŋ̍',    # syllabic ng
    'nx': 'ɾ̃',     # nasal flap

    # Semivowels and Glides
    'l': 'l',
    'r': 'ɹ',       # American English r
    'w': 'w',
    'y': 'j',
    'hh': 'h',
    'hv': 'ɦ',      # voiced h
    'el': 'l̩',     # syllabic l

    # Vowels - monophthongs
    'iy': 'i',      # beet
    'ih': 'ɪ',      # bit
    'eh': 'ɛ',      # bet
    'ae': 'æ',      # bat
    'aa': 'ɑ',      # bott
    'ah': 'ʌ',      # but
    'ao': 'ɔ',      # bought
    'uh': 'ʊ',      # book
    'uw': 'u',      # boot
    'ux': 'ʉ',      # fronted u
    'er': 'ɝ',      # bird
    'ax': 'ə',      # about (schwa)
    'ix': 'ɨ',      # debit (reduced vowel)
    'axr': 'ɚ',     # butter (r-colored schwa)
    'ax-h': 'ə̥',   # devoiced schwa

    # Vowels - diphthongs
    'ey': 'eɪ',     # bait
    'ay': 'aɪ',     # bite
    'oy': 'ɔɪ',     # boy
    'aw': 'aʊ',     # bout
    'ow': 'oʊ',     # boat

    # Silence/non-speech
    'pau': '',      # pause
    'epi': '',      # epenthetic silence
    'h#': '',       # begin/end marker
}


def parse_phn_file(phn_path: Path) -> List[Tuple[int, int, str]]:
    """
    Parse a .PHN file and return list of (start, end, phoneme) tuples.

    Args:
        phn_path: Path to .PHN file

    Returns:
        List of tuples containing (start_sample, end_sample, phoneme_code)
    """
    phonemes = []
    with open(phn_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, phoneme = parts
                phonemes.append((int(start), int(end), phoneme.lower()))
    return phonemes


def convert_arpabet_to_ipa(phonemes: List[Tuple[int, int, str]]) -> str:
    """
    Convert ARPABET phoneme sequence to IPA string.

    Args:
        phonemes: List of (start, end, phoneme_code) tuples

    Returns:
        IPA transcription string
    """
    ipa_chars = []
    for start, end, phoneme in phonemes:
        if phoneme in ARPABET_TO_IPA:
            ipa_char = ARPABET_TO_IPA[phoneme]
            if ipa_char:  # Skip empty strings (closures, silences)
                ipa_chars.append(ipa_char)
        else:
            print(f"Warning: Unknown phoneme '{phoneme}' - skipping")

    return ''.join(ipa_chars)


def process_timit_dataset(timit_root: Path, output_file: Path, split: str = 'TRAIN'):
    """
    Process TIMIT dataset and create JSON file with audio paths and IPA transcriptions.

    Args:
        timit_root: Root directory of TIMIT dataset (contains TRAIN/TEST)
        output_file: Output JSON file path
        split: 'TRAIN' or 'TEST'
    """
    dataset = []
    split_dir = timit_root / split

    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    # Iterate through all dialect regions
    for dr_dir in sorted(split_dir.glob('DR*')):
        if not dr_dir.is_dir():
            continue

        # Iterate through all speakers
        for speaker_dir in sorted(dr_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            # Iterate through all utterances
            for wav_file in sorted(speaker_dir.glob('*.WAV')):
                phn_file = wav_file.with_suffix('.PHN')
                txt_file = wav_file.with_suffix('.TXT')

                if not phn_file.exists():
                    print(f"Warning: Missing .PHN file for {wav_file}")
                    continue

                # Parse phoneme file
                phonemes = parse_phn_file(phn_file)

                # Convert to IPA
                ipa_transcription = convert_arpabet_to_ipa(phonemes)

                # Read orthographic text if available
                orthographic_text = ""
                if txt_file.exists():
                    with open(txt_file, 'r') as f:
                        line = f.read().strip()
                        # Format: start_sample end_sample text
                        parts = line.split(maxsplit=2)
                        if len(parts) == 3:
                            orthographic_text = parts[2]

                # Create dataset entry
                entry = {
                    'audio_path': str(wav_file.absolute()),
                    'ipa_transcription': ipa_transcription,
                    'orthographic_text': orthographic_text,
                    'speaker_id': speaker_dir.name,
                    'dialect_region': dr_dir.name,
                    'utterance_id': wav_file.stem,
                    'phoneme_count': len([p for p in phonemes if p[2] not in ['pau', 'epi', 'h#']]),
                    'arpabet_phonemes': [p[2] for p in phonemes]
                }

                dataset.append(entry)

    # Write to JSON file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n{split} Dataset Statistics:")
    print(f"  Total utterances: {len(dataset)}")
    print(f"  Output file: {output_file}")

    # Print sample entry
    if dataset:
        print(f"\nSample entry:")
        sample = dataset[0]
        print(f"  Audio: {sample['utterance_id']}")
        print(f"  Text: {sample['orthographic_text']}")
        print(f"  IPA:  {sample['ipa_transcription']}")
        print(f"  ARPABET: {' '.join(sample['arpabet_phonemes'][:10])}...")


def main():
    """Main function to process both TRAIN and TEST splits."""
    # Set paths
    project_root = Path(__file__).parent.parent
    timit_root = project_root / 'data' / 'TIMIT' / 'timit' / 'TIMIT'
    output_dir = project_root / 'data' / 'processed'

    print("=" * 80)
    print("TIMIT Dataset Preparation for IPA Transcription")
    print("=" * 80)

    # Process TRAIN split
    print("\nProcessing TRAIN split...")
    train_output = output_dir / 'timit_train_ipa.json'
    process_timit_dataset(timit_root, train_output, split='TRAIN')

    # Process TEST split
    print("\n" + "-" * 80)
    print("\nProcessing TEST split...")
    test_output = output_dir / 'timit_test_ipa.json'
    process_timit_dataset(timit_root, test_output, split='TEST')

    print("\n" + "=" * 80)
    print("Dataset preparation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
