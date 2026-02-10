"""
METU Turkish Dataset Preparation Script
Converts METUbet phonetic transcriptions to IPA format
and creates a dataset ready for Whisper fine-tuning.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

# METUbet to IPA mapping for Turkish phonemes
# Based on Turkish phonology and METU Turkish corpus documentation
METUBET_TO_IPA = {
    # Vowels - Turkish has 8 vowels with front/back and rounded/unrounded distinctions
    'A': 'a',       # open back unrounded vowel (as in "ağaç")
    'AA': 'aː',     # long a
    'E': 'e',       # close-mid front unrounded vowel (as in "el")
    'EE': 'eː',     # long e
    'I': 'ɯ',       # close back unrounded vowel (Turkish dotless i: ı)
    'IY': 'ɯː',     # long ı
    'O': 'o',       # close-mid back rounded vowel (as in "ok")
    'OE': 'ø',      # close-mid front rounded vowel (as in "ö" - German ö)
    'U': 'u',       # close back rounded vowel (as in "un")
    'UE': 'y',      # close front rounded vowel (as in "ü" - like German ü)

    # Consonants - stops
    'B': 'b',       # voiced bilabial stop
    'P': 'p',       # voiceless bilabial stop
    'D': 'd',       # voiced alveolar stop
    'T': 't',       # voiceless alveolar stop
    'G': 'ɡ',       # voiced velar stop
    'GG': 'ɟ',      # voiced palatal stop (soft g before front vowels)
    'K': 'k',       # voiceless velar stop
    'KK': 'c',      # voiceless palatal stop (before front vowels)

    # Affricates
    'C': 'tʃ',      # voiceless postalveolar affricate (Turkish ç)
    'J': 'dʒ',      # voiced postalveolar affricate (Turkish c)
    'CH': 'tʃ',     # alternative for Turkish ç

    # Fricatives
    'F': 'f',       # voiceless labiodental fricative
    'V': 'v',       # voiced labiodental fricative
    'VV': 'v',      # voiced labiodental fricative (geminate)
    'S': 's',       # voiceless alveolar fricative
    'Z': 'z',       # voiced alveolar fricative
    'SH': 'ʃ',      # voiceless postalveolar fricative (Turkish ş)
    'ZH': 'ʒ',      # voiced postalveolar fricative (Turkish j)
    'H': 'h',       # voiceless glottal fricative
    'RH': 'ɣ',      # voiced velar fricative (Turkish soft g: ğ - often silent/lengthening)

    # Nasals
    'M': 'm',       # bilabial nasal
    'N': 'n',       # alveolar nasal
    'NN': 'ŋ',      # velar nasal (before velar stops)

    # Liquids
    'L': 'l',       # alveolar lateral approximant
    'LL': 'ɫ',      # velarized/dark l (allophone)
    'R': 'ɾ',       # alveolar tap (Turkish r is typically a tap/trill)
    'RR': 'r',      # alveolar trill (geminate or emphatic r)

    # Glides
    'Y': 'j',       # palatal approximant (Turkish y)

    # Silence
    'SIL': '',      # silence marker
}


def parse_phn_file(phn_path: Path) -> List[Tuple[int, int, str]]:
    """
    Parse a .phn file and return list of (start, end, phoneme) tuples.

    Args:
        phn_path: Path to .phn file

    Returns:
        List of tuples containing (start_sample, end_sample, phoneme_code)
    """
    phonemes = []
    with open(phn_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, phoneme = parts
                phonemes.append((int(start), int(end), phoneme))
    return phonemes


def convert_metubet_to_ipa(phonemes: List[Tuple[int, int, str]]) -> str:
    """
    Convert METUbet phoneme sequence to IPA string.

    Args:
        phonemes: List of (start, end, phoneme_code) tuples

    Returns:
        IPA transcription string
    """
    ipa_chars = []
    for start, end, phoneme in phonemes:
        if phoneme in METUBET_TO_IPA:
            ipa_char = METUBET_TO_IPA[phoneme]
            if ipa_char:  # Skip empty strings (silence)
                ipa_chars.append(ipa_char)
        else:
            print(f"Warning: Unknown phoneme '{phoneme}' - skipping")

    return ''.join(ipa_chars)


def process_metu_dataset(metu_root: Path, output_file: Path):
    """
    Process METU Turkish dataset and create JSON file with audio paths and IPA transcriptions.

    Args:
        metu_root: Root directory of METU dataset
        output_file: Output JSON file path
    """
    dataset = []
    speech_text_dir = metu_root / 'data' / 'speech-text'
    alignments_dir = metu_root / 'data' / 'alignments'

    if not speech_text_dir.exists():
        raise ValueError(f"Speech-text directory not found: {speech_text_dir}")
    if not alignments_dir.exists():
        raise ValueError(f"Alignments directory not found: {alignments_dir}")

    # Iterate through all speakers
    for speaker_dir in sorted(speech_text_dir.glob('s*')):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name
        alignment_speaker_dir = alignments_dir / speaker_id

        if not alignment_speaker_dir.exists():
            print(f"Warning: No alignment directory for speaker {speaker_id}")
            continue

        # Iterate through all utterances
        for wav_file in sorted(speaker_dir.glob('*.wav')):
            phn_file = alignment_speaker_dir / f"{wav_file.stem}.phn"
            txt_file = wav_file.with_suffix('.txt')

            if not phn_file.exists():
                print(f"Warning: Missing .phn file for {wav_file}")
                continue

            # Parse phoneme file
            phonemes = parse_phn_file(phn_file)

            # Convert to IPA
            ipa_transcription = convert_metubet_to_ipa(phonemes)

            # Read orthographic text if available
            orthographic_text = ""
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    orthographic_text = f.read().strip()

            # Create dataset entry
            entry = {
                'audio_path': str(wav_file.absolute()),
                'ipa_transcription': ipa_transcription,
                'orthographic_text': orthographic_text,
                'speaker_id': speaker_id,
                'utterance_id': wav_file.stem,
                'phoneme_count': len([p for p in phonemes if p[2] != 'SIL']),
                'metubet_phonemes': [p[2] for p in phonemes],
                'language': 'turkish'
            }

            dataset.append(entry)

    # Write to JSON file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nMETU Turkish Dataset Statistics:")
    print(f"  Total utterances: {len(dataset)}")
    print(f"  Output file: {output_file}")

    # Print sample entry
    if dataset:
        print(f"\nSample entry:")
        sample = dataset[0]
        print(f"  Speaker: {sample['speaker_id']}")
        print(f"  Utterance: {sample['utterance_id']}")
        print(f"  Text: {sample['orthographic_text']}")
        print(f"  IPA:  {sample['ipa_transcription'][:50]}...")
        print(f"  METUbet: {' '.join(sample['metubet_phonemes'][:10])}...")


def main():
    """Main function to process METU Turkish dataset."""
    # Set paths
    project_root = Path(__file__).parent.parent
    metu_root = project_root / 'data' / 'TIMIT' / 'metu_turkish'
    output_dir = project_root / 'data' / 'processed'

    print("=" * 80)
    print("METU Turkish Dataset Preparation for IPA Transcription")
    print("=" * 80)

    # Process dataset
    output_file = output_dir / 'metu_turkish_ipa.json'
    process_metu_dataset(metu_root, output_file)

    print("\n" + "=" * 80)
    print("METU Turkish dataset preparation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
