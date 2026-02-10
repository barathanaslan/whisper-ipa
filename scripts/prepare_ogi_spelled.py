"""
OGI Spelled Speech Dataset Preparation Script
Converts phonetic transcriptions from .ptl files to IPA format
and creates a dataset ready for Whisper fine-tuning.

OGI Spelled Speech is a dataset of people spelling letters and words aloud.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Reuse ARPABET to IPA mapping from TIMIT (OGI uses ARPABET-like codes)
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
    'bcl': '',
    'dcl': '',
    'gcl': '',
    'pcl': '',
    'tcl': '',
    'kcl': '',
    'cl': '',       # generic closure

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
    'em': 'm̩',
    'en': 'n̩',
    'eng': 'ŋ̍',
    'nx': 'ɾ̃',

    # Semivowels and Glides
    'l': 'l',
    'r': 'ɹ',
    'w': 'w',
    'y': 'j',
    'hh': 'h',
    'h': 'h',
    'hv': 'ɦ',
    'el': 'l̩',

    # Vowels - monophthongs
    'iy': 'i',
    'ih': 'ɪ',
    'eh': 'ɛ',
    'ae': 'æ',
    'aa': 'ɑ',
    'ah': 'ʌ',
    'ao': 'ɔ',
    'uh': 'ʊ',
    'uw': 'u',
    'ux': 'ʉ',
    'er': 'ɝ',
    'ax': 'ə',
    'ix': 'ɨ',
    'axr': 'ɚ',
    'ax-h': 'ə̥',

    # Vowels - diphthongs
    'ey': 'eɪ',
    'ay': 'aɪ',
    'oy': 'ɔɪ',
    'aw': 'aʊ',
    'ow': 'oʊ',

    # R-colored vowels
    'ao-r': 'ɔɹ',   # r-colored ao
    'aa-r': 'ɑɹ',   # r-colored aa
    'ae-r': 'æɹ',   # r-colored ae
    'ay-': 'aɪ',    # incomplete diphthong
    'ax-': 'ə',     # incomplete schwa

    # Silence/non-speech
    'pau': '',
    'epi': '',
    '#h': '',       # OGI uses #h for silence
    'h#': '',

    # OGI-specific annotations (non-phonetic markers - should be skipped)
    'br': '',       # breath
    'ls': '',       # lip smack
    'ln': '',       # noise/laugh
    'ns': '',       # nasal sound/noise
    'pv': '',       # pause/voice pause
    'gx': '',       # garbage/unknown sound
    'glot': 'ʔ',    # glottal stop (treat as phonetic)
    'bn': '',       # background noise
    'xs': '',       # extra sound
    'unk': '',      # unknown
    '-': '',        # silence marker
}


def parse_ptl_file(ptl_path: Path) -> List[Tuple[int, int, str]]:
    """
    Parse a .ptl file and return list of (start, end, phoneme) tuples.

    PTL format:
    MillisecondsPerFrame: 3.0
    END OF HEADER
    start_ms  end_ms  phoneme

    Args:
        ptl_path: Path to .ptl file

    Returns:
        List of tuples containing (start_ms, end_ms, phoneme_code)
    """
    phonemes = []
    in_header = True

    with open(ptl_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip header
            if in_header:
                if line == 'END OF HEADER':
                    in_header = False
                continue

            # Parse phoneme line
            parts = line.split()
            if len(parts) >= 3:
                start_ms = int(parts[0])
                end_ms = int(parts[1])
                phoneme = parts[2].lower()
                phonemes.append((start_ms, end_ms, phoneme))

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


def process_ogi_dataset(ogi_root: Path, output_file: Path):
    """
    Process OGI Spelled Speech dataset and create JSON file with audio paths and IPA transcriptions.

    Args:
        ogi_root: Root directory of OGI dataset
        output_file: Output JSON file path
    """
    dataset = []
    speech_root = ogi_root / 'speech'
    handlabl_root = ogi_root / 'handlabl'

    if not speech_root.exists():
        raise ValueError(f"Speech directory not found: {speech_root}")
    if not handlabl_root.exists():
        raise ValueError(f"Handlabl directory not found: {handlabl_root}")

    # Find all .ptl files and match with corresponding .wav files
    for ptl_file in sorted(handlabl_root.rglob('*.ptl')):
        # Construct corresponding wav file path
        # handlabl/sp_lnamp/0/slp_68.ptl -> speech/sp_lnamp/0/slp_68.wav
        relative_path = ptl_file.relative_to(handlabl_root)
        wav_file = speech_root / relative_path.with_suffix('.wav')

        if not wav_file.exists():
            print(f"Warning: Missing .wav file for {ptl_file}")
            continue

        # Parse phoneme file
        try:
            phonemes = parse_ptl_file(ptl_file)
        except Exception as e:
            print(f"Warning: Error parsing {ptl_file}: {e}")
            continue

        # Convert to IPA
        ipa_transcription = convert_arpabet_to_ipa(phonemes)

        # Extract metadata from path
        # e.g., sp_lnamp/0/slp_68.ptl
        parts = relative_path.parts
        corpus_type = parts[0] if len(parts) > 0 else 'unknown'
        subset_id = parts[1] if len(parts) > 1 else 'unknown'

        # Create dataset entry
        entry = {
            'audio_path': str(wav_file.absolute()),
            'ipa_transcription': ipa_transcription,
            'orthographic_text': '',  # OGI doesn't include orthographic text in files
            'corpus_type': corpus_type,
            'subset_id': subset_id,
            'utterance_id': ptl_file.stem,
            'phoneme_count': len([p for p in phonemes if p[2] not in ['#h', 'h#', 'pau', 'epi']]),
            'arpabet_phonemes': [p[2] for p in phonemes],
            'language': 'english'
        }

        dataset.append(entry)

    # Write to JSON file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nOGI Spelled Speech Dataset Statistics:")
    print(f"  Total utterances: {len(dataset)}")
    print(f"  Output file: {output_file}")

    # Print sample entry
    if dataset:
        print(f"\nSample entry:")
        sample = dataset[0]
        print(f"  Utterance: {sample['utterance_id']}")
        print(f"  Corpus type: {sample['corpus_type']}")
        print(f"  IPA:  {sample['ipa_transcription']}")
        print(f"  ARPABET: {' '.join(sample['arpabet_phonemes'][:15])}...")


def main():
    """Main function to process OGI Spelled Speech dataset."""
    # Set paths
    project_root = Path(__file__).parent.parent
    ogi_root = project_root / 'data' / 'TIMIT' / 'ogi_spelled'
    output_dir = project_root / 'data' / 'processed'

    print("=" * 80)
    print("OGI Spelled Speech Dataset Preparation for IPA Transcription")
    print("=" * 80)

    # Process dataset
    output_file = output_dir / 'ogi_spelled_ipa.json'
    process_ogi_dataset(ogi_root, output_file)

    print("\n" + "=" * 80)
    print("OGI Spelled Speech dataset preparation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
