"""
Parse Taguchi et al. zero-shot test annotations (A3).

Reads both annotators' Excel sheets + multipa's test_data.csv,
matches to WAV files, cross-references to identify gold annotator,
outputs data/processed/zeroshot_test.json.
"""

import json
import os
import unicodedata
from pathlib import Path

import pandas as pd

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "test"
WAV_DIR = TEST_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

ARIGA_XLSX = TEST_DIR / "IPA_annotation_sheet_Ariga.xlsx"
HAMANISHI_XLSX = TEST_DIR / "IPA_annotation_sheet_Hamanishi.xlsx"
TEST_DATA_CSV = PROJECT_ROOT / "references" / "multipa" / "test_data.csv"

# Poor quality IDs per annotator (from paper/Excel inspection)
POOR_QUALITY_ARIGA = {41, 75}
POOR_QUALITY_HAMANISHI = {41, 80}


def normalize_ipa(text):
    """NFC normalize and strip whitespace from IPA string."""
    if not isinstance(text, str):
        return None
    text = unicodedata.normalize('NFC', text).strip()
    if not text or text == '?':
        return None
    return text


def parse_excel(path, poor_quality_ids):
    """Parse an annotator's Excel sheet.

    Returns dict mapping int ID -> {ipa, poor_quality, elapsed_time}.
    """
    df = pd.read_excel(path, engine='openpyxl')

    entries = {}
    for _, row in df.iterrows():
        # ID column
        raw_id = row.get('ID')
        if pd.isna(raw_id):
            continue
        try:
            entry_id = int(raw_id)
        except (ValueError, TypeError):
            continue

        ipa = normalize_ipa(str(row.get('IPA', '')) if pd.notna(row.get('IPA')) else None)
        poor = entry_id in poor_quality_ids

        elapsed = row.get('Elapsed Time (sec)')
        if pd.notna(elapsed):
            try:
                elapsed = float(elapsed)
            except (ValueError, TypeError):
                elapsed = None
        else:
            elapsed = None

        entries[entry_id] = {
            'ipa': ipa,
            'poor_quality': poor,
            'elapsed_time': elapsed,
        }

    return entries


def parse_test_data_csv(path):
    """Parse multipa's test_data.csv.

    Returns dict mapping int ID -> {ipa, done}.
    Handles header artifacts (extra 'Total:' row).
    """
    df = pd.read_csv(path)

    entries = {}
    for _, row in df.iterrows():
        raw_id = row.get('ID')
        if pd.isna(raw_id):
            continue
        try:
            entry_id = int(raw_id)
        except (ValueError, TypeError):
            continue

        done = row.get('Done')
        try:
            done = int(done) == 1
        except (ValueError, TypeError):
            done = False

        ipa = normalize_ipa(str(row.get('IPA', '')) if pd.notna(row.get('IPA')) else None)

        entries[entry_id] = {
            'ipa': ipa,
            'done': done,
        }

    return entries


def build_wav_index(wav_dir):
    """Scan WAV directory and map integer IDs to absolute paths.

    Filenames are like '1_123808906.wav'. Skips 'Copy of' files.
    """
    index = {}
    for f in wav_dir.iterdir():
        if not f.suffix.lower() == '.wav':
            continue
        if f.name.startswith('Copy of'):
            continue
        # ID is the part before the first underscore
        parts = f.stem.split('_', 1)
        try:
            file_id = int(parts[0])
        except (ValueError, IndexError):
            continue
        index[file_id] = str(f.resolve())
    return index


def cross_reference(test_csv_entries, ariga_entries, hamanishi_entries):
    """Compare test_data.csv IPA against both annotators.

    Returns (ariga_matches, hamanishi_matches, total_compared).
    """
    ariga_matches = 0
    hamanishi_matches = 0
    total = 0

    for entry_id, csv_entry in test_csv_entries.items():
        csv_ipa = csv_entry['ipa']
        if csv_ipa is None:
            continue

        # Strip spaces for comparison (annotators may use different spacing)
        csv_clean = csv_ipa.replace(' ', '')
        total += 1

        ariga_ipa = ariga_entries.get(entry_id, {}).get('ipa')
        if ariga_ipa is not None and ariga_ipa.replace(' ', '') == csv_clean:
            ariga_matches += 1

        hamanishi_ipa = hamanishi_entries.get(entry_id, {}).get('ipa')
        if hamanishi_ipa is not None and hamanishi_ipa.replace(' ', '') == csv_clean:
            hamanishi_matches += 1

    return ariga_matches, hamanishi_matches, total


def main():
    print("=" * 70)
    print("Parsing Zero-Shot Test Annotations (A3)")
    print("=" * 70)

    # 1a. Parse Excel sheets
    print("\n--- Parsing Excel sheets ---")
    ariga = parse_excel(ARIGA_XLSX, POOR_QUALITY_ARIGA)
    print(f"Ariga: {len(ariga)} entries, "
          f"{sum(1 for e in ariga.values() if e['ipa'] is not None)} with IPA")

    hamanishi = parse_excel(HAMANISHI_XLSX, POOR_QUALITY_HAMANISHI)
    print(f"Hamanishi: {len(hamanishi)} entries, "
          f"{sum(1 for e in hamanishi.values() if e['ipa'] is not None)} with IPA")

    # 1b. Parse test_data.csv
    print("\n--- Parsing test_data.csv ---")
    test_csv = parse_test_data_csv(TEST_DATA_CSV)
    done_count = sum(1 for e in test_csv.values() if e['done'])
    ipa_count = sum(1 for e in test_csv.values() if e['ipa'] is not None)
    print(f"test_data.csv: {len(test_csv)} entries, {done_count} done, {ipa_count} with IPA")

    # 1c. Build WAV index
    print("\n--- Building WAV file index ---")
    wav_index = build_wav_index(WAV_DIR)
    print(f"WAV files found: {len(wav_index)}")

    # 1d. Cross-reference
    print("\n--- Cross-referencing test_data.csv against annotators ---")
    a_match, h_match, total = cross_reference(test_csv, ariga, hamanishi)
    print(f"Total IPA entries in test_data.csv: {total}")
    print(f"Exact matches with Ariga:     {a_match}/{total} ({a_match/total*100:.1f}%)")
    print(f"Exact matches with Hamanishi:  {h_match}/{total} ({h_match/total*100:.1f}%)")

    if a_match > h_match:
        gold_annotator = "ariga"
        print(f"\n=> Gold standard: ARIGA (test_data.csv matches Ariga's transcriptions)")
    elif h_match > a_match:
        gold_annotator = "hamanishi"
        print(f"\n=> Gold standard: HAMANISHI (test_data.csv matches Hamanishi's transcriptions)")
    else:
        gold_annotator = "unknown"
        print(f"\n=> INCONCLUSIVE — equal matches. Manual inspection needed.")

    # Show mismatches for debugging
    if gold_annotator != "unknown":
        gold_entries = ariga if gold_annotator == "ariga" else hamanishi
        mismatches = []
        for eid, csv_entry in test_csv.items():
            csv_ipa = csv_entry['ipa']
            if csv_ipa is None:
                continue
            gold_ipa = gold_entries.get(eid, {}).get('ipa')
            if gold_ipa is not None and gold_ipa.replace(' ', '') != csv_ipa.replace(' ', ''):
                mismatches.append(eid)
        if mismatches:
            print(f"  Mismatched IDs with {gold_annotator}: {mismatches}")

    # 1e. Build output JSON
    print("\n--- Building output JSON ---")
    all_ids = sorted(set(ariga.keys()) | set(hamanishi.keys()))

    output = []
    missing_wav = 0
    for entry_id in all_ids:
        a_entry = ariga.get(entry_id, {})
        h_entry = hamanishi.get(entry_id, {})
        csv_entry = test_csv.get(entry_id, {})

        # Only include entries where at least one annotator has IPA
        if a_entry.get('ipa') is None and h_entry.get('ipa') is None:
            continue

        a_ipa = a_entry.get('ipa')
        h_ipa = h_entry.get('ipa')
        a_poor = a_entry.get('poor_quality', False)
        h_poor = h_entry.get('poor_quality', False)

        has_both = a_ipa is not None and h_ipa is not None
        usable = has_both and not a_poor and not h_poor

        # Gold IPA from the identified gold annotator
        if gold_annotator == "ariga":
            gold_ipa = a_ipa
        elif gold_annotator == "hamanishi":
            gold_ipa = h_ipa
        else:
            gold_ipa = None

        wav_path = wav_index.get(entry_id)
        if wav_path is None:
            missing_wav += 1

        record = {
            "id": entry_id,
            "audio_path": wav_path,
            "ipa_ariga": a_ipa,
            "ipa_hamanishi": h_ipa,
            "ipa_test_csv": csv_entry.get('ipa'),
            "poor_quality_ariga": a_poor,
            "poor_quality_hamanishi": h_poor,
            "has_both_annotators": has_both,
            "usable_for_iaa": usable,
            "gold_annotator": gold_annotator,
            "gold_ipa": gold_ipa,
            "language": None,
            "dataset_source": "multipa_zeroshot_test",
        }
        output.append(record)

    # Summary
    iaa_count = sum(1 for r in output if r['usable_for_iaa'])
    both_count = sum(1 for r in output if r['has_both_annotators'])
    print(f"Total entries: {len(output)}")
    print(f"Both annotators: {both_count}")
    print(f"Usable for IAA: {iaa_count}")
    print(f"Missing WAV files: {missing_wav}")

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "zeroshot_test.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nWritten to: {output_path}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
