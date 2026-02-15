"""
Preprocess local CommonVoice dataset to produce audio + IPA transcription pairs.
Supports 3 modes: raw, filtered, improved.
"""
import pandas as pd
import json
import os
import re
import sys
import argparse

# Language code to dataset folder mapping
LANG_FOLDER = {
    "fi": "finnish",
    "el": "greek",
    "hu": "hungarian",
    "ja": "japanese",
    "mt": "maltese",
    "pl": "polish",
    "ta": "tamil",
}

# Epitran instance cache (expensive to create)
_epitran_cache = {}

# Converter class cache
_converters = {}


def _get_converter(lang, mode="raw"):
    """Get the converter for a language. If mode='improved', use improved converters for fi/ta."""
    cache_key = (lang, mode)
    if cache_key in _converters:
        return _converters[cache_key]

    if mode == "improved" and lang == "fi":
        from converters_improved.finnish_to_ipa import Finnish2IPA
        _converters[cache_key] = Finnish2IPA
    elif mode == "improved" and lang == "ta":
        from converters_improved.tamil_to_ipa import Tamil2IPA
        _converters[cache_key] = Tamil2IPA
    elif lang == "ja":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../multipa/converter"))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../multipa"))
        from japanese_to_ipa import Japanese2IPA
        _converters[cache_key] = Japanese2IPA
    elif lang == "mt":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../multipa/converter"))
        from maltese_to_ipa import Maltese2IPA
        _converters[cache_key] = Maltese2IPA
    elif lang == "fi":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../multipa/converter"))
        from finnish_to_ipa import Finnish2IPA
        _converters[cache_key] = Finnish2IPA
    elif lang == "el":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../multipa/converter"))
        from greek_to_ipa import Greek2IPA
        _converters[cache_key] = Greek2IPA
    elif lang == "ta":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../multipa/converter"))
        from tamil_to_ipa import Tamil2IPA
        _converters[cache_key] = Tamil2IPA
    elif lang in ("hu", "pl"):
        from epitran import Epitran
        _converters[cache_key] = Epitran
    return _converters.get(cache_key)


def _get_epitran(lang_code):
    """Get cached Epitran instance."""
    if lang_code not in _epitran_cache:
        from epitran import Epitran
        _epitran_cache[lang_code] = Epitran(lang_code)
    return _epitran_cache[lang_code]


def text_to_ipa(sent: str, lang: str, mode: str = "raw") -> str:
    """Convert orthographic text to IPA using language-specific converter."""
    _get_converter(lang, mode)
    cache_key = (lang, mode)
    converter_cls = _converters[cache_key]

    if lang == "ja":
        converter = converter_cls()
        ipa = converter.remove_ja_punct(sent)
        ipa = converter.convert_sentence_to_ipa(ipa)
    elif lang == "mt":
        ipa = converter_cls.maltese_generate_ipa(sent)
    elif lang == "fi":
        ipa = converter_cls.finnish_generate_ipa(sent)
    elif lang == "el":
        ipa = converter_cls.greek_generate_ipa(sent)
    elif lang == "hu":
        ipa = re.findall(r"[\s\w]", sent.lower(), re.MULTILINE)
        ipa = "".join(ipa)
        epi = _get_epitran("hun-Latn")
        ipa = epi.transliterate(ipa)
    elif lang == "pl":
        ipa = re.findall(r"[\s\w]", sent.lower(), re.MULTILINE)
        ipa = "".join(ipa)
        epi = _get_epitran("pol-Latn")
        ipa = epi.transliterate(ipa)
    elif lang == "ta":
        ipa = converter_cls.tamil_generate_ipa(sent)
    else:
        raise ValueError(f"Unknown language: {lang}")
    return "".join(ipa.split())


def get_audio_duration(audio_path):
    """Get audio duration in seconds using mutagen (reads MP3 header, no decoding)."""
    try:
        from mutagen.mp3 import MP3
        audio = MP3(audio_path)
        return audio.info.length
    except Exception:
        return None


def apply_filters(df):
    """Apply 5 quality filters to a dataframe. Returns filtered df."""
    before = len(df)

    # 1. Remove corrupted rows (len > 500)
    df = df[df["sentence"].astype(str).str.len() <= 500]

    # 2. Remove garbage (len < 2)
    df = df[df["sentence"].astype(str).str.len() >= 2]

    # 3. Remove down-voted (more than 1 negative vote)
    if "down_votes" in df.columns:
        df = df[df["down_votes"] <= 1]

    # 4. Remove duplicate sentences (keep first)
    df = df.drop_duplicates(subset="sentence", keep="first")

    # 5. Remove URLs
    df = df[~df["sentence"].astype(str).str.contains(r"https?://", regex=True, na=False)]

    after = len(df)
    print(f"  Filtering: {before} → {after} ({before - after} removed)")
    return df.reset_index(drop=True)


def process_language(lang, dataset_root, num_samples, mode="raw", split="train"):
    """Load local CommonVoice data, convert to IPA, return list of dicts."""
    folder = LANG_FOLDER[lang]
    lang_dir = os.path.join(dataset_root, folder, lang)
    # CommonVoice uses "dev.tsv" for the validation split
    tsv_name = "dev.tsv" if split == "validation" else f"{split}.tsv"
    tsv_path = os.path.join(lang_dir, tsv_name)
    clips_dir = os.path.join(lang_dir, "clips")

    if not os.path.exists(tsv_path):
        print(f"  WARNING: TSV not found: {tsv_path}, skipping {lang}")
        return []

    df = pd.read_csv(tsv_path, sep="\t", on_bad_lines="warn")
    print(f"[{lang}] Loaded {len(df)} rows from {split}.tsv")

    # Apply filters for filtered/improved modes
    if mode in ("filtered", "improved"):
        df = apply_filters(df)

    # Filter Tamil sentences containing "ச" (matches multipa preprocess.py)
    if lang == "ta":
        before_ta = len(df)
        df = df[~df["sentence"].str.contains("ச", na=False)]
        after_ta = len(df)
        if before_ta != after_ta:
            print(f"  Tamil ச filter: {before_ta} → {after_ta} ({before_ta - after_ta} removed)")

    # Filter out audio clips > 6 seconds
    print(f"[{lang}] Filtering audio duration > 6s...")
    durations = []
    for _, row in df.iterrows():
        clip_filename = row["path"]
        audio_path = os.path.join(clips_dir, clip_filename)
        dur = get_audio_duration(audio_path)
        durations.append(dur)
    df["_duration"] = durations
    before_dur = len(df)
    df = df[df["_duration"].notna() & (df["_duration"] <= 6.0)]
    after_dur = len(df)
    if before_dur != after_dur:
        print(f"  Duration filter: {before_dur} → {after_dur} ({before_dur - after_dur} removed)")

    # Random sample selection (deterministic with seed)
    limit = min(num_samples, len(df))
    if limit < num_samples:
        print(f"  WARNING: only {limit} samples available for {lang} (requested {num_samples})")
    df = df.sample(n=limit, random_state=42)
    print(f"[{lang}] Selected {limit} samples")

    results = []
    for idx, row in df.iterrows():
        sentence = row["sentence"]
        clip_filename = row["path"]
        audio_path_abs = os.path.join(clips_dir, clip_filename)

        if not os.path.exists(audio_path_abs):
            continue

        try:
            ipa = text_to_ipa(sentence, lang, mode)
        except Exception as e:
            print(f"  WARNING: IPA conversion failed for '{sentence}': {e}, skipping")
            continue

        # Post-conversion validation: drop empty, too-short, or corrupted IPA
        if not ipa or len(ipa) < 2:
            print(f"  WARNING: empty/short IPA for '{sentence}', skipping")
            continue
        if re.search(r"[0-9]", ipa):
            print(f"  WARNING: IPA contains digits (TSV parse error?) for '{sentence}', skipping")
            continue

        # Store relative path from dataset root
        audio_path_rel = os.path.join("dataset", folder, lang, "clips", clip_filename)

        results.append({
            "audio_path": audio_path_rel,
            "sentence": sentence,
            "ipa_transcription": ipa,
            "locale": lang,
            "path": clip_filename,
            "dataset_source": "commonvoice",
            "speaker_id": row.get("client_id", "unknown"),
        })

    print(f"[{lang}] Successfully processed {len(results)} samples")
    return results


def main(languages=None, num_samples=1000, mode="raw", dataset_root=None, output_dir=None, split="train"):
    """Main entry point. Can be called programmatically or from CLI."""
    if languages is None:
        languages = list(LANG_FOLDER.keys())

    if dataset_root is None:
        dataset_root = os.path.join(os.path.dirname(__file__), "../../dataset")

    if output_dir is None:
        mode_dirs = {"raw": "v1_raw", "filtered": "v2_filtered", "improved": "v3_improved"}
        output_dir = os.path.join(os.path.dirname(__file__), "../data", mode_dirs[mode])

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Mode: {mode} | Split: {split} | Samples: {num_samples} | Output: {output_dir}")
    print(f"{'='*60}")

    all_results = []
    counts = {}
    for lang in languages:
        results = process_language(lang, dataset_root, num_samples, mode, split)
        all_results.extend(results)
        counts[lang] = len(results)

        lang_output = os.path.join(output_dir, f"{lang}_{split}_ipa.json")
        with open(lang_output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[{lang}] Saved to {lang_output}")

    # Save combined
    combined_output = os.path.join(output_dir, f"combined_{split}_ipa.json")
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved combined dataset ({len(all_results)} samples) to {combined_output}")

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess local CommonVoice data to IPA")
    parser.add_argument("-l", "--languages", nargs="+", default=list(LANG_FOLDER.keys()),
                        help="Language codes (e.g. fi el hu ja mt pl ta)")
    parser.add_argument("-n", "--num_samples", type=int, default=1000)
    parser.add_argument("--mode", choices=["raw", "filtered", "improved"], default="raw")
    parser.add_argument("--dataset_root", type=str,
                        default=os.path.join(os.path.dirname(__file__), "../../dataset"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    main(
        languages=args.languages,
        num_samples=args.num_samples,
        mode=args.mode,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        split=args.split,
    )
