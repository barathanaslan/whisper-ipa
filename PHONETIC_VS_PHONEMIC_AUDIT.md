# Phonetic vs Phonemic Audit: Raw Evidence

This document collects raw evidence from three sources to determine whether each data source produces **phonetic** (narrow, allophonic detail) or **phonemic** (broad, contrastive distinctions only) transcriptions. Final classification is left to the research team.

---

## Source 1: Taguchi et al. Paper (IPA_PAPER.pdf) + multipa Repository

### Paper Text Evidence

| Location | Quote | Implies |
|----------|-------|---------|
| Title | "Universal Automatic **Phonetic** Transcription into the International Phonetic Alphabet" | Phonetic |
| Section 1 | "IPA, which is the standard used in **phonemic and phonetic** transcription" | Ambiguous — both mentioned |
| Section 3.1 | "Linguistic resources...often contain IPA transcriptions that are **not phonetic but phonemic**" | They frame phonemic as the lesser option |
| Section 3.1 | "the output of these G2P tools is **phonemic**, and their quality is not guaranteed" | G2P tools produce phonemic; they acknowledge this |
| Section 3.1 | "we particularly chose languages that have a **consistent orthography-to-pronunciation mapping**" | Their G2P is reliable because spelling ≈ pronunciation |
| Section 3.1 | "Non-phonetic characters including punctuation symbols and spaces were removed...tones and other **suprasegmental elements were not included**" | They strip suprasegmentals |
| Section 3.2 | "Since our goal is to train a model applicable to **unseen languages**, we did not use encoder-decoder models" | They avoid Whisper by design |
| Section 4.2 | "phone error rate (PER)" / "phone feature error rate (PFER)" | Uses "phone" not "phoneme" |
| Table 2 | Zero-shot test: "**Manual annotation**" by 2 trained annotators | Humans listened to audio → likely phonetic |

### multipa Repository Evidence (references/multipa/)

**README.md:**
> "MultIPA is yet another automatic speech transcription model into **phonetic IPA**. The idea is that, if we train a multilingual speech-to-IPA model with enough amount of good **phoneme representations**, the model's output will be **approximated to phonetic transcriptions**."

Note the tension: they say "phoneme representations" in training but "phonetic transcriptions" as output goal.

**G2P tools used (preprocess.py):**
- Hungarian, Polish: `Epitran` library (produces phonetic/surface-form IPA, not underlying phonemic)
- Japanese: Custom converter using MeCab + romanization + manual phonetic rules (e.g., geminate `pp` → `pː`)
- Finnish, Greek, Maltese: Custom rules-based converters
  - Maltese converter includes context-dependent voicing rules (phonetic behavior)
- Tamil: Epitran + manual allophonic rules (e.g., voicing between sonorants)
- English: `eng_to_ipa` module referencing CMU dictionary

**Allophonic filtering (data_utils.py lines 373-376):**
```python
# Filter IPA (don't include samples containing "t͡ɕ" and "d͡ʑ";
# they are subject to allophonic variations)
train = train.filter(lambda batch: "t͡ɕ" not in batch["ipa"] and "d͡ʑ" not in batch["ipa"])
```
They filter out phones with allophonic variation — this only makes sense if they're operating at the phonetic level.

**Vocabulary (full_vocab_ipa.txt):** 292 IPA symbols from PanPhon — a phonetic-level inventory, not a reduced phonemic set.

**Code comment (main.py lines 16-17):**
```python
# Change this function later at some point to create vocabulary based on
# phonemes, not on characters
```
They acknowledge their current vocabulary is character-level (phonetic), not phoneme-level.

**Test data (test_data.csv):** Contains diacritics (nasality `ɑ̃`), length marks, tie-bar affricates (`t͡ʃ`, `d͡ʒ`). Human annotators transcribed from audio without being told what languages were included.

### Key Tension in the Paper's Data

Their **training data** comes from G2P tools that convert orthographic text → IPA. G2P tools inherently produce outputs closer to phonemic (they map spelling to pronunciation rules, not actual acoustic events). However, the languages were chosen because their orthography closely matches pronunciation, making the G2P output a reasonable approximation of phonetic surface forms.

Their **test data** (zero-shot) is manually annotated by humans listening to audio — this is inherently phonetic since humans transcribe what they hear.

---

## Source 2: TIMIT Dataset

### TIMIT Official Documentation

**PHONCODE.DOC** (official TIMIT phone code documentation):
> "This file contains a table of all the **phonemic and phonetic** symbols used in the TIMIT lexicon and in the **phonetic transcriptions**."

It explicitly distinguishes:
- The **lexicon** (TIMITDIC.TXT) = "**quasi-phonemic**" representation
- The **.PHN files** = "**phonetic transcription**"

**TIMITDIC.DOC:**
> "The symbols in the lexical representation are abstract, **quasi-phonemic** marks representing the underlying sounds and typically correspond to a variety of different sounds in the actual recordings."

**README.DOC:** describes .PHN files as "Time-aligned **phonetic** transcription."

### TIMIT Raw .PHN Files: 61 Unique Phone Labels

The .PHN files contain these allophonic labels NOT in the lexicon:

| Label | Meaning | Count | Phonetic Detail |
|-------|---------|-------|----------------|
| bcl, dcl, gcl, pcl, tcl, kcl | Stop closure intervals | 24,292 total | Closure separate from release burst |
| dx | Alveolar flap | 3,649 | Allophone of /t/ and /d/ |
| nx | Nasal flap | 1,331 | Allophone (as in "winner") |
| q | Glottal stop | 4,834 | Allophone of /t/ or vowel onset marker |
| hv | Voiced h | 1,523 | Allophone of /h/ intervocalically |
| ux | Fronted u | 2,488 | Allophone of /uw/ in alveolar context |
| ax-h | Devoiced schwa | 493 | Between voiceless consonants |
| ix | Reduced high-front vowel | 11,587 | Allophone of unstressed vowels |
| axr | R-colored schwa (unstressed) | 4,790 | Distinct from stressed er/ɝ |

### Cross-Speaker Variation (SA1: "She had your dark suit in greasy wash water all year")

The same sentence transcribed differently per speaker — proof of phonetic (not phonemic) transcription:

| Word | Dictionary (quasi-phonemic) | Speaker FCJF0 (.PHN) | Speaker FDML0 (.PHN) |
|------|---------------------------|----------------------|----------------------|
| she | /sh iy/ | sh ix | sh iy |
| had | /hh ae d/ | hv eh | hv ae |
| your | /y uh r/ | dcl jh ih | dcl d y er |
| suit | /s uw t/ | s ux | s uw tcl t |
| water | /w ao t axr/ | w ao dx axr | w ao dx ax |

Different speakers → different phone labels for same words → phonetic transcription of actual speech.

### Our Preprocessing (prepare_timit_dataset.py)

**What it does:**
- Takes the 61-phone phonetic .PHN transcriptions as input
- Maps all 52 non-silence phones to unique IPA symbols (1-to-1, NO merges)
- Drops 9 labels: 6 closures (bcl/dcl/gcl/pcl/tcl/kcl) + 3 silence markers (pau/epi/h#)
- Removes word boundaries (continuous string output)
- Preserves ALL allophonic distinctions: dx→ɾ, nx→ɾ̃, q→ʔ, hv→ɦ, ux→ʉ, ax-h→ə̥, ix→ɨ

**What is preserved from TIMIT's phonetic detail:**
- Flaps (ɾ, ɾ̃) — allophones of /t,d,n/
- Glottal stops (ʔ) — allophonic insertions
- Voiced h (ɦ) — allophone of /h/
- Fronted u (ʉ) — allophone of /u/
- Devoiced schwa (ə̥) — allophonic reduction
- Reduced vowels (ɨ) — distinct from full schwa (ə)
- Speaker-specific variation in the same words

**What is lost from TIMIT's phonetic detail:**
- Stop closures (bcl/dcl/etc.) are simply dropped, NOT merged with release
- No aspiration marking (pʰ, tʰ, kʰ)
- No unreleased stop marking (p̚, t̚, k̚)
- No vowel nasalization
- No velarized/dark l (ɫ)

**Output inventory:** 46 unique IPA characters (from 52 distinct phone→IPA mappings, some being multi-character like tʃ, dʒ, eɪ, etc.)

### Sample Output

```
"She had your dark suit in greasy wash water all year."
→ ʃɨɦɛdʒɪdʌksʉʔn̩ɡɹɨsɨwɔʃwɔɾɚɔljɪɚ  (speaker FCJF0)
→ ʃiɦædjʊdɑɹksʉʔn̩ɡɹisiwɑʃwɑɾɚʔɑljɨɚ  (speaker FDAW0)
```

Different outputs for same sentence = phonetic transcription reflecting actual pronunciation.

---

## Source 3: Zero-Shot Test Annotations (Ariga & Hamanishi)

### Sheet Structure

Both annotators' Excel sheets have columns: ID, Poor quality, Done, IPA, Elapsed time, Other

| Annotator | Total with IPA | ID Range | Poor Quality |
|-----------|---------------|----------|-------------|
| Ariga | 126 entries | 1-547 (102 contiguous + 24 scattered) | 2 (ID 41, 75) |
| Hamanishi | 100 entries | 0-101 (contiguous) | 2 (ID 41, 80) |

Annotation time: Ariga median 3.0 min/sample, Hamanishi median 1.4 min/sample.

### Transcription Convention Evidence

**No bracket notation:** Neither annotator uses [ ] (phonetic) or / / (phonemic) brackets — just bare IPA strings.

**No stress marks:** Zero primary stress (ˈ) or secondary stress (ˌ) markers in either annotator's output.

**No syllable boundaries:** No dots (.) or other syllabic markers.

**No tone marks:** Consistent with the paper's statement that "tones and other suprasegmental elements were not included."

### Sample Transcriptions (Ariga, IDs 1-10)

```
ID 1: okunkuntaudiɹiɹabunakiɹiɹakubakiɹiɹa
ID 2: svojepobudovanjenamozekisebiomjeɹiti
ID 3: okunkujomuvaɹaokuɹabiɹaenvuɹababiɹi
ID 4: poʃemutovonadziɹajetvodiʃiɹomsmisʃeodnojemanijenabiɹoʒidannom
ID 5: mugjozazadakokubusaʃunavezaɹa
ID 6: gakoʃanbavoʃonbagaɹaɹagaɹaɹa
ID 7: etonuʒnobudetzdjeɹatspomoʃjuaɹtiskusʃtvjennojukoziizdaʃniɹoɹjopʃuɾɛtsa
ID 8: kugubonononovomufumanjabiɹikjozasotojobana
ID 9: ɹubejuːatuvakubuɹubuɹubuɹu
ID 10: bazvjazanajabɑ̃dɛɹoʃizkɹasojukɹafimisbuziɹa
```

**Notable features:** Mostly basic Latin and IPA characters. Nasality diacritic appears (ɑ̃ in ID 10). Length mark appears (uː in ID 9). Retroflex/specific IPA symbols appear (ɹ throughout). Languages appear to be Luganda, Upper Sorbian, Hakha Chin, Tatar (4 unseen languages, randomly mixed).

### Ariga's Scattered Entries (IDs > 102) — More Detail

These later entries show MORE phonetic detail:
```
ID 155: nokʷopakʰaᴊ         — contains aspiration mark kʰ, labialization kʷ
ID 295: alietinumpʰɯa        — contains aspiration mark pʰ
ID 325: ikʲɯbəɾəɯʃtrəlr     — contains palatalization kʲ
ID 339: ilepʰanʦɯnmo         — aspiration pʰ, affricate ʦ
ID 387: hænɪkɛ:ɣɛnɪm         — length mark :, velar fricative ɣ
ID 394: ʤʰibel               — aspirated affricate ʤʰ
ID 413: tɕnzɔkanirnraɪ       — alveolopalatal tɕ
ID 427: kɯnɯniagsʲi          — palatalization sʲ
```

Ariga's later annotations include aspiration (kʰ, pʰ, ʤʰ), palatalization (kʲ, sʲ), labialization (kʷ) — clearly phonetic-level detail. The earlier entries (1-101) are more conservative.

---

## Summary: Evidence At A Glance

| Data Source | Self-Description | Actual Content | Key Observation |
|-------------|-----------------|----------------|-----------------|
| **Taguchi training data** | "phonetic IPA" (README) | G2P from orthography → IPA | G2P is inherently closer to phonemic; but languages chosen for orthography≈pronunciation |
| **Taguchi test data** | "Manual annotation" | Human-transcribed from audio | Inherently phonetic (transcribing what was heard); some allophonic detail present |
| **TIMIT .PHN files** | "phonetic transcription" (official docs) | 61 phones with allophonic variants | Clearly phonetic: speaker variation, closures, flaps, glottal stops |
| **Our TIMIT preprocessing** | (no self-description) | 52 phones → 52 IPA, closures dropped | Retains most allophonic detail from .PHN; drops closures and silence |
| **Ariga annotations** | (no self-description) | Mix of basic IPA + some phonetic diacritics | Later entries show aspiration, palatalization; earlier entries more conservative |
| **Hamanishi annotations** | (no self-description) | Basic IPA, fewer diacritics | More consistent, less allophonic detail than Ariga |
