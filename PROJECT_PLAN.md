# Whisper IPA Preprint: 3-Week Project Plan

## Context

This project fine-tunes OpenAI's Whisper for IPA phonetic transcription. The baseline to beat is Taguchi et al. ("multipa", arXiv:2308.03917) which uses wav2vec2 + CTC and achieves **21.2% PFER** zero-shot. Our hypothesis: Whisper's 680k-hour pre-training and native IPA token support make it a superior backbone. We have their exact test data (annotation sheets from Ariga & Hamanishi). The goal is an arXiv preprint in ~3 weeks with a 2-3 person team.

---

## Team Streams

| Stream | Focus | Owner |
|--------|-------|-------|
| **A: Infrastructure & Eval** | Fix metrics, logging, test data parsing, figures | Person 1 |
| **B: Experiment 1 (Whisper + Their Data)** | Download CommonVoice, preprocess via multipa repo, train | Person 2 |
| **C: Experiment 2 (Whisper + TIMIT)** | TIMIT audit, primary training runs, hyperparameter tuning | Person 3 |

---

## WEEK 1: Foundation & Data

### Day 1-2

**R1. Literature review: recent advances in speech-to-IPA** (ALL — shared task)
- We took a ~1 year break. Search for papers published since Aug 2023 (when Taguchi et al. appeared) on automatic phonetic/phonemic transcription, speech-to-IPA, and universal phone recognition.
- Key questions: Has anyone beaten 21.2% PFER on the zero-shot benchmark? Are there new models, datasets, or techniques we need to cite or compare against?
- Check arXiv (cs.CL, eess.AS), Interspeech 2024, ACL 2024, ICASSP 2024-2025 proceedings.
- Notable: `neurlang/ipa-whisper-base` and `ipa-whisper-small` appeared on HuggingFace (March 2025) — fine-tuned Whisper on Common Voice 21, 70+ languages, 15k synthetic IPA-labeled samples via gruut phonemizer. Mixed results on tonal languages. Evaluate whether this is a baseline we must compare against.
- Deliverable: 1-page summary of relevant new work with citations, fed into Related Work section of the paper.

**R2. Review Whisper GitHub Discussion #318 and community progress**
- Thread: https://github.com/openai/whisper/discussions/318 ("Transcribe to IPA") — 30+ comments, active since Oct 2022
- Also check: Discussion #1875 ("Generate phonetical transcription/tokens") and HuggingFace Discussion #86 ("Phoneme Recognition")
- Key developments to track:
  - `neurlang/ipa-whisper-base` release and its approach (new tokenizer? same tokenizer? what data?)
  - Sanchit Gandhi's (HuggingFace) recommended approach: new phoneme tokenizer + resize embeddings + fine-tune — how does this compare to our decoder-only fine-tuning with existing BPE tokens?
  - Any other community models or experiments posted in the threads
- Deliverable: Summary of community approaches, how they differ from ours, and whether any of them should be included as baselines.

**A1. Fix IPA phone tokenization** (CRITICAL — blocks all evaluation)
- `scripts/evaluate_ipa.py` → `tokenize_ipa()` is character-level (`list(text)`)
- Must handle multi-character phones: `tʃ`, `dʒ`, `eɪ`, `aɪ`, `aʊ`, `oʊ` and combining diacritics (`n̩`)
- Use phone inventory from `prepare_timit_dataset.py` ARPABET_TO_IPA mapping (lines 14-93) as reference
- Consider `panphon.FeatureTable().word_fts()` for proper segmentation

**A2. Create `requirements.txt`** from venv

**B1. Download CommonVoice 11.0** for 7 languages (ja, pl, mt, hu, fi, el, ta)
- Source: HuggingFace `mozilla-foundation/common_voice_11_0` or commonvoice.mozilla.org
- ~20-50 GB total. Start immediately — this is on the critical path

**B2. Clone `https://github.com/ctaguchi/multipa`**, install deps, read their `preprocess.py`

**C1. Audit TIMIT data**
- TIMIT = English-only, 8 US dialect regions (DR1-DR8), 630 speakers
- Verify: 4,620 TRAIN / 1,680 TEST samples with matching .PHN files
- Document: TIMIT .PHN files are narrow phonetic (61 phone labels including closures); our ARPABET→IPA mapping drops closures → effectively broad phonetic/phonemic
- Deliverable: audit document with exact counts and phonetic-vs-phonemic classification

### Day 3-4

**A3. Parse zero-shot test annotations into evaluation-ready JSON**
- Input: `test/IPA_annotation_sheet_Ariga.xlsx`, `test/IPA_annotation_sheet_Hamanishi.xlsx`
- WAVs: 1,549 files in `test/test/`
- Match WAV IDs to IPA transcriptions, handle "Poor quality" flags
- Separate by language (Luganda, Upper Sorbian, Hakha Chin, Tatar) — cross-reference multipa repo for language labels
- Output: `data/processed/zeroshot_test.json` with both annotators' transcriptions

**A4. Add structured training logging** to `scripts/train_whisper_ipa.py`
- CSV logger: step, loss, lr, step_time_sec, samples_per_sec, timestamp
- Validation metrics (PER, PFER) appended at each validation step
- Save `training_config.json` at training start
- Record total wall-clock time at training end

**B3. Run multipa preprocessing** on downloaded CommonVoice data
- Produces IPA labels from orthographic text via Epitran/manual G2P rules
- Spot-check IPA quality for each language

**C2. Start Experiment 2a: Whisper-small + TIMIT**
- Model: `mlx-community/whisper-small-mlx`
- Data: `data/processed/timit_train_ipa.json` (4,620 samples)
- Settings: 10,000 steps, batch_size=12, lr=1e-5, validate_every=1000, save_every=1000
- ~4-5 hours on M3 Ultra

### Day 5-7

**A5. Validate PFER metric matches paper's definition**
- Paper: PanPhon 24 features, substitution = feature_mismatches/24, insertion/deletion = 1
- Cross-check our `evaluate_ipa.py` PFERCalculator implementation
- Test on known examples

**A6. Compute IAA between Ariga and Hamanishi**
- Should get ~19.6% PFER matching the paper
- This validates (a) annotation parsing is correct and (b) our PFER metric is equivalent

**A7. Write `scripts/prepare_commonvoice_dataset.py`**
- Convert multipa preprocessed output → our JSON format
- Apply NFC Unicode normalization
- Create 3 variants: 1k/lang, 2k/lang, full (matching paper's experimental settings)

**B4. Convert CommonVoice data to our JSON format** using A7 script
- Output: `commonvoice_train_1k.json` (7k samples), `commonvoice_train_2k.json`, `commonvoice_train_full.json`

**B5. Extract CommonVoice supervised test set**
- 100 samples per trained language (700 total), from CommonVoice test split
- Check multipa repo for their exact test split; if unavailable, create our own

**C3. Evaluate Experiment 2a** on TIMIT test (1,680 samples) with fixed tokenization
- Record PER and PFER
- Review loss curve, decide if more steps needed

**C4. Start Experiment 2b: Whisper-large-v3 + TIMIT**
- `mlx-community/whisper-large-v3-mlx` (128 mel bins)
- batch_size=4-8, ~8-10 hours

---

## WEEK 2: Core Experiments & Evaluation

### Day 8-9

**B6. Experiment 1a: Whisper-small + CommonVoice 1k/lang** (7,000 samples)
- Same hyperparameters as Exp 2a
- This is the KEY comparison: same data as paper, different model
- ~5 hours

**B7. Experiment 1b: Whisper-small + CommonVoice full** (~50k samples)

**C5. Evaluate Experiment 2b** (Whisper-large-v3 + TIMIT)

**A8. Build unified evaluation harness** (`scripts/run_full_evaluation.py`)
- Tests any checkpoint against: TIMIT test, CV supervised test, zero-shot test
- Outputs comparison table with PER, PFER, confidence intervals

### Day 10-11

**B8. Evaluate Experiment 1a on all test sets** — the money comparison
- Zero-shot: compare directly with paper's 21.2% PFER
- Supervised: compare with paper's 5.7% PFER

**B9. Evaluate Experiment 1b** on all test sets

**C6. Experiment 1c: Whisper-large-v3 + CommonVoice 1k/lang** (P1 priority)
- Best model + their data = isolates architecture advantage at same data size

**A9. Evaluate base (untrained) Whisper** on zero-shot test as baseline reference

### Day 12-14

**A10. Compile final results table** — all experiments vs. paper benchmarks vs. human IAA

**A11. Error analysis on zero-shot test**
- Per-language PER/PFER breakdown (Luganda, Upper Sorbian, Hakha Chin, Tatar)
- Common phone confusions

**A12. Generate publication figures**
- Training loss curves (all experiments overlaid)
- PER/PFER bar chart: our models vs. paper vs. human IAA
- Per-language zero-shot breakdown
- Model size vs. performance

**C7. Efficiency comparison table**
- Training time, hardware, estimated power per experiment
- Paper: 4x GTX1080Ti, ~4h, ~7.5kW vs. our M3 Ultra single machine

**ALL: Freeze experimental results** — no more training after day 14

---

## WEEK 3: Paper Writing & Submission

### Day 15-16 (Parallel drafting)

**Person A → Introduction + Related Work** (~2 pages)
- Motivation: IPA transcription for language documentation
- Gap: current SOTA uses wav2vec2+CTC; Whisper untested for this task
- Contributions: (1) first Whisper-for-IPA evaluation, (2) new SOTA on zero-shot benchmark, (3) consumer hardware efficiency
- Related Work must incorporate findings from R1 and R2: cite any post-2023 papers, position against `neurlang/ipa-whisper-*` and community approaches from Discussion #318

**Person B → Methods: Data + Model** (~2 pages)
- Whisper architecture, decoder-only fine-tuning, IPA tokenizer coverage
- TIMIT description, CommonVoice reuse, zero-shot test set
- Phonetic vs. phonemic discussion
- Training details: AdamW, lr, gradient clipping, MLX

**Person C → Experiments + Results** (~2-3 pages)
- Experiment table, metrics definitions (PER, PFER)
- Main results table (the centerpiece)
- Key findings: architecture advantage, data quality effect, model size scaling
- Efficiency comparison

### Day 17-18

**ALL → Discussion + Conclusion + Abstract**
- Why Whisper works better (pre-training scale, native IPA tokens, encoder-decoder advantage over CTC)
- Limitations: TIMIT is English-only, ARPABET mapping choices, no tones
- Future work: more languages, phonetic vs phonemic, real-time deployment
- Abstract: 150-250 words

**ALL → References (BibTeX)**
- Key: Radford et al. 2022 (Whisper), Baevski et al. 2020 (wav2vec2), Taguchi et al. 2023 (multipa), TIMIT, PanPhon, CommonVoice, Epitran

### Day 19-20

**ALL → Review, revise, polish**
- Cross-check all numbers against frozen results
- Ensure reproducibility from paper description alone
- LaTeX formatting (arXiv-compatible template, cs.CL + eess.AS categories)

### Day 21

**Submit to arXiv**

---

## Critical Path

```
B1 (Download CV) → B3 (Preprocess) → B4/B5 (Convert) → B6 (Train Exp 1a) → B8 (Evaluate) → A10 (Results table) → Week 3 paper
```

**Risk**: CommonVoice download+preprocessing takes 4-5 days. If it slips, paper writing gets squeezed.

**Mitigation**: TIMIT experiments (Stream C) are independent and can start Day 2. If Exp 2 (Whisper+TIMIT) already shows strong results, paper can be structured around those with Exp 1 results added when ready.

---

## Experiments Priority

| # | Experiment | Model | Data | Samples | Priority | ~Time |
|---|-----------|-------|------|---------|----------|-------|
| 2a | Whisper-small + TIMIT | whisper-small | TIMIT | 4,620 | **P0** | 4h |
| 1a | Whisper-small + CV 1k | whisper-small | CommonVoice | 7,000 | **P0** | 5h |
| 2b | Whisper-large + TIMIT | whisper-large-v3 | TIMIT | 4,620 | P1 | 8h |
| 1b | Whisper-small + CV full | whisper-small | CommonVoice | ~50k | P1 | 6h |
| 1c | Whisper-large + CV 1k | whisper-large-v3 | CommonVoice | 7,000 | P1 | 10h |
| 2c | Whisper-medium + TIMIT | whisper-medium | TIMIT | 4,620 | P2 | 6h |

All evaluated on: TIMIT test (1,680), CV supervised test (700), zero-shot test (~100, 4 unseen languages)

---

## Essential vs Nice-to-Have

**Must-have for submission:**
- Literature review of post-2023 work including `neurlang/ipa-whisper-*` models (R1, R2)
- Fixed phone-level tokenization (A1)
- IAA validation ≈ 19.6% PFER (A6) — proves our metrics match
- At least Exp 2a (Whisper+TIMIT) fully evaluated
- At least Exp 1a (Whisper+CV 1k) for direct model comparison
- Zero-shot evaluation on shared test set
- Training loss curves
- Paper draft

**Nice-to-have (strengthens paper):**
- Large model experiments (2b, 1c)
- Model-size ablation (tiny→large)
- Per-language error analysis with confusion matrices
- Cross-dataset generalization (TIMIT model on CV test and vice versa)
- Power/efficiency comparison

---

## Key Files to Modify

| File | Change | Blocks |
|------|--------|--------|
| `scripts/evaluate_ipa.py` | Fix `tokenize_ipa()` to phone-level | All evaluation |
| `scripts/train_whisper_ipa.py` | Add CSV logging, config save | All training |
| `test/IPA_annotation_sheet_*.xlsx` | Parse to JSON | Zero-shot eval |
| `scripts/evaluate_model.py` | Extend to unified eval harness | Results compilation |
| NEW: `scripts/prepare_commonvoice_dataset.py` | Convert multipa output to our format | Exp 1 training |
