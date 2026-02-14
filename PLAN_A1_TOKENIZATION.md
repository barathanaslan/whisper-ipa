# Plan: A1 — Fix IPA Phone Tokenization

**Branch:** `A1-fix-ipa-tokenization`
**File to modify:** `scripts/evaluate_ipa.py`
**Blocks:** All evaluation, IAA validation (A6), and indirectly the gold-label question

---

## 1. The Problem

`tokenize_ipa()` at `evaluate_ipa.py:35` does `return list(text)`, splitting on every Unicode codepoint. This breaks phones with combining diacritics — a base character + modifier gets split into two "phones".

**6 broken phones from TIMIT inventory:**

| ARPABET | IPA | Unicode | Current (broken) | Correct |
|---------|-----|---------|-------------------|---------|
| em | m̩ | U+006D + U+0329 | `['m', '̩']` | `['m̩']` |
| en | n̩ | U+006E + U+0329 | `['n', '̩']` | `['n̩']` |
| el | l̩ | U+006C + U+0329 | `['l', '̩']` | `['l̩']` |
| eng | ŋ̍ | U+014B + U+030D | `['ŋ', '̍']` | `['ŋ̍']` |
| nx | ɾ̃ | U+027E + U+0303 | `['ɾ', '̃']` | `['ɾ̃']` |
| ax-h | ə̥ | U+0259 + U+0325 | `['ə', '̥']` | `['ə̥']` |

Orphaned combining diacritics (e.g. bare `̩`) get zero-vector features in PanPhon, inflating both PER and PFER. ~1,000+ tokenization errors across the 1,680-sample TIMIT test set.

**Affricates and diphthongs (`tʃ`, `dʒ`, `eɪ`, `aɪ`, etc.) are NOT broken** — they're sequences of two base characters. Both panphon and the multipa code treat them as separate segments, which is linguistically standard when no tie bar (U+0361) is present. Our TIMIT mapping produces `tʃ` without tie bar, so these correctly tokenize as `['t', 'ʃ']`.

---

## 2. Implementation Options

### Option A: panphon `ipa_segs()` (recommended)

panphon is already imported at `evaluate_ipa.py:12`. Its `FeatureTable().ipa_segs(text)` properly groups base characters with their combining diacritics.

**Verified behavior:**
```
'n̩æp'  → ['n̩', 'æ', 'p']     ✓ syllabic kept together
'ə̥tʃ'  → ['ə̥', 't', 'ʃ']    ✓ voiceless diacritic kept, affricate split (correct)
'ɾ̃æ'   → ['ɾ̃', 'æ']          ✓ nasalized flap kept together
'kʰæt'  → ['kʰ', 'æ', 't']    ✓ aspiration kept (matters for test data)
'kʲɯ'   → ['kʲ', 'ɯ']         ✓ palatalization kept
```

**Known limitation:** `ŋ̍` (U+014B + U+030D) → `['ŋ']` — panphon drops the combining vertical line above. Only 5 occurrences in TIMIT test. Decide whether to handle this edge case or accept the minor data loss.

**Pros:** Battle-tested, handles the full Unicode combining range, already in the dependency chain, also correctly handles test annotation phones (aspiration, palatalization, labialization from Ariga's annotations).

**Cons:** ŋ̍ edge case. Minor: creates a FeatureTable instance per call unless cached.

### Option B: Regex-based (matches multipa approach)

The multipa repo uses `retokenize_ipa()` at `references/multipa/utils.py:86-105`. It iterates through the string and appends characters in the range U+02B0–U+036F to the previous character. Also handles tie bars (U+0361) which join three characters.

**Pros:** Matches the baseline's exact approach. Handles ŋ̍ correctly. No external dependency beyond regex. Handles tie bars explicitly.

**Cons:** Must be manually maintained. Doesn't benefit from panphon's Unicode knowledge.

### Option C: Hybrid

Use panphon `ipa_segs()` as the primary segmenter, with a post-pass to re-attach any orphaned combining marks (catches the ŋ̍ case and any future edge cases).

---

## 3. What to Change

### Primary: `tokenize_ipa()` at `evaluate_ipa.py:15-35`

Replace the function body. Keep the same signature `(text: str) -> List[str]`. The rest of the pipeline (`phone_error_rate`, `PFERCalculator.phone_feature_error_rate`, `evaluate_batch`, and all call sites in `evaluate_model.py`) calls `tokenize_ipa()` and expects a `List[str]` — no other changes needed.

### Secondary: Update test cases at `evaluate_ipa.py:243-250`

Add test cases that exercise combining diacritics:
- A pair where reference has `n̩` — verify it's counted as 1 phone, not 2
- A pair comparing `ɾ̃` vs `r` — verify correct PER (1 substitution, not 1 sub + 1 deletion)

### No changes needed to:
- `PFERCalculator` class (lines 66-171) — receives phones from tokenize_ipa, unchanged
- `get_phone_features()` (line 74) — will now receive `'n̩'` instead of `'n'` and `'̩'` separately
- `evaluate_model.py` — calls `phone_error_rate` and `phone_feature_error_rate` which internally call tokenize_ipa
- `train_whisper_ipa.py` — if it calls evaluation functions, they'll automatically benefit

---

## 4. Verification Steps

### Step 1: Unit test tokenize_ipa itself

Run the `if __name__ == "__main__"` block at `evaluate_ipa.py:237-284`. The existing test cases should still pass (they use simple single-codepoint phones). New diacritic test cases should also pass.

### Step 2: Spot-check on TIMIT data

```bash
source venv/bin/activate
python3 -c "
from scripts.evaluate_ipa import tokenize_ipa
import json
with open('data/processed/timit_test_ipa.json') as f:
    data = json.load(f)
# Pick a sample with syllabic consonants
for d in data[:50]:
    t = d['ipa_transcription']
    if 'n̩' in t or 'l̩' in t or 'ɾ̃' in t:
        phones = tokenize_ipa(t)
        print(f'{t[:40]}... -> {len(phones)} phones')
        # Verify no orphaned combining marks
        import unicodedata
        for p in phones:
            if len(p) == 1 and unicodedata.category(p).startswith('M'):
                print(f'  BUG: orphaned combining mark {p!r}')
        break
"
```

### Step 3: Validate IAA (this is the real test)

**Context:** Taguchi et al. report 19.6% PFER as inter-annotator agreement between their two human annotators (Ariga and Hamanishi). We have both annotation sheets at `test/IPA_annotation_sheet_Ariga.xlsx` and `test/IPA_annotation_sheet_Hamanishi.xlsx`.

**Important caveats for IAA validation:**

1. **Gold label is unspecified.** The paper says two annotators transcribed the same audio but doesn't say which one was used as the gold standard for model evaluation and which for IAA. This means:
   - Compute PFER(Ariga_as_ref, Hamanishi_as_hyp) AND PFER(Hamanishi_as_ref, Ariga_as_hyp)
   - These will differ because PFER is asymmetric (it divides by reference length)
   - Whichever direction gives ~19.6% tells us which annotator was likely the reference

2. **Annotator overlap.** Ariga has 126 entries (IDs 1-547, non-contiguous), Hamanishi has 100 entries (IDs 0-101). Only the overlapping IDs can be compared. Need to match by ID.

3. **Poor quality flags.** Both annotators flagged some samples as poor quality (Ariga: IDs 41, 75; Hamanishi: IDs 41, 80). The paper may have excluded these.

4. **Parsing not yet done.** Task A3 (parse Excel sheets to JSON) hasn't been implemented. To validate IAA, you need to either:
   - Write a quick script using `openpyxl` to extract IPA columns from both sheets
   - Or do A3 first, then validate IAA

5. **Normalization matters.** Before comparing, both transcriptions should be Unicode NFC normalized. The paper states "Non-phonetic characters including punctuation symbols and spaces were removed" and "tones and other suprasegmental elements were not included."

**If the result is NOT ~19.6%:** There may be additional preprocessing the paper applied that we're missing, or there may be a difference in how PFER is calculated (see Section 5 below).

---

## 5. Critical Discovery: PFER Calculation Discrepancy

**Our implementation** (`evaluate_ipa.py:95-120`):
```python
mismatches = np.sum(feat1 != feat2)
cost = mismatches / self.num_features  # mismatches / 24
```

**Taguchi's implementation** (`references/multipa/utils.py:156`):
```python
penalty = 1 - cos_sim(t1_f, t2_f)  # cosine distance
```

These are NOT the same metric. Our code counts binary feature mismatches (Hamming-style), their code uses cosine distance on feature vectors. For the same phone pair, these will produce different substitution costs.

**Additional difference:** Taguchi's `combine_features()` (utils.py:107-122) sums feature vectors for multi-character phones (e.g., for `tʃ`, it adds the feature vector of `t` and `ʃ`). Our code takes only the first segment's features via `word_to_vector_list(phone)[0]`.

**What this means:** Even with perfect tokenization, our PFER numbers may not match theirs exactly due to the different distance function. This should be investigated as a follow-up (possibly task A5: "Validate PFER metric matches paper's definition"). For A1, the priority is fixing tokenization — the PFER formula can be aligned separately.

**Where to find more info:**
- Paper's PFER definition: `IPA_PAPER.pdf` Section 4.2
- Taguchi's full evaluation code: `references/multipa/utils.py` (functions `LPhD_combined`, `cos_sim`, `combine_features`, `preprocessing_combine`)
- PanPhon feature table: `references/multipa/full_vocab_ipa.txt` (292 symbols) — compare with panphon's built-in table to check if they use the same feature set
- Our PFER: `scripts/evaluate_ipa.py` lines 66-171

---

## 6. Scope of This Task

**In scope (A1):**
- Fix `tokenize_ipa()` to properly handle combining diacritics
- Update test cases in `evaluate_ipa.py`
- Verify fix on TIMIT test data (no orphaned combining marks)

**Out of scope (but documented for follow-up):**
- A3: Parse annotation Excel sheets to JSON
- A5: Validate PFER formula matches paper (cosine vs Hamming discrepancy)
- A6: Compute IAA between Ariga and Hamanishi (depends on A3)
- Normalization/preprocessing alignment with Taguchi's pipeline
- Gold label determination (depends on A3 + A6)

---

## 7. Files Reference

| File | What to look at | Why |
|------|----------------|-----|
| `scripts/evaluate_ipa.py:15-35` | `tokenize_ipa()` — **the function to fix** | Primary target |
| `scripts/evaluate_ipa.py:66-171` | `PFERCalculator` class | Downstream consumer, no changes needed but understand how it uses phones |
| `scripts/evaluate_ipa.py:237-284` | `__main__` test block | Update with diacritic test cases |
| `scripts/evaluate_model.py:213-214` | Call sites for PER/PFER | Verify no changes needed |
| `scripts/prepare_timit_dataset.py:14-93` | ARPABET_TO_IPA mapping | Source of the 6 problematic multi-codepoint phones |
| `references/multipa/utils.py:86-105` | `retokenize_ipa()` | Taguchi's tokenization approach (Option B reference) |
| `references/multipa/utils.py:139-172` | `LPhD_combined()` | Their PFER uses cosine similarity, not feature mismatch count |
| `data/processed/timit_test_ipa.json` | Test data | For verification — spot-check that phones tokenize correctly |
| `test/IPA_annotation_sheet_*.xlsx` | Annotator transcriptions | Needed for IAA validation (A6, not this task) |
