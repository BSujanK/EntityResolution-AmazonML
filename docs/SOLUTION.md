> This is the methodology document submitted with the final package (2026-09-27), kept unchanged except for this note.
> In the submission zip the code lived in `code/business_entity_resolution/`; in this repository the same files are at the root
> (`src/`, `requirements.txt`). For a structured overview see the [README](../README.md),
> [ARCHITECTURE.md](ARCHITECTURE.md) and [RESULTS.md](RESULTS.md).

# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Vyrox
**Team Members:** B Sujan Kumar, Apeksha Sanjay Hudali (BMS College of Engineering, Bengaluru)
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A cascade built entirely on a 10-core / 12-thread laptop CPU: (1) script-aware normalisation (Indic → Latin transliteration,
legal-form and address canonicalisation, phonetic skeletons, OCR repair); (2) hashed multi-key blocking over
seven key families (≈98% pair recall at top-10), then **learned pruning** to the best S1 per record, leaving
**3.8 candidates per S1 entity**; (3) a 2-fold cross-fitted LightGBM pair classifier over 72 language-agnostic
features; (4) **fine-tuned transformer cross-encoders** (MiniLM, Apache-2.0) that re-read the raw text of every
uncertain pair, stacked with the pair features; (5) an entity-group **twin veto**, soft in countries with training
data and full for the unseen country; (6) per-country label-shift calibration and a **per-entity expected-F0.5
decision rule**. Best public leaderboard score: **0.980385** (v10c, final submission). Held-out validation (441,785 Source-1
entities never used in training): **0.98637**; the blocking ceiling is 0.9925.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the 2.2M / 5.0M / 5.3M training records (S1 / S2 / S3):

* **Structure.** Every S2/S3 id occurs in at most one ground-truth group (7,638,365 links,
  7,638,365 distinct targets) → each S2/S3 record belongs to **at most one** S1 entity. ~26% of
  S2/S3 records match nothing (distractors). S1 entities have 0–11 matches (mean ≈3.5);
  5.6% are singletons.
* **Country is always equal** inside a true match → used only to scope blocking keys (as an
  opaque string, so the unseen `France` label works unchanged). No country one-hot anywhere.
* **Name noise:** word re-ordering (`EE Co Legal`), injected/replaced generic words
  (`Center`, `Partners`, `Downtown`, `Services`), random accents (`QÚALITY`), character typos,
  legal-suffix swaps, domain forms (`eelegal.com`, `#nórthernosisko`), and **native scripts**
  (Devanagari, Bengali, Gujarati, Odia, Tamil, Kannada, Malayalam: 5–11% of S2/S3 names),
  including transliterated legal forms (`प्राइवेट लिमिटेड` → `praivet limited`, `प्रा. लि.`, `एलएलपी`).
* **Address noise:** `#`/`00` prefixes, abbreviations (`Rd`, `R.`, `Av.`, `Imp`), state
  names ↔ codes ↔ native script (`महाराष्ट्र`, `MH`, `Maharashtra`), missing house numbers,
  number typos (`330`→`33`), component re-ordering, landmark phrases, `NULL` tokens.
* **Hard negatives.** Many distractors are near-copies of a real entity — identical name,
  perturbed house number (`402 Main St` vs `40 Main St`) — while genuine matches also carry
  number typos. This is the main precision challenge.
* **Test shift:** test adds France (15% of S1) and is more India-heavy (47% vs 40%).

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise classifier + group-level re-scoring + metric-aware decision (hybrid)
**Core Innovation:** (a) multi-key blocking choosing each record's *rarest* tokens so keys are
stable under re-ordering and transliteration; (b) exploiting the one-to-one structure
(each S2/S3 record → best S1 only) plus *competition* features between candidates;
(c) group-consensus re-scoring that separates near-copy distractors from typo'd true matches;
(d) per-entity expected-F0.5 set selection instead of a global threshold.

---

## 3. Candidate Generation (Blocking)

Every record emits hashed (u64) keys, all prefixed by the country label:

| Key | Content | Catches |
|---|---|---|
| A | pair of name tokens (phonetic skeletons) | re-ordering, extra words |
| B | name token × address number | noisy names with the right house number |
| C | address number × address token | transliterated / DBA names |
| D | first 8 chars of concatenated core name | domains, glued names |
| E | first 7 chars of the phonetic name skeleton | typos, transliteration |
| F | name token × address token | missing house numbers |
| G | pair of address tokens | address-only matches without numbers |

* Only each record's 4 rarest name tokens / 5 rarest address tokens (document frequency among
  S1 records of that country) are used — consistent across re-ordered records.
* Keys shared by >200 S1 records are dropped. Pairs sharing keys are scored by
  Σ 1/√(block size); each S2/S3 record keeps its **top-10** S1 candidates. Per-key-type hit
  counts become model features.
* **Key-blocking output:** train 98.0M pairs (≈9.5 per S2/S3 record), test 94.7M (≈55 per S1 entity).
* **Learned pruning (final candidate set).** The stage-1 LightGBM scores the key-blocking pairs and keeps,
  for each S2/S3 record, only its best S1 candidate, and only if p ≥ 0.02 (each record belongs to at most one
  S1 entity). These pairs are exactly what the final matcher runs inference on, so they form
  `candidate_pairs.tsv`: **6,595,983 test pairs = 3.81 candidates per S1 entity**, a 14× reduction over
  key blocking and 99.99996% over the cross product. On training data this pruned set still contains 96.97% of
  all true links (final pipeline recall is 95.9%, so pruning costs almost nothing).
* **Targeted candidate expansion (v10b/v10c).** Only for *weak* records (best stage-1 p < 0.5), key blocking is
  re-run deeper (top-30; top-50 for India) and the expanded group is re-scored; the record still keeps a single best
  S1. This recovers blocking misses without widening the candidate set: it stays at one S1 per S2/S3 record.
* **Why the set cannot shrink much further.** Training data has **3.46 true links per S1 entity**; our candidate set has
  3.81 per S1 (about 10% above that floor), and 87.5% of candidate pairs end up as final matches. Tighter pruning
  directly deletes matches: on test, p ≥ 0.1 would give 3.59 candidates/S1 but drop 3,006 accepted matches;
  p ≥ 0.5 gives 3.40/S1 but drops 30,074. We keep p ≥ 0.02 as the best balance of candidate size and F0.5.
* **Scalability.** No step compares all pairs: keys are hashed per country and capped at 200 S1 records per key,
  so blocking cost is linear in the number of records; all later stages are chunked by record ID (peak RAM ~9 GB
  for ~12M test records on a laptop).
* **Recall safeguards:** measured on 500k training queries: key-hit recall 98.7%, recall@10 ≈97.9%,
  recall@1 95.1%. On the validation fold a perfect matcher over our candidates would reach
  macro F0.5 **0.9926** (the blocking ceiling). Iterations: the first design (name/number keys only)
  had 94.4% recall; adding address-token keys (F, G) and rarest-token selection raised it to 98.7%.

---

## 4. Matching Model

**Normalisation (`normalize.py`).** anyascii transliteration (non-ASCII strings only),
lower-casing, legal-form canonicalisation (pvt/praivet/praibhet/pra → private, ltd/limitet/li
→ limited, elelpi → llp, corporation → corp, company/cie → co, …), stop-word removal
(the/and/of/de/du/la/…), domain stripping, address canonicalisation to short forms
(street/saint/st → st, road → rd, rue → r, avenue → av, …), state names and native-script state
names → codes, ordinal stripping, digit/letter splitting, number extraction with leading-zero
removal, and a consonant *skeleton* (aspirates merged, vowels dropped, repeats collapsed) so that
`bharat green` and `भारत ग्रीन` → `bharat grin` coincide.

**Features used (stage 1: 72; stage 2: 27):**
- **Name:** rapidfuzz ratio / token-sort / token-set / partial on core names; Jaro-Winkler and
  partial ratio on concatenated names (domains); ratio / partial on phonetic skeletons; word-level
  TF-IDF cosine (IDF learned on the split); token Jaccard, common-token count, token counts;
  exact / prefix equality; IDF mass of each name (how distinctive it is); legal-form equality / conflict.
- **Address:** TF-IDF cosine, rapidfuzz ratio / token-set / partial, token Jaccard, token counts,
  IDF mass, empty-address flag.
- **Numbers:** first-number equality, fuzzy ratio, log absolute difference, same digit length,
  truncation/prefix relation, cross-containment, number-set Jaccard, counts and coverage in both directions.
  **Digit-position edit type:** suffix relation, Hamming distance, and the position of the first
  differing digit counted from the left and from the right, plus a "last digit ±≤2" flag. EDA showed that noise hits leading
  digits (P(true match) 0.98 when only the first digit is dropped or changed), while look-alike twin businesses differ in
  trailing digits (P(true) 0.33–0.46).
- **Blocking evidence:** block score, block rank, hits per key family (A–G).
- **Competition context (per S2/S3 record):** for name cosine, address cosine, token-set ratios,
  block score and number Jaccard: gap to the best candidate and margin over the second best;
  number of candidates; how many records have this S1 as their top candidate.
- **Other:** non-Latin-script flag, source (S2 vs S3). Deliberately **no country feature**.

**Model type:** LightGBM binary classifier (127 leaves, lr 0.05, early stopping), **2-fold
cross-fitted**: S2/S3 records are split into two folds by hash; model A is trained on 1.2M
sampled records of fold 0, model B on 1.2M records of fold 1 (≈11M candidate pairs each).
Every training pair is then scored by the model that did *not* see its record, giving honest
out-of-fold probabilities for all 98M training pairs, which stage 2 and the evaluation use.
Test pairs are scored with the average of A and B (a 2-model ensemble).

**OCR repair (feature time).** Name tokens that mix letters and digits have digits mapped back
to look-alike letters (0→o, 1→l, 3→e, 4→a, 5→s, 7→t, 8→b): `m0unt 8aptist` → `mount baptist`,
`nunezr0yalsierra` → `nunezroyalsierra`.

**Assignment.** Each S2/S3 record is attached only to its highest-probability S1 candidate
(one-to-one structure).

**Stage 2 — group consensus and twin detection (`stage2.py`): final matcher on the pruned candidates.** Each S1 entity collects a group
of attached records. Error analysis showed that most wrong merges are **twin entities**: businesses
with the same name at a nearby address (`903 Fox Run Rd` vs `90 Fox Run Rd`, `61 Wa Rd` vs `63 Wa Rd`)
that are not in Source 1 but have their own records in S2/S3. A true match with a number typo is a
one-off, whereas a twin's records *agree with each other*. For every attachment we therefore compute,
over the whole group (any probability): how many other members share this record's house number
while it disagrees with the S1 number (`g_twin_num`), how many members (and how much probability mass)
support the S1's own number, exact name / address agreement counts, and the best probability among
records sharing my number. We also compute consensus among confident members, the group size, sum / max of
the other members' probabilities, and the rank and relative probability within the group. A LightGBM (63 leaves, early
stopping) re-scores each attachment. **The final score is min(p_stage1, p_stage2)**: the group model
can veto an attachment (twins, look-alike clusters) but never promote one that the pair model finds unlikely.
Leaderboard feedback showed why this matters. Letting stage 2 promote low-p records (v3) improved validation
(0.9812 → 0.9833) but lowered the public score (0.972 → 0.965), because test has more look-alike clusters than
train (40% vs 26% distractor records; mid-confidence records are 1.6× more frequent). It is trained on the **out-of-fold** stage-1 scores of all
training groups (≈2.2M S1 entities), and evaluated on a held-out 20% of S1 entities.

**Label-shift correction (test-time, label-free).** The pair model is well calibrated on training data
(mean p of best attachments 0.950 vs true rate 0.951 for US; 0.929 vs 0.931 for India). Test, however, has
more look-alike records near the decision boundary. For each country we re-estimate the positive rate among
records' best attachments with the Saerens–Latinne–Decaestecker EM algorithm, using only the model's
test-set posteriors, and rescale every score's odds by (π_test/(1−π_test)) / (π_train/(1−π_train)).
Unseen countries use the pooled training rate. Estimated test rates: US 0.914, India 0.873, France 0.891
(training: 0.951, 0.931, pooled 0.943), giving odds factors ×0.55, ×0.51 and ×0.50.

**Threshold selection method:** per-entity **expected-F0.5 maximisation**. For each S1 we sort
its attached records by probability and keep the prefix size *m* maximising a plug-in estimate
E[F0.5](m) = 1.25·S_m / (1.25·S_m + 0.25·(T − S_m) + (m − S_m)), where S_m is the sum of the
top-m probabilities and T the sum of all; predicting nothing scores Π(1 − p_i) (probability that
the entity is a singleton). This needs no tuned threshold, which matters for the unseen country,
and slightly beats the best global threshold on validation.

---

## 5. Results & Error Analysis

Validation protocol: S1 entities with hash(id) % 40 == 0 (54,940 entities, singletons included)
are held out. *Every* S2/S3 record that has one of them as a candidate or as its true match
(2.19M records) is scored with its full candidate list, so false merges into held-out
entities are counted exactly as by the official metric. Training uses only records that never
touch a held-out entity.

| Configuration | Validation macro F0.5 |
|---|---|
| Blocking ceiling (perfect matcher on candidates) | 0.9926 |
| Stage 1, best global threshold (0.64) | 0.9764 |
| Stage 1, expected-F0.5 decision | 0.9768 |
| + numeric house-number features, 1.2M training queries (stage 1, global threshold 0.66) | 0.9791 |
| + expected-F0.5 decision (stage 1) | 0.9792 |
| + stage 2 group consensus (confident members only; 98k training rows) | 0.9812 |
| v3: OCR repair + 2-fold cross-fitted stage 1, out-of-fold over all 2.2M S1 | 0.9795 |
| **v3: + stage 2 with twin-entity features trained on all out-of-fold groups (441,785 held-out S1)** | **0.9833** |

| v4: stage-2 used as a veto only, min(p1, p2) (held-out 441,785 S1) | 0.9813 |

### Version history with leaderboard scores

| Version | Change | Held-out F0.5 | Public LB |
|---|---|---|---|
| v2 | numeric house-number features, stage-2 consensus, expected-F rule | 0.9812 | 0.972 |
| v3 | cross-fitting, twin features; stage 2 allowed to **promote** | 0.9833 | 0.965 |
| v4a | stage 2 as a **veto only**; learned pruning (3.8 candidates/S1) | 0.9813 | 0.973 |
| v5a | per-country EM label-shift calibration | – | 0.973 |
| v6 | digit-position house-number edit features | 0.9815 | 0.973612 |
| v7b | + cross-encoder (90k pairs/fold) + 75-input stacker | 0.9840 | 0.977 |
| v9 | + v8 cross-encoder (220k pairs/fold), ensemble; **soft veto** in US/India | 0.9853 | 0.979189 |
| probe | soft veto also for France | – | 0.978 |
| v10a | v9 + stricter France calibration (odds ×0.5) | 0.9853 | 0.979396 |
| v10b | + candidate expansion: top-30 key blocking for weak records (best p1 < 0.5), expanded groups re-scored, cross-encoders on changed pairs | 0.9861 | 0.980128 |
| probe | v10b with India / US calibration odds ×0.7 | – | 0.980022 / 0.980105 |
| **v10c (final)** | + second expansion for weak India records: key-blocking ranks 31–50 | **0.9864** | **0.980385** |

Why each step: v3's drop showed that group features which *promote* records do not transfer to test (test holds
~40% decoys vs 26% in train, and more look-alike twins), hence the veto-only rule. The cross-encoders (v7+) cut
errors on the uncertain pairs by learning typo-vs-twin noise from raw text (uncertain-pair AUC 0.913 → 0.981), and
the gain grew on test (validation +0.0025 → leaderboard +0.0034). The France probe showed that France's borderline
pairs are mostly wrong, so the unseen country gets the full veto and stricter calibration.

- **F_0.5 Score (macro):** **0.98637** (validation, final pipeline, 441,785 held-out S1): US 0.98860, India 0.98302. Public leaderboard: **0.980385**.
- **Loss decomposition** (v10b pipeline, 1 − F = 0.0147): blocking misses 0.0076 (true links never retrieved); empty-address records 0.0065 in total (97.7% of them are true matches, but with only a name the S1 is often ambiguous: 49% of S1 entities share their exact core name with another S1 in the same country); rejected look-alikes 0.0030; false merges 0.0019.
- **Common false positives (wrong merges):** ~85% are distractor records that are near-copies of a
  real entity — same name, house number off by a digit or truncated (`402` vs `40`, `1811` vs
  `1812`) — or generic names (`Board Aging`, `Tri State Group Center`) with an empty address.
  Stage 2 targets exactly these by checking agreement with the entity's other matched records.
- **Common false negatives (missed matches):** (i) ~43% never reach the candidate set: typo'd or
  gibberish/DBA names *and* an empty or heavily truncated address; (ii) true matches whose house
  number was also perturbed (`52` vs `54`), which are indistinguishable from distractors by the
  record pair alone; (iii) transliterated Indic names whose skeleton diverges
  (`indiyan entarpraijes`) combined with a partial address.

---

## 6. Conclusion

High-recall, structure-aware blocking, a gradient-boosted matcher and fine-tuned cross-encoders on the uncertain
pairs reach 0.9864 held-out and 0.9804 on the public leaderboard with 3.8 candidates per Source-1 entity, on a
laptop CPU. The largest precision gains
came from modelling *competition* (each record belongs to one entity) and *consensus* (a real
entity's records agree with each other). Optimising the macro F0.5 directly per entity was better
than a single threshold. Lesson learned: under a precision-heavy metric, the right unit of decision
is the entity group, not the isolated pair.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — see its `README.md` for exact commands.

| File | Role |
|---|---|
| `src/run_pipeline.py` | **part 1 entry point**: `python src/run_pipeline.py --data-dir <dataset>` runs prep → block → train → predict |
| `src/config.py` | paths and constants (block cap 200, top-10, validation fold, sampling) |
| `src/normalize.py` | transliteration and canonicalisation |
| `src/prep.py` | normalised-record and ground-truth caches |
| `src/blocking.py` | multi-key hashed blocking, candidate scoring and top-K |
| `src/features.py` | pair and competition features (rapidfuzz, TF-IDF cosine, numbers) |
| `src/model.py` | LightGBM, argmax assignment, macro F0.5, expected-F decision, EM calibration |
| `src/stage2.py` | entity-group consensus / twin veto |
| `src/ce_*.py` | cross-encoder export, training, scoring, stacker3 and the v10a prediction |
| `src/ext_*.py`, `src/ext_phase2.sh` | candidate expansion for weak records (v10b) and India ranks 31–50 (v10c, final) |
| `src/run_ce_stage.sh` | **part 2 entry point**: cross-encoder stage and both expansions → final outputs |

External resources: no external data, lookups or APIs. Models: LightGBM (MIT) and the open pretrained cross-encoder
`cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0, 22.7M parameters, downloaded from Hugging Face), fine-tuned only on
the provided training pairs; both far below the 8B-parameter limit. `anyascii` (ISC) is a deterministic character-table
transliterator.

Hardware: Intel Core i7-1355U laptop CPU (10 cores, 12 threads), 16 GB RAM, no GPU.

### B. Additional Results

**Test-set prediction profile** (final v10c, the 1.73M test S1 entities): 5,772,673 matched links, 6,595,983 candidates; the official validator
(`--check-ids`) reports PASS. Predicted statistics per country compared with the training ground truth
(3.46 matches per S1 entity, 5.6% singletons):

| Country | S1 entities | Predicted matches / S1 | Predicted singleton rate |
|---|---|---|---|
| France (unseen in training) | 259,452 | 3.31 | 5.6% |
| US | 663,106 | 3.37 | 5.8% |
| India | 809,986 | 3.31 | 5.9% |

The unseen country behaves like the training countries, so the model shows no sign of over- or under-merging on France.

**Top stage-1 features by gain:** block-score margin over the second-best candidate, block rank,
log house-number difference, block-score gap to the best candidate, number-set Jaccard, address
token-set ratio (and its gap to the best candidate), name skeleton ratio, legal-form equality,
name IDF mass, name ratio, name TF-IDF cosine. The ranking confirms that *relative* evidence
(how a candidate compares with its competitors) matters more than absolute similarity.

**Validation threshold curve (stage 1, global threshold):** flat between 0.56 and 0.74 (±0.0005),
so the decision is not sensitive to the exact operating point. Expected-F selection needs no threshold at all.

**Runtime (i7-1355U, 10 cores / 12 threads, 16 GB RAM):** normalisation ~3.5 min, blocking ~38 min per split,
training stage (features, 2 cross-fitted models, out-of-fold scoring of 98M pairs, stage 2) ~97 min, test scoring with the 2-model ensemble ~107 min.
