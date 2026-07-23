# Codex build — Obj A: scale-validate `oracle_margin_mean` under a PRE-REGISTERED, block-LOO protocol

Build the honest scale-up validation of the D3 recoverability lead, in
`research/drift_recovery/estimator/`, reusing `harness_bridge.py`, `EmbeddingPack`, `features.py`,
`baseline_analysis.py`. A fresh adversarial reviewer (Grok) approved this only PROCEED-WITH-CHANGES;
the changes below are BINDING and are the whole point — do not relax them to manufacture a positive.

Offline only (HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1), sequential, modest GPU (shared 3090 — do not
OOM). First READ `research/drift_recovery/out/estimator/offline_cache_scope.md` (just produced) for the
exact models/datasets available offline.

## The pre-registration (write this FIRST, freeze it, THEN build packs)
Write `research/drift_recovery/estimator/prereg_objA.md` and a machine `prereg_objA.json` BEFORE
freezing any new pack, containing:
- **Primary hypothesis:** univariate `oracle_margin_mean` predicts ridge oracle-gap recovery R better
  than the trivial nulls.
- **Primary metric (single, frozen):** block-LOO Spearman(R̂, R) from a univariate OLS on
  `oracle_margin_mean` ONLY. No multi-feature map, no q50/sign_rate fallback.
- **Nulls it MUST beat (frozen):** (i) `oracle_gap → R` OLS, (ii) `mean-R`. "Beat" = strictly better
  on BOTH block-LOO Spearman AND block-LOO MAE. Also report a locked partial-Spearman of
  (oracle_margin_mean, R) controlling for oracle_gap — the increment over gap is the real question.
- **Block-LOO definition (binding):** the held-out unit is a whole **(dataset)** and separately a whole
  **(encoder-family)** — NOT a hyperparam cell. Report block-n (# independent dataset×encoder-family
  blocks) alongside cell-n. Claim power only at block-n.
- **Success bar (frozen, AND not OR):** margin beats BOTH nulls on BOTH metrics (Spearman AND MAE)
  under BOTH block schemes (dataset-block AND encoder-block). Even if met, the result is labeled
  **PROMISING-BUT-UNDERPOWERED** (not "validated") because the offline cache caps us at 3
  dataset-blocks / 2 encoder-family-blocks. If margin fails to beat gap-only under either scheme →
  **NEGATIVE**.
- **Pseudo-replication rule:** drift-magnitude / anchor-fraction variants of the SAME
  (dataset,encoder) pair are logged as cells but do NOT increment block-n and are NOT used to pad the
  primary block-LOO.

## Offline reality (from the cache scope — do not exceed it)
The cache supports exactly **6 independent (dataset × encoder-family) cells**: the existing 4
(SciFact/NFCorpus × mpnet/bge) plus **FiQA2018→mpnet** and **FiQA2018→bge-large** (anchor 0.40). Build
EXACTLY those two new packs and reuse the 4 existing. Do NOT build anchor-fraction pseudo-replicates.
FiQA (57k corpus, financial domain) is the decisive new-dataset robustness test: if
`oracle_margin_mean`'s n=4 +1.0 collapses or stops beating gap-only once FiQA is held out, that is the
informative finding — report it plainly. `run_estimator.py` rejects non-positive oracle gaps, so if a
FiQA pack has a degenerate gap, log it and proceed with what builds.

## Build
1. Freeze every NEW regime the offline cache supports (from the scope doc), maximizing the number of
   independent **(dataset × encoder-family) blocks** — prefer new datasets and new encoder families
   over more magnitudes/fractions of existing pairs. Reuse the 4 existing packs. Each new pack MUST
   reproduce its ridge recovery through the SAME NDCG implementation (assert_harness_parity ≤1e-12) and
   carry a leakage-safe fit index. Do NOT fabricate a regime whose model/dataset is not cached — skip
   and log it.
2. Compute `oracle_margin_mean` (+ gap, floor, mean-R) per regime; run the frozen block-LOO analysis.
3. If block-n < 6 after using everything offline, do NOT fudge — report the achieved block-n and mark
   the result **UNDERPOWERED-NEGATIVE**; still report the point estimates and the must-beat-gap test.

## Deliverables
1. `prereg_objA.md` + `prereg_objA.json` (frozen before packs).
2. New packs under `out/estimator/packs/`; `objA_validation.md` + `objA_validation.json` with: the
   block-LOO Spearman/MAE for margin vs the two nulls under both block schemes, the partial-Spearman
   increment over gap, block-n, cell-n, and the verdict (POSITIVE only if the frozen AND-bar is met;
   else NEGATIVE/UNDERPOWERED).
3. `tests/test_objA.py`: block-LOO never trains on the held-out block; univariate-only primary; nulls
   computed on the same regimes; leakage guard. Run the suite; paste pass/fail.
4. `objA_REPORT.md` (≤1 page): what was built, block-n achieved, the verdict, and — brutally — whether
   `oracle_margin_mean` actually beats oracle-gap-alone or just re-labels oracle ranking ease.

Be brutally honest. Grok's prior is that +1.0 on n=4 regresses toward the ~0.6 gap-level or noise at
block-n; if it does, report the NEGATIVE plainly. A validated positive REQUIRES beating gap-only under
independent blocks — nothing less counts.
