# Codex build — Obj A v2: the POWERED margin-predictor validation (expanded blocks)

The operator downloaded the two unblocking pieces: `intfloat/e5-base-v2` (3rd encoder family, 768-d)
and `mteb/arguana` (4th dataset; harness task name `ArguAna`). Both verified loading fully OFFLINE.
Extend the Obj A validation from 6 → 12 cells: datasets {SciFact, NFCorpus, FiQA2018, ArguAna} ×
encoder swaps {all-mpnet-base-v2, BAAI/bge-large-en-v1.5, intfloat/e5-base-v2}. That gives
**4 dataset-blocks and 3 encoder-family-blocks** — the powered test the v1 report said was owed.

Work in `research/drift_recovery/` (estimator/ + run_estimator.py + tests/). Offline flags ON
(HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1 — everything needed is cached). Sequential, modest GPU
(shared 3090). Reuse the 6 existing frozen packs; build only the 6 new ones:
ArguAna×{mpnet, bge-large, e5} + {SciFact, NFCorpus, FiQA2018}×e5.

## Pre-registration v2 (write and freeze BEFORE building any pack; v1 files UNTOUCHED)
`estimator/prereg_objA_v2.md` + `prereg_objA_v2.json`:
- Primary, nulls, metrics, AND-bar: IDENTICAL to v1 (univariate `oracle_margin_mean` OLS block-LOO;
  must beat `oracle_gap→R` and `mean-R` nulls on BOTH Spearman AND MAE under BOTH block schemes;
  partial-Spearman(margin, R | gap) reported).
- Regime set: the 12 cells above, anchor fraction 0.40 everywhere, seed 42 (matching v1 pack recipe).
- Frozen labels: **POSITIVE** if the AND-bar is met with ≥4 dataset-blocks AND ≥3 encoder-blocks;
  **PROMISING-BUT-UNDERPOWERED** if met on fewer blocks (e.g. cells fail to build);
  **NEGATIVE** if the bar fails. No other labels.
- Degenerate-cell rule: a pack whose oracle gap is non-positive is excluded and logged (existing
  `run_estimator.py` gate); it reduces block counts honestly rather than being imputed.

## Build notes
- e5-base-v2 is used through the SAME plain harness encode path as mpnet/bge (no "query:"/"passage:"
  prefixes). Disclose this in the report: it may understate e5's absolute quality but keeps the
  encoder-swap protocol identical across families; the estimator predicts recovery of whatever swap
  actually happened.
- Every new pack: assert_harness_parity ≤1e-12, leakage-safe fit index, frozen per-query scores —
  identical recipe to the existing packs (mirror how the FiQA packs were registered/built).
- ArguAna corpus ~8.7k docs: sample/slice exactly per the existing dataset recipe (1,200-doc slice if
  that is what the other packs use — MATCH the existing recipe, do not invent a new one).

## Run + deliverables
1. `out/estimator/objA_v2_validation.md` + `.json`: per-scheme block-LOO tables (margin vs both nulls,
   Spearman + MAE), partial-Spearman, per-cell R table, block counts, and the frozen-label verdict.
2. `out/estimator/objA_v2_REPORT.md` (≤1 page): what was built, which cells (if any) failed/degenerate,
   the verdict, and — brutally — whether margin still beats gap-only at 4 dataset-blocks, whether the
   ArguAna and e5 holdouts (fully novel dataset and encoder family) individually agree or disagree with
   the aggregate, and the FiQA-style within-block tie check for the new blocks.
3. Tests: extend with `tests/test_objA_v2.py` (or extend test_objA) — block integrity for the 12-cell
   set, v2 prereg frozen before packs (assert file hash/timestamp ordering if practical), nulls computed
   on same regimes, no eval-qrel leakage in new packs. RUN the estimator + objA test suites; paste
   pass/fail.
4. Do NOT modify v1 outputs (objA_validation.*, prereg_objA.*) — v2 is a separate, additive artifact set.

Be brutally honest. Grok's standing prior: the n=4 perfect rank was cheap and the honest expectation is
regression toward gap-level. If margin stops beating gap-only at 4 dataset-blocks, the verdict is
NEGATIVE and that is a publishable, clean answer — do not soften it. If it holds, POSITIVE per the
frozen label and say exactly what scale caveats remain.
