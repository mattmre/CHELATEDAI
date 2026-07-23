# Codex build task — D3: the recoverability ESTIMATOR (P1, locked plan)

Build the calibrated recoverability **estimator** in a new `research/drift_recovery/estimator/`
package, on top of the frozen D1 packs and `harness_bridge.py`. This is P1 in the locked
agreed-build-plan-v1.md. It is an *estimator*, NOT a "certificate"/"theorem": Gram-distortion alone
≠ NDCG, and top-k retrieval breaks the clean free-alignment theorems, so the deliverable is a
calibrated predictor with an honest validation verdict — including an explicit **negative/scoping
note** if the data cannot support cross-regime validation.

Work ONLY inside `research/drift_recovery/`. Do NOT touch `docs/.../paper-draft/`. Reuse
`harness_bridge.py` as the ONLY harness importer; reuse `EmbeddingPack` (artifacts.py) and the
`stats/` primitives. Keep harness parity ≤1e-12. GPU is available (torch cuda); run offline
(HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1), sequential, modest — do not OOM the 3090 (shared machine).

## What the estimator predicts
Given a drift regime represented by a frozen pack (cached doc vectors `Do`, oracle-target doc
positions `Dor`, drifted eval queries `Qd`, qrels), predict the **oracle-gap recovery R̂** that a
*trivial regularized linear corrector* (ridge) will achieve — WITHOUT running the corrector's full
eval — and emit a calibrated lower band. The ground-truth R for each regime is the D1-measured ridge
recovery on that pack.

## Core signal (margin-bound features)
For each eval query q with relevant doc y_r and top competing non-relevant doc y_j, retrieval order
is preserved under a corrector g iff the **margin**
  m(q) = qᵀ g(y_r) − qᵀ g(y_j)  >  0,
and the identity/no-correction error decomposes via e_d = g(y_d) − y_d* (y_d* = oracle position, i.e.
`Dor`). The plan's bound: order is preserved when
  qᵀy_r* − qᵀy_j*  >  ‖q‖(‖e_r‖ + ‖e_j‖).
Build features per query from the pack:
1. **Gram/margin distribution** — per-query clean margin qᵀy_r* − max_j qᵀy_j* (oracle space), its
   sign rate, and low-quantiles across the eval set.
2. **Correction-error magnitude** — ‖e_d‖ distribution for the ridge map fit on the pack's `fit_idx`
   (leakage contract: NEVER use eval qrels in the fit; use the pack's leakage-safe fit index for the
   estimator's own fit-side features).
3. **Predicted inversion rate** — fraction of eval queries where the bound is violated
   (‖q‖(‖e_r‖+‖e_j‖) ≥ clean margin) → maps to a predicted recovery band.

## Package layout (`research/drift_recovery/estimator/`)
- `features.py` — per-query & per-regime feature extraction from an `EmbeddingPack` (margins, error
  norms, inversion-rate estimate). Pure numpy; no harness import (operates on frozen packs).
- `margin_bound.py` — the order-preservation bound + per-query inversion predictor; unit-tested on a
  tiny synthetic pack with a hand-checkable margin.
- `calibration.py` — cross-fitted map from features → R̂ + a lower band (isotonic or a small linear
  calibrator; cross-fit so no regime's own R trains its own prediction).
- `predictor.py` — `estimate_recovery(pack) -> {R_hat, lower_band, features}`.
- `validation.py` — **leave-one-regime-out** validation across the available packs. Held-out axis =
  regime (encoder family / dataset / n_anchor), NEVER same-run (that is circular). Reports
  Spearman(R̂, R) and 3-way bin accuracy (low/med/high recovery).

## Regimes (data)
The estimator needs ≥3–4 regimes for leave-one-regime-out. `scifact_evalsplit_pack` already exists
(SciFact, MiniLM→mpnet). Freeze the additional regimes you can via `harness_bridge` **if the cached
models are available offline**: NFCorpus/MiniLM→mpnet, SciFact/MiniLM→bge-large, NFCorpus/MiniLM→bge-
large (and optionally a different `anchor_fraction` on SciFact as a 5th regime). Each new pack MUST
reproduce its ridge recovery through the SAME NDCG implementation (assert_harness_parity ≤1e-12) and
carry a leakage-safe fit index. **If a model/dataset is not cached offline and cannot be fetched, do
NOT fabricate the regime** — log it and proceed with whatever regimes exist.

## Success bar (locked)
- **Ship-positive** iff leave-one-regime-out Spearman(R̂, R) ≥ 0.7 OR 3-way bin accuracy ≥ 80% across
  ≥3 held-out regimes.
- **Otherwise ship an explicit negative/scoping note**: report the achieved Spearman/accuracy, how
  many regimes were available, and precisely why the data is insufficient (e.g. too few regimes to
  validate cross-regime; the per-query bound predicts inversion but the feature→R map is not
  calibratable at n regimes). A well-scoped negative is an acceptable, honest deliverable here — do
  NOT inflate a 2-regime fit into a validated estimator.

## Deliverables
1. `research/drift_recovery/estimator/` package (files above), unit-tested.
2. `research/drift_recovery/out/estimator/` artifacts: per-regime feature JSON, the frozen packs you
   built (or a manifest of which regimes were available vs skipped-and-why), and
   `estimator_validation.md` with the Spearman/bin-accuracy table + the positive-or-negative verdict.
3. `tests/test_estimator.py`: margin-bound correctness on a synthetic pack, feature determinism,
   leakage guard (estimator fit never sees eval qrels), and a leave-one-regime-out smoke test.
4. Run the test suite; paste pass/fail. Regenerate no D1 artifact; D3 is additive.
5. A `research/drift_recovery/out/estimator/D3_REPORT.md` (≤1 page): what was built, regimes used,
   the verdict, and the single most important limitation.

Be brutally honest. If only 1–2 regimes are cheaply available, say so and deliver the scoping-negative
rather than a circular same-run "validation." Report the smallest concrete thing that does not work.
