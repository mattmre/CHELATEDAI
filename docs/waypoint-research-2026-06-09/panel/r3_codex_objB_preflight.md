# Codex build — Obj B CPU preflight falsifier (NO GPU; gates whether Obj B is worth a GPU run)

Grok's adversarial design red-team returned BUILD-WITH-CHANGES **conditional on a cheap CPU preflight
that tries to KILL the regime first**. Build that falsifier in
`research/drift_recovery/regimes/` + `research/drift_recovery/d2b_preflight/`. **No GPU, no embedding
model** — use synthetic Gaussian-cluster vectors (or a tiny cached slice) where dense/closed-form
local linear fits are computable. The goal is to answer ONE question: after a **detector-gated,
CV-λ-tuned, parameter-matched per-cluster local ridge**, is there any residual left for a bounded
nonlinear corrector to capture — or does the local ridge swallow it? If it swallows it, Obj B CLOSES
with no GPU.

## The regime (build per Grok's mandatory changes)
- Synthetic corpus: N docs in d dims as K Gaussian clusters (well-separated + realistic within-cluster
  covariance). A set of queries with known relevant docs so NDCG@10 is computable (each query = a
  perturbed cluster member or centroid; relevance = same-cluster membership or a planted positive).
- **Sparse harm:** corrupt a fraction s∈{0.15,0.25} of clusters; leave the rest clean.
- **Non-affine, NON-RADIAL warp** (de-match method and harm — Grok change #2): within each harmed
  cluster apply a DISTINCT per-cluster non-affine map that is NOT radial. **Test TWO warp families**
  (report the admission gate for each separately, so the verdict does not hinge on one warp choice):
  (F1) a per-cluster random **quadratic form** x ↦ x + ε·(xᵀA_c x)·u_c (small), and
  (F2) a per-cluster **soft-fold / piecewise** map (e.g. a smooth hinge along a per-cluster direction).
  Parameters differ per cluster so no single global linear W inverts all harmed clusters, and each map
  is non-affine on the cluster support. If EITHER family leaves a non-empty residual band, report it;
  if BOTH are swallowed by the gated local ridge, that is a stronger CLOSE.
- **Magnitude knob γ** and **anchor-count knob n** (anchors per harmed cluster) are the two sweep axes.

## Methods (all detector-gated with the SAME τ policy — Grok change #3)
- Global ridge / orthogonal Procrustes (expected to fail the dual residual).
- **Gated per-cluster LOCAL ridge — THE CO-PRIMARY** — CV-λ selected on an anchor dev split (never on
  eval queries), parameter/rank-budget-matched. Also a gated local **low-rank** ridge.
- Clusters inferred on **corrupted** vectors only (NO clean cluster-ID leakage — Grok change #4);
  clean partitions allowed only as a diagnostic upper bound.
- Oracle = clean vectors.
(The bounded chelation adapter itself is NOT required in the preflight — the preflight only tests
whether a residual EXISTS after the best gated local ridge. Chelation enters only if we proceed to GPU.)

## Three recovery ladders (Grok change #5 — report all three)
1. oracle − floor
2. oracle − best **global** map (ridge/Procrustes)
3. oracle − best **detector-gated local ridge** (the admission gate)

## Admission gate (Grok change #1 — the decisive test)
Sweep (γ, n) over a grid. For each cell compute mean(oracle NDCG − gated-local-ridge NDCG) = the
**residual band**. Admit the regime for GPU ONLY if there exists a (γ, n) where simultaneously:
- residual-after-gated-local-ridge ≥ 0.05 NDCG (non-empty band), AND
- the warp is still "small magnitude" (per-doc displacement bounded — report the mean/max), AND
- oracle − floor is in a sane discriminating range (not degenerate), AND
- it survives corrupted-space clustering (no clean-ID leakage).
Enforce hyperparameter parity: λ (and rank) selected by the SAME nested/dev-split protocol you would
use for chelation's α (Grok change #4/Q2).

## Verdict
- **CLOSE** (no GPU): if for ALL small-magnitude non-affine (γ, n) the gated local ridge swallows the
  residual (band < 0.05), output the honest one-liner: *"Obj B fails constructively — locality and
  gating matter; chelation's bounded/unpaired mechanism does not uniquely occupy the residual."*
- **PROCEED-TO-GPU:** if a non-empty residual band exists under small-magnitude non-affine warps,
  report the exact (γ, n) operating point + the residual size, as the preregistered cell for a future
  GPU chelation-vs-gated-local-ridge dual-CI run.

## Deliverables
1. `regimes/sparse_local_nonaffine.py` (the warp + query/relevance construction), 
   `d2b_preflight/preflight.py` (methods, three ladders, sweep, admission logic), unit-tested.
2. `out/d2b_preflight/preflight_report.md` + `preflight_grid.json`: the (γ,n) grid with all three
   ladders per cell, the residual band, and the CLOSE / PROCEED verdict with the operating point or the
   close one-liner.
3. `tests/test_sparse_local_preflight.py`: warp is non-affine + non-radial + sparse; no clean-cluster-ID
   leakage into fits; global ridge fails while a dense oracle-ID local fit can partially recover
   (sanity); λ selection uses only the anchor dev split. Run the suite; paste pass/fail.

Be brutally honest and try to KILL the regime — a CLOSE verdict here is a valuable, GPU-saving result,
not a failure. Do NOT tune the warp class to favor chelation, do NOT give the local ridge a worse λ
protocol than chelation would get, and do NOT leak clean cluster IDs. Report the smallest concrete
reason the residual band is empty (or the exact operating point if it is not).
