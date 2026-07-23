# Grok adversarial review C2 — try to DISPROVE the D3 recoverability estimator

A fresh implementer (Codex) built the D3 recoverability **estimator** in
`research/drift_recovery/estimator/` with validation artifacts in `research/drift_recovery/out/
estimator/`. Your job is NOT to verify — it is to **try to disprove** that this estimator predicts
recovery. Read the actual code and artifacts (start with `out/estimator/D3_REPORT.md` and
`out/estimator/estimator_validation.md`), recompute what you can, and return a verdict:
**PASS / PASS-WITH-FIXES / FAIL**, with specific bugs at file:line.

## Context
- The estimator predicts the oracle-gap recovery R̂ a trivial ridge corrector achieves on a drift
  regime (frozen `EmbeddingPack`), WITHOUT running the full eval, plus a calibrated lower band.
- Ground-truth R per regime = the D1-measured ridge recovery. D1 itself was just hardened: ridge
  recovery 84.3% [77.0,91.0] on SciFact/mpnet is a *leaky-waypoint* point estimate (leakage-safe
  ≈79%); the study is underpowered (G2 fail, 60 queries). Bring that skepticism here.
- Core signal: per-query margin m = qᵀy_r* − qᵀy_j* vs ‖q‖(‖e_r‖+‖e_j‖); predicted inversion rate →
  recovery band. Success bar (locked): leave-one-regime-out Spearman(R̂,R) ≥ 0.7 OR 3-way bin
  accuracy ≥ 80% across ≥3 held-out regimes; else an explicit negative/scoping note.

## What Codex actually shipped (an honest NEGATIVE — pressure-test it)
Verdict: **VALIDATION-NEGATIVE**. 4 regimes (scifact/nfcorpus × mpnet/bge, all real & offline-built;
9/9 estimator tests pass, leakage-guarded). Leave-one-regime-out Spearman(R̂,R) = **−0.400**, 3-way
bin accuracy **75%** — both below the 0.7 / 80% bars. Stated root cause: the margin-bound violation
rate **saturates at 100% in ALL 4 regimes**, so the core inversion-rate feature does not discriminate
recoveries spanning 0.65–0.84.

**The three sharpest attacks for a negative like this:**
- **A. Is it a real kill, or just n=4 underpowered?** Spearman on 4 points is nearly meaningless
  (a single swap flips the sign). Is the honest verdict "estimator fails" or "insufficient regimes to
  decide"? Which does the report claim, and is that the defensible one? Under-selling a real signal
  and over-claiming a kill are BOTH findings.
- **B. Wrong-feature vs dead-idea.** The *sufficient* margin bound saturating at 100% violation is
  expected for catastrophic drift — it may just be the wrong feature, not proof the idea is dead. Did
  Codex test whether the *continuous* margin/error features (margin quantiles, ‖e‖ distribution) —
  not the binary violation rate — correlate with R? If a cheap continuous feature would have
  discriminated and Codex only reported the saturated binary one, the negative is premature.
- **C. Does R̂ beat a trivial predictor?** Even at Spearman −0.4, check the dumb baselines (predict
  mean R; predict from floor NDCG or oracle gap alone). If the estimator is no worse than trivial,
  that's the honest framing.

## Attack these specifically
1. **Circularity in validation.** Is leave-one-regime-out ACTUALLY held-out, or does a regime's own R
   leak into its own prediction (shared calibration fit, same-run features, normalization computed on
   the full set)? If cross-fitting is faked, the Spearman is inflated — FAIL.
2. **Trivial-predictor confound.** Does R̂ actually use the margin-bound signal, or is it just tracking
   floor NDCG / oracle gap / a constant? Check: does a dumb baseline (predict mean R, or predict from
   floor NDCG alone) match the estimator's Spearman? If the margin features add nothing over a trivial
   predictor, the estimator is theater even if Spearman looks OK.
3. **Regime count honesty.** How many regimes were actually available? With <3 held-out regimes a
   "validated" positive is unsupportable — the report must ship a scoping-negative. If it claims a
   positive on 1–2 regimes, FAIL. If regimes were fabricated (models not actually cached offline),
   FAIL hard.
4. **Leakage.** Does the estimator's own ridge fit (for the error features) ever touch eval qrels? It
   must use the leakage-safe fit index.
5. **Margin-bound correctness.** Is the order-preservation inequality implemented as stated, and is the
   unit test on the synthetic pack actually hand-checkable (not tautological)?
6. **Positive-or-negative framing.** If Codex shipped a positive, is it real under 1–4? If Codex
   shipped a negative, is it honest, or is there a real signal being under-sold? Either error is a
   finding.

## Deliverable
`PASS / PASS-WITH-FIXES / FAIL` + a table of blockers (severity, file:line, fix), + a one-sentence
bottom line on whether the estimator predicts recovery or is a trivial-predictor / circular artifact.
Be adversarial. Recompute the Spearman and the trivial-baseline comparison yourself if the arrays are
in the artifacts. Default to skepticism: if you cannot confirm the held-out Spearman independently,
say so and do not grant PASS.
