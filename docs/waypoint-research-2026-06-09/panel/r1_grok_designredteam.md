# Grok design red-team (do this BEFORE any GPU is spent)

You are adversarially red-teaming two experiment DESIGNS before we build them. Goal: kill bad designs
cheaply. For each, give: fatal flaws, confounds, whether the "success bar" is honest or riggable, and
the single change that most improves validity. Be terse and specific. Read the relevant code under
`research/drift_recovery/` if useful (estimator/, d2/, out/estimator/, out/d2/).

## Design 1 — validate `oracle_margin_mean` as a recoverability predictor at scale
Context: on 4 regimes (SciFact/NFCorpus × mpnet/bge), the preregistered α=10 multi-feature calibrator
FAILED leave-one-regime-out (Spearman −0.40), but a single continuous feature `oracle_margin_mean`
(mean over eval queries of qᵀy_r* − max_j qᵀy_j* in oracle space) achieved perfect LOO rank (Spearman
+1.0, MAE 0.033) — post-hoc and on n=4. Proposed design: freeze ≥8 regimes (more datasets × encoder
families × drift magnitudes × anchor fractions), **preregister** `oracle_margin_mean` (+ q50, sign_rate)
as the feature set, run honest LOO, success = Spearman≥0.7 OR bin-acc≥80% across ≥8 held-out regimes.
Attack: (a) Is `oracle_margin_mean` just a proxy for the oracle gap or floor NDCG (i.e. predicting R
from something trivially correlated with R)? How do we prove it adds signal over oracle-gap-alone?
(b) With correlated regimes (same datasets/encoders reused), is n=8 "independent" or pseudo-replicated —
what's the honest effective n? (c) Is perfect rank on n=4 likely to survive n=8, or is it overfit to a
monotone accident? (d) What is the HONEST success bar and what pre-registration prevents post-hoc
feature-picking round 2?

## Design 2 — β severity-calibration preflight, then re-run the D2 kill-screen
Context: D2's synthetic cluster-collapse at β=0.10 was degenerate — oracle gap ≈0/negative in every
cell, so G3 (min gap ≥0.05) was a-priori unpassable and chelation did nothing. Proposed design: sweep
β (e.g. {0.1,0.2,0.4,0.6,0.8}) with a preregistered rule to pick the β whose mean oracle-NDCG−floor
lands in a discriminating band (~[0.05,0.20]), freeze it, re-run the 4 cells × 5 seeds under the locked
G2→G3 dual-CI gates. Attack: (a) Does calibrating β to GUARANTEE a recoverable gap bias the verdict
(selecting a regime where re-embed/ridge necessarily wins, i.e. rebuilding the C2-oracle hazard)?
(b) Is there any β where synthetic centroid-collapse is BOTH discriminating AND a fair test of
bounded/local chelation (its design premise), or does a big-enough β just become global drift where
ridge trivially wins again — making the test unwinnable-by-construction for chelation regardless of β?
(c) If (b) is true, is this objective worth GPU at all, or should it be declared unfalsifiable and cut?

## Deliverable
For each design: `PROCEED / PROCEED-WITH-CHANGES / CUT`, the fatal-flaw list, and the one highest-value
design change. If Design 2 is unwinnable-by-construction, say so plainly — cutting a doomed GPU run is a
win. One-paragraph bottom line each.
