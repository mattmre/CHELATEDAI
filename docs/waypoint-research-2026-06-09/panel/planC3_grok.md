# Grok adversarial review C3 — try to DISPROVE the D2 kill-screen's verdict

A fresh implementer (Codex) built the lean D2 crossover kill-screen in
`research/drift_recovery/d2/`, `methods/`, `regimes/`, with artifacts in
`research/drift_recovery/out/d2/`. Your job is to **try to disprove** its verdict and expose any
dishonest framing. Read the code + `out/d2/D2_REPORT.md` + `out/d2/d2_decision.md`, recompute what you
can, return **PASS / PASS-WITH-FIXES / FAIL** with file:line.

## What Codex shipped
Regime-C synthetic cluster-collapse (β=0.10) on SciFact + NFCorpus, 8/16 clusters, 5 seeds. 23/23
tests pass. Verdict: **NO_GLOBAL_G3_VERDICT_UNDERPOWERED_INVALID_OR_UNAVAILABLE**. Per cell:

| Cell | G2 half-width | Invalid gap draws | min oracle gap | Harm AUPRC | chel−ridge ΔNDCG | verdict |
|---|---:|---:|---:|---:|---:|---|
| SciFact/8 | 0.0190 (FAIL) | 1.53% | 0.0000 | 0.225 | +0.0012 | NO_G3_VERDICT |
| SciFact/16 | 0.0148 (PASS) | 23.76% (INVALID) | −0.0046 | 0.151 | −0.0017 | NO_G3_VERDICT |
| NFCorpus/8 | 0.0275 (FAIL) | 76.15% (INVALID) | −0.0078 | 0.269 | +0.0209 | NO_G3_VERDICT |
| NFCorpus/16 | 0.0255 (FAIL) | 60.86% (INVALID) | −0.0091 | 0.307 | +0.0224 [0.0001,0.048] | NO_G3_VERDICT |

## Attack these specifically
1. **Is the regime degenerate, and is that disclosed honestly?** Every cell's min oracle gap ≈ 0 or
   negative — i.e. β=0.10 cluster-collapse barely hurts retrieval, so there is almost no recoverable
   gap to compete over (floor ≈ oracle). Is "NO_G3_VERDICT underpowered/invalid" the honest call, or
   should the report state plainly that **β=0.10 was mis-calibrated (drift too mild) and a
   severity-calibration step — like D1 §4.4's discriminating-band — should have set β BEFORE running**?
   Is this a protocol miss being softened into "underpowered"?
2. **Detector AUPRC 0.15–0.31 ≪ 0.80 everywhere.** Per the locked G3 rule, detector AUPRC<0.80 on harm
   labels → terminate the detector-gated mechanism (detection-only-pass → park as router). Does the
   AUPRC failure constitute a real KILL of the detector-gated chelation mechanism that the report
   *under*-states by leading with "no verdict"? OR is AUPRC meaningless here because near-zero harm
   means near-zero positive labels (single-class / degenerate AUPRC)? Check which, from the harm-label
   code + the actual label prevalence per cell.
3. **Any hidden chelation "win"?** NFCorpus chel−ridge is +0.021/+0.022 and one CI barely excludes 0
   ([0.0001, 0.0484]). Confirm the report correctly does NOT claim a win: it fails the dual-CI rule
   (must beat BOTH CBIE and hubness — chel−cbie is NEGATIVE on NFCorpus; min-gap≥0.05 fails; Holm
   non-significant). Verify chelation is genuinely not winning, not being suppressed.
4. **Are the locked rules actually implemented?** G2-hard-gates-G3, dual-CI (≥5 pts AND ≥0.02 NDCG AND
   CI excludes 0 AND same sign 5 seeds AND min gap ≥0.05), Holm family frozen to {chel-vs-cbie,
   chel-vs-hubness}, α=0.05 arm excluded from the decision family, harm labels = NDCG loss not
   injection membership, detector rejects train/eval overlap, no eval-qrel leakage. Spot-check each
   against code, not just test names.
5. **The excluded pilot.** The report says protocol v2 was locked only after an excluded SciFact pilot
   exposed bugs. Is excluding it honest (it had real implementation bugs), or is it p-hacking (ran,
   saw a result, relabeled the protocol)? Check what changed between pilot and v2.

## Deliverable
`PASS / PASS-WITH-FIXES / FAIL` + a blocker table (severity, file:line, fix) + a one-sentence bottom
line: is D2 an honest "inconclusive because the regime was mis-calibrated," a real "detector-gated
chelation killed by AUPRC," or something being spun? Recompute the oracle gaps and AUPRC label
prevalence yourself if the arrays are in the artifacts. Default to skepticism.
