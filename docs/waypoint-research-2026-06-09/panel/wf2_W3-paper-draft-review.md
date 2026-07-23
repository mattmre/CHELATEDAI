# Adversarially review the paper-updates draft

Read `D:\GITHUB\CHELATEDAI\docs\waypoint-research-2026-06-09\panel\paper-updates-2026-07-10-draft.md` and try to DISPROVE its fitness:
- Verify EVERY number against research/drift_recovery/out/estimator/objA_validation.md and
  research/drift_recovery/out/d2b_preflight/preflight_report.md. Any number not in an artifact = FAIL.
- Hunt L13 overclaims: does any sentence upgrade the estimator toward "validated", call anything a
  "ceiling", claim local-ridge-dominates-chelation, or soften the FiQA-holdout tie?
- Check the home-turf paragraph uses the corrected causal framing (small absolute gaps; local often < floor).
- Is the §8 limitations text honest about 3 dataset-blocks and synthetic-only preflight?
Return PASS / PASS-WITH-FIXES / FAIL with quoted offending sentences and exact replacement text.
