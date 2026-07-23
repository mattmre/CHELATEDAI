# C3 adversarial-review fix report (D2 kill-screen)

Applied by the chair (Claude) directly — Codex was rate-limited. Numbers machine-verified from the
frozen per-seed / detector artifacts (`out/d2/cells/*/seed_*.json`, `detector_auprc.json`), not
hand-copied from the review. Grok C3 verdict was **PASS-WITH-FIXES**: the procedural `NO_G3_VERDICT`
is correct and could not be disproven; the fixes are all report-honesty / framing.

| Sev | Finding | Fix applied | Evidence |
|---|---|---|---|
| HIGH | Verdict led with "underpowered/invalid"; did not name β=0.10 regime degeneracy | `D2_REPORT.md` rewritten: scientific verdict is now "inconclusive because β=0.10 was mis-calibrated (drift too mild)"; added per-cell oracle-gap column (min ≈0 / negative; NFCorpus point-means negative) and stated G3 was a priori unpassable + the missing severity-calibration preflight | machine: gaps min {+0.000, −0.005, −0.008, −0.009}, NFCorpus point-mean −0.002/−0.001 |
| HIGH | SciFact chelation NDCG-identical to floor; "Primary movement" column implied action | Report now states max\|chel−floor\| NDCG = **0.0 across all 5 seeds on both SciFact cells** (≤5e−3 NFCorpus); dropped the misleading vector-displacement "Primary movement" column and explained it | machine: max\|chel−floor\| = 0.0e+00 (SciFact/8,16), 2.4e−4 / 4.7e−3 (NFCorpus) |
| MEDIUM | AUPRC printed without prevalence / single-class context → reads as detector death | Report adds pos/N per cell (5/160…33/160), per-seed positives, 2 single-class seeds on SciFact/8, median harm near the 1e−8 floor; states AUPRC<0.80 is a symptom of regime mildness, not an independent kill; does NOT invoke the AUPRC→KILL path | machine: AUPRC {0.225,0.151,0.269,0.307}, prevalence 3–21% |
| MEDIUM | β locked without recoverability preflight | Documented as the primary caveat + named the required preregistered β-calibration step (abort if min gap < threshold) as the one experiment still owed. Not implemented (confirmatory re-run is post-gate per the locked plan). | `D2_REPORT.md` headline caveat |
| LOW | "single most important caveat" = pilot exclusion | Demoted pilot to a secondary caveat; promoted regime degeneracy to the headline | `D2_REPORT.md` |
| LOW | `decisions.py` reason string said "every seed"; check is identity-in-any-seed | Reworded to "judged primary arm was identity: moved no documents in any seed" | `d2/decisions.py:111`; 23/23 tests still pass |

## Non-fixes (Grok attacks that FAILED to disprove honesty — left as-is)
No hidden chelation win (dual-CI + CBIE-beats-chelation + Holm + gap all correctly fail); pilot
exclusion justified by real bugs; G2-before-G3, leakage guards, harm-label definition, α-exclusion all
real in code; 23/23 tests pass (chair re-ran).

## Test evidence
`python -m unittest research.drift_recovery.tests.test_d2_decisions research.drift_recovery.tests.test_synthetic_collapse` → **Ran 23 tests … OK** (chair-run, offline).
