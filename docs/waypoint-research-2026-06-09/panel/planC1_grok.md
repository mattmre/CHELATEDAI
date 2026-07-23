ROLE — ADVERSARIAL VALIDATOR (Grok, multi-agent). Codex just built the D1 deliverable of a drift-recovery paper in this worktree. Your job: adversarially VALIDATE the build and its findings. Spin up internal sub-checks (statistics, code-correctness, leakage, claims). READ THE ACTUAL FILES under research/drift_recovery/ (do not trust my summary):
- research/drift_recovery/out/d1/ci_report.md, bootstrap_cis.json, contrasts_holm.json, learning_curve.json, run_manifest.json, paper_edits.md, section3_draft.md
- research/drift_recovery/stats/paired_bootstrap.py, stats/learning_curve.py, stats/multiple_testing.py
- research/drift_recovery/harness_bridge.py, artifacts.py, contracts.py, d1/paper_stats.py, tests/test_d1_stats.py

Headline results to validate: ridge 84.3% [77.0,91.0]; mlp 81.6% [74.2,88.6]; c3a 19.4% [9.4,30.6]; ridge-mlp Delta=0.022 [-0.033,0.077] Holm p=0.43 (NOT significant); ridge-c3a Delta=0.524 Holm p=0.0004 (significant); G2 power FAIL (half-width 0.089 vs 0.015); a leakage flag (eval-positive docs in the original fit permutation).

Attack, specifically:
1. Is the paired bootstrap implemented CORRECTLY (ratio-of-means recovery; same draws for floor/oracle/methods; percentile CIs; invalid-draw handling; Holm)? Any statistical bug that would make the CIs wrong?
2. Is the harness-parity + leakage handling real, or does it silently reintroduce leakage / diverge from the merged ndcg?
3. Does the 'ridge-mlp not significant -> no linear ceiling' conclusion hold, or is it just the 60-query underpowering (i.e. absence of evidence vs evidence of absence)? What must the paper claim vs not claim now?
4. Is the section3 draft accurate to the real harness definitions, or does it invent formulas?
5. Are the paper_edits complete and correct?
Deliver: PASS / PASS-WITH-FIXES / FAIL, the specific bugs/overclaims with file:line, and the exact required fixes before D1 ships. Be brutal and concrete.
