# Grok Tier B — rung 13 disintegration loop (commit 4e2927c1). Try to disprove it.

Official Tier B scorer for commit `4e2927c1` on branch `lattice/rung13-disintegration-20260714`
(cwd = repo root). This adds detector-driven Evidence-DAG edge pruning. Try to DISPROVE that it is a
real, honest, fail-closed disintegration loop worth BHS 100. Read `evidence_dag_disintegration.py`,
the `prune_edges`/ledger additions in `evidence_dag.py`, `test_evidence_dag_disintegration.py`, and the
real `isomer_detector.py` / `convergence_monitor.py` to confirm the signal mapping is honest.

## Attack
1. **Is the score real or faked?** Confirm `fitness` derives from the detectors' ACTUAL emitted
   fields, not a hardcoded/random/constant. Check `IsomerDetector.detect_*` really emits per-query
   `strength` in [0,1] with high=worse (so `fitness=1-strength` is correct-signed), and
   `ConvergenceMonitor.get_summary()` really has `epochs_without_improvement`/`patience`/`converged`.
   If the field names or semantics are wrong, the mapping is theater → FAIL.
2. **Fail-closed correctness.** Prove that a missing/unmatched/immature signal yields fitness 1.0 and
   therefore CANNOT prune. Try to construct an input where absent evidence causes a prune (that would
   break the core safety claim). Confirm an all-healthy DAG prune is a genuine no-op.
3. **Transactional integrity.** In `prune_edges` and `reanneal_edges`, does a scorer that raises leave
   the DAG half-mutated or the ledger stale? Try to break the transaction. Does `validate_evidence_dag`
   still pass after prune AND after reanneal?
4. **Duplicate/edge-identity.** EvidenceDAG allows duplicate edges; does prune/reanneal handle N equal
   edges correctly (the test claims duplicates are all reannealed) or can it drop/double-restore?
5. **Tests tautological?** Do the 12 disintegration tests actually exercise the real mapping, or do they
   hand-feed a fitness scorer that bypasses `detector_signals_from_outputs`? At least one test must
   drive the real detector-output → fitness path end to end. If all tests use `lambda edge: 0.0`, the
   detector wiring is untested → PASS-WITH-FIXES at best.
6. **Private-attr coupling.** It touches `dag._edges` / `dag._pruned_edge_ledger`. Is that a landmine
   (breaks if internals change) or acceptable for an in-module extension?
7. **BHS body honesty.** Any overclaim; is the "engine auto-prune deferred" scoping honest?

Deliver `BHS_TIER_B: <0-100>` + PASS/PASS-WITH-FIXES/FAIL, defect table (severity, file:line, fix), and
a one-line bottom line: is this a real detector-driven fail-closed loop, or does something not bind?
Fresh-agent: you = grok-4.5, implementer = codex/Fable — different. Recompute/run the tests yourself.
