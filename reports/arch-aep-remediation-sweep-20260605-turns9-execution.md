# ARCH-AEP Remediation Sweep — 2026-06-05 (Turns 1-9 execution)

## 9-Turn Plan Status

1. Branch / working state confirmed and scoped.
2. Baseline debt posture captured (`check_block_flag`, `next-session` scan).
3. Focused test validation of SHIM evidence and scheduler proof paths.
4. PHASE loop hardening: scheduler slice now fails on stale/invalid evidence.
5. 10-minute BHS loop and smoke verification run to confirm no regressions.
6. Parallelizable checks run while heavy tests executed (artifact checks + smoke tiers).
7. Failure-mode audit of scheduler evidence behavior and test isolation.
8. Remediation report drafted and risk-ranked.
9. Execution-readiness summary prepared with concrete next steps.

## What was done this pass

- Added evidence freshness gating for scheduler completion in `scripts/phase_development_loop.py`.
  - `cd06_scheduler_evidence` is now only considered complete when the latest
    scheduler artifact is `found && verified`.
  - Stale completion markers are pruned in-memory via `prune_invalid_completions()`.
  - `tests/test_phase_development_loop_scheduler_handler.py` now verifies this behavior with a regression test.

- Isolated scheduler evidence output to avoid cross-run contamination.
  - `scripts/record_shim_scheduler_evidence.py` now supports `CHELATED_SHIM_EVIDENCE_DIR`.
  - `tests/test_shim_scheduler_evidence.py` uses a temp artifact directory for each run and validates output without mutating production artifacts.

- Re-validated hardening:
  - `python -m unittest -q tests.test_shim_scheduler_evidence tests.test_phase_development_loop_scheduler_handler`
  - `python -m unittest -q tests.test_check_block_flag`
  - `python scripts/run_10min_priority_bhs_loop.py --minutes 0.1 --bhs-target 100` (turns reached `0`, all checks passed).

## Current objective posture

- `python scripts/phase_development_loop.py --recommend-only` now surfaces:
  - `next_executable: SHIM-SLICE-SCHEDULER-06`
  - `pending_executable_count: 1`
  - `primary_action: execute_handler`
- `verify_shim_development` is still a mixed mode:
  - it writes scheduler evidence, but in this runtime scheduler probe is still not positively proven unless strict verification is enforced by command context.
- Open blocking debt in `docs/next-session.md` remains at 5 rows (`SHIM-CD-01/02/06/08/09`).

## Highest-risk gaps (validated)

1. **SHIM-CD-06 proof gap is now correctly blocking execution**
   - External scheduler evidence cannot be treated as complete if latest artifact lacks positive proof (`found=false` / `verified=false`).
   - This is good for safety, but also means queue cannot advance on stale test- or environment-only artifacts.

2. **Verification environment mismatch remains**
  - Some environments still cannot expose scheduler `019e669bf1bb` through default probes.
  - Need explicit policy: either enforce strict environment prerequisites or provide a deterministic provenance path for scheduler verification in CI.

3. **Operational debt drift risk**
  - Block flag is currently clear but 8 open debt rows remain in one session; this is intentional under current rulebook but increases the chance of non-substrate progress claims if process gates are not enforced.

## 9-Turn completion assessment

- Turn plan completed end-to-end with no unresolved blocking in this pass.
- Remediation shifted from “pass-through despite weak proof” to proof-aware execution gating.

## Recommended architecture/roadmap adjustments (next 9 turns)

1. **Close one blocking debt at a time with strict gates**
   - SHIM-CD-01 first, SHIM-CD-02 second, then SHIM-CD-06.
2. **Treat scheduler proof as production infrastructure debt, not test debt**
   - Require scheduler command, worker profile, and one successful execution trace before marking any cycle ready.
3. **Add debt-aging automation**
   - Gate stale debt transitions by cycle age to avoid manual drift and repeated doc-only progress claims.
4. **Keep 10-minute loops proof-tight**
   - In loop handlers, require `verify_shim_development` strict mode and artifact hash checks.
5. **Protect evidence namespace**
   - Preserve per-run evidence isolation in scripts and CI to prevent environment cross-contamination.
