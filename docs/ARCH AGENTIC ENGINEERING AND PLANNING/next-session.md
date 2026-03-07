# Next Session Checklist

Purpose: Review the Session 28 PR stack first, then decide whether to resume bounded evaluation or continue the computational-storage hardware follow-through when a real device is available.

## Session Start
- Review `session-log-2026-03-07-session28.md`.
- Review open PRs:
  - `#96` `feat: harden weight refinement campaign recovery`
  - `#97` `docs: normalize stale roadmap documents`
- Sync local `main` to the latest `origin/main` before resuming any experiment or hardware task.

## Priority Order
1. Review PR `#96`, especially:
   - real-engine benchmark wiring
   - adapter checkpoint isolation
   - `--resume-run-dir` behavior
   - `docs/weight-refinement-campaign-results-2026-03-06-session28.md`
2. Review PR `#97` and confirm the historical-note framing is the right cleanup level for the legacy hardening docs.
3. If PR `#96` lands, decide whether to:
   - resume from the missing `phase4_beir_medium` stage, or
   - rerun a fresh bounded campaign from a clean run directory
4. If an actual RP2040 / Pico device is attached, run `python computational_storage_poc/capture_hardware_evidence.py <drive-number-or-path> --notes "..."`.
5. Keep the RP2040 claim locked to deterministic transport proof until all promotion gates in `docs/computational-storage-transport-scope-decision.md` are satisfied.
6. Revisit the dated cleanup review on or after 2026-04-05 using `docs/computational-storage-retention-policy-2026-03-06.md`.

## Current State
- PRs `#90`, `#91`, `#92`, and `#93` remain merged on `main`.
- PR `#96` is open for benchmark recovery, adapter isolation, and Session 28 result documentation.
- PR `#97` is open for stale-roadmap cleanup in the historical hardening docs.
- Session 28 did not rerun the test suite by request; it only recorded:
  - `python -m py_compile ...` for the benchmark-recovery branch
  - `git diff --check` for both overnight branches
- The partial bounded campaign completed:
  - Phase 1 sweep
  - Phase 2 distillation weight study
  - Phase 3 multitask suites
  - Phase 4 BEIR small
- The resumed `phase4_beir_medium` process was intentionally stopped once the overnight priority shifted to documentation and cleanup.
- No RP2040-class device was attached during Session 27, so no real-hardware evidence report exists yet.

## Handoff Notes
- Do not add `pytest` imports to `test_*.py`; CI does not install `pytest`.
- If imported code must run under Python `3.9` CI, avoid runtime `X | None` annotations unless the module uses deferred annotations.
- `gh pr merge` can still require `--admin` even when checks are green. Keep the root worktree on `main` when merging stacked PRs.
- The benchmark-recovery work found a real risk of cross-configuration contamination through the shared `adapter_weights.pt` checkpoint.
- If a long-running local benchmark stops being the active priority, stop the process explicitly rather than leaving it alive unattended.
- The firmware, emulator, and host reader must keep sharing the same deterministic payload contract. Do not reintroduce fixed demo strings or mock-only transport output.
- Use the hardware capture tool from `main` for the first physical RP2040 evidence run; do not handcraft the report.
- Explicit Windows raw-device paths like `\\.\PhysicalDrive2` are valid inputs for the host reader and evidence capture tool.
- Do not use unrelated removable USB storage as proxy hardware evidence.
- `ruff check` does not validate GitHub Actions YAML. Keep workflow-file review separate from Python lint.
- Local `git status` may show `?? .claude/`; that is expected because local worktree metadata and retired-branch artifacts live there and are not part of the product tree.

## Cycle ID
- AEP-2026-03-06
