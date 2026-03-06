# Next Session Checklist

Purpose: Resume from the fully landed computational-storage split on `main` without reopening the stale Session 22 branch line.

## Session Start
- Review `session-log-2026-03-06-session25.md`.
- Check open PRs first. Expected state after Session 25: no open PRs.
- Sync local `main` to `2ff7ebb` or newer.
- Confirm whether any new local worktrees or backup refs were created after Session 25 before starting new branch work.

## Priority Order
1. Capture real hardware evidence for the merged RP2040 transport path and document the exact results.
2. Decide whether to add an emulator-path CI check on top of the existing firmware build and Python validation.
3. Decide whether the merged transport path remains a deterministic toy-graph proof or should graduate to a validated on-device workload.
4. Review the `backup/retired-*` refs and `.claude/retired-branch-artifacts/` retention policy once the current `main` state has aged enough to be trusted without them.

## Current State
- PRs `#86`, `#87`, and `#88` were all merged to `main` on 2026-03-06.
- The stale computational-storage PR `#84` is closed and should remain retired.
- The root repository is back on `main`; there are no open PRs.
- Final validation on `main` after the payload merge:
  - `python -m ruff check .`
  - `python -m unittest discover -s . -p "test_*.py" -v`
- Result on `main` as of 2026-03-06: `966` tests passing.
- Payload transport coverage now exists in `test_computational_storage_payload.py`.
- Firmware build automation exists in `.github/workflows/build_firmware.yml`.
- Archived rollback refs/artifacts kept locally:
  - `backup/retired-session22-online-correction-2026-03-06`
  - `backup/retired-computational-storage-poc-2026-03-06`
  - `backup/retired-computational-storage-poc-rescue-2026-03-06`
  - `.claude/retired-branch-artifacts/2026-03-06-session22-online-correction/`

## Handoff Notes
- Do not add `pytest` imports to `test_*.py`; CI does not install `pytest`.
- If imported code must run under Python `3.9` CI, avoid runtime `X | None` annotations unless the module uses deferred annotations.
- `gh pr merge` can fail if some local worktree is holding `main`; keep the root worktree on `main` unless there is a specific reason to split it again.
- The firmware, emulator, and host reader should keep sharing the same deterministic payload contract. Do not reintroduce fixed demo strings or mock-only transport output.
- Local `git status` may show `?? .claude/`; that is expected because local worktree metadata and retired-branch artifacts live there and are not part of the product tree.

## Cycle ID
- AEP-2026-03-06
