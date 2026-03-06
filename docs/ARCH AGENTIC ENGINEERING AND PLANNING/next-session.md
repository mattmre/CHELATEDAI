# Next Session Checklist

Purpose: Resume from the Session 26 overnight PR stack and finish the real hardware follow-through without reopening the stale Session 22 branch line.

## Session Start
- Review `session-log-2026-03-06-session26.md`.
- Check open PRs first. Expected state after Session 26: PRs `#90`, `#91`, `#92`, and `#93` are open against `main`.
- Sync local `main` to `2ff7ebb` or newer before reviewing or merging.
- Confirm whether any RP2040 / Pico hardware is attached before attempting physical evidence capture.

## Priority Order
1. Review PRs `#90`, `#91`, `#92`, and `#93` and merge the accepted ones.
2. Once the evidence-capture tooling is available locally, connect the RP2040 device and capture real hardware evidence for the merged transport path.
3. If the hardware evidence passes, open a follow-up PR with the generated report rather than editing old PRs ad hoc.
4. Revisit the dated cleanup review on or after 2026-04-05 using the retention-policy doc.

## Current State
- PRs `#86`, `#87`, and `#88` are already merged on `main`.
- Session 26 opened four new follow-up PRs:
  - `#90` -- hardware evidence capture tooling
  - `#91` -- emulator-path CI coverage
  - `#92` -- transport scope lock
  - `#93` -- retention policy
- No RP2040-class device was attached during Session 26, so no real-hardware evidence report exists yet.
- `main` still reflects the post-payload baseline from Session 25:
  - `python -m ruff check .`
  - `python -m unittest discover -s . -p "test_*.py" -v`
  - Result on `main` as of 2026-03-06: `966` tests passing
- Archived rollback refs/artifacts still exist locally and now have an explicit review target of 2026-04-05 in PR `#93`.

## Handoff Notes
- Do not add `pytest` imports to `test_*.py`; CI does not install `pytest`.
- If imported code must run under Python `3.9` CI, avoid runtime `X | None` annotations unless the module uses deferred annotations.
- `gh pr merge` can fail if some local worktree is holding `main`; keep the root worktree on `main` unless there is a specific reason to split it again.
- The firmware, emulator, and host reader should keep sharing the same deterministic payload contract. Do not reintroduce fixed demo strings or mock-only transport output.
- Prefer the pure-Python emulation validation path for hosted CI instead of trying to force Docker/FUSE into Actions.
- Use the hardware capture tool from PR `#90` for the first physical RP2040 evidence run; do not handcraft the report.
- Local `git status` may show `?? .claude/`; that is expected because local worktree metadata and retired-branch artifacts live there and are not part of the product tree.

## Cycle ID
- AEP-2026-03-06
