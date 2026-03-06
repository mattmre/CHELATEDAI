# Next Session Checklist

Purpose: Resume from the post-merge Session 23 state without re-tracing the stale Session 19-22 handoff.

## Session Start
- Review `session-log-2026-02-27-session23.md`.
- Check open PRs first. Expected remaining PRs after Session 23:
  - `#84` `feat: computational storage POC with firmware and emulation tooling`
  - the session-wrap docs PR from `docs/session23-wrap`, if it has not been merged yet
- Sync local `main` to `24f56fb` or newer.
- Prune stale worktrees/branches before attempting any merge operations.

## Priority Order
1. Review PR `#84` and decide whether to merge it as an experimental POC, split it, or keep it draft because it includes binaries and hardware/emulation assets.
2. Merge the session-wrap docs PR so `CLAUDE.md`, tracker files, and this handoff become canonical.
3. Clean up stale local/remote branches and old worktrees after the remaining PRs land.
4. If the PR backlog is clear, decide whether the computational-storage POC becomes an active roadmap track or stays out-of-band research.

## Current State
- PRs `#80`, `#83`, and `#82` were merged to `main` on 2026-03-05 in that order.
- The shared CI failure was fixed by converting `test_noise_injection.py` to native `unittest`.
- Final post-merge validation on `main` passed:
  - `python -m ruff check .`
  - `python -m unittest discover -s . -p "test_*.py" -v`
- Result on `main`: `957` tests passing.

## Handoff Notes
- Do not add `pytest` imports to `test_*.py`; CI does not install `pytest`.
- `gh pr merge` can fail if some local worktree is holding `main`; remove/prune merged worktrees first.
- Repository policy may still show PRs as blocked even after green checks; Session 23 required admin merges for `#80`, `#83`, and `#82`.
- Remaining non-doc product work that was not part of the Session 22 stack is isolated in PR `#84`.

## Cycle ID
- AEP-2026-02-27
