# Architecture — 2026-03-06 Session 27 Merge Follow-Through

## Objective

Complete the Session 26 follow-up cycle safely by revalidating the open PR stack, landing the implementation branches, preserving audit continuity, and carrying the hardware blocker forward without overstating the system state.

## Execution Plan

### Phase 1: Fresh Review Passes

- Review each open PR independently to avoid context leakage between concerns.
- Treat the code-bearing PRs (`#90`, `#91`) as higher regression risk than the documentation branches.

### Phase 2: Targeted Fixes

- If any review finding appears, patch only the owning branch.
- Re-run the smallest sufficient validation set before merge.

### Phase 3: Merge Order

1. `#90` — hardware evidence capture tooling
2. `#91` — emulation CI coverage
3. `#92` — transport scope lock
4. `#93` — retention policy
5. `#94` — wrap/handoff refresh after the prior four are already on `main`

This order keeps the wrap branch from describing a pre-merge state after the implementation PRs have already landed.

### Phase 4: Post-Merge Validation

- Run `python -m ruff check .` on `main`.
- Run `python -m unittest discover -s . -p "test_*.py" -v` on `main`.
- Record the results in the cycle verification log.

### Phase 5: Hardware Gate

- Probe the local machine for an RP2040 / Pico-class device before attempting physical evidence capture.
- If no device is present, explicitly record the blocker and do not attempt software-only substitution.

### Phase 6: Wrap Refresh

- Add a new session log for the follow-through session.
- Update `next-session.md`, `CLAUDE.md`, the tracker pointer/index, and phase summaries to the post-merge state.

## Risk Controls

- Use admin merge only after checks are green because the repo branch policy can remain blocked even when CI is fully passing.
- Preserve explicit Windows raw-device paths during hardware evidence capture to match the documented CLI contract.
- Keep the RP2040 claim scope locked until the promotion gates in `docs/computational-storage-transport-scope-decision.md` are met.
- Treat unrelated removable USB storage as non-evidence.

## Expected Outputs

- PRs `#90`-`#93` merged on `main`
- Full post-merge validation evidence
- Session 27 research, architecture, tracker, and handoff artifacts
- Remaining backlog reduced to real hardware evidence capture plus the dated retention review
