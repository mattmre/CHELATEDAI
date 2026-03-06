# Research — 2026-03-06 Session 27 PR Review And Merge Follow-Through

## Scope

Re-review the Session 26 follow-up PR stack (`#90`-`#94`) with fresh passes, fix any newly discovered issues before merge, and confirm whether physical RP2040 hardware evidence is actionable in the same cycle.

## Findings

- PR `#90` had a correctness gap during review: `resolve_drive_path()` on Windows rewrote already-formed raw-device paths like `\\.\PhysicalDrive2` into malformed paths. The tool contract explicitly allowed device paths, so the branch needed a narrow fix plus regression coverage before merge.
- PR `#91` remained technically sound after a fresh local pass. The pure-Python emulation job is the correct hosted-CI gate because it avoids privileged FUSE while still proving payload parity.
- PR `#92` correctly narrows the RP2040 claim boundary to deterministic transport proof and aligns the firmware/POC docs around the same promotion gates.
- PR `#93` correctly defers deletion. The live local refs and `.claude/retired-branch-artifacts` inventory still justify a dated manual review rather than immediate cleanup.
- The repository branch policy still reports validated PRs as non-mergeable without escalation. Session 27 confirmed that admin merges were required again even after all required checks passed.
- The local hardware probe on 2026-03-06 did not reveal an RP2040 / Raspberry Pi Pico / TinyUSB mass-storage device. The only removable USB disk visible was a SanDisk drive, so physical evidence remained blocked.
- The open wrap PR (`#94`) could not be merged as-is after `#90`-`#93` landed because its handoff assumptions would be stale immediately. It needed to be refreshed against post-merge reality.

## Sources Reviewed

- `CLAUDE.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/session-log-2026-03-06-session26.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-index.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-pointer.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/verification-log.md`
- Open PRs `#90`, `#91`, `#92`, `#93`, `#94`
- Local branch diffs for `feat/session26-hardware-evidence-capture`, `feat/session26-emulation-ci`, `docs/session26-transport-scope-lock`, `docs/session26-retention-policy`, and `docs/session26-wrap`
- Local device inventory from `Get-PnpDevice`, `Get-Disk`, and `Get-Volume`

## Recommended Actions

1. Fix PR `#90` to preserve explicit Windows device paths and add regression coverage.
2. Merge PRs `#90`, `#91`, `#92`, and `#93` after fresh validation.
3. Run full post-merge validation on `main`.
4. Refresh the wrap/handoff PR against the actual merged state.
5. Keep hardware evidence blocked until a real RP2040/Pico device is attached.
