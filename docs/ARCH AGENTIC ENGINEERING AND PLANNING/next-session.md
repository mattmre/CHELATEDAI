# Next Session Checklist

Purpose: Continue from the post-merge computational-storage follow-through and complete the first real RP2040 evidence capture when actual hardware is available.

## Session Start
- Review `session-log-2026-03-06-session27.md`.
- Sync local `main` to the latest `origin/main` before attempting hardware follow-through.
- Confirm whether an actual RP2040 / Raspberry Pi Pico device is attached before attempting physical evidence capture.
- If a removable USB disk is present, verify that it is the RP2040 target and not unrelated storage. Session 27 only observed a SanDisk removable drive.

## Priority Order
1. If an actual RP2040 / Pico device is attached, run `python computational_storage_poc/capture_hardware_evidence.py <drive-number-or-path> --notes "..."`.
2. Review the generated JSON and Markdown reports under `docs/computational-storage-hardware-evidence/` and confirm they match the deterministic payload contract.
3. Open a follow-up PR with the generated evidence report. If capture fails, open a blocker/remediation PR with the failing report and device notes instead of editing old PRs ad hoc.
4. Keep the RP2040 claim locked to deterministic transport proof until all promotion gates in `docs/computational-storage-transport-scope-decision.md` are satisfied.
5. Revisit the dated cleanup review on or after 2026-04-05 using `docs/computational-storage-retention-policy-2026-03-06.md`.

## Current State
- PRs `#90`, `#91`, `#92`, and `#93` are merged on `main`.
- `main` now includes:
  - reusable hardware evidence capture tooling,
  - dedicated computational-storage emulation CI coverage,
  - a canonical transport scope decision, and
  - the dated retention policy.
- `main` passes:
  - `python -m ruff check .`
  - `python -m unittest discover -s . -p "test_*.py" -v`
  - Result on `main` as of 2026-03-06: `973` tests passing
- No RP2040-class device was attached during Session 27, so no real-hardware evidence report exists yet.
- Archived rollback refs/artifacts still exist locally and remain under the explicit review window that opens on 2026-04-05.

## Handoff Notes
- Do not add `pytest` imports to `test_*.py`; CI does not install `pytest`.
- If imported code must run under Python `3.9` CI, avoid runtime `X | None` annotations unless the module uses deferred annotations.
- `gh pr merge` can still require `--admin` even when checks are green. Keep the root worktree on `main` when merging stacked PRs.
- The firmware, emulator, and host reader must keep sharing the same deterministic payload contract. Do not reintroduce fixed demo strings or mock-only transport output.
- Prefer the pure-Python emulation validation path for hosted CI instead of trying to force Docker/FUSE into Actions.
- Use the hardware capture tool from `main` for the first physical RP2040 evidence run; do not handcraft the report.
- Explicit Windows raw-device paths like `\\.\PhysicalDrive2` are valid inputs for the host reader and evidence capture tool.
- Do not use unrelated removable USB storage as proxy hardware evidence.
- `ruff check` does not validate GitHub Actions YAML. Keep workflow-file review separate from Python lint.
- Local `git status` may show `?? .claude/`; that is expected because local worktree metadata and retired-branch artifacts live there and are not part of the product tree.

## Cycle ID
- AEP-2026-03-06
