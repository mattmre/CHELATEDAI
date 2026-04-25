# Next Session Checklist

Purpose: Start from the Session 32 no-promotion outcome instead of relaunching the broad campaign. Current defaults remain unchanged.

## Session Start
- Review `docs/weight-refinement-campaign-results-2026-04-25-session32.md`.
- Review `session-log-2026-04-24-session32.md` for the exact PR and process history.
- Sync local `main` to `origin/main` after the Session 32 wrap PR is merged.
- Confirm there is no stray `run_weight_refinement_campaign.py` or `benchmark_distillation.py` process still alive before starting new research work.

## Priority Order
1. **Do not promote presets from Session 32.**
   - The run ended with an explicit no-promotion outcome.
   - `mlp` + `teacher_weight=0.3` was the only local Phase 2 win, but it was not repeatable across 2 or more independent runs.
2. **If more evaluation is explicitly desired, run a focused repeatability check instead of the full broad campaign.**
   - target only `mlp` + `teacher_weight=0.3`
   - use a clean run directory
   - require a second independent SciFact win before doing anything else
3. **Only if repeatability is confirmed**, run transfer checks for that single candidate:
   - Phase 3 multitask (small + medium)
   - Phase 4 BEIR (small + medium)
   - reject the candidate immediately if transfer or stability regresses
4. **Keep optional research variants behind the repeatability gate.**
   - `bounded=True` adapter wrapping
   - `infonce` or `hybrid` sedimentation loss variants
   - Kalman-gain LR variants
5. If an RP2040 / Pico device is attached, run `python computational_storage_poc/capture_hardware_evidence.py <drive-number-or-path> --notes "..."`.

## Current State
- Session 32 merged PRs `#107`, `#108`, and `#110`
- the Session 32 broad campaign was intentionally stopped once promotion gates were already failed
- retention review is complete
- no default or preset refresh is justified from Session 32

## Handoff Notes
- Do not add `pytest` imports to `test_*.py`; CI does not install `pytest`.
- Python 3.9 CI: avoid runtime `X | None` annotations unless module uses deferred annotations.
- `gh pr merge` may require `--admin` even when checks are green.
- If a benchmark campaign is no longer the active task, stop it instead of leaving it alive in the background.
- `ruff check` does not validate GitHub Actions YAML.
- Local `git status` may show `?? .claude/`; that is expected.

## Cycle ID
- AEP-2026-04-24
