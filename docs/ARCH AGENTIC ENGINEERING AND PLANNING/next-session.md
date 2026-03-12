# Next Session Checklist

Purpose: Launch the overnight evaluation campaign with Session 29-31 fixes in place, then analyze results to decide on preset promotion.

## Session Start
- Review `session-log-2026-03-12-session31.md` for context on the 4 new features merged.
- Sync local `main` to `origin/main` (1082 tests, all PRs through #103 merged).
- Verify `run_overnight_campaign.py` defaults match Phase 1 sweet spot (LR=0.01, threshold=1).

## Priority Order
1. **Launch overnight campaign** (`python run_overnight_campaign.py`):
   - Phase 1: Standard sweep (SciFact, 81 configs) with sediment_time tracking
   - Phase 2: Distillation — 3 adapter types x 3 teacher weights x 3 modes (all-mpnet-base-v2 teacher)
   - Phase 3: Multitask (small + medium suites)
   - Phase 4: BEIR (small + medium tiers)
   - Phase 5: Online-correction ablation (3 loss types)
   - Phase 6: Large sweep (optional, `--launch-large-sweep`)
2. **Analyze campaign results** against the 4 decision rules from `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md`:
   - Beats or matches frozen baseline on SciFact
   - No material regression on multi-task suites
   - No instability (negative gain drift, collapse, poor stability metrics)
   - Gains repeatable across 2+ independent runs
3. **If promotion candidate found**, refresh presets:
   - `ChelationConfig.SEDIMENTATION_TUNED_PRESETS`
   - `ChelationConfig.TEACHER_WEIGHT_SCHEDULE_PRESETS`
   - Consider `BOUNDED_ADAPTER_PRESETS` and `KALMAN_LR_PRESETS` tuning
4. **Test new features in campaign context** (optional, if time):
   - Run campaign variants with `bounded=True` adapter wrapping
   - Run campaign variants with `infonce` or `hybrid` sedimentation loss
   - Run campaign variants with Kalman-gain LR enabled
5. If an RP2040 / Pico device is attached, run `python computational_storage_poc/capture_hardware_evidence.py <drive-number-or-path> --notes "..."`.
6. Revisit the dated cleanup review on or after 2026-04-05 using `docs/computational-storage-retention-policy-2026-03-06.md`.

## New Features Available (Session 31)
- `create_adapter(..., bounded=True)` — BoundedAdapter wrapper for INT8-safe corrections
- `engine.set_sedimentation_loss("infonce"|"hybrid")` — contrastive/hybrid loss
- `engine.enable_kalman_lr(process_noise=0.1)` — variance-aware adaptive LR
- `DimensionProjection.project_tensor()` — gradient-preserving projection (now auto-used in training)

## Current State
- 1082 tests passing on main, lint clean
- PRs #96-#103 all merged (Session 31)
- 3 research docs in `docs/ARCH AGENTIC ENGINEERING AND PLANNING/research-2026-03-12-session31-*.md`
- No overnight campaign has been run with the Session 29-31 fixes yet

## Handoff Notes
- Do not add `pytest` imports to `test_*.py`; CI does not install `pytest`.
- Python 3.9 CI: avoid runtime `X | None` annotations unless module uses deferred annotations.
- `gh pr merge` may require `--admin` even when checks are green.
- Keep the root worktree on `main` when merging stacked PRs.
- `ruff check` does not validate GitHub Actions YAML.
- Local `git status` may show `?? .claude/`; that is expected.

## Cycle ID
- AEP-2026-03-12
