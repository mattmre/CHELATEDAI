# Next Session Checklist

Purpose: Start from the Session 33 follow-up no-promotion outcome instead of reopening the same candidate path. Current defaults remain unchanged.

## Session Start
- Review `docs/weight-refinement-campaign-results-2026-04-25-session32.md`.
- Review `docs/weight-refinement-follow-up-results-2026-04-25-session33.md`.
- Review `session-log-2026-04-25-session33.md` for the exact follow-up decisions and tooling changes.
- Sync local `main` to `origin/main` after the Session 33 PRs are merged.
- Confirm there is no stray `run_weight_refinement_campaign.py`, `benchmark_distillation.py`, or candidate-transfer helper process still alive before starting new research work.

## Priority Order
1. **Do not promote presets from Session 32 or Session 33.**
   - Session 32 ended with an explicit no-promotion outcome.
   - Session 33 confirmed SciFact repeatability for `mlp` + `teacher_weight=0.3`, but the corrected candidate-specific small multitask gate failed on `NFCorpus`.
2. **Do not continue medium multitask or BEIR for the same `mlp` + `teacher_weight=0.3` candidate.**
   - the first corrected transfer gate already failed
   - more depth on the same candidate is not justified
3. **If implementation PRs are still open, finish review and merge the tooling/docs only.**
   - PR `#111`: stale-roadmap wording cleanup
   - PR `#112`: repeatability helper plus candidate-transfer gate tooling
4. **If more evaluation is explicitly desired, start from a new candidate or changed hypothesis.**
   - do not reuse the Session 33 candidate as if it were still promotion-viable
   - require candidate-specific transfer evidence via `run_candidate_transfer_gate.py`
5. **Keep optional research variants behind a new candidate that first clears repeatability and small transfer.**
   - `bounded=True` adapter wrapping
   - `infonce` or `hybrid` sedimentation loss variants
   - Kalman-gain LR variants
6. If an RP2040 / Pico device is attached, run `python computational_storage_poc/capture_hardware_evidence.py <drive-number-or-path> --notes "..."`.

## Current State
- Session 32 merged PRs `#107`, `#108`, and `#110`
- Session 33 confirmed the only positive Session 32 candidate is repeatable on SciFact but not transfer-safe on the corrected small multitask gate
- `run_candidate_transfer_gate.py` is now the required path for future single-candidate transfer evidence
- no default or preset refresh is justified from Session 32 or Session 33

## Handoff Notes
- Do not add `pytest` imports to `test_*.py`; CI does not install `pytest`.
- Python 3.9 CI: avoid runtime `X | None` annotations unless module uses deferred annotations.
- `gh pr merge` may require `--admin` even when checks are green.
- If a benchmark campaign is no longer the active task, stop it instead of leaving it alive in the background.
- The generic `benchmark_multitask.py` and `benchmark_beir.py` outputs are auxiliary repo benchmarking, not promotion evidence for a specific distillation candidate.
- `ruff check` does not validate GitHub Actions YAML.
- Local `git status` may show `?? .claude/`; that is expected.

## Cycle ID
- AEP-2026-04-24
