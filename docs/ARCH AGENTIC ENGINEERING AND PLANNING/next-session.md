# Next Session Checklist

Purpose: Resume from the road-course, self-healing, and attribution batch without reopening disproven promotion paths. Current defaults remain unchanged except for the conservative `0.01` threshold guardrail.

## Session Start
- Review `docs/road-course-results-2026-04-27.md`.
- Review `docs/self-adapting-chelation-seal-eggroll-analysis-2026-04-28.md`.
- Review `docs/seal-eggroll-multipanel-architecture-2026-04-28.md`.
- Review `session-log-2026-04-29-session34.md`.
- Sync local `main` to `origin/main`.
- Confirm there is no stray `run_road_course_tuning_loop.py`, `run_thousand_query_tuning.py`, `static_mask_probe.py`, or `run_live_fire_diagnostics.py` process still alive before starting new research work.

## Priority Order
1. **Do not promote active chelation or defaults beyond `ChelationConfig.DEFAULT_CHELATION_THRESHOLD = 0.01`.**
   - road-course, 1,000-query, 5,000-query, gate-learning, and masking probes produced no shippable active candidate
   - aggressive thresholds around `0.0004-0.0025` remain research-only
2. **Treat self-healing and learned-mask tooling as advisory or evaluation-only surfaces.**
   - `self_healing_chelation.py` is adapter-only and intentionally non-persistent
   - static, conditional, regularized, and classifier-gated masks are not safe candidates from current holdout evidence
3. **If more retrieval evaluation is desired, start from query-conditional or route-limited candidates only.**
   - `reform_rrf_v2` is the only weak retest lead from current attribution evidence
   - require holdout validation, transfer checks, quantization survival, and structural-health gates before any promotion discussion
4. **Prefer broader evidence over more local slice fishing.**
   - compare baseline, `0.01` guard, and limited reformulation on broader BEIR or multitask slices
   - use `research_pathway_analyzer.py` and query-attribution rows to justify any new candidate family
5. **Keep synthetic-collapse results scoped correctly.**
   - `synthetic_collapse_benchmark.py` proves the fixture only
   - `learned_mask_policy.py` and `static_mask_probe.py` are research scaffolds until a holdout-positive candidate exists
6. If an RP2040 / Pico device is attached, run `python computational_storage_poc/capture_hardware_evidence.py <drive-number-or-path> --notes "..."`.

## Current State
- The safety harness now covers instrumentation, component benches, control surfaces, closed-course calibration, failure injection, and road-course campaign runners.
- The conservative `0.01` threshold guardrail is the only default change with current support.
- Self-healing planning, candidate provenance, eval probes, sandbox seams, and live-fire reporting are implemented, but persistent self-adaptation is still gated.
- Gate-learning, query attribution, synthetic collapse, learned masking, static/conditional masks, and research meta-analysis are implemented; no golden setting or shippable gate emerged.

## Handoff Notes
- Do not add `pytest` imports to `test_*.py`; CI does not install `pytest`.
- Python 3.9 CI: avoid runtime `X | None` annotations unless module uses deferred annotations.
- `gh pr merge` may require `--admin` even when checks are green.
- If a benchmark campaign is no longer the active task, stop it instead of leaving it alive in the background.
- `ruff check` does not validate GitHub Actions YAML.
- Local `git status` may show `?? .claude/`; that is expected.

## Cycle ID
- AEP-2026-04-27
