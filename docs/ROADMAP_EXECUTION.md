# Execution queue (operator priority)

**Effective:** 2026-06-03  
**Policy:** SHIM substrate rows (SHIM-CD-01, 02, 06, 08, 09) are **on hold / last**. Core engine and research-validity work runs **one track at a time** until each step meets its exit criteria.

## Block flag

Merge gate: **CLEAR** for this sequence (`docs/next-session.md`). SHIM debts stay **OPEN** but **non-blocking** until the queue below finishes.

## Sequence (do not parallelize)

| Step | Track | Exit criteria |
|------|--------|----------------|
| 1 | **ML correctness — projection** | `DimensionProjection` trains (no erroneous `.detach()` on teacher path); regression test fails if reverted |
| 2 | **ML correctness — InfoNCE** | `SedimentationInfoNCELoss` does not use inter-doc false negatives; tests green |
| 3 | **Infra hygiene** | `isolated_adapter_state()` leak fixed; `*.pt` backup pattern gitignored; no stray backup files |
| 4 | **Sweep / CI cost** | `run_large_sweep.py` O(N²) JSON path fixed or bounded; CI torch cache documented |
| 5 | **Packaging** | Missing `py-modules` in `pyproject.toml` added; `pip install -e .` smoke passes |
| 6 | **Docs truth** | `CLAUDE.md` + `CHANGELOG.md` reflect current entrypoints and smoke paths |
| 7 | **Model-Scope integration** | Shadow steering pilot on Qwen3.5-9B (one policy, provenance persist) per architecture-2026-05-01 phases 4–6 slice |
| 8 | **E2E learning loop** | Single test: ingest → sedate → measurable metric delta on fixture corpus |
| **Last** | **SHIM substrate** | Resume SHIM-CD-01/02/06/08/09 only after step 8; restore `Blocking=YES` when re-entering shim program |

### Execution posture

- Track work one **step** at a time, but do not leave long checks idle: if any test command in a step runs longer than ~30 seconds, launch at least one independent, lower-cost validation in parallel (state checks, docs check, packaging import smoke, or debt table audits).
- Keep the operator queue explicit: no step is considered complete until the long-running validation and the parallel companion check both pass.
- Use this automation to enforce the policy during execution:
  - `bash scripts/run_step_with_checks.sh --always --threshold 30 --companion "python -m unittest tests.test_check_block_flag -v" -- python -m unittest -v tests/test_model_scope_runtime.py`
  - Replace the primary command for each queue step; add step-specific companions as needed (for example, a docs audit or focused evidence smoke) and keep each check independent of the long-running step.

## Automation

- Phase loop prefers **CORE-SLICE-*** handlers (low `priority` number) before **SHIM-SLICE-*** (priority ≥ 200).
- Ten-minute loops: `bash scripts/loop_core_10m.sh` (core track) instead of shim BHS loop unless explicitly requested.

## What we are not doing (while on this queue)

- No new doc-only shim cycles or 10-agent backlog slices.
- No claiming SHIM-CD **CLOSED** without full substrate DoD.
- No Model-Scope + full shim + EGGROLL at full throttle in parallel.
