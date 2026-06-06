# Stowed SIP draft notes (from production comment blocks)

Extracted 2026-06-02 from uncommitted `[DRAFT] THIN GUARDED SIP WRAPPER` comment blocks in
`antigravity_engine.py` and `tts_pipeline.py`. These were research-only placeholders (Agent 4,
BHS Cycle 010); they are **not** executable wiring.

## SHIM-CD-01 target seams

| Host | Location | Stage |
|------|----------|-------|
| `antigravity_engine.py` | ~2452 | Post-embed, pre-TTS / retrieval |
| `antigravity_engine.py` | ~2582 | Pre-variance / chelation decision |
| `tts_pipeline.py` | `VectorSteerer.steer` | Primary steering SIP (first prod probe) |
| `tts_pipeline.py` | `clear_signals` | Secondary (not drafted in removed block) |

## Guard convention

- Env: `CHELATED_SHIM_RESEARCH=1` (default OFF).
- Harness reference: `shim_collapse_benchmark_extension.py` (`--research-shim`).
- Goal doc: `docs/steering_chelation_rag_dag_research/BHS_5MIN_SHIM_LOOP_GOAL.md` (backlog #1, #9).

## MinMax pre-filter sketch (not implemented in comments)

When research guard is on, a future `MinMaxBlockRelevanceScorer` (research artifacts only) would
gate expensive paths (TTS, full steer, variance) via cheap block min/max signals. No production
import until BHS promotion.

## Antigravity post-embed seam (was ~2452)

- Context: after `q_vec` embed + static mask, before `_tts.apply` and retrieval.
- Backlog #1 (minimal SIP) + #9 (MinMax pre-filter).
- Zero change when guard false.

## Antigravity variance seam (was ~2582)

- Context: before `dim_variances` / `global_variance` chelation gate.
- Proximity to existing broad `except` and chelation threshold — future wiring must not add L11 risk.

## VectorSteerer seam (implemented)

Guarded behavior in `tts_pipeline.py` `VectorSteerer.steer` when `CHELATED_SHIM_RESEARCH=1`.

## Antigravity seams (implemented, metadata-only)

- `AntigravityEngine.post_embed` — before TTS intercept
- `AntigravityEngine.chelation_variance` — before variance / chelation decision
- `_build_runtime_diagnostics` includes `research_shim` when preflight ran

Shared helpers: `chelated_shim_research.py` (incl. `collect_research_probe_from_tts_metadata`).
Evidence: `scripts/record_shim_prod_evidence.py`, `scripts/record_shim_tts_intercept_evidence.py`.
Verification: `bash scripts/verify_shim_development.sh`.

## References

- `docs/next-session.md` SHIM-CD-01 … SHIM-CD-09
- `docs/steering_chelation_rag_dag_research/loop_02/22_agentB_build_SHIM_CD_01_VectorSteerer_minimal_guarded_diff.md`
- Cycle-010 evidence: `artifacts/bhs_shim_evidence_Cycle-010-20260527_0400.json`