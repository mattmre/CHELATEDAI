# 10-minute priority loop — goal and worker charter

**Duration:** 10 minutes wall clock  
**BHS target:** 100/100 on every file touched in this loop  
**Block gate:** `docs/next-session.md` may remain BLOCKED until SHIM blocking rows close; debt-draining work uses `python scripts/check_block_flag.py --allow-debt-prs`.

## Goal (operator)

Drain Tier A shim substrate debt without doc-only slices:

1. Refresh production-path evidence: `VectorSteerer.steer`, `TTSPipeline.apply`, `AntigravityEngine.run_inference` + `enable_tts` under `CHELATED_SHIM_RESEARCH=1`.
2. Promote minimal `ShimRegistry` surface to repo root (`shim_node_promoted.py`) when BHS smoke passes — not import from `docs/.../artifacts/` at runtime.
3. Iteratively raise BHS on `chelated_shim_research.py`, `tts_pipeline.py` seam, and promotion module until structural validator reports no blocking issues.
4. Update `docs/next-session.md` only with honest **CLOSED** rows backed by tests + artifacts.

## Worker roles (iterative improvement)

| Role | Mandate |
|------|---------|
| **Goal** | Keep scope on executable slices in `scripts/phase_development_loop.py`; reject doc-only backlog additions while SHIM-CD-01/02/05/06/08/09 are OPEN. |
| **Implementer (B/C)** | Smallest diff that adds observable runtime evidence; match existing style; no broad `except Exception: pass`. |
| **Reviewer (D/J)** | Score each change toward BHS 100: real tests, no stubs, no keyword-padding in findings. Re-run failing tests before claiming done. |
| **Integrator (E)** | Run `bash scripts/verify_shim_development.sh` and `python scripts/check_block_flag.py` after each turn; record artifact paths. |

## BHS 100 checklist (per changed module)

- Load-bearing code is exercised by unittest (not only scripts).
- Research guards default OFF (`CHELATED_SHIM_RESEARCH` unset in CI).
- Evidence JSON written under `artifacts/` with `research_shim_guard` when env on.
- No new imports from `docs/steering_chelation_rag_dag_research/artifacts/` except via promotion copy to root.
- Carried-debt table updated only when Status can truthfully start with **CLOSED**.

## Stop conditions

- Wall clock elapsed (10 min), or
- All executable slices in phase loop `completed`, or
- PAUSE gate with no executable slice and OVERRIDE not active (sleep/retry per loop driver).