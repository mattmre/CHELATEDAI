# ARCH-AEP + BHS v3.7.1 inventory — 2026-08-29

Cycle `AEP-20260829-1`; scope lock and control plane: [08-cycle-control.md](08-cycle-control.md). Authoritative tracker: [04-master-backlog.md](04-master-backlog.md). Durable cycle index: [../README.md](../README.md).

## Scope and posture

Plan-only FE/backend/boundary remediation audit. No product code, existing product artifact, GitHub state, branch, or index was mutated; only this new audit directory was created. Tests created or updated ignored runtime scratch under their normal contracts, which was not treated as a product edit. Completion claims were treated as unproven until opened evidence survived an adversarial check. Unit tests are recorded as bounded evidence only.

Audit output: `docs/aep-remediation/20260829/` (new cycle; no prior directory was overwritten).

## Method authority

- EVOKORE workflow resolution selected `aep-framework` / ARCH-AEP, `panel-of-experts` / code refinement, and `brutal-honesty-validation`.
- Repository guidance opened: `CLAUDE.md`, `docs/CLAUDE.md`, `docs/next-session.md`, and the local BHS mirror.
- Requested canonical BHS authority opened from the public kit implementation path: [v3.7.1 charter](https://github.com/mattmre/Brutal-Honesty-Kit/blob/main/v3.5/CHARTER-v3.7.1.md) and [v3.7.1 rulebook](https://github.com/mattmre/Brutal-Honesty-Kit/blob/main/v3.5/rulebook/brutal-honesty-rulebook.md).
- The checkout still embeds BHS v3.3 and L1–L13 (`CLAUDE.md:7,30`; `scripts/bhs_validator.py:2,41,57`). Therefore v3.7.1 L14–L16 were applied manually in this audit; no claim is made that repository automation enforces them.

## Repository snapshot

| Item | Value |
|---|---|
| origin | `https://github.com/mattmre/CHELATEDAI.git` |
| branch | `codex/prime-ring-onion-method-dev` |
| HEAD | `65c9085cd048e8a7351a53e87666fdd5639e612b` |
| upstream delta | `+0/-0` against its upstream; `+39/-32` against `origin/main` at inventory time |
| root status | dirty: 7 tracked modifications, 74 untracked paths |
| linked worktrees | 20 total; 6 dirty |
| open PRs | 10 |
| block flag | `BLOCKED`; expired `CD-MLR-01`, `CD-R13-01`, `CD-R16-01` (`docs/next-session.md:20-25,63-65`) |

Tracked root modifications: `.gitattributes`, `docs/next-session.md`, `findings.md`, `paired_intervention_experiments.py`, `progress.md`, `run_paired_intervention_sanity.py`, `tests/test_paired_intervention_experiments.py`. Untracked files span paired intervention, QSCCI, EGV private/canary helpers, protocols, reports, and evidence; see `01-worktrees-index.md` and finding `AEP-20260829-WT001-001`.

## FE and backend roots

| Surface | Roots and entry points |
|---|---|
| Browser client | `dashboard/index.html`; inline fallback in `dashboard_server.py` |
| Dashboard boundary | `dashboard_server.py`; `/dashboard/`; `/api/*`; `python dashboard_server.py --port 8080` |
| Core runtime | `antigravity_engine.py`, `chelation_adapter.py`, `embedding_backend.py`, `vector_store.py`, `checkpoint_manager.py` |
| AEP/orchestration | `aep_orchestrator.py`, `scripts/bhs_validator.py`, `scripts/validate_pr_brutal_honesty.py` |
| Research DAG/routing | `evidence_dag.py`, `recursive_decomposer.py`, `adapter_router.py`, semantic-cache and prime-ring modules |
| EGV | `egv/` plus EGV campaign worktrees |
| RHPC | uncommitted `D:/GITHUB/CHELATEDAI-RHPC1/rhpc/`, runner, protocol, runbook, and tests |
| Computational storage | `computational_storage_poc/` |

## Canonical commands and current results

| Command | Result | Evidence boundary |
|---|---|---|
| `python scripts/smoke_pipeline.py` | PASS, exit 0 | Real floor imports plus AntigravityEngine construction/ingest/embed. Adapter checkpoint failed dimensional load and fell back to identity; the smoke still passed, so it does not prove checkpoint compatibility. |
| `python -m unittest test_dashboard_server -v` | 67 passed, 2 skipped | Handler/loader unit evidence. A real token-enabled HTTP probe still reproduced FE001. |
| `python -m unittest tests.test_paired_intervention_experiments -v` | 25 passed | Dirty root slice only; not immutable-head acceptance. |
| RHPC: `python -m unittest tests.test_rhpc_stage_a -v` | 15 passed | Dirty RHPC worktree only; tests include a fabricated source SHA accepted as official. |
| `python scripts/validate_v33_schema_drift.py` | PASS, exit 0 | Proves local v3.3 mirror consistency only; not v3.7.1 enforcement. |
| `python -m ruff check .` | PASS, exit 0 | Direct `ruff` executable was absent, but the installed Python module ran successfully: `All checks passed!` |
| full `python -m unittest discover -s . -p "test_*.py" -v`, then quiet full rerun | first run observed 2 failures but raw stream was not sealed; rerun passed 3,633 with 16 skips | The pass is sealed in `full-suite-raw-receipt.txt`; the first run is diagnostic only. A separate sealed deterministic checkpoint probe establishes BE002-001. |
| `gh pr list/view/checks`, GraphQL review threads, failed logs and diffs | completed | All ten PRs and 27 unresolved threads were opened; PR #257 diff used exact local base..head because GitHub diff API returned 406 at 351 files. |

## Agent ownership

- A1 Worktree Scout: worktree inventory and Tier 1 findings; later finding normalizer.
- A2 PR Auditor: all open PRs, comments, threads, checks, failed logs, exact diffs; later BE convergence.
- A3 FE Panel / Alex Rivera: dirty/PR FE lanes followed by full-repo FE fallback and live HTTP probe.
- A4 root / Sofia Andersson lead, challenged by James Okafor and Margaret Chen: backend, reliability, implementation quality, synthesis, and audit self-score.

Tier B is deliberately not one of A1–A4 and is recorded in `07-bhs-adversarial-scorecard.md`.
