# Next Session

<!--
  This file is the Tier C (cross-PR / cross-session) state surface for the
  Brutal Honesty Rulebook v3.2. It is read by:
    - scripts/check_block_flag.py  (the merge gate for new feature work)
    - the Tier B reviewer of the next PR you open
    - whoever picks the work back up after a session boundary

  Schema is fixed. Do not rename headings. Do not collapse the tables. Empty
  state is honest state — empty rows do not get deleted, they get filled with
  the "_none yet_" placeholder which the validator recognizes.

  Cycle definition (per rulebook §6.1): 1 cycle = the next operator-initiated
  session OR 5 calendar days, whichever comes FIRST. Items with TTL=1 cycle
  must be cleared before the next cycle starts or the block flag flips to
  BLOCKED automatically.
-->

## Block flag

**Current**: `CLEAR` — no Carried Debt items have expired.

When the flag is `BLOCKED`, no new feature work may merge until Carried Debt
is empty. The flag is set automatically by `scripts/check_block_flag.py`:
- `CLEAR` if no Carried Debt rows OR all open rows are still in their first
  cycle (TTL not yet expired).
- `BLOCKED` if any open Carried Debt row has survived a full cycle without
  being closed.

The script does not know cycle age — that is set by the operator at session-
wrap by inspecting the TTL column. The `**Current**:` line above is the
authoritative source; everything else is advisory.

## Carried Debt

| ID | Item | Source | TTL | Blocking | Status |
|----|------|--------|-----|----------|--------|
| CD-001 | smoke_pipeline.py ceiling-tier not yet implemented; floor-tier only (`run_ceiling_smoke()` returns sentinel 2). Ceiling gap = no real end-to-end fixture exercise of AntigravityEngine | kit install 2026-05-10 | 1 cycle | NO — honestly disclosed per Rule 5 | **CLOSED** by PR <pending consolidation PR> — `run_ceiling_smoke()` now constructs `AntigravityEngine(qdrant_location=":memory:", model_name="all-MiniLM-L6-v2")`, ingests 4 docs, runs `get_chelated_vector()` + `embed()` against the production code path, asserts non-zero vector with `vector_size=384`; honest skip path retained for envs missing torch / sentence-transformers / qdrant; covered by `test_smoke_pipeline_ceiling.py` (8 tests) |
| CD-002 | `scripts/smoke.sh` Stage 1 exits non-zero: `tests/test_e2e_smoke.py` does not exist; smoke.sh is the `bash`-mode entry point but the repo has no e2e smoke test file. The Python `smoke_pipeline.py` path (used by CI and operator) is unaffected. | kit v3.3 upgrade 2026-05-12 | 1 cycle | NO — CI uses `smoke_pipeline.py` directly; gap is only in the `bash scripts/smoke.sh` code path | **CLOSED** by PR <pending consolidation PR> — added `tests/test_e2e_smoke.py` (unittest surface-boot covering `antigravity_engine` + 8 load-bearing modules and the `AntigravityEngine` entry-point class); swapped `smoke.sh` Stage 1 invocation from `python -m pytest tests/test_e2e_smoke.py` to `python -m unittest -v tests.test_e2e_smoke` per CLAUDE.md (CI has no pytest) |
| CD-244-01 | `scripts/bhs_validator.py:43-50` `validate_pr_brutal_honesty()` returns hardcoded `BHSResult(score=0.0)`; `:53-58` `run_smoke_pipeline()` always returns `True`. AEP orchestrator hooks call these so `summary["avg_bhs_score"]` is always `0.0`. L1 + L4. Violates Session Rule #1. | PR #244 (2026-05-16) | 1 cycle | YES — load-bearing stub | **CLOSED** by PR #245 |
| CD-244-02 | `aep_orchestrator.py:679, 745, 919-926` consume `bhs_metadata` from stub call; `summary["avg_bhs_score"]` always `0.0`. After CD-244-01 lands, verify the score actually varies with finding content and is surfaced in operator-facing closure summary. | PR #244 (2026-05-16) | 1 cycle | YES — depends on CD-244-01 | **CLOSED** by PR #245 |
| CD-244-03 | New comp-storage modules (`computational_storage_poc/moe_reap.py`, `sparse_cpu_inference.py`, `packed_graph.py`, `packed_cpu_inference.py`, `repo_graph_memory.py`, `integrated_repo_runtime.py`, `phase7_system_evaluation.py`, `disk_llm_estimator.py`, `cpu_backends.py` + benchmarks) have unit tests but zero references from production paths. L4 + L8. | PR #244 (2026-05-16) | 1 cycle | NO — POC scoped + honestly disclosed | **CLOSED** by PR #247 — `disk_llm_estimator` wired into `dashboard_server.py` `/api/disk_llm_estimate` rendered in dashboard campaigns-tab panel; other 8 modules marked `EXPERIMENTAL = True` + load-bearing `mark_experimental()` import-time check; tabulated in `computational_storage_poc/README.md` Status section |
| CD-244-04 | `aep_orchestrator.py:33` catches `Exception` (not `ImportError`) around the BHS import; any future runtime error in `scripts.bhs_validator` is silently absorbed. L11 risk. | PR #244 (2026-05-16) | 1 cycle | NO — post-CD-244-01 cleanup | **CLOSED** by PR #245 |
| CD-244-05 | `computational_storage_poc/model.cspg` binary artifact tracked in repo via PR #244; decide ignore/LFS/remove. | PR #244 (2026-05-16) | 1 cycle | NO — hygiene | **CLOSED** by PR #246 — Option 3 (untrack + gitignore `*.cspg`); README documents regeneration commands |
| CD-245-01 | `scripts/bhs_validator.py` `_score_finding` rubric is length-based + keyword-based, not semantic. Three Tier B iterations (92/96/85) converged on this: `"xxxxxxxxxxxx"` (12 identical chars) passes the min-content check; padded keyword-bait can hit score 100 without committed prose. L13 (soft-prose-claimed-as-mechanical). Per rulebook §6.1, same gap class surviving 2 iterations escalates — flagging as Tier C debt rather than looping further. | PR #245 (2026-05-16) | 1 cycle | NO — rubric depth research, not an L1/L4 in production data flow | **CLOSED** by PR <pending consolidation PR> — Candidate A + B per research-agent recommendation: entropy/unique-token/dominant-token content-quality penalty in `scripts/bhs_validator.py::_content_quality_penalty` (drops `"xxxxxxxxxxxx"` to 85), `_score_finding` renamed to `_score_finding_structure` with backwards-compat alias, scope documented in `docs/bhs-rubric-scope.md`, operator audit script at `scripts/audit_findings.py` for periodic human sample-grade. Residual diverse-but-meaningless gap (`"foo bar baz qux at handler.py:42"` still scores 100) is honestly acknowledged in the scope doc and asserted in `test_bhs_validator.py::test_diverse_but_meaningless_prose_acknowledged_gap_scores_100` so any future "we closed it" claim must actually change the rubric. |

**Schema**:
- `ID`: stable identifier, prefix `CD-` + sequential number (CD-001, CD-002, ...).
- `Item`: one short sentence stating the gap. Reference file:line where useful.
- `Source`: PR number or session ID that opened the debt.
- `TTL`: `1 cycle`, `expired`, or `—` (placeholder row only).
- `Blocking`: `YES` (forbids new feature work in next cycle if not cleared)
  or `NO — <reason>` (cosmetic / honestly-disclosed gap that does not block).
- `Status`: `OPEN — <note>`, `**CLOSED** by PR #N`, or `_—_` (placeholder).
  Rows starting with `CLOSED` (case-insensitive, stripping `**` markdown bold)
  are filtered out of the active-debt count by `check_block_flag.py`.

## Deferred Scope

Items from this cycle's PRs whose `DEFERRED_SCOPE:` field captured ≥25% of
original scope. These are NOT debt — they are honestly-bounded
not-in-scope items. They appear here so the next planner sees them.

| ID | Item | Source PR | Why deferred |
|----|------|-----------|--------------|
| _none yet_ | _—_ | _—_ | _—_ |

## Aggregate BHS trend

Track `BHS_OFFICIAL` per merged PR over the last 5 cycles. Falling trend = the
remediation loop is succeeding. Flat-at-100 trend = either real progress or the
loop is being gamed (Tier B agents not adversarial enough). Spot-check the
Tier B reports if the trend looks suspicious.

| Cycle | PRs merged | Avg BHS_OFFICIAL | Avg LOOP_ITERATIONS | OPERATOR_OVERRIDE count |
|-------|------------|------------------|---------------------|------------------------|
| _current_ | _—_ | _—_ | _—_ | _—_ |

## Operator overrides log

Every PR merged at `BHS_OFFICIAL < 100` (i.e. with `OPERATOR_OVERRIDE:`
populated) gets a permanent row here. The override creates an automatic top-
priority Carried Debt entry; this log is the audit trail.

| PR | BHS_OFFICIAL at merge | Override reason | Override author | Out-of-band ref |
|----|----------------------|-----------------|-----------------|-----------------|
| #244 | 55 | reconciliation foundation must land so follow-up cycle can implement CD-244-01..05 against canonical main | mattmre | `docs/next-session.md` Carried Debt CD-244-01..05 |
| #245 | 85 | three Tier B iterations converged on rubric-depth gameability (length-based not semantic); per §6.1 same-gap-2-iterations rule, escalating to Tier C as CD-245-01 rather than looping further | mattmre | `docs/next-session.md` Carried Debt CD-245-01 |

---

**Last session**: 2026-05-16 — PR #244 reconciliation merge (BHS_OFFICIAL=55, OPERATOR_OVERRIDE)
**Last validated by `check_block_flag.py`**: run after this PR merges
