# Closeout addendum — 2026-09-03 (post-merge record)

Supplements the 20260902 closeout package. New file; no prior round folders
were modified (several are currently missing from the workspace — see incident
note below).

## STATIC-01 Critical re-rank question: CLOSED (stays High, fixed)

The re-rank condition was unauthenticated dotfile reach. Live probes
2026-09-03 against post-#311 `main` (token set, no creds):
`GET /dashboard/index.html` → 401, `HEAD /dashboard/index.html` → 401,
`GET /.git/config` → 401, `HEAD /.git/config` → 401.
The earlier `HEAD /.git/config → 200` observation was on pre-#311 code; #311's
`do_HEAD` (auth + `/dashboard/` confinement) plus the fail-closed default gate
close the oracle. No Critical rank. No further action.

## P1-01 HEAD bypass: CLOSED (fixed by #311)

Same probe series: unauthenticated HEAD now 401s (was 200 pre-#311). The panel
finding is retained as the discovery record; no separate fix PR needed.

## Already-fixed-by-#309–311 (verified on `main`, no new PR)

- P2-02 (`data.reason`), P2-03 (`outcome`/`nodeid`), P1-11 (disk-LLM error):
  all interpolations now go through `escapeHTML()` — 14 escaped sites.
- P3-04 (bind guard): `run_server` raises `ValueError` on non-loopback bind
  without a token; startup banner prints the auth mode. No change needed.
- PROV-01 core (`get_git_metadata`) was already `try/except`-guarded; Batch A
  adds 10s timeouts against hangs.

## Incident: workspace docs loss (~17:56 EDT 2026-09-02)

Missing from the worktree: `docs/aep-remediation/20260902/`,
`20260902-r2/`, `20260902-r3/`, `20260902-r4/`, and most top-level files of
`20260902-closeout/` (only `findings/` + `verification-log.md` survive).
Cause unknown; not caused by the merge sequence (all merge work is accounted
for above; deletion predates it). Surviving sources: p1–p3 rounds,
`panels-index.md`, PR bodies #309–#311/#317+ (contain BHS records), CI logs.
Recovery: attempted only on operator request — no content is reconstructed
from memory in this file.

## PR295-01: CLOSED as FALSE_POSITIVE (hammer + audit, 2026-09-03)

Lock-scope audit of `adapter_router.py`: every shared-state access
(`_routes`, `_last_route_outcome`, `_route_history`) is under
`with self._lock` — `register` (:50), `select` snapshot (:57-58),
`record_outcome` (:112-114), all getters (:124,128,132). `select` copies
under lock then computes outside (safe pattern); no cross-call
check-then-act exists. Threaded hammer (16 threads x 400 mixed
select/record/register/getters, project venv numpy 2.5.2):
0 errors, 0 stuck, history correctly capped at 256, 8/8 routes intact
→ HAMMER_PASS. The "race" premise is falsified both statically and
empirically; no product change needed. Locked in by regression test
`test_adapter_router_concurrency.py` (8 threads x 100 ops, history-cap
asserted — re-runnable in CI, no /tmp dependency). Demote-or-drop resolved as
drop. PR #295 itself (lattice rung16) is unaffected by this verdict.

## Deferred for a free-Spark window (owner: operator, TTL: next window)

- Browser-DOM-fire proof for XSS-class fixes (no harness installed).
- BEIR/model transfer checks and torch-dependent validation (CI covers the
  mockable subset; EGV campaigns untouched).
- PR295-01: resolved FALSE_POSITIVE (see above; regression test landed).
- Full-suite combined-tree matrix post-merge (CI matrix is the gate).
