# Tier 3 — full-repo FE/backend map

Tier 3 began only after Tier 1 worktrees and Tier 2 PR/comments were completed.

## Critical paths opened

| Path | Contract | Audit result |
|---|---|---|
| `dashboard/index.html` ↔ `dashboard_server.py` | browser auth, API status, artifact rendering, lifecycle, keyboard operation | six FE findings; real token HTTP probe disproved browser usability |
| `antigravity_engine.py` → embedding/adapter/vector store | runtime construction, checkpoint load, ingest/embed | canonical smoke passed, but adapter checkpoint dimension mismatch fell back to identity; no checkpoint-compatibility claim accepted |
| `aep_orchestrator.py`, BHS scripts/workflows | callback isolation and honesty gates | open PR callback defect; local enforcement remains v3.3, not requested v3.7.1 |
| evidence DAG/routing/pool lines in #293–#295 | mutation, rollback, publication, promotion | bounded unit slices pass; later exact adversarial probes and durable record reject #293/#295 and require #294 replacement |
| RHPC admission/artifact publication | trusted source, predecessor binding, atomic absent-target publication | three distinct trust failures in dirty worktree; no official run eligible |
| report/sweep producers ↔ dashboard consumers | versioned schema and lifecycle | nonexistent test producer, obsolete schema, inferred “Running” status |

## Stop-the-line scan

Bounded source scan covered non-test Python outside artifacts/experiment outputs for `TODO`, `FIXME`, `NotImplementedError`, `except Exception`, `except BaseException`, bare `pass`, and permissive status assertions. Results were triaged rather than converted mechanically into findings. Abstract methods in `embedding_backend.py`, `vector_store.py`, and computational-storage interfaces were not mislabeled as stubs; explicitly logged fail-closed boundaries were not mislabeled L11. Real findings require a demonstrated contract failure.

The scan reinforced these boundaries:

- dashboard loaders contain broad handler catches, but FE006 is about the client laundering real non-2xx responses, not merely the existence of `except Exception`;
- local BHS scripts identify only L1–L13, creating a v3.7.1 enforcement gap (`AEP-20260829-BE001-001`);
- the production smoke does not cover dashboard JavaScript, browser auth, hostile artifact strings, RHPC official admission, concurrent publication, or current PR heads.

## README/docs versus code (L9/L13)

- Test Tracking documents `generate_report_json.py`, which does not exist, and consumes a pytest-style schema despite canonical unittest commands (FE002).
- RHPC runbook requires the same reviewed immutable source SHA, but production admission only reports whichever clean HEAD it observes (WT020-002).
- RHPC claims absent-target atomic publication, but check-then-replace is not atomic no-replace (WT020-004).
- Live PR completion bodies contradict the later durable rejection record (PR296-001).

## Evidence explicitly not obtained

- No production deployment, remote dashboard session, screen-reader run, Spark official RHPC run, or current replacement PR existed to inspect.
- No full-suite exact-head acceptance was claimed from this mixed dirty checkout.
- Direct `ruff` was unavailable, but `python -m ruff check .` passed. The full suite ran and exposed two suite-load timing failures (BE002-001).
- Tests, docs, file presence, GitHub mergeability, and self-attestation were not promoted to runtime acceptance.
