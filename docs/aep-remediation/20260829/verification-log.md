# Verification log

| Time/order | Command or inspection | Exit/result | Interpretation |
|---:|---|---|---|
| T0 | remote/branch/HEAD/upstream/status/worktree inventory | completed | exact local snapshot; 20 worktrees, 6 dirty |
| T0 | `gh pr list --state open` | completed | inventory only: 10 PRs; deep review was intentionally deferred to T2 |
| T1 | `python -m unittest tests.test_paired_intervention_experiments -v` | 25 passed | dirty root slice only |
| T1 | RHPC `python -m unittest tests.test_rhpc_stage_a -v` | 15 passed | dirty worktree; does not close provenance gates |
| T1 | initial direct test-file invocation | failed import (`ModuleNotFoundError`) | operator invocation error; corrected module invocation passed; not a product finding |
| T2 | #293 unit slices | 60 passed | regression evidence, not later runtime-disproof override |
| T2 | #294 unit slice | 15 passed | regression evidence only |
| T2 | #295 unit slices | 49 passed | regression evidence only |
| T2 | per-PR bodies/files/checks/reviews/comments, GraphQL threads, failed logs and diffs | completed | 27 unresolved threads; no required checks; strict tier order preserved |
| T3 | `python -m unittest test_dashboard_server -v` | 67 passed, 2 skipped | unit boundary only |
| T3 | token-enabled real HTTPServer probe | page 401 no header / 200 with header; API 401 no header; authorized request reached handler | reproduces FE001; sealed command/output: `http-auth-probe-receipt.txt` |
| T3 | `python scripts/smoke_pipeline.py` | PASS exit 0 | real core runtime path; checkpoint dimensional mismatch warned and identity fallback used |
| T3 | `python scripts/validate_v33_schema_drift.py` | PASS exit 0 | local v3.3 consistency only |
| T3 | `ruff check .` | invocation unavailable | superseded by the successful module invocation below; not a skipped gate |
| T3 | `python -m ruff check .` | PASS: `All checks passed!` | local lint gate cleared |
| T3 | full unittest discovery | observed FAIL: 3,633 run, 2 failures, 16 skips in 278.109s; raw stream not sealed | diagnostic context only; not controlling evidence and not release acceptance |
| T3 | focused rerun of the two failed Qwen smoke tests | 2 passed in 0.257s | proves timing sensitivity, not closure |
| T3 | quiet full unittest discovery rerun | PASS: 3,633 run, 16 skips in 262.471s; aggregate receipt `full-suite-raw-receipt.txt` | contradicts the preceding full-run failure and strengthens BE002-001 nondeterminism; not deterministic release acceptance |
| T3 | deterministic pre-promotion deadline probe | exit 0; `target_is_dir=False` | replayably proves BE002-001's stage-selection assumption; `be002-stage-probe-receipt.txt` |
| T3 | 20 focused reruns of the two deadline tests | 40 tests passed; zero failures | bounded stability result; does not refute deterministic pre-stage expiry probe |
| T3 | bounded stop-line `rg` and README/code comparison | completed | candidate sites triaged; no grep-only finding |
| T4 | corrected cold-command set | PR296 live query exit 0; RHPC 15 tests pass from correct worktree; HTTP and deadline probes exit 0; WT019 visible=3,898/check-ignore=1 | sealed in `cold-command-verification.md` and raw receipts; challenged commands no longer inferred from prose |

An attempted temporary RHPC CLI command was rejected by the execution safety boundary before running because it included recursive cleanup. It produced no product evidence and was not retried destructively.

`SKIPPED_GATES: none` for validation of this plan-only audit artifact after Tier A iteration 4 cold-command execution. Opt-in hardware/model/deployment evidence and operator-only mutations are deferred scope, not converted into passing audit gates. The sealed quiet full suite passed; the earlier unsealed failure is diagnostic only, and BE002-001 is controlled by its sealed deterministic stage probe.

This log is frozen into the Tier A candidate manifest. Final manifest verification and the independent verdict are recorded in the manifest-excluded `07-bhs-adversarial-scorecard.md`; they are not appended here after candidate freeze.
