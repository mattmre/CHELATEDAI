# Phase I step 4 — sweep JSON persistence (2026-09-23)

`docs/ROADMAP_EXECUTION.md` step 4 asks for the `run_large_sweep.py` O(N²) JSON path to be fixed or bounded, and for the CI torch cache to be documented. This change does the JSON path only.

Before this change, each of the 7,350 configurations did `json.load` on `{prefix}_results.json`, appended one object, and `json.dump` of the whole list. CSV writes were already append-only.

Now `sweep_result_store.append_jsonl` adds one line to `{prefix}_results.jsonl` and flushes it. `migrate_json_array_to_jsonl` copies an existing JSON array into that log once, and does nothing if the log is already there. `materialize_json_array` reads the log once after the loop and replaces the JSON array through a temporary file. `run_large_sweep.py` no longer calls `json.load` or `json.dump`.

`test_sweep_result_store.py` checks five appends, a one-time migration, a bad JSONL line, and that the sweep script does not load the array. It does not run the 7,350-config grid, MTEB, or sedimentation.

Not in this change: `run_large_sweep` is still absent from `pyproject.toml` `py-modules`. The CI torch cache is not documented. The sweep still reuses one engine and one Qdrant path across configurations (`docs/ARCH AGENTIC ENGINEERING AND PLANNING/planning/2026-05-reconciliation/panel-analysis/03-performance-scale.md` V-02). That reuse is unchanged.

The historical ledger `docs/status-corrections-2026-09-23.md` still describes the old read-modify-write on an earlier commit. It was not rewritten.
