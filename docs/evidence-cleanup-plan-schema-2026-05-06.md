# Evidence Cleanup Plan Schema

Purpose: document the generated JSON contract for cleanup dry-run plans and the dashboard API that summarizes them. Cleanup planning is read-only and must never delete files.

## Generator

```bash
python plan_evidence_artifact_cleanup.py --root experiment_runs --keep-latest 1 --output experiment_runs/evidence-cleanup/latest/cleanup_plan.json
```

The command prints the plan to stdout and optionally writes the same JSON to `--output`.

## `evidence_artifact_cleanup_plan`

| Field | Type | Meaning |
| --- | --- | --- |
| `record_type` | string | Always `evidence_artifact_cleanup_plan` |
| `dry_run` | boolean | Always `true`; this command does not delete files |
| `root` | string | scanned generated-output root |
| `keep_latest` | integer | number of newest artifacts retained per artifact type |
| `summary.candidate_count` | integer | total files listed as cleanup candidates |
| `summary.retained_count` | integer | total files retained by the keep-latest rule |
| `summary.candidate_bytes` | integer | total byte size of candidate files |
| `candidates[]` | array | candidate file records |
| `retained[]` | array | retained file records, included by the CLI output for auditability |

Candidate and retained records use the same shape:

| Field | Type | Meaning |
| --- | --- | --- |
| `artifact_type` | string | artifact family matched by the planner |
| `path` | string | repo-relative or root-relative POSIX-style path |
| `modified_at` | number | filesystem modified timestamp |
| `size_bytes` | integer | file size in bytes |
| `disposition` | string | `candidate` or `retain_latest` |

## Dashboard API

The dashboard reads a compact version at:

```text
GET /api/evidence_cleanup_plan?keep_latest=1&limit=25
```

Response fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `record_type` | string | cleanup plan record type |
| `dry_run` | boolean | always `true` |
| `root` | string | scanned generated-output root |
| `keep_latest` | integer | keep-latest value used for the plan |
| `summary.candidate_count` | integer | total cleanup candidates before row limiting |
| `summary.retained_count` | integer | total retained artifacts |
| `summary.candidate_bytes` | integer | total candidate bytes |
| `summary.candidate_types` | string array | artifact types currently represented in candidates |
| `candidates[]` | array | row-limited candidate records |

The dashboard intentionally omits the full `retained[]` list and displays only `summary.retained_count`, keeping the response compact while preserving deletion-safety context.

## Interpretation

- A candidate is safe to inspect, not safe to delete automatically.
- A retained artifact is the newest file for its artifact type under the current `keep_latest` rule.
- If generated files are deleted manually, regenerate the evidence index and rerun the freshness audit.
- Source scripts, schemas, runbooks, phase summaries, and test fixtures are outside the cleanup planner's deletion scope.
