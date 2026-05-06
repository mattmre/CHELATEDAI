# Evidence Cleanup Plan Schema

Purpose: document the generated JSON contract for cleanup dry-run plans and the dashboard API that summarizes them. Cleanup planning is read-only and must never delete files.

## Generator

```bash
python plan_evidence_artifact_cleanup.py \
  --root experiment_runs \
  --keep-latest 1 \
  --evidence-index experiment_runs/evidence-index/latest/evidence_index.json \
  --freshness-audit experiment_runs/evidence-index/latest/freshness_audit.json \
  --output experiment_runs/evidence-cleanup/latest/cleanup_plan.json
```

The command prints the plan to stdout and optionally writes the same JSON to `--output`.
Add `--fail-on-blocked-review` when a local or CI caller should exit with status `2` if `summary.cleanup_review_allowed` is `false`.

## `evidence_artifact_cleanup_plan`

| Field | Type | Meaning |
| --- | --- | --- |
| `record_type` | string | Always `evidence_artifact_cleanup_plan` |
| `dry_run` | boolean | Always `true`; this command does not delete files |
| `root` | string | scanned generated-output root |
| `keep_latest` | integer | number of newest artifacts retained per artifact type |
| `source_artifacts.evidence_index` | string or null | evidence index path the plan was generated after |
| `source_artifacts.freshness_audit` | string or null | freshness-audit path the plan was generated after |
| `source_status.evidence_index.present` | boolean | whether the linked evidence index exists when the plan is generated |
| `source_status.freshness_audit.present` | boolean | whether the linked freshness audit exists when the plan is generated |
| `summary.cleanup_review_allowed` | boolean | `false` when linked source artifacts are missing |
| `summary.missing_source_artifacts` | string array | missing linked source artifact names |
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
| `summary.cleanup_review_allowed` | boolean or null | `false` when linked source artifacts are missing; `null` only for legacy plans without the field |
| `summary.missing_source_artifacts` | string array | linked source artifact names missing when the dashboard plan is generated |
| `summary.candidate_types` | string array | artifact types currently represented in candidates |
| `candidates[]` | array | row-limited candidate records |
| `source_artifacts.evidence_index` | string or null | linked evidence index path |
| `source_artifacts.freshness_audit` | string or null | linked freshness audit path |
| `source_status.evidence_index.present` | boolean | whether the linked evidence index exists |
| `source_status.freshness_audit.present` | boolean | whether the linked freshness audit exists |

The dashboard intentionally omits the full `retained[]` list and displays only `summary.retained_count`, keeping the response compact while preserving deletion-safety context.

## Interpretation

- A candidate is safe to inspect, not safe to delete automatically.
- A retained artifact is the newest file for its artifact type under the current `keep_latest` rule.
- Missing source status means the cleanup plan is stale against its linked evidence-index or freshness-audit path.
- `cleanup_review_allowed: false` means operators should regenerate source artifacts before using the candidate list. The dashboard renders this as `Cleanup Review: blocked`.
- If generated files are deleted manually, regenerate the evidence index and rerun the freshness audit.
- Source scripts, schemas, runbooks, phase summaries, and test fixtures are outside the cleanup planner's deletion scope.
