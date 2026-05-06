# Evidence Dashboard Runbook

Purpose: explain how to read the dashboard evidence panels and when to regenerate the generated evidence artifacts that feed them. This runbook is operational only; it does not authorize a default change.

## Dashboard Surface

Start the local dashboard with:

```bash
python dashboard_server.py --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/dashboard/` and use the Campaign History tab for the default-promotion evidence panels.

| Panel | API | Source artifact | Operator reading |
| --- | --- | --- | --- |
| Campaign reports | `/api/campaign_history` | `experiment_runs/**/campaign_report.json` | historical model-scope campaign decisions and linked overlay evidence |
| Validation bundle | `/api/validation_history` | `experiment_runs/validation-bundles/**/validation_summary.json` | latest overlay/model-scope validation pass state and failed commands |
| Promotion preflight | `/api/preflight_history` | `experiment_runs/default-promotion-preflight/**/default-promotion-preflight.json` | fail-closed blocker status for default-promotion review |
| Evidence index | `/api/evidence_index` | `experiment_runs/evidence-index/latest/evidence_index.json` | compact artifact counts and latest linked chain/preflight states |
| Evidence chains | `/api/evidence_chain_history` | `experiment_runs/default-promotion-evidence-chain/**/evidence_chain_summary.json` | whether the evidence collection chain ran cleanly and whether review is allowed |
| Evidence cleanup | `/api/evidence_cleanup_plan` | generated dry-run plan from `plan_evidence_artifact_cleanup.py` | cleanup candidate counts, retained count, candidate paths, and linked evidence-index/freshness-audit paths |

The dashboard is read-only. If a panel is empty, malformed, or stale, regenerate the source artifacts instead of editing dashboard output.

For the Evidence cleanup panel, `present` source status means the cleanup plan can still see the evidence-index or freshness-audit artifact it was linked against. `missing` means the linked source path is absent and the operator should regenerate the evidence index, rerun the freshness audit, and refresh the cleanup plan before making deletion decisions.
`Cleanup Review` is fail-closed: `blocked` means at least one linked source artifact is missing and cleanup candidates should not be used for deletion decisions.

## Regeneration Triggers

Regenerate the evidence set when any of these are true:

| Trigger | Action |
| --- | --- |
| New validation, campaign, preflight, or evidence-chain artifacts were created | regenerate the evidence index and freshness audit |
| `/api/evidence_index` shows missing or stale linked paths | regenerate the index, then rerun the freshness audit |
| The dashboard shows no latest evidence-chain report after a campaign or validation run | run the evidence chain into `experiment_runs/default-promotion-evidence-chain/latest` |
| CI collected evidence through the manual workflow | download the uploaded artifact bundle or rerun locally before making a promotion-readiness call |
| Generated evidence directories were removed or archived | regenerate the evidence index and freshness audit |

## Local Refresh

Use this full refresh when the operator needs current local dashboard evidence:

```bash
python run_default_promotion_evidence_chain.py --output-dir experiment_runs/default-promotion-evidence-chain/latest --timeout-seconds 300
python generate_evidence_index.py --root experiment_runs --output experiment_runs/evidence-index/latest/evidence_index.json
python audit_evidence_index_freshness.py --index experiment_runs/evidence-index/latest/evidence_index.json --output experiment_runs/evidence-index/latest/freshness_audit.json
```

Expected current posture: the chain can pass while `review_allowed` remains `false`. That means evidence collection is healthy and the fail-closed promotion gate is doing its job.

## Manual Workflow

Use the GitHub Actions `Default Promotion Evidence` workflow when evidence should be collected in CI. The workflow now performs this sequence:

1. collect the default-promotion evidence chain;
2. generate the cross-artifact evidence index;
3. run the evidence-index freshness audit;
4. write a dry-run evidence cleanup plan linked to the index and freshness audit;
5. fail the workflow if that cleanup plan is blocked by missing linked source artifacts.

It uploads `experiment_runs/default-promotion-evidence-chain/ci/`, `experiment_runs/evidence-index/ci/`, and `experiment_runs/evidence-cleanup/ci/`.
The cleanup step remains dry-run only and does not delete files.

## Interpretation Rules

- Treat `chain_passed: false` as an execution problem to debug before interpreting promotion readiness.
- Treat `chain_passed: true` with `review_allowed: false` as a valid fail-closed result when blockers are present.
- Treat a stale or dangling evidence index as a navigation problem, not as evidence that the underlying validation passed or failed.
- Treat missing cleanup source status as a stale planning problem. Regenerate the linked index and freshness audit before using cleanup candidates.
- Do not commit generated evidence artifacts unless they are deliberately curated as small fixtures or source contracts.
- Do not change defaults from dashboard state alone; the promotion contract and preflight gate remain authoritative.
