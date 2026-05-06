# Evidence Artifact Retention Policy

Date: 2026-05-06

Purpose: define which generated evidence artifacts should be retained, regenerated, or kept out of git during default-promotion and overlay validation work.

## Policy

Generated evidence under `experiment_runs/` is runtime output, not source. Keep it locally or upload it as a CI artifact unless a small, curated fixture is explicitly needed for tests or documentation.

## Retain Locally

Keep the latest local copies of:

| Artifact | Why retain |
| --- | --- |
| `validation_summary.json` | focused validation pass/fail and command tails |
| `promotion-linkage-audit.json` | linkage blockers for campaign reports |
| `attnres-repeat-seed-decision.json` | no-default-change or candidate-repeat decision |
| `default-promotion-preflight.json` | fail-closed review-readiness decision |
| `evidence_chain_summary.json` | linked summary across the promotion evidence chain |
| `evidence_index.json` | compact navigation index over generated evidence |
| `freshness_audit.json` | stale or dangling evidence-index link audit |

These files are operational state. They should be regenerated when the underlying reports change.

## CI Artifacts

For GitHub Actions, upload evidence bundles instead of committing them:

- default-promotion evidence chain output
- overlay/model-scope validation output
- freshness audit output when added to a workflow

CI artifacts are the right retention layer for run-specific proof because they preserve exact run output without bloating the repo.

## Commit Only Source Contracts

Commit:

- runner scripts
- audit scripts
- schemas and runbooks
- phase summaries
- small unit-test fixtures when they are hand-authored and deterministic

Do not commit:

- full `experiment_runs/` trees
- generated campaign outputs
- generated dashboard history JSON
- large raw traces
- provider/model outputs not needed by tests

## Regeneration Commands

Refresh the default-promotion evidence set with:

```bash
python run_default_promotion_evidence_chain.py --output-dir experiment_runs/default-promotion-evidence-chain/latest --timeout-seconds 300
python generate_evidence_index.py --root experiment_runs --output experiment_runs/evidence-index/latest/evidence_index.json
python audit_evidence_index_freshness.py --index experiment_runs/evidence-index/latest/evidence_index.json --output experiment_runs/evidence-index/latest/freshness_audit.json
```

The expected current result is still fail-closed: evidence collection can pass while default-promotion review remains blocked by repeat-seed evidence.

## Safe Deletion

Before deleting generated evidence:

1. Confirm it is under `experiment_runs/` or another explicit generated-output directory.
2. Confirm no source doc links to it as an irreplaceable artifact.
3. Keep at least one recent local or CI artifact copy for the latest merged evidence chain.
4. Regenerate the evidence index after deletion.
5. Run the freshness audit; it must pass or explicitly name the deleted stale links.

Never delete source scripts, schemas, runbooks, phase summaries, or test fixtures as part of generated-output cleanup.
