# Default-Promotion Evidence Runbook

Date: 2026-05-06

Purpose: explain how to collect and read the evidence required before starting any default-promotion review. This runbook does not authorize a default change. It describes the fail-closed evidence path.

## One-Command Evidence Chain

Run:

```bash
python run_default_promotion_evidence_chain.py --output-dir experiment_runs/default-promotion-evidence-chain/latest --timeout-seconds 300
```

The command writes:

| Artifact | Meaning |
| --- | --- |
| `overlay-model-scope-validation/validation_summary.json` | focused overlay/model-scope regression and smoke validation |
| `promotion-linkage-audit.json` | scan for campaign reports missing artifact-card or rollback linkage |
| `attnres-repeat-seed-decision.json` | repeat-seed AttnRes decision summary |
| `default-promotion-preflight.json` | fail-closed review-readiness decision |
| `evidence_chain_summary.json` | linked summary of all evidence artifacts |

The chain can exit `0` while `review_allowed` is `false`. That is expected when the evidence was collected successfully but does not support a promotion review.

Use `--fail-on-blocked-review` only when a caller wants blocked review status to become a nonzero exit.

## Cross-Artifact Index

Run:

```bash
python generate_evidence_index.py --root experiment_runs --output experiment_runs/evidence-index/latest/evidence_index.json
```

The index is a compact navigation artifact. It links the latest validation summaries, promotion-linkage audits, repeat-seed decisions, preflights, evidence-chain summaries, campaign reports, and adaptive overlay artifact cards. It is not a promotion decision.

## Manual CI

Use the `Default Promotion Evidence` workflow when the evidence chain should run on GitHub Actions. The workflow uploads `experiment_runs/default-promotion-evidence-chain/ci/`, `experiment_runs/evidence-index/ci/`, and `experiment_runs/evidence-cleanup/ci/` as artifacts.
The workflow defaults `cleanup-guard-mode` to `fail`, which runs cleanup planning with `--fail-on-blocked-review` after generating the CI evidence index and freshness audit, so a missing linked source artifact fails the workflow before anyone relies on cleanup candidates.
Set `cleanup-guard-mode` to `warn` only when intentionally collecting nonblocking diagnostics; cleanup remains dry-run only and the workflow emits a cleanup-review allowed/blocked summary without failing.
If cleanup review is blocked, artifact upload still runs before the workflow reports failure so operators can inspect the available chain, index, freshness, and cleanup outputs.
The failure step also writes a compact cleanup-review diagnostic to the GitHub step summary, including missing source artifacts and cleanup candidate counts.
If the diagnostic says `Cleanup plan missing`, the cleanup planner did not leave the expected artifact for the diagnostic step. Inspect the earlier cleanup-planning step and uploaded artifacts, rerun the manual evidence workflow after fixing the generation failure, and do not use cleanup candidates from that run.
If the diagnostic says `Cleanup plan unreadable`, treat the cleanup plan as unusable workflow output: inspect the uploaded cleanup artifact, regenerate the evidence chain, evidence index, freshness audit, and cleanup plan, and do not use any cleanup candidates from that run.
The guard mode contract is documented in `docs/evidence-cleanup-plan-schema-2026-05-06.md`.

The workflow is intentionally manual. Promotion evidence should not become background noise on every commit.

For local diagnostic rendering, use:

```bash
chelatedai-cleanup-review-diagnostic --plan experiment_runs/evidence-cleanup/latest/cleanup_plan.json --mode warn
```

From a source checkout before installation, use `python -m cleanup_review_diagnostic` with the same arguments.
The optional `--github-summary` flag appends the same diagnostic to `GITHUB_STEP_SUMMARY` inside GitHub Actions. In local shells where `GITHUB_STEP_SUMMARY` is unset, the command still prints the diagnostic and exits cleanly.
The diagnostic field mapping is documented in `docs/evidence-cleanup-plan-schema-2026-05-06.md`.

## Preflight Fields

| Field | Meaning |
| --- | --- |
| `review_allowed` | all required evidence says a promotion review may start |
| `default_change_allowed` | always false in current tooling; actual default changes remain out of scope |
| `blockers` | reasons review must not start |
| `artifacts` | validation, audit, and repeat-decision inputs used by preflight |

## Current Blocker Meanings

| Blocker | Meaning | Operator action |
| --- | --- | --- |
| `validation_summary_missing` | validation artifact was not found | run the evidence chain or validation bundle |
| `validation_summary_unreadable` | validation artifact could not be parsed | regenerate the validation artifact |
| `validation_bundle_failed` | focused validation or smoke command failed | inspect failed command tails in `validation_summary.json` |
| `promotion_linkage_audit_missing` | linkage audit artifact was not found | run the evidence chain or linkage audit |
| `promotion_linkage_audit_failed` | a campaign report lacks required card or rollback linkage | inspect `blocked_reports` in the audit output |
| `repeat_seed_decision_missing` | repeat-seed decision artifact was not found | run the evidence chain or AttnRes decision command |
| `repeat_seed_evidence_does_not_support_default_promotion` | repeat-seed evidence does not justify a default-promotion review | keep defaults unchanged and continue evidence collection |

## No-Default-Change Path

The expected current state is:

1. validation bundle passes
2. promotion-linkage audit passes
3. repeat-seed decision says `no_default_change`
4. preflight reports `review_allowed: false`
5. evidence-chain summary reports `chain_passed: true`

That combination is not a failed implementation. It means the system can collect and link evidence, and the evidence correctly blocks promotion review.

## Promotion Boundary

Do not change production defaults unless a future evidence chain shows all of the following:

- validation and smoke evidence pass
- artifact-card and rollback linkage pass
- repeat-seed evidence is positive across required tasks
- quantization gates pass where applicable
- no active-negative or hard-negative blockers remain
- the promotion contract still reports fail-closed readiness for the candidate

Until then, default behavior remains unchanged.
