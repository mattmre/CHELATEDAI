# Model-Scope Overlay Bundle Schema - 2026-05-05

Purpose: document the generated artifacts from a Model-Scope campaign that receives adaptive overlay evidence. This is an operator contract, not a default-promotion claim.

## Generation Commands

Focused deterministic smoke:

```bash
python run_model_scope_overlay_smoke.py --output-dir experiment_runs/model-scope-overlay-smoke/latest
```

Focused validation bundle:

```bash
python run_overlay_model_scope_validation.py --output-dir experiment_runs/overlay-model-scope-validation/latest --timeout-seconds 300
```

Promotion-linkage audit:

```bash
python audit_promotion_linkage.py --root experiment_runs --output experiment_runs/promotion-linkage-audit/latest/audit.json
```

## Campaign Bundle

The canonical campaign report is `campaign_report.json`. Its `outputs` object links the replayable sidecar artifacts.

| artifact | producer | purpose | promotion role |
| --- | --- | --- | --- |
| `memory_snapshot.json` | `ModelScopeSegmentedMemory.save` | Captures segmented Model-Scope memory state used by the campaign. | Evidence only |
| `evidence_bundle.json` | `evidence_contract.write_evidence_bundle` | Replayable event bundle across Model-Scope and evaluator surfaces. | Required input to promotion contract |
| `replay_bundle.json` | `model_scope_trainer` path | Replay entries used for candidate shadow-policy training. | Required input to replay scoring |
| `comparison_report.json` | `run_model_scope_campaign.py` | Per-query/profile comparison records. | Required replay evidence |
| `shadow_policy_candidate.json` | `ModelScopeShadowPolicyTrainer` | Candidate learned shadow policy. | Candidate artifact, not default runtime |
| `evaluator_summary.json` | `evaluator_fabric` | Evaluator vote and agreement summary. | Promotion gate input |
| `trace_grade.json` | `evidence_contract` trace grader | Checks required evidence surfaces are present. | Promotion gate evidence |
| `compute_budget_summary.json` | `compute_budget_policy` | Budget action counts and escalation summary. | Diagnostic evidence |
| `reward_overoptimization_report.json` | reward divergence check | Compares training and heldout score behavior. | Promotion blocker if divergent |
| `feature_scorecard.json` | `model_scope_features` | Feature support, polarity, and intervention-risk summary. | Advisory only |
| `holdout_report.json` | campaign input or default fail-closed report | Holdout gate result. | Required promotion gate input |
| `safety_report.json` | campaign input or default fail-closed report | Safety gate result. | Required promotion gate input |
| `promotion_decision.json` | `promotion_contract.evaluate_promotion_candidate` | Single fail-closed promotion decision. | Authoritative promotion state |

## Overlay Sidecars

These artifacts are emitted only when adaptive overlay evidence is supplied.

| artifact | required fields | purpose | promotion role |
| --- | --- | --- | --- |
| `adaptive_overlay_report.json` | `schema_version`, `record_type`, `readiness`, `branch_set_metrics` | Replay overlay readiness and blocker evidence. | Required if overlay readiness is configured |
| `adaptive_overlay_holdout_report.json` | same as replay report | Holdout overlay readiness evidence. | Required for validation-ready overlay bundle |
| `adaptive_overlay_validation_report.json` | `validation_ready`, `blockers`, replay and holdout readiness | Combines replay and holdout overlay status. | Evidence embedded in artifact card |
| `adaptive_overlay_collection_policy.json` | `decision`, `advisory_only`, budget features | Recommends next collection width. | Advisory only, never a runtime controller |
| `verifier_evidence_cards.json` | `record_type`, `advisory_only`, `result`, `rubric` | Stores verifier/rubric outputs as review cards. | Advisory evidence only |
| `adaptive_overlay_artifact_card.json` | `card_id`, `candidate_id`, `readiness`, `evidence`, `rollback_path`, `limitations` | Compact review card linking the candidate, evidence, limitations, and rollback target. | Required linkage for overlay-backed promotion decisions |

## Promotion Decision Linkage

Overlay-backed or promotion-ready campaign reports must carry:

- `promotion_decision.artifact_card_reference.path` or `promotion_decision.artifact_card_reference.card_id`
- `promotion_decision.rollback_path`

`audit_promotion_linkage.py` scans campaign reports and blocks reports that require linkage but lack either field.

Promotion remains fail-closed. A clean artifact card, validation report, or dashboard row is not a default change unless `promotion_decision.promotion_ready` is true and the governance review accepts the decision.

## Validation Summary

`run_overlay_model_scope_validation.py` writes `validation_summary.json` with:

| field | meaning |
| --- | --- |
| `record_type` | Always `overlay_model_scope_validation_bundle` |
| `passed` | True only if every bundled command exits zero |
| `command_count` | Number of commands executed |
| `failed_commands` | Names of failed or timed-out commands |
| `results[]` | Per-command return code, duration, and stdout/stderr tails |

The dashboard reads this summary through `/api/validation_history`.

## Operator Notes

- Treat `experiment_runs` as generated output unless a run artifact is explicitly promoted into docs.
- Use the validation bundle before claiming an overlay/model-scope campaign is review-ready.
- Use the promotion-linkage audit before reviewing any campaign report for default promotion.
- Do not infer default promotion from a smoke campaign. The smoke bundle proves wiring, not empirical lift.
