# Frontier Adaptive Overlay Research Refresh

Date: 2026-05-05

Purpose: Re-check current frontier work before continuing implementation after the HeavySkill/adaptive-overlay sequence. This document records accepted and rejected ideas for ChelatedAI's engine and model overlay queue.

## Sources Reviewed

| Source | Date | Main idea | ChelatedAI interpretation |
| --- | --- | --- | --- |
| ODAR: Principled Adaptive Routing for LLM Reasoning via Active Inference, `https://arxiv.org/abs/2602.23681` | 2026-02-27 | Route between fast and slow reasoning agents with difficulty estimation and free-energy/risk-sensitive answer fusion. | Strong fit for overlay collection policy: estimate difficulty and allocate branch search only where uncertainty or historical blockers justify it. Do not copy the agent harness. |
| Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained Policy Optimization, `https://arxiv.org/abs/2604.14853` | 2026-04-16 | Solve budgeted compute allocation as constrained optimization, then train a lightweight classifier to imitate oracle allocation. | Strong fit for future overlay budget policy: learn when broader replay or extra branches are worth cost under an average budget. |
| TARo: Token-level Adaptive Routing for LLM Test-time Alignment, `https://arxiv.org/abs/2603.18411` | 2026-03-19 | Use step-wise reward signals and a token-level router to steer a frozen LLM at inference time. | Conceptual fit only. ChelatedAI should not implement token-level steering now; use it as evidence that frozen-model overlays are viable when verifier signals are fine-grained. |
| TIDE: Trajectory-based Diagnostic Evaluation of Test-Time Improvement in LLM Agents, `https://huggingface.co/papers/2602.02196` | 2026-02 | Evaluate test-time improvement through trajectory dynamics, looping, and memory burden. | Strong fit for diagnostics: add overlay trajectory health metrics before adding more branching. |
| DeepVerifier: Inference-Time Scaling of Verification, `https://arxiv.org/abs/2601.15808` | 2026-01-22, v2 2026-04-29 | Use rubric-guided verification as a plug-in test-time module that feeds back refinement. | Fit for verifier-card design and failure taxonomies. Keep as optional evaluator evidence; no default self-refinement loop. |
| BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute, `https://www.microsoft.com/en-us/research/publication/best-route-adaptive-llm-routing-with-test-time-optimal-compute/` | 2025-06 | Route queries across models and consider multiple cheap samples before escalating to expensive models. | Good analogy for branch-set budget control: compare cheap overlay variants before broader or slower validation. |

## Accepted Ideas

1. Add budget-aware overlay validation before adding more branches.
2. Track trajectory health, not just branch-set lift.
3. Treat verifier/rubric output as evidence cards, not as a default runtime controller.
4. Add explicit overlay artifact cards so promoted or candidate overlays carry provenance, evals, safety checks, and rollback.
5. Add a later overlay collection policy that combines difficulty, uncertainty, coverage novelty, and blocker history.

## Rejected Or Deferred Ideas

1. Full agent harness implementation: deferred. HeavySkill and ODAR-like harnesses remain inspiration for evidence architecture, not product architecture.
2. Token-level routing over base-model logits: deferred. TARo-style steering requires reward-model and decoding integration that is outside the current engine-safe scope.
3. Automatic self-refinement loops: deferred. DeepVerifier-style feedback should first appear as evaluator evidence and blocker taxonomy.
4. Default route promotion from adaptive overlay readiness alone: rejected. Readiness only authorizes broader validation.

## Queue Impact

The highest-value next implementation slice is an overlay artifact card and validation bundle, not more branching. The card should summarize:

- purpose and candidate id
- source overlay report hash or path
- readiness status and blockers
- replay, holdout, hard-negative, evaluator, and safety evidence
- limitations and known failure modes
- rollback path
- promotion decision linkage

Artifact cards landed as the next reporting slice: `build_overlay_artifact_card` now creates compact candidate review objects from overlay readiness, branch-set metrics, replay, holdout, hard-negative, evaluator, safety, promotion, limitations, and rollback evidence. Model-Scope campaign runs that receive an adaptive overlay report now write `adaptive_overlay_artifact_card.json` next to the full overlay report.

Broader replay/holdout validation landed as the next validation slice: `build_overlay_validation_report` now requires replay readiness plus holdout readiness and fails closed on missing or regressing holdout evidence. Model-Scope campaigns can accept `--adaptive-overlay-holdout-report`, write `adaptive_overlay_validation_report.json`, and embed validation evidence into the artifact card.

Overlay trajectory diagnostics landed as an observational report field on overlay reports and artifact cards:

- loop or repeated-branch count
- memory/report burden
- oracle gap trend
- blocker recurrence by channel
- budget used per safe pass

Budget-aware overlay collection policy landed as advisory-only evidence. It uses readiness blockers, trajectory warnings, uncertainty, coverage novelty, and blocker history to choose observe-only, standard collection, or broadened collection without changing runtime routing.

The next implementation target should be verifier/rubric evidence card integration.

## Updated Priority Recommendation

1. Overlay artifact cards.
2. Broader adaptive-overlay replay and holdout validation.
3. Overlay trajectory health diagnostics.
4. Budget-aware overlay collection policy.
5. Verifier/rubric evidence card integration.
