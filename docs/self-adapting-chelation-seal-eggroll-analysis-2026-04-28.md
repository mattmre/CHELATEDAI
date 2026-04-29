# Self-Adapting Chelation: SEAL + EGGROLL Adaptation

## Sources reviewed

- SEAL: Self-Adapting Language Models, arXiv `2506.10943`
- EGGROLL / Evolution Strategies at the Hyperscale, reviewed in [Evolution Strategies Hyperscale Comparison](evolution-strategies-hyperscale-chelatedai-analysis.md)

## Core synthesis

SEAL supplies the missing control language for self-healing chelation. EGGROLL supplies the low-rank, scalar-fitness optimization substrate. Together they map cleanly to ChelatedAI:

| Paper concept | ChelatedAI adaptation | Implementation surface |
|---|---|---|
| SEAL self-edit | A structured directive describing synthetic examples, update mode, hyperparameters, and safety gates | `SelfEditDirective` in `self_healing_chelation.py` |
| SEAL inner update | Adapter-only SFT, online contrastive update, low-rank ES, or retention replay | `adaptation_mode` values such as `adapter_sft`, `online_contrastive_update`, `eggroll_es` |
| SEAL reward | Retrieval/structural/retention/quantization fitness after applying a candidate | `SelfHealingChelationPlanner.evaluate_directives()` |
| ReSTEM / filtered behavior cloning | Accept only positive-reward edits; reject zero/negative, forgetful, or quantization-collapsed candidates | `accepted` and `reasons` in `SelfEditEvaluation` |
| Knowledge incorporation | Generate implications and retrieval QA examples from new context | deterministic synthetic examples in `generate_directives()` |
| Few-shot TTT | Create query-local contrastive updates when runtime diagnostics show retrieval anomalies | `seal_retrieval_ttt` directive |
| Catastrophic forgetting limitation | Require retention score before accepting persistent/self-healing edits | `min_retention_score` gate |
| EGGROLL low-rank ES | Adapter-only population search using scalar fitness, rank-1 perturbations, quantization-aware scoring | `eggroll_low_rank_self_edit` directive |
| EGGROLL int8/quantized training | Reject candidates whose FP32 improvement disappears under simulated quantization | `QuantizationPromotionGate` integration |

## SEAL findings relevant to ChelatedAI

- The paper frames LLMs as static systems that need a way to transform new context into their own update data and update directives.
- A self-edit can restructure source information, generate synthetic training examples, specify optimization settings, or invoke tools for data augmentation and gradient updates.
- The training loop is nested: an outer reinforcement loop samples self-edits and scores them by downstream performance after an inner update loop.
- The paper's ReSTEM-style training reinforces only useful edits, which maps to ChelatedAI's existing promotion-gate culture.
- Knowledge incorporation improved no-context SQuAD accuracy from roughly the low-30% range to `47.0%` after RL-trained self-edits, with self-generated data outperforming GPT-4.1 synthetic data in the single-passage setting reported by the paper.
- Few-shot ARC-style adaptation improved self-edit success substantially compared with no-adaptation and non-RL self-edit baselines.
- The paper explicitly reports two risks that matter here: self-edit reward loops are computationally expensive, and repeated edits can cause catastrophic forgetting.

## Engineering stance

The repository should not let a model freely rewrite base weights during retrieval. The safe ChelatedAI adaptation is:

1. Generate candidate self-edits from context and diagnostics.
2. Convert them into adapter-only update directives.
3. Score each directive with retrieval fitness and structural health.
4. Reject any directive that harms retention or fails quantization survival.
5. Treat accepted directives as advisory unless an explicitly tested persistent-update mode is enabled.

This preserves the SEAL idea of self-directed adaptation while respecting the safety-testbed rule that no control should silently change runtime behavior without evidence.

## What was added

- `self_healing_chelation.py`: deterministic SEAL/EGGROLL-inspired self-edit planner.
- Candidate provenance ledger, self-generated eval probes, sandbox execution seam, and adaptive validation loops for repeated debug/replay rounds.
- `test_self_healing_chelation.py`: unit coverage for directive generation, ReSTEM-style filtering, retention rejection, quantization rejection, and JSON-safe update plans.
- `run_live_fire_diagnostics.py`: now includes a self-healing update-plan section so live-fire output proves the self-edit layer is operable and logged.
- `pyproject.toml`: package module registration.
- Documentation/index/research-track updates so the self-adapting chelation track is visible alongside EGGROLL, safety, and road-course work.

## Practical workflow

```python
from self_healing_chelation import SelfHealingChelationConfig, build_self_healing_update_plan

plan = build_self_healing_update_plan(
    context=["new evidence", "observed retrieval collapse"],
    diagnostics={"retrieval_policy": {"high_variance_fast_path": True}},
    fitness=lambda directive: score_directive(directive),
    config=SelfHealingChelationConfig(baseline_fitness=0.5),
)
```

The returned plan is JSON-safe and includes accepted/rejected directives, reasons, best fitness, and safety metadata. By default, `mode` is `advisory`.

## Live-fire output surface

`python run_live_fire_diagnostics.py` now includes:

```text
live_fire.self_healing_update_plan
```

That plan records validation rounds, accepted and rejected self-edits, reward deltas, quantization-gate details where available, self-generated eval probes, candidate-ledger hashes, and retention/adapter-only safety metadata. The live-fire reward now comes from `RetrievalFitnessEvaluator.evaluate_engine()` over generated probes instead of fixed synthetic reward deltas.

For the deeper expert-panel synthesis, see [SEAL + EGGROLL Multi-Panel Architecture Review](seal-eggroll-multipanel-architecture-2026-04-28.md).

## Research implications

SEAL changes the ChelatedAI roadmap from "detect collapse and apply fixed repair rules" to "detect collapse, synthesize candidate repairs, and reinforce only repairs that improve downstream behavior." EGGROLL keeps those repairs efficient by focusing on low-rank adapter perturbations and scalar-fitness evaluation. The combined path is the repo's clearest implementation route for self-healing chelation.

## Boundaries

- No base embedding model mutation is enabled.
- Accepted directives do not prove broad chelation lift; they require road-course validation before promotion.
- The deterministic synthetic examples are a safe local stand-in for generated SEAL data, not a claim that the repo now trains a full self-editing LLM policy.
- Quantization and retention gates are required because both papers expose failure modes: EGGROLL emphasizes low-precision viability, while SEAL reports catastrophic forgetting risk.
