# Research Agent 2: Current Chelation Implementation Audit Report
**Loop 1 — CHELATION + OPSD Upgrade Program**

**Agent**: Research Agent 2 - Current Chelation Implementation Auditor  
**Date**: 2026-05-15  
**Repo Root**: /mnt/d/GITHUB/CHELATEDAI  
**Output Location**: `docs/chelation_opsd_research/loop_01/02_chelation_system_audit.md`

---

## Executive Summary

The CHELATEDAI "chelation" system is a sophisticated but **fundamentally incomplete** attempt at iterative, diagnostic-driven self-correction of embedding representations via lightweight residual adapters + variance/spectral masking + sedimentation training cycles.

**Core Architecture**:
- **ChelationAdapter family** (chelation_adapter.py): Multiple residual correction mechanisms (`x + δ(x)` + L2 normalize) with near-identity initialization.
- **Self-healing layer** (self_healing_chelation.py): SEAL/EGGROLL-inspired generator of `SelfEditDirective`s. Currently **advisory-only** ("base_model_mutation_allowed": False, "adapter_only": True).
- **Sedimentation pipeline** (sedimentation.py + *_loss.py + *_trainer.py): The actual mechanism that trains the adapter on "collapse" events recorded in `chelation_log`.
- **Engine integration** (antigravity_engine.py): Spectral masking during query + adapter application only in local-mode `embed()` + `run_sedimentation_cycle()` + `run_hierarchical_sedimentation()`.

**Ruthless Verdict**: 
The system **never actually closes the self-correction loop in a stable, sample-efficient, persistent way**. Self-healing produces plans but does not train. Sedimentation trains but is plagued by documented critical instabilities, data leakage, and forgetting risks. Persistent self-adaptation is explicitly **not claimed** ("safe persistent self-adaptation" remains future work per seal-eggroll-multipanel-architecture-2026-04-28.md).

This audit identifies **dozens of concrete failure modes** (many labeled CRITICAL/HIGH in the 2026-05-15 panel review) that map **directly** to the exact problems OPSD/SDPO techniques (KL control, on-policy dense supervision, filtered/asymmetric distillation, MIS-PO variants) were designed to solve.

**Evidence Sources** (liberally quoted below):
- Panel analysis: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/planning/2026-05-reconciliation/panel-analysis/02-data-engineering-ml.md` (F-ML-001 to F-ML-047)
- Code refinement panel: `.../01-code-refinement.md` (JO-, PS-, RT-, MC- findings)
- `self-adapting-chelation-seal-eggroll-analysis-2026-04-28.md`
- `seal-eggroll-multipanel-architecture-2026-04-28.md`
- Direct source in `chelation_adapter.py`, `self_healing_chelation.py`, `sedimentation.py`, `antigravity_engine.py`, `config.py`
- `task_plan.md`, `findings.md`, `test_self_healing_chelation.py`

---

## 1. Inventory of All Chelation-Related Components

**Core Implementation** (194 files reference "chelat|SelfEditDirective" patterns):
- `chelation_adapter.py` (520 LOC): Primary adapter hierarchy + factory.
- `self_healing_chelation.py` (722 LOC): SEAL-inspired directive planner, evaluator, sandbox, ledger.
- `chelation_logger.py` (486 LOC): Structured JSON logging (query, training, checkpoint events).
- `sedimentation.py` (262 LOC): `HierarchicalSedimentationEngine`.
- `sedimentation_loss.py` (165 LOC): InfoNCE, Hybrid, HardNegativeMiner.
- `sedimentation_trainer.py` (138 LOC): `compute_homeostatic_target`, `sync_vectors_to_qdrant`.
- `antigravity_engine.py` (2711 LOC): Main runtime — `embed()`, `_spectral_chelation_ranking()`, `run_sedimentation_cycle()`, `chelation_log`, adapter application.
- `online_updater.py`: Micro-gradient inference-time adapter updates.
- `stability_tracker.py`: Diagnostic tracking of masks, collapse, adapter drift.
- `checkpoint_manager.py`: `SafeTrainingContext` for rollback on training failure.
- `config.py`: `ChelationConfig` (ADAPTER_TYPE, BOUNDED_*, thresholds, etc.).

**Tests & Benchmarks**:
- `test_self_healing_chelation.py`, `test_chelation_logger.py`, `test_sedimentation_loss.py`, `test_sedimentation_trainer.py`, `test_attnres_adapter.py`, `test_stability_tracker.py`, `benchmark_distillation.py`, `benchmark_comparative.py`, many `test_run_*_campaign.py`.

**Documentation & Research**:
- `docs/chelation_opsd_research/CHELATION_OPSD_RESEARCH_PLAN.md`
- `docs/self-adapting-chelation-seal-eggroll-analysis-2026-04-28.md`
- `docs/seal-eggroll-multipanel-architecture-2026-04-28.md`
- Panel analyses (esp. 02-data-engineering-ml.md, 01-code-refinement.md)
- `task_plan.md`, `findings.md`, `TECHNICAL_ANALYSIS.md`, `REFACTORING_PLAN.md`.

**Other Integration**:
- `model_scope*` (steering, trainer, runtime, features): Use adapter indirectly via engine.
- `evolution_strategies_optimizer.py`: Used for EGGROLL-style ES directives.
- `quantization_promotion_gate.py`: Used by self-healing quant gate.
- `fitness_interfaces.py`, `RetrievalFitnessEvaluator`.

**Key Observation**: Self-healing chelation and the actual sedimentation training path are **loosely coupled**. Directives propose "adapter_sft", "eggroll_es", etc., but the planner only evaluates via fitness callables — it never actually instantiates trainers or runs gradient steps on accepted directives in the current code.

---

## 2. Deep Audit: chelation_adapter.py

**Architecture Quote** (lines 10-18):
```python
class ChelationAdapter(nn.Module):
    """
    A lightweight residual adapter that learns to 'chelate' (correct) embeddings.
    ...
    This ensures that at initialization (or with 0 weights), it acts as an identity function,
    preserving the original model's strong baseline.
    """
```

**Core Pattern** (all variants):
- `forward`: `delta = correction_net(x)`; `out = x + delta`; `out = F.normalize(out, p=2, dim=1)`
- Near-identity init: `nn.init.normal_(..., std=0.001)` + zeros bias (or V=0 in LoRA-style).

**Variants Implemented**:
1. **ChelationAdapter (MLP)**: Sequential Linear-ReLU-Linear. `regularization_loss()` returns 0.0.
2. **OrthogonalProcrustesAdapter** (lines 91-161): Cayley parameterization of orthogonal W + **DSM** (Diagonal Scaling Matrix `_scale`). Regularization: Frobenius on skew A.
   - Quote: "DSM improves recall recovery from 95-97% to 98-99%" (comment referencing Drift-Adapter arXiv:2509.23471).
3. **LowRankAffineAdapter** (lines 164-220): `delta = x @ U @ V^T + b` (asymmetric LoRA init: V=zeros). Reg: 0.0.
4. **BoundedAdapter** (lines 223-321): Wrapper enforcing `min_correction=0.01` (above INT8 noise ~0.0078) to `max_correction=0.5`. Scales delta and adds per-dim `dim_scale`. Reg includes scale penalty (hardcoded 0.001).
5. **BlockAttnResAdapter** (lines 324-415): Multi-block residuals + learned cross-block attention (MoonshotAI AttnRes 2025 inspiration).
6. **LayerAttentionAggregator**: For aggregating transformer layer embeddings.

**Factory** (`create_adapter`, lines 472-520): Supports "mlp", "procrustes", "low_rank", "attnres". Bounded wrapper optional. **Bugs noted in panel**:
- Silently discards kwargs (MC-008 in 01-code-refinement.md).
- `ADAPTER_TYPE = "mlp"` default in config; `BOUNDED_ADAPTER_ENABLED = False` but engine constructor **ignores the flag** (PS-016).

**save/load**: Per-class duplication, incomplete exception handling (only RuntimeError caught in most; TOCTOU race in exists+load).

**Regularization**: Inconsistent — only Procrustes and Bounded have meaningful terms. No unified interface for all.

**Integration with OPSD**: The residual "correction as small delta" is perfect substrate for **asymmetric distillation** (privileged diagnostic context as teacher) and **KL-regularized self-distillation** to keep corrections from destroying base knowledge.

---

## 3. Deep Audit: self_healing_chelation.py

**Core Claim** (module docstring, lines 1-7):
> "It is advisory by default: accepted directives describe what should be adapted, but they do not mutate the base embedding model."

**Data Structures**:
- `SelfEditDirective`: strategy, adaptation_mode (adapter_sft, online_contrastive_update, eggroll_es, quantization_gate, retention_replay, structural_health_penalty), synthetic_examples, optimization_params, source ("seal" or "eggroll").
- `SelfEditEvaluation`: fitness, reward, accepted, reasons, quantization_gate.
- `CandidateProvenanceLedger`: append-only hash-tracked history (context_hash, diagnostics_hash, directive_hash).
- `SelfGeneratedEvalProbe`: knowledge_recall + retention probes with negatives/confusers.

**Planner** (`SelfHealingChelationPlanner`):
- `generate_directives()`: Hardcoded heuristics based on diagnostics (retrieval anomaly → retrieval_ttt; quant failure → quant survival; structural_health < 0.7 → structural_repair; always includes seal_implication_sft + eggroll + retention_guard).
- `evaluate_directives()`: reward = fitness - baseline; gates:
  - `reward > reward_threshold`
  - `retention_score >= min_retention_score` (default 0.8)
  - QuantizationPromotionGate (retained_gain_threshold, minimum_fp32_gain)
  - min_structural_health, latency regression.
- `build_update_plan()` / `execute_shadow_round()` / `run_adaptive_validation_loop()`: Produce JSON-safe plans. `mode = "advisory"` unless `allow_persistent_update=True`.
- Sandbox executor: `isolated=True`, `base_model_mutation_allowed=False`.

**Test Evidence** (test_self_healing_chelation.py):
- Explicitly tests that only "good" (positive reward + retention + quant pass) is accepted; others rejected with specific reasons ("retention_below_threshold", "quantization_gate_failed", "reward_not_positive").
- Plan always has `"mode": "advisory"`, `"base_model_mutation_allowed": False`.

**Pain Point Quote** (seal-eggroll-multipanel...md):
> "Not claimed yet: ... safe persistent self-adaptation"

**Integration Gap**: No code path from accepted `SelfEditDirective` → actual `SedimentationInfoNCELoss` / Adam training of a cloned adapter. Directives are diagnostic proposals, not executable training jobs.

This is the **exact gap** OPSD on-policy self-distillation + filtered behavior cloning is designed to fill (ReSTEM-style filtering of self-generated trajectories).

---

## 4. Deep Audit: Sedimentation Pipeline (The "Real" Training Mechanism)

**Flow**:
1. Inference populates `chelation_log[doc_id].append(center_of_mass)` in dense clusters (`_spectral_chelation_ranking`).
2. `run_hierarchical_sedimentation(threshold, lr, epochs)` or legacy `run_sedimentation_cycle()`:
   - Filter targets where `len(v) >= threshold`.
   - Compute homeostatic targets (`compute_homeostatic_target`: push away from avg noise).
   - Cluster by variance partitioning.
   - Per-cluster Adam + MSE (hardcoded in hierarchical path!).
   - Global refinement (LR * 0.1, fresh optimizer).
   - `adapter(input_tensor)` → Qdrant upsert (with payload preservation).
   - `chelation_log.clear()`.

**Loss Problems** (from 02-data-engineering-ml.md — CRITICAL):
- **F-ML-001 [CRITICAL]**: `temperature=0.07` "dangerously small for typical batch sizes" (sedimentation uses 10-100 items). "causes mode collapse into the hardest pair in the batch. No batch-size-aware temperature scaling."
  - Quote: "gradients become extremely sparse... only the near-duplicate pairs receive gradient signal"
- **F-ML-004 [HIGH]**: Hybrid loss scale mismatch — InfoNCE dominates by 10-100x; "effectively pure InfoNCE".
- **F-ML-006 [HIGH]**: Hierarchical path **ignores** `set_sedimentation_loss()` and hardcodes `MSELoss()` (sedimentation.py:134-135). Inconsistency.
- **F-ML-007 [HIGH]**: "No gradient clipping anywhere in any training loop". Compounds InfoNCE small-temp issue → "weight explosion".

**Target & Sync Issues**:
- **F-ML-009 [MEDIUM]**: `compute_homeostatic_target` push not scale-invariant.
- **F-ML-016 [MEDIUM]**: Training only on thresholded targets, but `chelation_log.clear()` discards sub-threshold signal unconditionally.
- **F-ML-023 [CRITICAL]**: "Training/serving parity broken": training does `adapter(input).numpy()` (no explicit normalize after MSE), but inference relies on cosine (unit-norm assumption). "stored vectors may have varying norms".
- **F-ML-040 [HIGH]**: "No held-out evaluation set for sedimentation training" — "direct data leakage path: the adapter learns to minimize loss on exactly the vectors it will be evaluated on."
- **F-ML-041 [HIGH]**: Chelation log sole signal "with no de-duplication". "A single query repeated 100 times would make one document appear as the most 'collapsed'".

**Training Loop Issues** (from panels):
- No warmup (F-ML-010).
- Fresh Adam every cycle → momentum reset (F-ML-012).
- No seeds → non-reproducible (F-ML-020).
- No experiment tracking / artifact logging of loss curves or adapter states (F-ML-021 [CRITICAL]).
- Adapter checkpoint overwrite risk (F-ML-022 [CRITICAL]).

**Robustness** (01-code-refinement.md):
- Duplicate training loop code between sedimentation and offline_distillation (MC-007).
- Empty training_inputs returns without clearing log → potential infinite retry (JO-002).
- `SafeTrainingContext` rollback has exception-loss issues (JO-007, JO-008).

**OPSD Relevance**: Sedimentation is a crude, offline, sparse-reward (collapse count threshold) self-correction. OPSD provides **dense token-level on-policy supervision from own generations**, proper KL scheduling to avoid "KL shocks", and filtered sampling for efficiency — exactly the upgrades needed to replace or augment this brittle cycle.

---

## 5. Integration Points & Runtime Behavior

**AntigravityEngine**:
- Adapter created in `__init__` via `create_adapter(ChelationConfig.ADAPTER_TYPE, ...)`.
- Applied **only** in local mode `embed()` (line ~177): `adapted_embeddings = self.adapter(tensor_inputs)`.
- Ollama mode bypasses adapter entirely.
- `chelation_log` populated only during dense `_spectral_chelation_ranking` (center-of-mass of local cluster).
- Sedimentation called externally (campaign scripts).
- `get_chelated_vector` for external MTEB benchmarking uses masking, not adapter.

**Online Updater**: Exists for micro-updates (`perform_online_update`), stability tracking of adapter weights/grads. But opt-in (`ONLINE_UPDATE_ENABLED = False`).

**Model Scope Components**: Steering/trainer/runtime use engine or hooks; no direct self-healing wiring observed.

**Quantization Path**: BoundedAdapter + simulate_int8 + QuantizationPromotionGate. But many parity issues (F-ML-023).

**Safety / Promotion**: Heavy reliance on gates, but "persistent_update" is deliberately gated behind `allow_persistent_update=False`.

---

## 6. Catalog of Concrete Limitations, Failure Modes & Gaps (Evidence-Based)

### 6.1 Initialization & Near-Identity Pathologies
- Near-identity init broken in related projection code (F-ML-002, F-ML-011).
- Adapter itself starts good, but downstream components destroy the property.

### 6.2 Training Instability & Gradient Pathology (CRITICAL cluster)
- InfoNCE temperature too low → mode collapse (F-ML-001).
- No gradient clipping (F-ML-007).
- Loss scale mismatch in hybrid (F-ML-004).
- Scheduler inverted logic (F-ML-005).
- No LR warmup, cold Adam restarts (F-ML-010, F-ML-012).
- **Quote (panel)**: "a single pathological batch can cause weight explosion."

### 6.3 Forgetting, Retention & Catastrophic Failure
- Self-healing has retention gate but **no actual replay mechanism** implemented beyond proposal.
- Panel explicitly notes SEAL paper risk: "repeated edits can cause catastrophic forgetting."
- Sedimentation has no held-out retention benchmark during training.
- Advisory-only design is a symptom of the unsolved problem.

### 6.4 Sample Efficiency & Data Quality
- Threshold-based sparse signal (collapse count) vs dense on-policy.
- Duplication bias in chelation_log (F-ML-041).
- No dedup, no curriculum.
- Leakage: train on future eval vectors (F-ML-040).

### 6.5 Quantization Survival
- BoundedAdapter exists but training does not guarantee normalized outputs (F-ML-023).
- Many parity and device issues (F-ML-027, F-ML-034).

### 6.6 Reproducibility, Checkpointing & MLOps (CRITICAL)
- No seeds, no experiment tracking (F-ML-020, F-ML-021).
- Checkpoint overwrite + no integrity verification (F-ML-022, F-ML-032).
- Training/serving norm parity broken.
- Unbounded histories in StabilityTracker (PS-015).

### 6.7 Code & Architectural Debt
- Massive duplication in training loops.
- Inconsistent regularization interfaces.
- Factory silent failures.
- Self-healing completely decoupled from actual optimizer execution.
- `allow_persistent_update` is a dead flag with no safe implementation path.

### 6.8 Integration Gaps with Self-Healing
- Directives propose `eggroll_es`, `adapter_sft` but no executor wires them to `evolution_strategies_optimizer` or a cloned-adapter SFT loop inside the planner.
- Live-fire diagnostics use the planner but reward is retrieval fitness on probes — not a full inner training loop.

**From seal-eggroll-multipanel (exact quote)**:
> "Implemented now: advisory SEAL/EGGROLL planning... Not claimed yet: ... safe persistent self-adaptation"

---

## 7. Mapping Weaknesses to OPSD / SDPO / @ar0cket1 Techniques

| Current Chelation Weakness | OPSD / SDPO Counterpart | How It Solves It |
|----------------------------|--------------------------|------------------|
| Brittle MSE/InfoNCE on collapse counts; mode collapse from bad temp | On-policy self-distillation with dense token-level supervision from own generations | Replaces sparse threshold signal with rich per-token gradients from model rollouts |
| No KL control → forgetting during adapter updates | Explicit KL scheduling + "KL shock" mitigation (low-entropy/high-divergence region handling) | Prevents correction from destroying useful base behavior (exact "catastrophic forgetting" risk) |
| Advisory-only; no safe persistent reinforcement | Filtered on-policy sampling + ReSTEM-style positive filtering + MIS-PO / SDPO variants | Turns accepted SelfEditDirectives into actually reinforced, stable LoRA/adapter updates |
| Training on eval docs (leakage); poor sample efficiency | On-policy vs filtered sampling; privileged-context asymmetric distillation (diagnostics as teacher context) | Much higher sample efficiency; teacher (with full diagnostics) vs student (normal context) |
| Hardcoded losses, no grad clip, unstable schedulers | Gradient shape analysis + stabilized objectives in SDPO | Stable training signals for residual corrections |
| No experiment tracking of self-correction trajectories | Online RL + self-distillation loops (live LoRA from feedback) | Auditability + continual improvement without collapse |
| BoundedAdapter is post-hoc magnitude hack | Quantization-aware training in the distillation objective itself | Corrections survive quant by construction |
| Self-healing generates but never executes real inner loops | Model acting as its own teacher on privileged vs student contexts | Direct pattern for making directives "stick" via self-distilled reinforcement |

**Strongest Mapping**: The `SelfEditDirective` + sandbox + fitness gate is a **proto-ReSTEM / proto-SDPO** outer loop. The missing piece is the **inner on-policy distillation** that actually updates the adapter using the model's own (privileged diagnostic) generations as dense teacher signal, with KL regularization to the frozen base.

This is precisely what @ar0cket1's Hermes-Agent-Online-RL + SDPO/MIS-PO work demonstrates at scale for live self-improving adapters.

---

## 8. Recommendations for Subsequent Loops

**Immediate Priorities for Loop 2+**:
1. Implement a **cloned-adapter execution path** inside `SelfEditSandboxExecutor` / a new `SelfEditTrainer` that actually runs SFT / contrastive / ES updates on a copy of the current adapter for accepted directives.
2. Add **KL divergence regularizer** to all adapter training (Procrustes-friendly + general).
3. Replace/fix InfoNCE temperature and add batch-aware scaling + gradient clipping in sedimentation (as baseline).
4. Design **asymmetric privileged-context distillation** objective where diagnostics (structural health, quant gate, retrieval anomalies) are injected as privileged teacher context.
5. Build the `chelation_benchmark_suite` (Agent 6) with metrics: correction gain, retention delta over N iterations, sample efficiency (queries per NDCG lift), quant survival rate, "self-healing success rate" (accepted → persistent lift without regression).
6. Wire EGGROLL ES + SDPO-style filtering into the directive execution for Variant A/B/C implementers.

**Risks to Track**: Overfitting to synthetic probes; KL weight scheduling sensitivity; quantization during online updates.

This audit provides the **complete evidence base** for the 10-agent program. The current chelation system has excellent scaffolding and safety culture, but the **practical self-correction engine is missing** — and OPSD research supplies exactly the missing production-grade techniques.

---

**End of Agent 2 Loop 1 Report**

Next agents should reference this document heavily when designing losses, training regimes, and variants.

All quotes and findings are directly attributable to the cited files in the 2026-05-15 CHELATEDAI repository state.