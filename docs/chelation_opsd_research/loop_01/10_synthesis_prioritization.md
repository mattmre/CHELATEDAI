# Research Agent 10: Synthesis & Prioritization Report (Loop 1)
**CHELATION + ON-POLICY SELF-DISTILLATION (OPSD) Upgrade Program**

**Agent**: Research Agent 10 — Synthesis & Prioritization Agent  
**Loop**: 1 (Deep Research & Mapping)  
**Date**: 2026-05-15  
**Repo Root**: /mnt/d/GITHUB/CHELATEDAI  
**Output Location**: `docs/chelation_opsd_research/loop_01/10_synthesis_prioritization.md`  
**Synthesized Inputs** (from parallel 10-agent swarm):
- Agent 1: `01_literature_deep_dive.md` (OPSD arXiv:2601.18734, SDPO arXiv:2601.20802, MIS-PO arXiv:2602.10604, @ar0cket1/Hermes-Agent-Online-RL practical LoRA online RL)
- Agent 2: `02_chelation_system_audit.md` (ruthless code + panel audit of chelation_adapter.py, self_healing_chelation.py, sedimentation*, antigravity_engine.py, 20+ CRITICAL/HIGH failure modes)
- Agent 4: `04_on_policy_dynamics.md` (current regime is mostly off-policy sparse sedimentation + advisory self-edits; only micro on-policy in OnlineUpdater)
- Agent 7: `07_sample_efficiency_filtering.md` (extreme sample inefficiency from unfiltered chelation_log, thresholded targets, no divergence/entropy selection)
- Agent 11: `11_evaluation_framework_skeleton.md` (core metrics: correction_gain, retention@K, stability_over_iterations, sample_efficiency, quant_robustness, self_healing_success_rate)
- Agent 12: `12_pain_point_to_opsd_mapping.md` (11 pain points mapped to OPSD techniques with priority tiers)
- Agent 13: `13_candidate_upgrade_patterns.md` (Tier S/A/B patterns seeded from literature + audit)
- Additional: CHELATION_OPSD_RESEARCH_PLAN.md, task_plan.md, findings.md, panel reviews (esp. 02-data-engineering-ml.md F-ML-*), chelation_adapter.py (520 LOC, 6 adapter variants + Bounded), self_healing_chelation.py (722 LOC, SEAL/EGGROLL directives), sedimentation_loss/trainer, etc.
- External research: Direct arXiv PDF fetches, GitHub repo clones/analyses, X post on KL shocks/low-entropy/high-divergence regions.

**Program Goal Recap**: Transform the elegant but currently advisory/incomplete "chelation" (residual correction adapters + diagnostic-driven self-edit planning) into a **practically viable, stable, sample-efficient, persistent, continual self-correction system** by integrating 2026 state-of-the-art On-Policy Self-Distillation (OPSD/SDPO/MIS-PO) techniques. No external larger teacher required; model teaches its "weaker self" using privileged diagnostics + ground-truth.

---

## 1. Executive Synthesis: Coherent Picture of the Opportunity

**Current CHELATEDAI Chelation System (Strengths + Fatal Gaps)**

**Strengths (what works conceptually and partially in practice)**:
- **Residual near-identity adapters** (chelation_adapter.py): `x + δ(x)` + L2 normalize. Variants:
  - ChelationAdapter (MLP residual, tiny-init std=0.001)
  - OrthogonalProcrustesAdapter (Cayley skew + DSM diagonal scaling for norm preservation; reg = ||A||_F^2)
  - LowRankAffineAdapter (LoRA-style U/V asymmetric init V=0)
  - BlockAttnResAdapter (multi-block residuals + learned cross-block attention, Moonshot AttnRes 2025)
  - BoundedAdapter wrapper (min_correction=0.01 > INT8 noise ~0.0078, max=0.5 + per-dim scale)
  - LayerAttentionAggregator (cross-layer attention for transformer hidden states)
- Factory `create_adapter()` supports "mlp|procrustes|low_rank|attnres" + bounded flag.
- **Self-healing planner** (self_healing_chelation.py): Generates 4-6 `SelfEditDirective`s (seal_implication_sft, retrieval_ttt, eggroll_low_rank_es, quantization_survival, structural_repair, retention_replay_guard) from context + diagnostics. Evaluates via `FitnessEvaluation`, retention score (>=0.8), QuantizationPromotionGate, self-generated probes (SEAL-style). Builds `CandidateProvenanceLedger`, advisory or persistent plan.
- **Sedimentation** (sedimentation.py + loss.py + trainer.py + antigravity_engine.py): Hierarchical engine records "collapse" events in chelation_log during retrieval, replays for InfoNCE/hybrid loss training of adapter. OnlineUpdater does micro on-policy contrastive steps. Spectral masking + chelation in embed().
- Near-identity init + regularization + normalization + quantization gate + retention probes + structural_health = strong foundation for **controlled reversible correction** without immediate base model destruction.
- Integration with fitness_interfaces, evolution_strategies_optimizer (EGGROLL), model_scope*, computational_storage_poc, stability_tracker.

**Fatal Gaps (why it is not yet production-viable — directly mapped from Agent 2 audit + panels F-ML-001..047, 01-code-refinement)**:
- **Advisory-only self-healing**: `build_update_plan` / `execute_shadow_round` produces plans/ledgers but **never instantiates trainers or runs gradient steps on accepted directives** in the main path (config.allow_persistent_update defaults False; "safe persistent self-adaptation" is future work per seal-eggroll docs).
- **Off-policy sparse sedimentation dominates**: Collapse logs under old policy → batch replay with thresholded sparse signals (collapse freq >= threshold). InfoNCE temp=0.07 "dangerously small" → mode collapse, sparse gradients (only hard pairs get signal). Hybrid loss scale mismatch (InfoNCE dominates 10-100x). Hardcoded MSE in hierarchical path. No warmup, no seeds (non-repro), no tracking (F-ML-021 CRITICAL), checkpoint overwrite risk (F-ML-022 CRITICAL).
- **No KL / divergence control**: Self-edits cause "KL shocks" (sudden embedding distribution shifts), unpredictable drift, catastrophic forgetting of useful retrieval facts. regularization_loss() often returns 0.0 for MLP/LowRank/AttnRes.
- **Training/serving parity broken**: Adapter forward in training (no post-MSE normalize?) vs inference (unit-norm cosine assumption) → stored vectors have varying norms (F-ML-023 CRITICAL).
- **Data leakage + poor selection**: chelation_log trains on future eval data (F-ML-040 HIGH). No de-dupe; repeated queries pollute (F-ML-041). No held-out. Uniform sampling of all noise events.
- **Sample inefficiency**: Fitness/probe generation + sedimentation cycles are expensive and slow-converging. No dense per-dimension / per-"token" (per-query) supervision. Current is closer to naive RL than modern self-distillation.
- **Quantization survival hacky**: BoundedAdapter post-hoc wrapper (hardcoded 0.01/0.5) not in learning objective. Survival not first-class.
- **Loose coupling**: SelfEditDirectives propose "adapter_sft" / "eggroll_es" / "online_contrastive" but execution is not wired to actual OPSD-style training loops.
- **Other**: Weak experiment tracking, no inversion tests for chelation (does adapter invert cleanly?), limited continual/online hot-swap robustness, no integration of chelation with disk-resident computational storage paths.

**The OPSD/SDPO Insight (from Agent 1 + external research)**: This is **exactly** the missing practical layer. OPSD ("Self-Distilled Reasoner", Zhao et al. arXiv:2601.18734) + SDPO ("RL via Self-Distillation", Hübotter et al. arXiv:2601.20802) + MIS-PO (ar0cket1 practical online LoRA implementation in Hermes-Agent-Online-RL) solve **precisely** the self-correction stability + efficiency problem at scale in 2026:

**Core OPSD Mechanism** (adapted from paper Eq. 1,6):
- Single model = both teacher & student (different contexts).
- Student policy p_S(· | x): normal query/context (in CHELATEDAI: normal query + current ChelationAdapter state).
- Teacher policy p_T(· | x, y*): privileged context (ground-truth solution y*, or in chelation: full diagnostic trace + fitness vector + quant gate results + successful SelfEditDirective + ground-truth relevant docs + self-generated high-fitness probes).
- Student generates **on-policy rollouts** (in chelation: corrected embeddings / retrieval trajectories from current adapter on live queries or synthetic probes).
- Minimize **per-token (here: per-dimension or per-query) divergence** D(p_T || p_S) along the student's own trajectory (forward KL, reverse KL, JSD). Gradients **only through student**.
- **Stability tricks**: Per-token pointwise KL clipping (prevents stylistic/generic dimensions dominating semantically critical correction dimensions — analog of "stylistic tokens vs math tokens"); KL scheduling; teacher EMA; small LR (e.g. 2e-6).
- **Why it beats priors**: Dense token-level signal (vs sparse RL scalar reward that vanishes on uniform batch outcomes); on-policy realism (vs off-policy exposure bias); no external larger teacher (uses ground-truth + rationalization "evaluation easier than generation").

**SDPO/MIS-PO extensions** (ar0cket1 practical):
- SDPO: Hybrid scalar reward loss + self-distillation (top-K token distillation from self-teacher conditioned on rich feedback: fitness, critiques, quant survival).
- MIS-PO: Metropolis Independence Sampling + Filtered Policy Optimization. Importance sampling ratio filtering (ρ_min ≤ ratio ≤ ρ_max) on tokens/trajectories + small KL β≈0.001. Works with binary/scalar feedback only. Low-variance REINFORCE-style for online human/feedback loops. Perfect for hot-LoRA / hot-adapter updates.
- @ar0cket1 X (May 2026): Heavy emphasis on **KL shocks** in low-entropy/high-divergence regions, gradient shape analysis, why naive self-distill fails, and practical online continual self-improvement of adapters from live feedback.

**Perfect Fit for CHELATEDAI**:
- "Privileged" = rich diagnostics from SelfHealingChelationPlanner + FitnessEvaluation + QuantizationPromotionGate + structural_health + self-generated probes + ground-truth retrieval labels.
- "Student rollouts" = on-policy corrected embeddings / retrieval results from current adapter in AntigravityEngine.embed() / run_sedimentation_cycle().
- "Distillation target" = teacher embeddings/logits/deltas from privileged view.
- Residual structure (x + δ) + normalization + existing variants (Procrustes for orthogonality, Bounded for quant) are **ideal substrates** for KL-regularized asymmetric distillation of deltas.
- SelfEditDirective strategies map directly to "policy" that can itself be self-distilled (SDPO on directive generator).
- Sedimentation can be **replaced/augmented** by filtered on-policy self-distill data (entropy/divergence-based selection from Agent 7).
- Result: RL-like self-correction power + distillation-like sample efficiency + stability via KL control + filtering + retention.

**Evidence of Readiness**: The system already has the "self-edit planning + evaluation + ledger" (SEAL/EGGROLL), the "small residual correction substrate", the "retrieval fitness + probes", the "quant gate". OPSD supplies the **training dynamics glue** that was missing.

---

## 2. Consolidated Pain Point → OPSD Technique Mapping (Synthesized from Agents 2,4,7,12)

**Tier S (Very High Priority — Address in Loops 2-3)**:
| Pain Point (Evidence) | OPSD/SDPO/MIS-PO Technique | How It Directly Solves | Implementation Notes for CHELATEDAI |
|-----------------------|----------------------------|------------------------|-------------------------------------|
| P2: No KL/divergence control ("KL shocks", drift) (Agent 2 audit, panel F-ML, @ar0cket1 X) | Explicit KL penalty to base/reference + per-token/pointwise dim KL clipping + KL scheduling | Bounds distribution shift; prevents stylistic/generic dimensions from dominating corrections | Add `kl_to_base(adapter_out, base_embed, beta=0.001)` + pointwise clip(τ=0.1-0.5) in new `opsd_losses.py`. Schedule beta from stability_tracker.structural_health. |
| P3: Self-healing advisory-only — no training loop for accepted directives | Asymmetric privileged-context on-policy self-distillation (OPSD core) | Turns diagnostic signals + probes into dense training targets for adapter deltas | New `ChelationOPSDTrainer`: teacher = frozen(adapter + privileged_diagnostics), student = trainable(adapter + normal_query). Distill on student's on-policy retrieval rollouts. Wire SelfEditDirective.accepted → training examples. |
| P7: Poor sample efficiency (expensive probes/fitness, slow convergence) | Dense per-token/dim divergence supervision + on-policy rollouts (vs sparse scalar) | 5-10x better token/sample efficiency than RL/InfoNCE replay (per papers) | Replace/augment InfoNCE in sedimentation_loss with OPSD-style divergence on embedding deltas or projected logits. Use student's own generations. |
| P1: InfoNCE temp=0.07 → mode collapse + sparse grads | Temperature annealing + entropy-aware + divergence objectives + filtering | Prevents over-sharpening/hardest-pair domination | Hybrid loss: OPSD_divergence + annealed_InfoNCE. Filter batch by entropy/divergence (Agent 7). |

**Tier A (High Priority — Loops 3-5)**:
| Pain Point | Technique | Benefit | Notes |
|------------|-----------|---------|-------|
| P4: Catastrophic forgetting on repeated self-edits | KL-to-base + retention replay of high-retention on-policy probes + inversion tests | Preserves base retrieval NDCG while allowing correction | Mandatory retention probes in every OPSD batch. Add `inversion_loss` (adapter(adapter(x)) ≈ x). Use in Procrustes variant. |
| P10: Quant survival hacky (post-hoc Bounded) | Quantization-aware self-distillation + integrate bounds into loss | Corrections survive INT8 by construction | Fake-quant or real INT8 forward in teacher/student during distillation. Soften Bounded constraints via penalty/Lagrangian in loss. |
| P5/P6: Data leakage + train/serve parity (norm drift) | Proper on-policy held-out splits + consistency regularization + filtering (MIS-PO ratios) | No leakage; norm-consistent vectors | Agent 7 filtering + on-policy generation ensures fresh data. Add norm-consistency term in loss. Use held-out diagnostic sets for eval. |
| P9: Weak regularization on most adapters | KL-to-base + Frobenius (Procrustes) + scale penalty + low-rank during distillation | Prevents adapter explosion | Make regularization_loss() first-class in OPSD objective for all variants. |

**Tier B (Medium, Strategic — Loops 6-10)**:
- P8 (tracking): Combine with modern experiment tracking (not OPSD-specific).
- P11 (disk-first integration): Near-data self-distillation on computational storage POC (unique CHELATEDAI advantage).
- Weak continual/hot-swap: MIS-PO filtered online micro-updates (ar0cket1 Hermes pattern) for live adapter hot-load in AntigravityEngine.

**Cross-Cutting from Agent 4 (Dynamics)**: Current = off-policy replay (sedimentation) + advisory + micro-on-policy (OnlineUpdater). Target = **full on-policy asymmetric self-distill** (privileged diagnostics as teacher context) + **filtered sampling** + **dense delta divergence** + **KL control**.

**Cross-Cutting from Agent 7 (Filtering)**: Use divergence/entropy/low-fitness signals + MIS-PO importance ratio filtering + top-K (per-dim or per-probe) to select only high-value correction examples from chelation_log / probes. Avoids redundant/destructive data.

---

## 3. Ranked List of Most Promising Upgrade Patterns / Architectural Options (Consolidated & Prioritized)

Synthesized from Agent 1 (8 variants), Agent 13 (Tier S/A/B), Agent 4 (dynamics families), Agent 7 (filtering), Agent 12 (mappings). Each includes variances, pros/cons/risks, test ideas. **Ranked by leverage (impact on core gaps) + feasibility (existing substrate) + risk (forgetting/collapse/quant).**

### Tier S — Highest Leverage / Lowest Risk Starting Points (Implement First in Loops 2-4)
**S1: Asymmetric Privileged-Diagnostic OPSD Chelation (Core Recommendation — Variant A from Agent 1, S1 from Agent 13)**
- **Description**: Teacher = frozen ChelationAdapter (or variant) + full privileged context (diagnostics trace + fitness vector + quant gate + ground-truth docs + high-fitness probes from SelfEditEvaluation). Student = trainable ChelationAdapter + normal query context. Generate on-policy "rollouts" = corrected embeddings / retrieval results from student on training queries/probes. Minimize clipped forward-KL (or JSD) between teacher and student deltas/distributions. Pointwise per-dim clipping. Optional teacher EMA.
- **Variances**:
  - A1: Full-vocab analog (project embeddings to vocab-like via linear head for KL) vs direct vector MSE/KL on deltas.
  - A2: Teacher sees "rationalized" privileged (LLM-generated explanation of why correction helps) vs raw diagnostics.
  - A3: Per-adapter-variant (MLP vs Procrustes vs LowRank vs BlockAttnRes).
- **Pros**: Directly solves P3 (turns advisory into training), P2 (KL control), P7 (dense signal), P4 (KL-to-base implicit). Matches OPSD paper exactly. Leverages existing SelfHealingChelationPlanner + fitness + probes.
- **Cons/Risks**: Teacher may be weak on hard cases (mitigate: curriculum on probe difficulty; fallback scalar). Memory for full distributions (mitigate: top-K per-dim importance sampling + tail norm approx).
- **Test Ideas** (from Agent 11/1): correction_gain ΔNDCG, retention@K on held-out, kl_shock metric (max div during training), sample_efficiency (probes per +0.01 gain). Compare vs current sedimentation baseline.
- **Entry Point**: New `chelation_self_distillation.py` (OPSDLoss, AsymmetricChelationTrainer). Wire into SelfHealing...build_update_plan when accepted.

**S2: KL-Regularized Residual + Procrustes Self-Distillation (S2 + D from Agents)**
- **Description**: During any distillation (S1 or sedimentation replacement), add explicit `beta * D_KL(base || chelated) + existing regularization_loss()` (Frobenius for Procrustes, scale for Bounded). Use inversion test (adapter o adapter ≈ identity).
- **Variances**: KL on raw embeddings vs on deltas; scheduled beta (high on structural_health low); combined with LowRankAffine.
- **Pros**: Directly attacks forgetting (P4) + weak reg (P9). Procrustes variant already has strong orthogonality prior — perfect for KL.
- **Cons/Risks**: Over-regularization slows correction (mitigate: small beta 0.001, adaptive).
- **Priority**: Very high — cheap to add to any loss.

**S3: MIS-PO Filtered On-Policy Sedimentation Replacement + Online Micro-Updates (C from Agent 1, S3 from Agent 13, Agent 7 focus)**
- **Description**: Replace/augment chelation_log replay with filtered on-policy data: only high-value (high div or fitness gain) correction examples. Use MIS-PO ratio filtering on delta changes (trajectory + per-dim). Binary reward from SelfEdit accepted/high-gain. Small KL reg. Hot-swap adapter in engine (like Hermes LoRA hot-load).
- **Variances**: Pure filtering vs hybrid with OPSD dense loss; online micro (OnlineUpdater extended) vs batch; importance sampling weights.
- **Pros**: Solves P1/P5/P7 (filtering + on-policy + efficiency). Enables true continual self-improvement. Practical (ar0cket1 code is production-tested for online adapters).
- **Cons/Risks**: Ratio estimation variance (mitigate: clip ρ bounds 0.1-10); hot-swap instability (use SafeTrainingContext + validation).
- **Entry Point**: Extend `sedimentation_trainer.py` + `online_updater.py` with MISPOFilter + OPSD hybrid loss.

### Tier A — Strong / Medium Risk (Loops 4-6)
**A1: Quantization-Aware Self-Distillation for BoundedAdapter (E + A1)**
- Train with fake-quant or simulated INT8 in forward pass of teacher/student. Integrate min/max correction as soft penalty in loss (not just wrapper).
- **Pros**: Makes quant survival first-class (P10). Critical for deployment.
- **Risks**: Quant noise in gradients (mitigate: straight-through estimator + small noise injection).
- **Test**: quant_robustness (FP32 vs INT8 post-update Δ fitness).

**A2: Multi-Objective Self-Distillation (Correction + Retention + Efficiency + Structural Health) (A2)**
- Joint loss: OPSD_divergence + retention_KL + low-rank penalty + structural_health penalty (from diagnostics).
- **Variances**: Scalarized vs Pareto / evolutionary (tie to EGGROLL).
- **Pros**: Holistic; prevents single-metric gaming.
- **Risks**: Tuning weights hard (mitigate: use config + meta-optimization).

**A3: SDPO-Style Rich-Feedback Self-Distill on SelfEditDirective Policy (B)**
- Treat directive generator (strategy + params) as policy. After eval, rich feedback f = (fitness, retention, quant, structural, latency). Self-teacher (generator + f) provides dense supervision on chosen directive tokens/choices. Hybrid scalar reward + top-K distill.
- **Pros**: Self-improves the planner itself (meta self-healing).
- **Risks**: Complex (directive space discrete); start with simple strategies first.

### Tier B — Interesting / Higher Complexity or Strategic (Loops 7-10)
**B1: Low-Rank + EGGROLL + OPSD Hybrid**
- LowRankAffineAdapter + evolution strategies (existing) + OPSD distillation on the low-rank factors.
**B2: BlockAttnRes + LayerAttention Multi-Depth Self-Distill (G)**
- Teacher/Student at multiple block depths; cross-block attention distilled.
**B3: Computational Storage / Near-Data Self-Distillation (P11)**
- Perform teacher logprobs or partial distillation inside the disk-resident POC (unique differentiator).
**B4: Continual Lifelong with KL Scheduling + Retention Replay (H)**
- Dynamic beta/clip based on recent forgetting signals + replay buffer of successful past on-policy corrections.

**Overall Ranking Rationale (Agent 10)**: S1/S2/S3 first because they map 1:1 to the 4 Tier S pain points, leverage 80%+ of existing code (adapters, planner, engine, fitness), have lowest implementation risk (residual + existing reg + bounded substrate), and deliver measurable gains on core metrics (correction + retention + efficiency) with minimal new infra. Tier A adds quant/holistic depth. Tier B for long-term differentiation.

**Variances to Test Across All** (from literature + Agent 7):
- Divergence: forward_KL vs reverse_KL vs JSD vs hybrid.
- Clipping: per-dim pointwise τ vs global.
- Teacher: frozen at cycle start vs EMA (0.999) vs rationalizing LLM.
- Data: pure on-policy (student generations) vs mixed synthetic (from SelfEditDirective.synthetic_examples) vs filtered chelation_log.
- Adapter scope: only delta vs full adapter params (low-rank on adapter).
- Reward shaping: pure divergence vs + scalar fitness + retention bonus.

---

## 4. Highest-Leverage, Lowest-Risk Starting Points for Loops 2+

**Immediate (Loop 2 — Architecture)**:
1. Design `chelation_self_distillation.py` (new module): OPSD Loss (forward_KL + clipped + KL_to_base), AsymmetricPrivilegedContextBuilder (from SelfEditEvaluation + diagnostics), ChelationOPSDTrainer interface (compatible with existing adapters + create_adapter).
2. Extend `SelfHealingChelationConfig` + `build_update_plan` to optionally return "training_directives" with privileged_context dict.
3. Sketch training regime in `antigravity_engine.py` (or new `opsd_trainer.py`): on-policy generation hook during embed() or dedicated `run_opsd_sedimentation_cycle()`.
4. Update CHELATION_OPSD_RESEARCH_PLAN.md with this synthesis + agent links.
5. Baseline: current sedimentation vs naive SFT on accepted directives (for comparison).

**Loop 3 (Loss + Regime)**: Implement S1 core + S2 KL reg. 3-4 loss variants. Basic on-policy loop (no full filtering yet).

**Loop 4 (Stability)**: Add clipping, scheduling, inversion tests, Procrustes integration, retention replay buffer.

**Loop 5 (Filtering/Sample Eff)**: MIS-PO style filters + entropy/divergence selection (Agent 7). Replace 50% of sedimentation data with filtered on-policy.

**Loop 6 (Quant + Low-Rank)**: A1 quant-aware + B1 low-rank hybrid.

**Loop 7 (Directive Integration)**: A3 SDPO on SelfEditDirective "policy".

**Loop 8 (Eval Harness)**: Full `chelation_benchmark_suite.py` (extend benchmark_distillation.py + test_self_healing_chelation.py) with all metrics from Agent 11. Synthetic drift-correction tasks + BEIR/Scifact held-outs. Ablation matrix.

**Loop 9 (Impl + Tests)**: Top 3 patterns (S1+S2+S3) fully implemented + unit/integration tests + hot-swap validation.

**Loop 10 (Comparative + Roadmap)**: Head-to-head (all variants + baselines) on 5+ campaigns. Integration into aep_orchestrator, model_scope, computational storage, road-course. Final upgrade PR + docs + "viable production pattern" recommendation.

**Highest-Leverage Code Touchpoints**:
- `chelation_adapter.py`: Add `distillation_target(teacher_out)` or hooks for teacher mode.
- `self_healing_chelation.py`: Expose privileged_context in evaluations.
- `sedimentation_loss.py`: Hybrid OPSD + InfoNCE class.
- `antigravity_engine.py`: Add `run_chelation_opsd_cycle()` + on-policy generation path.
- `online_updater.py`: Extend to full MIS-PO micro SDPO.
- New: `opsd_losses.py`, `chelation_self_distillation.py`, `chelation_benchmark_suite.py` (in tests or new).

**Resource Claims / Parallelization**: Use fleet/orchestration for variant impls (Agents 7-9 style). Claim "adapter_training" + "qdrant_eval" resources.

---

## 5. Risks, Open Questions, Mitigations & Test Requirements

**Top Risks** (from literature + Agent 1/2/4/7):
- Teacher weaker than student on hard diagnostics → curriculum + fallback to GRPO-style group scalar reward.
- Overfitting to specific diagnostic signatures → strong KL-to-base (beta>=0.001) + diverse retention probes mandatory + held-out eval.
- KL shocks in low-entropy embedding regions (high-divergence "correction" dimensions) → pointwise clipping + entropy-aware filtering + small LR.
- Quant noise corrupting distillation gradients → straight-through + quant-aware loss + Bounded min_correction as soft target.
- Compute/memory (full teacher rollouts) → top-K dim sampling + parallel teacher (frozen) + low-rank always.
- Non-repro / checkpoint corruption → enforce seeds + SafeTrainingContext + SHA verification + experiment tracking (MLflow or local JSONL).
- Forgetting base capabilities on continual runs → mandatory retention@K benchmark every N cycles + inversion tests.

**Open Questions for Later Loops**:
- Best divergence for embedding space (vector KL vs projected token KL vs contrastive)?
- How to construct "privileged rationalization" without extra LLM calls (use existing fitness metadata)?
- Optimal filtering thresholds for MIS-PO ratios in embedding delta space?
- Can we do true "self-teacher on rich feedback" for the directive planner itself without discrete RL complications?
- Integration depth with disk-first computational storage (near-data teacher?).

**Mandatory Test Requirements (from Agent 11 skeleton + literature)**:
- Metrics: correction_gain (Δ NDCG/Recall/MRR on held-out BEIR/Scifact after update), retention_score (probe recall preservation vs baseline), stability (drift in base NDCG over 10+ self-heal cycles, max KL shock during training), sample_efficiency (queries/probes/cycles per 0.01 gain), quant_robustness (Δ after fake/real INT8), self_healing_success_rate (% directives that yield net positive after training), inversion_error (||adapter(adapter(x)) - x||).
- Ablations: divergence type, clip τ, teacher type (frozen/EMA/rationalize), filtering (none/MIS-PO/entropy), adapter variant, on-policy vs replay.
- Regressions: no worse than current sedimentation on latency, no base model degradation.
- Full AntigravityEngine integration tests + hot-swap + persistence across sessions + quantization_promotion_gate end-to-end.
- Synthetic controlled tasks: inject known embedding drift, measure correction + retention.

---

## 6. Immediate Next Actions & Deliverables for the Program

**For Orchestrator / Next Grok Turn**:
- Merge this synthesis into master `CHELATION_OPSD_RESEARCH_PLAN.md` (update status, add links to all loop_01/*.md, revise Loop 2+ deliverables).
- Create `docs/chelation_opsd_research/loop_01/10_synthesis_prioritization.md` (this file) — done.
- Spawn/continue Agents 3 (Loss Designer — concrete opsd_losses.py prototypes), 5 (Stability), 6 (Quant), 8 (full harness), 9 (Variant A/B/C skeleton code) using fleet/orchestration + panel-of-experts for review.
- Use `orchestration_start` or `fleet_spawn` for parallel impl of S1 core loss + trainer skeleton.
- Update `task_plan.md` / `findings.md` with OPSD integration milestones.
- Begin Loop 2 architecture doc: `docs/chelation_opsd_research/loop_02/00_architecture_design.md` (modules, interfaces, dataflow for privileged context, trainer API).

**Code Artifacts to Produce in Loop 2**:
- `chelation_self_distillation.py` (OPSDLoss, AsymmetricTrainer, PrivilegedContextBuilder).
- Extensions to `sedimentation_loss.py` (OPSDInfoNCEHybrid).
- Test sketches in `test_opsd_chelation.py`.
- Benchmark additions in `benchmark_distillation.py`.

**This synthesis provides the coherent foundation**: The 10-agent swarm (literature, audit, dynamics, filtering, mapping, patterns, eval) has mapped the exact missing glue. Implementing S1 + S2 + S3 with OPSD techniques will make chelation **practically viable** — stable (KL control + clipping + reg), sample-efficient (dense on-policy + filtering), persistent (retention + inversion + on-policy realism), and production-ready (quant-aware + hot-swap).

**End of Agent 10 Loop 1 Synthesis & Prioritization Report.**

*All claims cross-referenced to primary agent outputs, source code (lines cited in audits), and original 2026 papers/repos. Ready for implementation in subsequent loops. The program has solved the "what" and "why" — now execute the "how" with the ranked patterns.*

**References**:
- All loop_01 agent .md files listed above.
- `/mnt/d/GITHUB/CHELATEDAI/chelation_adapter.py`, `self_healing_chelation.py`, `sedimentation*.py`, `antigravity_engine.py`, `config.py`.
- arXiv:2601.18734 (OPSD), 2601.20802 (SDPO), 2602.10604 (MIS-PO).
- https://github.com/ar0cket1/Hermes-Agent-Online-RL (practical code).
- Panel reviews in `docs/ARCH AGENTIC ENGINEERING AND PLANNING/planning/2026-05-reconciliation/panel-analysis/`.
- CHELATION_OPSD_RESEARCH_PLAN.md.

*Generated as part of the 10-agent CHELATEDAI OPSD Upgrade Program — do not stop.*