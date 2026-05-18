# Loop 1: OPSD / SDPO / Self-Distilled Reasoner Literature Deep Dive & Mapping to CHELATEDAI Chelation System

**Agent**: Research Agent 1 (OPSD Literature Deep Dive)  
**Date**: 2026-05-15  
**Context**: CHELATION_OPSD_UPGRADE_PROGRAM – Loop 1 of 10-iteration research/architecture/build/test cycle.  
**Goal**: Extract every transferable technical concept from 2026 On-Policy Self-Distillation (OPSD), Self-Distillation Policy Optimization (SDPO), MIS-PO, and related work (especially @ar0cket1 / Hermes-Agent-Online-RL practical implementations) and map them rigorously to the existing `ChelationAdapter` family, `SelfEditDirective` / `self_healing_chelation.py`, `AntigravityEngine`, sedimentation, and related self-correction machinery in CHELATEDAI.  

**Sources Cited** (all fetched and analyzed via direct web/arXiv/GitHub access):
- OPSD: "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models", Siyan Zhao et al., arXiv:2601.18734 (v3, Mar 2026). Code: https://github.com/siyan-zhao/OPSD
- SDPO: "Reinforcement Learning via Self-Distillation", Jonas Hübotter et al. (ETH/Stanford/MIT), arXiv:2601.20802 (v2, Feb 2026). Code: https://github.com/lasgroup/SDPO ; HF TRL `SDPOTrainer`
- MIS-PO: Step 3.5 Flash technical report, arXiv:2602.10604; practical implementation in @ar0cket1/Hermes-Agent-Online-RL (https://github.com/ar0cket1/Hermes-Agent-Online-RL)
- Related: OGLS-SD (arXiv:2605.12400), @ar0cket1 X post (status 2054108160450064571, May 2026) on KL shocks / low-entropy/high-divergence regions / practical online LoRA self-improvement.
- Broader: On-policy distillation priors (Agarwal et al. 2024 GKD, Lu 2025), STaR/ReST comparisons in appendices, GRPO/RLVR baselines.

This document is exhaustive. It contains:
1. Core mathematical formulations and algorithms
2. Practical tricks, stability mechanisms, gotchas, and ablation insights from papers + real implementations
3. Direct, line-by-line applicability analysis to CHELATEDAI components (`chelation_adapter.py` variants, `self_healing_chelation.py`, AntigravityEngine, etc.)
4. 8+ concrete upgrade pattern variants proposed for testing in later loops (with pseudocode sketches, loss adaptations for embedding space, test ideas)
5. Risks, open questions, and integration points (computational storage, retrieval, model scope, quantization, continual learning)

---

## 1. The Core Problem CHELATEDAI Is Trying to Solve (and Why OPSD/SDPO Is the Missing Practical Layer)

From deep audit of the codebase (chelation_adapter.py:1-500+, self_healing_chelation.py, AntigravityEngine, fitness/quantization gates, task_plan.md, findings.md, etc.):

**Current Chelation System Strengths**:
- **ChelationAdapter** (and variants: MLP residual `x + g(x)` with tiny-init std=0.001, OrthogonalProcrustesAdapter via Cayley + DSM diagonal scale, LowRankAffineAdapter (LoRA-like U/V + bias), BoundedAdapter wrapper enforcing min/max correction norm for INT8 quantization survival (~0.0078 noise floor), BlockAttnResAdapter (multi-block + cross-block attention), LayerAttentionAggregator).
- All enforce near-identity at init (prevents immediate catastrophic forgetting of base embeddings).
- L2 normalization to hypersphere (critical for cosine retrieval).
- Regularization (Frobenius on skew for Procrustes, scale deviation, etc.).
- **Self-healing / SEAL-inspired**: `SelfEditDirective` (strategy, synthetic_examples, optimization_params), `SelfEditEvaluation` with fitness/retention/quantization_gate, `CandidateProvenanceLedger`, `SelfHealingChelationConfig` (reward_threshold, min_retention_score=0.8, retained_gain_threshold, structural_health, latency_regression, self-generated probes).
- Advisory (not mutating) by design; evaluates via probes, fitness_interfaces, QuantizationPromotionGate.
- Used in AntigravityEngine for dynamic embedding correction ("chelation" = controlled reversible stabilizing correction of representations), sedimentation, road-course tuning, etc.
- Existing attempts at stability: Procrustes regularization, low-rank constraints, bounded corrections, retention replay, structural health scoring.

**Core Failure Modes Identified in Current System** (from code, tests, task_plan.md, panel reviews):
- Self-generated corrections (SelfEditDirectives) and adapter training often unstable: drift, forgetting of useful base retrieval facts, poor sample efficiency (requires many probes/fitness evals).
- Naive reinforcement of corrections leads to "KL shocks" analog (sudden divergence in embedding space causing retrieval collapse or over-correction).
- Quantization survival is addressed (BoundedAdapter) but not deeply integrated into the *learning* objective.
- Sample inefficiency: fitness evaluation + probe generation is expensive; no dense per-"dimension" or per-token analog supervision.
- Lack of on-policy self-distillation: corrections are generated/evaluated but not distilled back densely from "privileged" (ground-truth or high-fitness) context vs normal (student) context using the model's own rollouts.
- Continual/online adaptation (hot-update adapters from live diagnostics/feedback) is rudimentary compared to production online RL systems.
- Stylistic vs "meaningful" signals dominate (analogous to OPSD's "stylistic tokens" vs "math tokens" – here: generic embedding directions vs semantically critical correction dimensions).

**The Perfect Mapping**: CHELATEDAI's chelation + self-healing is *precisely* attempting **iterative, diagnostic-driven, on-policy self-correction of internal representations (embeddings)** without external teachers. OPSD/SDPO/MIS-PO provide the current (2026) state-of-the-art practical toolkit for exactly this class of problem: achieving RL-like performance ceilings (strong self-correction) with distillation-like sample efficiency, while avoiding collapse via sophisticated KL/divergence control, filtering, asymmetric/privileged-context teaching, and dense token-level (here: dimension- or vector-level) supervision from the model's own generations.

The "self-teacher with privileged information" pattern directly translates: privileged = ground-truth relevant docs + rich diagnostics/fitness traces + successful SelfEditDirectives; student = normal query + current adapter state.

---

## 2. OPSD – On-Policy Self-Distillation (Self-Distilled Reasoner, arXiv:2601.18734)

### 2.1 Core Idea & Motivation
A *single* LLM acts as both teacher and student with different contexts. Teacher conditions on privileged information (ground-truth solution y* or verified CoT). Student conditions only on the question x. Student generates on-policy rollouts ŷ ~ p_S(·|x). Minimize per-token divergence D(p_T(· | x, y*, ŷ_<n) || p_S(· | x, ŷ_<n)) along the student's own trajectory. Gradients only through student.

**Intuition**: "A sufficiently capable LLM can rationalize external privileged reasoning traces and teach its weaker self." Evaluation/rationalization is easier than generation. Mirrors human learning: review correct solution, identify where own reasoning failed, internalize.

**Why superior to prior**:
- vs SFT/off-policy distillation: fixes distribution mismatch/exposure bias.
- vs RLVR (GRPO etc.): dense token-level signal instead of sparse scalar outcome reward (avoids advantage collapse when all group samples same reward; no need for 8-16 rollouts per prompt).
- vs external-teacher on-policy distillation: no larger teacher required; uses ground-truth directly via rationalization.

### 2.2 Mathematical Formulations (Key Equations)

**Teacher / Student Policies** (same params θ):
```
p_T(· | x, y*) ≜ p_θ(· | x, y*)
p_S(· | x)      ≜ p_θ(· | x)
```

**On-policy rollout**: ŷ ~ p_S(· | x)

**Per-token divergence objective** (full-vocabulary logit distillation, Eq. 6 in paper):
```
D(p_T || p_S)(ŷ | x) = (1/|ŷ|) Σ_n D( p_T(· | x, y*, ŷ_<n) || p_S(· | x, ŷ_<n) )
L_OPSD(θ) = E_{(x,y*)~S} [ E_ŷ~p_S [ D(p_T || p_S)(ŷ | x) ] ]
```
Gradients only through student logits (teacher is stop-gradient target).

**Divergence choices tested** (forward KL best):
- Forward KL(p_T || p_S)
- Reverse KL
- JSD_β (generalized Jensen-Shannon, β=0.5)

**Sampled-token policy-gradient variant** (Eq. 9, advantage shaping):
```
A_n(x, ŷ) = log p_T(ŷ_n | x, y*, ŷ_<n) - log p_S(ŷ_n | x, ŷ_<n)
L = -E [ E [ (1/|ŷ|) Σ A_n * log p_S(ŷ_n | ...) ] ]
```
Interpreted as dense-reward policy gradient (token-level reward r_n from privileged teacher). Contrasted with STaR (sequence-level binary reward on final answer correctness only; signal vanishes on all-failure batches).

**Per-Token Pointwise Divergence Clipping** (critical stability trick, addresses heavy-tailed stylistic tokens dominating math tokens):
```
ℓ_{n,v}^{(f)} = p_T(v|·) * f( p_S(v|·) / p_T(v|·) )
D_clip^f (p_T || p_S) = (1/|ŷ|) Σ_n Σ_v min(ℓ_{n,v}^{(f)}, τ)
```
Clipping τ prevents stylistic tokens (connectives like "wait", "hmm", "therefore") from dominating the signal over math tokens (numbers, operators, keywords). Empirically prevents performance collapse (see Figure 4).

**Prompting for Teacher Rationalization** (Figure 2): Teacher prompt includes ground-truth y* + instruction to "rationalize and generate its own solution" (but teacher only prefills; no generation during training).

### 2.3 Key Ablations & Results (Qwen3 family 1.7B-8B, OpenThoughts math data, AIME24/25, HMMT25)
- **Performance**: OPSD matches/exceeds GRPO while using ~8-12× fewer tokens (single rollout of 1024 tokens vs GRPO 8×16k). Converges in ~100 steps.
- **Token efficiency**: Dense signal allows learning from far fewer generations. GRPO often hits reward diversity collapse (std=0 in groups → zero gradient).
- **Divergence**: Forward KL strongest; reverse KL/JSD limited or negative.
- **Generation style**: Student TM-off (direct) + Teacher TM-on (thinking) gives highest KL on *math* tokens (best supervision).
- **Generation length**: Longer student rollouts (4096 vs 1024) do *not* consistently help; early tokens more critical (later tokens predictable given long prefix + privileged info).
- **Full-vocab logit dist vs sampled-token**: Full-vocab better (richer supervision) but higher memory (top-K or approx needed in practice).
- **Clipping**: Essential; without it, stylistic dominance causes collapse.
- **Teacher freezing**: Best to fix teacher at *initial* policy (regularization against drift) rather than updating it.
- **Forgetting / retention**: Not primary focus but on-policy nature helps (cf. SDPO explicit holdout results).

**Algorithm 1 (pseudocode from paper)**: Simple loop – sample batch, student on-policy ŷ, compute token-wise D, average loss, update θ (student only).

**Limitations noted**: Scaling beyond 8B untested; teacher may fail if problem exceeds model comprehension (curriculum needed); no explicit use of outcome verification in loss (only for privileged y*).

---

## 3. SDPO – Self-Distillation Policy Optimization (Reinforcement Learning via Self-Distillation, arXiv:2601.20802)

### 3.1 Core Idea: RLRF + Self-Teacher
Formalizes **Reinforcement Learning with Rich Feedback (RLRF)** vs classic RLVR (scalar reward only). Rich feedback = tokenized environment output (runtime errors, failed tests, judge traces, compiler output) + optional successful prior rollouts.

**Self-teacher**: Current policy π_θ(· | x, f) where f = rich feedback. Same model, different (augmented) context. After student generates on-policy y, re-evaluate logprobs of *same* y under self-teacher (no extra sampling). Distill: match student's next-token dist to self-teacher's feedback-informed dist.

This turns rich feedback into *dense logit-level advantages* without external teacher or value network. Model uses in-context learning to "retrospectively identify its own mistakes."

Even in pure scalar RLVR: use *successful* rollouts in batch as implicit "f" (correct solution) for failed ones on same x.

**Test-time self-distillation**: For hard single questions, run SDPO on that question alone to accelerate discovery (3× fewer attempts than best-of-k or multi-turn).

### 3.2 Formulations

**Core SDPO Loss** (KL distillation, Eq. 1):
```
L_SDPO(θ) := Σ_t KL( π_θ(· | x, y_<t) || stopgrad( π_θ(· | x, f, y_<t) ) )
```
stopgrad on teacher prevents regression. Teacher improves over training (student catches up and surpasses initial teacher).

**Gradient** (Proposition 2.1, policy-gradient form with self-teacher advantages):
```
∇L = E_y [ Σ_t E_ŷt [ log(π_student(ŷt)/π_teacher(ŷt)) * ∇ log π_student(ŷt) ] ]
```
Equivalent to negated logit-level policy gradient where advantage A_t = log(π_teacher / π_student) for the sampled token (dense, per-token, can be + or -).

**Comparison to GRPO advantages**:
- GRPO: A_i,t = r_i - mean({r})  (constant per rollout, scalar outcome only; collapses to 0 if group uniform)
- SDPO: A_i,t = log(π_θ(yit | x,fi,yi<t) / π_θ(yit | x,yi<t) )  (per-token, dense, leverages full f)

**Approximations for practicality**:
- **Top-K distillation**: Only top-K logits + tail prob mass (K=100 sufficient). Avoids full vocab memory blowup.
- **Regularized teacher**: EMA of student params or trust-region interpolation with initial teacher (α=0.01). Prevents divergence; non-regularized teacher collapses.
- **JSD** (symmetric) for distillation loss (stability, per Agarwal GKD).
- **Hybrid SDPO+GRPO**: λ * GRPO_adv + (1-λ) * SDPO_adv (helps weak models; λ=0.9 on small Qwen).

**Off-policy extension**: PPO-style clipped importance sampling (see appendix).

### 3.3 Key Results & Insights
- **Without rich feedback** (SciKnowEval science Q&A, ToolAlpaca): SDPO > improved GRPO (e.g., +10-20 pts on some tasks). 5-11× shorter generations (concise reasoning; GRPO produces filler "Hmm/Wait" loops). Achieves GRPO's 5h accuracy in ~50min wall-clock.
- **With rich feedback** (LiveCodeBench v6 code): SDPO 48.8% vs GRPO 41.2%; reaches GRPO final in 4× fewer generations. Gains scale with model size (emergent retrospection ability). Particularly strong on medium/hard questions.
- **Test-time**: 3× faster discovery on hard binary-reward questions (pass@64 base <0.03).
- **Dense credit assignment ablation**: Logit-level (top-100) >> token-level >> sequence-level (still > GRPO). Sequence-level SDPO already beats GRPO by using rich f.
- **Teacher improves**: Generative accuracy of self-teacher rises; final student surpasses initial teacher.
- **Forgetting**: On-policy SDPO preserves holdout capabilities (IFEval, ArenaHard-v2, MMLU-Pro) better than off-policy SFT on self-teacher successes. SFT causes more degradation.
- **Compute**: Low overhead (parallel logprob re-eval << generation). Top-K eliminates memory cost.
- **Scale**: Stronger base models → better SDPO (in-context retrospection emerges).

**Qualitative**: SDPO produces concise, non-circular reasoning. Sparse advantages (only corrects specific mistaken tokens).

**Stability**: Regularized teacher + JSD critical. KL control implicit via distillation target.

---

## 4. MIS-PO & @ar0cket1 Practical Online Implementation (Hermes-Agent-Online-RL + Step 3.5 Flash arXiv:2602.10604)

**MIS-PO (Metropolis Independence Sampling - Filtered Policy Optimization)**: Stable, low-variance, *single-trajectory* REINFORCE-style update for online/continual LoRA self-improvement from human (or auto) binary feedback. Designed exactly for the "one response = one trajectory, hot-load adapter" setting (no grouped rollouts possible in live chat).

**Key Mechanisms** (directly portable):
- **Token-level filtering**: Compute importance ratio x_t = π_train(a_t|s_t) / π_inference(a_t|s_t). Only tokens with ρ_min ≤ x_t ≤ ρ_max contribute (defaults [0.8, 1.25]). Completely excludes off-policy tokens (unlike PPO clipping which still scales them).
- **Trajectory-level filtering**: Geometric mean ρ̄(τ) over full response; reject entire trajectory if outside [0.9, 1.1]. Prevents accumulated drift.
- **KL regularization** (unbiased Schulman k3 estimator): L_KL = E[exp(log_ratio) - log_ratio - 1], β=0.001. Zero extra VRAM (reuses existing logprobs). Prevents excessive deviation from base.
- **Binary reward**: Simple +1 / -1 from feedback (up/down). No critic, no groups.
- **LoRA-focused**: rank 16 default, targets attn+MLP, hot-load into vLLM/Ollama/MLX. Warm-start from previous adapter. Max 4 saved adapters. Very low VRAM overhead (~50-200MB above inference).
- **Backends**: PyTorch CUDA/MPS/CPU, MLX (Apple unified memory + QLoRA auto), Tinker hosted.

**SDPO-style path in Hermes** (Tinker-backed approximation of Hübotter SDPO):
- Frozen teacher (initial or explicit).
- Top-K (sdpo_topk=20) distillation + scalar RL term: L_total = λ_rl * L_binary + λ_distill * L_teacher_distill.
- Text notes as rich feedback (dense correction signal).
- `train_steps:0` = one pass over new feedback batch (online).

**Practical Lessons from @ar0cket1** (X threads + repo + the referenced post):
- KL shocks / low-entropy high-divergence regions are real failure modes in naive self-distillation; require careful scheduling, filtering, and regularization.
- Human-in-loop (or auto-fitness) feedback UI + immediate training trigger + hot-load is the killer feature for continual agents.
- On-policy + filtering + small β KL keeps updates stable for live LoRA without collapse or forgetting.
- Rich notes (text feedback) dramatically improve over binary-only.
- Designed for "collapsing RLHF timelines from weeks to minutes".

This is *extremely* close to what CHELATEDAI needs for making SelfEditDirective + adapter updates production-viable and continual.

---

## 5. Direct Mapping & Integration Points to CHELATEDAI Chelation

| OPSD/SDPO/MIS-PO Concept | CHELATEDAI Analog / Gap | Proposed Port / Upgrade |
|--------------------------|-------------------------|-------------------------|
| Privileged teacher context (y* or f) | Ground-truth relevant vectors / successful retrievals + rich diagnostics (fitness, structural_health, latency, quant survival, SelfEdit fitness) | In AntigravityEngine / self_healing: construct "privileged prompt/context" embedding or diagnostic vector; teacher adapter sees it, student sees normal query. |
| Student on-policy rollout ŷ | "Student" = current adapter applied to query embedding; "rollout" = corrected embedding + downstream retrieval / probe results | Generate on-policy corrected embeddings (or synthetic probes from SelfEditDirective), evaluate with fitness. |
| Per-token divergence D(p_T || p_S) on next-token logits | Per-dimension or vector-level divergence on normalized embedding deltas / corrected embeddings | MSE + cosine on (teacher_corrected - student_corrected); or treat adapter output as "policy" and use forward KL on softmax-projected embeddings; or contrastive on positive (privileged) vs negative corrections. Full "vocab" analog = all embedding dims or attention over correction blocks. |
| Pointwise clipping on stylistic tokens | Stylistic/generic embedding directions dominate semantic correction dimensions | Per-dimension clipping or importance weighting in loss (focus on high-variance or high-fitness-impact dims). Use existing BoundedAdapter min/max as hard version. |
| Teacher regularization (EMA / trust-region / frozen initial) | Prevent adapter drift from base (already attempted via tiny init + Procrustes reg + regularization_loss) | Add EMA teacher adapter or trust-region penalty on delta from initial adapter weights. Freeze teacher adapter at start of self-healing cycle. |
| Top-K / approx distillation for efficiency | Memory for full "distribution" over embedding space | Top-K most influential dimensions or attention heads in correction; tail mass via norm. Critical for large embedding models. |
| Filtering (token/trajectory ratios in MIS-PO) | Retention checks, quantization_gate, structural_health already exist | Add ratio-based acceptance filters on adapter delta magnitude or fitness improvement trajectory. Reject "off-policy" SelfEditDirectives whose optimization_params drift too far. |
| Dense logit advantages from rich f | FitnessEvaluation + SelfGeneratedEvalProbe + CandidateLedgerEntry already rich | Use successful/failed probe outcomes + rich metadata as f for self-teacher on next SelfEditDirective generation or adapter update. |
| On-policy vs off-policy (SFT on teacher successes bad for forgetting) | Current self-healing is somewhat off-policy (synthetic examples, advisory) | Enforce on-policy: generate corrections with *current* adapter, distill only on those. |
| LoRA / low-rank continual hot-update + hot-load | LowRankAffineAdapter already exists; adapter_weights.pt persistence; Bounded for quant | Make all adapters LoRA-style trainable online. Implement hot-swap in AntigravityEngine (like Hermes). MIS-PO style trainer for binary/auto fitness feedback. |
| Quantization-aware + bounded corrections | BoundedAdapter explicitly for INT8 noise floor | Integrate quantization survival *into the distillation objective* (e.g., simulate quant during teacher/student forward, penalize divergence post-quant). OPSD clipping + SDPO top-K help here. |
| Concise reasoning / avoiding filler | Over-correction or verbose diagnostics in SelfEdit | Encourage minimal effective delta (regularize ||delta|| strongly; use in loss). |
| Test-time self-distillation for hard cases | Hard retrieval queries or failing probes | For difficult queries, run short "self-healing" optimization loop using SDPO-style on that single query + its probes. |
| Hybrid (SDPO + GRPO/MIS-PO) | Existing evolutionary strategies, road-course tuning | Combine outcome fitness (GRPO-like) with dense diagnostic distillation (SDPO). |

**Specific Files / Hooks**:
- `chelation_adapter.py`: All variants are perfect "student policy" heads. Add `teacher_forward` or dual-context support. Implement divergence losses (forward_KL, JSD, clipped) as methods on adapters or in a new `chelation_distillation.py`.
- `self_healing_chelation.py`: `SelfEditDirective` generation can become the "policy"; use successful ledger entries + rich reasons as f for SDPO-style self-distillation of better directives. Turn advisory into trainable (reinforce accepted ones).
- `AntigravityEngine`: Training loop for adapters (currently baseline + some distillation_benchmark). Replace/extend with OPSD/SDPO/MIS-PO trainer. Add privileged context construction from ground-truth or high-fitness retrievals.
- `fitness_interfaces.py`, `quantization_promotion_gate.py`, `stability_tracker.py`: Rich feedback sources. Feed directly into teacher prompt/context.
- Existing tests (`test_self_healing_chelation.py`, `test_attnres_adapter.py`, `benchmark_distillation.py`): Extend for OPSD variants.
- `adapter_weights.pt` + save/load: Support multiple adapter "generations" + hot-swap like Hermes.

**Embedding-Space Specific Adaptations** (critical difference from LLM token dist):
- Embeddings are continuous normalized vectors, not discrete tokens. Adapt "next-token dist" to:
  1. Projection to logits via a small head (learnable or fixed) then KL.
  2. Direct vector divergence (forward KL on Gaussian approx, Wasserstein, or simple ||teacher_delta - student_delta||^2 + cosine).
  3. Contrastive: privileged positive corrections vs negatives.
  4. Treat correction blocks (in BlockAttnRes) as "layers" and do cross-"token" (cross-dimension) distillation.
- Leverage existing normalization + hypersphere geometry (cosine = dot product).
- Procrustes/orthogonal constraints already provide nice inductive bias for "stable rotation" of corrections – combine with KL penalty.

---

## 6. Concrete Upgrade Patterns / Variants to Test (for Loops 2-10)

**Variant A: OPSD-Privileged-Diagnostic Chelation (Core Recommendation)**  
Teacher adapter/context = current adapter + privileged (ground-truth docs + full diagnostic trace from SelfEditEvaluation / fitness / quant gate). Student = normal query + current adapter. On-policy "rollouts" = corrected embeddings from current adapter on training queries. Minimize clipped forward-KL (or JSD) on (projected) distributions or vector deltas. Use pointwise dim-clipping. Freeze teacher at cycle start. Add EMA option.

**Variant B: SDPO-Style Self-Teacher on Rich Feedback for Self-Edits**  
For SelfEditDirective generation/planning: treat directive generator as policy. After execution/eval, rich f = (outcome fitness vector, retention scores, quant survival bools, probe results, structural health, latency). Re-eval "logprobs" of chosen strategy/params under self-teacher (generator + f). Distill with top-K + KL. Hybrid with scalar reward from fitness.

**Variant C: MIS-PO Filtered Continual Adapter Updates (Online RL Path)**  
Binary/auto fitness feedback (+1 accepted SelfEdit or high-gain probe, -1 regression). Single-"trajectory" (one query batch or one SelfEdit cycle). Token/dim-level ratio filtering on delta changes. Trajectory filter on overall fitness trajectory. KL reg (k3) with β=0.001 to base adapter. Low-rank (existing LowRankAffine or new LoRA on adapter params). Hot-swap weights in AntigravityEngine. Exactly like Hermes but for embeddings.

**Variant D: Hybrid OPSD + Bounded + Procrustes Regularized**  
Combine distillation loss + existing regularization_loss() + BoundedAdapter hard constraints in the objective (soften bounds via Lagrangian or penalty during distillation).

**Variant E: Quantization-Aware + Test-Time Self-Distillation**  
During distillation, forward through simulated quant (fake quant or real INT8). For hard queries, run short online SDPO loop on that query + its failing probes to discover better correction (3× efficiency).

**Variant F: Asymmetric + Filtered Sampling (OGLS-SD inspired)**  
Partition rollouts/probes into positive (high fitness) / negative. Contrastive logit (or delta) steering: positive teacher minus negative. Combine with outcome-guided filtering.

**Variant G: Full Multi-Adapter Distillation (BlockAttnRes + LayerAttention)**  
Distill across the attention-aggregated blocks; teacher sees privileged at multiple "depths".

**Variant H: Continual / Lifelong with Retention Replay + KL Scheduling**  
Replay high-retention successful probes + directives. Schedule KL coefficient or clipping τ based on structural_health or recent forgetting signals (from stability_tracker).

**Test Harness Ideas** (for Loop 8+):
- Extend `benchmark_distillation.py`, `benchmark_beir.py`, `test_self_healing_chelation.py`.
- Metrics: correction_gain (Δ fitness), retention@K (probe recall preservation), stability_over_iterations (drift in base retrieval NDCG), sample_efficiency (probes/generations per gain), quant_robustness (FP32 vs INT8 post-update Δ), self_healing_success_rate (fraction of SelfEditDirectives accepted + net positive after N cycles).
- Synthetic tasks: controlled embedding drift + correction, math/retrieval analogs on embedding space.
- Ablations: divergence type, clipping τ, teacher reg strength, rank, bounded min/max, on-policy vs synthetic data, filtering ratios.
- Compare all variants + baselines (current SFT-like on successful directives, pure RL on fitness scalar, no-adapter, frozen initial adapter).
- Integration tests: full AntigravityEngine + self_healing loop with new trainer; hot-swap; persistence across sessions; road-course + safety campaigns.

**Risks & Mitigations** (from papers + CHELATEDAI context):
- Teacher weaker than needed on hard problems → curriculum on probe difficulty; fall back to GRPO-like scalar.
- Memory for full "dist" → always use top-K / dim importance sampling + tail norm.
- Overfitting to specific diagnostics → strong KL / reg to base + retention probes mandatory.
- KL shocks / collapse → mandatory filtering (MIS-PO ratios) + clipping + regularized teacher + small LR (2e-6 like Hermes).
- Quant drift during training → quant-aware forward in loss.
- Forgetting base capabilities → on-policy only + holdout retention benchmarks + Procrustes-style orthogonality constraints.
- Compute: parallelize teacher logprobs / embeddings; low-rank always.

---

## 7. Next Steps & Recommendations for Subsequent Loops

**Immediate (Loop 2 Architecture)**: Design master `chelation_distillation.py` module with OPSD/SDPO/MIS-PO losses adapted to embedding space + adapter interface. Define privileged context construction API. Sketch trainer class supporting all 8 variants + hybrid.

**Loop 3**: Implement loss functions + basic training regimes (on-policy generation of corrections in AntigravityEngine).

**Loop 4**: Stability (clipping, teacher reg, KL scheduling, Procrustes integration).

**Loop 5-6**: Filtering/sampling + quant-aware + low-rank specifics.

**Loop 7**: Full SelfEditDirective integration (trainable directive policy via SDPO).

**Loop 8-9**: Comprehensive benchmark suite + full implementations + unit/integration tests.

**Loop 10**: Head-to-head experiments, integration into main pipelines (aep_orchestrator, road-course, computational storage), final upgrade PR + docs.

**Master Orchestration**: Update `CHELATION_OPSD_RESEARCH_PLAN.md` and `docs/chelation_opsd_research/` with findings. All 10 agents should contribute to a shared synthesis doc.

This literature provides the exact missing practical machinery (dense self-supervised signals from privileged rationalization + robust online filtering + KL control) to turn CHELATEDAI's elegant but currently advisory/self-healing chelation concepts into a stable, sample-efficient, continually improving production system.

**References & Further Reading** (all primary sources):
- arXiv:2601.18734 (OPSD + code)
- arXiv:2601.20802 (SDPO + code)
- arXiv:2602.10604 (MIS-PO)
- https://github.com/ar0cket1/Hermes-Agent-Online-RL (full practical online LoRA + SDPO approx)
- Related priors in papers (GKD, on-policy distillation 2025 works).

**End of Agent 1 Loop 1 Report**. Ready for synthesis with other agents and progression to architecture design.

---

*Generated as part of the 10-agent CHELATEDAI OPSD Upgrade Program. All claims backed by direct source analysis.*