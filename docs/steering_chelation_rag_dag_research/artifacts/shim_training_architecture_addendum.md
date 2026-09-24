# Shim Training & Architecture Addendum
## OPSD-Style Distillation, EGGROLL Population Search, Micro-SLM Route Policy Objectives, and Loop Updates for Shim Cascades + MTP Lookahead

**Program**: Steering-Chelation-RAGDAG-MicroSLM (10-Loop BHS-Governed)  
**Agent Slice**: Agent 5 — Training, OPSD/EGGROLL & Architecture Integration  
**Date**: 2026-05-26  
**Status**: Focused research addendum (Loop 1 synthesis input for Loop 2 architecture)  
**Required Reading (this document assumes)**:  
- `docs/steering_chelation_rag_dag_research/shim_nodes_mtp_lookahead_nomenclature.md` (full; SV/SN/SC/PCS/MSL/URS/Shim Backdoor/SE-RDAG definitions + integration table lines 113-126 + BHS token-reduction rules lines 160-164)  
- `docs/steering_chelation_rag_dag_research/STEERING_CHELATION_RAGDAG_MICROSLM_RESEARCH_PLAN.md` (Loop 2/3/7 placements lines 61,63,67; "successful shim cascades" as privileged traces line 55; micro-SLM as "route policy head" line 54)  
- `docs/chelation_opsd_research/loop_01/10_synthesis_prioritization.md` (S1 Asymmetric Privileged-Diagnostic OPSD lines 116-125; Tier S pain-point mapping; cross-cutting from Agents 4/7)  
- `docs/chelation_opsd_research/loop_01/12_pain_point_to_opsd_mapping.md` (P2 KL control, P3 advisory-only → training, P7 sample efficiency lines 58-63)  
- `docs/evolution-strategies-hyperscale-chelatedai-analysis.md` (EGGROLL alignment matrix, Phase 1-8 roadmap, low-rank E = A B^T, scalar fitness, Kalman sigma)  
- `docs/llm-architecture-ai-engineering-adaptation-review-2026-04-27.md` (MTP arXiv:2404.19737 mapped to speculative retrieval: cheap proposer + exact verifier lines 277-286 and 95; speculative decoding → retrieval analogue)  
- `self_healing_chelation.py:778` (SelfEditDirectiveOPSDIntegrator + OPSDTrainingBatch with privileged_diagnostics; build_asymmetric_teacher_student_objective:931; compute_embedding_kl_regularization:1002)  
- `evolution_strategies_optimizer.py:152` (LowRankEvolutionStrategyOptimizer; _sample_parameter_perturbation low-rank left@right.t()/sqrt(rank) lines 201-211; KalmanSigmaScheduler; population fitness shaping)  

**Cross-Program Inheritance**: Re-uses CHELATION_OPSD_BHS_RESEARCH_RUBRIC.md baseline + STEERING_CHELATION_BHS_RESEARCH_RUBRIC.md extensions (budget-adjusted lift, route cohesion, quantization survival delta, provenance cards for traces, rollback success). All CLAUDE.md + brutal-honesty-rulebook.md v3.3 rules apply (evidence rule §2 Rule 1; L13 soft-prose-claimed-as-mechanical; Tier B independence).

---

## 1. OPSD-Style Asymmetric Privileged Distillation for Shim Cascades and MTP Shim Lookahead

### Core Pattern (Exact Prior OPSD)
From OPSD Loop 01 synthesis (10_synthesis...md:116-125 S1 "Asymmetric Privileged-Diagnostic OPSD Chelation") and mapping (12_...:58-63 P3 "Self-healing advisory-only → on-policy self-distillation"):

- **Single model (or policy head) = teacher + student (different contexts)**.
- **Student policy** p_S(· | x): normal query + current adapter / route state (no privileged signals).
- **Teacher policy** p_T(· | x, y*): privileged context = full diagnostic trace + fitness vector + quant gate + successful SelfEditDirective + ground-truth relevant docs + self-generated high-fitness probes.
- Student generates **on-policy rollouts** (corrected embeddings / retrieval trajectories / now: shim selections + cascades under current policy).
- Minimize **per-"token" (here: per-dimension or per-shim-decision) divergence** D(p_T || p_S) along the student's trajectory (forward KL preferred, with pointwise clipping to prevent stylistic/generic dimensions dominating; + KL-to-base for retention).
- Gradients only through student. Teacher is frozen (or EMA). Dense supervision vs sparse scalar reward.

**Existing Substrate (self_healing_chelation.py:758-1000)**:
- `OPSDTrainingBatch`: student_inputs (normal view), teacher_targets (privileged), privileged_diagnostics (structural_health, quantization_gate, retrieval_anomaly, runtime), positive/negative pairs from probes.
- `SelfEditDirectiveOPSDIntegrator.directive_to_onpolicy_batch(...)`: converts accepted directive + diagnostics → batch. `_build_privileged_teacher_targets` injects "PRIVILEGED: ..." meta-signals (high variance, collapse risk, quant failure).
- `build_asymmetric_teacher_student_objective(...)`: MSE(student_emb, teacher_emb) + KL-proxy on delta vs base (kl_weight from directive). Explicit L4 note in docstring: "sketch... assumes torch... returns diagnostic dict without computing".
- `compute_embedding_kl_regularization`: retention term for any adapter variant.

### Privileged Shim Trace (Definition for This Program)
A **privileged shim trace** is the direct analogue of OPSD privileged_diagnostics + teacher_targets, but for the shim layer inside SE-RDAG / micro-SLM route policy:

```
PrivilegedShimTrace = {
  "query_context": str | embedding,                    # student view
  "chelation_variance": float, "isomer_drift": float, "structural_health": float,  # triggers
  "candidate_shims": List[shim_id],                    # registry hits at SIP
  "activated_cascade": List[Tuple[shim_id, tier, token_delta, fitness_delta]],  # full SC execution trace
  "mtp_lookahead_predictions": List[shim_id],          # what MTP head proposed pre-activation
  "token_cost_total": int,                             # including cascade + verification
  "cascade_success": bool, "route_cohesion_lift": float, "final_ndcg_delta": float,  # outcomes
  "quant_survival": bool, "rollback_feasible": bool,   # safety
  "provenance": {"source": "organic| synthetic| teacher_oracle", "checksum": str, "seed": int},
  "teacher_rationalization": str | embedding | fitness_vector  # optional LLM or probe-derived "why this cascade worked"
}
```

**Training Use**:
- **Student (micro-SLM route policy head or steering policy)**: sees only query + current DAG state + chelation signals → selects SN or proposes cascade.
- **Teacher (privileged)**: sees full trace above (including what *actually* succeeded in a prior high-fitness execution or oracle replay) → provides dense target distribution over shim selections / cascade steps / MTP lookahead labels.
- On-policy rollouts: during live or simulated SE-RDAG expansion, the policy's shim choices generate the student trajectory; privileged replay supplies teacher targets.
- Loss: asymmetric KL (or JSD) over shim-selection logits + cascade-acceptance head + MTP-head predictions, plus token_cost_penalty and cohesion terms (see §3).
- MTP Shim Lookahead (nomenclature:79-84) is trained as an auxiliary head: when SN_i is selected, predict top-k useful SN_{i+1..} that historically compounded well (supervised by successful privileged traces).

**Pseudocode Sketch (extends existing integrator pattern)**:

```python
# In extended ShimOPSDIntegrator (new, modeled on SelfEditDirectiveOPSDIntegrator:778)
def shim_trace_to_training_batch(activated_cascade_trace: PrivilegedShimTrace,
                                 student_policy_output: Dict) -> ShimOPSDBatch:
    student_shim_logits = student_policy_output["shim_selection_logits"]
    teacher_shim_dist = build_teacher_dist_from_trace(activated_cascade_trace)  # softmax over successful shims + MTP preds
    return ShimOPSDBatch(
        student_inputs=trace["query_context"],  # normal view only
        teacher_targets=teacher_shim_dist,
        privileged_trace=trace,  # full for KL shaping / filtering
        token_cost=trace["token_cost_total"],
        cascade_success=trace["cascade_success"]
    )

def shim_asymmetric_distill_loss(batch, student_logits, base_policy_logits=None):
    # Forward KL on shim decisions (dense per-shim-decision signal)
    distill = F.kl_div(F.log_softmax(student_logits), batch.teacher_targets, reduction='batchmean')
    # Pointwise clip per "decision" (nomenclature-style tier escalation)
    distill = clip_per_decision(distill, tau=0.5)
    kl_base = 0.001 * mse_delta(student_logits, base_policy_logits) if base else 0  # retention
    token_penalty = 0.01 * batch.token_cost * (1 if not batch.cascade_success else 0.5)
    return distill + kl_base + token_penalty
```

**Cross-Ref**: Exactly mirrors OPSD S1 (10_synthesis:117 "Teacher = frozen... + full privileged context... Student = trainable... on-policy rollouts... clipped forward-KL"). Extends to shims per research plan line 55 ("successful shim cascades" as privileged traces). MTP lookahead training uses the same trace's "mtp_lookahead_predictions" as dense labels (cf. llm-arch MTP → speculative retrieval verifier).

---

## 2. EGGROLL Low-Rank Population Search for Precomputed Shims and Cascade Combinations

### Exact EGGROLL Mapping (from evolution-strategies...md + optimizer code)
EGGROLL = Evolution Guided GeneRal Optimisation via Low-rank Learning (paper validated in evolution...md:32-43). Core:

- Black-box scalar fitness (no gradients through objective).
- Low-rank perturbations: E = A B^T / sqrt(r) for each population member (exact code: evolution_strategies_optimizer.py:206-211: left = randn(rows,rank); right=...; return (left @ right.t()) / sqrt(rank)).
- Aggregate over large population → effective full-rank update.
- Hardware win: cheap low-rank batched work + base matmul → looks like inference.
- Quant/int8 compatible (paper + ChelatedAI BoundedAdapter alignment, evolution...md:52).
- Kalman-style sigma adaptation from fitness variance (optimizer:176 KalmanSigmaScheduler; evolution...md Phase 3:202-210 "fitness variance modulates ES sigma").

**ChelatedAI-Native Fitness** (evolution...md:188-198 + research plan + BHS rubric):
NDCG@10 gain + collapse reduction + topology cohesion + isomer penalty + quant survival + **new for shims**: route_cohesion_lift, -token_cost_delta, cascade_success_rate, MTP lookahead hit-rate.

### Application to Precomputed Shims (PCS) and Cascades (nomenclature:72-76,61-68)
- **Black-box over shim fitness**: Shim Registry entries (or low-rank embeddings of shim vectors) + discrete/embedded cascade combinations treated as "parameters".
- Population = N hypotheses of (base shim + small cascade tree of depth ≤k, or PCS materialization).
- Each member: apply low-rank perturbation to shim vector params or to a learned embedding of "cascade recipe"; evaluate full SE-RDAG insertion + retrieval on fixture queries using scalar fitness above.
- Elite archive (existing in optimizer:170) retains high-utility PCS.
- Successful high-fitness low-token cascades are promoted to Precomputed Shim status (O(1) registry lookup) and become privileged traces for OPSD (§1).
- Synthetic seeding (nomenclature open Q5:173): teacher models + EGGROLL search generate initial useful SN without months of organic usage.

**Pseudocode Sketch (extends LowRankEvolutionStrategyOptimizer:152 directly)**:

```python
# ShimPopulationEvaluator (black-box; no diff through DAG)
class ShimCascadeESOptimizer(LowRankEvolutionStrategyOptimizer):
    def __init__(self, shim_registry: ShimRegistry, ...):
        # params = low-rank factors for shim vectors + cascade embedding table
        super().__init__(shim_embedding_module, config)
    
    def evaluate_population(self, population_perturbations, query_fixtures, fitness_fn):
        fitnesses = []
        for member_pert in population_perturbations:
            candidate_shims = materialize_perturbed_shims(self.base_shims, member_pert)
            cascades = propose_cascades_from_embeddings(candidate_shims)  # small trees
            scores = []
            for q in query_fixtures:
                trace = se_rdag_execute_with_shims(q, candidate_shims, cascades)  # insert-once
                scores.append(fitness_fn(trace))  # route_cohesion - lambda*token_cost + success
            fitnesses.append(mean(scores))
        return fitnesses

    # After pop eval: weighted update (existing ES logic) → promote top PCS to registry
    # Kalman sigma from var(fitnesses)
```

**Cross-Ref**: Direct from evolution...md Phase 1 (add "eggroll_es" optimizer around adapters → extend to shim registry), Phase 2/4 (scalar fitness + micro-pop search), Phase 5 (quant scoring mandatory before promotion). Optimizer code provides the exact low-rank sampling + antithetic + elite machinery. Aligns with research plan "EGGROLL low-rank population search over route + shim combinations" (line 9,55).

---

## 3. Proposed Training Objectives for the Micro-SLM Route Policy Head

Per research plan: micro-SLM (2-4GB) is the "learned 'router of routes'" / "route policy head" (lines 10,54). Inputs: chelation signals + DAG state + active shim context + Model-Scope features. Outputs: reroute proposals, shim selections + cascade proposals, commit.

**Multi-Objective (must be jointly optimized; budget-aware per BHS rubric)**:

Primary scalarized or Pareto loss for policy head (and MTP auxiliary head):

L_total = L_route_cohesion + α * L_token_cost + β * L_cascade_success + γ * L_opsd_distill + δ * L_kl_base + ε * L_mtp_lookahead

Where (pseudocode):

```python
# Route cohesion (topology/isomer analogue over proposed route family)
L_route_cohesion = -mean( pairwise_cosine_sim(proposed_shim_vectors) ) + isomer_penalty(proposed_set)

# Token cost (minimize; includes full cascade + verification; budget-adjusted)
L_token_cost = mean( trace.token_cost for trace in onpolicy_rollouts )   # or surrogate from policy

# Cascade success (binary or shaped reward from privileged outcomes)
L_cascade_success = -mean( success * (1 + cohesion_lift) - failure_penalty )

# OPSD asymmetric distill (from §1 privileged shim traces)
L_opsd_distill = shim_asymmetric_distill_loss(batch, policy.shim_logits, ...)

# Retention
L_kl_base = compute_embedding_kl_regularization(base_policy, current_policy)  # or logit KL

# MTP lookahead auxiliary (predict useful next shims; trained on successful traces)
L_mtp_lookahead = cross_entropy( mtp_head(current_shim), teacher_mtp_labels_from_trace )
```

**Hyperparameters**: Start small α/β (0.01-0.1) tuned via EGGROLL outer loop. Use MIS-PO-style filtering (from OPSD Agent 7 cross-cut) on high-divergence / high-gain shim decisions only.

**Integration**: Micro-SLM forward pass at SIPs (nomenclature:56). MTP head runs cheaply on selected shim to pre-fetch candidates (speculative, gated by policy + budget — never unconditional per nomenclature:135).

**Cross-Ref**: Extends OPSD loss families (synthesis §4 Loop 3) + EGGROLL scalar fitness (evolution Phase 2) + MTP speculative principle (llm-arch P2).

---

## 4. Concrete Updates Needed to Loop 2 Architecture, Loop 3 Loss Families, and Loop 7 Self-Edit Integration

### Loop 2 (Architecture — per plan line 61 + nomenclature §5:145-149)
- Define `ShimNode` dataclass + `ShimRegistry` interface (register/lookup_by_context/get_cascade/update_usage + versioning/rollback for BHS).
- Extend RerouteDAG → SE-RDAG with shim node expansion rules + SIP hooks (post-embed, VectorSteerer, micro-SLM forward, block-graph dispatch).
- Specify MTP Shim Lookahead head interface (aux head on micro-SLM or standalone lightweight; input current SN activation + context; output ranked next-shim proposals + confidence).
- Micro-SLM route policy: explicit shim_selection_head + cascade_proposal_head + mtp_lookahead_head; input schema includes "active_shim_context".
- Drive-node dispatch contract: small PCS + cascades can be block-graph payloads (computational_storage_poc/block_graph.py parity required).
- First artifact: minimal ShimRegistry + SE-RDAG skeleton (stub-free per CLAUDE Rule 1; or explicitly scoped "Loop 2 architecture slice" with L1 disclosure).

### Loop 3 (Loss Families — per plan line 63 + OPSD synthesis Loop 3)
- New module or extension: `shim_opsd_losses.py` (or augment sedimentation_loss + existing OPSD sketches).
- Add ShimCascadeKL, TokenBudgetPenalty, CascadeSuccessShapedReward, MTPLookaheadCE terms (exact pseudocode in §3).
- Hybrid: OPSD shim distill + annealed route-cohesion contrastive + EGGROLL-compatible scalar fitness path (for non-diff pop search).
- Stability: pointwise per-shim-decision clipping + KL scheduling from structural_health (exact OPSD P2 mitigation, 12_...:58).
- Filtering: MIS-PO ratio on shim-decision trajectories (high-divergence shim choices prioritized).

### Loop 7 (Self-Edit Directive Integration — per plan line 67 + nomenclature:119)
- Extend `SelfEditDirective` (self_healing_chelation.py:22) with variants: `shim_insertion`, `shim_promotion`, `shim_deprecation`, `cascade_edit`, `mtp_lookahead_tune`.
- `SelfEditDirectiveOPSDIntegrator` extended → also emits `PrivilegedShimTrace` batches when directive involves shims.
- Planner (build_update_plan) can now propose shim registry mutations from diagnostics (high-variance SIP → "insert new PCS shim" directive).
- Execution of accepted shim directives feeds directly into OPSD + EGGROLL pop search for the registry.
- Ledger (CandidateProvenanceLedger) must record shim provenance + token deltas for BHS replay.

**BHS Gate for All Loops**: No architecture doc or loss sketch counts as "implemented" until wired (even minimally) and smoke-tested per Rule 5; claims require EVIDENCE + SMOKE in any associated PR.

---

## 5. BHS Evidence Requirements Specific to Claiming "Shim Backdoors Reduce Token Usage"

Per nomenclature §5 BHS Considerations (160-164): "Any claim of 'token reduction via shim backdoors' requires before/after token accounting on the exact same query set with identical quality gates. Cascade depth and fan-out must be reported; ... Precomputed shims must show they were derived from evidence (not hand-crafted)..."

**Mandatory (non-negotiable, inherits + extends STEERING_CHELATION_BHS_RESEARCH_RUBRIC + rulebook §0 evidence rule + §2 Rule 1)**:
- **Runtime evidence only**: Command output / artifact from production SE-RDAG + micro-SLM route policy path (not test fixtures, not doc prose). Must survive fresh checkout.
- **Exact before/after on identical queries**: Same held-out query set (golden collapse + road-course fixtures extended with "shim insertion under noise" per nomenclature:157). Report total prompt+completion+retrieval+verification tokens (or proxy if generative RAG not yet active).
- **Quality gates preserved**: NDCG/MRR/Recall@K + route_cohesion + structural_health + isomer score **no regression beyond pre-registered tolerance**. Budget-adjusted lift = raw_lift / extra_tokens_used.
- **Provenance + replay**: Every shim backdoor trace in the "after" set must carry PrivilegedShimTrace-style card (checksum, seed, source organic vs synthetic vs EGGROLL-derived). Replay must reproduce the token delta.
- **Cascade bounds**: Max depth/fan-out reported; unbounded/high-variance cascades = failure (L4 if claimed as win).
- **Quant + rollback**: Delta must survive INT8/Bounded floor; rollback success rate ≥ threshold on the same queries.
- **No L13**: A doc claiming "shim backdoors reduce tokens via MTP" without the above runtime artifacts + Tier B disprove attempt is soft-prose-claimed-as-mechanical (L13). EGGROLL-derived PCS must cite the exact population run artifact (not "we ran ES").
- **PR body lines** (mandatory template §4 rulebook): EVIDENCE: (token-accounting command output + artifact path), SMOKE: (scripts/smoke... on shim path), BHS_* fields, CARRIED DEBT for any ceiling-tier gaps.
- **Independent Tier B**: Fresh agent given diff + brutal-honesty + EVIDENCE + smoke command + this rulebook + nomenclature BHS notes; task = try to disprove the token claim.

If any of the above is missing, the claim is L4 (partial-as-complete) or L9 (doc-as-implementation). Empty "we will measure later" is unjustified.

---

## Pseudocode Summary (Consolidated)

See §1 (shim_asymmetric_distill_loss), §2 (ShimCascadeESOptimizer.evaluate_population), §3 (L_total + components) for executable sketches. All extend existing classes (LowRank...Optimizer, SelfEditDirectiveOPSDIntegrator, OPSDTrainingBatch) rather than greenfield.

---

## Brutal Honesty on This Addendum (Per Rulebook §4 + Program Rubric)

**What I did NOT implement that the title might imply**: No code changes, no new loss modules, no micro-SLM training runs, no ShimRegistry, no SE-RDAG, no first shim experiment executed. This is pure synthesis + proposal for Loops 2+.

**What I stubbed / sketched (with file:line analogs)**: All pseudocode is research sketch (L1/L4 pattern identical to self_healing_chelation.py:951 "Brutal honesty (L4): This method assumes... returns a diagnostic dict without computing"). No production path exercised.

**Conditionals that exist ONLY because real path missing**: None added (this is a doc).

**Broad try/except**: N/A.

**Tests that do NOT exercise production**: N/A (no tests written).

**Claimed "complete" without e2e smoke**: None. This addendum is explicitly "Loop 1 synthesis input"; promotion of any pattern requires full evidence chain per rubric.

**Lie-taxonomy self-classification**: L9 risk if this doc is later cited as "shim training implemented" without the runtime artifacts it demands. No other L1-L13 instances in the diff (doc-only). I claim nothing was overstated; all mappings are directly quoted from source files:lines.

**Visibility status**: Feature is hidden — not exposed via UI/API/docs beyond this research artifact. No roadmap tick or "working" implication.

**EVIDENCE**: This file itself + cross-referenced source docs + code paths cited (all survive checkout today). No runtime shim training evidence exists yet.

**SMOKE**: N/A — docs/analysis artifact only. (Future shim PRs must name floor/ceiling tier.)

**BHS_SELF_DRAFT**: 85 (solid cross-refs and definitions grounded in 8+ source files; gaps in first-experiment data reqs are explicitly called out below rather than minimized).  
**BHS_SELF_DRAFT_AGENT**: session 2026-05-26 Agent 5 (this slice).  
**BHS_TIER_B / BHS_TIER_B_AGENT / BHS_OFFICIAL**: (to be assigned by independent adversarial fresh agent per Rule 4; must differ).  
**BHS_TIER_B_SEVERITY**: "important" (research proposal; token-reduction BHS rules are load-bearing but untested in shim context).  
**CARRY_FORWARD**: "First shim experiment data requirements and micro-SLM training feasibility blockers (see §6 below) — target Loop 2 architecture PR".  
**DEFERRED_SCOPE**: none (scope exactly matches assigned slice).  
**LOOP_ITERATIONS**: 1.  
**OPERATOR_OVERRIDE**: empty.

---

## 6. Short Brutal Honesty on Training Feasibility and Data Requirements for the First Shim Experiments

**Feasibility Assessment (evidence-grounded, no speculation)**:

- **Substrate readiness (high)**: Excellent leverage from existing (FeatureDirectionBank → ShimRegistry natural, SelfEditDirectiveOPSDIntegrator + OPSD batch machinery at self_healing...py:778 already produces privileged traces that can be extended to PrivilegedShimTrace with <50 LOC delta, LowRankEvolutionStrategyOptimizer + Kalman ready for black-box shim fitness, SE-RDAG extension points documented in nomenclature + plan). No greenfield from scratch.
- **First experiment blocker (data)**: Zero organic shim activation traces today. Synthetic seeding (teacher + EGGROLL pop search over synthetic collapse fixtures) is the only viable path for Loop 2 smoke (nomenclature open Q5 explicitly flags this). Requires: (a) extended synthetic_collapse_benchmark.py or road-course fixtures with "shim insertion under noise" tasks (plan line 157), (b) oracle/teacher that can label "successful cascade" outcomes with token costs + cohesion, (c) provenance cards for every synthetic trace.
- **Compute / micro-SLM**: 2-4 GB class (existing quantized zoo + llama.cpp per plan) is plausible host for route policy head. But: any training that mutates core without frozen+adapter discipline = automatic L4 per STEERING BHS rubric. First smoke must be adapter/steering-head only on frozen micro-SLM.
- **Quant / stability risk (high)**: Shim vectors + cascades must survive Bounded/INT8 by construction (nomenclature:136 + BHS rubric quant survival delta mandatory). EGGROLL Phase 5 + OPSD quant-aware (synthesis A1) are non-optional gates. KL shocks in shim-decision space are direct analogue of P2 (12_...:18-21); pointwise clipping + retention replay required from day one.
- **Token-reduction claim risk (critical)**: Per §5 + nomenclature:161, the very first "shim backdoors reduce token usage" statement requires identical-query before/after token accounting + quality preservation. With synthetic data only, this is L4 until held-out real-usage or road-course transfer proven. Budget-adjusted lift is the only honest primary metric.
- **Minimal viable first smoke (honest floor)**: (1) ShimRegistry + 3-5 hand-audited seed PCS (disclosed as such), (2) micro-SLM route head stub that can select from registry at one SIP, (3) EGGROLL pop search over 8-16 cascade combos on 50 synthetic queries with scalar fitness including -token_cost, (4) one OPSD-style privileged trace replay using extended integrator, (5) floor-tier smoke (import + registry roundtrip + policy forward that emits a shim_id) + explicit "ceiling-tier gap: no real token reduction evidence on held-out" in Carried Debt. Anything claiming more is L4.
- **Data volume estimate (from OPSD patterns)**: OPSD papers + synthesis emphasize dense per-decision signal buys 5-10x sample efficiency vs sparse RL/InfoNCE. Still: hundreds to low-thousands of high-quality (synthetic + filtered) privileged shim traces likely needed before stable MTP lookahead head or PCS promotion. No free lunch on diversity (P5/P6 leakage risks apply directly to shim traces).
- **Overall**: First shim experiments are feasible in Loop 2-3 **as scoped research slices with explicit L1 disclosures** (scaffolds labeled, no UI/API surfacing, no "reduces tokens" claim without the exact accounting). The BHS 100 gate + adversarial Tier B will correctly block any overclaim. The real gating item is not code volume — it is constructing and proving the first non-hand-crafted PrivilegedShimTrace dataset + repeatable EGGROLL fitness that survives the full rubric (quant, rollback, transfer, budget-adjusted).

This addendum is intentionally narrow. It does not claim a training run or a working shim backdoor. It supplies the precise mappings and pseudocode so a fresh Loop 2 agent can begin architecture without re-auditing the priors. All gaps are named.

*End of addendum. Update STEERING_CHELATION_RAGDAG_MICROSLM_RESEARCH_PLAN.md + BHS rubric with these definitions and gates in the next synthesis cycle.*