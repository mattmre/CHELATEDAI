# 15: Shim Concepts Mapping — First Cross-Thread Rigorous Draft (Loop 1)

**Agent**: Agent Sub-5 (Shim) per updated 00_kickoff_brief.md:43  
**Date**: 2026-05 (post-nomenclature release)  
**Status**: Analysis + mapping only. No production code, no stubs, no new classes. Feeds 02_substrate_audit.md (Shim Substrate Readiness subsection) and 10_master_synthesis. BHS-enforced: every claim cites file:line; promotion requires full evidence per STEERING_CHELATION_BHS_RESEARCH_RUBRIC.md and nomenclature 160-164,177-182.

**Cross-References (mandatory reads)**:  
- shim_nodes_mtp_lookahead_nomenclature.md (full; primitives at 36-110, integration table 113-126, open Qs 167-174)  
- 00_kickoff_brief.md:8-20 (required reading + updated Q1-Q3 naming SIPs, insert-once, MTP heads)  
- 01_literature_deep_dive_starter.md:42-60 (Shim Concepts section with SAE-RSV/LogicRAG/MTP mappings)  
- STEERING_CHELATION_RAGDAG_MICROSLM_RESEARCH_PLAN.md:51-56,61 (SE-RDAG + shim elevation in Loop 2)  
- Core surfaces: feature_direction_bank.py:16-78, tts_pipeline.py:27-129+213-227, self_healing_chelation.py:22-35+287-349, antigravity_engine.py:528-593, computational_storage_poc/block_graph.py:35-95 + mock_array.py:55-67, steering_policy.py:39-62.

---

## 1. Pain Points in Current Steering/Chelation That Shims Target (Grounded)

1. **Ephemeral-only corrections with no registration or versioning** (nomenclature 39 explicitly calls this out):  
   `SteeringSignal` (tts_pipeline.py:27-31: direction/strength/source only) is appended to `_signals` (39-41), cleared on every feature_event path (216-222: `self._steerer.clear_signals()`), and summed with clamp in `steer()` (63-74). No identity, no provenance ledger entry per insertion, no "insert-once" contract that survives across steps or DAG expansions. Result: every inference re-derives the same correction; no compounding or backdoor learning.

2. **Masking / point additive deltas only; no structured directional overrides or tiered escalation** (nomenclature 19-29,53):  
   `_chelate_toxicity` (antigravity_engine.py:539-551) computes per-dim variance and returns a 0/1 mask applied as `q_vec * mask` (589). `VectorSteerer.steer()` adds scaled directions but never registers a "shim node" that carries cascade metadata or tier (ST-k per nomenclature 69-71). No mechanism for Order-0/1/k escalation or meta-shims.

3. **No composition, lookahead, or usage-driven refinement** (nomenclature 61-92):  
   FeatureDirectionBank (feature_direction_bank.py:32-52) is a pure lookup (get_direction + overrides dict); zero telemetry for activation_count / success_rate / token_cost_delta / compounding_frequency. No MTP-style prediction of "next shims" (contrast MTP ref in docs/llm-architecture-ai-engineering-adaptation-review-2026-04-27.md:82 "speculative retrieval" P2 and nomenclature 79-84). Recurring high-utility patterns still pay full retrieval + steering cost.

4. **SelfEditDirective surface is adapter-only; no shim proposal path** (self_healing_chelation.py:299-349):  
   `generate_directives` emits only "seal_implication_sft", "seal_retrieval_ttt", "eggroll_low_rank_self_edit" (no `shim_directive` variant proposed at nomenclature 119). Directives carry optimization_params for adapter scope only (305-310); no registration into a Shim Registry or SE-RDAG mutation proposal.

5. **Block-graph payloads and drive-node racing have no shim payload contract** (block_graph.py:35-44: only layer_matrices; mock_array.py:55-67: only probability_branches of nodes):  
   Speculative dispatch exists for ES populations / route candidates but cannot carry or execute a "Precomputed Shim (PCS)" vector or small cascade as O(1) near-data insertion (nomenclature 72-76,122).

6. **LogicRAG-style dynamic DAGs still pay full decomposition/retrieval for patterns that could become backdoors** (01_literature...md:11-13,21-23):  
   Even with chelation variance annotations (proposed at 16-19), there is no "shim + minimal verification" pathway (nomenclature 96) or MTP Shim Lookahead to turn expensive branches into learned cascades.

These are L2/L4/L5/L8 per BHS (escape conditionals absent, partial surfaces presented as complete actuators, untested production paths for DAG mutation).

---

## 2. Best Existing Host Components for First Shim Implementation (Ranked by SIP Proximity)

Per nomenclature 51-58 and 152 (Loop 1 audit mandate):

**Tier-1 Hosts (implement first; direct SIPs)**:
- `FeatureDirectionBank` (feature_direction_bank.py:16-78) + `VectorSteerer` (tts_pipeline.py:33-129): Registry extension lives here. `get_direction` + `update_from_activation` already support Gaussian seeds + SAE overrides — exact seed material for Shim Vectors (nomenclature 117). `steer()` is the insertion execution site; minimal delta: add `insert_shim_once(shim_id)` path that respects insert-once (131) and records to provenance.
- `SelfHealingChelationPlanner.generate_directives` + `SelfEditDirective` (self_healing_chelation.py:287-349,22-35): Natural generator of `shim_directive` proposals (nomenclature 119). Ledger (174-214) already does provenance + quant gates; extend for shim utility ledger.

**Tier-2 Hosts (trigger + dispatch surfaces)**:
- `AntigravityEngine.get_chelated_vector` + `_chelate_toxicity` (antigravity_engine.py:553-593,528-551): Chelation variance (539) becomes first-class "consider shim insertion" decision surface (nomenclature 53,121). Post-embed SIP.
- `ArraySimulation.speculative_multipath_racing` + block_graph payload builders (mock_array.py:55-67; block_graph.py:35-44,74-95): PCS and small cascades compile to block payloads for drive-node execution (nomenclature 57,122). Existing sharded_population_evaluation (79+) is direct analog for shim candidate scoring.

**Tier-3 (policy / training surface)**:
- `ModelScopeSteeringPolicy` + rules (steering_policy.py:39-62): Extend `SteeringRule` or add `ShimSelectionRule`; shadow/active modes map to advisory MTP lookahead (nomenclature 135).
- OPSD/EGGROLL paths (self_healing + evolution_strategies_optimizer; OPSD Loop 01 10_synthesis): Privileged traces now include successful shim cascades (nomenclature 123,55 in plan).

**Rejected hosts for v1**: Full micro-SLM training (deferred per kickoff anti-goals and nomenclature 125); base model weight mutation (scope lock in plan 71).

---

## 3. Tier S/A Candidate Shim Patterns (Pseudocode Sketches — Analysis Only)

**S1 (Tier S): Static Precomputed Shim (PCS) Seeded from Feature Bank + SAE Refinement (addresses pain 1+3; host: FeatureDirectionBank + VectorSteerer)**  
Minimal viable: register a versioned shim from existing bank + optional SAE row; insert-once at steer() or post-chelation SIP when variance exceeds threshold.  
Pseudocode sketch (not code):
```
# In extended FeatureDirectionBank / new ShimRegistry (analysis)
def register_precomputed_shim(shim_id: str, feature_seeds: list[str], sae_override: Optional[np.ndarray] = None, metadata: dict):
    vectors = [bank.get_direction(fid) for fid in feature_seeds]
    if sae_override: vectors.append(normalize(sae_override))
    # version + provenance hash
    registry[shim_id] = {"vectors": vectors, "tier": 0, "version": v, "usage": {}}

# In VectorSteerer (or SIP wrapper)
def insert_shim_once(self, shim_id: str, context: np.ndarray) -> Tuple[np.ndarray, dict]:
    if shim_id in self._inserted_this_pass: return v, {"skipped": "insert-once"}
    shim = registry.lookup(shim_id)
    delta = sum(shim.vectors) * shim.strength   # or gated
    if norm(delta) > max: delta = clamp...
    self._inserted_this_pass.add(shim_id)
    record_to_ledger(shim_id, outcome=None)  # filled post-eval
    return v + delta, {"shim_inserted": shim_id, "tier": 0}
```
Evidence surface needed: before/after token delta + NDCG on held-out noisy queries (BHS rubric route metrics). Risk: just another additive adapter (L3 if no distinct registry/ledger).

**A1 (Tier A): Dynamic Usage-Refined Shim (URS) via OPSD on Cascade Traces (addresses pain 3+4; host: SelfEdit + OPSD surfaces)**  
After N activations, update vectors/cascade partners from telemetry (nomenclature 85-92). Use successful cascades as privileged OPSD targets.  
Pseudocode sketch:
```
def refine_urs_from_outcome(shim_id, activation_context, downstream_fitness_delta, token_cost_delta, cascade_partners):
    entry = ledger[shim_id]
    entry.activation_count += 1
    entry.success_rate = ema(entry.success_rate, 1.0 if fitness_delta > 0 else 0)
    entry.avg_token_delta = ema(...)
    if entry.activation_count > N and entry.success_rate > thresh:
        # OPSD privileged trace: (context, shim_cascade) -> high fitness
        privileged_traces.append((context, [shim_id] + cascade_partners))
    if low_utility: demote_or_prune(shim_id)
```
Ties to SAE-RSV (lit 32-41): refined directions feed URS. Evidence: retention on legacy cases + token-accounted lift (plan 74, nomenclature 161).

**S2 (Tier S): MTP Shim Lookahead Head for Compounding Cascades (addresses pain 2+6; host: micro-SLM route policy + existing speculative surfaces)**  
Tiny auxiliary head (or micro-SLM extension) predicts next 1-N shims given current shim + chelation sig + DAG state (nomenclature 79-84,148; MTP analogy from arch review 82,400). Advisory only (135).  
Pseudocode sketch (training/inference separation):
```
# Inference (route policy forward)
active_shim = ...
chelation_sig = ...
dag_state = ...
predicted_next = mtp_head.predict_next_shims(active_shim, chelation_sig, dag_state, k=3)  # advisory
for p in predicted_next:
    if budget_allows and policy_score(p) > gate:
        insert_shim_once(p)  # may trigger further lookahead

# Training objective (Loop 3-4 per nomenclature 154)
loss = opsd_asymmetric( privileged_successful_cascade_traces, student_proposals )
      + route_cohesion(cascade_depth, fanout)
      + mtp_lookahead_hit_rate( predicted, actual_high_utility_next )
```
Host for dispatch: mock_array speculative racing extended to shim branches. Evidence: MTP hit rate on held-out usage traces + cascade boundedness (nomenclature 162).

**A2 (Tier A): Block-Graph Compiled Drive Shim for Near-Data Insertion (addresses pain 5; host: computational_storage_poc)**  
Compile small PCS vector or 1-2 step cascade into block payload; dispatch via drive node for speculative execution hidden behind other work.  
Pseudocode sketch:
```
# Compiler extension (analysis only)
def compile_shim_cascade_to_blocks(shim_vectors: list[np.ndarray], next_dispatch: int) -> bytes:
    # pack as 512x512 float16 matrices (or vector slices) + pointer
    return build_graph_payload([v.reshape(...) for v in shim_vectors] + [control_block])

# Dispatch (ArraySimulation extension)
def race_shim_candidates(shim_candidates, query_vec):
    branches = [compile_shim_cascade_to_blocks(c.vectors, ...) for c in shim_candidates]
    latency = speculative_multipath_racing(branches)  # existing 55-67
    # winner shim vector returned for insertion at SIP
```
Evidence: software parity + latency model vs pure CPU insertion (rubric drive-node section).

---

## 4. Risks and BHS Mitigations (Explicit, Non-Negotiable)

- **Cascade explosion / unbounded fan-out** (nomenclature 132, plan 78 "Over-fragmentation"): Mitigate with hard max_depth + budget-aware collection (reuse adaptive overlay). Must report depth/fan-out in every artifact card (nomenclature 162). Failure mode: route cohesion collapse.
- **No runtime evidence exists** (nomenclature 179 "No code yet implements"; 177-182 full BHS requirement): All patterns above are hypotheses. Any "token reduction via shim backdoors" claim requires identical-query before/after accounting + quality gates (161). Empty "Brutal Honesty" sections in future PRs are L13 violations.
- **Quantization / boundedness survival** (nomenclature 136): Shim Vectors must obey same INT8/BoundedAdapter floor as corrections. Evidence: quant survival delta on route metrics (rubric).
- **Rollback / provenance gaps** (nomenclature 137): Every insertion affecting result must survive replay on fresh checkout. Current SelfEdit ledger is adapter-only; shim ledger must integrate without duplicating.
- **Scope violation into full agent harness** (plan 81): Shims are retrieval + early reasoning substrate only. Any claim involving long-horizon planning is explicit rejection.
- **L4/L5 presentation risk**: Treating nomenclature table (113-126) or these sketches as "implemented" or "ready" without 02_audit + evidence chain is forbidden (kickoff anti-goals 21-24, success gate 40).

**Deferred Scope (per BHS conventions referenced in CLAUDE.md)**: Full micro-SLM MTP head training, real drive-node hardware shim dispatch, multi-month usage-refined backdoor emergence — all Loop 6+.

---

**Brutal Honesty on This Mapping Itself**: This is the first cross-thread artifact produced under the explicit shim nomenclature mandate. It names concrete hosts (file:line) and patterns with sketches, but contains zero runtime evidence, zero new artifacts surviving checkout, and zero Tier B review. It directly discharges the "shim substrate readiness" requirement added to 00_kickoff_brief.md:24 and nomenclature 152. Next required: Agent Sub-5 (or equivalent) output incorporated into 02_substrate_audit.md + 03_pain_point... + 10_master... with BHS score update. No pattern here is promoted.

*End of 15_shim_concepts_mapping.md (Loop 1 shim slice — Agent 1 complete for nomenclature integration).*
