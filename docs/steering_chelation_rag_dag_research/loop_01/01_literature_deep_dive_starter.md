# Loop 1 Partial: Literature Deep Dive Starter (New Papers + Cross-Mapping)

**Agent**: Integration Lead (initial pull) + dispatched literature agents to expand  
**Date**: Program kickoff session  
**Status**: Starter — contains two high-signal 2025 papers fetched fresh + initial mappings. Not exhaustive. Full version will incorporate OPSD Loop 01 citations + more.

## 1. LogicRAG — You Don't Need Pre-built Graphs for RAG (arXiv:2508.06105, AAAI 2026)

**Core Contribution**:
- Pre-built GraphRAG is expensive (token cost + update latency) and brittle because real queries require *different* logical structures.
- LogicRAG decomposes the query into subproblems at inference time, builds a *dynamic DAG* of logical dependencies among them, topological-sorts for coherent execution order, then prunes redundant retrieval and irrelevant context.
- Achieves better performance + efficiency than static GraphRAG baselines without any pre-constructed knowledge graph.

**Exact Mapping / Non-Duplicative Opportunity for This Program**:
- The dynamic DAG construction at inference time is *almost exactly* the "modified RAG-DAG with live reroutes" vision.
- Chelation variance (local neighborhood noise in embedding space) is a natural *per-subproblem-node signal* that can trigger: (a) spawn parallel speculative branches, (b) invoke a steering node to propose an alternative decomposition or edge, or (c) mark the node for drive-node multi-path racing.
- The topological sort + pruning logic is a perfect place to *insert* TTS-style vector relocation or multi-cast steering interventions *before* the subproblem is handed to retrieval or the micro-SLM.
- **Do not copy**: We do not want to reimplement their decomposition or linearization. We want to *annotate and mutate* their (or an equivalent) DAG using the repo's existing chelation + steering + fitness surfaces.
- Highest-leverage integration point: make the LogicRAG-style subproblem DAG a *first-class citizen* inside an extended `RerouteDAG` that also carries chelation metadata and steering actuator handles on every node/edge.

**Risks from the paper that mirror our own**:
- Decomposition quality determines everything downstream (our chelation signals must not be fooled by bad subproblem boundaries).
- Token budget explosion from multiple branches (our budget-aware collection + pruning policies from adaptive overlay work are directly relevant).

## 2. SAE-RSV — Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement (arXiv:2509.23799)

**Core Contribution**:
- Steering vectors learned from small/limited data are noisy (task-irrelevant features dominate).
- Use a trained SAE to *semantically denoise* (remove irrelevant features) and *augment* (add missing task-relevant features via semantic similarity) the raw steering vector.
- Dramatic empirical gains over raw steering vectors and even SFT in limited-data regimes.

**Exact Mapping / Non-Duplicative Opportunity**:
- This is almost a direct "chelation for steering vectors" paper. Our spectral chelation (variance-based dimension masking + centering) is the embedding-space analog of what they do with SAE features for steering vectors.
- The Model-Scope sparse feature path + `feature_direction_bank.py` + existing steering vectors are the perfect substrate to apply SAE-RSV-style refinement *before* they are used by `VectorSteerer` or ModelScope actuators.
- Matryoshka SAEs (other 2025 work) add hierarchical/nested structure that maps beautifully onto our dimension masking + BoundedAdapter + low-rank work.
- **Do not copy** the specific SAE training or refinement procedure blindly. Adapt the *denoise + semantically complete the direction* idea into our chelation + steering node loop, using our existing topology/isomer/structural health signals as additional supervision.
- Highest-leverage: make every steering signal or route proposal pass through a "refinement gate" that can use SAE (where available for the base) or our own learned mask predictors / dimension banks to clean the proposal before it is cast as a multi-reroute.

**Connection to OPSD Loop 01**:
- The "privileged vs student" distinction in OPSD maps to "clean/privileged steering direction (after SAE refinement or successful reroute) vs noisy on-policy proposal". Asymmetric distillation can train the micro-SLM route head to prefer the refined directions.

## 2.5 Shim Concepts — Early Mapping (Shim Nodes, MTP Lookahead, SE-RDAG)
**Source**: `shim_nodes_mtp_lookahead_nomenclature.md` (full read required per updated 00_kickoff_brief.md:8-15). Introduces Shim Vector (SV), Shim Node (SN), Shim Insertion Point (SIP), Shim Cascade (SC), MTP Shim Lookahead (MSL), Precomputed Shim (PCS), Usage-Refined Shim (URS), Shim Backdoor, Shim Registry (SR as extension of FeatureDirectionBank), and SE-RDAG (evolution of RerouteDAG with first-class shim nodes).

**Relation to Existing Repo Surfaces (file:line grounded)**:
- **FeatureDirectionBank** (feature_direction_bank.py:16-78): Currently provides deterministic SHA-256 Gaussian unit vectors (or SAE decoder overrides via update_from_activation at 42-52) for SteeringSignal construction. Per nomenclature 117 and 40, becomes low-level provider for registered, versioned Shim Vectors (Gaussian seeds + SAE overrides). Gap: no versioning, no cascade metadata, no registry lookup_by_context API.
- **TTS / VectorSteerer + SteeringSignal** (tts_pipeline.py:27-31 for dataclass; 33-129 VectorSteerer; 47-80 steer() applies additive deltas with max_strength clamp; 213-227 in TTSPipeline clears signals per feature_event path): Ephemeral per-inference additive corrections (nomenclature 39 explicitly distinguishes from registered/cascadable Shim Vector). SIP candidate inside steer() or extended ModelScopeShadowSteerer (nomenclature 54). Current from_sparse_feature_event (83-129) is natural host for static shim seeding but lacks insert-once + cascade semantics.
- **SelfEditDirective / self-healing** (self_healing_chelation.py:22-35 dataclass; generate_directives at 287+ produces adapter_sft / eggroll_es / retrieval_ttt variants only): Nomenclature 119 proposes new `shim_directive` variant for proposing insertion/promotion/deprecation of Shim Nodes. No current support (grep in file returns zero shim/steer references).
- **AntigravityEngine chelation paths** (antigravity_engine.py:538-551 _chelate_toxicity computes dim_variance mask; 553-593 get_chelated_vector applies it post-embed): Nomenclature 53,121: high local variance or isomer drift is natural trigger for "shim insertion" action (alongside or instead of classic rerank/mask). Current surface only produces mask; no action surface for registered directional override.
- **Block graph / drive nodes** (computational_storage_poc/block_graph.py:35-44 build_graph_payload / create_block for matrix payloads; mock_array.py:55-67 speculative_multipath_racing): Nomenclature 57,122: Shim Vectors + small cascades can be compiled into block-graph payloads for O(1) speculative dispatch and lookup. Current payload is dense matrices only; no shim vector serialization or cascade dispatch contract.
- **ModelScopeSteeringPolicy** (steering_policy.py:39-62 + SteeringRule): Rules are feature-triggered suppress/scale. Nomenclature 120: policies can now select/score Shim Nodes in addition. Gap: no shim registry integration.
- **OPSD / EGGROLL surfaces** (referenced via self_healing + evolution_strategies_optimizer; OPSD Loop 01 artifacts): Nomenclature 123: successful shim cascades become privileged traces for asymmetric distillation; low-rank pop search over shim combinations. Extends existing "privileged successful reroute traces".

**Relation to Papers Already Discussed**:
- **SAE-RSV (arXiv:2509.23799, this file lines 25-41)**: SAE-RSV denoises/augments raw steering vectors. Shims are a *registered, composable, usage-refined* form of such directions (nomenclature 85-92 URS + 72-76 PCS). Mapping: SAE-refined vectors (or our chelation-denoised equivalents) become the seed material for the first Static/Dynamic Shim Nodes in the Registry. Non-duplicative: SAE-RSV is per-vector cleanup; shims add node identity, cascades, MTP lookahead, and backdoor learning over DAG telemetry.
- **LogicRAG (arXiv:2508.06105, this file lines 7-24)**: Dynamic DAG at inference. Shims provide the *mechanism* for live mutation: a shim node activation (triggered by chelation variance on a subproblem node) can reroute or backdoor without full re-decomposition. SE-RDAG (nomenclature 105-110) is the target substrate where LogicRAG-style nodes can be annotated with shim-augmented edges. MTP Shim Lookahead (nomenclature 79-84) supplies speculative next-shim proposals analogous to LogicRAG pruning but at the level of directional overrides.
- **MTP (arXiv:2404.19737, referenced in docs/llm-architecture-ai-engineering-adaptation-review-2026-04-27.md:24,52,82,400 and nomenclature 80)**: Paper provides multi-token prediction / speculative decoding. The arch review (line 82) already flags it as P2 "speculative retrieval: cheap candidate proposal + exact verification". Shim extension (nomenclature 79-84): MTP-style head on micro-SLM or dedicated lookahead predicts next 1-N Shim Nodes for pre-activation, turning point corrections into compounding cascades (nomenclature 61-68). Speculative shim activation is the direct analog of speculative token decoding but for reasoning primitives / backdoors. Non-duplicative with LogicRAG: MTP lookahead operates on the shim vocabulary inside the route policy, not on token sequences or DAG nodes directly.

**BHS Note (per nomenclature 177-182 and rubric)**: No code implements these yet. All mappings are hypotheses. Any promoted shim pattern requires full evidence chain (token-accounted quality-preserving gains, bounded cascades, rollback provenance).

## 3. Initial Cross-Thread Synthesis (starter)

**Tier S Patterns Emerging (to be stress-tested in full Loop 1)**:

**S1: Chelation-Annotated Dynamic RAG-DAG with Steering-Node Multi-Cast**
- Every subproblem node in a LogicRAG-style DAG carries a chelation variance / structural health vector.
- High noise → steering node (extended TTS) is allowed to emit N low-rank route deltas (EGGROLL-style population) instead of one.
- Cheap evaluation (existing fitness surfaces + possible drive-node dispatch for the candidates) selects which (if any) to commit.
- Micro-SLM (small head) learns the policy "given this chelation signature + sparse features + current DAG state, which reroutes or topology mutations are worth proposing?"
- Training: OPSD on (privileged successful reroute traces) vs (on-policy student proposals), plus route-cohesion auxiliary loss.

**S2: SAE-Refined / Chelation-Denoised Steering Vectors as First-Class Route Actuators**
- All existing steering vectors / feature directions are passed through a refinement stage (SAE where available, or our spectral + mask predictor analog).
- Refined directions become the *vocabulary* from which multi-reroute proposals and new token routes are composed.
- This directly attacks the "noisy neighborhood" problem at the steering level, not just the embedding level.

**S3: Drive-Node Speculative Racing for Reroute Candidate Evaluation**
- The existing `mock_array.py` multi-drive speculative dispatch + block_graph format is extended so that low-rank route delta evaluations or micro-SLM head forward passes (tiny) can be dispatched as block-graph payloads.
- Primary value: hide the latency of "try 8 reroutes" behind parallel storage-node work, exactly as EGGROLL hides ES population cost behind inference-like matmuls.
- Scope: software proof + emulation first; real hardware only for transport contract.

**Rejected or Deferred in Starter (examples)**:
- Full pre-built static GraphRAG ingestion pipeline: rejected — we want inference-time adaptability.
- Training the entire 2-4 GB micro-SLM from scratch in Loop 1: deferred (feasibility probe only).
- Claiming "this will give us hyperscale convergence": analysis only until we have population-based route search actually running with measurable convergence behavior on the DAG.

## 4. Immediate Gaps This Starter Exposes (for next agents)
- We have almost no *actual* SAE surfaces wired for the models we actually run (Qwen-Scope SAEs are referenced in the 2026-05-01 Model-Scope arch doc, but the current `model_scope_features.py` is still largely summary/probe based).
- The TTS `VectorSteerer` currently applies *additive signals*; extending it to true multi-cast *disjoint route proposals* (that can change which downstream nodes/edges are even considered) requires a larger interface change than a simple delta. (Shim-specific: no support for registered/insert-once/cascadable vectors per shim_nomenclature.md:36-41,169.)
- Existing road-course and synthetic collapse fixtures are per-query / per-corpus, not per-DAG-topology. The evaluation harness work in Loop 8 will be substantial. (Shim extension needed: "shim insertion under noise" and "cascade token-efficiency" families per nomenclature 158.)
- Data for OPSD-style training of a route policy: we have collapse logs and fitness traces, but not yet "successful reroute trajectory" traces with privileged context. That data collection surface must be designed in Loop 2-3. (Shim addition: provenance for successful shim cascades + MTP lookahead hit-rate traces.)
- **Shim substrate gaps (new)**: No Shim Registry, no SIP hooks in any production path, no versioning/rollback for directional overrides beyond current adapter ledgers, no MTP-style head interface even sketched. See updated kickoff Q1-Q3 and nomenclature open questions 167-174. The dedicated Agent Sub-5 (Shim) + 15_shim_concepts_mapping.md are required to close this.

**Next for full Loop 1**: Dispatch the specialized literature and substrate agents listed in the (updated) kickoff brief, including Agent Sub-5 (Shim). Their outputs + integration will produce the master `10_master_synthesis_and_prioritization.md` and `15_shim_concepts_mapping.md`.

*This starter is intentionally partial and opinionated to seed the swarm. All claims here are hypotheses to be attacked by the other agents.*