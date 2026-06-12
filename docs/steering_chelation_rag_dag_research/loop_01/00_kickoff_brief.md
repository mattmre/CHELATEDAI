# Loop 1 Kickoff Brief — Steering-Chelation-RAGDAG-MicroSLM Program

**Loop**: 1 (Deep Research & Mapping)  
**Dates**: Program kickoff session onward  
**Orchestrator**: Integration Lead (current Grok session + dispatched agents)  
**Goal**: Produce the master literature + substrate audit synthesis that will rank concrete upgrade patterns for Loops 2-10. Do not build code yet.

## Required Reading (Mandatory for Every Agent in This Loop)
- `STEERING_CHELATION_RAGDAG_MICROSLM_RESEARCH_PLAN.md` (top-level program definition, including shim elevation at lines 51-56 and Loop 2 target at line 61)
- `shim_nodes_mtp_lookahead_nomenclature.md` (full; canonical definitions of Shim Vector, Shim Node, SIP, Shim Cascade, MTP Shim Lookahead (MSL), Usage-Refined Shim, Shim Backdoor, Shim Registry (extension of FeatureDirectionBank), SE-RDAG; integration table at lines 113-126; open questions at lines 167-174; BHS considerations at 160-164)
- `STEERING_CHELATION_BHS_RESEARCH_RUBRIC.md` (route-specific metrics, carried debt re-audit rules)
- `README.md` (lists shim doc as canonical major new primitive at line 17)
- `STEERING_CHELATION_10_LOOP_BHS_PROGRAM.md` (Loop 1 definition at lines 15-16, refined by this brief)

No agent output is valid for Loop 1 synthesis unless it cites the shim nomenclature with file:line and audits shim substrate readiness.

## Three Mandatory Questions This Loop Must Answer
1. **What exactly from the new 2025-2026 literature (LogicRAG dynamic DAGs [arXiv:2508.06105], SAE-RSV steering vector refinement [arXiv:2509.23799], Matryoshka SAEs, MTP [arXiv:2404.19737 and the repo arch review at docs/llm-architecture-ai-engineering-adaptation-review-2026-04-27.md:82], latest OPSD/SDPO, spectral methods, hyperscale ES, plus the shim nomenclature itself) maps *cleanly and non-duplicatively* onto the five existing ChelatedAI substrates (chelation, TTS steering nodes, Model-Scope sparse features, computational-storage block graphs + drive-node racing, EGGROLL/OPSD surfaces) — with explicit treatment of how Shim Vectors / Shim Nodes / MTP Shim Lookahead / SE-RDAG would extend FeatureDirectionBank (lines 16-78 of feature_direction_bank.py), VectorSteerer.steer() + SteeringSignal (tts_pipeline.py:27-80, 213-227), SelfEditDirective (self_healing_chelation.py:22-35), antigravity chelation decision paths (antigravity_engine.py:538-551), block_graph payloads (computational_storage_poc/block_graph.py:35-44), and ModelScopeSteeringPolicy (steering_policy.py:39-62)?**
2. **What are the actual (not aspirational) failure modes and integration seams when we try to make chelation signals drive live DAG mutations and multi-cast reroutes through the existing TTS + Model-Scope actuators — specifically including seams for Shim Insertion Points (SIP per shim_nomenclature.md:51-58), insert-once vs ephemeral additive semantics (contrast SteeringSignal at tts_pipeline.py:27-31 vs registered Shim Vector at shim_nomenclature.md:36-41), cascade bounding, and chelation variance as shim trigger vs current _chelate_toxicity masking?**
3. **What is the minimal viable micro-SLM interface (2-4 GB class, quantization-survivable, tied to legacy weights via adapters/banks) that can be trained with the OPSD + EGGROLL-style regime against reroute fitness — including honest blockers for first smoke of shim selection policy + MTP Shim Lookahead heads (per shim_nomenclature.md:79-84, 125, 148), data provenance for successful shim cascades, and retention of base cases when shim backdoors / SE-RDAG mutations are active?**

## Minimal Deliverables to Close the Loop (BHS-enforced)
- `01_literature_deep_dive.md` (or split by topic): exhaustive but mapped — every cited paper must have a 1-2 paragraph "exact mapping or explicit rejection" to a named file/line or concept in the current repo.
- `02_substrate_audit.md`: ruthless walk through `tts_pipeline.py`, `model_scope_*`, `computational_storage_poc/{block_graph,mock_array,repo_graph_memory,CHELATEDAI_integration_demo}`, `evolution_strategies_optimizer.py`, `self_healing_chelation.py`, `antigravity_engine.py` chelation paths, and the OPSD Loop 01 synthesis. Every seam, every missing hook, every quantization or stability landmine must be called out with file:line. **Must include dedicated "Shim Substrate Readiness" subsection auditing FeatureDirectionBank / VectorSteerer / SIP candidates / SelfEditDirective extension points against shim_nomenclature.md:113-126 and open questions 167-174.**
- `03_pain_point_to_pattern_mapping.md`: table (or multiple) that turns the audited pains into candidate upgrade patterns, each with Tier (S/A/B), pseudocode sketch, primary risk, and primary evidence surface needed.
- `10_master_synthesis_and_prioritization.md`: the single living document that the rest of the program will reference. Contains the ranked Tier S patterns that Loops 2+ will actually implement, the BHS Research Score for the program at close of Loop 1, and the explicit carried-debt re-audit.
- Optional but high-value: `04_microslm_feasibility_probe.md` (data requirements, candidate base models from existing quantized zoo, first training sketch, blocker list).

## Anti-Goals for This Loop (explicit rejection criteria)
- Writing any new production-path code (stubs, adapters, new classes) — analysis and docs only.
- Claiming "we can just bolt LogicRAG on top" without auditing how chelation variance would actually annotate or mutate its DAG nodes.
- Ignoring the existing BHS culture or the carried debt from Sessions 32-34 and OPSD Loop 01.
- Treating the micro-SLM as a full replacement model rather than a learned route-policy head that must preserve compatibility with legacy base cases.

## Parallel Agent Dispatch Plan (recommended)
- Agent Lit-1: LogicRAG + GraphRAG adaptive/DAG papers (focus 2508.06105 + related)
- Agent Lit-2: SAE steering refinement papers (2509.23799 SAE-RSV + AxBench + Matryoshka SAEs + SAIF)
- Agent Lit-3: OPSD/SDPO/MIS-PO 2026 updates + any new hyperscale ES papers post-EGGROLL analysis
- Agent Sub-1: Full TTS + VectorSteerer + feature bank audit (explicitly map to Shim Vector / Shim Node hosting per shim_nomenclature.md:169 and integration table 117-118)
- Agent Sub-2: Model-Scope steering + hook bus + sparse feature reality check (what is actually implemented vs architected) + policy extensions for shim node scoring
- Agent Sub-3: Computational storage graph surfaces (block_graph, mock_array racing, repo_graph_memory, integration demo) + drive node dispatch seams (incl. shim vector / cascade payload compilation candidates)
- Agent Sub-4: EGGROLL optimizer + existing evolution_strategies_optimizer.py + sedimentation/OPSD loss surfaces + OPSD extensions for successful shim cascade traces
- Agent Sub-5 (Shim): Dedicated cross-audit of all SIP candidates (antigravity post-embed, VectorSteerer, RerouteDAG expansion, micro-SLM policy, block-graph dispatch) + first mapping of pain points to Tier S/A shim patterns (output feeds 15_shim_concepts_mapping.md)
- Agent Feas-1: MicroSLM 2-4 GB candidates from current models/ + llama.cpp + training data sketch from existing collapse/road-course traces
- Agent Int-1 (orchestrator): Cross-thread integration + master synthesis

All agents must use the trigger phrases from the rubric and produce brutally honest output with file:line citations into the live codebase.

## Success Gate for Loop Close
The master synthesis must be able to name 2-4 Tier S patterns with enough specificity that a fresh agent in Loop 2 can begin architecture design *without* having to re-do the literature or substrate audit. If the synthesis is vague ("more research needed on X"), Loop 1 has failed its mandate.

**Kickoff artifacts created in initiating session**: This brief + the top-level program plan + rubric. First specialized agent outputs land in `loop_01/` as they complete.

*Brutal honesty on this brief itself*: It is a kickoff scaffold. No experiments run, no new evidence generated under this program name yet. All strength comes from the prior connected work it references.