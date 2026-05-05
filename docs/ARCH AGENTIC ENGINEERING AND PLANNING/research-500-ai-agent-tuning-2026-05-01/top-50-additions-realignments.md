# Top 50 Additions And Realignments

Date: 2026-05-01

This ranking is seeded from the first 100 analyzed sources and should be revised after each additional 100-item report. The current bias is toward changes that fit the repo's existing Engine-Scope and Model-Scope work without requiring unsafe base-model mutation.

| Rank | Addition or realignment | Primary repo surface | Source families | Posture |
| ---: | --- | --- | --- | --- |
| 1 | Unified episode evidence schema for engine, model, retrieval, tool, and evaluator events | `model_scope_artifacts.py`, `engine_scope.py` | Reflexion, DSPy, RAGAS, PRM | adopt now |
| 2 | Comparator-first promotion contract for all self-improvement outputs | `expectation_comparator.py`, `run_*_campaign.py` | Self-Refine, SELF, DPO, PRM | adopt now |
| 3 | Adaptive test-time compute scheduler | new `compute_budget_policy.py` | ToT, Self-Consistency, TTS scaling | adopt now |
| 4 | Multi-verifier evaluator fabric with agreement scores | new `evaluator_fabric.py` | G-Eval, Prometheus, PRM, MT-Bench | adopt now |
| 5 | Engine-Scope route/action telemetry rows | `engine_scope.py` | MRKL, RouteLLM, Toolformer | adopt now |
| 6 | Retrieval intervention gate covering expand, rerank, correct, abstain | learned gates | Self-RAG, CRAG, FLARE, RAGAS | adopt now |
| 7 | Feature scorecard for SAE and activation features | `model_scope_features.py` | Monosemanticity, SAE scaling, Qwen-Scope, SAE fragility papers | adopt now |
| 8 | Tiny-model hook and activation-patching fixture | `model_hook_bus.py` | IOI, induction heads, TransformerLens | adopt now |
| 9 | Hard-negative miner from active blocker signatures | `engine_scope_negatives.py` | Qwen-Scope pattern, CRAG, STaR | adopt now |
| 10 | Reflection buffer as candidate evidence, not persistent memory | `model_scope_memory.py` | Reflexion, Self-Refine | adopt now |
| 11 | Replay bundle builder for every promoted artifact | `model_scope_artifacts.py` | PRM, SWE-bench, AlphaCode | adopt now |
| 12 | Evaluator provenance and evaluator drift tracking | campaign reports | G-Eval, Prometheus, MT-Bench | adopt now |
| 13 | Trace-level grading for full tool/model/retrieval trajectories | evaluator fabric | AgentOps, OpenAI trace grading, AgentEvals | adopt now |
| 14 | Durable checkpoint/resume primitives for campaign runners | `checkpoint_manager.py`, `run_*_campaign.py` | LangGraph durable execution, AutoGen runtime | adopt now |
| 15 | Reward-overoptimization guardrail with heldout evaluator divergence | evaluator fabric | RM scaling laws, RewardBench, uncertainty RLHF | adopt now |
| 16 | Synthetic-data provenance and anti-recursion guard | campaign artifacts | Self-Instruct, Self-Rewarding LM, Curse of Recursion, MAD | adopt now |
| 17 | OpenAI-compatible local runtime boundary with capability records | model runtime | vLLM, SGLang, llama.cpp, Ollama | adopt now |
| 18 | LoRA/QLoRA/PEFT adapter artifact manifest | `model_scope_artifacts.py`, `model_scope_trainer.py` | LoRA, QLoRA, PEFT | adopt now |
| 19 | RAG faithfulness matrix: correctness, support, groundedness, attribution | diagnostics | ARES, RAGTruth, RAGChecker, attribution papers | adopt now |
| 20 | OpenTelemetry-compatible internal trace schema | diagnostics/artifacts | OTel GenAI, Phoenix, OpenInference, AgentOps | adopt now |
| 21 | Tool-capability manifest and allowlisted MCP-shaped tool contracts | tool runtime | MCP, CaMeL, Instruction Hierarchy, OWASP | adopt now |
| 22 | Artifact cards and dataset datasheets generated from campaign evidence | docs/artifacts | Model Cards, Datasheets, System Cards, NIST RMF | adopt now |
| 23 | Eval contamination audit and private holdout flags | eval manifests | contamination papers, Dynabench | adopt now |
| 24 | Process-step scoring for multi-step campaigns | evaluator fabric | PRM800K, process/outcome feedback | adopt now |
| 25 | Plan/no-plan gate for agentic escalation | `recursive_decomposer.py`, campaign runners | Learning When to Plan, AgentTTS | shadow prototype |
| 26 | Runtime benchmark matrix for replay-heavy, RAG-heavy, adapter-heavy workloads | model runtime | vLLM, SGLang, llama.cpp, Ollama | adopt now |
| 27 | Tool retrieval layer to reduce prompt/tool overload | new `tool_retrieval_gate.py` | Toolformer, Gorilla, tool retrieval paper | shadow prototype |
| 28 | Cost-aware model routing | new `model_route_gate.py` | FrugalGPT, RouteLLM | shadow prototype |
| 29 | Best-of-N plus verifier selector mode | campaign runners | verifier papers, AlphaCode | shadow prototype |
| 30 | Sequential self-refine mode with hard stop budget | campaign runners | Self-Refine, Reflexion | adopt now |
| 31 | Query expansion portfolio: HyDE, Query2doc, RAG-Fusion | `query_reformulator.py` | HyDE, Query2doc, RAG-Fusion | shadow prototype |
| 32 | Late-interaction reranker option for hard queries | retrieval backend | ColBERTv2, RankGPT | prototype |
| 33 | Hybrid sparse+dense retrieval mode | retrieval backend | SPLADE, E5, BGE, Qdrant hybrid | prototype |
| 34 | Branch-and-prune pathway search | `research_pathway_analyzer.py` | ToT, GoT, LATS, AFlow | shadow prototype |
| 35 | Selective retrieval trigger from uncertainty | retrieval policy | FLARE, Self-RAG, Rowen | shadow prototype |
| 36 | Corrective retrieval branch when evaluator flags low support | retrieval policy | CRAG, RARR | adopt now |
| 37 | Citation-backed verify-and-edit postprocessor | answer/research path | Verify-and-Edit, RARR, WebGPT, SAFE | adopt now |
| 38 | Conflict replay suite for internal-prior vs external-evidence clashes | eval harness | ClashEval, FreshQA | adopt now |
| 39 | Graph/evidence memory prototype for replay bundles | `model_scope_memory.py` | GraphRAG, LightRAG, HippoRAG, KAG | prototype |
| 40 | Hierarchical memory index | `model_scope_memory.py` | RAPTOR, MemGPT, LongRAG | prototype |
| 41 | Preference-loss plugin interface | `model_scope_trainer.py` | DPO, KTO, ORPO, SimPO, IPO | overlay only |
| 42 | Adapter/LoRA-only promotion lane for preference tuning | `model_scope_trainer.py` | DPO, ORPO, SimPO, QLoRA | overlay only |
| 43 | Policy critique pass before high-risk actions | `steering_policy.py` | Constitutional AI, Deliberative Alignment | adopt now |
| 44 | Safety regression set for activation steering and agent actions | `model_scope_steering.py`, `run_safety_testbed.py` | Rogue Scalpel, jailbreak papers, AgentHarm, HarmBench | adopt now |
| 45 | Shadow-mode steering provenance record | `model_scope_steering.py` | Representation Engineering, ActAdd, ITI | adopt now |
| 46 | Feature-level intervention off switch and rollback metadata | steering artifacts | RepE, Rogue Scalpel, model editing failures | adopt now |
| 47 | Qwen-Scope small-target smoke-test adapter before 9B benchmark | `qwen_scope_adapter.py` | Qwen-Scope model cards | adopt now |
| 48 | Prompt/program optimization artifact with parentage and rollback | campaign artifacts | OPRO, PromptBreeder, AutoPDL, AFlow | prototype |
| 49 | Reproducible dev/eval service stack | docker/dev tooling | Docker Compose, Phoenix, Qdrant, pgvector | adopt now |
| 50 | Base-weight mutation prohibition until overlay gates prove durable | architecture docs/config | Model-Scope roadmap, model editing failures, steering risks | enforce now |
