# Source Ledger 401-500

Date: 2026-05-01

| ID | Title | Year | Source | Mechanism | ChelatedAI mapping | Critique / risk | Improvement posture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 401 | HarmBench | 2024 | https://arxiv.org/abs/2402.04249 | harmful-behavior red-team suite | safety campaign | benchmark overfit | red-team lane |
| 402 | AgentHarm | 2024 | https://arxiv.org/abs/2410.09024 | harmful agent tasks | tool guardrails | scoring hard | agent evals |
| 403 | CyberSecEval 2 | 2024 | https://arxiv.org/abs/2404.13161 | prompt injection/interpreter abuse | code/tool safety | dual-use ambiguity | adopt with FRR |
| 404 | JailbreakBench | 2024 | https://jailbreakbench.github.io/ | reproducible jailbreaks | regression suite | attacks stale | continuous benchmark |
| 405 | StrongREJECT | 2024 | https://arxiv.org/abs/2402.10260 | refusal eval | refusal/utility score | over-refusal | pair with utility |
| 406 | AdvBench | 2023 | https://arxiv.org/abs/2307.15043 | suffix jailbreaks | attack corpus | attack-specific | seed mutations |
| 407 | PAIR | 2023 | https://arxiv.org/abs/2310.08419 | iterative jailbreak agent | red-team agent | collusion | sandbox |
| 408 | TAP | 2023 | https://arxiv.org/abs/2312.02119 | tree-search jailbreaks | attack search | query cost | shadow |
| 409 | CoP Agentic Red-Teaming | 2025 | https://arxiv.org/abs/2506.00781 | jailbreak principle composition | safety principle library | abuse synthesis | restricted lab |
| 410 | SafeDialBench | 2025 | https://arxiv.org/abs/2502.11090 | multi-turn jailbreak benchmark | multi-turn safety | auto-judge | benchmark |
| 411 | Prompt Injection Benchmarking | 2023 | https://arxiv.org/abs/2311.16119 | indirect prompt injection | retrieval/tool boundary | changing attacks | adopt |
| 412 | StruQ | 2024 | https://arxiv.org/abs/2402.06363 | structured data/instruction separation | prompt isolation | formatting brittle | prototype |
| 413 | Instruction Hierarchy | 2024 | https://arxiv.org/abs/2404.13208 | privilege levels | policy stack | enforcement needed | concept |
| 414 | CaMeL | 2025 | https://arxiv.org/abs/2503.18813 | capability-based secure agents | tool permissions | complexity | security architecture |
| 415 | Rebuff | active | https://github.com/protectai/rebuff | prompt-injection detection | pre-tool filter | evasion | compare |
| 416 | Lakera Gandalf | active | https://gandalf.lakera.ai/ | jailbreak challenge | attack mining | skew | reference |
| 417 | OWASP Top 10 for LLM Apps | 2025 | https://owasp.org/www-project-top-10-for-large-language-model-applications/ | LLM app risk taxonomy | risk checklist | broad | governance |
| 418 | MITRE ATLAS | active | https://atlas.mitre.org/ | AI attack taxonomy | incident taxonomy | enterprise skew | map telemetry |
| 419 | NIST AI RMF | 2023 | https://www.nist.gov/itl/ai-risk-management-framework | govern/map/measure/manage | promotion governance | high-level | risk spine |
| 420 | NIST GenAI Profile AI 600-1 | 2024 | https://doi.org/10.6028/NIST.AI.600-1 | genAI risk actions | safety controls | not agent-specific | checklist |
| 421 | ISO/IEC 42001 | 2023 | https://www.iso.org/standard/81230.html | AI management system | release governance | overhead | defer |
| 422 | EU AI Act | 2024 | https://artificialintelligenceact.eu/ | risk-tiered obligations | deployment classification | jurisdiction | monitor |
| 423 | Anthropic Responsible Scaling Policy | 2023 | https://www.anthropic.com/news/anthropics-responsible-scaling-policy | capability thresholds | promotion gates | vendor-specific | adapt |
| 424 | Google DeepMind Frontier Safety Framework | 2024 | https://deepmind.google/discover/blog/introducing-the-frontier-safety-framework/ | critical capability levels | dangerous evals | frontier skew | adapt |
| 425 | OpenAI Preparedness Framework v2 | 2025 | https://openai.com/index/updating-our-preparedness-framework/ | catastrophic risk categories | risk gates | shifting categories | risk-card format |
| 426 | OpenAI Operator System Card | 2025 | https://openai.com/index/operator-system-card/ | computer-use red-teaming | agent guardrails | product-specific | template |
| 427 | OpenAI Deep Research System Card | 2025 | https://openai.com/research/deep-research-system-card/ | browsing-agent safety | research monitoring | contamination risk | template |
| 428 | OpenAI o3/o4-mini System Card | 2025 | https://openai.com/index/o3-o4-mini-system-card/ | reasoning/tool safety evals | tool-capable scoring | opaque | compare |
| 429 | Model Cards | 2019 | https://arxiv.org/abs/1810.03993 | model reporting | adapter manifests | incomplete metadata | artifact card |
| 430 | Datasheets for Datasets | 2021 | https://arxiv.org/abs/1803.09010 | dataset provenance | eval/train ledger | labor | adopt |
| 431 | System Cards | 2022 | https://arxiv.org/abs/2206.03216 | system transparency | release cards | selective disclosure | internal cards |
| 432 | Safety Cases for Frontier AI | 2025 | https://arxiv.org/abs/2503.04744 | structured safety argument | release evidence | weak arguments | release gate |
| 433 | Model Evaluation for AI Governance | 2023 | https://arxiv.org/abs/2305.15324 | governance evals | policy-linked benchmarks | eval lag | reference |
| 434 | Inspect | active | https://inspect.aisi.org.uk/ | eval solvers/scorers | safety harness | migration cost | evaluate |
| 435 | HELM | 2022 | https://arxiv.org/abs/2211.09110 | holistic evals | scorecards | broad | reference |
| 436 | HELM Safety | 2024 | https://crfm.stanford.edu/helm/safety/ | safety taxonomy | dashboard | coverage gaps | compare |
| 437 | Dynabench | 2021 | https://arxiv.org/abs/2108.05286 | dynamic adversarial data | living eval queue | annotation cost | prototype |
| 438 | Data Contamination Quiz | 2023 | https://arxiv.org/abs/2311.06233 | memorization detection | contamination audits | imperfect | adopt |
| 439 | Benchmark Data Contamination of LLMs | 2023 | https://arxiv.org/abs/2311.09783 | contamination measurement | eval integrity | uncertain detection | adopt |
| 440 | RE-Bench | 2024 | https://metr.org/blog/2024-11-22-evaluating-frontier-models-for-dangerous-capabilities/ | autonomous AI R&D eval | self-improvement risk suite | expensive | shadow |
| 441 | SWE-bench Verified | 2024 | https://www.swebench.com/ | verified coding fixes | utility sidecar | expensive | benchmark |
| 442 | METR Autonomy Evaluations | 2024 | https://metr.org/ | long-horizon capability evals | autonomy thresholds | private details | monitor |
| 443 | Apollo Scheming Evaluations | 2024 | https://www.apolloresearch.ai/ | deception probes | monitoring tests | artificial | canaries |
| 444 | Uncertainty-Based Abstention in LLMs | 2024 | https://arxiv.org/abs/2404.10960 | uncertainty abstention | answer gate | drift | prototype |
| 445 | Conformal Abstention | 2024 | https://arxiv.org/abs/2405.01563 | conformal bounds | threshold gate | sampling cost | benchmark |
| 446 | Semantic Entropy | 2024 | https://www.nature.com/articles/s41586-024-07421-0 | meaning-level uncertainty | hallucination monitor | sample cost | selective |
| 447 | Semantic Entropy Probes | 2024 | https://arxiv.org/abs/2406.15927 | hidden-state hallucination probe | model-scope monitor | model access | prototype |
| 448 | Abstain-QA | 2024 | https://arxiv.org/abs/2407.16221 | abstention benchmark | abstain testbed | QA-centric | subset |
| 449 | Don't Hallucinate, Abstain | 2024 | https://arxiv.org/abs/2402.00367 | knowledge-gap detection | evaluator ensemble | cost | shadow |
| 450 | I-CALM | 2026 | https://arxiv.org/abs/2604.03904 | confidence-aware abstention reward | abstention RM | new | research |
| 451 | Model Context Protocol Spec | 2024 | https://modelcontextprotocol.io/specification/2024-11-05/index | JSON-RPC tool lifecycle | tool boundary | host auth | adopt with allowlist |
| 452 | MCP Tools | 2024 | https://modelcontextprotocol.io/specification/2024-11-05/server/tools | `tools/list` and `tools/call` | tool registry | injection/exfiltration | gated prototype |
| 453 | MCP Prompts | 2024 | https://modelcontextprotocol.info/specification/2024-11-05/server/prompts/ | discoverable prompt templates | prompt assets | embedded injection | validation |
| 454 | OpenInference MCP Tracing | 2025 | https://www.arize.com/docs/phoenix/integrations/model-context-protocol/mcp-tracing | tool trace propagation | campaign artifacts | instrumentation | adopt |
| 455 | OpenTelemetry GenAI Agent Spans | active | https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/ | agent/tool spans | logger compatibility | evolving spec | adapter |
| 456 | OpenTelemetry GenAI Metrics | active | https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/ | token/latency metrics | runtime matrix | churn | export |
| 457 | OpenTelemetry GenAI Events | active | https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/ | inference/eval events | evaluator rows | unstable | shadow |
| 458 | OpenInference Spec | active | https://arize-ai.github.io/openinference/ | LLM/retrieval/tool tracing | span schema | overlap | map to OTEL |
| 459 | LangGraph Durable Execution | active | https://docs.langchain.com/oss/python/langgraph/durable-execution | checkpoint graph runs | campaign resume | framework lock-in | copy pattern |
| 460 | AutoGen v0.4 | 2025 | https://www.microsoft.com/en-us/research/blog/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/ | layered agent framework | evaluator harness | API churn | compare |
| 461 | AutoGen AgentChat | active | https://microsoft.github.io/autogen/0.4.0/index.html | multi-agent API | panel/evaluator agents | hidden evidence | prototype |
| 462 | CrewAI Flows | active | https://docs.crewai.com/en/introduction | stateful flows | orchestration templates | product claims | compare |
| 463 | LlamaIndex Workflows | active | https://docs.llamaindex.ai/en/stable/workflows/ | event/step workflow | RAG/eval runner | ecosystem tie | prototype |
| 464 | Phoenix AI Observability | active | https://arize.com/docs/phoenix | traces/evals/datasets | local dashboard | service dependency | optional |
| 465 | Phoenix Datasets and Experiments | active | https://arize.com/docs/phoenix/datasets-and-experiments/how-to-experiments/using-evaluators | dataset evaluators | golden-set runner | judge drift | fixed seeds |
| 466 | MLflow 3 GenAI | 2025 | https://mlflow.org/docs/latest/genai/mlflow-3 | traces/evals/prompts/agents | experiment registry | heavy | selective |
| 467 | MLflow Eval Datasets | active | https://mlflow.org/docs/latest/genai/datasets/ | golden datasets | road-course fixtures | overhead | pattern |
| 468 | MLflow Prompt Evaluation | active | https://www.mlflow.org/docs/3.5.0/genai/prompt-registry/evaluate-prompts/ | prompt registry/eval | prompt versions | complexity | prototype |
| 469 | MLflow Model Registry | active | https://www.mlflow.org/docs/3.0.1/model-registry | lineage/versions/tags | adapter registry | alias misuse | manifest subset |
| 470 | Hugging Face Hub Model Cards | active | https://huggingface.co/docs/hub/en/model-cards | artifact metadata | adapter card | drift | template |
| 471 | Hugging Face Hub Versioned Repos | active | https://huggingface.co/docs/hub/index | versioned repos | artifact publishing | privacy | optional |
| 472 | lakeFS Reproducibility | active | https://docs.lakefs.io/latest/understand/use_cases/reproducibility/ | data commits/tags | dataset snapshots | ops burden | pattern |
| 473 | DVC | active | https://dvc.org/doc | data/pipeline versioning | benchmark data | lockfile churn | large data |
| 474 | W&B Artifacts | active | https://docs.wandb.ai/guides/artifacts/ | artifact lineage | external lineage | SaaS | optional |
| 475 | Inspect AI | 2024 | https://doi.org/10.5281/zenodo.18434279 | reproducible evals | agent benchmark harness | dependency surface | benchmark |
| 476 | OpenAI Evals | active | https://github.com/openai/evals | eval registry | private regression | provider coupling | adapt concepts |
| 477 | DeepEval | active | https://deepeval.com/docs/introduction | pytest-style LLM tests | CI smoke evals | judge reliability | adopt smoke |
| 478 | promptfoo | active | https://www.promptfoo.dev/docs/intro/ | prompt/RAG/red-team matrix | prompt gates | YAML sprawl | adopt |
| 479 | Ragas | active | https://docs.ragas.io/en/stable/references/evaluate/ | RAG metrics | retrieval diagnostics | judge sensitivity | one evaluator |
| 480 | NVIDIA RAG Blueprint Evals | active | https://docs.nvidia.com/rag/latest/evaluate.html | RAG metric bundle | metric reference | vendor bias | reference |
| 481 | AgentBench | 2023 | https://arxiv.org/abs/2308.03688 | interactive environments | agent baseline | generic | benchmark |
| 482 | tau-bench | 2024 | https://arxiv.org/abs/2406.12045 | tool/user/policy interaction | policy suite | simulated users | scenario style |
| 483 | AgentHarm | 2024 | https://arxiv.org/abs/2410.09024 | harmful tool tasks | safety injection | dual-use content | gated |
| 484 | SWE-bench Verified | 2024 | https://www.swebench.com/ | real issue repair | coding regression | cost | subset |
| 485 | SWE-agent | 2024 | https://arxiv.org/abs/2405.15793 | agent-computer interface | coding-agent pattern | overfit | compare |
| 486 | OSWorld | 2024 | https://arxiv.org/abs/2404.07972 | real computer tasks | UI/tool benchmark | heavy | defer |
| 487 | MTEB v2/MMTEB | 2025 | https://embeddings-benchmark.github.io/mteb/ | embedding benchmark | retrieval gate | contamination | selected tasks |
| 488 | BEIR | 2021 | https://arxiv.org/abs/2104.08663 | heterogeneous retrieval | baseline | partly used | control |
| 489 | BRIGHT | 2024 | https://arxiv.org/abs/2407.12883 | reasoning retrieval | hard queries | small ecosystem | pilot |
| 490 | MIRACL | 2022 | https://arxiv.org/abs/2210.09984 | multilingual retrieval | cross-lingual distillation | variance | prototype |
| 491 | Qdrant Hybrid Search | active | https://qdrant.tech/articles/sparse-vectors/ | dense+sparse fusion | retrieval backend | fusion tuning | configurable |
| 492 | Qdrant Quantization | active | https://qdrant.tech/documentation/manage-data/quantization/ | vector compression | memory/perf | quality loss | benchmark |
| 493 | Weaviate Hybrid Search | active | https://docs.weaviate.io/weaviate/search/hybrid | BM25+vector fusion | backend candidate | service complexity | compare |
| 494 | LanceDB | active | https://docs.lancedb.com/ | vector/full-text/SQL table | corpus/artifact store | newer | prototype |
| 495 | Milvus Hybrid Search | active | https://milvus.io/docs | sparse+dense indexing | scale backend | ops footprint | benchmark only |
| 496 | pgvector | active | https://github.com/pgvector/pgvector | Postgres vectors | local backend | scale limits | fallback |
| 497 | Ray Serve LLM | active | https://docs.ray.io/en/latest/serve/llm/ | deployment graphs | cluster serving | overhead | defer |
| 498 | BentoML | active | https://docs.bentoml.com/ | model packaging/services | service packaging | layer overhead | optional |
| 499 | Docker Compose Watch/Develop | active | https://docs.docker.com/compose/ | reproducible service stack | one-command dev stack | version mismatch | adopt |
| 500 | Nix Flakes | active | https://nixos.wiki/wiki/Flakes | pinned dev shells/build graph | benchmark environment | Windows ergonomics | optional |
