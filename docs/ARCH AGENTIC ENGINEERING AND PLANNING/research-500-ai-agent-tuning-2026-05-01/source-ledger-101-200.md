# Source Ledger 101-200

Date: 2026-05-01

| ID | Title | Year | Source | Mechanism | ChelatedAI mapping | Critique / risk | Improvement posture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | Toy Models of Superposition | 2022 | https://www.anthropic.com/research/toy-models-of-superposition | sparse features in superposition | feature-collision model for masks | toy setting | theory baseline |
| 102 | Interpretability in the Wild: IOI in GPT-2 Small | 2022 | https://arxiv.org/abs/2211.00593 | causal circuit/path patching | `model_hook_bus.py` validation | narrow task | hook test |
| 103 | Causal Scrubbing | 2022 | https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing | hypothesis-level interventions | comparator for mechanistic claims | setup-heavy | feature-claim testing |
| 104 | A Mathematical Framework for Transformer Circuits | 2021 | https://transformer-circuits.pub/2021/framework/index.html | residual stream/circuit formalism | hook schema discipline | early/small models | architecture reference |
| 105 | In-Context Learning and Induction Heads | 2022 | https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html | induction-head circuit | activation-patching benchmark | limited behavior class | diagnostics |
| 106 | Towards Automated Circuit Discovery | 2023 | https://arxiv.org/abs/2304.14997 | ACDC edge pruning | causal component mining | metric-sensitive | prototype |
| 107 | Finding Neurons in a Haystack | 2023 | https://arxiv.org/abs/2305.01610 | knowledge-neuron localization | feature-to-behavior index | neuron units too coarse | compare vs SAE |
| 108 | Locating and Editing Factual Associations in GPT | 2022 | https://arxiv.org/abs/2202.05262 | causal tracing and ROME | reversible intervention template | factual scope only | tracing pattern |
| 109 | Mass Editing Memory in a Transformer | 2022 | https://arxiv.org/abs/2210.07229 | MEMIT batched edits | multi-feature promotion/rollback | persistence risk | overlay only |
| 110 | Discovering Latent Knowledge in Language Models Without Supervision | 2022 | https://arxiv.org/abs/2212.03827 | unsupervised probes | latent readout comparator | probe instability | shadow evaluator |
| 111 | Inference-Time Intervention | 2023 | https://arxiv.org/abs/2306.03341 | head-level truth vectors | truth steering | truth/helpfulness tradeoff | bounded steering |
| 112 | Activation Addition | 2023 | https://arxiv.org/abs/2308.10248 | prompt-derived steering vectors | reversible steering | off-manifold risk | hook-gated prototype |
| 113 | Steering Llama 2 via Contrastive Activation Addition | 2023 | https://arxiv.org/abs/2312.06681 | contrastive residual vectors | vector registry | dataset artifacts | prototype |
| 114 | Representation Engineering | 2023 | https://arxiv.org/abs/2310.01405 | read/control representations | model-scope control plane | coarse directions | strategic anchor |
| 115 | Towards Monosemanticity | 2023 | https://transformer-circuits.pub/2023/monosemantic-features/index.html | SAE dictionary features | feature UI | small model first | feature browser |
| 116 | Sparse Autoencoders Find Highly Interpretable Features | 2023 | https://arxiv.org/abs/2309.08600 | L1 SAE decomposition | `model_scope_features.py` | reconstruction hides error | SAE baseline |
| 117 | Scaling Monosemanticity | 2024 | https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html | large SAE features | feature store/labels | proprietary model | design reference |
| 118 | Gemma Scope | 2024 | https://huggingface.co/google/gemma-scope-2b-pt-res | open Gemma SAEs | feature extraction tests | model-specific | fixture |
| 119 | Open Source Sparse Autoencoders for all Residual Stream Layers of GPT-2 Small | 2024 | https://arxiv.org/abs/2406.04093 | layerwise SAE suite | hook regression fixture | GPT-2 scale | testbed |
| 120 | Scaling and Evaluating Sparse Autoencoders | 2024 | https://arxiv.org/abs/2406.04093 | SAE eval methods | SAE scorecard | proxy metrics | scorecard |
| 121 | Qwen-Scope: SAE-Res-Qwen3-8B-Base | 2026 | https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 | Qwen residual SAEs | Qwen adapter target | license/version drift | integrate first |
| 122 | Qwen-Scope: SAE-Res-Qwen3.5-2B-Base | 2026 | https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100 | small Qwen SAE | cheap local tests | very new | smoke-test |
| 123 | Qwen-Scope: SAE-Res-Qwen3.5-9B-Base | 2026 | https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100 | mid-size Qwen SAE | production-like feature UI | hardware cost | benchmark later |
| 124 | Qwen-Scope: SAE-Res-Qwen3-30B-A3B | 2026 | https://huggingface.co/Qwen/SAE-Res-Qwen3-30B-A3B-Base-W128K-L0_100 | MoE Qwen SAE | MoE feature/routing | expensive | defer |
| 125 | TransformerLens | active | https://transformerlensorg.github.io/TransformerLens/ | hook/cache/patching APIs | hook bus design | API churn | pattern source |
| 126 | TransformerLens Activation Patching | active | https://jbloomaus.github.io/TransformerLens/generated/code/transformer_lens.patching.html | clean/corrupt patch API | causal runner | task setup bias | mirror API |
| 127 | TransformerLens Hook Points | active | https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.hook_points.html | forward/backward hooks | `model_hook_bus.py` | framework-specific | naming convention |
| 128 | Tuned Lens | 2023 | https://arxiv.org/abs/2303.08112 | layerwise translators to logits | intermediate diagnostics | lens can mislead | read-only probes |
| 129 | Logit Lens | 2020+ | https://transformer-circuits.pub/2021/framework/index.html | residual-to-logit readout | activation summaries | correlation not causation | pair with patching |
| 130 | Eliciting Latent Predictions with Tuned Lens | 2023 | https://arxiv.org/abs/2303.08112 | calibrated lens probes | observability | trained artifact | diagnostic |
| 131 | Causal Abstraction | 2025 | https://www.jmlr.org/papers/v26/23-0058.html | formal patching/steering/SAE abstraction | promotion vocabulary | theory-heavy | criteria vocabulary |
| 132 | Distributed Alignment Search | 2023 | https://arxiv.org/abs/2303.02536 | causal abstraction search | feature-task alignment | compute-heavy | research prototype |
| 133 | The Hydra Effect | 2023 | https://arxiv.org/abs/2307.15771 | self-repair after ablation | rollback tests | interventions masked | compensation checks |
| 134 | Does Circuit Analysis Interpretability Scale? | 2023 | https://arxiv.org/abs/2307.09458 | circuit scalability stress test | claim limits | pessimistic evidence | risk memo input |
| 135 | Universal and Transferable Adversarial Attacks on Aligned LMs | 2023 | https://arxiv.org/abs/2307.15043 | jailbreak suffixes | safety eval set | attacks evolve | red-team suite |
| 136 | Jailbreaking Black Box LLMs in Twenty Queries | 2023 | https://arxiv.org/abs/2310.08419 | adaptive attack search | steering safety tests | benchmark gaming | adversarial replay |
| 137 | The Geometry of Truth | 2023 | https://arxiv.org/abs/2310.06824 | truth directions | truthfulness readout | brittle concept | shadow signal |
| 138 | Truth is Universal | 2024 | https://arxiv.org/abs/2407.12831 | cross-model truth reps | transfer probe | disputed | cross-model probe |
| 139 | Linear Representations of Sentiment in LMs | 2017 | https://arxiv.org/abs/1704.01444 | sentiment neuron/vector | steering precedent | old architecture | context |
| 140 | Knowledge Neurons in Pretrained Transformers | 2021 | https://arxiv.org/abs/2104.08696 | factual neuron attribution | neuron-vs-feature baseline | polysemantic neurons | compare |
| 141 | ROME | 2022 | https://rome.baulab.info/ | rank-one model edit | intervention provenance | persistent mutation | overlay analogy |
| 142 | MEMIT | 2022 | https://memit.baulab.info/ | mass factual edit | batched rollback | catastrophic edits | avoid base mutation |
| 143 | SERAC | 2022 | https://arxiv.org/abs/2203.08406 | external edit memory | reversible edit memory | retrieval mismatch | external overlay |
| 144 | MEND | 2021 | https://arxiv.org/abs/2110.11309 | learned edit updates | edit controller | unsafe persistence | defer |
| 145 | Model Editing at Scale Leads to Gradual and Catastrophic Forgetting | 2024 | https://arxiv.org/abs/2401.07453 | edit accumulation failure | rollback caution | method-specific | risk input |
| 146 | CorrSteer | 2025 | https://arxiv.org/abs/2508.12535 | correlation-selected SAE steering | runtime feature selector | new/unreplicated | shadow experiment |
| 147 | GSAE | 2025 | https://arxiv.org/abs/2512.06655 | graph-regularized SAE controller | multi-feature safety steering | preprint | watchlist |
| 148 | Mechanistic Knobs in LLMs | 2026 | https://arxiv.org/abs/2601.02978 | high-order semantic SAE knobs | behavior feature retrieval | very recent | cautious eval |
| 149 | YaPO | 2026 | https://arxiv.org/abs/2601.08441 | learnable sparse steering vectors | reversible steering objective | overfit | later overlay |
| 150 | When the Coffee Feature Activates on Coffins | 2026 | https://arxiv.org/abs/2601.03047 | SAE fragility audit | Qwen-Scope failure tests | negative result | required critique |
| 151 | ARES | 2023 | https://arxiv.org/abs/2311.09476 | synthetic-trained RAG judges | evaluator layer | judge drift | eval harness |
| 152 | RAGTruth | 2024 | https://arxiv.org/abs/2401.00396 | word-level hallucination labels | hallucination detector | schema mismatch | benchmark/training seed |
| 153 | CRAG Benchmark | 2024 | https://arxiv.org/abs/2406.04744 | long-tail factual QA | retrieval stress test | QA bias | external eval |
| 154 | RAGBench and TRACe | 2024 | https://arxiv.org/abs/2407.11005 | explainable RAG labels | RAG scorecard | vendor task mix | metric taxonomy |
| 155 | RAGChecker | 2024 | https://arxiv.org/abs/2408.08067 | fine-grained RAG diagnostics | failed-row diagnosis | metric complexity | failure taxonomy |
| 156 | Lynx / HaluBench | 2024 | https://arxiv.org/abs/2407.08488 | hallucination judge | hallucination gate | false positives | shadow detector |
| 157 | LRP4RAG | 2024 | https://arxiv.org/abs/2408.15533 | relevance propagation | token/context attribution | interpretability cost | audit prototype |
| 158 | ReDeEP | 2024 | https://arxiv.org/abs/2410.11414 | external vs parametric knowledge use | retrieved-evidence compliance | internals dependence | research |
| 159 | HALT-RAG | 2025 | https://arxiv.org/abs/2509.07475 | NLI ensemble and abstention | contradiction gate | NLI brittleness | abstention baseline |
| 160 | Correctness is not Faithfulness in RAG Attributions | 2024 | https://arxiv.org/abs/2412.18004 | citation faithfulness distinction | evidence citation audit | costly labels | metric |
| 161 | Evaluation of Attribution Bias in RAG LLMs | 2024 | https://arxiv.org/abs/2410.12380 | source attribution skew | provenance QA | domain-specific bias | risk test |
| 162 | ClashEval | 2024 | https://arxiv.org/abs/2404.10198 | internal prior vs evidence conflict | retrieval override eval | synthetic conflict | replay suite |
| 163 | MultiHop-RAG | 2024 | https://arxiv.org/abs/2401.15391 | multi-hop evidence | pathway analyzer eval | breadth over precision | graph retrieval tests |
| 164 | RetrievalQA: Assessing Adaptive RAG | 2024 | https://arxiv.org/abs/2402.16457 | retrieval-needed eval | retrieval gate signal | QA bias | adaptive eval |
| 165 | CRUD-RAG | 2024 | https://arxiv.org/abs/2401.17043 | knowledge lifecycle tasks | memory freshness tests | corpus mismatch | task taxonomy |
| 166 | Rowen | 2024 | https://arxiv.org/abs/2402.10612 | uncertainty triggers retrieval | adaptive retrieval gate | perturbation cost | trigger policy |
| 167 | PAIRS | 2025 | https://arxiv.org/abs/2508.04057 | parametric verification before retrieval | retrieval avoidance | confidence error | shadow routing |
| 168 | AIR-RAG | 2026 | https://doi.org/10.1016/j.neucom.2025.132272 | adaptive iterative retrieval | retriever controller | loop latency | hard-query prototype |
| 169 | AdaPCR | 2025 | https://arxiv.org/abs/2507.04069 | passage-combination reranking | evidence bundle selector | combinatorial cost | multi-hop replay |
| 170 | Query Decomposition for RAG | 2025 | https://arxiv.org/abs/2510.18633 | subquery planning | subquery budget | noise | planner gate |
| 171 | AdaCQR | 2024 | https://arxiv.org/abs/2407.01965 | sparse+dense CQR | reformulation gate | conversational bias | training source |
| 172 | Promptagator | 2022 | https://arxiv.org/abs/2209.11755 | synthetic query generation | retrieval campaigns | synthetic drift | data generation |
| 173 | InPars v2 | 2023 | https://arxiv.org/abs/2301.01820 | synthetic query/reranker data | hard-negative generation | filtering needed | evaluator-filtered data |
| 174 | UDAPDR | 2023 | https://arxiv.org/abs/2303.00807 | unsupervised domain adaptation | retriever adaptation | overfit | per-corpus prototype |
| 175 | RankVicuna | 2023 | https://arxiv.org/abs/2309.15088 | open listwise reranker | local rerank option | instruction bias | benchmark |
| 176 | RankZephyr | 2023 | https://arxiv.org/abs/2312.02724 | distilled listwise reranking | cheaper reranker | transfer risk | rerank candidate |
| 177 | LiT5-Distill | 2024 | https://arxiv.org/abs/2402.14838 | listwise distillation | low-latency reranking | stale teacher labels | curated replay only |
| 178 | DRAGON+ | 2022 | https://arxiv.org/abs/2204.06031 | diverse dense retriever supervision | embedding baseline | older | baseline |
| 179 | Contriever | 2021 | https://arxiv.org/abs/2112.09118 | unsupervised dense retrieval | cold-start baseline | weak without adaptation | baseline |
| 180 | Atlas | 2022 | https://arxiv.org/abs/2208.03299 | retrieval-augmented LM training | retrieval-training reference | heavy | defer |
| 181 | RETRO | 2021 | https://arxiv.org/abs/2112.04426 | retrieval during generation | memory-augmented model | requires training | long-term |
| 182 | kNN-LM | 2019 | https://arxiv.org/abs/1911.00172 | nearest-neighbor datastore | token memory concept | scale/latency | reference |
| 183 | MemGPT | 2023 | https://arxiv.org/abs/2310.08560 | virtual context and archival memory | memory eviction/promotion | bad fact persistence | schema ideas |
| 184 | MemoryBank | 2023 | https://arxiv.org/abs/2305.10250 | long-term memory with forgetting | retention scoring | privacy/staleness | heuristics |
| 185 | HippoRAG | 2024 | https://arxiv.org/abs/2405.14831 | graph-inspired retrieval memory | graph evidence store | graph errors | graph replay |
| 186 | GraphRAG | 2024 | https://www.microsoft.com/en-us/research/project/graphrag/ | entity graph and summaries | evidence synthesis | build cost | prototype |
| 187 | LightRAG | 2024 | https://arxiv.org/abs/2410.05779 | lightweight graph retrieval | cheaper GraphRAG | update consistency | evaluate |
| 188 | LongRAG | 2024 | https://arxiv.org/abs/2406.15319 | long-context retrieval units | chunk policy | context cost | benchmark |
| 189 | KAG | 2024 | https://arxiv.org/abs/2409.13731 | KG plus logical forms | structured evidence | ontology burden | evidence design |
| 190 | FActScore | 2023 | https://arxiv.org/abs/2305.14251 | atomic fact verification | claim-level metric | open-domain bias | atomic scorer |
| 191 | TRUE | 2022 | https://arxiv.org/abs/2204.04991 | factual consistency suite | evaluator calibration | pre-RAG | judge sanity |
| 192 | SelfCheckGPT | 2023 | https://arxiv.org/abs/2303.08896 | sampled consistency | hallucination signal | stochastic cost | fallback |
| 193 | SAFE | 2024 | https://arxiv.org/abs/2403.18802 | search-augmented factuality | audit evaluator | web variability | audit mode |
| 194 | FreshLLMs / FreshQA | 2023 | https://arxiv.org/abs/2310.03214 | fresh knowledge eval | freshness replay | ages quickly | replay |
| 195 | SPADE | 2024 | https://arxiv.org/abs/2401.03038 | data quality assertions | pipeline contracts | assertion quality | adopt |
| 196 | OpenTelemetry GenAI Semantic Conventions | 2025 | https://opentelemetry.io/docs/specs/semconv/gen-ai/ | model/tool/retrieval spans | campaign traces | evolving spec | version-pin |
| 197 | OpenTelemetry GenAI Agent Spans | 2025 | https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/ | agent/tool spans | replay trace format | evolving names | internal mapping |
| 198 | TruLens RAG Triad | 2023 | https://www.trulens.org/getting_started/core_concepts/rag_triad/ | relevance/groundedness metrics | dashboard metrics | KB quality dependent | dashboard view |
| 199 | LangSmith Tracing and Evaluation | 2023 | https://docs.smith.langchain.com/ | traces, datasets, evaluators | replay/eval UX | vendor coupling | emulate patterns |
| 200 | Arize Phoenix / OpenInference | 2023 | https://docs.arize.com/phoenix | LLM traces and evals | local observability | overhead | trace export |
