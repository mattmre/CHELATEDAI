# Source Ledger 001-100

Date: 2026-05-01

This ledger captures the first analyzed block. Items 001-050 were collected by the model/runtime tuning agent. Items 051-100 were collected by the agent/process tuning agent and renumbered from its original 251-350 slice.

| ID | Title | Year | Source | Mechanism | ChelatedAI mapping | Critique / risk | Improvement posture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | Constitutional AI: Harmlessness from AI Feedback | 2022 | https://arxiv.org/abs/2212.08073 | critique-revision and RLAIF from principles | `steering_policy.py`, safety gates | principle set can encode blind spots | policy-conditioned self-correction |
| 002 | Deliberative Alignment: Reasoning Enables Safer Language Models | 2024 | https://openai.com/index/deliberative-alignment/ | explicit policy reasoning before answer | `steering_policy.py`, `antigravity_engine.py` | hidden reasoning hard to audit | policy-aware pre-answer gate |
| 003 | OpenAI o1 System Card | 2024 | https://openai.com/index/openai-o1-system-card/ | eval-gated deployment and risk tiers | `run_*_campaign.py` | vendor metrics not fully reproducible | campaign scorecards |
| 004 | Reflexion | 2023 | https://arxiv.org/abs/2303.11366 | episodic verbal feedback memory | memory loop, retry supervisor | can amplify wrong lessons | self-improvement loop |
| 005 | Self-Refine | 2023 | https://arxiv.org/abs/2303.17651 | generate-feedback-refine loop | `run_*_campaign.py` | latency and biased self-judgment | bounded refinement passes |
| 006 | SELF: Self-Evolution with Language Feedback | 2023 | https://arxiv.org/abs/2310.00533 | iterative self-generated feedback data | learned gates, tuning campaigns | synthetic drift | offline improvement campaigns |
| 007 | Voyager | 2023 | https://arxiv.org/abs/2305.16291 | skill library and curriculum | `research_pathway_analyzer.py` | sandbox transfer risk | reusable pathway memory |
| 008 | Generative Agents | 2023 | https://arxiv.org/abs/2304.03442 | memory stream, reflection, planning | model memory | believability not task quality | memory consolidation |
| 009 | ReAct | 2022 | https://arxiv.org/abs/2210.03629 | reason/action/observation loop | `antigravity_engine.py` | tool errors poison reasoning | baseline agent loop |
| 010 | Tree of Thoughts | 2023 | https://arxiv.org/abs/2305.10601 | search over reasoning branches | `research_pathway_analyzer.py` | branch explosion | branch and prune planner |
| 011 | Graph of Thoughts | 2023 | https://arxiv.org/abs/2308.09687 | graph-structured thought transformations | pathway analyzer | orchestration complexity | DAG candidate transformations |
| 012 | Least-to-Most Prompting | 2022 | https://arxiv.org/abs/2205.10625 | decomposition before solution | task planner | bad decomposition cascades | campaign planner primitive |
| 013 | Automatic Prompt Optimization | 2023 | https://arxiv.org/abs/2305.03495 | prompt search with feedback | campaign tuning | eval overfit | prompt tuning campaign mode |
| 014 | DSPy | 2023 | https://arxiv.org/abs/2310.03714 | metric-driven pipeline compilation | `run_*_campaign.py` | framework lock-in | metric-driven module tuning |
| 015 | TextGrad | 2024 | https://arxiv.org/abs/2406.07496 | textual gradients | campaign optimizer | judge instability | iterative prompt/code tuning |
| 016 | AgentBench | 2023 | https://arxiv.org/abs/2308.03688 | multi-environment agent evals | campaign eval | benchmark mismatch | agent eval taxonomy |
| 017 | SWE-bench | 2023 | https://arxiv.org/abs/2310.06770 | real issue resolution benchmark | coding eval harness | costly sandboxing | external coding benchmark |
| 018 | SWE-agent | 2024 | https://arxiv.org/abs/2405.15793 | agent-computer interface | execution loop | interface dominates ability | action-space design |
| 019 | WebArena | 2023 | https://arxiv.org/abs/2307.13854 | realistic web task benchmark | browser/runtime eval | UI brittleness | web adaptation eval |
| 020 | ToolLLM / ToolBench | 2023 | https://arxiv.org/abs/2307.16789 | tool-use tuning/eval | tool gating | API hallucination | tool-selection supervision |
| 021 | Gorilla | 2023 | https://arxiv.org/abs/2305.15334 | API retrieval and invocation | tool routing | API drift | tool-call grounding |
| 022 | MRKL Systems | 2022 | https://arxiv.org/abs/2205.00445 | modular expert router | model/tool gates | router failure point | neuro-symbolic routing |
| 023 | FrugalGPT | 2023 | https://arxiv.org/abs/2305.05176 | model cascade | model routing | poor uncertainty estimates | cost-aware cascade |
| 024 | RouteLLM | 2024 | https://arxiv.org/abs/2406.18665 | learned strong/weak router | learned gates | preference transfer risk | model routing gate |
| 025 | Mixture-of-Agents | 2024 | https://arxiv.org/abs/2406.04692 | layered multi-agent aggregation | campaign ensembles | correlated failures | hard-task ensemble |
| 026 | Speculative Decoding | 2023 | https://arxiv.org/abs/2211.17192 | draft and verifier model | runtime adaptation | mostly latency | faster campaigns |
| 027 | Medusa | 2024 | https://arxiv.org/abs/2401.10774 | multi-token prediction heads | local inference | model modification required | speedup path |
| 028 | Retrieval-Augmented Generation | 2020 | https://arxiv.org/abs/2005.11401 | parametric plus nonparametric memory | retrieval layer | retrieval bottleneck | foundational RAG |
| 029 | HyDE | 2022 | https://arxiv.org/abs/2212.10496 | hypothetical answer retrieval | query reformulation | hallucinated pseudo-docs | optional reformulation |
| 030 | Query2doc | 2023 | https://arxiv.org/abs/2303.07678 | pseudo-doc query expansion | retrieval optimizer | precision loss | recall-heavy expansion |
| 031 | RAG-Fusion | 2024 | https://github.com/Raudaschl/rag-fusion | multi-query RRF | retrieval optimizer | latency | eval-gated fusion |
| 032 | Corrective RAG | 2024 | https://arxiv.org/abs/2401.15884 | retrieval quality evaluator | learned retrieval gate | fallback noise | adaptive retrieval correction |
| 033 | Self-RAG | 2023 | https://arxiv.org/abs/2310.11511 | retrieve and critique tokens | retrieval gate | training burden | control-token pattern |
| 034 | FLARE | 2023 | https://arxiv.org/abs/2305.06983 | uncertainty-triggered retrieval | runtime retrieval | weak uncertainty proxy | just-in-time retrieval |
| 035 | RAPTOR | 2024 | https://arxiv.org/abs/2401.18059 | hierarchical retrieval summaries | memory index | compression loss | hierarchical memory |
| 036 | ColBERTv2 | 2021 | https://arxiv.org/abs/2112.01488 | late-interaction retrieval | retrieval core | heavier infra | precision upgrade |
| 037 | SPLADE v2 | 2021 | https://arxiv.org/abs/2109.10086 | sparse lexical expansion | hybrid retrieval | larger indexes | hybrid retrieval |
| 038 | BGE Embeddings | 2023 | https://arxiv.org/abs/2309.07597 | embedding/reranker models | embedding backend | version drift | embedding baseline |
| 039 | E5 Text Embeddings | 2022 | https://arxiv.org/abs/2212.03533 | contrastive embeddings | embedding layer | prefix discipline | retrieval baseline |
| 040 | RankGPT | 2023 | https://arxiv.org/abs/2304.09542 | listwise LLM reranking | rerank gate | expensive and biased | hard-query reranking |
| 041 | Direct Preference Optimization | 2023 | https://arxiv.org/abs/2305.18290 | PPO-free preference loss | model tuning | preference quality | offline tuning |
| 042 | KTO | 2024 | https://arxiv.org/abs/2402.01306 | binary feedback optimization | preference tuning | noisy polarity | simple feedback tuning |
| 043 | ORPO | 2024 | https://arxiv.org/abs/2403.07691 | SFT plus preference objective | tuning pipeline | task variance | simple alignment stage |
| 044 | IPO | 2023 | https://arxiv.org/abs/2310.12036 | regularized preference objective | evaluator | theory-heavy | loss-selection guidance |
| 045 | Distilling Step-by-Step | 2023 | https://arxiv.org/abs/2305.02301 | rationale distillation | local scope models | rationale quality | distill agent traces |
| 046 | Orca | 2023 | https://arxiv.org/abs/2306.02707 | explanation-trace imitation | campaign distillation | trace contamination | scope-model training |
| 047 | Toolformer | 2023 | https://arxiv.org/abs/2302.04761 | self-supervised tool calls | tool gates | filtering burden | tool-use supervision |
| 048 | Representation Engineering | 2023 | https://arxiv.org/abs/2310.01405 | activation directions | steering policy | brittle steering | prototype hook steering |
| 049 | Activation Steering work | 2023 | https://arxiv.org/abs/2310.01405 | inference-time latent control | steering policy | fluency/safety degradation | cautious steering hooks |
| 050 | The Rogue Scalpel | 2025 | https://arxiv.org/abs/2509.22067 | steering safety failure modes | safety instrumentation | recent preprint | steering regression tests |
| 051 | ReAct | 2022 | https://arxiv.org/abs/2210.03629 | reason/action/observation | planner-action layer | brittle actions | tool-use engine |
| 052 | Toolformer | 2023 | https://arxiv.org/abs/2302.04761 | self-supervised API call learning | tool policy | affordance quality | tool-use learning |
| 053 | Reflexion | 2023 | https://arxiv.org/abs/2303.11366 | failure reflection memory | retry supervisor | wrong lessons | self-correction |
| 054 | Tree of Thoughts | 2023 | https://arxiv.org/abs/2305.10601 | reasoning search | branch evaluator | cost | test-time search |
| 055 | Graph of Thoughts | 2023 | https://arxiv.org/abs/2308.09687 | graph reasoning | DAG planner | complexity | candidate transformation graph |
| 056 | Self-Refine | 2023 | https://arxiv.org/abs/2303.17651 | critique-revise loop | revise layer | shallow critique | cheap integration |
| 057 | Self-Consistency | 2022 | https://arxiv.org/abs/2203.11171 | multiple chains plus consensus | N-sample consensus | linear cost | baseline consensus |
| 058 | Least-to-Most | 2022 | https://arxiv.org/abs/2205.10625 | ordered subtasks | decomposition planner | cascade errors | task planner |
| 059 | Plan-and-Solve | 2023 | https://arxiv.org/abs/2305.04091 | explicit plan prepass | planning layer | performative plans | planning gate |
| 060 | HuggingGPT | 2023 | https://arxiv.org/abs/2303.17580 | expert model coordination | model/tool router | dependency sprawl | router |
| 061 | Voyager | 2023 | https://arxiv.org/abs/2305.16291 | lifelong skill library | skill memory | domain transfer | skill accumulation |
| 062 | Generative Agents | 2023 | https://arxiv.org/abs/2304.03442 | memory stream | episodic memory | not task-quality proof | memory schema |
| 063 | AutoGen | 2023 | https://arxiv.org/abs/2308.08155 | conversable multi-agent programming | role orchestration | loop guardrails | multi-agent architecture |
| 064 | CAMEL | 2023 | https://arxiv.org/abs/2303.17760 | role-playing agents | role separation | role drift | worker/evaluator split |
| 065 | MetaGPT | 2023 | https://arxiv.org/abs/2308.00352 | SOP-driven agent workflow | handoff contracts | heavy workflow | process templates |
| 066 | AgentVerse | 2023 | https://arxiv.org/abs/2308.10848 | agent topology experiments | topology analyzer | benchmark sensitivity | topology testing |
| 067 | ChatDev | 2023 | https://arxiv.org/abs/2307.07924 | software-dev role agents | code task pipeline | demo-heavy | workflow inspiration |
| 068 | SWE-agent | 2024 | https://arxiv.org/abs/2405.15793 | agent-computer interface | shell/editor action layer | sandbox rigor | coding-agent engine |
| 069 | Impact of Tool Retrieval on LLM Agents | 2024 | https://arxiv.org/abs/2402.13431 | dynamic tool retrieval | tool selection layer | retrieval misses | tool prompt reduction |
| 070 | Gorilla | 2023 | https://arxiv.org/abs/2305.15334 | API retrieval | tool schema ranking | API drift | grounded tool call |
| 071 | MRKL Systems | 2022 | https://arxiv.org/abs/2205.00445 | modular routing | expert router | early eval | routing |
| 072 | Constitutional AI | 2022 | https://arxiv.org/abs/2212.08073 | RLAIF critique | policy critique | bias | synthetic preferences |
| 073 | Helpful and Harmless Assistant with RLHF | 2022 | https://arxiv.org/abs/2204.05862 | preference model plus RL | reward model | reward hacking | alignment reference |
| 074 | InstructGPT | 2022 | https://arxiv.org/abs/2203.02155 | SFT/RM/PPO | alignment pipeline | expensive | foundation |
| 075 | Direct Preference Optimization | 2023 | https://arxiv.org/abs/2305.18290 | preference loss | tuning campaign | data quality | high-value offline tuning |
| 076 | SLiC-HF | 2023 | https://arxiv.org/abs/2305.10425 | sequence likelihood calibration | preference tuning | narrower than RLHF | lightweight preference tuning |
| 077 | RRHF | 2023 | https://arxiv.org/abs/2304.05302 | rank-based alignment | ranking loss | signal granularity | rank preference loss |
| 078 | IPO | 2023 | https://arxiv.org/abs/2310.12036 | preference regularization | tuning objective | assumptions | alternate loss |
| 079 | KTO | 2024 | https://arxiv.org/abs/2402.01306 | binary feedback | unpaired eval feedback | polarity calibration | scarce-pair tuning |
| 080 | ORPO | 2024 | https://arxiv.org/abs/2403.07691 | odds-ratio preference objective | SFT alignment | complex prefs | simple tuning |
| 081 | SimPO | 2024 | https://arxiv.org/abs/2405.14734 | reference-free preference loss | memory-cheap tuning | margin sensitivity | local preference tuning |
| 082 | RLAIF | 2023 | https://arxiv.org/abs/2309.00267 | AI-generated feedback | evaluator-label pipeline | evaluator bias | synthetic reward data |
| 083 | Deep RL from Human Preferences | 2017 | https://arxiv.org/abs/1706.03741 | pairwise reward learning | reward model | non-LLM | foundation |
| 084 | InstructGPT duplicate reference | 2022 | https://arxiv.org/abs/2203.02155 | instruction following from feedback | alignment baseline | cost | foundation |
| 085 | Let's Verify Step by Step | 2023 | https://arxiv.org/abs/2305.20050 | process reward models | step evaluator | step labels needed | process supervision |
| 086 | Training Verifiers to Solve Math Word Problems | 2021 | https://arxiv.org/abs/2110.14168 | verifier selects solution | verifier module | math domain | verifier pattern |
| 087 | Large Language Monkeys | 2024 | https://arxiv.org/abs/2407.21787 | repeated sampling | compute scheduler | diminishing returns | repeated-sample budget |
| 088 | Scaling Test-Time Compute Optimally | 2024 | https://arxiv.org/abs/2408.03314 | adaptive inference compute | compute policy | difficulty estimator | dynamic compute |
| 089 | Scaling LLM Test-Time Compute Optimally | 2024 | https://arxiv.org/abs/2408.03314 | best-of-N vs refinement | inference optimizer | duplicate family | budget optimizer |
| 090 | Quiet-STaR | 2024 | https://arxiv.org/abs/2403.09629 | internal rationale generation | hidden reasoning | training complexity | later model training |
| 091 | STaR | 2022 | https://arxiv.org/abs/2203.14465 | bootstrap rationales | curriculum from traces | spurious rationales | trace curriculum |
| 092 | AlphaZero | 2017 | https://doi.org/10.1038/nature24270 | search plus self-play | planning analogy | non-language | search-value blueprint |
| 093 | AlphaCode | 2022 | https://doi.org/10.1126/science.abq1158 | massive sampling/filtering | candidate pool | compute-heavy | candidate rerank |
| 094 | AlphaCode 2 Technical Report | 2023 | https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf | sampling plus search/ranking | code-agent evaluator | contest-specific | code candidate ranking |
| 095 | G-Eval | 2023 | https://arxiv.org/abs/2303.16634 | LLM-as-judge form scoring | evaluator model | prompt sensitivity | rubric scoring |
| 096 | MT-Bench / Chatbot Arena | 2023 | https://arxiv.org/abs/2306.05685 | pairwise evaluation | comparative harness | style bias | pairwise eval |
| 097 | Judging LLM-as-a-Judge | 2023 | https://arxiv.org/abs/2306.05685 | evaluator agreement | eval calibration | same-model favoritism | judge calibration |
| 098 | Prometheus | 2023 | https://arxiv.org/abs/2310.08491 | rubric-trained evaluator | local evaluator | rubric limits | rubric eval |
| 099 | Prometheus 2 | 2024 | https://arxiv.org/abs/2405.01535 | direct and pairwise evaluator | local judge | judge bias | evaluator option |
| 100 | RAGAS | 2023 | https://arxiv.org/abs/2309.15217 | reference-free RAG metrics | retrieval monitoring | RAG-specific | RAG eval |
