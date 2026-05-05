# Source Ledger 201-300

Date: 2026-05-01

| ID | Title | Year | Source | Mechanism | ChelatedAI mapping | Critique / risk | Improvement posture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 201 | Training Language Models to Follow Instructions with Human Feedback | 2022 | https://arxiv.org/abs/2203.02155 | SFT + RM + PPO | post-training stack | costly, hackable RM | reference pipeline |
| 202 | Learning to Summarize from Human Feedback | 2020 | https://arxiv.org/abs/2009.01325 | preference reward shaping | evidence summaries | narrow transfer | adapt |
| 203 | WebGPT | 2021 | https://arxiv.org/abs/2112.09332 | browse-assisted RLHF | citation rewards | citation gaming | source audits |
| 204 | Helpful and Harmless RLHF | 2022 | https://arxiv.org/abs/2204.05862 | dual preference models | multi-axis alignment | objective conflict | reward cards |
| 205 | Constitutional AI | 2022 | https://arxiv.org/abs/2212.08073 | critique/revision RLAIF | policy constitution | rule brittleness | versioned principles |
| 206 | Scaling Laws for Reward Model Overoptimization | 2022 | https://arxiv.org/abs/2210.10760 | RM overoptimization curves | reward hacking detector | proxy collapse | heldout RM cap |
| 207 | Direct Preference Optimization | 2023 | https://arxiv.org/abs/2305.18290 | classification preference loss | preference tuning | beta sensitivity | offline baseline |
| 208 | RRHF | 2023 | https://arxiv.org/abs/2304.05302 | rank loss | ranking finetune | sample quality | low-resource pilot |
| 209 | SLiC-HF | 2023 | https://arxiv.org/abs/2305.10425 | sequence likelihood calibration | contrastive alignment | pair overfit | diverse negatives |
| 210 | Preference Ranking Optimization | 2023 | https://arxiv.org/abs/2306.17492 | listwise preference objective | richer labels | label cost | ranking mode |
| 211 | IPO | 2023 | https://arxiv.org/abs/2310.12036 | identity preference optimization | DPO alternative | underfit | test alternative |
| 212 | Statistical Rejection Sampling Improves Preference Optimization | 2023 | https://arxiv.org/abs/2309.06657 | rejection-sampled pairs | cleaner synthetic prefs | sampler bias | filter |
| 213 | Nash Learning from Human Feedback | 2024 | https://proceedings.mlr.press/v235/munos24a.html | Nash policy from pairwise prefs | intransitive preferences | complex training | evaluate later |
| 214 | KTO | 2024 | https://arxiv.org/abs/2402.01306 | unary feedback utility | thumbs-up logs | utility assumptions | useful |
| 215 | ORPO | 2024 | https://arxiv.org/abs/2403.07691 | odds-ratio SFT alignment | simple SFT replacement | weak KL control | efficient |
| 216 | SimPO | 2024 | https://arxiv.org/abs/2405.14734 | reference-free preference loss | small-model tuning | margin sensitivity | strong default |
| 217 | CPO for Machine Translation | 2024 | https://arxiv.org/abs/2401.08417 | contrastive avoid-bad tuning | answer-quality deltas | narrow | adapt |
| 218 | Controllable Preference Optimization | 2024 | https://arxiv.org/abs/2402.19085 | score-conditioned multi-objective PO | objective sliders | calibration | prototype |
| 219 | Relative Preference Optimization | 2024 | https://arxiv.org/abs/2402.10958 | cross-prompt contrast matrix | related case mining | embedding dependence | mine related cases |
| 220 | DPO with an Offset | 2024 | https://arxiv.org/abs/2402.10571 | preference-strength margin | severity-weighted training | graded labels | severity labels |
| 221 | Online DPO: Fast-Slow Chasing | 2024 | https://arxiv.org/abs/2406.05534 | continual DPO with LoRA speeds | live adaptation | forgetting | gated continual tuning |
| 222 | Self-Play Preference Optimization | 2024 | https://arxiv.org/abs/2405.00675 | self-play Nash objective | self-improving policy | drift | human anchors |
| 223 | Self-Improving Robust Preference Optimization | 2024 | https://arxiv.org/abs/2406.01660 | min-max offline RLHF | OOD robustness | conservative behavior | stress-test |
| 224 | Chain of Preference Optimization | 2024 | https://arxiv.org/abs/2406.09136 | stepwise ToT preferences | reasoning-process tuning | synthetic bias | verifier labels |
| 225 | Distributional Preference Alignment via Optimal Transport | 2024 | https://arxiv.org/abs/2406.05882 | distributional preference alignment | mixed corpora | batch sensitivity | later |
| 226 | Anchored Preference Optimization | 2024 | https://arxiv.org/abs/2408.06266 | anchors and contrastive revisions | underspecified pair repair | anchor choice | rationale pairs |
| 227 | Self-Consistency Preference Optimization | 2024 | https://arxiv.org/abs/2411.04109 | prefer self-consistent answers | self-training | wrong consensus | verifier veto |
| 228 | FocalPO | 2025 | https://arxiv.org/abs/2501.06645 | focus on rankable pairs | signal efficiency | ambiguity ignored | curriculum |
| 229 | RePO | 2025 | https://arxiv.org/abs/2503.07426 | ReLU preference loss | simple PO | new | benchmark |
| 230 | MaPPO | 2025 | https://arxiv.org/abs/2507.21183 | prior reward knowledge | domain priors | prior dominates | audited priors |
| 231 | PRM800K / Let's Verify Step by Step | 2023 | https://arxiv.org/abs/2305.20050 | process reward labels | step verifier | annotation cost | active label |
| 232 | Process and Outcome Feedback | 2022 | https://arxiv.org/abs/2211.14275 | process vs outcome feedback | evidence chains | math bias | port |
| 233 | Process Reward Models That Think | 2025 | https://arxiv.org/abs/2504.16828 | verbalized verifier PRM | step verifier | hidden rationale | short traces |
| 234 | Generative Verifiers | 2022 | https://arxiv.org/abs/2110.14168 | verifier reranking | best-of-N | compute heavy | high-stakes |
| 235 | Training Verifiers to Solve Math Word Problems | 2021 | https://arxiv.org/abs/2110.14168 | outcome verifier | validation layer | flawed reasoning miss | pair with PRM |
| 236 | RewardBench | 2024 | https://arxiv.org/abs/2403.13787 | RM benchmark | reward model selection | overfitting | private split |
| 237 | How to Evaluate Reward Models for RLHF | 2024 | https://arxiv.org/abs/2410.14872 | RM usefulness tests | RM protocol | costly | proxy plus spot-train |
| 238 | Judging LLM-as-a-Judge | 2023 | https://arxiv.org/abs/2306.05685 | arena/judge evaluation | scalable evaluator | bias | randomized order |
| 239 | Prometheus | 2023 | https://arxiv.org/abs/2310.08491 | rubric evaluator | local judge | rubric leakage | rubric discipline |
| 240 | Prometheus 2 | 2024 | https://arxiv.org/abs/2405.01535 | stronger open judge | local evaluator | bias | calibrate |
| 241 | G-Eval | 2023 | https://arxiv.org/abs/2303.16634 | structured auto-eval | evaluator prompt | prompt sensitivity | freeze prompts |
| 242 | Chatbot Arena | 2024 | https://arxiv.org/abs/2403.04132 | crowdsourced pairwise Elo | preference signal | population bias | cohort segment |
| 243 | AlpacaEval 2 | 2024 | https://arxiv.org/abs/2404.04475 | length-controlled win rate | regression eval | judge artifacts | pair with task evals |
| 244 | Taming Overconfidence in LLMs | 2024 | https://arxiv.org/abs/2410.09724 | reward calibration | confidence-aware output | helpfulness tradeoff | separate abstention |
| 245 | Uncertainty-aware RLHF | 2024 | https://arxiv.org/abs/2410.23726 | RM uncertainty | avoid reward chasing | hard uncertainty | ensembles |
| 246 | Calibrated Language Models Must Hallucinate | 2023 | https://arxiv.org/abs/2311.14648 | calibration limits | abstention realism | theoretical | explicit unknown |
| 247 | Teaching Models to Express Uncertainty | 2022 | https://arxiv.org/abs/2205.14334 | verbal confidence | user-facing confidence | prose mismatch | action mapping |
| 248 | SelfCheckGPT | 2023 | https://arxiv.org/abs/2303.08896 | consistency sampling | uncertainty | false consensus | combine checks |
| 249 | STaR | 2022 | https://arxiv.org/abs/2203.14465 | rationale self-training | reasoning self-training | bad rationales | verifier-filter |
| 250 | RLAIF vs RLHF / AI Feedback for Summarization | 2023 | https://arxiv.org/abs/2309.00267 | AI preference labels | cheaper labels | judge bias | human calibration |
| 251 | AgentOps | 2024 | https://arxiv.org/abs/2411.05285 | lifecycle trace taxonomy | campaign traces | taxonomy only | trace schema |
| 252 | LangGraph Durable Execution | active | https://docs.langchain.com/oss/python/langgraph/durable-execution | checkpoint/resume workflows | campaign runners | framework coupling | durability primitives |
| 253 | OpenAI Agents SDK Tracing | 2025 | https://openai.github.io/openai-agents-python/tracing/ | workflow/span tracing | replay logs | vendor backend | local exporter |
| 254 | OpenAI Trace Grading | 2025 | https://platform.openai.com/docs/guides/trace-grading | graders over traces | regression suite | judge drift | trace-level evals |
| 255 | LangChain AgentEvals | 2025 | https://docs.langchain.com/langsmith/trajectory-evals | trajectory evals | tool-call replay | brittle references | deterministic trajectory tests |
| 256 | AutoGen v0.4 Runtime | 2025 | https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/ | async stateful runtime | AEP orchestration | churn | reference |
| 257 | Magentic-One | 2024 | https://arxiv.org/abs/2411.04468 | orchestrator/specialists | `aep_orchestrator.py` | benchmark-specific | pattern |
| 258 | AutoGenBench | 2024 | https://www.microsoft.com/en-us/research/publication/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/ | repeated isolated agent evals | replay harness | AutoGen tie | side-effect eval |
| 259 | CrewAI Flows | active | https://docs.crewai.com/ | flows, guardrails, memory | campaign flow DSL | product abstraction | inspiration |
| 260 | AFlow | 2024 | https://arxiv.org/abs/2410.10762 | MCTS workflow generation | pathway analyzer | benchmark overfit | prototype |
| 261 | Multi-Agent Collaboration via Evolving Orchestration | 2025 | https://arxiv.org/abs/2505.19591 | RL-trained orchestrator | adaptive orchestrator | instability | learn later |
| 262 | Beyond the Strongest LLM | 2025 | https://arxiv.org/abs/2509.23537 | propose/vote consensus | consensus mode | herding | guarded mode |
| 263 | AgentsNet | 2025 | https://arxiv.org/abs/2507.08616 | graph coordination tasks | topology analyzer | synthetic | stress tests |
| 264 | MultiAgentBench | 2025 | https://arxiv.org/abs/2503.01935 | collaboration/competition eval | multi-agent metrics | representativeness | collab metrics |
| 265 | Revisiting Multi-Agent Debate as Test-Time Scaling | 2025 | https://arxiv.org/abs/2505.22960 | conditional debate gains | merge stage | conditional gains | debate gate |
| 266 | Can LLM Agents Really Debate? | 2025 | https://arxiv.org/abs/2511.07784 | debate process analysis | consensus diagnostics | narrow | failure taxonomy |
| 267 | Multiple LLM Agents Debate for Cultural Alignment | 2025 | https://arxiv.org/abs/2505.24671 | diverse debate | safety eval | ambiguity | diversity panel |
| 268 | Scaling Test-time Compute for LLM Agents | 2025 | https://arxiv.org/abs/2506.12928 | rollouts/verifiers/merging | autopilot scheduler | cost | budgeted scheduler |
| 269 | Learning When to Plan | 2025 | https://arxiv.org/abs/2509.03581 | dynamic planning decision | plan/no-plan gate | needs RL/SFT | gate |
| 270 | AgentTTS | 2025 | https://arxiv.org/abs/2508.00890 | compute-optimal subtask strategy | route/profile scheduler | overhead | allocation |
| 271 | Large Language Monkeys | 2024 | https://arxiv.org/abs/2407.21787 | repeated sampling scaling | budget curves | brute force | baseline |
| 272 | Robotouille | 2025 | https://arxiv.org/abs/2502.05227 | async long-horizon planning | scheduler | toy domain | concurrency eval |
| 273 | GAIA | 2023 | https://arxiv.org/abs/2311.12983 | real-world assistant tasks | hard-task suite | web volatility | periodic benchmark |
| 274 | AssistantBench | 2024 | https://arxiv.org/abs/2407.15711 | web assistant eval | browser/tool replay | annotation cost | adapter |
| 275 | OSWorld | 2024 | https://arxiv.org/abs/2404.07972 | OS GUI/CLI tasks | side-effect sandbox | heavy | future |
| 276 | TheAgentCompany | 2024 | https://arxiv.org/abs/2412.14161 | company workflow benchmark | AEP simulation | expensive setup | long-horizon target |
| 277 | Tau-bench | 2024 | https://arxiv.org/abs/2406.12045 | user/tool/policy interaction | policy evaluator | two domains | policy benchmark |
| 278 | AgentBoard | 2024 | https://arxiv.org/abs/2401.13178 | progress-rate analytics | dashboard metric | uneven envs | progress score |
| 279 | R-Judge | 2024 | https://arxiv.org/abs/2401.10019 | risk awareness | safety testbed | aging labels | risk classifier |
| 280 | AgentHarm | 2024 | https://arxiv.org/abs/2410.09024 | harmful agent request eval | safety regression | adversary coverage | misuse benchmark |
| 281 | Agent-SafetyBench | 2024 | https://arxiv.org/abs/2412.14470 | interactive safety cases | safety fixtures | prompt insufficiency | interactive gates |
| 282 | SafeAgentBench | 2024 | https://arxiv.org/abs/2412.13178 | safe planning | risk-aware planner | embodied mismatch | hazard cases |
| 283 | Reliable Weak-to-Strong Monitoring of LLM Agents | 2025 | https://arxiv.org/abs/2508.19461 | monitor red-teaming | shadow monitor | monitor assumptions | adversarial monitor |
| 284 | SafetyFlow | 2025 | https://arxiv.org/abs/2508.15526 | safety benchmark generation | safety dataset generator | synthetic redundancy | generate cases |
| 285 | LLM Agent Honeypot | 2025 | https://arxiv.org/abs/2410.13919 | prompt-injection honeypots | ops security monitor | weak attribution | abuse detection |
| 286 | Agent Laboratory | 2025 | https://arxiv.org/abs/2501.04227 | autonomous research pipeline | research campaign automation | hallucinated science | reviewed research loop |
| 287 | Open Source Planning & Control for Scientific Discovery | 2025 | https://arxiv.org/abs/2507.07257 | planning/control for science | experiment orchestration | domain-heavy | template |
| 288 | VLM-Guided Autonomous Scientific Discovery | 2025 | https://arxiv.org/abs/2511.14631 | visual checkpoint judging | artifact QA | VLM fragility | visual QA |
| 289 | STELLA | 2025 | https://arxiv.org/abs/2507.02004 | self-evolving templates/tools | pathway analyzer | tool sprawl | template library |
| 290 | ALAS | 2025 | https://arxiv.org/abs/2508.15805 | curriculum, retrieval, SFT/DPO loop | model trainer | drift | gated self-update |
| 291 | AURA | 2025 | https://arxiv.org/abs/2506.02507 | schema-validated curriculum RL | curriculum generator | robotics transfer | YAML spec |
| 292 | EduPlanner | 2025 | https://arxiv.org/abs/2504.05370 | evaluator/optimizer/analyst agents | curriculum design | education bias | adversarial curriculum |
| 293 | Helmsman | 2025 | https://arxiv.org/abs/2510.14512 | multi-agent code synthesis/eval | codegen campaign | narrow | sandbox pattern |
| 294 | From LLM Reasoning to Autonomous AI Agents Review | 2025 | https://arxiv.org/abs/2504.19678 | benchmark taxonomy | coverage checklist | survey lag | checklist |
| 295 | AgentEvals Spec | 2026 | https://agentevals.io/ | declarative YAML evals | eval manifests | young standard | portable eval format |
| 296 | OpenAI Agent Evals | 2025 | https://platform.openai.com/docs/guides/agent-evals | trace grading/eval runs | eval flywheel | hosted dependency | pattern |
| 297 | LangSmith Evaluation Concepts | active | https://docs.langchain.com/langsmith/evaluation-concepts | eval lifecycle | dashboard panels | vendor lock-in | lifecycle |
| 298 | LangGraph Production: Latency, Replay, Scale | 2024 | https://aerospike.com/blog/langgraph-production-latency-replay-scale | state storage for replay | checkpoint design | vendor framing | persistence memo |
| 299 | OpenAI Agents JS Tracing | 2025 | https://openai.github.io/openai-agents-js/guides/tracing/ | trace export controls | cross-runtime trace parity | runtime diff | trace processor |
| 300 | CrewAI Observability Stack | active | https://docs.crewai.com/core-concepts/Agents/ | agents/process observability | AEP workflow templates | abstraction leakage | compare DSLs |
