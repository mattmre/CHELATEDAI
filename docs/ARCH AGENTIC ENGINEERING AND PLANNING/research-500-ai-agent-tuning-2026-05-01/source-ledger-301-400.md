# Source Ledger 301-400

Date: 2026-05-01

| ID | Title | Year | Source | Mechanism | ChelatedAI mapping | Critique / risk | Improvement posture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 301 | A Self-Improving Coding Agent | 2025 | https://arxiv.org/abs/2504.15228 | agent edits own code with overseer | AEP code mutation | reward-hacked tests | gated self-patch lab |
| 302 | Building Self-Evolving Agents via Experience-Driven Lifelong Learning | 2025 | https://arxiv.org/abs/2508.19005 | exploration, memory, skill internalization | replay and memory | simulation drift | lifecycle blueprint |
| 303 | EvolveR | 2025 | https://arxiv.org/abs/2510.16079 | self-distilled principles | pathway memory | fossilized bad habits | strategy memory |
| 304 | Agent0 | 2025 | https://arxiv.org/abs/2511.16043 | curriculum and tool co-evolution | curriculum generator | tool overfit | co-evolution sandbox |
| 305 | Agentic Context Engineering | 2025 | https://arxiv.org/abs/2510.04618 | generator/reflector/curator context playbooks | prompt/context registry | context bloat | curated overlays |
| 306 | Self-Evolving Curriculum for LLM Reasoning | 2025 | https://arxiv.org/abs/2505.14970 | bandit curriculum | task scheduler | reward proxy drift | adaptive curriculum |
| 307 | LADDER | 2025 | https://arxiv.org/abs/2503.00735 | recursive decomposition with rewards | `recursive_decomposer.py` | distorted subtasks | verifier-filtered recursion |
| 308 | EvoCurr | 2025 | https://arxiv.org/abs/2508.09586 | behavior-code curriculum generation | task-profile generator | unsafe env code | sandbox synthesis |
| 309 | AutoPDL | 2025 | https://arxiv.org/abs/2504.04365 | prompt-program AutoML | prompt optimizer | search gaps | executable variants |
| 310 | PromptWizard | 2024 | https://arxiv.org/abs/2405.18369 | critique/synthesis prompt loop | reformulation prompts | self-critique bias | prompt evolution |
| 311 | OPRO | 2023 | https://arxiv.org/abs/2309.03409 | optimization by prompted history | config tuning | history contamination | black-box tuning |
| 312 | APE | 2022 | https://arxiv.org/abs/2211.01910 | instruction search | prompt baselines | eval overfit | seed generator |
| 313 | PromptBreeder | 2023 | https://arxiv.org/abs/2309.16797 | evolving prompts and mutation prompts | recursive prompt mutation | complexity | prompt lab |
| 314 | EvoPrompt | 2023 | https://arxiv.org/abs/2309.08532 | genetic prompt operators | config population | stochastic cost | ES comparison |
| 315 | GrIPS | 2022 | https://arxiv.org/abs/2203.07281 | edit-based prompt search | prompt ablations | local optima | cheap sweeps |
| 316 | Automatic Instruction Optimization | 2023 | https://arxiv.org/abs/2309.10259 | LLM instruction candidates | prompt CI pool | validation leakage | candidate pool |
| 317 | Self-Instruct | 2022 | https://arxiv.org/abs/2212.10560 | synthetic instruction bootstrap | task generation | duplicates | filtered corpus |
| 318 | WizardLM / Evol-Instruct | 2023 | https://arxiv.org/abs/2304.12244 | evolved instruction difficulty | hard-query generator | style artifacts | curriculum expansion |
| 319 | Self-Rewarding Language Models | 2024 | https://arxiv.org/abs/2401.10020 | self-generated preference data | internal reward loop | evaluator collapse | external holdout |
| 320 | SPIN | 2024 | https://arxiv.org/abs/2401.01335 | self-play fine-tuning | self-play tuning | distribution narrowing | offline pilot |
| 321 | LLM See, LLM Do | 2024 | https://arxiv.org/abs/2407.01490 | synthetic data active inheritance | data controller | generator leakage | data audits |
| 322 | CRAFT Your Dataset | 2024 | https://arxiv.org/abs/2409.02098 | retrieval-augmented synthetic data | corpus expansion | copied noise | grounded synthesis |
| 323 | SoftSRV Prompting | 2024 | https://arxiv.org/abs/2410.16534 | soft-prompt synthetic generator | targeted generator | less inspectable | controlled generator |
| 324 | Balancing Cost and Effectiveness of Synthetic Data Generation | 2024 | https://arxiv.org/abs/2409.19759 | generation strategy cost model | budget policy | task-specific | budget policy |
| 325 | Self-Alignment with Instruction Backtranslation | 2023 | https://arxiv.org/abs/2308.06259 | generate instructions from high-quality responses | log mining | quality bias | mine strong traces |
| 326 | MetaMath | 2023 | https://arxiv.org/abs/2309.12284 | math QA transformations | difficulty templates | math-only | templates |
| 327 | Orca-Math | 2024 | https://arxiv.org/abs/2402.14830 | teacher synthetic math data | distillation corpus | teacher errors | verifier-filtered |
| 328 | OpenMathInstruct-1 | 2024 | https://arxiv.org/abs/2402.10176 | synthetic reasoning dataset | public baseline | contamination | baseline |
| 329 | ExpeL | 2023 | https://arxiv.org/abs/2308.10144 | lessons from trajectories | lesson memory | wrong lessons | promotion gates |
| 330 | CLIN | 2023 | https://arxiv.org/abs/2310.10134 | continual natural-language memory | learning notes | conflict | versioned notes |
| 331 | AriGraph | 2024 | https://arxiv.org/abs/2407.04363 | semantic/episodic KG world model | graph memory | compounding errors | prototype |
| 332 | ReadAgent | 2024 | https://arxiv.org/abs/2402.09727 | long-document episodic paging | document memory | segmentation | memory policy |
| 333 | LongMem | 2023 | https://arxiv.org/abs/2306.07174 | cached memory network | external memory | architecture-specific | reference |
| 334 | Memory3 | 2024 | https://arxiv.org/abs/2407.01178 | explicit memory module | persistent memory | complexity | target |
| 335 | LLMs as Compilers | 2024 | https://arxiv.org/abs/2406.12952 | prompt program transforms | prompt IR | brittle metaphor | prompt-program IR |
| 336 | AFlow | 2024 | https://arxiv.org/abs/2410.10762 | MCTS workflow graphs | pathway analyzer | heavy framework | compare |
| 337 | Evolving Excellence | 2026 | https://www.research.ed.ac.uk/en/publications/evolving-excellence-automated-optimization-of-llm-based-agents/ | tunes prompts/tools/params | agent config optimizer | preprint | full-stack tuning |
| 338 | SyNeg | 2024 | https://arxiv.org/abs/2412.17250 | synthetic hard negatives | `engine_scope_negatives.py` | false negatives | negative campaign |
| 339 | Don't Retrieve, Generate | 2025 | https://arxiv.org/abs/2504.21015 | corpus-free query/negative generation | retrieval data | unnatural negatives | fallback |
| 340 | GPL | 2021 | https://arxiv.org/abs/2112.07577 | generative pseudo labeling | domain adaptation | pseudo noise | bootstrap |
| 341 | ANCE | 2020 | https://arxiv.org/abs/2007.00808 | async hard-negative mining | retriever loop | stale negatives | online mining |
| 342 | RocketQA | 2020 | https://arxiv.org/abs/2010.08191 | denoised hard negatives | retrieval baseline | complex | robust recipe |
| 343 | SimANS | 2022 | https://arxiv.org/abs/2210.11773 | ambiguous negative sampling | graded negatives | label confusion | calibrated negatives |
| 344 | DINO | 2021 | https://arxiv.org/abs/2104.14294 | self-distillation | teacher-student analogy | vision transfer | reference |
| 345 | MAML | 2017 | https://proceedings.mlr.press/v70/finn17a.html | fast-adaptation initialization | meta adapter init | expensive | reference |
| 346 | Reptile | 2018 | https://arxiv.org/abs/1803.02999 | first-order meta-learning | meta update | weaker | cheap loop |
| 347 | Online Meta-Learning | 2019 | https://arxiv.org/abs/1902.08438 | stream meta-learning | continual scheduler | instability | streaming policy |
| 348 | The Curse of Recursion | 2023 | https://arxiv.org/abs/2305.17493 | synthetic recursion collapse | synthetic data guard | broad setting | anti-collapse |
| 349 | Self-Consuming Generative Models Go MAD | 2023 | https://arxiv.org/abs/2307.01850 | model autophagy | synthetic corpus QA | provenance critical | quotas |
| 350 | Model Evaluation for Extreme Risks | 2023 | https://arxiv.org/abs/2305.15324 | dangerous capability evals | promotion risk gate | eval lag | risk suite |
| 351 | vLLM / PagedAttention | 2023 | https://arxiv.org/abs/2309.06180 | paged KV and batching | local serving benchmark | runtime coupling | adopt |
| 352 | SGLang / RadixAttention | 2023 | https://arxiv.org/abs/2312.07104 | prefix KV reuse | agent-loop caching | shared-prefix dependency | prototype |
| 353 | TensorRT-LLM IFB | 2024 | https://developer.nvidia.com/blog/nvidia-tensorrt-llm-now-accelerates-encoder-decoder-models-with-in-flight-batching/ | in-flight batching | high-throughput lane | engine friction | benchmark |
| 354 | FlashAttention-3 | 2024 | https://arxiv.org/abs/2407.08608 | Hopper FP8 attention | long context | hardware skew | hardware-gated |
| 355 | Layer-Condensed KV Cache | 2024 | https://arxiv.org/abs/2405.10637 | layer-selective KV | VRAM reduction | quality loss | shadow |
| 356 | ClusterKV | 2024 | https://arxiv.org/abs/2412.03213 | semantic-cluster KV | long trace replay | recall bugs | research |
| 357 | SmoothQuant | 2022 | https://arxiv.org/abs/2211.10438 | W8A8 quantization | INT8 serving | calibration sensitivity | benchmark |
| 358 | GPTQ | 2022 | https://arxiv.org/abs/2210.17323 | Hessian PTQ | 3/4-bit local scoring | backend variance | benchmark |
| 359 | AWQ | 2023 | https://arxiv.org/abs/2306.00978 | activation-aware quantization | consumer GPU deployment | domain activation misses | option |
| 360 | bitsandbytes | active | https://huggingface.co/docs/bitsandbytes/main/en/index | 4-bit base and LoRA | memory-efficient overlays | CUDA fragility | adopt |
| 361 | QLoRA | 2023 | https://arxiv.org/abs/2305.14314 | NF4 and paged optimizers | small-GPU adapter training | quant noise | adopt |
| 362 | LoRA | 2021 | https://arxiv.org/abs/2106.09685 | low-rank deltas | base mutation alternative | rank tuning | adopt |
| 363 | PEFT LoRA | active | https://huggingface.co/docs/peft/developer_guides/lora | adapter injection/switching | overlay format | API churn | adopt |
| 364 | DoRA | 2024 | https://arxiv.org/abs/2402.09353 | magnitude/direction adapter | stronger adapter | merge complexity | benchmark |
| 365 | PiSSA | 2024 | https://arxiv.org/abs/2404.02948 | SVD LoRA init | faster convergence | SVD cost | prototype |
| 366 | CorDA | 2024 | https://huggingface.co/docs/peft/developer_guides/lora | covariance LoRA init | preserve knowledge | bias | research |
| 367 | aLoRA | 2025 | https://huggingface.co/docs/peft/developer_guides/lora | invocation-token adapter | verifier/corrector adapters | boundary errors | prototype |
| 368 | Arrow adapter routing | active | https://huggingface.co/docs/peft/developer_guides/lora | token-wise LoRA routing | adapter-router | opacity | shadow |
| 369 | ZeRO-Offload | 2020 | https://www.deepspeed.ai/tutorials/zero-offload/ | CPU optimizer offload | single-GPU larger tuning | PCIe bottleneck | selective |
| 370 | ZeRO-Infinity | 2021 | https://arxiv.org/abs/2104.07857 | CPU/NVMe hierarchy | large experiment fallback | slow | benchmark only |
| 371 | FSDP-QLoRA | active | https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora | sharded quantized training | multi-GPU adapters | debug cost | prototype |
| 372 | Unsloth MoE kernels | 2025 | https://unsloth.ai/docs/new | Triton grouped GEMM | MoE adapter experiments | validate claims | benchmark |
| 373 | Speculative Decoding | 2023 | https://arxiv.org/abs/2211.17192 | draft/target verification | latency | draft mismatch | prototype |
| 374 | Big Little Decoder | 2023 | https://arxiv.org/abs/2302.07863 | small/large decoder pairing | cheap assistant model | memory | research |
| 375 | SpecInfer | 2023 | https://arxiv.org/abs/2305.09781 | token tree verification | batched completions | tree overhead | prototype |
| 376 | Medusa | 2024 | https://proceedings.mlr.press/v235/cai24b.html | multi-decoding heads | acceleration heads | artifact class | benchmark |
| 377 | EAGLE | 2024 | https://proceedings.mlr.press/v235/li24bt.html | feature-level speculative sampling | lossless acceleration | internals access | prototype |
| 378 | Lookahead Decoding | 2023 | https://arxiv.org/abs/2309.08168 | n-gram candidate verification | no-draft acceleration | workload-dependent | benchmark |
| 379 | Prompt Lookup Decoding | 2023 | https://github.com/apoorvumang/prompt-lookup-decoding | prompt n-gram draft | RAG/replay acceleration | weak novel output | cheap test |
| 380 | KV cache quantization | 2024 | https://arxiv.org/abs/2401.18079 | low-bit KV | long context VRAM | attention degradation | benchmark |
| 381 | KIVI | 2024 | https://arxiv.org/abs/2402.02750 | 2-bit KV | 32k context | kernel support | prototype |
| 382 | H2O | 2023 | https://arxiv.org/abs/2306.14048 | heavy-hitter KV eviction | bounded traces | evicts evidence | shadow |
| 383 | StreamingLLM | 2023 | https://arxiv.org/abs/2309.17453 | attention sinks | long-running context | incomplete recall | prototype |
| 384 | SnapKV | 2024 | https://arxiv.org/abs/2404.14469 | observation-window KV compression | long-doc eval | retention bias | benchmark |
| 385 | FastGen | 2023 | https://arxiv.org/abs/2310.01801 | adaptive KV compression | cache telemetry | complexity | research |
| 386 | MInference | 2024 | https://arxiv.org/abs/2407.02490 | sparse prefill | replay/campaign prefill | pattern errors | benchmark |
| 387 | DeepSpeed-MoE | 2022 | https://arxiv.org/abs/2201.05596 | MoE compression | MoE serving constraints | complexity | benchmark |
| 388 | Tutel | 2022 | https://arxiv.org/abs/2206.03382 | MoE parallelism | expert runtime | cluster assumptions | research |
| 389 | MegaBlocks | 2022 | https://arxiv.org/abs/2211.15841 | block-sparse MoE | safe routing | kernel burden | research |
| 390 | Switch Transformer | 2021 | https://arxiv.org/abs/2101.03961 | top-1 expert routing | routing baseline | expert collapse | background |
| 391 | GShard | 2020 | https://arxiv.org/abs/2006.16668 | sharded MoE | scale background | TPU-centric | background |
| 392 | Mixtral 8x7B | 2024 | https://arxiv.org/abs/2401.04088 | top-2 sparse experts | MoE benchmark | high memory | benchmark |
| 393 | DeepSeekMoE | 2024 | https://arxiv.org/abs/2401.06066 | fine/shared experts | router design | complex | research |
| 394 | Qwen3 MoE and Qwen-Scope | 2025 | https://huggingface.co/Qwen | MoE with SAE/adapter surfaces | future target | drift | defer |
| 395 | Punica | 2023 | https://arxiv.org/abs/2310.18547 | multi-tenant LoRA serving | adapter serving | scheduling complexity | prototype |
| 396 | S-LoRA | 2023 | https://arxiv.org/abs/2311.03285 | scalable multi-LoRA batching | adapter A/B | fragmentation | benchmark |
| 397 | LoRAX | active | https://github.com/predibase/lorax | dynamic adapter routing | adapter-serving substrate | dependency | evaluate |
| 398 | llama.cpp | active | https://github.com/ggml-org/llama.cpp | GGUF local serving | fallback runtime | feature gaps | fallback |
| 399 | Ollama | active | https://github.com/ollama/ollama | local packaging/server | launcher path | architecture lag | fallback |
| 400 | OpenAI-compatible local servers | active | https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html | stable API facade | runtime swap | partial compatibility | adopt |
