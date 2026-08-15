# ChelatedAI Related Works - Human-Readable Catalog

Auto-generated from `related_works.tsv` (source of truth).

Totals: 78 works; strong=67; supporting=11; structural_duplicates=4.

## Strong works by claim
### c01 [chelation] - Post-hoc residual embedding correction (~x+delta(x)+L2 norm) improves retrieval in noisy neighborhoods without mutating frozen base weights
- w05 (cousin): SimCSE - arXiv:2104.08821 - Unsupervised contrastive embeddings; baseline for sedimentation-style contrastive training
- w07 (relevant): Drift-Adapter (Orthogonal-Procrustes residual) - arXiv:2509.23471 - EMNLP 2025; inspired OrthogonalProcrustes/LowRankAffine adapters (REFERENCES.md)
- w08 (relevant): Online-Optimized RAG for Tool Use and Function Calling - arXiv:2509.20415 - deployment-time online gradient refinement; verified on arXiv abs 2025-09-24
- w10 (cousin): Adaptive Semantic Prompt Caching with VectorQ - arXiv:2502.03771 - online-learned embedding similarity thresholds with correctness feedback; structural cousin of adaptive chelation threshold guard; verified on arXiv abs (NOTE: repo REFERENCES.md description differs from actual paper)
- w11 (structural_duplicate): CRAG Corrective Retrieval Augmented Generation - arXiv:2401.15884 - Retrieval correction/rerank-when-unstable; direct structural cousin of chelation trigger loop
- w12 (cousin): Adaptive-RAG - arXiv:2403.14403 - Query-complexity routing; fast/slow retrieval path selection analog
- w26 (additive): OPSD Self-Distilled Reasoner - arXiv:2601.18734 - Dense privileged-teacher on-policy distillation; verified (siyan-zhao/OPSD)
- w30 (additive): Attention Residuals AttnRes (MoonshotAI Kimi Linear) - as-cited-in-repo - Source of BlockAttnRes adapters in repo; GPQA-Diamond +7.5% at 1.25x compute

### c02 [chelation] - Bounded corrections (~0.0078 INT8 quantization noise floor) keep adapters quantizable after training
- w06 (relevant): Matryoshka Representation Learning - arXiv:2205.13147 - Per-dimension importance; basis for learned dimension mask predictor (REFERENCES.md)
- w46 (relevant): Learning to Select: Query-Aware Adaptive Dimension Selection for Dense Retrieval - arXiv:2602.03306 - verified on arXiv abs 2026-02-03; per-query dimension masking; REFERENCES.md mislabels this id as MRL (real MRL is w06)

### c03 [dimension-masking] - Learned per-dimension masks steer which embedding dimensions are corrected or masked
- w03 (structural_duplicate): BERT-Whitening (How to Find Your Friendly Neighborhood) - arXiv:2105.00554 - Post-hoc linear transform removing anisotropy; closest prior art to spectral chelation masking/centering
- w06 (relevant): Matryoshka Representation Learning - arXiv:2205.13147 - Per-dimension importance; basis for learned dimension mask predictor (REFERENCES.md)
- w17 (cousin): ColBERT - arXiv:2004.12832 - Late interaction dimension-level operations; comparison point for dimension masking
- w44 (relevant): Randomly Removing 50% of Dimensions in Text Embeddings has Minimal Impact on Retrieval and Classification Tasks - arXiv:2508.17744 - verified on arXiv abs 2025-08-25; random removal of up to 50% across 6 text encoders and 26 retrieval/classification tasks causes less than 10% performance drop; primary-source support for the repo dimension-masking premise
- w46 (relevant): Learning to Select: Query-Aware Adaptive Dimension Selection for Dense Retrieval - arXiv:2602.03306 - verified on arXiv abs 2026-02-03; per-query dimension masking; REFERENCES.md mislabels this id as MRL (real MRL is w06)

### c04 [spectral-reranking] - Center-of-mass / spectral centering reranks documents inside noisy neighborhoods
- w03 (structural_duplicate): BERT-Whitening (How to Find Your Friendly Neighborhood) - arXiv:2105.00554 - Post-hoc linear transform removing anisotropy; closest prior art to spectral chelation masking/centering
- w04 (structural_duplicate): BERT-flow (Sentence Embeddings of PLMs) - arXiv:2011.05864 - Normalizing-flow post-processing of embeddings; cousin of spectral centering
- w55 (math_foundation): Neural ODE / continuous normalizing flows - arXiv:1806.07366 - continuous-depth transform rationale for embedding correction as flow; math foundation for spectral/geometric chelation
- w56 (relevant): Temperature Scaling: On Calibration of Modern Neural Networks - arXiv:1706.04599 - temperature parameter in spectral chelation ranking scores (REFERENCES.md)
- w69 (math_foundation): Deep CORAL: Correlation Alignment for Deep Domain Adaptation - arXiv:1607.01719 - verified on arXiv abs 2016-07-06; aligns second-order source/target statistics with linear/nonlinear transforms; mathematical basis for covariance-aware embedding correction and drift thresholds
- w75 (structural_duplicate): All-but-the-Top: Simple and Effective Postprocessing for Word Representations - arXiv:1702.01417 - verified on arXiv abs 2017-02-05; removes common mean vector and top dominating directions; closest structural analog to spectral centering / common-component removal in embeddings

### c05 [sedimentation] - Contrastive sedimentation on collapse-event logs tunes adapters for future retrieval
- w05 (cousin): SimCSE - arXiv:2104.08821 - Unsupervised contrastive embeddings; baseline for sedimentation-style contrastive training
- w22 (additive): STaR: Bootstrapping Reasoning With Reasoning - arXiv:2203.14465 - rationalize-failures data synthesis for sample-efficient self-correction (canonical id 2203.14465)
- w31 (additive): EGGROLL Evolution Strategies at the Hyperscale - https://eshyperspace.github.io/ - Low-rank E=A*B^T/sqrt(r) evolution strategies; int8 quantized training evidence; verified against paper PDF
- w48 (relevant): BGE-M3 with Self-Knowledge Distillation - arXiv:2402.03216 - Self-KD across dense/sparse/multivector heads; multilingual; cousin of hybrid distillation + cross-lingual
- w60 (additive): DAPO: An Open-Source LLM Reinforcement Learning System at Scale - arXiv:2503.14476 - verified on arXiv abs 2025-03-21; Decoupled Clip-Higher + Dynamic Sampling + token-level PG prevent entropy collapse; annealing/stability rationale for online loops
- w63 (relevant): GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval - arXiv:2112.07577 - verified on arXiv abs 2021-12-14; query generation + pseudo labeling for unsupervised retrieval adaptation; domain-transfer analog to sedimentation; up to +9.3 nDCG@10

### c06 [drift-detection] - Topology / isomer / structural signals expose degradation that ranking metrics alone miss
- w58 (relevant): DriftLens: Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time - arXiv:2406.17813 - verified on arXiv abs 2024-06-24; distribution-distance drift detection in embedding/representation space + per-label characterization; direct method for lattice drift-triggered disintegration and repo drift experiments
- w59 (relevant): Measuring Distributional Shifts in Text: The Advantage of LLM-Based Embeddings - arXiv:2312.02337 - verified on arXiv abs 2023-12-04; clustering-based drift measurement over LLM embeddings; drift sensitivity metric; ~18mo production (Fiddler); supports isomer/topology drift premise
- w66 (relevant): Test-Time Training with Self-Supervision for Generalization under Distribution Shifts - arXiv:1909.13231 - verified on arXiv abs 2019-09-29; self-supervised updates on unlabeled test samples, including online streams; foundational test-time adaptation analog
- w67 (relevant): Tent: Fully Test-time Adaptation by Entropy Minimization - arXiv:2006.10726 - verified on arXiv abs 2020-06-18; source-free online affine adaptation; entropy objective is a caution for collapse-safe gates
- w68 (additive): CoTTA: Continual Test-Time Domain Adaptation - arXiv:2203.13591 - verified on arXiv abs 2022-03-25; weight/augmentation averaging plus stochastic source restoration reduce error accumulation and forgetting under non-stationary shift
- w69 (math_foundation): Deep CORAL: Correlation Alignment for Deep Domain Adaptation - arXiv:1607.01719 - verified on arXiv abs 2016-07-06; aligns second-order source/target statistics with linear/nonlinear transforms; mathematical basis for covariance-aware embedding correction and drift thresholds
- w70 (math_foundation): Domain-Adversarial Training of Neural Networks (DANN) - arXiv:1505.07818 - verified on arXiv abs 2015-05-28; domain-invariant representation learning with gradient reversal; foundation for source-target drift adaptation and safe routing
- w71 (math_foundation): A Kernel Method for the Two-Sample Problem (MMD) - arXiv:0805.2368 - verified on arXiv abs 2008-05-15; RKHS two-sample test with linear approximations; principled drift detector candidate for embedding windows
- w78 (relevant): Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time - arXiv:2211.14238 - verified on arXiv abs 2022-11-25; NeurIPS 2022 Datasets-and-Benchmarks; 5 temporal-shift datasets, Eval-Stream streaming protocol, ~20% average ID->OOD drop; drift-benchmark standard for online distribution-shift evaluation

### c07 [self-healing] - Adapter-only self-healing recovers retrieval after induced drift (rotation 3/3 recovery@12; noise 0/3 on uncalibrated severity)
- w01 (math_foundation): Elastic Weight Consolidation - arXiv:1612.00796 - Retention/forgetting anchor; basis for retention gates and replay guards
- w10 (cousin): Adaptive Semantic Prompt Caching with VectorQ - arXiv:2502.03771 - online-learned embedding similarity thresholds with correctness feedback; structural cousin of adaptive chelation threshold guard; verified on arXiv abs (NOTE: repo REFERENCES.md description differs from actual paper)
- w13 (cousin): Self-RAG - arXiv:2310.11511 - Reflection tokens gate retrieval quality; cousin to SelfEditDirective evaluation gates
- w21 (additive): Self-Rewarding Language Models - arXiv:2401.10020 - Iterative self-judging improvement loop; outer-loop engine for self-healing
- w22 (additive): STaR: Bootstrapping Reasoning With Reasoning - arXiv:2203.14465 - rationalize-failures data synthesis for sample-efficient self-correction (canonical id 2203.14465)
- w23 (additive): ReST-MCTS-star - arXiv:2406.03816 - Process-reward-guided search + filtering; denser fitness signal than outcome-only
- w24 (additive): Constitutional AI - arXiv:2212.08073 - Critique-revise self-edits against explicit principles
- w25 (additive): SDFT Self-Distillation Enables Continual Learning - arXiv:2601.19897 - On-policy continual self-distillation with EMA teacher; verified on arXiv abs (MIT/ETH)
- w26 (additive): OPSD Self-Distilled Reasoner - arXiv:2601.18734 - Dense privileged-teacher on-policy distillation; verified (siyan-zhao/OPSD)
- w27 (additive): SDPO Reinforcement Learning via Self-Distillation - arXiv:2601.20802 - Dense logit-level self-teacher advantages; verified (lasgroup/SDPO)
- w29 (additive): A Survey of Process Reward Models - arXiv:2510.08049 - Process-supervision survey; verified on arXiv abs (v3 2026-04-29)
- w31 (additive): EGGROLL Evolution Strategies at the Hyperscale - https://eshyperspace.github.io/ - Low-rank E=A*B^T/sqrt(r) evolution strategies; int8 quantized training evidence; verified against paper PDF
- w32 (additive): SEAL Self-Adapting Language Models - arXiv:2506.10943 - Source of SEAL planner inspiration in repo; verified on arXiv; replaces wrong id 2412.01122
- w64 (relevant): LoRA Learns Less and Forgets Less - arXiv:2405.09673 - verified on arXiv abs 2024-05-15; low-rank finetuning preserves base-model retention better than full finetuning; retention evidence for adapter-only correction path
- w68 (additive): CoTTA: Continual Test-Time Domain Adaptation - arXiv:2203.13591 - verified on arXiv abs 2022-03-25; weight/augmentation averaging plus stochastic source restoration reduce error accumulation and forgetting under non-stationary shift
- w74 (relevant): AdapterFusion: Non-Destructive Task Composition for Transfer Learning - arXiv:2005.00247 - verified on arXiv abs 2020-05-01; separates task-specific adapter extraction from non-destructive composition, evaluates 16 NLU tasks, and avoids full-finetuning forgetting; direct adapter-retention comparator
- w77 (additive): Process-based Self-Rewarding Language Models - arXiv:2503.03746 - verified on arXiv abs 2025-03-05; long-thought reasoning + step-wise LLM-as-judge + step-wise preference optimization; process-dense credit assignment for self-edit acceptance

### c08 [distillation] - Teacher-guided (hybrid) distillation improves retrieval and transfers across SciFact / NFCorpus road-course runs
- w02 (math_foundation): Generalized Knowledge Distillation (GKD) - arXiv:2306.13649 - on-policy distillation with JSD from self-generated outputs; verified on arXiv abs 2023-06-23; basis for SDPO/OPSD stability
- w07 (relevant): Drift-Adapter (Orthogonal-Procrustes residual) - arXiv:2509.23471 - EMNLP 2025; inspired OrthogonalProcrustes/LowRankAffine adapters (REFERENCES.md)

### c09 [online-correction] - Inference-time adapter / online updates improve quality without destabilizing retention and stability
- w01 (math_foundation): Elastic Weight Consolidation - arXiv:1612.00796 - Retention/forgetting anchor; basis for retention gates and replay guards
- w02 (math_foundation): Generalized Knowledge Distillation (GKD) - arXiv:2306.13649 - on-policy distillation with JSD from self-generated outputs; verified on arXiv abs 2023-06-23; basis for SDPO/OPSD stability
- w08 (relevant): Online-Optimized RAG for Tool Use and Function Calling - arXiv:2509.20415 - deployment-time online gradient refinement; verified on arXiv abs 2025-09-24
- w09 (cousin): TTARAG Predict the Retrieval! - arXiv:2601.11443 - Test-time adaptation of RAG via prefix-suffix self-supervision; ICASSP 2026; verified on arXiv abs
- w25 (additive): SDFT Self-Distillation Enables Continual Learning - arXiv:2601.19897 - On-policy continual self-distillation with EMA teacher; verified on arXiv abs (MIT/ETH)
- w26 (additive): OPSD Self-Distilled Reasoner - arXiv:2601.18734 - Dense privileged-teacher on-policy distillation; verified (siyan-zhao/OPSD)
- w27 (additive): SDPO Reinforcement Learning via Self-Distillation - arXiv:2601.20802 - Dense logit-level self-teacher advantages; verified (lasgroup/SDPO)
- w28 (additive): MIS-PO filtered policy optimization (Step 3.5 Flash) - arXiv:2602.10604 - Binary token+trajectory ratio filtering for stable off-policy RL; verified on arXiv abs
- w32 (additive): SEAL Self-Adapting Language Models - arXiv:2506.10943 - Source of SEAL planner inspiration in repo; verified on arXiv; replaces wrong id 2412.01122
- w60 (additive): DAPO: An Open-Source LLM Reinforcement Learning System at Scale - arXiv:2503.14476 - verified on arXiv abs 2025-03-21; Decoupled Clip-Higher + Dynamic Sampling + token-level PG prevent entropy collapse; annealing/stability rationale for online loops
- w64 (relevant): LoRA Learns Less and Forgets Less - arXiv:2405.09673 - verified on arXiv abs 2024-05-15; low-rank finetuning preserves base-model retention better than full finetuning; retention evidence for adapter-only correction path
- w66 (relevant): Test-Time Training with Self-Supervision for Generalization under Distribution Shifts - arXiv:1909.13231 - verified on arXiv abs 2019-09-29; self-supervised updates on unlabeled test samples, including online streams; foundational test-time adaptation analog
- w67 (relevant): Tent: Fully Test-time Adaptation by Entropy Minimization - arXiv:2006.10726 - verified on arXiv abs 2020-06-18; source-free online affine adaptation; entropy objective is a caution for collapse-safe gates
- w68 (additive): CoTTA: Continual Test-Time Domain Adaptation - arXiv:2203.13591 - verified on arXiv abs 2022-03-25; weight/augmentation averaging plus stochastic source restoration reduce error accumulation and forgetting under non-stationary shift
- w70 (math_foundation): Domain-Adversarial Training of Neural Networks (DANN) - arXiv:1505.07818 - verified on arXiv abs 2015-05-28; domain-invariant representation learning with gradient reversal; foundation for source-target drift adaptation and safe routing
- w74 (relevant): AdapterFusion: Non-Destructive Task Composition for Transfer Learning - arXiv:2005.00247 - verified on arXiv abs 2020-05-01; separates task-specific adapter extraction from non-destructive composition, evaluates 16 NLU tasks, and avoids full-finetuning forgetting; direct adapter-retention comparator
- w76 (relevant): SelfAug: Mitigating Catastrophic Forgetting in Retrieval-Augmented Generation via Distribution Self-Alignment - arXiv:2509.03934 - verified on arXiv abs 2025-09-04; aligns input-sequence logits to preserve semantic distribution, mitigating catastrophic forgetting in RAG fine-tuning with code released; input-anchored KL evidence for adapter retention

### c10 [cross-lingual] - Language-aware teacher routing generalizes embedding correction across languages
- w47 (relevant): Making Monolingual Sentence Embeddings Multilingual Using KD - arXiv:2004.09813 - Cross-lingual teacher distillation; basis for repo cross_lingual_distillation + language routing
- w48 (relevant): BGE-M3 with Self-Knowledge Distillation - arXiv:2402.03216 - Self-KD across dense/sparse/multivector heads; multilingual; cousin of hybrid distillation + cross-lingual

### c11 [lattice] - Unified annealing temperature schedule (temperature controller) unifies explore/stabilize across sedimentation, ES, online updates
- w38 (additive): Exploratory Annealed Decoding (EAD) - arXiv:2510.05251 - Explore-early/exploit-late token temperature schedule for RLVR; annealing rationale for lattice temperature controller
- w60 (additive): DAPO: An Open-Source LLM Reinforcement Learning System at Scale - arXiv:2503.14476 - verified on arXiv abs 2025-03-21; Decoupled Clip-Higher + Dynamic Sampling + token-level PG prevent entropy collapse; annealing/stability rationale for online loops

### c12 [lattice] - Evidence DAG over the attribution pool links query-doc-actuator relationships dynamically
- w14 (cousin): GraphRAG - arXiv:2404.16130 - Evidence-graph RAG with preprocessing pools; closest structural cousin of lattice evidence DAG + precomputed pools
- w39 (relevant): DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal RAG - arXiv:2607.28580 - verified on arXiv abs 2026-07-30; GNN+DAG message passing with serialized evidence graph; lattice evidence-DAG match
- w40 (relevant): MemGraphRAG: Memory-based Multi-Agent System for Graph RAG - arXiv:2606.00610 - verified on arXiv abs 2026-05-30; hierarchical index + source evidence graph for graph retrieval
- w41 (relevant): LogicRAG: You Don't Need Pre-built Graphs for RAG - arXiv:2508.06105 - verified on arXiv abs 2025-08-08; query-specific DAG of subproblems with adaptive retrieval; evidence-DAG cousin
- w42 (relevant): GNN-RAG: Graph Neural Retrieval for LLM Reasoning - arXiv:2405.20139 - verified on arXiv abs 2024-05-30; GNN dense subgraph reasoner for KG RAG; prior art to lattice GNN prototype
- w57 (cousin): Sparse Autoencoders Find Highly Interpretable Features in Language Models - arXiv:2309.08600 - verified on arXiv abs 2023-09-15; sparse feature directions for steering; basis for Model-Scope steering and lattice shims
- w65 (cousin): LightRAG: Simple and Fast Retrieval-Augmented Generation - arXiv:2410.05779 - dual-level entity-relation graph retrieval; evidence-graph retrieval cousin; verified on arXiv abs

### c13 [lattice] - Drift-triggered disintegration plus re-annealing keeps the pool healthy under sparsification without recall collapse
- w11 (structural_duplicate): CRAG Corrective Retrieval Augmented Generation - arXiv:2401.15884 - Retrieval correction/rerank-when-unstable; direct structural cousin of chelation trigger loop
- w14 (cousin): GraphRAG - arXiv:2404.16130 - Evidence-graph RAG with preprocessing pools; closest structural cousin of lattice evidence DAG + precomputed pools
- w23 (additive): ReST-MCTS-star - arXiv:2406.03816 - Process-reward-guided search + filtering; denser fitness signal than outcome-only
- w29 (additive): A Survey of Process Reward Models - arXiv:2510.08049 - Process-supervision survey; verified on arXiv abs (v3 2026-04-29)
- w31 (additive): EGGROLL Evolution Strategies at the Hyperscale - https://eshyperspace.github.io/ - Low-rank E=A*B^T/sqrt(r) evolution strategies; int8 quantized training evidence; verified against paper PDF
- w39 (relevant): DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal RAG - arXiv:2607.28580 - verified on arXiv abs 2026-07-30; GNN+DAG message passing with serialized evidence graph; lattice evidence-DAG match
- w40 (relevant): MemGraphRAG: Memory-based Multi-Agent System for Graph RAG - arXiv:2606.00610 - verified on arXiv abs 2026-05-30; hierarchical index + source evidence graph for graph retrieval
- w42 (relevant): GNN-RAG: Graph Neural Retrieval for LLM Reasoning - arXiv:2405.20139 - verified on arXiv abs 2024-05-30; GNN dense subgraph reasoner for KG RAG; prior art to lattice GNN prototype
- w43 (cousin): The Molecular Structure of Thought (Mole-Syn) - arXiv:2601.06002 - verified on arXiv abs 2026-01-09; behavioral topology over CoT bond types; lattice disintegration/annealing analog
- w58 (relevant): DriftLens: Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time - arXiv:2406.17813 - verified on arXiv abs 2024-06-24; distribution-distance drift detection in embedding/representation space + per-label characterization; direct method for lattice drift-triggered disintegration and repo drift experiments
- w65 (cousin): LightRAG: Simple and Fast Retrieval-Augmented Generation - arXiv:2410.05779 - dual-level entity-relation graph retrieval; evidence-graph retrieval cousin; verified on arXiv abs
- w78 (relevant): Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time - arXiv:2211.14238 - verified on arXiv abs 2022-11-25; NeurIPS 2022 Datasets-and-Benchmarks; 5 temporal-shift datasets, Eval-Stream streaming protocol, ~20% average ID->OOD drop; drift-benchmark standard for online distribution-shift evaluation

### c14 [quantization] - Steerable quant-surviving shims gate promotion of adapter routes (quantization survival + retrieval fitness)
- w13 (cousin): Self-RAG - arXiv:2310.11511 - Reflection tokens gate retrieval quality; cousin to SelfEditDirective evaluation gates
- w28 (additive): MIS-PO filtered policy optimization (Step 3.5 Flash) - arXiv:2602.10604 - Binary token+trajectory ratio filtering for stable off-policy RL; verified on arXiv abs
- w57 (cousin): Sparse Autoencoders Find Highly Interpretable Features in Language Models - arXiv:2309.08600 - verified on arXiv abs 2023-09-15; sparse feature directions for steering; basis for Model-Scope steering and lattice shims
- w61 (cousin): Mixture-of-Depths: Dynamically Allocating Compute in Transformers - arXiv:2404.02258 - verified on arXiv abs 2024-04-02; top-k capacity-limited routing = steerable per-token compute gates; shim-routing analog
- w72 (cousin): Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity - arXiv:2101.03961 - verified on arXiv abs 2021-01-11; capacity-constrained top-1 expert routing, load-balancing, lower-precision training, and stability controls; comparator for quant-surviving shim route selection

### c15 [computational-storage] - Block-graph payloads with parity can serve retrieval / model shards from disk with host parity verified
- w50 (additive): TRACE: Unlocking Effective CXL Bandwidth via Lossless Compression and Precision Scaling - arXiv:2509.03377 - verified on arXiv abs 2025-09-03; BF16 KV footprint -46.9% lossless, 4.24x throughput at 128k tokens; real CXL tier latency evidence
- w51 (additive): Scalable PNM for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management - arXiv:2511.00321 - verified on arXiv abs 2025-10-31; PNM-KV/PnG-KV up to 21.9x throughput, 60x lower energy/token; disk/tier pool evidence
- w52 (additive): HILOS: A Cost-Effective Near-Storage Processing Solution for Offline Long-Context LLM Inference - arXiv:2502.09921 - verified on arXiv abs 2025-02-14; real 16 SmartSSDs, up to 7.86x throughput, -85% energy; near-storage fitness scoring analog
- w53 (additive): HyMCache: A CXL Memory Rack for Multi-Turn LLM Serving - arXiv:2607.18141 - verified on arXiv abs 2026-07-20; SSD-backed CXL-HM KV reuse; disk-scale pool infra evidence
- w54 (additive): ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories - arXiv:2606.12556 - verified on arXiv abs 2026-06-10; SK Hynix CMM + FPGA prototype, up to 35.7% throughput; pooled-memory evidence

### c16 [evaluation] - Multi-dataset (BEIR-style) evaluation shows retrieval adaptations generalize beyond SciFact
- w15 (relevant): BEIR - arXiv:2104.08663 - Zero-shot retrieval benchmark suite used in repo (REFERENCES.md)
- w16 (relevant): MTEB - arXiv:2210.07316 - Embedding benchmark suite used in repo (REFERENCES.md)
- w63 (relevant): GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval - arXiv:2112.07577 - verified on arXiv abs 2021-12-14; query generation + pseudo labeling for unsupervised retrieval adaptation; domain-transfer analog to sedimentation; up to +9.3 nDCG@10
- w73 (cousin): Ragas: Automated Evaluation of Retrieval Augmented Generation - arXiv:2309.15217 - verified on arXiv abs 2023-09-26; reference-free RAG metrics separate retrieval relevance, context focus, faithfulness, and generation quality; task-centric evaluation cousin for BEIR/MTEB generalization

### c17 [distillation] - Asymmetric privileged-context self-distillation (teacher sees diagnostics, student does not) converts advisory self-edit directives into stable persistent adapter updates without mutating base weights
- w26 (additive): OPSD Self-Distilled Reasoner - arXiv:2601.18734 - Dense privileged-teacher on-policy distillation; verified (siyan-zhao/OPSD)
- w27 (additive): SDPO Reinforcement Learning via Self-Distillation - arXiv:2601.20802 - Dense logit-level self-teacher advantages; verified (lasgroup/SDPO)

### c18 [online-correction] - Explicit KL divergence anchoring to a reference policy during adapter training prevents catastrophic forgetting across repeated self-edits
- w25 (additive): SDFT Self-Distillation Enables Continual Learning - arXiv:2601.19897 - On-policy continual self-distillation with EMA teacher; verified on arXiv abs (MIT/ETH)
- w76 (relevant): SelfAug: Mitigating Catastrophic Forgetting in Retrieval-Augmented Generation via Distribution Self-Alignment - arXiv:2509.03934 - verified on arXiv abs 2025-09-04; aligns input-sequence logits to preserve semantic distribution, mitigating catastrophic forgetting in RAG fine-tuning with code released; input-anchored KL evidence for adapter retention

### c19 [sedimentation] - Trust-region binary filtering of on-policy correction samples (MIS-PO-style) reduces gradient variance and collapse risk in adapter training versus sparse thresholded replay
- w28 (additive): MIS-PO filtered policy optimization (Step 3.5 Flash) - arXiv:2602.10604 - Binary token+trajectory ratio filtering for stable off-policy RL; verified on arXiv abs

### c20 [self-healing] - Constitutional critique-and-revise of proposed self-edit directives raises directive acceptance rate and long-cycle stability
- w24 (additive): Constitutional AI - arXiv:2212.08073 - Critique-revise self-edits against explicit principles

### c21 [evaluation] - Process-reward (step-level) supervision gives denser credit assignment for self-edit acceptance than outcome-only fitness
- w23 (additive): ReST-MCTS-star - arXiv:2406.03816 - Process-reward-guided search + filtering; denser fitness signal than outcome-only
- w29 (additive): A Survey of Process Reward Models - arXiv:2510.08049 - Process-supervision survey; verified on arXiv abs (v3 2026-04-29)
- w77 (additive): Process-based Self-Rewarding Language Models - arXiv:2503.03746 - verified on arXiv abs 2025-03-05; long-thought reasoning + step-wise LLM-as-judge + step-wise preference optimization; process-dense credit assignment for self-edit acceptance

## Category index
### relevant (26)
- w06 [strong]: Matryoshka Representation Learning (arXiv:2205.13147)
- w07 [strong]: Drift-Adapter (Orthogonal-Procrustes residual) (arXiv:2509.23471)
- w08 [strong]: Online-Optimized RAG for Tool Use and Function Calling (arXiv:2509.20415)
- w15 [strong]: BEIR (arXiv:2104.08663)
- w16 [strong]: MTEB (arXiv:2210.07316)
- w39 [strong]: DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal RAG (arXiv:2607.28580)
- w40 [strong]: MemGraphRAG: Memory-based Multi-Agent System for Graph RAG (arXiv:2606.00610)
- w41 [strong]: LogicRAG: You Don't Need Pre-built Graphs for RAG (arXiv:2508.06105)
- w42 [strong]: GNN-RAG: Graph Neural Retrieval for LLM Reasoning (arXiv:2405.20139)
- w44 [strong]: Randomly Removing 50% of Dimensions in Text Embeddings has Minimal Impact on Retrieval and Classification Tasks (arXiv:2508.17744)
- w45 [supporting]: Dimension Mask Layer: Optimizing Embedding Efficiency for Scalable ID-based Models (arXiv:2510.15308)
- w46 [strong]: Learning to Select: Query-Aware Adaptive Dimension Selection for Dense Retrieval (arXiv:2602.03306)
- w47 [strong]: Making Monolingual Sentence Embeddings Multilingual Using KD (arXiv:2004.09813)
- w48 [strong]: BGE-M3 with Self-Knowledge Distillation (arXiv:2402.03216)
- w49 [supporting]: LaBSE: Language-agnostic BERT Sentence Embedding (arXiv:2007.01852)
- w56 [strong]: Temperature Scaling: On Calibration of Modern Neural Networks (arXiv:1706.04599)
- w58 [strong]: DriftLens: Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time (arXiv:2406.17813)
- w59 [strong]: Measuring Distributional Shifts in Text: The Advantage of LLM-Based Embeddings (arXiv:2312.02337)
- w62 [supporting]: A Survey on Contextualised Semantic Shift Detection (arXiv:2304.01666)
- w63 [strong]: GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval (arXiv:2112.07577)
- w64 [strong]: LoRA Learns Less and Forgets Less (arXiv:2405.09673)
- w66 [strong]: Test-Time Training with Self-Supervision for Generalization under Distribution Shifts (arXiv:1909.13231)
- w67 [strong]: Tent: Fully Test-time Adaptation by Entropy Minimization (arXiv:2006.10726)
- w74 [strong]: AdapterFusion: Non-Destructive Task Composition for Transfer Learning (arXiv:2005.00247)
- w76 [strong]: SelfAug: Mitigating Catastrophic Forgetting in Retrieval-Augmented Generation via Distribution Self-Alignment (arXiv:2509.03934)
- w78 [strong]: Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time (arXiv:2211.14238)

### cousin (19)
- w05 [strong]: SimCSE (arXiv:2104.08821)
- w09 [strong]: TTARAG Predict the Retrieval! (arXiv:2601.11443)
- w10 [strong]: Adaptive Semantic Prompt Caching with VectorQ (arXiv:2502.03771)
- w12 [strong]: Adaptive-RAG (arXiv:2403.14403)
- w13 [strong]: Self-RAG (arXiv:2310.11511)
- w14 [strong]: GraphRAG (arXiv:2404.16130)
- w17 [strong]: ColBERT (arXiv:2004.12832)
- w18 [supporting]: DPR Dense Passage Retrieval (arXiv:2004.04906)
- w19 [supporting]: Sentence-BERT (arXiv:1908.10084)
- w20 [supporting]: SPLADE (arXiv:2107.05720)
- w33 [supporting]: LLM in a Flash (arXiv:2312.11514)
- w34 [supporting]: FlexGen (arXiv:2303.06865)
- w37 [supporting]: Representation Degeneration Problem (anisotropy cone) (arXiv:1907.12009)
- w43 [strong]: The Molecular Structure of Thought (Mole-Syn) (arXiv:2601.06002)
- w57 [strong]: Sparse Autoencoders Find Highly Interpretable Features in Language Models (arXiv:2309.08600)
- w61 [strong]: Mixture-of-Depths: Dynamically Allocating Compute in Transformers (arXiv:2404.02258)
- w65 [strong]: LightRAG: Simple and Fast Retrieval-Augmented Generation (arXiv:2410.05779)
- w72 [strong]: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (arXiv:2101.03961)
- w73 [strong]: Ragas: Automated Evaluation of Retrieval Augmented Generation (arXiv:2309.15217)

### structural_duplicate (4)
- w03 [strong]: BERT-Whitening (How to Find Your Friendly Neighborhood) (arXiv:2105.00554)
- w04 [strong]: BERT-flow (Sentence Embeddings of PLMs) (arXiv:2011.05864)
- w11 [strong]: CRAG Corrective Retrieval Augmented Generation (arXiv:2401.15884)
- w75 [strong]: All-but-the-Top: Simple and Effective Postprocessing for Word Representations (arXiv:1702.01417)

### additive (21)
- w21 [strong]: Self-Rewarding Language Models (arXiv:2401.10020)
- w22 [strong]: STaR: Bootstrapping Reasoning With Reasoning (arXiv:2203.14465)
- w23 [strong]: ReST-MCTS-star (arXiv:2406.03816)
- w24 [strong]: Constitutional AI (arXiv:2212.08073)
- w25 [strong]: SDFT Self-Distillation Enables Continual Learning (arXiv:2601.19897)
- w26 [strong]: OPSD Self-Distilled Reasoner (arXiv:2601.18734)
- w27 [strong]: SDPO Reinforcement Learning via Self-Distillation (arXiv:2601.20802)
- w28 [strong]: MIS-PO filtered policy optimization (Step 3.5 Flash) (arXiv:2602.10604)
- w29 [strong]: A Survey of Process Reward Models (arXiv:2510.08049)
- w30 [strong]: Attention Residuals AttnRes (MoonshotAI Kimi Linear) (as-cited-in-repo)
- w31 [strong]: EGGROLL Evolution Strategies at the Hyperscale (https://eshyperspace.github.io/)
- w32 [strong]: SEAL Self-Adapting Language Models (arXiv:2506.10943)
- w38 [strong]: Exploratory Annealed Decoding (EAD) (arXiv:2510.05251)
- w50 [strong]: TRACE: Unlocking Effective CXL Bandwidth via Lossless Compression and Precision Scaling (arXiv:2509.03377)
- w51 [strong]: Scalable PNM for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management (arXiv:2511.00321)
- w52 [strong]: HILOS: A Cost-Effective Near-Storage Processing Solution for Offline Long-Context LLM Inference (arXiv:2502.09921)
- w53 [strong]: HyMCache: A CXL Memory Rack for Multi-Turn LLM Serving (arXiv:2607.18141)
- w54 [strong]: ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories (arXiv:2606.12556)
- w60 [strong]: DAPO: An Open-Source LLM Reinforcement Learning System at Scale (arXiv:2503.14476)
- w68 [strong]: CoTTA: Continual Test-Time Domain Adaptation (arXiv:2203.13591)
- w77 [strong]: Process-based Self-Rewarding Language Models (arXiv:2503.03746)

### math_foundation (8)
- w01 [strong]: Elastic Weight Consolidation (arXiv:1612.00796)
- w02 [strong]: Generalized Knowledge Distillation (GKD) (arXiv:2306.13649)
- w35 [supporting]: GPTQ (arXiv:2210.17323)
- w36 [supporting]: CMA-ES and Natural Evolution Strategies (arXiv:1604.00772)
- w55 [strong]: Neural ODE / continuous normalizing flows (arXiv:1806.07366)
- w69 [strong]: Deep CORAL: Correlation Alignment for Deep Domain Adaptation (arXiv:1607.01719)
- w70 [strong]: Domain-Adversarial Training of Neural Networks (DANN) (arXiv:1505.07818)
- w71 [strong]: A Kernel Method for the Two-Sample Problem (MMD) (arXiv:0805.2368)

