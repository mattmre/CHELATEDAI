# EGGROLL Strategic Analysis and Iterative Improvement Plan

## Purpose

This plan converts the EGGROLL implementation panel review into a strategic architecture roadmap for ChelatedAI. It identifies current weaknesses, reusable repository functionality, missing platform capabilities, research opportunities, and follow-up experiments.

The plan was created through three iterative loops:

1. **Loop 1: Baseline gap audit and current-research check**
   - Compared the implementation panel review against the repository.
   - Classified items as existing, partial, or missing.
   - Rechecked current research around EGGROLL, RoZO, LOREN/LOZO, QZO/QuZO, adaptive RAG, and near-data storage.

2. **Loop 2: Expanded expert-panel refinement**
   - Re-ran an architecture/research panel to broaden beyond the original P0 gaps.
   - Added candidate archives, composite retrieval/health fitness, storage-backed fitness interfaces, query reformulation, adaptive population sizing, and provenance telemetry.

3. **Loop 3: Convergence and sequencing**
   - Reduced the expanded idea set into an implementable P0/P1/P2/P3 roadmap.
   - Separated blocking validation work from speculative research.
   - Identified claims that must remain explicitly out of scope.

## Executive conclusion

ChelatedAI now has a strong opt-in EGGROLL-inspired foundation, but the platform is not yet ready to claim that ES improves retrieval quality or that storage-backed ES is validated.

The key weakness is still the same after three loops: the ES path is **not yet retrieval-native**. It can optimize adapter loss, but it does not yet optimize direct retrieval outcomes such as NDCG@10, MRR, or Recall@K inside the ES generation loop.

The strategic next move is to make candidate optimization:

- retrieval-native
- quantization-gated
- reproducible
- structurally health-aware
- archive-backed
- pluggable across RAM and simulated near-data storage evaluators

Defaults should remain unchanged until the new gates prove value over Adam/backprop baselines.

## What already exists, what is partial, and what is missing

| Capability | Status | Existing files / evidence | Follow-up |
|---|---|---|---|
| Low-rank ES adapter optimizer | Exists | `evolution_strategies_optimizer.py` | Extend with elite tracking, retrieval fitness, and pluggable evaluators. |
| Quantization simulation | Exists | `simulate_int8_quantization()` in `evolution_strategies_optimizer.py` | Wrap in a promotion gate comparing FP32 and INT8 retrieval fitness. |
| Kalman sigma adaptation | Exists | `KalmanSigmaScheduler` | Keep opt-in and document as heuristic noise control, not full curvature modeling. |
| NDCG benchmark logic | Exists outside ES | `benchmark_distillation.py`, benchmark utilities | Move direct NDCG/MRR/Recall scoring into a reusable ES fitness evaluator. |
| Online loss abstractions | Partial | `online_updater.py` | Reuse but do not force unification; expose optional shared fitness objects. |
| Sedimentation losses | Partial | `sedimentation_loss.py` | Bridge to ES fitness only after retrieval-native scorer exists. |
| Topology/isomer/stability diagnostics | Partial | `topology_analyzer.py`, `isomer_detector.py`, `stability_tracker.py` | Convert passive diagnostics into optional structural-health penalties. |
| Storage-sharded population scoring | Partial | `computational_storage_poc/mock_array.py` | Connect through a `DistributedFitnessEvaluator` interface. |
| Benchmark result metadata | Partial | `benchmark_comparative.py`, checkpoint utilities | Add commit, seed, device, optimizer, quantization, adapter hash, and dataset hash. |
| Repeatability/transfer gates | Partial | `run_repeatability_check.py`, `run_candidate_transfer_gate.py` | Add multi-seed and quantization gates. |
| Best-individual / elite tracking | Missing | No persistent best-params state in ES | Add `EliteArchive` or `CandidateArchive`. |
| Retrieval-aware ES fitness | Missing | No in-generation NDCG/MRR/Recall fitness | Add `RetrievalFitnessEvaluator` and `RetrievalFitnessComposer`. |
| Quantization promotion gate | Missing | Simulation exists but no gate | Add `QuantizationPromotionGate`. |
| Reproducibility context | Missing | Metadata is not end-to-end | Add `ReproducibilityContext` and result schema. |
| Device-class storage profiles | Missing | Single latency constants only | Add profiles for RP2040, consumer NVMe, SmartSSD/FPGA, and DPU-backed storage. |

## Current research signals

| Research area | Insight for ChelatedAI | Strategic consequence |
|---|---|---|
| EGGROLL / low-rank ES | Low-rank black-box updates can scale if the scalar objective is meaningful. | Use ES for adapter surfaces, but make scalar fitness retrieval-native. |
| LOREN / LOZO | Low-rank and curvature-aware ZO can reduce variance and improve sample efficiency. | Treat Kalman sigma as a first step; later add curvature/rank diagnostics. |
| RoZO | Geometry-aware ZO on low-rank adapters reduces unstable updates. | Add optional structural-health and adapter-geometry penalties. |
| QZO / QuZO | Quantized ZO is viable, but quantization must be part of scoring. | Gate candidates by post-quantization retrieval fitness, not only FP32 loss. |
| Adaptive/self-correcting RAG | Retrieval systems need feedback, telemetry, and dynamic correction. | Add retrieval telemetry and make chelation logs feed fitness. |
| HILOS / SmartANNS / NDSEARCH | Near-data systems help with vector search, KV/cache movement, and high-throughput evaluation. | Keep storage claims scoped to near-data retrieval/candidate evaluation until real hardware evidence exists. |

## Strategic themes

### Theme A: Retrieval-driven optimization

Current ES is loss-driven. That is not enough for ChelatedAI because the platform exists to improve retrieval behavior under collapse, quantization, and transfer pressure.

Add a retrieval fitness layer that can evaluate ES candidates using:

- NDCG@10
- MRR
- Recall@K
- collapse-frequency weighting
- hard-negative severity
- optional structural-health penalties
- optional quantization survival score

### Theme B: Candidate promotion and safety

Candidate promotion needs guardrails. ES can move away from the best individual, quantization can erase corrections, and seed variance can make one-off wins look real.

Add:

- elite archive
- rollback to best candidate
- quantization promotion gate
- reproducibility metadata
- multi-seed gate
- candidate lineage and provenance

### Theme C: Structural health as optimization context

Topology, isomer, and stability data should not remain only observational. They should become optional constraints on candidate promotion.

Do not make structural health the primary objective. Retrieval quality remains primary. Structural health should act as a penalty, veto, or sigma-control signal when an adapter destabilizes the embedding topology.

### Theme D: Storage-resident candidate evaluation

The storage thesis should be reframed as near-data population scoring and retrieval filtering, not passive SSD model execution.

Add a pluggable distributed evaluator so the same ES population can be scored by:

- local RAM evaluator
- mock storage evaluator
- future remote evaluator
- future real hardware evaluator

### Theme E: Online/offline adaptation continuity

Online ES, online updater losses, sedimentation losses, and offline distillation should not evolve as isolated optimization tracks.

The first step should be shared interfaces, not forced unification. Keep local online objectives available, but allow ES to consume common fitness objects where appropriate.

## Prioritized roadmap

### P0: Blocking foundations

These must be completed before any major benchmark campaign or claim that ES improves the platform.

| Workstream | Module / interface | Existing reuse | Expected outcome |
|---|---|---|---|
| Retrieval-native fitness | `retrieval_fitness_evaluator.py`, `RetrievalFitnessComposer` | NDCG logic from benchmarks, vector store, benchmark utilities | ES candidates can be scored by NDCG/MRR/Recall inside generation loops. |
| Elite candidate tracking | `elite_archive.py` or `candidate_archive.py` | `CheckpointManager`, ES seed replay | Best candidate parameters survive bad generations and can be rolled back. |
| Quantization promotion gate | `quantization_promotion_gate.py` | `simulate_int8_quantization`, BoundedAdapter findings | Candidates that improve only before INT8 simulation are rejected or flagged. |
| Reproducibility context | `reproducibility_context.py` | benchmark result dataclasses, checkpoint metadata | Runs record commit, seed, config hash, device, quantization flag, optimizer, adapter hash, and dataset hash. |
| Multi-seed smoke gate | `test_seed_reproducibility.py` or benchmark wrapper | repeatability and transfer scripts | Candidate promotion sees seed variance before large campaigns. |

### P1: Core capability layer

These should follow immediately after P0 because they convert the foundation into a differentiated ChelatedAI platform.

| Workstream | Module / interface | Existing reuse | Expected outcome |
|---|---|---|---|
| Structural-health score | `structural_health_score.py` | topology, isomer, stability modules | ES can penalize collapse, isomer drift, or topology damage. |
| Distributed fitness evaluator | `distributed_fitness_evaluator.py` | `mock_array.sharded_population_evaluation()` | ES can call RAM or mock-storage population scoring through one interface. |
| Shared fitness/loss interfaces | `fitness_interfaces.py` or integrated ABCs | online updater and sedimentation losses | Online and offline adaptation can share fitness objects where useful. |
| Quantized benchmark gate | transfer-gate integration | benchmark scripts and quantization gate | INT8 survival becomes part of candidate promotion. |
| Candidate archive telemetry | archive metadata + reports | checkpoint manager and benchmark metadata | Candidate lineage, replay, and warm starts become possible. |

### P2: Optimization and scale

These are valuable but should not block P0/P1.

| Workstream | Purpose |
|---|---|
| Adaptive population sizing | Increase population when variance or stagnation suggests more exploration; shrink when convergence is stable. |
| Antithetic sampling | Reduce ES estimator variance using paired positive/negative perturbations. |
| Rank-based fitness shaping | Make ES more robust to outlier scores and noisy retrieval metrics. |
| Effective-rank diagnostics | Learn whether rank 1, 2, 4, or adaptive rank is best for each adapter family. |
| Device-class storage profiles | Replace single storage latency constants with RP2040, consumer NVMe, SmartSSD/FPGA, and DPU profiles. |
| Storage population cache | Persist population state, cached fitness, replay seeds, and fault-recovery data. |
| Storage-resident ANN simulator | Simulate local top-k filtering on drive shards before host reranking. |

### P3: Research and future bets

These are strategically interesting but should stay behind the main roadmap.

| Idea | Why it might matter | Dissent / caution |
|---|---|---|
| Query reformulator | Hard queries could be expanded or relaxed when retrieval quality is weak. | Adds latency and may require LLM/heuristic dependencies. |
| MoE adapter routing | Per-cluster or per-domain adapters could improve long-tail recall. | Adds routing and promotion complexity. |
| Curriculum ES | Seed future ES from prior elite candidates and gradually harder query sets. | Needs archive first. |
| Cross-lingual transfer prediction | Could predict expensive transfer-gate outcomes earlier. | Requires enough labeled campaign history. |
| Causal retrieval fitness | Could separate real improvements from noise. | Expensive and speculative. |
| Symbolic adapter circuits | Could improve interpretability for certain collapse rules. | Diverges from current neural adapter track. |

## Interfaces to design

### FitnessFunctionInterface

Purpose: a common scoring surface for ES, online adaptation, and future distributed evaluators.

Expected implementations:

- `TrainingLossFitness`
- `RetrievalNDCGFitness`
- `RetrievalMRRFitness`
- `RecallAtKFitness`
- `HybridRetrievalFitness`
- `StructuralHealthPenalty`
- `QuantizationSurvivalFitness`

### RetrievalFitnessEvaluator

Purpose: evaluate a candidate adapter using actual retrieval behavior.

Required capabilities:

- batch query evaluation
- top-k ranking metrics
- qrels/corpus compatibility
- optional query subsampling
- deterministic seeds
- reusable cached corpus embeddings
- per-query telemetry

### EliteArchive / CandidateArchive

Purpose: preserve best candidates and support rollback, warm starts, lineage, and diversity analysis.

Required capabilities:

- store best-k candidate states
- store generation, seed, fitness, quantization status, and metadata
- restore best candidate
- compare final vs best candidate
- export archive report

### QuantizationPromotionGate

Purpose: verify candidate gains survive quantization.

Required capabilities:

- compare FP32 and simulated INT8 retrieval fitness
- configurable tolerance threshold
- report per-candidate and aggregate loss
- flag quantization-safe candidates
- integrate with transfer gate

### ReproducibilityContext

Purpose: make every benchmark/promotion result replayable.

Required fields:

- git commit
- working tree cleanliness marker
- Python version
- platform/device
- model name
- adapter type/hash
- optimizer type/config hash
- ES seed and population seed strategy
- dataset and qrels hash
- quantization settings
- benchmark command

### DistributedFitnessEvaluator

Purpose: decouple population scoring from the local process.

Required implementations:

- `LocalFitnessEvaluator`
- `MockStorageFitnessEvaluator`
- future `RemoteFitnessEvaluator`
- future `HardwareStorageFitnessEvaluator`

## Research questions and experiments

| ID | Question | Experiment | Gate |
|---|---|---|---|
| RQ1 | Does retrieval-native ES beat loss-driven ES? | Compare MSE/InfoNCE ES vs retrieval-fitness ES on at least three benchmarks. | Retrieval ES must improve NDCG/MRR/Recall consistently enough to justify campaigns. |
| RQ2 | Does elite tracking prevent ES drift? | Run ES with and without elite archive; compare final fitness to best-seen fitness. | Elite-backed ES should never finish materially worse than best-seen candidate. |
| RQ3 | Does quantization gating catch invisible corrections? | Compare FP32 vs INT8 candidate fitness and classify candidates that lose most gains. | Gate must identify candidates whose post-quantization gain collapses. |
| RQ4 | Does structural-health scoring improve transfer? | Compare retrieval-only ES vs retrieval plus structural penalty. | Structural penalty should reduce collapse/isomer drift without unacceptable NDCG loss. |
| RQ5 | Can storage-backed scoring be practical in simulation? | Compare local RAM vs mock storage population scoring. | Mock storage path should report honest latency/quality tradeoffs, not assumed speedup. |
| RQ6 | What rank is needed for retrieval adapters? | Sweep rank 1, 2, 4, 8 across adapter types and tasks. | Identify rank defaults or adaptive-rank heuristics. |
| RQ7 | How sensitive is ES to seed variance? | Run fixed seed matrix across candidates. | Promotion gates should account for observed seed variance. |
| RQ8 | Can candidate archives accelerate convergence? | Warm-start ES from elite archive vs cold start. | Warm starts should reduce generations-to-threshold without hurting transfer. |

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Scope creep from too many loop-2 ideas | High | P0/P1/P2/P3 sequencing; no P2/P3 work until P0 is complete unless it is trivial documentation. |
| Retrieval fitness is noisy | High | Use fixed query subsets, larger batches for gates, moving averages, and seed matrices. |
| ES overfits a single benchmark | High | Use repeatability, multitask transfer, and quantization gates before promotion. |
| Quantization simulation differs from real vector DB quantization | Medium | Start as warning gate, then calibrate against actual Qdrant/FAISS quantization if available. |
| Structural penalties fight retrieval quality | Medium | Keep optional and low weight; compare retrieval-only vs health-weighted ES. |
| Storage claims outrun evidence | High | Keep storage path simulation-only until hardware evidence exists. |
| Candidate archive grows without bound | Low | Store best-k plus metadata; prune by age, fitness, and diversity. |
| Unified loss interface over-constrains online updates | Medium | Use pluggable interfaces; do not force one objective everywhere. |

## What not to claim

Do not claim:

1. ES is better than Adam until retrieval-native ES passes repeatability, transfer, quantization, and seed gates.
2. SSDs can run the full AI workload until real hardware evidence exists.
3. Storage-backed ES is validated by `mock_array` alone.
4. Quantization is solved by INT8 simulation alone.
5. Structural health is the primary objective.
6. Online ES is production-safe without latency and stability gates.
7. Candidate archives prevent bad candidates without downstream promotion gates.

## Recommended implementation sequence

1. Add retrieval fitness interfaces and `RetrievalFitnessEvaluator`.
2. Add elite archive and ES rollback/best-state reporting.
3. Add quantization promotion gate and benchmark output.
4. Add reproducibility metadata schema and seed matrix gate.
5. Run RQ1 to compare retrieval-fitness ES vs loss-driven ES.
6. Add structural-health scoring if RQ1 justifies continued ES work.
7. Add distributed fitness evaluator and mock storage backend integration.
8. Add P2 optimizer hygiene: antithetic sampling, rank shaping, adaptive population size.
9. Add device-class storage profiles and storage-resident ANN simulation.
10. Only then run broad benchmark/regression campaigns.

## Immediate next-session prompt

Implement the P0 foundations from `docs/eggroll-strategic-analysis-plan-2026-04-27.md`: retrieval-native ES fitness, elite archive/rollback, quantization promotion gate, reproducibility metadata, and a multi-seed smoke gate. Preserve existing defaults, keep ES opt-in, and add focused unittest coverage before benchmark campaigns.

