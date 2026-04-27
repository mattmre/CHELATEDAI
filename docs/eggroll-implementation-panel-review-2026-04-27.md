# EGGROLL Implementation Expert Panel Review

## Purpose

This document records a multi-panel expert review of the current EGGROLL-inspired ChelatedAI implementation. The review used the Evokore panel-of-experts methodology: multiple domain panels, contrarian stress testing, convergence, and feasibility-oriented prioritization.

Panels convened:

- zeroth-order optimization / low-rank adapter / quantized fine-tuning panel
- retrieval / chelation / evaluation panel
- computational-storage / vector database / distributed systems panel
- validation / reproducibility / CI panel

## Executive conclusion

The current implementation is a strong opt-in foundation, but it is not yet a complete research-grade EGGROLL/ChelatedAI stack. The major missing piece is that the new ES optimizer is still mostly **target-loss-driven** rather than **retrieval-fitness-driven**.

Current strengths:

- adapter-only low-rank ES is implemented without changing defaults
- deterministic seed replay is present
- quantization-aware fitness simulation exists
- Kalman-style sigma adaptation exists
- online ES micro-population updates exist
- storage-sharded population scoring simulation exists
- Session 33 transfer-gate discipline remains intact

Primary gaps:

1. ES fitness is not yet direct retrieval fitness.
2. Best-individual / elite tracking is missing from the ES optimizer.
3. Quantization-aware scoring exists but is not yet an end-to-end promotion gate.
4. Online ES and sedimentation losses are not unified.
5. Topology/isomer health remains passive rather than a fitness/sigma signal.
6. Storage-sharded population scoring is standalone and not connected to ES execution.
7. Seed, artifact, and benchmark reproducibility need stronger infrastructure before large campaigns.

## Converged gap matrix

| Priority | Gap | Why it matters | Recommended action |
|---|---|---|---|
| P0 | Retrieval-aware ES fitness | EGGROLL-style optimization should score actual retrieval/ranking outcomes, not only MSE/InfoNCE target loss. | Add a `RetrievalFitnessEvaluator` that can score mini-batch NDCG/MRR/Recall@K inside ES generations. |
| P0 | Best-individual tracking | Current ES updates by weighted average only; a bad generation can move away from the best candidate. | Track elite parameters and expose best fitness/params in optimizer results. |
| P0 | Quantization gate | Session 31 showed corrections can disappear after INT8 quantization; current quantization-aware mode is not promotion-tested. | Add INT8-vs-FP32 validation and reject candidates whose gains vanish post-quantization. |
| P0 | Reproducibility/provenance | Benchmark results lack a consistent seed/device/config/artifact schema. | Add benchmark metadata schema and seed policy before campaigns. |
| P1 | Unified online/sedimentation fitness | Online ES uses a local pos-neg margin while sedimentation uses MSE/InfoNCE/hybrid. | Make online ES accept shared loss/fitness objects. |
| P1 | Structural health as fitness | Topology/isomer/stability metrics are computed but not fed back into ES. | Add optional structural-health fitness penalty and/or sigma control. |
| P1 | Storage evaluator integration | `mock_array.sharded_population_evaluation()` does not participate in ES. | Add a `DistributedFitnessEvaluator` interface with mock storage backend. |
| P1 | Multi-seed gates | Repeatability currently proves one clean rerun, not seed sensitivity. | Add 3+ seed sweeps for candidate promotion. |
| P2 | Rank adaptation | Rank is configurable but not adaptive per parameter/layer. | Add future effective-rank diagnostics and rank sweep support. |
| P2 | Device-class storage model | Storage latency constants are single-value and not device-class calibrated. | Add device profiles for RP2040, consumer NVMe, SmartSSD/FPGA, DPU-backed storage. |
| P2 | Population persistence on storage | Current storage simulation is single-pass/in-memory. | Model population state, cached fitness, replay seeds, and fault recovery. |

## Expert-panel findings

### Optimization panel

The optimizer panel judged the low-rank perturbation implementation sound. The rank-1 default, deterministic seed replay, adapter-only target, and no-default-change posture are correct.

Critical concerns:

- no best-individual / elite memory
- no antithetic sampling or rank-based fitness shaping yet
- current scalar fitness is primarily negated loss, not retrieval quality
- Kalman sigma currently uses population fitness variance as a proxy for measurement noise, which is useful but not theoretically identical
- rank is fixed per config rather than adaptive per parameter shape/effective rank

Recommended next improvements:

1. Add elite/best-parameter tracking.
2. Add optional antithetic pairs.
3. Add retrieval-aware fitness and rank-normalized fitness shaping.
4. Keep Kalman sigma opt-in, but document it as a heuristic.

### Retrieval and chelation panel

The retrieval panel agreed that the architecture is promising but incomplete as an adaptive retrieval optimizer.

Critical concerns:

- ES is currently optimizing embedding targets, not direct retrieval ranking.
- Quantization-aware scoring is available but not part of transfer gates.
- Online ES uses a separate local margin objective from sedimentation.
- Topology/isomer reports are observational; they do not penalize risky ES candidates.
- Chelation logs feed sedimentation, but ES fitness does not weight high-frequency collapse cases.

Recommended next improvements:

1. Add mini-batch retrieval fitness for ES.
2. Add structural-health penalty terms.
3. Weight ES fitness by collapse frequency or hard-negative severity.
4. Add an integrated benchmark comparing Adam vs ES vs ES plus online ES.

### Computational-storage panel

The storage panel found the scope boundary well documented, but the new storage-sharded population simulation remains too abstract to support the SSD/database thesis.

Critical concerns:

- latency constants are single-value and not tied to real device classes
- queue depth, garbage collection stalls, controller scheduling, and I/O priority inversion are not modeled
- storage-sharded candidate scoring is not connected to ES optimizer execution
- population persistence, seed replay on storage, cached fitness, and fault recovery are not modeled
- real hardware evidence remains absent

Recommended next improvements:

1. Add device-class profiles.
2. Add queue-depth/contention and latency variance tests.
3. Add a mock `DistributedFitnessEvaluator` interface that can plug into the ES optimizer.
4. Separate software validation gates from hardware validation gates so software progress is not blocked by missing hardware.

### Validation panel

The validation panel found the test suite broad but not yet sufficient for real benchmark campaigns around ES.

Critical concerns:

- no global seed/reproducibility context
- no seed matrix across benchmark tasks
- no artifact provenance schema for adapter weights/results
- no automated quantization gate
- no stress test for degenerate fitness, NaNs, or failed population members
- CI does not run quick retrieval/quantization benchmark gates

Recommended next improvements:

1. Add `test_seed_reproducibility.py` or a benchmark wrapper with fixed seed matrix.
2. Add `test_quantization_impact.py` or equivalent benchmark gate.
3. Add result metadata schema: commit, config, seeds, device, quantization, optimizer, adapter hash.
4. Add failure-mode tests for ES population evaluation.

## Feasibility-ranked next work

### Immediate P0 queue

1. **RetrievalFitnessEvaluator**
   - mini-batch NDCG@10/MRR/Recall@K
   - optional hard-negative/collapse weighting
   - callable from ES fitness loop

2. **Elite ES tracking**
   - retain best candidate parameters
   - expose best fitness and best generation in results
   - optional rollback to best if final weighted update underperforms

3. **Quantization promotion gate**
   - compare FP32 and simulated INT8 fitness
   - reject candidates with invisible corrections
   - add benchmark/report output

4. **Reproducibility metadata**
   - standard result schema for seeds/config/device/commit
   - ensure benchmark scripts emit it

### P1 queue

1. Shared online/sedimentation fitness object.
2. Structural-health-aware ES penalty.
3. DistributedFitnessEvaluator mock interface.
4. Multi-seed candidate gate.

### P2 queue

1. Rank-adaptation diagnostics.
2. Device-class storage latency profiles.
3. Population persistence and fault-recovery simulation.
4. Hardware runbook expansion for computational-storage ES scoring.

## Current answer to "are we missing anything?"

Yes. The current version covers the skeleton of the EGGROLL/ChelatedAI strategy but not the most important research-grade validation loop.

The next best architectural move is not another broad optimizer feature. It is to make the ES fitness function **retrieval-native**, quantization-gated, reproducible, and structurally aware.

Until those pieces exist, the implementation should remain:

- opt-in
- non-default
- not promoted as better than Adam
- not used to claim SSD/database execution beyond simulation

## Recommended next-session prompt

Implement the P0 queue from `docs/eggroll-implementation-panel-review-2026-04-27.md`: retrieval-aware ES fitness, elite ES tracking, quantization promotion gate, and reproducibility metadata. Preserve current defaults and add focused tests before running benchmark campaigns.

