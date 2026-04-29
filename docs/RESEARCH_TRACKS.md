# Research Tracks

This document describes the main research themes in the repository and where each one currently stands.

## Portfolio Summary

| Track | Primary question | Proof required | Current state |
|---|---|---|---|
| Adaptive retrieval | Can noisy embedding neighborhoods be detected and corrected before they degrade retrieval? | Frozen baseline values plus NDCG@10/MRR/Recall@K deltas after chelation | Implemented; deterministic live-fire and non-saturated closed-course safety harnesses validated; first MiniLM/SciFact road course supports a safer threshold guardrail, not aggressive chelation |
| Chelation fitness validation | Does chelation improve retrieval metrics directly, not just adapter loss? | Retrieval-fitness campaigns with baseline, adaptive, transfer, quantized comparisons, query-level attribution rows, and controlled collapse fixtures | Harness, evaluator, calibration profiles, failure-injection tests, first small-model road course, query attribution, and synthetic collapse benchmark exist; broader benchmark campaigns remain next |
| Distillation and cross-lingual routing | Can teacher-guided correction generalize across models and languages? | Multitask/cross-lingual campaign results with no structural-health regression | Implemented and under evaluation |
| Online correction | Can inference-time updates improve quality without destabilizing the system? | Ablations showing norm/drift and ranking stability under online updates | Implemented; requires controlled ablation |
| Self-adapting chelation | Can the system synthesize, sandbox, and filter its own repair directives from new context and diagnostics? | Positive-reward self-edit candidates that retain prior behavior, survive quantization, and pass road-course gates | SEAL/EGGROLL-inspired advisory planner, self-generated eval probes, candidate ledger, sandbox execution seam, and adaptive validation loops implemented; cloned-adapter persistence remains gated |
| Structural diagnostics | Can topology and isomer signals reveal degradation that ranking metrics miss? | Structural-health gates correlated with retrieval outcomes | Implemented and test-backed |
| Multi-dataset evaluation | Do improvements transfer beyond SciFact? | Transfer gates over held-out query sets and BEIR-style tasks | Implemented; campaign execution is ongoing work |
| LLM architecture adaptation | Which modern LLM architecture and serving patterns should become retrieval-native ChelatedAI subcomponents? | Live-fire telemetry for routing, norm drift, local/global policy, and gate recommendations | Research review complete; runtime diagnostics wired |
| Computational storage and drive nodes | Can some model or control-plane work move toward storage-resident execution? | Hardware evidence for physical transport; separate proof for any full storage-resident runtime claim | Mixed maturity: software proof is strong, hardware claim remains scope-locked |
| Agentic remediation process | Can repository changes be triaged and delivered through a durable AEP workflow? | Verification log entries for each implementation/campaign round | Implemented and extensively documented |

## 1. Adaptive Retrieval And Chelation

### Question

When a query lands in a semantically noisy local neighborhood, can the system detect that condition and rerank or adapt before the result set collapses?

### Main files

- `antigravity_engine.py`
- `chelation_adapter.py`
- `vector_store.py`
- `config.py`

### What exists now

- adaptive retrieval path selection
- spectral or centered reranking
- adapter-backed correction
- structured logging and checkpointing

## 2. Distillation, Scheduling, And Cross-Lingual Routing

### Question

Can a teacher-guided correction loop outperform passive post-processing, and can it stay aligned across language boundaries?

### Main files

- `teacher_distillation.py`
- `teacher_weight_scheduler.py`
- `cross_lingual_distillation.py`
- `language_detector.py`

### What exists now

- teacher-student dimension projection
- hybrid target generation
- teacher-weight schedule families
- language-aware teacher routing

## 3. Online Correction And Structural Health

### Question

Can the runtime update itself during inference without introducing instability that outweighs any quality gains?

### Main files

- `online_updater.py`
- `stability_tracker.py`
- `embedding_quality.py`
- `dimension_mask_predictor.py`
- `topology_analyzer.py`
- `isomer_detector.py`

### What exists now

- multiple online loss families
- scheduler and diagnostics support
- SEAL/EGGROLL-inspired self-edit planning in `self_healing_chelation.py`
- topology snapshots
- isomer drift detection
- structural health reporting

## 4. Evaluation And Weight Refinement

### Question

Do measured gains survive beyond a single benchmark or hyperparameter setting?

### Main files

- `benchmark_beir.py`
- `benchmark_multitask.py`
- `benchmark_comparative.py`
- `benchmark_distillation.py`
- `run_sweep.py`
- `run_large_sweep.py`

### Current state

The implementation surface is present. The main remaining work is experiment execution, artifact capture, and preset refinement.

The canonical repo-wide audit is:

- [roadmap-audit-and-weight-refinement-plan-2026-03-06.md](roadmap-audit-and-weight-refinement-plan-2026-03-06.md)

That audit concludes that the missing work is mostly evaluation and research iteration, not unbuilt runtime features.

## 5. LLM Architecture And AI Engineering Adaptation

### Question

Which modern LLM architecture patterns and practical AI-engineering operations should inform ChelatedAI's adaptive retrieval engine?

### Main files

- `adapter_router.py`
- `adaptive_gate_orchestrator.py`
- `fitness_composition_orchestrator.py`
- `integrated_diagnostics_report.py`
- `dimension_mask_predictor.py`
- `embedding_quality.py`
- `retrieval_fitness_evaluator.py`
- `vector_store.py`
- `dashboard_server.py`

### What exists now

- adapter routing and query reformulation hooks
- structural-health scoring and integrated diagnostics
- quantization promotion gates
- retrieval-fitness evaluation
- storage-profile metadata and mock distributed scoring

### Research conclusion

Modern LLMs repeatedly use sparse routing, compressed memory, local/global context schedules, normalization, speculative execution, and serving/runtime optimization. ChelatedAI should adapt those ideas as retrieval-native workflows rather than copying transformer internals directly.

The project-car safety testbed now covers instrumentation, component benches, control-surface dyno sweeps, non-saturated closed-course loops, calibration profiles, and failure injection. Defaults remain unchanged until the road-course campaign plan passes BEIR, multitask transfer, repeatability, and quantization-survival gates.

The canonical review is:

- [LLM Architecture And AI Engineering Adaptation Review](llm-architecture-ai-engineering-adaptation-review-2026-04-27.md)

## 6. Computational Storage And Drive-Resident Nodes

### Question

Can selected model subgraphs or routing primitives live closer to storage media so that the host only coordinates coarse-grained control instead of performing every fetch and multiplication itself?

### Main files

- `computational_storage_poc/block_graph.py`
- `computational_storage_poc/mock_nvme.py`
- `computational_storage_poc/mock_array.py`
- `computational_storage_poc/payload_contract.py`
- `computational_storage_poc/usb_host_inference.py`
- `computational_storage_poc/capture_hardware_evidence.py`
- `computational_storage_poc/firmware/`
- `computational_storage_poc/emulation/`

### What is already demonstrated

- software block-graph traversal from flash-like payloads
- parity between host and mock-storage execution for the digits-model path
- theoretical latency comparison for the mock NVMe path
- speculative multi-drive racing as a storage-node thought experiment
- deterministic sector-100 transport contract shared by firmware and emulator
- RP2040 firmware build success in CI
- emulation-path validation in CI

### What is not yet demonstrated

- full hard-drive-hosted or SSD-hosted LLM inference
- promoted on-device digits-model validation on real RP2040 hardware
- trustworthy physical-latency evidence for the transport path

This distinction matters. The repo contains genuine drive-node research, but the merged hardware claim remains intentionally narrow:

- current hardware claim: deterministic transport/control-plane proof
- not yet proven: full drive-resident LLM runtime

See [COMPUTATIONAL_STORAGE_DRIVE_NODES.md](COMPUTATIONAL_STORAGE_DRIVE_NODES.md) for the detailed summary.

## 7. Agentic Engineering And Planning

### Question

Can this repository keep delivery quality high while multiple sessions and remediation passes accumulate over time?

### Main files

- `aep_orchestrator.py`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/`

### What exists now

- severity-tiered finding model
- tracker and verification workflow
- long-lived session archive
- scope-lock and risk-memo templates

## Recent Storage-Track Milestones

The most recent merged storage-track additions on `main` include:

- deterministic payload transport path across firmware, emulator, and host reader
- hardware evidence capture tooling
- emulation-path validation coverage in CI
- explicit transport-scope decision record
- retention policy for generated hardware evidence artifacts

The remaining practical follow-through is physical RP2040 evidence capture when compatible hardware is actually attached.

## 8. Testing And Validation Roadmap For Chelation Proof

### Current live-fire status

`run_live_fire_diagnostics.py` now validates the end-to-end control and reporting path without external services:

- baseline retrieval values via `InitialChelatedValues`
- query reformulation through `AntigravityEngine`
- adapter routing and route-effectiveness telemetry
- norm drift and stability diagnostics
- retrieval fitness, structural health, quantization gate, and storage metadata composition
- integrated diagnostics and adaptive gate recommendations
- dashboard summary aggregation over emitted events

The first deterministic run passed all hard checks and emitted two expected warnings: the tiny synthetic corpus saturated baseline retrieval (`fitness=1.0`), and its forced-noisy setup produced a CHELATE rate of `1.00`. That means the harness proves wiring and reporting, not retrieval lift.

### Calibration starting points

| Control | Current default/evidence | Recommended start | Adjust when |
|---|---:|---:|---|
| Chelation variance threshold | `0.0004`; presets `0.0002`, `0.0004`, `0.0008`; adaptive bounds `0.0001-0.01` | Start `0.0004`; explore `0.0002-0.001`; target 20-40% CHELATE rate | CHELATE rate drifts, threshold oscillation rises, or NDCG/recall drops |
| Quantization retention | retained-gain gate `0.8` | Require `0.8-0.9`; add minimum FP32 gain only after baseline campaigns | Quantized fitness loses FP32 gain or falls below baseline |
| Norm drift | advisory hard band `<0.5` or `>2.0` | Watch `0.75-1.33`; hard fail outside `0.5-2.0` | Repeated hard-band exits or monotonic query/result norm deltas |
| Route effectiveness | advisory disable below mean Jaccard `0.25` | Require at least 20 samples; warn below `0.5`, disable below `0.25` | Low route Jaccard combines with no NDCG lift or latency regression |
| Retrieval fitness | `0.6 NDCG@10 + 0.2 MRR + 0.2 Recall@K` | Promote only if at or above baseline, preferably +1-3% | NDCG/MRR/Recall fall more than 1-2%, empty results appear, or p95 latency regresses |
| Structural health | adaptive gate minimum `0.6` | Start min health `0.7`; never promote below `0.6` | Persistent collapse, isomer drift, or topology drift increases |

### P0 live-fire campaign gates

1. Freeze corpus, query set, qrels, config hash, and seed into `InitialChelatedValues`.
2. Run baseline retrieval and capture NDCG@10, MRR, Recall@K, latency, and action mix.
3. Enable chelation/adaptive controls and compare retrieval deltas against the frozen baseline.
4. Re-run candidate rankings through quantized simulation and require retained gain >= 80%.
5. Check structural health, norm drift, route effectiveness, and adaptive gate recommendations.
6. Publish a dated campaign result document and verification-log entry.

## Read Next

- [LLM Architecture And AI Engineering Adaptation Review](llm-architecture-ai-engineering-adaptation-review-2026-04-27.md)
- [Live-Fire Diagnostics And Calibration](live-fire-diagnostics-2026-04-27.md)
- [COMPUTATIONAL_STORAGE_DRIVE_NODES.md](COMPUTATIONAL_STORAGE_DRIVE_NODES.md)
- [roadmap-audit-and-weight-refinement-plan-2026-03-06.md](roadmap-audit-and-weight-refinement-plan-2026-03-06.md)
- [INDEX.md](INDEX.md)
