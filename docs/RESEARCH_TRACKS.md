# Research Tracks

This document describes the main research themes in the repository and where each one currently stands.

## Portfolio Summary

| Track | Primary question | Current state |
|---|---|---|
| Adaptive retrieval | Can noisy embedding neighborhoods be detected and corrected before they degrade retrieval? | Implemented and benchmarkable |
| Distillation and cross-lingual routing | Can teacher-guided correction generalize across models and languages? | Implemented and under evaluation |
| Online correction | Can inference-time updates improve quality without destabilizing the system? | Implemented, requires controlled ablation |
| Structural diagnostics | Can topology and isomer signals reveal degradation that ranking metrics miss? | Implemented and test-backed |
| Multi-dataset evaluation | Do improvements transfer beyond SciFact? | Implemented; campaign execution is ongoing work |
| Computational storage and drive nodes | Can some model or control-plane work move toward storage-resident execution? | Mixed maturity: software proof is strong, hardware claim remains scope-locked |
| Agentic remediation process | Can repository changes be triaged and delivered through a durable AEP workflow? | Implemented and extensively documented |

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

## 5. Computational Storage And Drive-Resident Nodes

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

## 6. Agentic Engineering And Planning

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

## Read Next

- [COMPUTATIONAL_STORAGE_DRIVE_NODES.md](COMPUTATIONAL_STORAGE_DRIVE_NODES.md)
- [roadmap-audit-and-weight-refinement-plan-2026-03-06.md](roadmap-audit-and-weight-refinement-plan-2026-03-06.md)
- [INDEX.md](INDEX.md)
