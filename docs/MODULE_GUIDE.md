# Module Guide

This guide groups the codebase by responsibility rather than by file age.

## Core Retrieval Runtime

| File | Responsibility |
|---|---|
| `antigravity_engine.py` | Main runtime for ingestion, retrieval, adaptive chelation, and training hooks |
| `embedding_backend.py` | Backend routing between local SentenceTransformers and Ollama HTTP embeddings |
| `vector_store.py` | Qdrant-backed vector-store abstraction |
| `chelation_adapter.py` | Near-identity adapter variants used for post-hoc correction |
| `config.py` | Central preset and validation surface |
| `checkpoint_manager.py` | Safe rollback and checkpoint integrity flow |
| `chelation_logger.py` | Structured JSON logging for runs and diagnostics |

## Distillation, Scheduling, And Adaptation

| File | Responsibility |
|---|---|
| `teacher_distillation.py` | Teacher-guided correction, dimension projection, hybrid target generation |
| `teacher_weight_scheduler.py` | Schedule families for teacher weighting |
| `cross_lingual_distillation.py` | Language-aware teacher routing and mapping |
| `sedimentation.py` | Earlier hierarchical sedimentation logic |
| `sedimentation_trainer.py` | Shared target and training helpers for correction |
| `online_updater.py` | Inference-time loss functions, schedulers, diagnostics, and updater runtime |
| `convergence_monitor.py` | Patience-oriented training stabilization |
| `dimension_mask_predictor.py` | Learned masking experiments |

## Structural Diagnostics

| File | Responsibility |
|---|---|
| `topology_analyzer.py` | Graph-like topology snapshots and connectivity metrics |
| `isomer_detector.py` | Isomer drift detection built on topology signals |
| `embedding_quality.py` | Per-document quality scoring |
| `stability_tracker.py` | Structural health and instability metrics |

## Evaluation And Experiment Drivers

| File | Responsibility |
|---|---|
| `benchmark_beir.py` | BEIR dataset registry, loading, execution, and reporting |
| `benchmark_comparative.py` | Side-by-side configuration comparisons |
| `benchmark_distillation.py` | Teacher-guided benchmarking |
| `benchmark_multitask.py` | Cross-task stability and learning-gain evaluation |
| `benchmark_evolution.py` | Earlier evaluation runner |
| `benchmark_rlm.py` | RLM-oriented evaluation harness |
| `benchmark_utils.py` | Shared benchmark helpers and data loading |
| `run_sweep.py` | Standard parameter sweep |
| `run_large_sweep.py` | Larger grid-search style run |
| `dashboard_server.py` | Local HTTP server for logs and result inspection |
| `dashboard/index.html` | Browser UI for dashboard views |

## Retrieval Extensions And Process Utilities

| File | Responsibility |
|---|---|
| `recursive_decomposer.py` | Recursive retrieval / decomposition engine |
| `aep_orchestrator.py` | Agentic Engineering and Planning workflow runtime |
| `language_detector.py` | Lightweight language detection used by cross-lingual routing |

## Computational Storage And Drive-Node Research

### Core POC files

| File | Responsibility |
|---|---|
| `computational_storage_poc/block_graph.py` | Packs layer matrices into flash-like blocks and replays graph traversal |
| `computational_storage_poc/compiler.py` | Compiles a graph payload into a binary image |
| `computational_storage_poc/train_and_compile.py` | Trains a digits classifier and compiles it into the block format |
| `computational_storage_poc/mock_nvme.py` | Mock storage inference path and theoretical host-vs-storage latency comparison |
| `computational_storage_poc/mock_array.py` | Multi-drive speculative racing simulation |
| `computational_storage_poc/CHELATEDAI_integration_demo.py` | Small bridge demo showing how recursive node requests could dispatch to a storage array |
| `computational_storage_poc/payload_contract.py` | Shared deterministic payload contract for firmware and emulator |
| `computational_storage_poc/usb_host_inference.py` | Host raw-sector reader and decoder |
| `computational_storage_poc/capture_hardware_evidence.py` | Evidence-capture tool for real RP2040 transport verification |

### Firmware and emulation

| Area | Responsibility |
|---|---|
| `computational_storage_poc/firmware/` | RP2040/TinyUSB USB mass-storage transport proof |
| `computational_storage_poc/emulation/` | File-backed emulation and validation flow |

## Documentation And Process Archives

| Area | Responsibility |
|---|---|
| `docs/README.md` | Canonical docs home |
| `docs/INDEX.md` | Broad document index |
| `docs/ARCH AGENTIC ENGINEERING AND PLANNING/` | Session logs, trackers, templates, verification logs, and process artifacts |
| `CLAUDE.md` | Internal operator guidance for coding sessions |

## Representative Tests

| Test file | What it validates |
|---|---|
| `test_unit_core.py` | Adapter variants and foundational runtime behavior |
| `test_antigravity_engine.py` | Retrieval-engine behavior under full ML dependencies |
| `test_benchmark_beir.py` | BEIR evaluation stack |
| `test_cross_lingual_distillation.py` | Language-aware teacher routing |
| `test_online_updater.py` | Online loss and update logic |
| `test_topology_analyzer.py` | Structural topology tracking |
| `test_computational_storage_poc.py` | Block-graph parity, latency invariants, digits round-trip |
| `test_computational_storage_payload.py` | Deterministic transport contract |
| `test_computational_storage_emulation.py` | Emulator semantics |
| `test_computational_storage_hardware_evidence.py` | Evidence capture and Windows raw-device handling |
