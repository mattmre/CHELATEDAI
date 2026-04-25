# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

ChelatedAI is a research prototype for adaptive vector search with self-correcting embeddings. It detects "semantic collapse" in RAG systems (where unrelated concepts get similar embeddings) and fixes it through dynamic dimension masking and neural adaptation.

## Dependencies & Setup

Dependencies are managed via `pyproject.toml` (PEP 621) and `requirements.txt`:

```bash
pip install -r requirements.txt
# Or install as editable package:
pip install -e .
```

Optional: Ollama for Docker-based embeddings (`ollama:nomic-embed-text` model prefix).

## CI Pipeline

GitHub Actions workflow at `.github/workflows/test.yml`:
- **Lint job:** `ruff check .` on Python 3.11
- **Test job:** `unittest discover` across Python 3.9, 3.10, 3.11, 3.12 matrix

GitHub Actions workflow at `.github/workflows/build_firmware.yml`:
- **Firmware build job:** builds the RP2040/TinyUSB computational-storage firmware when `computational_storage_poc/firmware/**` changes and uploads UF2/ELF/BIN artifacts

GitHub Actions workflow at `.github/workflows/test.yml` also includes:
- **Computational-storage emulation job:** validates emulator semantics on hosted CI without privileged FUSE

## Running Tests

All tests use Python `unittest` (not pytest). CI does **not** install `pytest`, so do not add `pytest` imports or pytest-only fixtures to `test_*.py`. Run via CI or locally:

```bash
# Run a single test file
python test_unit_core.py

# Run a specific test class or method
python -m unittest test_unit_core.TestChelationAdapter.test_adapter_forward_pass

# Run all test files (bash glob)
for f in test_*.py; do python "$f"; done

# Discover and run all tests
python -m unittest discover -s . -p "test_*.py" -v
```

**Representative test files (`1082` tests passing on `main` as of 2026-03-12):**
- `test_unit_core.py` - Core adapter variants, BoundedAdapter, DimensionProjection training
- `test_noise_injection.py` - Noise injection validation under `unittest`
- `test_online_updater.py`, `test_dimension_mask_predictor.py`, `test_stability_tracker.py`
- `test_benchmark_beir.py`, `test_benchmark_comparative.py`, `test_dashboard_server.py`
- `test_sedimentation_loss.py` - InfoNCE, hybrid loss, hard negative miner, factory
- `test_kalman_lr.py` - Kalman-gain adaptive LR, variance behavior, clamping, engine integration
- `test_computational_storage_poc.py` - block-graph parity, latency invariants, and real-data storage round-trip validation
- `test_computational_storage_payload.py` - deterministic trigger-sector payload, host decoding, and virtual-disk interception validation
- `test_computational_storage_hardware_evidence.py` - deterministic evidence capture and Windows raw-device path handling
- `test_computational_storage_emulation.py` - dependency-light emulator parity and file-image validation
- `test_cross_lingual_distillation.py`, `test_language_detector.py`
- `test_topology_analyzer.py`, `test_isomer_detector.py`, `test_structural_health_report.py`
- `test_teacher_distillation.py`, `test_teacher_weight_scheduler.py`, `test_aep_orchestrator.py`

**Environment-dependent tests:** `test_antigravity_engine.py`, `test_adaptive_threshold.py`, and `test_memory_optimization.py` require full `torch` + `sentence-transformers` installed. They pass in CI but may fail locally without those dependencies.

## Architecture

### Flat file layout

All `.py` files live at the project root. No packages, no `__init__.py`. Imports are direct module references (e.g., `from antigravity_engine import AntigravityEngine`).

### Core module dependency graph

```
AntigravityEngine (antigravity_engine.py)  <- central entry point
|-- embedding_backend.py       # F-045: "ollama:model" -> HTTP, else -> SentenceTransformer
|-- vector_store.py            # F-044: QdrantVectorStore abstraction
|-- chelation_adapter.py       # 4 adapter types via create_adapter() factory + BoundedAdapter wrapper
|-- config.py                  # ChelationConfig with presets + validation
|-- chelation_logger.py        # get_logger() -> JSON structured logging
|-- teacher_distillation.py    # Offline/hybrid modes + DimensionProjection + EnsembleTeacherHelper
|-- teacher_weight_scheduler.py # 5 schedule types: constant/linear/cosine/step/adaptive
|-- sedimentation_loss.py      # InfoNCE, hybrid, hard-negative mining loss functions
|-- kalman_lr_scheduler.py     # Kalman-gain adaptive LR for sedimentation training
|-- sedimentation_trainer.py   # Shared homeostatic target logic
|-- checkpoint_manager.py      # SafeTrainingContext with SHA256 verification
|-- convergence_monitor.py     # Phase 1: patience-based early stopping
|-- online_updater.py          # Phase 3: inference-time micro-updates
|-- dimension_mask_predictor.py # Phase 4: learned dimension masking
|-- embedding_quality.py       # Phase 4: per-doc quality with decay weighting
`-- stability_tracker.py       # Phase 5: structural health diagnostics

RecursiveRetrievalEngine (recursive_decomposer.py)
|-- Uses AntigravityEngine for retrieval
|-- sedimentation.py           # HierarchicalSedimentationEngine
`-- config.py, chelation_logger.py, checkpoint_manager.py

AEPOrchestrator (aep_orchestrator.py)  <- 7-phase agentic remediation workflow
`-- chelation_logger.py

Dashboard & Sweeping (dashboard_server.py, run_sweep.py, run_large_sweep.py)
`-- Serves live metrics to localhost:8080 and manages parameter grid search.

Evaluation & Analysis Modules
|-- benchmark_beir.py                # Multi-dataset BEIR benchmarking + report generation
|-- cross_lingual_distillation.py    # Language-aware teacher routing for distillation
|-- language_detector.py             # Lightweight language detection with caching/fallbacks
|-- topology_analyzer.py             # Topology snapshots, bond matrices, cluster connectivity
`-- isomer_detector.py               # Query-result isomer detection built on topology signals
```

### Key design patterns

- **Adapter factory:** `create_adapter("mlp"|"procrustes"|"low_rank", input_dim, bounded=False)` returns one of three adapter types, optionally wrapped in `BoundedAdapter` for INT8-safe corrections. All initialize near-identity (small weight std 0.001) to preserve base model quality.
- **Config presets:** `ChelationConfig.get_preset(name, type)` supports `chelation`, `adapter`, `convergence`, `adapter_type`, `rlm`, `sedimentation`, `sedimentation_tuned`, `ensemble`, `cross_lingual`, `teacher_weight_schedule`, `teacher_encoding`, `online_update`, `beir`, `topology`, `isomer`, `bounded_adapter`, `sedimentation_loss`, and `kalman_lr`.
- **Sedimentation loss:** `engine.set_sedimentation_loss("mse"|"infonce"|"hybrid")` switches loss function. InfoNCE uses batch contrastive alignment; hybrid combines MSE + InfoNCE.
- **Kalman-gain adaptive LR:** `engine.enable_kalman_lr(process_noise, min_lr_ratio, max_lr_ratio)` modulates learning rate based on loss variance — high variance lowers LR, low variance raises it.
- **Noise Injection:** Dynamically scaled noise injection during sedimentation training.
- **Embedding backend routing:** Model names prefixed with `ollama:` use the HTTP API; all others use local SentenceTransformers.
- **Teacher distillation:** `DimensionProjection` for teacher-student dim mismatches, `EnsembleTeacherHelper` for multi-teacher weighted averaging, `TeacherWeightScheduler` for 5 dynamic schedule types.
- **AEP data model:** `Finding` objects with `Severity` (CRITICAL/HIGH/MEDIUM/LOW), `FindingStatus`, and `EffortSize` (S=1, M=3, L=5). Tiered remediation processes Critical->High->Medium->Low with no skipping.
- **Structural health reporting:** `antigravity_engine.py` now uses config-driven thresholds for persistent collapse, oscillation, topology cohesion, and isomer drift.

## Test Conventions

- **Mock logger:** `patch('module_name.get_logger')` returning `MagicMock()` -- used in nearly every test file.
- **In-memory Qdrant:** `qdrant_location=":memory:"` for isolation.
- **Local model for tests:** `model_name="all-MiniLM-L6-v2"` (requires sentence-transformers).
- **Temp files:** Tests use `tempfile` for filesystem isolation with cleanup in `tearDown`.
- **Sklearn usage:** Keep core runtime code on numpy/torch. `scikit-learn` is only acceptable in isolated validation flows such as the computational-storage digits test track.
- **Python 3.9 CI compatibility:** If a module imported by tests uses `X | None` annotations, add `from __future__ import annotations` or use `Optional[...]`.

## Git Workflow Notes

- The repository branch policy may still show PRs as blocked even after all required checks are green. Session 23 required admin merges for `#80`, `#83`, and `#82`.
- Session 27 required admin merges for `#90`, `#91`, `#92`, and `#93` even after all required checks passed.
- `gh pr merge` can fail if a local worktree is holding `main`. Before merging stacked PRs, remove/prune merged worktrees or switch them off `main`.
- The computational-storage split is complete on `main` as of 2026-03-06: `#86` landed the validation foundation, `#87` landed the payload transport path, and `#88` landed the session-wrap docs.
- Session 26 follow-up PRs `#90` (hardware evidence capture tool), `#91` (emulation CI), `#92` (transport scope lock), and `#93` (retention policy) are merged on `main` as of 2026-03-06. The dated retention review was executed and merged as PR `#108` on 2026-04-24; the remaining computational-storage follow-through is real hardware evidence capture on actual RP2040 hardware.
- Session 31 merged PRs `#96`–`#103` (weight refinement, docs cleanup, 4 features, session wrap). The adapter checkpoint contamination risk from Session 28 is resolved (isolated checkpoints per config).
- Session 32 ended with an explicit no-promotion outcome for preset/default changes. Do not treat `mlp` + `teacher_weight=0.3` as promotable until a second independent run plus multi-task/BEIR transfer checks confirm it. See `docs/weight-refinement-campaign-results-2026-04-25-session32.md`.
- Do not revive the stale computational-storage PR `#84` or the old `feat/session22-online-correction` branch line. If historical comparison is needed, use the local `backup/retired-*` refs instead.
- If no RP2040 device is attached, do not fabricate hardware evidence. Use `computational_storage_poc/capture_hardware_evidence.py` once actual hardware is available.
- Explicit Windows raw-device paths like `\\.\PhysicalDrive2` are valid inputs to `usb_host_inference.py` and `capture_hardware_evidence.py`; do not rewrite them into a second `PhysicalDrive` prefix.
- Do not treat unrelated removable USB storage as RP2040 evidence. Session 27's local probe only found a SanDisk removable drive, which was not used as a proxy.
- If a resumed benchmark campaign is no longer the active task, stop the live process instead of leaving it consuming CPU in the background.
- `ruff check` does not validate GitHub Actions YAML. Keep workflow-file review separate from Python lint.
- Local `git status` may show `?? .claude/`; that directory holds local worktree metadata and retired-branch artifacts and is not, by itself, a product-code diff.

## Reference Material

- `rlm_reference/` -- Cloned RLM paper implementation (read-only, do not modify)
- `docs/rlm-analysis.md` -- Analysis of RLM source code
- `docs/REFERENCES.md` -- 17 research paper citations
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/` -- Full AEP workflow documentation (60+ files)
- `docs/INDEX.md` -- Documentation index
