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

**Representative test files (`962` tests passing on this branch as of 2026-03-05):**
- `test_unit_core.py` - Core adapter variants
- `test_noise_injection.py` - Noise injection validation under `unittest`
- `test_online_updater.py`, `test_dimension_mask_predictor.py`, `test_stability_tracker.py`
- `test_benchmark_beir.py`, `test_benchmark_comparative.py`, `test_dashboard_server.py`
- `test_computational_storage_poc.py` - block-graph parity, latency invariants, and real-data storage round-trip validation
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
|-- chelation_adapter.py       # 3 adapter types via create_adapter() factory
|-- config.py                  # ChelationConfig with presets + validation
|-- chelation_logger.py        # get_logger() -> JSON structured logging
|-- teacher_distillation.py    # Offline/hybrid modes + DimensionProjection + EnsembleTeacherHelper
|-- teacher_weight_scheduler.py # 5 schedule types: constant/linear/cosine/step/adaptive
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

- **Adapter factory:** `create_adapter("mlp"|"procrustes"|"low_rank", input_dim)` returns one of three adapter types. All initialize near-identity (small weight std 0.001) to preserve base model quality.
- **Config presets:** `ChelationConfig.get_preset(name, type)` supports `chelation`, `adapter`, `convergence`, `adapter_type`, `rlm`, `sedimentation`, `sedimentation_tuned`, `ensemble`, `cross_lingual`, `teacher_weight_schedule`, `teacher_encoding`, `online_update`, `beir`, `topology`, and `isomer`.
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
- **Core modules avoid sklearn:** The computational-storage POC is the exception and uses `scikit-learn`'s digits dataset for round-trip validation.

## Git Workflow Notes

- The repository branch policy may still show PRs as blocked even after all required checks are green. Session 23 required admin merges for `#80`, `#83`, and `#82`.
- `gh pr merge` can fail if a local worktree is holding `main`. Before merging stacked PRs, remove/prune merged worktrees or switch them off `main`.

## Reference Material

- `rlm_reference/` -- Cloned RLM paper implementation (read-only, do not modify)
- `docs/rlm-analysis.md` -- Analysis of RLM source code
- `docs/REFERENCES.md` -- 17 research paper citations
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/` -- Full AEP workflow documentation (60+ files)
- `docs/INDEX.md` -- Documentation index
