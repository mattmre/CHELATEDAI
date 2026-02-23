# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

ChelatedAI is a research prototype for adaptive vector search with self-correcting embeddings. It detects "semantic collapse" in RAG systems (where unrelated concepts get similar embeddings) and fixes it through dynamic dimension masking and neural adaptation.

## Dependencies & Setup

No requirements.txt or pyproject.toml exists. Install manually:

```bash
pip install numpy torch sentence-transformers qdrant-client mteb requests
```

Optional: Ollama for Docker-based embeddings (`ollama:nomic-embed-text` model prefix).

## Running Tests

All tests use Python `unittest` (not pytest). There is no unified test runner, Makefile, or CI pipeline. Run each test file individually:

```bash
# Run a single test file
python test_unit_core.py

# Run a specific test class or method
python -m unittest test_unit_core.TestChelationAdapter.test_adapter_forward_pass

# Run all test files (bash glob)
for f in test_*.py; do python "$f"; done
```

**Key test files (400+ tests passing):**
- `test_unit_core.py` (40) - Core adapter variants
- `test_noise_injection.py` (2) - Noise injection validation
- `test_convergence_monitor.py` (22) - Phase 1 early stopping
- `test_online_updater.py` (18) - Phase 3 online updates
- `test_dimension_mask_predictor.py` (26) - Phase 4 masking
- `test_stability_tracker.py` (19) - Phase 5 diagnostics
- `test_checkpoint_manager.py` (27) - Checkpoint/rollback
- `test_recursive_decomposer.py` (48) - Query decomposition
- `test_aep_orchestrator.py` (22) - Agentic workflow
- `test_benchmark_comparative.py` (23), `test_benchmark_utils.py` (26), `test_benchmark_rlm.py` (42)

**Known issues:** `test_antigravity_engine.py`, `test_adaptive_threshold.py`, `test_memory_optimization.py`, and integration tests have pre-existing failures due to huggingface-hub version incompatibility. The `PytestCollectionWarning` about `TestingAgent` is harmless.

## Architecture

### Flat file layout

All `.py` files live at the project root. No packages, no `__init__.py`. Imports are direct module references (e.g., `from antigravity_engine import AntigravityEngine`).

### Core module dependency graph

```
AntigravityEngine (antigravity_engine.py)  ← central entry point
├── embedding_backend.py       # F-045: "ollama:model" → HTTP, else → SentenceTransformer
├── vector_store.py            # F-044: QdrantVectorStore abstraction
├── chelation_adapter.py       # 3 adapter types via create_adapter() factory
├── config.py                  # ChelationConfig with presets + validation
├── chelation_logger.py        # get_logger() → JSON structured logging
├── teacher_distillation.py    # Offline/hybrid training modes
├── sedimentation_trainer.py   # Shared homeostatic target logic
├── checkpoint_manager.py      # SafeTrainingContext with SHA256 verification
├── convergence_monitor.py     # Phase 1: patience-based early stopping
├── online_updater.py          # Phase 3: inference-time micro-updates
├── dimension_mask_predictor.py # Phase 4: learned dimension masking
└── stability_tracker.py       # Phase 5: structural health diagnostics

RecursiveRetrievalEngine (recursive_decomposer.py)
├── Uses AntigravityEngine for retrieval
├── sedimentation.py           # HierarchicalSedimentationEngine
└── config.py, chelation_logger.py, checkpoint_manager.py

AEPOrchestrator (aep_orchestrator.py)  ← 7-phase agentic remediation workflow
└── chelation_logger.py

Dashboard & Sweeping (dashboard_server.py, run_large_sweep.py)
└── Serves live metrics to localhost:8080 and manages parameter grid search.
```

### Key design patterns

- **Adapter factory:** `create_adapter("mlp"|"procrustes"|"low_rank", input_dim)` returns one of three adapter types. All initialize near-identity (small weight std 0.001) to preserve base model quality.
- **Config presets:** `ChelationConfig.get_preset(name, type)` for `chelation`, `adapter`, `convergence`, `adapter_type`, `rlm`, `sedimentation` categories.
- **Noise Injection:** Dynamically scaled noise injection during sedimentation training.
- **Embedding backend routing:** Model names prefixed with `ollama:` use the HTTP API; all others use local SentenceTransformers.
- **AEP data model:** `Finding` objects with `Severity` (CRITICAL/HIGH/MEDIUM/LOW), `FindingStatus`, and `EffortSize` (S=1, M=3, L=5). Tiered remediation processes Critical→High→Medium→Low with no skipping.

## Test Conventions

- **Mock logger:** `patch('module_name.get_logger')` returning `MagicMock()` — used in nearly every test file.
- **In-memory Qdrant:** `qdrant_location=":memory:"` for isolation.
- **Local model for tests:** `model_name="all-MiniLM-L6-v2"` (requires sentence-transformers).
- **Temp files:** Tests use `tempfile` for filesystem isolation with cleanup in `tearDown`.
- **No sklearn:** Use numpy and torch only.

## Reference Material

- `rlm_reference/` — Cloned RLM paper implementation (read-only, do not modify)
- `docs/rlm-analysis.md` — Analysis of RLM source code
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/` — Full AEP workflow documentation (30+ files)
- `docs/INDEX.md` — Documentation index
