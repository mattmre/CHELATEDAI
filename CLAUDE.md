# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# All tests (67 total, should all pass)
python -m pytest test_unit_core.py test_recursive_decomposer.py test_aep_orchestrator.py test_integration_rlm.py -v

# Unit tests only (no external services needed)
python -m pytest test_unit_core.py test_recursive_decomposer.py test_aep_orchestrator.py -v

# Integration tests (requires sentence-transformers installed)
python -m pytest test_integration_rlm.py -v

# Single test file
python -m pytest test_unit_core.py -v

# Single test method
python -m pytest test_unit_core.py::TestChelationAdapter::test_forward_shape -v

# Benchmarks (require Ollama docker + MTEB)
python benchmark_evolution.py --task SciFact --lr 0.5
python benchmark_rlm.py --task SciFact --decomposer mock --max-depth 3 --aggregation rrf
```

No `requirements.txt` or `pyproject.toml` exists. Install dependencies manually:
```bash
pip install numpy torch sentence-transformers qdrant-client mteb requests
```

## Architecture

ChelatedAI fixes semantic collapse in vector search through adaptive dimension masking and self-correcting embeddings. Three major subsystems:

### 1. Core Retrieval Engine
`antigravity_engine.py` is the central module. It embeds queries (dual-mode: Ollama HTTP or local SentenceTransformers), scouts a top-50 neighborhood from Qdrant, measures per-dimension variance, and takes one of two paths:
- **High variance** -> chelation path: masks noisy dimensions, reranks via center-of-mass spectral centering, logs noise centers to `chelation_log`
- **Low variance** -> fast path: trusts the quantized index directly

`chelation_adapter.py` is a small residual PyTorch module (Linear->ReLU->Linear + skip connection + L2 norm) that wraps around the base embedder. Identity-initialized so it starts as a no-op.

The **sedimentation cycle** (`run_sedimentation_cycle`) trains the adapter on accumulated noise centers from `chelation_log`, then rewrites corrected vectors back into Qdrant.

### 2. Recursive Decomposition (RLM Integration)
`recursive_decomposer.py` wraps `AntigravityEngine` via composition (zero modifications to the engine). It builds a decomposition tree:
- **Decomposers** (strategy pattern): `MockDecomposer` splits on conjunctions (deterministic, used in tests), `OllamaDecomposer` uses LLM via Ollama generate API
- **RecursiveRetrievalEngine** walks the tree, retrieves at leaf nodes, aggregates up via RRF, union, or intersection
- **HierarchicalSedimentationEngine** does variance-based recursive clustering (no sklearn) + two-phase adapter training

### 3. Agentic Remediation (ARCH-AEP)
`aep_orchestrator.py` encodes the 7-phase ARCH-AEP workflow as executable Python:
- Specialist agents (`ArchitectureAgent`, `SecurityAgent`, `TestingAgent`, `PerformanceAgent`) each implement `analyze()`, `propose_fix()`, `verify()`
- `AEPTracker` manages findings with severity tiers, markdown/JSON export
- `AEPOrchestrator.run_full_cycle()` executes: scope_lock -> discovery -> parallel_revalidation -> synthesis -> tiered_remediation -> verification -> closure

### Module Dependency Graph
```
antigravity_engine.py  <-- chelation_adapter.py
        ^
        |  (constructor injection)
recursive_decomposer.py  --> config.py, chelation_logger.py
aep_orchestrator.py       --> chelation_logger.py
benchmark_rlm.py          --> all of the above
```

## Conventions

- **Flat file structure**: all `.py` files at project root, no packages
- **Testing**: unittest-style (not pytest classes), run via pytest. Mock logger with `patch('module.get_logger')` returning `MagicMock()`
- **Dependencies**: numpy, torch, qdrant-client only. No sklearn. Hierarchical clustering is hand-rolled via recursive median-split
- **Config**: `ChelationConfig` (all static methods) in `config.py` with preset system: `get_preset(name, preset_type)` where `preset_type` is one of `chelation`, `adapter`, `rlm`, `sedimentation`
- **Logging**: `get_logger()` singleton from `chelation_logger.py`, structured JSON to JSONL files
- **Integration tests**: use in-memory Qdrant (`qdrant_location=":memory:"`) and local model `all-MiniLM-L6-v2`
- **Reference material**: `rlm_reference/` is a read-only clone of the official RLM repo -- never modify it

## ARCH-AEP Workflow

The `docs/ARCH AGENTIC ENGINEERING AND PLANNING/` directory contains the full workflow specification for agentic code remediation cycles. Key entry points:
- `orchestrator-briefing.md` -- session start narrative and folder index
- `workflow.md` -- 7-phase specification with guardrails
- `next-session.md` -- current state and session handoff checklist
- `tracker-pointer.md` -- authoritative link to the active tracker file
- `docs/INDEX.md` -- master index of all documentation

## Known Quirks

- `TestingAgent` class name in `aep_orchestrator.py` triggers a PytestCollectionWarning -- harmless, does not affect tests
- `GITHUBCHELATEDAIrlm_reference/` is a failed git clone artifact -- safe to delete, gitignored
- `README.md` file structure and test commands are partially outdated (references removed files like `homeostatic_engine.py`, `test_dynamic_adaptation.py`)
