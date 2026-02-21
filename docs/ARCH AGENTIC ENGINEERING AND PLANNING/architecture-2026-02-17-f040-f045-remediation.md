# Architecture Plan: F-040..F-045 Remediation

**Date:** 2026-02-17  
**Cycle:** AEP-2026-02-13  
**Scope:** F-040, F-041, F-042, F-044, F-045

---

## Goals

1. Reduce unnecessary payload transfer/storage overhead during retrieval (F-040).
2. Remove benchmark utility duplication and centralize shared functions (F-041).
3. Move hierarchical sedimentation logic to a dedicated module while preserving imports (F-042).
4. Introduce dependency inversion for vector-store operations (F-044).
5. Extract embedding backend behavior behind explicit abstractions (F-045).

## Non-Goals

- No behavior changes to ranking logic or sedimentation math.
- No removal of existing public API symbols used by tests/callers.
- No broad refactor of `AntigravityEngine` beyond abstractions required by F-044/F-045.

---

## Dependency Graph

```text
F-041 ─┐
F-042 ─┼─ independent quick wins
F-040 ─┘

F-045 ──> F-044 pattern alignment (recommended order)

F-044 + F-045 enable future F-046 decomposition work.
```

---

## Detailed Design

## F-040: Payload minimization

### Design
- Add config flags to control payload persistence and retrieval:
  - `STORE_FULL_TEXT_IN_PAYLOAD` (default `True` for backward compatibility)
  - `INFERENCE_WITH_PAYLOAD` (default `False`)
  - `SCOUT_WITH_PAYLOAD` (default `False`)
- In ingestion paths, include `"text"` only when `STORE_FULL_TEXT_IN_PAYLOAD=True`.
- Keep metadata fields (e.g., `original_id`) untouched.
- In `query_points()` calls used for inference/scouting, explicitly pass `with_payload` from config.

### Compatibility
- Default behavior keeps text in payload unless config is changed.
- Existing tests expecting `payload["text"]` still pass under default config.

---

## F-041: Benchmark utility extraction

### Design
- Create `benchmark_utils.py` with shared functions:
  - `dcg_at_k`, `ndcg_at_k`, `find_keys`, `find_payload`, `load_mteb_data`
- Update `benchmark_rlm.py` and `benchmark_evolution.py` to import from `benchmark_utils`.
- Keep function signatures unchanged.

### Compatibility
- Re-export imported functions from `benchmark_rlm.py` so existing tests importing from `benchmark_rlm` continue to work.

---

## F-042: Hierarchical sedimentation relocation

### Design
- Create dedicated module `sedimentation.py` containing `HierarchicalSedimentationEngine`.
- Remove class definition from `recursive_decomposer.py`.
- Add compatibility import in `recursive_decomposer.py`:
  - `from sedimentation import HierarchicalSedimentationEngine`
- Move or adjust tests to import from new module where appropriate, but keep old import path valid.

### Compatibility
- Existing consumers importing from `recursive_decomposer` remain functional.

---

## F-045: Embedding backend abstraction

### Design
- Add `embedding_backend.py` with:
  - `EmbeddingBackend` protocol/ABC:
    - `embed(texts: List[str]) -> np.ndarray`
    - `vector_size` property
  - `OllamaEmbeddingBackend`
  - `LocalSentenceTransformerBackend`
  - `create_embedding_backend(model_name, logger, adapter)` factory
- `AntigravityEngine` delegates all embedding logic to backend instance.
- Keep `model_name` prefix parsing in factory, not scattered across engine methods.

### Compatibility
- Preserve current `model_name` contract (`ollama:*` vs local model).
- Keep output dtype/shape behavior consistent with existing tests.

---

## F-044: Vector store abstraction

### Design
- Add `vector_store.py` with:
  - `VectorStore` protocol/ABC for required operations:
    - `collection_exists`, `create_collection`
    - `query_points`, `retrieve`, `upsert`, `scroll`, `close`
  - `QdrantVectorStore` adapter wrapping `QdrantClient`
- `AntigravityEngine` owns `self.vector_store` and routes operations through it.
- Keep `self.qdrant` compatibility alias pointing to underlying client adapter to minimize ripple.

### Compatibility
- Existing tests patching `QdrantClient` continue to function.
- Existing direct `engine.qdrant` usages preserved for now.

---

## Migration Strategy

1. Implement F-041/F-042/F-040 as low-risk isolated commits.
2. Implement F-045 abstraction, keep engine behavior identical.
3. Implement F-044 abstraction with compatibility alias.
4. Run full regression after each finding and once at tranche end.

---

## Test Strategy

- Keep baseline green: `python -m pytest (Get-ChildItem -Name test_*.py) -q`
- Add targeted tests:
  - `test_antigravity_engine.py` for payload flags and backend/vector-store delegation.
  - `test_benchmark_rlm.py` and benchmark-specific tests for utility extraction.
  - `test_recursive_decomposer.py` (and/or new sedimentation test file) for moved class import compatibility.

---

## PR Stack Mapping

1. `pr/f041-benchmark-utils`
2. `pr/f042-hierarchical-module-relocation`
3. `pr/f040-payload-minimization`
4. `pr/f045-embedding-backend-abstraction`
5. `pr/f044-vector-store-inversion`
6. `pr/session9-tracking-docs`

Each PR contains one finding plus only necessary tests/docs updates.
