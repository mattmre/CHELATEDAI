# Research Artifact: F-040..F-045 Implementation Plan

**Date:** 2026-02-17  
**Cycle:** AEP-2026-02-13  
**Scope:** Low-priority performance and architecture findings (Tier 5)  
**Target Session:** Session 9  
**Current Baseline:** 438 tests passing (1 warning)  
**Backlog Status:** 40/55 resolved (15 remaining)

---

## Executive Summary

This research artifact covers the next 5 findings from the low-priority backlog. These are primarily architectural improvements and performance optimizations that will improve maintainability, reduce technical debt, and prepare the codebase for future scalability needs.

**Findings Summary:**
- **F-040:** Memory/bandwidth optimization (remove full document text from Qdrant payload)
- **F-041:** Code duplication removal (benchmark utilities consolidation)
- **F-042:** Module organization (relocate HierarchicalSedimentationEngine)
- **F-044:** Dependency inversion (abstract vector store interface)
- **F-045:** Backend abstraction (extract embedding mode branching)

**Total effort:** 3S + 2M (estimated 1-2 sessions)  
**Risk level:** Low-Medium (F-044/F-045 are medium effort with moderate risk)  
**Dependency status:** All unblocked; F-046 (god object decomposition) depends on F-044/F-045

**Strategic Notes:**
- F-040, F-041, F-042 are quick wins with minimal risk
- F-044 and F-045 are foundational refactorings that enable F-046 (deferred to later)
- All changes are backward-compatible and tested via existing regression suite

---

## Finding F-040: Full document text stored in Qdrant payload

### Current State Analysis

**Severity:** Low  
**Effort:** S  
**Impact:** Increased memory usage, network transfer overhead, and storage costs

**Current Behavior:**
The `ingest()` method stores complete document text in Qdrant payloads:

```python
# antigravity_engine.py:213 (within ingest method)
points.append(PointStruct(
    id=chunk["id"],
    vector=embeddings[i],
    payload={"text": chunk["text"], "metadata": chunk.get("metadata", {})}
))
```

**Actual Usage Patterns:**
- Most vector queries use `with_payload=False` (only vectors needed)
- Full text only retrieved in specific debugging/display scenarios
- Payloads are transferred on every Qdrant operation even when unused

**Cost Analysis:**
- Average document size: ~500-2000 characters
- Large collections (100K+ documents): 50-200MB of unnecessary payload data
- Network transfer overhead on every retrieval operation
- No current use case requires full text at query time

### Impacted Files and Symbols

**Primary:**
- `antigravity_engine.py:213` - `ingest()` method payload construction
- `antigravity_engine.py:268-287` - `get_chelated_vector()` (may retrieve payloads)
- `antigravity_engine.py:222-236` - `_gravity_sensor()` (query_points calls)
- `antigravity_engine.py:517-584` - `run_inference()` (query_points calls)

**Secondary:**
- `recursive_decomposer.py:454-459` - `_retrieve_for_node()` calls engine.run_inference
- `benchmark_rlm.py:318-321` - ID mapping via payload retrieval
- `benchmark_evolution.py:43-56` - payload retrieval for ID mapping

**Test Files:**
- `test_antigravity_engine.py` - Tests that verify ingest/retrieval behavior
- `test_integration_rlm.py` - Integration tests with full ingest/query cycle
- `test_benchmark_rlm.py` - Benchmark tests that may inspect payloads

### Constraints from Existing Tests

**Test Coverage:**
1. `test_antigravity_engine.py::test_ingest_validation_*` (8 tests) - validates ingest behavior
2. `test_antigravity_engine.py::test_run_inference_*` (6 tests) - validates retrieval paths
3. `test_integration_rlm.py::test_full_pipeline_*` (3 tests) - end-to-end with Qdrant
4. `test_benchmark_rlm.py::test_map_predicted_ids` - relies on payload['original_id']

**Critical Constraint:**
The `original_id` field in payload is used for benchmark ID mapping and **must be preserved**. Only the large `text` field should be removed or made optional.

### Risks

1. **Breaking Change Risk:** LOW
   - Existing code that assumes `payload["text"]` exists will break
   - Mitigation: Preserve small metadata fields, only remove large text field

2. **Feature Regression Risk:** LOW
   - Full text may be needed for debugging or future features
   - Mitigation: Add optional `with_full_text=False` parameter to ingest()

3. **Migration Risk:** LOW
   - Existing databases already have full text in payloads
   - Mitigation: Changes only affect new ingestions; existing data unaffected

### Validation Recommendations

**Test Strategy:**
1. Add unit test for `ingest()` with `include_full_text=False` parameter
2. Verify payload size reduction via mock assertions
3. Run full benchmark suite to ensure ID mapping still works
4. Integration test confirms retrieval works without text payload

**Regression Coverage:**
- All existing `test_antigravity_engine.py` tests must pass unchanged
- Benchmark tests must still complete successfully
- `test_integration_rlm.py` should show no behavioral change

**Performance Validation:**
- Measure payload size before/after on small test collection
- Verify network transfer reduction via Qdrant client metrics (if available)

---

## Finding F-041: Benchmark code duplication across two files

### Current State Analysis

**Severity:** Low  
**Effort:** S  
**Impact:** Code duplication leads to divergence risk and maintenance burden

**Duplicated Code:**

1. **`dcg_at_k` function** - Identical in both files:
   ```python
   # benchmark_rlm.py:26-32
   def dcg_at_k(r, k):
       r = np.asarray(r, dtype=float)[:k]
       if r.size:
           return np.sum(r / np.log2(np.arange(2, r.size + 2)))
       return 0.
   
   # benchmark_evolution.py:9-13
   def dcg_at_k(r, k):
       r = np.asarray(r, dtype=float)[:k]
       if r.size:
           return np.sum(r / np.log2(np.arange(2, r.size + 2)))
       return 0.
   ```

2. **`ndcg_at_k` function** - Identical in both files:
   ```python
   # benchmark_rlm.py:35-40
   def ndcg_at_k(r, k):
       dcg_max = dcg_at_k(sorted(r, reverse=True), k)
       if not dcg_max:
           return 0.
       return dcg_at_k(r, k) / dcg_max
   
   # benchmark_evolution.py:15-20
   def ndcg_at_k(r, k):
       dcg_max = dcg_at_k(sorted(r, reverse=True), k)
       if not dcg_max:
           return 0.
       return dcg_at_k(r, k) / dcg_max
   ```

3. **MTEB helper functions** - Similar but slightly different:
   - `find_keys()` - Identical in both files
   - `find_payload()` - Identical in both files
   - `load_mteb_data()` - Identical core logic, different comments

**Current State:**
- 5 functions duplicated across 2 files (~100 lines of code)
- Already tested in `test_benchmark_rlm.py` (39 tests cover the functions)
- No divergence detected yet, but risk increases over time

### Impacted Files and Symbols

**Primary:**
- `benchmark_rlm.py` - Lines 26-114 (duplicated functions section)
- `benchmark_evolution.py` - Lines 8-114 (duplicated functions section)
- **NEW:** `benchmark_utils.py` (to be created)

**Secondary:**
- All files that import from benchmark modules
- Test files that mock or patch these functions

**Test Files:**
- `test_benchmark_rlm.py` - Currently tests functions from benchmark_rlm
- Will need to update imports after extraction

### Constraints from Existing Tests

**Test Coverage:**
- `test_benchmark_rlm.py::TestDcgAtK` (7 tests) - Pure function tests
- `test_benchmark_rlm.py::TestNdcgAtK` (7 tests) - Pure function tests
- `test_benchmark_rlm.py::TestFindKeys` (6 tests) - Nested dict search tests
- `test_benchmark_rlm.py::TestFindPayload` (7 tests) - Recursive search tests
- `test_benchmark_rlm.py::TestMapPredictedIds` (5 tests) - ID mapping with mocks

**Critical Constraint:**
All 39 existing tests must continue to pass without modification. The extraction must maintain exact function signatures and behavior.

### Risks

1. **Import Chain Risk:** MEDIUM
   - Both benchmark files are imported in various test and script contexts
   - Mitigation: Create `benchmark_utils.py` and update imports in one atomic commit

2. **Circular Import Risk:** LOW
   - New shared module should not import from benchmark_rlm or benchmark_evolution
   - Mitigation: Only extract pure utility functions with no module-level dependencies

3. **Test Brittleness Risk:** LOW
   - Tests currently import from `benchmark_rlm`
   - Mitigation: Update test imports to use `benchmark_utils`

### Validation Recommendations

**Implementation Order:**
1. Create `benchmark_utils.py` with extracted functions
2. Add tests for benchmark_utils (reuse existing test cases)
3. Update `benchmark_rlm.py` to import from benchmark_utils
4. Update `benchmark_evolution.py` to import from benchmark_utils
5. Update test imports
6. Remove old duplicate definitions

**Test Strategy:**
1. Run `test_benchmark_rlm.py` before changes (baseline: all pass)
2. Create `test_benchmark_utils.py` by copying test cases
3. After extraction, verify all 39 tests still pass
4. Run full test suite to catch any import issues

**Regression Coverage:**
- All benchmark tests must pass unchanged
- No performance degradation in benchmark scripts
- Import paths should be the only visible change

---

## Finding F-042: HierarchicalSedimentationEngine in wrong module

### Current State Analysis

**Severity:** Low  
**Effort:** S  
**Impact:** Poor module organization; sedimentation logic mixed with decomposition logic

**Current Location:**
```python
# recursive_decomposer.py:578-773 (196 lines)
class HierarchicalSedimentationEngine:
    """
    Hierarchical sedimentation with variance-based clustering.
    """
    def __init__(self, engine: 'AntigravityEngine', ...):
        ...
    
    def run_cycle(self, threshold=10, epochs=50, learning_rate=0.01):
        """Full hierarchical sedimentation cycle."""
        ...
    
    def _simple_partition(self, vectors, indices, n_clusters):
        """Variance-based clustering."""
        ...
```

**Why It's Misplaced:**
- `recursive_decomposer.py` is about query decomposition and aggregation
- Sedimentation is about adapter training and vector refinement
- The module already has 773 lines; sedimentation adds significant bulk
- Existing `sedimentation_trainer.py` module already contains related helpers

**Architectural Context:**
- `sedimentation_trainer.py` (created in Session 3, F-011) contains:
  - `compute_homeostatic_target()` - target vector computation
  - `sync_vectors_to_qdrant()` - batch vector updates
- These are used by both `antigravity_engine.py` and `HierarchicalSedimentationEngine`
- Natural home: Move `HierarchicalSedimentationEngine` to same module

### Impacted Files and Symbols

**Primary:**
- **Source:** `recursive_decomposer.py:578-773` - HierarchicalSedimentationEngine class
- **Destination:** `sedimentation_trainer.py` (new location)
- **Imports:** `recursive_decomposer.py` imports section (lines 1-40)

**Secondary Files Importing HierarchicalSedimentationEngine:**
- `benchmark_evolution.py:94-104` - creates and uses engine
- `test_integration_rlm.py:165-189` - integration test for hierarchical sedimentation
- `test_recursive_decomposer.py:765-794` - unit test for SafeTrainingContext integration

**Current Import Pattern:**
```python
# benchmark_evolution.py
from recursive_decomposer import RecursiveRetrievalEngine, HierarchicalSedimentationEngine

# After refactoring:
from recursive_decomposer import RecursiveRetrievalEngine
from sedimentation_trainer import HierarchicalSedimentationEngine
```

### Constraints from Existing Tests

**Test Coverage:**
1. `test_integration_rlm.py::test_hierarchical_sedimentation_cycle` (1 test)
   - Tests full sedimentation cycle with real Qdrant
   - Verifies adapter training and vector updates
   - Must continue to work after move

2. `test_recursive_decomposer.py::test_hierarchical_sedimentation_uses_safe_training_context` (1 test)
   - Tests SafeTrainingContext integration
   - Verifies checkpoint creation during training
   - Must continue to work after move

3. `test_recursive_decomposer.py::TestHierarchicalSedimentationEngine` (class with multiple tests)
   - Edge case tests for clustering logic
   - Must remain in test_recursive_decomposer or move to new test file

**Critical Constraint:**
All existing tests must pass without modification except for updated imports.

### Risks

1. **Import Chain Risk:** MEDIUM
   - HierarchicalSedimentationEngine is currently exported from recursive_decomposer
   - Multiple files import it
   - Mitigation: Update all import statements in one commit

2. **Circular Import Risk:** MEDIUM
   - HierarchicalSedimentationEngine takes AntigravityEngine as constructor parameter
   - sedimentation_trainer.py must not create circular dependency
   - Mitigation: Use TYPE_CHECKING for forward references

3. **Test Organization Risk:** LOW
   - Tests are currently in test_recursive_decomposer.py
   - Decision needed: move tests or keep them there with updated imports
   - Mitigation: Keep tests in current location, only update imports

### Validation Recommendations

**Implementation Order:**
1. Add HierarchicalSedimentationEngine to sedimentation_trainer.py
2. Update imports in recursive_decomposer.py (keep backward-compat re-export)
3. Update imports in benchmark_evolution.py
4. Update imports in test files
5. Remove class from recursive_decomposer.py
6. Remove backward-compat re-export after verification

**Test Strategy:**
1. Run full test suite before move (baseline: 438 passing)
2. After move, run targeted tests:
   - `test_integration_rlm.py::test_hierarchical_sedimentation_cycle`
   - `test_recursive_decomposer.py::test_hierarchical_sedimentation_*`
3. Run full suite to catch import issues
4. Verify benchmark_evolution.py still runs

**Backward Compatibility:**
Optional: Keep temporary re-export in recursive_decomposer.py for one release:
```python
# recursive_decomposer.py (temporary backward compatibility)
from sedimentation_trainer import HierarchicalSedimentationEngine
__all__ = [..., 'HierarchicalSedimentationEngine']
```

---

## Finding F-044: No dependency inversion for vector store

### Current State Analysis

**Severity:** Low  
**Effort:** M  
**Impact:** Tight coupling to Qdrant prevents swapping vector store implementations

**Current Architecture:**
The `AntigravityEngine` directly instantiates and uses `QdrantClient`:

```python
# antigravity_engine.py:74-103
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class AntigravityEngine:
    def __init__(self, qdrant_location=":memory:", ...):
        # Direct instantiation
        if qdrant_location.startswith("http"):
            self.qdrant = QdrantClient(url=qdrant_location)
        else:
            self.qdrant = QdrantClient(path=qdrant_location)
        
        # Direct method calls throughout:
        self.qdrant.create_collection(...)      # Line 95
        self.qdrant.query_points(...)           # Lines 222-236, 517-584
        self.qdrant.retrieve(...)               # Lines 268-287
        self.qdrant.upsert(...)                 # Line 213
```

**Coupling Analysis:**
- 15+ direct method calls to QdrantClient throughout engine
- Collection creation logic hardcoded
- Query syntax specific to Qdrant (query_points, with_vectors, etc.)
- Point structure tied to Qdrant models
- No abstraction boundary between engine logic and storage layer

**Why This Matters:**
- Cannot swap to alternative vector stores (Weaviate, Pinecone, Milvus, etc.)
- Difficult to create lightweight test doubles
- Qdrant-specific behavior leaks into business logic
- Storage decisions tightly coupled to algorithm implementation

### Impacted Files and Symbols

**Primary:**
- `antigravity_engine.py` - 15+ direct QdrantClient usages
  - `__init__` (lines 74-103) - client instantiation and collection creation
  - `ingest()` (lines 194-220) - upsert operations
  - `get_chelated_vector()` (lines 268-287) - query + retrieve
  - `_gravity_sensor()` (lines 222-236) - query_points
  - `run_inference()` (lines 517-584) - query_points
  - `run_sedimentation_cycle()` (lines 347-484) - retrieve + upsert
  - `close()` (lines 586-591) - F-039 cleanup

**Secondary:**
- `sedimentation_trainer.py:69-132` - `sync_vectors_to_qdrant()` uses QdrantClient directly
- `recursive_decomposer.py:578-773` - HierarchicalSedimentationEngine delegates to engine.qdrant
- `checkpoint_manager.py` - No vector store coupling (good separation)

**Test Files:**
- `test_antigravity_engine.py` - Mocks QdrantClient in 15+ tests
- `test_integration_rlm.py` - Uses real Qdrant instance
- `test_recursive_decomposer.py` - Mocks engine.qdrant

### Proposed Architecture

**Create VectorStore ABC:**

```python
# NEW: vector_store.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class VectorPoint:
    """Storage-agnostic vector point representation."""
    id: Any
    vector: List[float]
    payload: Dict[str, Any]

@dataclass
class QueryResult:
    """Storage-agnostic query result."""
    id: Any
    score: float
    vector: Optional[List[float]] = None
    payload: Optional[Dict[str, Any]] = None

class VectorStore(ABC):
    """Abstract interface for vector storage backends."""
    
    @abstractmethod
    def create_collection(self, name: str, vector_size: int, 
                         distance: str = "cosine") -> None:
        """Create a new collection."""
        pass
    
    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        pass
    
    @abstractmethod
    def upsert(self, collection: str, points: List[VectorPoint]) -> None:
        """Insert or update vectors."""
        pass
    
    @abstractmethod
    def query(self, collection: str, query_vector: List[float], 
              limit: int = 10, with_vectors: bool = False,
              with_payload: bool = True) -> List[QueryResult]:
        """Query for similar vectors."""
        pass
    
    @abstractmethod
    def retrieve(self, collection: str, ids: List[Any],
                with_vectors: bool = False) -> List[QueryResult]:
        """Retrieve specific points by ID."""
        pass
    
    @abstractmethod
    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Get collection metadata."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Cleanup resources."""
        pass
```

**Qdrant Adapter Implementation:**

```python
# NEW: qdrant_adapter.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from vector_store import VectorStore, VectorPoint, QueryResult

class QdrantVectorStore(VectorStore):
    """Qdrant implementation of VectorStore interface."""
    
    def __init__(self, location: str):
        if location.startswith("http"):
            self.client = QdrantClient(url=location)
        else:
            self.client = QdrantClient(path=location)
    
    def create_collection(self, name: str, vector_size: int,
                         distance: str = "cosine") -> None:
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map.get(distance, Distance.COSINE)
            )
        )
    
    def collection_exists(self, name: str) -> bool:
        return self.client.collection_exists(name)
    
    def upsert(self, collection: str, points: List[VectorPoint]) -> None:
        qdrant_points = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ]
        self.client.upsert(collection_name=collection, points=qdrant_points)
    
    def query(self, collection: str, query_vector: List[float],
              limit: int = 10, with_vectors: bool = False,
              with_payload: bool = True) -> List[QueryResult]:
        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            with_vectors=with_vectors,
            with_payload=with_payload
        )
        return [
            QueryResult(
                id=p.id,
                score=p.score,
                vector=p.vector if with_vectors else None,
                payload=p.payload if with_payload else None
            )
            for p in results.points
        ]
    
    def retrieve(self, collection: str, ids: List[Any],
                with_vectors: bool = False) -> List[QueryResult]:
        points = self.client.retrieve(
            collection_name=collection,
            ids=ids,
            with_vectors=with_vectors
        )
        return [
            QueryResult(id=p.id, score=0.0, vector=p.vector, payload=p.payload)
            for p in points
        ]
    
    def get_collection_info(self, name: str) -> Dict[str, Any]:
        info = self.client.get_collection(name)
        return {
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance.name,
            "points_count": info.points_count
        }
    
    def close(self) -> None:
        self.client.close()
```

**Update AntigravityEngine:**

```python
# antigravity_engine.py
from vector_store import VectorStore, VectorPoint, QueryResult
from qdrant_adapter import QdrantVectorStore

class AntigravityEngine:
    def __init__(self, vector_store: Optional[VectorStore] = None,
                 qdrant_location: str = ":memory:", ...):
        # Dependency injection with backward compatibility
        if vector_store is None:
            self.vector_store = QdrantVectorStore(qdrant_location)
        else:
            self.vector_store = vector_store
        
        # Replace all self.qdrant.* calls with self.vector_store.*
```

### Constraints from Existing Tests

**Test Coverage:**
- `test_antigravity_engine.py` - 50+ tests mock QdrantClient
  - All mocks will need updating to mock VectorStore interface
  - Or: Keep using QdrantVectorStore and mock the Qdrant client internally

**Test Strategy Options:**
1. **Option A (Recommended):** Mock VectorStore ABC in tests
   - Cleaner test isolation
   - Tests engine logic without Qdrant specifics
   - More refactoring required

2. **Option B:** Continue using QdrantVectorStore, mock underlying client
   - Less test churn
   - Still tests some Qdrant-specific behavior
   - Easier migration path

**Integration Tests:**
- `test_integration_rlm.py` uses real Qdrant - should continue to work
- May want to add integration test with mock VectorStore implementation

### Risks

1. **Breaking Change Risk:** HIGH
   - This is a significant architectural change
   - All code that directly accesses `engine.qdrant` will break
   - Mitigation: Keep `engine.qdrant` as deprecated property pointing to underlying client

2. **Test Maintenance Risk:** HIGH
   - 50+ tests mock QdrantClient
   - All mocks need updating or new test doubles created
   - Mitigation: Implement in stages, update tests incrementally

3. **Performance Risk:** LOW
   - Thin abstraction layer adds minimal overhead
   - Most operations remain identical
   - Mitigation: Benchmark before/after to verify no regression

4. **Scope Creep Risk:** MEDIUM
   - Easy to over-engineer the abstraction
   - Mitigation: Start with minimal interface covering current usage only

### Validation Recommendations

**Implementation Phases:**

**Phase 1: Create Abstraction (Non-Breaking)**
1. Create `vector_store.py` with ABC
2. Create `qdrant_adapter.py` with QdrantVectorStore
3. Add unit tests for QdrantVectorStore adapter
4. No changes to AntigravityEngine yet

**Phase 2: Add Backward-Compatible Injection**
1. Add optional `vector_store` parameter to AntigravityEngine.__init__
2. Default to creating QdrantVectorStore(qdrant_location)
3. Keep `self.qdrant` as property for backward compatibility:
   ```python
   @property
   def qdrant(self):
       """Deprecated: Use vector_store instead."""
       if isinstance(self.vector_store, QdrantVectorStore):
           return self.vector_store.client
       raise AttributeError("Direct qdrant access not available")
   ```
4. Run full test suite - should pass unchanged

**Phase 3: Migrate Internal Usage**
1. Replace `self.qdrant.*` calls with `self.vector_store.*`
2. Update one method at a time
3. Run tests after each method migration
4. Update sedimentation_trainer.py to accept VectorStore

**Phase 4: Update Tests (Future)**
1. Create MockVectorStore test double
2. Update test_antigravity_engine.py to use MockVectorStore
3. Remove Qdrant-specific test dependencies

**Test Strategy:**
1. **Adapter tests:** Unit tests for QdrantVectorStore adapter
2. **Backward compat:** All existing tests pass with no changes
3. **Integration:** test_integration_rlm.py continues to work
4. **Regression:** Full test suite passes after each phase

**Success Criteria:**
- [ ] VectorStore ABC defined with minimal interface
- [ ] QdrantVectorStore adapter passes unit tests
- [ ] AntigravityEngine accepts VectorStore via dependency injection
- [ ] All 438 existing tests pass unchanged
- [ ] Backward compatibility maintained via deprecated properties
- [ ] Documentation updated with migration guide

---

## Finding F-045: Embedding mode branching via string prefix

### Current State Analysis

**Severity:** Low  
**Effort:** M  
**Impact:** Conditional branching scattered throughout engine makes adding new backends difficult

**Current Architecture:**

The engine uses string prefix checking to branch between Ollama and local modes:

```python
# antigravity_engine.py:29-64
if model_name.startswith("ollama:"):
    # Ollama Mode - 30+ lines of initialization
    self.mode = "ollama"
    self.model_name = model_name.replace("ollama:", "")
    self.ollama_url = ChelationConfig.OLLAMA_URL
    # ... connection test, vector size detection, etc.
else:
    # Local/Torch Mode - 20+ lines
    self.mode = "local"
    from sentence_transformers import SentenceTransformer
    self.model = SentenceTransformer(...)
    # ... vector size from model, device setup, etc.

# antigravity_engine.py:105-192 - embed() method
def embed(self, texts):
    if self.mode == "ollama":
        # 70+ lines of Ollama-specific logic
        # ThreadPoolExecutor, requests, retries, etc.
        ...
    else:
        # 20+ lines of local mode logic
        # SentenceTransformer.encode(), device management, etc.
        ...
```

**Branching Locations:**
- `__init__` (lines 29-64): Mode detection and initialization
- `embed()` (lines 105-192): Dual implementation based on self.mode
- Mode-specific error handling throughout

**Why This Is Problematic:**
- Adding new embedding backend (OpenAI, Cohere, etc.) requires modifying engine
- Branching logic violates Open/Closed Principle
- Cannot easily test embedding backends in isolation
- Difficult to compose or switch backends at runtime

### Impacted Files and Symbols

**Primary:**
- `antigravity_engine.py:29-64` - Mode detection in __init__
- `antigravity_engine.py:105-192` - Dual implementation in embed()
- `config.py` - Embedding-related configuration constants

**Secondary:**
- All code that instantiates AntigravityEngine (tests, benchmarks, scripts)
- Mock setup in tests that stub embedding behavior

**Test Files:**
- `test_antigravity_engine.py::test_embed_local_mode_*` (10 tests)
- `test_antigravity_engine.py::test_embed_ollama_mode_*` (8 tests)
- Tests rely on model_name prefix to determine mocking strategy

### Proposed Architecture

**Create EmbeddingBackend ABC:**

```python
# NEW: embedding_backend.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingBackend(ABC):
    """Abstract interface for embedding generation."""
    
    @abstractmethod
    def get_vector_size(self) -> int:
        """Return the dimensionality of embeddings."""
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Returns:
            np.ndarray of shape (len(texts), vector_size) with dtype float32
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Cleanup resources."""
        pass
```

**Ollama Backend Implementation:**

```python
# NEW: ollama_backend.py
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List
from embedding_backend import EmbeddingBackend
from config import ChelationConfig
from chelation_logger import get_logger

class OllamaEmbeddingBackend(EmbeddingBackend):
    """Ollama embedding backend with retry and truncation."""
    
    def __init__(self, model_name: str, ollama_url: str = None):
        self.model_name = model_name
        self.ollama_url = ollama_url or ChelationConfig.OLLAMA_URL
        self.logger = get_logger()
        
        # Detect vector size
        try:
            test_vec = self.embed(["test"])[0]
            self._vector_size = len(test_vec)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.ollama_url}"
            ) from e
        except Exception as e:
            self.logger.log_error("connection", f"Ollama connection test failed: {e}")
            self._vector_size = ChelationConfig.DEFAULT_VECTOR_SIZE
    
    def get_vector_size(self) -> int:
        return self._vector_size
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using Ollama API with retries and truncation.
        
        Implements the full retry logic from current AntigravityEngine.embed():
        - Parallel requests via ThreadPoolExecutor
        - Automatic truncation on context limit errors
        - Zero-vector fallback for failed embeddings
        """
        embeddings = np.zeros((len(texts), self._vector_size), dtype=np.float32)
        
        def _get_embedding(i: int, txt: str):
            def attempt(text):
                try:
                    res = requests.post(
                        self.ollama_url,
                        json={
                            "model": self.model_name,
                            "prompt": text,
                            "options": {"num_ctx": ChelationConfig.OLLAMA_NUM_CTX}
                        },
                        timeout=ChelationConfig.OLLAMA_TIMEOUT
                    )
                    if res.status_code == 200:
                        return np.array(res.json()["embedding"], dtype=np.float32)
                    return None
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                       KeyError) as e:
                    self.logger.log_error("embedding", f"Ollama error for doc {i}", exception=e)
                    return None
            
            # Try truncation levels
            for limit in ChelationConfig.OLLAMA_TRUNCATION_LIMITS:
                emb = attempt(txt[:limit])
                if emb is not None:
                    return i, emb
            
            # Fallback to zero vector
            self.logger.log_error("embedding_failed", f"Failed doc {i} after retries")
            return i, np.zeros(self._vector_size, dtype=np.float32)
        
        with ThreadPoolExecutor(max_workers=ChelationConfig.OLLAMA_MAX_WORKERS) as executor:
            futures = [executor.submit(_get_embedding, i, txt) for i, txt in enumerate(texts)]
            for idx, future in enumerate(futures):
                try:
                    _, emb = future.result(timeout=ChelationConfig.OLLAMA_TIMEOUT)
                    embeddings[idx] = emb
                except TimeoutError:
                    self.logger.log_error("timeout", f"Embedding timeout for doc {idx}")
                    embeddings[idx] = np.zeros(self._vector_size, dtype=np.float32)
        
        return embeddings
    
    def close(self) -> None:
        """No cleanup needed for stateless HTTP client."""
        pass
```

**Local (SentenceTransformers) Backend:**

```python
# NEW: local_backend.py
import numpy as np
from typing import List
from embedding_backend import EmbeddingBackend
from sentence_transformers import SentenceTransformer
from chelation_logger import get_logger

class LocalEmbeddingBackend(EmbeddingBackend):
    """Local SentenceTransformers embedding backend."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.logger = get_logger()
        
        self.model = SentenceTransformer(model_name, device=device)
        self._vector_size = self.model.get_sentence_embedding_dimension()
        
        self.logger.log_event(
            "initialization",
            f"Loaded local model: {model_name}",
            model_name=model_name,
            vector_size=self._vector_size,
            device=self.device
        )
    
    def get_vector_size(self) -> int:
        return self._vector_size
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local SentenceTransformer model."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device
        )
        return embeddings.astype(np.float32)
    
    def close(self) -> None:
        """No cleanup needed for local model."""
        pass
```

**Backend Factory:**

```python
# NEW: embedding_factory.py
from typing import Optional
from embedding_backend import EmbeddingBackend
from ollama_backend import OllamaEmbeddingBackend
from local_backend import LocalEmbeddingBackend

def create_embedding_backend(model_name: str, 
                            device: str = "cpu") -> EmbeddingBackend:
    """
    Factory function to create appropriate embedding backend.
    
    Supports:
    - "ollama:MODEL_NAME" -> OllamaEmbeddingBackend
    - Any other string -> LocalEmbeddingBackend (SentenceTransformers)
    """
    if model_name.startswith("ollama:"):
        actual_model = model_name.replace("ollama:", "")
        return OllamaEmbeddingBackend(actual_model)
    else:
        return LocalEmbeddingBackend(model_name, device=device)
```

**Update AntigravityEngine:**

```python
# antigravity_engine.py
from embedding_backend import EmbeddingBackend
from embedding_factory import create_embedding_backend

class AntigravityEngine:
    def __init__(self, 
                 embedding_backend: Optional[EmbeddingBackend] = None,
                 model_name: str = 'ollama:nomic-embed-text',
                 qdrant_location: str = ":memory:",
                 ...):
        """
        embedding_backend: Optional injected backend. If None, created from model_name.
        model_name: Used only if embedding_backend is None (backward compatibility).
        """
        # Dependency injection with backward compatibility
        if embedding_backend is None:
            self.embedding_backend = create_embedding_backend(model_name)
        else:
            self.embedding_backend = embedding_backend
        
        self.vector_size = self.embedding_backend.get_vector_size()
        
        # Remove all mode-specific initialization
        # Remove self.mode, self.model_name, self.ollama_url, etc.
    
    def embed(self, texts):
        """Simplified: delegate to backend."""
        return self.embedding_backend.embed(texts)
    
    def close(self):
        """Cleanup both vector store and embedding backend."""
        if hasattr(self, 'embedding_backend'):
            self.embedding_backend.close()
        if hasattr(self, 'vector_store'):
            self.vector_store.close()
```

### Constraints from Existing Tests

**Test Coverage:**
- `test_antigravity_engine.py` has 18 tests for embedding behavior
  - 10 local mode tests
  - 8 Ollama mode tests
- Tests currently mock at the library level (requests, SentenceTransformer)
- After refactoring, can mock at backend level (cleaner)

**Test Strategy Options:**

1. **Option A (Recommended):** Mock EmbeddingBackend in tests
   ```python
   def test_embed_calls_backend(self):
       mock_backend = MagicMock(spec=EmbeddingBackend)
       mock_backend.get_vector_size.return_value = 768
       mock_backend.embed.return_value = np.zeros((5, 768), dtype=np.float32)
       
       engine = AntigravityEngine(embedding_backend=mock_backend)
       result = engine.embed(["a", "b", "c", "d", "e"])
       
       mock_backend.embed.assert_called_once()
       self.assertEqual(result.shape, (5, 768))
   ```

2. **Option B:** Keep existing library-level mocks
   - Tests continue to mock requests and SentenceTransformer
   - Factory creates real backends, but underlying libraries are mocked
   - Less test churn

### Risks

1. **Breaking Change Risk:** HIGH
   - Major refactoring of core engine functionality
   - All code that relies on engine.mode or engine.model will break
   - Mitigation: Keep deprecated properties for backward compatibility

2. **Test Maintenance Risk:** HIGH
   - 18 embedding tests need updating
   - Mock strategy changes significantly
   - Mitigation: Implement in phases, update tests incrementally

3. **Regression Risk:** MEDIUM
   - Embedding behavior is critical path for all operations
   - Any bug will cause widespread failures
   - Mitigation: Extensive validation at each step

4. **Performance Risk:** LOW
   - Thin abstraction layer adds minimal overhead
   - Backend implementations preserve existing logic
   - Mitigation: Benchmark embed() latency before/after

### Validation Recommendations

**Implementation Phases:**

**Phase 1: Create Abstraction (Non-Breaking)**
1. Create `embedding_backend.py` with ABC
2. Create `ollama_backend.py` - extract Ollama logic from engine
3. Create `local_backend.py` - extract local logic from engine
4. Create `embedding_factory.py`
5. Add unit tests for each backend in isolation
6. No changes to AntigravityEngine yet

**Phase 2: Add Backward-Compatible Injection**
1. Add optional `embedding_backend` parameter to AntigravityEngine.__init__
2. Default to calling `create_embedding_backend(model_name)`
3. Keep `self.mode` as deprecated property
4. Run full test suite - should pass unchanged

**Phase 3: Migrate embed() Method**
1. Replace embed() implementation with `return self.embedding_backend.embed(texts)`
2. Remove mode-specific branching
3. Run targeted embedding tests after change
4. Run full test suite

**Phase 4: Cleanup (Future)**
1. Remove deprecated properties (self.mode, self.model_name, etc.)
2. Update all tests to use backend mocking
3. Add tests for new backend implementations (OpenAI, etc.)

**Test Strategy:**

1. **Backend Unit Tests:**
   - `test_ollama_backend.py` - Test Ollama retry logic, truncation, etc.
   - `test_local_backend.py` - Test SentenceTransformers wrapping
   - Mock underlying libraries (requests, SentenceTransformer)

2. **Integration Tests:**
   - `test_embedding_factory.py` - Test factory creates correct backends
   - `test_integration_rlm.py` - End-to-end with real backends

3. **Backward Compatibility:**
   - All existing engine tests pass with no changes
   - model_name parameter still works as before

**Success Criteria:**
- [ ] EmbeddingBackend ABC defined
- [ ] OllamaEmbeddingBackend passes unit tests
- [ ] LocalEmbeddingBackend passes unit tests
- [ ] Factory creates correct backends based on model_name
- [ ] AntigravityEngine accepts backend via dependency injection
- [ ] All 438 existing tests pass unchanged
- [ ] embed() performance unchanged (benchmark)
- [ ] Documentation updated

---

## Cross-Finding Dependencies and Implementation Order

### Dependency Graph

```
F-040 (payload optimization) - Independent
F-041 (benchmark utils) - Independent
F-042 (module relocation) - Independent
F-044 (vector store abstraction) - Blocks F-046 (future)
F-045 (embedding backend abstraction) - Blocks F-046 (future)
```

**No blocking dependencies within this tranche.** All findings can be implemented in parallel or any order.

### Recommended Implementation Order

**Session 9 - Quick Wins (F-040, F-041, F-042):**
1. **F-041** - Benchmark utils extraction (lowest risk, highest value for developers)
2. **F-042** - Module relocation (improves code organization for F-044/F-045 work)
3. **F-040** - Payload optimization (simple, concrete improvement)

**Session 10 - Architectural Refactoring (F-044, F-045):**
4. **F-045** - Embedding backend abstraction (simpler than F-044, good warm-up)
5. **F-044** - Vector store abstraction (more complex, builds on F-045 patterns)

**Rationale:**
- Tackle low-risk, high-value items first to build momentum
- Get organizational cleanup (F-042) done before architectural changes
- Leave architectural abstractions (F-044, F-045) for when more time is available
- F-044 and F-045 together provide foundation for future F-046 (god object decomposition)

---

## Test Impact Analysis

### Current Test Baseline
- **Total:** 438 tests passing (1 warning)
- **Coverage:** Core modules well-tested, integration paths validated
- **Runtime:** ~15-30 seconds for full suite

### Expected Test Changes

**F-040 (Payload Optimization):**
- New tests: +3 (payload validation, size verification, with_payload parameter)
- Modified tests: 0 (backward compatible)
- Risk: Low

**F-041 (Benchmark Utils):**
- New tests: +5 (benchmark_utils module tests, duplicate of existing)
- Modified tests: ~10 (update imports)
- Risk: Low

**F-042 (Module Relocation):**
- New tests: 0
- Modified tests: ~5 (update imports)
- Risk: Low

**F-044 (Vector Store Abstraction):**
- New tests: +15 (VectorStore ABC tests, QdrantVectorStore adapter tests)
- Modified tests: ~50 (update mocks from QdrantClient to VectorStore)
- Risk: Medium-High

**F-045 (Embedding Backend):**
- New tests: +12 (backend unit tests, factory tests)
- Modified tests: ~18 (update embedding test mocks)
- Risk: Medium-High

**Total for Tranche:**
- New tests: +35
- Modified tests: ~83
- Final count estimate: 473 tests

---

## Risk Mitigation Strategy

### High-Risk Items (F-044, F-045)

**Mitigation Measures:**
1. **Phased Implementation:** Implement abstraction layer without breaking existing code first
2. **Backward Compatibility:** Keep deprecated accessors (.qdrant, .mode) during transition
3. **Incremental Migration:** Update one method at a time with test validation
4. **Feature Flags:** Consider adding config flag to enable/disable abstraction layer
5. **Extensive Validation:** Run full test suite + benchmarks after each phase

### Medium-Risk Items (F-040, F-041, F-042)

**Mitigation Measures:**
1. **Atomic Commits:** One finding per commit with full test validation
2. **Backward Compatibility:** F-040 adds optional parameter, existing behavior unchanged
3. **Import Safety:** F-041/F-042 update all imports in single commit
4. **Rollback Plan:** Each change can be reverted independently

---

## Success Metrics

### Quantitative Metrics
- [ ] All 438 existing tests pass after each finding
- [ ] New tests bring total to ~473 tests
- [ ] No performance regression (benchmark runtime within 5%)
- [ ] Code duplication reduced (F-041: -100 lines)
- [ ] Module size reduced (F-042: recursive_decomposer.py -196 lines)

### Qualitative Metrics
- [ ] Code organization improved (F-042)
- [ ] Maintainability improved (F-041, F-044, F-045)
- [ ] Extensibility improved (F-044, F-045 enable new backends)
- [ ] Technical debt reduced
- [ ] Documentation updated for all changes

---

## Documentation Requirements

### Code Documentation
- [ ] Docstrings for all new classes (VectorStore, EmbeddingBackend, etc.)
- [ ] Inline comments explaining design decisions
- [ ] Migration guide for F-044/F-045 (how to create custom backends)

### Architecture Documentation
- [ ] Update architecture diagrams to show new abstraction layers
- [ ] Document abstraction boundaries and extension points
- [ ] Add examples of custom backend implementations

### User Documentation
- [ ] Update README with new initialization options (dependency injection)
- [ ] Add examples of using custom vector stores and embedding backends
- [ ] Document backward compatibility and deprecation timeline

---

## Handoff to Architecture Agent

### Key Decisions Needed

1. **F-044 - VectorStore Interface:**
   - Confirm minimal interface covers all current use cases
   - Decide on backward compatibility strategy
   - Determine migration timeline for test updates

2. **F-045 - EmbeddingBackend Interface:**
   - Confirm interface sufficient for OpenAI/Cohere future support
   - Decide on error handling strategy in backends vs engine
   - Determine device management for local backends

3. **F-042 - Test Organization:**
   - Keep HierarchicalSedimentationEngine tests in test_recursive_decomposer.py?
   - Or create new test_sedimentation_trainer.py?

### Implementation Artifacts to Create

**For Quick Wins (F-040, F-041, F-042):**
- Minimal architecture notes (pattern to follow)
- Direct implementation based on this research

**For Architectural Changes (F-044, F-045):**
- Detailed architecture decision documents
- Interface specifications
- Migration plan with rollback strategy
- Test strategy document

### Next Steps

1. **Session 9 Goal:** Complete F-040, F-041, F-042 (quick wins)
2. **Session 10 Planning:** Detailed architecture review for F-044/F-045
3. **F-046 Preparation:** F-044/F-045 lay groundwork for god object decomposition

---

## Appendix: Current Code Statistics

### Module Sizes (lines of code)
- `antigravity_engine.py`: 591 lines
- `recursive_decomposer.py`: 773 lines
- `benchmark_rlm.py`: 555 lines
- `benchmark_evolution.py`: 280 lines
- `sedimentation_trainer.py`: 132 lines

### Duplicated Code (F-041)
- `dcg_at_k`: 7 lines × 2 = 14 lines
- `ndcg_at_k`: 6 lines × 2 = 12 lines
- `find_keys`: 8 lines × 2 = 16 lines
- `find_payload`: 10 lines × 2 = 20 lines
- `load_mteb_data`: 30 lines × 2 = 60 lines
- **Total duplication:** ~122 lines

### Abstraction Opportunities
- Vector store operations: 15+ call sites in antigravity_engine.py
- Embedding operations: 2 major implementations (~90 lines combined)
- Sedimentation logic: Shared via sedimentation_trainer.py (F-011 resolved)

---

## References

- Backlog: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/backlog-2026-02-13.md`
- Session 8 Log: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/session-log-2026-02-17-impl-8.md`
- Previous Research: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/research-2026-02-17-f025-f039-implementation.md`
- Previous Architecture: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/architecture-2026-02-17-f025-f039-remediation.md`
- Test Baseline: 438 passing, 1 warning (pytest output from Session 8)

---

**End of Research Artifact**

**Status:** Ready for Architecture Agent review  
**Next Agent:** Architecture Agent (for detailed design of F-044/F-045) or Implementer Agent (for F-040/F-041/F-042)  
**Session 9 Priority:** F-040, F-041, F-042 (quick wins, 3S effort, low risk)
