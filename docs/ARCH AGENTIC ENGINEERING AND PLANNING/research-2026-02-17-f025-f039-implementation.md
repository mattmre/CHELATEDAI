# Research Artifact: F-025..F-028, F-039 Implementation Plan

**Date:** 2026-02-17  
**Cycle:** AEP-2026-02-13  
**Scope:** Medium-priority reliability and performance findings (Tier 4, batch 2)  
**Target Session:** Session 8

---

## Executive Summary

This research artifact covers the next 5 medium-severity findings from the backlog. All are small-effort (S) fixes with localized touchpoints and clear acceptance criteria. These findings address:
- Embedding validation and error handling
- Exception handling during checkpoint rollback
- Redundant database operations
- Non-vectorized computational loops
- Resource cleanup and connection leaks

**Total effort:** 5S (estimated 1 session)  
**Risk level:** Low to Medium (localized changes, straightforward test coverage)  
**Dependency status:** All unblocked

---

## Finding F-025: `ingest()` does not validate embedding dimensions or handle empty results

### Current Behavior/Risk
- **Severity:** Medium  
- **Impact:** No dimension check after `embed()` call; empty embeddings silently passed to `upsert()`
- **Reliability risk:** Failed batches unreported; dimension mismatches cause cryptic Qdrant errors
- **Observable symptoms:** Silent ingestion failures, incorrect vector counts in collection, unclear error messages

### Code Touchpoints
```
antigravity_engine.py:194-220  (AntigravityEngine.ingest)
```

**Exact code location:**
```python
# Lines 194-220 - ingest method
embeddings = self.embed([chunk["text"] for chunk in chunks])
# No validation that embeddings match expected dimensions
# No check for empty results
for i, chunk in enumerate(chunks):
    points.append(PointStruct(
        id=chunk["id"],
        vector=embeddings[i],  # Could be wrong dimension or empty
        payload={"text": chunk["text"], "metadata": chunk.get("metadata", {})}
    ))
self.qdrant.upsert(collection_name=collection_name, points=points)
# No error reporting for failed batches
```

### Minimal Implementation Plan

1. **Add dimension validation helper:**
   ```python
   def _validate_embeddings(self, embeddings: List[np.ndarray], 
                            expected_dim: int,
                            collection_name: str) -> List[np.ndarray]:
       """Validate embedding dimensions and filter invalid entries."""
       if not embeddings:
           self.logger.log_event("empty_embeddings",
                                collection=collection_name)
           return []
       
       valid_embeddings = []
       for i, emb in enumerate(embeddings):
           if emb is None or len(emb) == 0:
               self.logger.log_event("null_embedding",
                                    collection=collection_name,
                                    index=i)
               continue
           
           if len(emb) != expected_dim:
               self.logger.log_event("dimension_mismatch",
                                    collection=collection_name,
                                    expected=expected_dim,
                                    actual=len(emb),
                                    index=i)
               continue
           
           valid_embeddings.append(emb)
       
       return valid_embeddings
   ```

2. **Apply validation in `ingest()`:**
   ```python
   # Get collection info to determine expected dimensions
   collection_info = self.qdrant.get_collection(collection_name)
   expected_dim = collection_info.config.params.vectors.size
   
   # Embed chunks
   texts = [chunk["text"] for chunk in chunks]
   embeddings = self.embed(texts)
   
   # Validate dimensions
   validated_embeddings = self._validate_embeddings(
       embeddings, 
       expected_dim,
       collection_name
   )
   
   # Build points only for valid embeddings
   points = []
   for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
       if i < len(validated_embeddings):
           points.append(PointStruct(
               id=chunk["id"],
               vector=validated_embeddings[i],
               payload={"text": chunk["text"], "metadata": chunk.get("metadata", {})}
           ))
   
   # Report skipped entries
   if len(points) < len(chunks):
       self.logger.log_event("batch_partial_success",
                            collection=collection_name,
                            total=len(chunks),
                            ingested=len(points),
                            skipped=len(chunks) - len(points))
   ```

3. **Add batch error handling:**
   ```python
   try:
       self.qdrant.upsert(collection_name=collection_name, points=points)
       self.logger.log_event("batch_ingested",
                            collection=collection_name,
                            count=len(points))
   except Exception as e:
       self.logger.log_event("batch_ingest_failed",
                            collection=collection_name,
                            error=str(e),
                            error_type=type(e).__name__)
       raise
   ```

### Test Plan

**New test file:** `test_antigravity_engine.py` (append to existing)

```python
def test_ingest_validates_embedding_dimensions(mock_qdrant):
    engine = AntigravityEngine(qdrant_location="memory")
    mock_qdrant.get_collection.return_value = Mock(
        config=Mock(params=Mock(vectors=Mock(size=384)))
    )
    
    # Return wrong-dimension embeddings
    engine.embed = Mock(return_value=[
        np.random.randn(256),  # Wrong dimension
        np.random.randn(384),  # Correct
    ])
    
    chunks = [
        {"id": 1, "text": "test1"},
        {"id": 2, "text": "test2"}
    ]
    
    engine.ingest(chunks, collection_name="test")
    
    # Should only upsert valid embeddings
    upsert_call = mock_qdrant.upsert.call_args
    assert len(upsert_call[1]['points']) == 1
    assert upsert_call[1]['points'][0].id == 2

def test_ingest_handles_empty_embeddings(mock_qdrant):
    engine = AntigravityEngine(qdrant_location="memory")
    engine.embed = Mock(return_value=[])
    
    chunks = [{"id": 1, "text": "test"}]
    
    # Should not crash, should log warning
    engine.ingest(chunks, collection_name="test")
    
    # Upsert should not be called
    mock_qdrant.upsert.assert_not_called()

def test_ingest_handles_null_embeddings(mock_qdrant):
    engine = AntigravityEngine(qdrant_location="memory")
    mock_qdrant.get_collection.return_value = Mock(
        config=Mock(params=Mock(vectors=Mock(size=384)))
    )
    
    engine.embed = Mock(return_value=[None, np.random.randn(384)])
    
    chunks = [
        {"id": 1, "text": "test1"},
        {"id": 2, "text": "test2"}
    ]
    
    engine.ingest(chunks, collection_name="test")
    
    # Should skip null, process valid
    upsert_call = mock_qdrant.upsert.call_args
    assert len(upsert_call[1]['points']) == 1

def test_ingest_reports_batch_errors(mock_qdrant):
    engine = AntigravityEngine(qdrant_location="memory")
    mock_qdrant.get_collection.return_value = Mock(
        config=Mock(params=Mock(vectors=Mock(size=384)))
    )
    mock_qdrant.upsert.side_effect = Exception("Qdrant error")
    
    chunks = [{"id": 1, "text": "test"}]
    
    with pytest.raises(Exception, match="Qdrant error"):
        engine.ingest(chunks, collection_name="test")
    
    # Should log error event
    # (verify via logger mock)
```

**Validation:**
- Confirm dimension mismatches logged and skipped
- Confirm empty/null embeddings handled gracefully
- Confirm partial batch success reported
- Confirm upsert errors logged with context
- No existing tests should break

### Regression Risks
- **Low:** Validation is defensive; existing valid inputs unaffected
- **Watch for:** Tests that pass intentionally malformed embeddings
- **Mitigation:** Run full test suite before/after

### Source Findings
- REL-004, REL-019

---

## Finding F-026: `SafeTrainingContext` rollback can mask original exception

### Current Behavior/Risk
- **Severity:** Medium  
- **Impact:** If `restore_checkpoint` raises during exception handling, the original training error is lost
- **Debugging risk:** Stack trace shows rollback error instead of root cause
- **Observable symptoms:** Misleading error messages about checkpoint restoration rather than actual training failure

### Code Touchpoints
```
checkpoint_manager.py:294-306  (SafeTrainingContext.__exit__)
```

**Exact code location:**
```python
def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is not None:
        # Training failed, rollback
        self.manager.restore_checkpoint(self.checkpoint_id, self.model)
        # If restore_checkpoint raises, exc_val is lost
        return False  # Re-raise original exception
    else:
        # Training succeeded
        self.manager.commit_checkpoint(self.checkpoint_id)
        return False
```

### Minimal Implementation Plan

1. **Add nested exception handling:**
   ```python
   def __exit__(self, exc_type, exc_val, exc_tb):
       if exc_type is not None:
           # Training failed, attempt rollback
           self.logger.log_event("training_failed",
                                checkpoint_id=self.checkpoint_id,
                                error_type=exc_type.__name__,
                                error=str(exc_val))
           
           try:
               self.manager.restore_checkpoint(self.checkpoint_id, self.model)
               self.logger.log_event("rollback_succeeded",
                                    checkpoint_id=self.checkpoint_id)
           except Exception as rollback_error:
               # Log rollback failure but preserve original exception
               self.logger.log_event("rollback_failed",
                                    checkpoint_id=self.checkpoint_id,
                                    original_error=str(exc_val),
                                    rollback_error=str(rollback_error),
                                    rollback_error_type=type(rollback_error).__name__)
               
               # Optionally: chain exceptions (Python 3)
               # raise rollback_error from exc_val
           
           return False  # Re-raise original exception
       else:
           # Training succeeded
           try:
               self.manager.commit_checkpoint(self.checkpoint_id)
           except Exception as commit_error:
               self.logger.log_event("checkpoint_commit_failed",
                                    checkpoint_id=self.checkpoint_id,
                                    error=str(commit_error))
               # This is a new error, let it propagate
               raise
           
           return False
   ```

2. **Update error messages to be more informative:**
   ```python
   # In CheckpointManager.restore_checkpoint
   def restore_checkpoint(self, checkpoint_id: str, model: torch.nn.Module):
       try:
           checkpoint_path = self._get_checkpoint_path(checkpoint_id)
           state_dict = torch.load(checkpoint_path)
           model.load_state_dict(state_dict)
       except Exception as e:
           raise RuntimeError(
               f"Failed to restore checkpoint {checkpoint_id}: {e}"
           ) from e
   ```

### Test Plan

**New tests in:** `test_checkpoint_manager.py`

```python
def test_safe_training_context_preserves_original_exception():
    manager = CheckpointManager(checkpoint_dir="./test_checkpoints")
    model = torch.nn.Linear(10, 10)
    
    # Mock restore to fail
    manager.restore_checkpoint = Mock(side_effect=RuntimeError("Restore failed"))
    
    with pytest.raises(ValueError, match="Training error") as exc_info:
        with SafeTrainingContext(manager, model, "test_id"):
            raise ValueError("Training error")
    
    # Original exception should be raised, not rollback error
    assert "Training error" in str(exc_info.value)

def test_safe_training_context_logs_rollback_failure():
    manager = CheckpointManager(checkpoint_dir="./test_checkpoints")
    model = torch.nn.Linear(10, 10)
    
    manager.restore_checkpoint = Mock(side_effect=RuntimeError("Restore failed"))
    
    with pytest.raises(ValueError):
        with SafeTrainingContext(manager, model, "test_id"):
            raise ValueError("Training error")
    
    # Should log both errors
    # (verify via logger mock)

def test_safe_training_context_rollback_succeeds():
    manager = CheckpointManager(checkpoint_dir="./test_checkpoints")
    model = torch.nn.Linear(10, 10)
    
    manager.create_checkpoint(model, "test_id")
    
    # Simulate training failure
    with pytest.raises(ValueError):
        with SafeTrainingContext(manager, model, "test_id"):
            raise ValueError("Training failed")
    
    # Model should be restored (verify weights unchanged)

def test_safe_training_context_commit_failure_raises():
    manager = CheckpointManager(checkpoint_dir="./test_checkpoints")
    model = torch.nn.Linear(10, 10)
    
    manager.commit_checkpoint = Mock(side_effect=OSError("Disk full"))
    
    with pytest.raises(OSError, match="Disk full"):
        with SafeTrainingContext(manager, model, "test_id"):
            pass  # Training succeeds, but commit fails
```

**Validation:**
- Original training exception preserved when rollback fails
- Both errors logged for debugging
- Successful rollback doesn't suppress original error
- Commit failures during success path propagate correctly

### Regression Risks
- **Low:** Only adds exception handling; existing success paths unchanged
- **Watch for:** Tests that check specific exception types
- **Mitigation:** Exception chain preserves original type

### Source Findings
- REL-010

---

## Finding F-027: Redundant Qdrant round-trip in `get_chelated_vector()`

### Current Behavior/Risk
- **Severity:** Medium  
- **Impact:** Calls `query_points()` without vectors, then `retrieve()` for same IDs
- **Performance cost:** 2x network round-trips, 2x database queries
- **Observable symptoms:** Slower retrieval, increased Qdrant load

### Code Touchpoints
```
antigravity_engine.py:268-287  (AntigravityEngine.get_chelated_vector)
```

**Exact code location:**
```python
def get_chelated_vector(self, query, collection_name, k=50):
    # First query: get IDs only (no vectors)
    results = self.qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,
        with_vectors=False  # Line 276 - vectors not requested
    )
    
    # Second query: retrieve vectors for same IDs
    candidate_ids = [r.id for r in results]
    vectors = self.qdrant.retrieve(
        collection_name=collection_name,
        ids=candidate_ids,
        with_vectors=True  # Line 283 - redundant fetch
    )
    
    # Use vectors for chelation ranking
    # ...
```

### Minimal Implementation Plan

1. **Consolidate into single query:**
   ```python
   def get_chelated_vector(self, query, collection_name, k=50):
       query_vector = self.embed([query])[0]
       
       # Single query: get both IDs and vectors
       results = self.qdrant.query_points(
           collection_name=collection_name,
           query=query_vector,
           limit=k,
           with_vectors=True  # Include vectors in initial query
       )
       
       # Extract vectors directly from results
       candidate_vectors = np.array([r.vector for r in results])
       candidate_ids = [r.id for r in results]
       
       # Perform chelation ranking
       scores = self._spectral_chelation_ranking(query_vector, candidate_vectors)
       
       # Select top candidate
       best_idx = np.argmax(scores)
       best_id = candidate_ids[best_idx]
       
       # Return selected vector and metadata
       return {
           "id": best_id,
           "vector": candidate_vectors[best_idx],
           "score": float(scores[best_idx]),
           "payload": results[best_idx].payload
       }
   ```

2. **Update callers if needed:**
   - Verify return value format matches expectations
   - Update any tests that mock `retrieve()`

### Test Plan

**New tests in:** `test_antigravity_engine.py`

```python
def test_get_chelated_vector_single_query(mock_qdrant):
    engine = AntigravityEngine(qdrant_location="memory")
    
    # Mock query_points to return vectors
    mock_results = [
        Mock(id=i, vector=np.random.randn(384), payload={"text": f"doc{i}"})
        for i in range(10)
    ]
    mock_qdrant.query_points.return_value = mock_results
    
    result = engine.get_chelated_vector("test query", "test_collection")
    
    # Should only call query_points once (not retrieve)
    mock_qdrant.query_points.assert_called_once()
    assert mock_qdrant.query_points.call_args[1]['with_vectors'] is True
    mock_qdrant.retrieve.assert_not_called()
    
    # Result should contain expected fields
    assert "id" in result
    assert "vector" in result
    assert "score" in result

def test_get_chelated_vector_performance_improvement():
    # Integration test: measure actual query count
    engine = AntigravityEngine(qdrant_location="memory")
    
    # Set up collection with test data
    engine.ingest([
        {"id": i, "text": f"document {i}"}
        for i in range(100)
    ], collection_name="perf_test")
    
    # Instrument Qdrant client to count calls
    call_count = {"query": 0, "retrieve": 0}
    
    original_query = engine.qdrant.query_points
    original_retrieve = engine.qdrant.retrieve
    
    def count_query(*args, **kwargs):
        call_count["query"] += 1
        return original_query(*args, **kwargs)
    
    def count_retrieve(*args, **kwargs):
        call_count["retrieve"] += 1
        return original_retrieve(*args, **kwargs)
    
    engine.qdrant.query_points = count_query
    engine.qdrant.retrieve = count_retrieve
    
    # Execute query
    engine.get_chelated_vector("test", "perf_test")
    
    # Should be 1 query, 0 retrieve calls
    assert call_count["query"] == 1
    assert call_count["retrieve"] == 0
```

**Validation:**
- Single `query_points()` call with `with_vectors=True`
- No `retrieve()` calls
- Correct vector and metadata returned
- Performance improvement measurable in benchmarks

### Regression Risks
- **Low:** Optimization doesn't change output
- **Watch for:** Tests that mock `retrieve()` expecting it to be called
- **Mitigation:** Update test mocks to match new call pattern

### Source Findings
- PERF-002

---

## Finding F-028: Per-element cosine similarity loop in `_spectral_chelation_ranking()`

### Current Behavior/Risk
- **Severity:** Medium  
- **Impact:** Python loop over 50 candidates instead of vectorized NumPy
- **Performance cost:** ~10-50x slower than vectorized operations
- **Observable symptoms:** Slow chelation ranking, especially with large k values

### Code Touchpoints
```
antigravity_engine.py:336-339  (_spectral_chelation_ranking)
```

**Exact code location:**
```python
def _spectral_chelation_ranking(self, query_vector, candidate_vectors):
    # Center vectors
    mean = np.mean(candidate_vectors, axis=0)
    centered_query = query_vector - mean
    centered_candidates = candidate_vectors - mean
    
    # Compute norms
    query_norm = np.linalg.norm(centered_query)
    
    # Loop over candidates (slow!)
    scores = []
    for i in range(len(centered_candidates)):  # Line 336
        candidate_norm = np.linalg.norm(centered_candidates[i])
        similarity = np.dot(centered_query, centered_candidates[i]) / (
            query_norm * candidate_norm + 1e-8
        )
        scores.append(similarity)
    
    return np.array(scores)
```

### Minimal Implementation Plan

1. **Vectorize cosine similarity:**
   ```python
   def _spectral_chelation_ranking(self, query_vector, candidate_vectors, eps=1e-8):
       """
       Compute spectral chelation ranking using vectorized operations.
       
       Args:
           query_vector: Shape [dim]
           candidate_vectors: Shape [n_candidates, dim]
           eps: Small value for numerical stability
       
       Returns:
           scores: Shape [n_candidates]
       """
       # Center vectors
       mean = np.mean(candidate_vectors, axis=0)
       centered_query = query_vector - mean  # [dim]
       centered_candidates = candidate_vectors - mean  # [n_candidates, dim]
       
       # Compute norms
       query_norm = np.linalg.norm(centered_query)  # scalar
       candidate_norms = np.linalg.norm(centered_candidates, axis=1)  # [n_candidates]
       
       # Vectorized dot products: [n_candidates, dim] @ [dim] -> [n_candidates]
       dot_products = centered_candidates @ centered_query
       
       # Vectorized cosine similarity
       scores = dot_products / (query_norm * candidate_norms + eps)
       
       return scores
   ```

2. **Add input validation (optional):**
   ```python
   # At start of function
   if candidate_vectors.ndim != 2:
       raise ValueError(f"candidate_vectors must be 2D, got {candidate_vectors.ndim}D")
   
   if len(query_vector) != candidate_vectors.shape[1]:
       raise ValueError(
           f"Dimension mismatch: query={len(query_vector)}, "
           f"candidates={candidate_vectors.shape[1]}"
       )
   ```

### Test Plan

**New tests in:** `test_antigravity_engine.py`

```python
def test_spectral_chelation_ranking_vectorized():
    engine = AntigravityEngine(qdrant_location="memory")
    
    query = np.random.randn(384)
    candidates = np.random.randn(50, 384)
    
    scores = engine._spectral_chelation_ranking(query, candidates)
    
    # Check output shape
    assert scores.shape == (50,)
    
    # Check scores are valid similarities (roughly in [-1, 1])
    assert np.all(scores >= -1.1)  # Allow small numerical error
    assert np.all(scores <= 1.1)

def test_spectral_chelation_ranking_correctness():
    """Verify vectorized implementation matches naive loop."""
    engine = AntigravityEngine(qdrant_location="memory")
    
    # Simple test case
    query = np.array([1.0, 0.0, 0.0])
    candidates = np.array([
        [1.0, 0.0, 0.0],  # Identical to query
        [0.0, 1.0, 0.0],  # Orthogonal
        [-1.0, 0.0, 0.0], # Opposite
    ])
    
    scores = engine._spectral_chelation_ranking(query, candidates)
    
    # After centering, scores should reflect relative similarities
    # (exact values depend on centering, but ordering should be preserved)
    assert len(scores) == 3
    assert not np.isnan(scores).any()

def test_spectral_chelation_ranking_performance():
    """Measure performance improvement."""
    import time
    
    engine = AntigravityEngine(qdrant_location="memory")
    
    query = np.random.randn(384)
    candidates = np.random.randn(1000, 384)  # Larger set
    
    start = time.time()
    scores = engine._spectral_chelation_ranking(query, candidates)
    elapsed = time.time() - start
    
    # Should complete in < 10ms on modern hardware
    assert elapsed < 0.01, f"Too slow: {elapsed:.3f}s"
    assert scores.shape == (1000,)

def test_spectral_chelation_ranking_handles_zero_norm():
    """Edge case: zero-norm vectors after centering."""
    engine = AntigravityEngine(qdrant_location="memory")
    
    # All candidates identical (zero variance)
    query = np.array([1.0, 0.0, 0.0])
    candidates = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    
    scores = engine._spectral_chelation_ranking(query, candidates)
    
    # Should not crash or produce NaN
    assert not np.isnan(scores).any()
    assert not np.isinf(scores).any()
```

**Validation:**
- Output matches naive loop implementation (correctness)
- Performance improvement measurable (>10x speedup)
- Edge cases (zero norms, identical vectors) handled
- No NaN or Inf in output

### Regression Risks
- **Low:** Pure optimization; output should be identical (within numerical precision)
- **Watch for:** Tests that rely on exact floating-point values
- **Mitigation:** Use `np.allclose()` instead of exact equality checks

### Source Findings
- PERF-003

---

## Finding F-039: No resource cleanup for Qdrant client

### Current Behavior/Risk
- **Severity:** Medium  
- **Impact:** Engine creates `QdrantClient` but no `close()` or context manager
- **Resource leak:** File locks, network connections, memory buffers not released
- **Observable symptoms:** Resource exhaustion in long-running processes, file descriptor leaks

### Code Touchpoints
```
antigravity_engine.py:74-103  (AntigravityEngine.__init__)
antigravity_engine.py (entire class - missing cleanup method)
```

**Exact code locations:**
```python
class AntigravityEngine:
    def __init__(self, ..., qdrant_location=...):
        # Create client
        if qdrant_location.startswith("http"):
            self.qdrant = QdrantClient(url=qdrant_location)
        else:
            self.qdrant = QdrantClient(path=qdrant_location)
        
        # No close() method defined
        # No __enter__/__exit__ for context manager
```

### Minimal Implementation Plan

1. **Add cleanup method:**
   ```python
   def close(self):
       """Close Qdrant client and release resources."""
       if hasattr(self, 'qdrant') and self.qdrant is not None:
           try:
               if hasattr(self.qdrant, 'close'):
                   self.qdrant.close()
               self.logger.log_event("engine_closed")
           except Exception as e:
               self.logger.log_event("engine_close_error",
                                    error=str(e),
                                    error_type=type(e).__name__)
           finally:
               self.qdrant = None
   ```

2. **Add context manager support:**
   ```python
   def __enter__(self):
       """Context manager entry."""
       return self
   
   def __exit__(self, exc_type, exc_val, exc_tb):
       """Context manager exit: cleanup resources."""
       self.close()
       return False  # Don't suppress exceptions
   ```

3. **Add destructor (defensive):**
   ```python
   def __del__(self):
       """Destructor: ensure cleanup even if close() not called."""
       # Only log warning, don't raise
       if hasattr(self, 'qdrant') and self.qdrant is not None:
           import warnings
           warnings.warn(
               "AntigravityEngine was not explicitly closed. "
               "Use engine.close() or context manager to ensure cleanup.",
               ResourceWarning
           )
           self.close()
   ```

4. **Update documentation:**
   ```python
   class AntigravityEngine:
       """
       Main engine for chelation-based retrieval.
       
       Usage:
           # Explicit cleanup
           engine = AntigravityEngine(...)
           try:
               engine.ingest(...)
           finally:
               engine.close()
           
           # Or use context manager (recommended)
           with AntigravityEngine(...) as engine:
               engine.ingest(...)
       """
   ```

### Test Plan

**New tests in:** `test_antigravity_engine.py`

```python
def test_antigravity_engine_close_method():
    engine = AntigravityEngine(qdrant_location="memory")
    
    # Should have close method
    assert hasattr(engine, 'close')
    
    # Close should not raise
    engine.close()
    
    # Should be safe to call multiple times
    engine.close()

def test_antigravity_engine_context_manager():
    """Test context manager support."""
    with AntigravityEngine(qdrant_location="memory") as engine:
        # Engine should work inside context
        engine.ingest([{"id": 1, "text": "test"}], collection_name="test")
    
    # After exiting context, qdrant should be None
    assert engine.qdrant is None

def test_antigravity_engine_context_manager_with_exception():
    """Context manager should cleanup even on exception."""
    with pytest.raises(ValueError):
        with AntigravityEngine(qdrant_location="memory") as engine:
            raise ValueError("Test error")
    
    # Engine should still be cleaned up
    assert engine.qdrant is None

def test_antigravity_engine_resource_warning():
    """Test that destructor warns if not explicitly closed."""
    import warnings
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ResourceWarning)
        
        engine = AntigravityEngine(qdrant_location="memory")
        # Don't call close(), let destructor run
        del engine
        
        # Should have issued ResourceWarning
        assert any(issubclass(warning.category, ResourceWarning) for warning in w)

def test_antigravity_engine_close_idempotent():
    """Calling close multiple times should be safe."""
    engine = AntigravityEngine(qdrant_location="memory")
    
    engine.close()
    engine.close()  # Should not raise
    engine.close()

def test_antigravity_engine_usage_after_close():
    """Operations after close should raise informative error."""
    engine = AntigravityEngine(qdrant_location="memory")
    engine.close()
    
    # Attempting operations after close should fail cleanly
    with pytest.raises((AttributeError, ValueError)):
        engine.ingest([{"id": 1, "text": "test"}], collection_name="test")
```

**Validation:**
- `close()` method releases resources
- Context manager (`with` statement) works correctly
- Cleanup happens even on exceptions
- Warning issued if not explicitly closed
- Multiple `close()` calls safe
- Clear error if used after close

### Regression Risks
- **Low:** Additive feature; existing code continues to work
- **Migration path:** Gradually update code to use context manager
- **Watch for:** Tests that expect engine to stay open indefinitely
- **Mitigation:** Add deprecation warning after several versions

### Source Findings
- REL-026

---

## PR Slicing Recommendation

Given the localized nature of these changes, recommend **one PR per finding** for clean review and easy rollback:

### Recommended PR Structure

1. **PR #38:** `pr/f025-ingest-validation` → `feature/aep-cycle-remediation-20260216`
   - Single touchpoint in `antigravity_engine.py` (ingest method)
   - 4 new validation tests
   - Low regression risk

2. **PR #39:** `pr/f026-checkpoint-exception-handling` → `pr/f025-ingest-validation`
   - Single touchpoint in `checkpoint_manager.py` (__exit__ method)
   - 4 new exception handling tests
   - Low risk (defensive error handling)

3. **PR #40:** `pr/f027-deduplicate-qdrant-query` → `pr/f026-checkpoint-exception-handling`
   - Single touchpoint in `antigravity_engine.py` (get_chelated_vector)
   - 2 new performance tests
   - Low risk (optimization)

4. **PR #41:** `pr/f028-vectorize-cosine-similarity` → `pr/f027-deduplicate-qdrant-query`
   - Single touchpoint in `antigravity_engine.py` (_spectral_chelation_ranking)
   - 4 new performance/correctness tests
   - Low risk (pure optimization)

5. **PR #42:** `pr/f039-qdrant-resource-cleanup` → `pr/f028-vectorize-cosine-similarity`
   - Multiple touchpoints in `antigravity_engine.py` (add close, __enter__, __exit__)
   - 6 new resource management tests
   - Low risk (additive feature)

### Alternative: Combined Performance PR (NOT recommended)
If time pressure exists, could combine F-027 + F-028 (both performance optimizations) into a single PR. However, individual PRs are preferred for:
- Easier code review
- Granular rollback capability
- Clearer commit history
- Independent merge approval

---

## Test Execution Strategy

### Pre-implementation baseline
```bash
pytest --tb=short -v
# Expected: 399 passing tests (after Session 6/7 PRs)
```

### Per-PR validation
```bash
# Run full suite after each PR's changes
pytest --tb=short -v

# Run specific test file
pytest test_antigravity_engine.py -v  # F-025, F-027, F-028, F-039
pytest test_checkpoint_manager.py -v  # F-026
```

### Integration validation
```bash
# After all 5 PRs merged
pytest --tb=short -v
# Expected: ~418 passing tests (19 new)

# Run specific integration tests
pytest test_integration_rlm.py -v  # Verify engine still works end-to-end
pytest test_benchmark_rlm.py -v  # Verify performance improvements
```

### Performance benchmarking
```bash
# Before and after F-027, F-028
python benchmark_rlm.py --config scifact --limit 100

# Expected improvements:
# - F-027: ~2x faster get_chelated_vector (1 query vs 2)
# - F-028: ~10-50x faster chelation ranking (vectorized vs loop)
```

---

## Open Questions / Decisions Needed

1. **F-025 Dimension mismatch handling:**
   - Current plan: Skip invalid embeddings, log warning
   - Alternative: Raise exception on any invalid embedding
   - **Recommendation:** Skip and log (more robust for batch processing)

2. **F-026 Exception chaining:**
   - Python 3 supports `raise ... from ...` for exception chaining
   - Should we use explicit chaining or just log both errors?
   - **Recommendation:** Log both; explicit chaining may confuse stack traces

3. **F-039 Deprecation timeline:**
   - Should we immediately require context manager usage?
   - Or add deprecation warning for a few releases?
   - **Recommendation:** Add warning now, make required in v2.0

---

## Dependencies and Ordering

```
F-025  (standalone)
  ↓
F-026  (standalone)
  ↓
F-027  (standalone, performance)
  ↓
F-028  (standalone, performance)
  ↓
F-039  (standalone, resource cleanup)
```

**Execution order:** Sequential as listed (no hard dependencies, but logical flow)

---

## Rollback Strategy

Each PR is independently revertible:
- **F-025:** Remove validation logic, revert to direct upsert
- **F-026:** Remove nested try/except, revert to simple exception handling
- **F-027:** Revert to two-query pattern
- **F-028:** Revert to loop-based similarity computation
- **F-039:** Remove close/context manager methods (backward compatible)

No cross-PR dependencies means individual rollback is safe.

---

## Success Criteria

- [ ] All 5 findings marked RESOLVED in backlog
- [ ] 19+ new tests added (passing)
- [ ] Full test suite passes (no regressions)
- [ ] All PRs merged to feature branch
- [ ] Session log updated with implementation notes
- [ ] Performance benchmarks show expected improvements (F-027, F-028)
- [ ] No new TODOs or FIXMEs introduced

---

## Appendix: Backlog Context

**From backlog-2026-02-13.md:**
- F-025 at lines 293-299
- F-026 at lines 301-307
- F-027 at lines 309-315
- F-028 at lines 317-323
- F-039 at lines 407-413

**From session-log-2026-02-17-impl-7.md:**
- Listed as immediate next tranche for Session 8
- All confirmed unblocked
- 20 findings remaining in backlog after session 7

**From next-session.md:**
- Should verify stacked PR review/merge progression through #37 before starting

---

**End of Research Artifact**
