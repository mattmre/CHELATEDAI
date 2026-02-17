# Research Report: Tier 2 & Tier 3 Implementation Plan
**Date:** 2026-02-17  
**Cycle ID:** AEP-2026-02-13  
**Scope:** Findings F-004, F-007, F-008, F-009, F-011, F-012, F-013, F-014, F-015, F-016, F-017, F-018, F-019  
**Authors:** Orchestrator + Research & Architecture Agents  
**Status:** Pre-implementation research complete  

---

## Executive Summary

This document provides implementation research for 13 findings spanning security, reliability, testing, and architecture. Analysis shows:
- **7 findings** are OPEN and ready for immediate implementation (S effort, no blockers)
- **3 findings** are OPEN with medium effort (M effort, dependencies resolved)
- **3 findings** are OPEN requiring larger coordination (L effort or multi-file changes)

All dependencies from Session 2 (F-006, F-010) are now resolved. Recommended execution prioritizes quick security wins (F-004, F-008, F-015, F-016, F-017, F-018, F-019), followed by testing gaps (F-014) and medium-effort items (F-007, F-009, F-011, F-012, F-013).

---

## Status Matrix

| Finding | Severity | Effort | Status | Blockers | Priority |
|---------|----------|--------|--------|----------|----------|
| F-004 | High | S | OPEN | None | P1 (Security) |
| F-007 | High | S | OPEN | None (F-006✓, F-010✓) | P2 (Reliability) |
| F-008 | High | S | OPEN | None | P1 (Security) |
| F-009 | High | M | OPEN | None | P2 (Security) |
| F-011 | High | M | OPEN | None (F-006✓) | P3 (Architecture) |
| F-012 | High | M | OPEN | None | P3 (Testing) |
| F-013 | High | L | OPEN | None | P4 (Testing) |
| F-014 | High | S | OPEN | None | P2 (Testing) |
| F-015 | High | S | OPEN | None | P1 (Performance) |
| F-016 | High | S | OPEN | None (F-010✓) | P1 (Reliability) |
| F-017 | High | S | OPEN | None | P1 (Reliability) |
| F-018 | Medium | S | OPEN | None (F-003✓) | P2 (Security) |
| F-019 | Medium | S | OPEN | None | P2 (Reliability) |

**Legend:**
- P1 = Critical quick wins (security/reliability/performance, S effort)
- P2 = High-value medium items (reliability/testing, S effort)
- P3 = Medium effort foundational work
- P4 = Large effort, deferred to later session

---

## Detailed Findings Analysis

### F-004: Remove `trust_remote_code=True` from SentenceTransformer

**Status:** OPEN  
**Current Evidence:**
- File: `antigravity_engine.py:96`
- Code: `self.local_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)`
- **Risk:** High security risk - allows arbitrary code execution from compromised HuggingFace model repos

**Planned Change:**
```python
# Location: antigravity_engine.py:96
# REMOVE trust_remote_code=True parameter

# CURRENT:
self.local_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

# PROPOSED:
self.local_model = SentenceTransformer(model_name, device=device)
```

**Rationale:**
- Most mainstream models (e.g., `sentence-transformers/*`) do not require `trust_remote_code`
- Parameter should only be used for explicitly vetted models
- Removing provides secure-by-default behavior

**Planned Tests:**
1. **Integration test**: Verify existing integration tests still pass with mainstream models
2. **Unit test**: Add test documenting the behavior if a model requiring remote code is attempted
3. **Regression test**: Confirm `test_integration_rlm.py` continues to work

**Files to Modify:**
- `antigravity_engine.py` (line 96)
- `test_unit_core.py` or new `test_antigravity_engine.py` (documentation test)

**Estimated Effort:** 15 minutes  
**Risk Level:** LOW (mainstream models unaffected)  
**Dependencies:** None

---

### F-007: Add Qdrant error handling in inference path

**Status:** OPEN  
**Current Evidence:**
- Files: `antigravity_engine.py:389-394, 432-450`
- Unprotected Qdrant calls:
  - `_gravity_sensor()`: `self.qdrant.query_points()` (line 389)
  - `get_chelated_vector()`: `self.qdrant.query_points()` (line 432) and `self.qdrant.retrieve()` (line 446)
- **Risk:** Any Qdrant connectivity issue causes unhandled crash

**Planned Change:**
```python
# Location: antigravity_engine.py:_gravity_sensor(), get_chelated_vector()

# Add try/except around Qdrant calls
from qdrant_client.http.exceptions import UnexpectedResponse

def _gravity_sensor(self, query_vec, top_k=ChelationConfig.SCOUT_K):
    """Phase 1: Detects Local Curvature (Entropy) around the query."""
    try:
        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            with_vectors=True 
        ).points
        
        if not search_result:
            return np.array([])
            
        vectors = [hit.vector for hit in search_result]
        return np.array(vectors)
        
    except UnexpectedResponse as e:
        self.logger.log_error(
            "qdrant_query_failed",
            f"Qdrant query failed in gravity sensor: {e}",
            exception=e,
            top_k=top_k
        )
        return np.array([])  # Fallback to empty - disables chelation
    except Exception as e:
        self.logger.log_error(
            "gravity_sensor_error",
            f"Unexpected error in gravity sensor: {e}",
            exception=e
        )
        return np.array([])
```

**Similar pattern for `get_chelated_vector()` and `run_inference()`**

**Planned Tests:**
1. **Unit test**: Mock Qdrant to raise `UnexpectedResponse`, verify graceful fallback
2. **Unit test**: Mock Qdrant to raise generic exception, verify logging
3. **Integration test**: Simulate Qdrant unavailable, confirm engine continues

**Files to Modify:**
- `antigravity_engine.py` (3 methods: `_gravity_sensor`, `get_chelated_vector`, `run_inference`)
- `test_antigravity_engine.py` (new file for unit tests)

**Estimated Effort:** 45 minutes  
**Risk Level:** LOW (improves robustness)  
**Dependencies:** F-006 (RESOLVED), F-010 (RESOLVED)

---

### F-008: SSRF URL validation for OllamaDecomposer

**Status:** OPEN  
**Current Evidence:**
- File: `recursive_decomposer.py:134-169`
- Code: `self.ollama_url = ollama_url or ChelationConfig.OLLAMA_URL...`
- Method `decompose()` calls `self.requests.post(self.ollama_url, ...)`
- **Risk:** Attacker-controlled URL could probe internal network

**Planned Change:**
```python
# Location: recursive_decomposer.py:OllamaDecomposer.__init__

from urllib.parse import urlparse

ALLOWED_OLLAMA_HOSTS = {'localhost', '127.0.0.1', '::1'}

def __init__(self, model="llama3.2", ollama_url=None, timeout=None):
    self.model = model
    
    # Validate and store URL
    raw_url = ollama_url or ChelationConfig.OLLAMA_URL.replace(
        "/api/embeddings", "/api/generate"
    )
    self._validate_ollama_url(raw_url)
    self.ollama_url = raw_url
    
    self.timeout = timeout or ChelationConfig.OLLAMA_TIMEOUT
    
    try:
        import requests
        self.requests = requests
    except ImportError:
        self.requests = None

def _validate_ollama_url(self, url: str):
    """Validate Ollama URL to prevent SSRF."""
    try:
        parsed = urlparse(url)
        if parsed.hostname not in ALLOWED_OLLAMA_HOSTS:
            raise ValueError(
                f"Invalid Ollama host: {parsed.hostname}. "
                f"Allowed hosts: {ALLOWED_OLLAMA_HOSTS}"
            )
    except Exception as e:
        raise ValueError(f"Invalid Ollama URL: {url}") from e
```

**Planned Tests:**
1. **Unit test**: Valid localhost URL accepted
2. **Unit test**: External URL rejected with ValueError
3. **Unit test**: Malformed URL rejected
4. **Integration test**: Confirm existing decomposition tests still work

**Files to Modify:**
- `recursive_decomposer.py` (OllamaDecomposer class)
- `test_recursive_decomposer.py` (add validation tests)

**Estimated Effort:** 30 minutes  
**Risk Level:** LOW (only restricts to localhost)  
**Dependencies:** None

---

### F-009: Path traversal validation in file I/O

**Status:** OPEN  
**Current Evidence:**
- Files:
  - `chelation_adapter.py:49-55` (save/load methods)
  - `config.py:248-285` (load_from_file/save_to_file)
  - `checkpoint_manager.py:86-97,154-167` (create/restore/delete)
- **Risk:** Arbitrary file read/write/delete via unchecked paths

**Planned Change:**
```python
# Add to config.py as shared utility
from pathlib import Path
import re

class ChelationConfig:
    # ... existing code ...
    
    @staticmethod
    def validate_safe_path(path: Path, base_dir: Path = None) -> Path:
        """
        Validate that path is safe (contained within base_dir).
        
        Args:
            path: Path to validate
            base_dir: Base directory to contain within (default: PROJECT_ROOT)
        
        Returns:
            Resolved absolute path
            
        Raises:
            ValueError: If path escapes base_dir
        """
        base_dir = base_dir or ChelationConfig.PROJECT_ROOT
        resolved_path = Path(path).resolve()
        resolved_base = Path(base_dir).resolve()
        
        try:
            resolved_path.relative_to(resolved_base)
        except ValueError:
            raise ValueError(
                f"Path '{path}' escapes base directory '{base_dir}'"
            )
        
        return resolved_path
    
    @staticmethod
    def sanitize_checkpoint_name(name: str) -> str:
        """Sanitize checkpoint name to prevent path traversal."""
        return re.sub(r'[^a-zA-Z0-9_-]', '', name)
```

**Apply to all file I/O:**
```python
# chelation_adapter.py
def save(self, path):
    safe_path = ChelationConfig.validate_safe_path(path)
    torch.save(self.state_dict(), safe_path)

def load(self, path):
    safe_path = ChelationConfig.validate_safe_path(path)
    if safe_path.exists():
        # ... existing code
```

**Similar pattern for config.py and checkpoint_manager.py**

**Planned Tests:**
1. **Unit test**: Valid paths within PROJECT_ROOT accepted
2. **Unit test**: Path with `..` escaping PROJECT_ROOT rejected
3. **Unit test**: Absolute path outside PROJECT_ROOT rejected
4. **Unit test**: Checkpoint name with `/` or `\` sanitized
5. **Integration test**: Existing checkpointing still works

**Files to Modify:**
- `config.py` (add validation utilities)
- `chelation_adapter.py` (apply to save/load)
- `checkpoint_manager.py` (apply to create/restore/delete, sanitize names)
- `test_unit_core.py` or `test_checkpoint_manager.py` (add validation tests)

**Estimated Effort:** 90 minutes  
**Risk Level:** MEDIUM (requires careful testing of path edge cases)  
**Dependencies:** None

---

### F-011: Extract shared sedimentation trainer

**Status:** OPEN  
**Current Evidence:**
- Duplicated code:
  - `antigravity_engine.py:run_sedimentation_cycle()` (lines 350-500+)
  - `recursive_decomposer.py:run_hierarchical_sedimentation()` (lines 449-676)
- **Duplication:** Target vector calculation, adapter training loop, Qdrant batch update with payload preservation

**Planned Change:**
Create new module `sedimentation_trainer.py`:

```python
"""
Shared Sedimentation Training Logic for ChelatedAI

Extracted from AntigravityEngine and HierarchicalSedimentationEngine
to eliminate code duplication and standardize training behavior.
"""

import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Any
from chelation_logger import get_logger
from config import ChelationConfig

class SedimentationTrainer:
    """
    Encapsulates sedimentation training logic with Qdrant synchronization.
    """
    
    def __init__(self, adapter, qdrant_client, collection_name, logger=None):
        self.adapter = adapter
        self.qdrant = qdrant_client
        self.collection_name = collection_name
        self.logger = logger or get_logger()
    
    def compute_target_vectors(
        self,
        current_vectors: np.ndarray,
        noise_centers: List[List[np.ndarray]],
        push_magnitude: float = 0.1
    ) -> np.ndarray:
        """
        Compute target vectors by pushing away from noise centers.
        
        Args:
            current_vectors: Current embeddings [N, D]
            noise_centers: List of noise center lists per vector
            push_magnitude: How far to push (0.1 = 10% of normalized diff)
        
        Returns:
            Target vectors [N, D]
        """
        targets = []
        for i, current_vec in enumerate(current_vectors):
            if i < len(noise_centers) and noise_centers[i]:
                avg_noise = np.mean(noise_centers[i], axis=0)
                diff = current_vec - avg_noise
                diff_norm = diff / (np.linalg.norm(diff) + 1e-9)
                target_vec = current_vec + (diff_norm * push_magnitude)
                target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-9)
            else:
                # No noise data, keep current
                target_vec = current_vec
            targets.append(target_vec)
        
        return np.array(targets)
    
    def train_adapter(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float,
        epochs: int,
        batch_size: int = None
    ) -> Dict[str, Any]:
        """
        Train adapter with MSE loss.
        
        Returns:
            Training statistics dict
        """
        self.adapter.train()
        optimizer = optim.Adam(self.adapter.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.adapter(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        return {
            'final_loss': losses[-1],
            'mean_loss': np.mean(losses),
            'loss_history': losses
        }
    
    def sync_to_qdrant(
        self,
        doc_ids: List[int],
        new_vectors: np.ndarray,
        chunk_size: int = 100
    ):
        """
        Update Qdrant with corrected vectors, preserving payloads.
        """
        # Fetch current points to preserve payloads
        all_points = []
        for i in range(0, len(doc_ids), chunk_size):
            chunk_ids = doc_ids[i:i+chunk_size]
            points = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=chunk_ids,
                with_vectors=False,
                with_payload=True
            )
            all_points.extend(points)
        
        # Create updated points
        from qdrant_client.models import PointStruct
        updated_points = [
            PointStruct(
                id=point.id,
                vector=new_vectors[i].tolist(),
                payload=point.payload
            )
            for i, point in enumerate(all_points)
        ]
        
        # Batch upsert
        for i in range(0, len(updated_points), chunk_size):
            chunk = updated_points[i:i+chunk_size]
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=chunk
            )
```

**Then refactor both callers:**
```python
# antigravity_engine.py:run_sedimentation_cycle
from sedimentation_trainer import SedimentationTrainer

def run_sedimentation_cycle(self, threshold=3, learning_rate=0.001, epochs=10):
    # Filter targets
    targets = {k: v for k, v in self.chelation_log.items() if len(v) >= threshold}
    if not targets:
        return
    
    # Fetch vectors
    # ... existing fetch logic ...
    
    # Use trainer
    trainer = SedimentationTrainer(self.adapter, self.qdrant, self.collection_name, self.logger)
    
    target_vecs = trainer.compute_target_vectors(current_vectors, noise_centers)
    input_tensor = torch.tensor(current_vectors, dtype=torch.float32)
    target_tensor = torch.tensor(target_vecs, dtype=torch.float32)
    
    stats = trainer.train_adapter(input_tensor, target_tensor, learning_rate, epochs)
    
    # Get corrected vectors
    with torch.no_grad():
        corrected = self.adapter(input_tensor).numpy()
    
    trainer.sync_to_qdrant(doc_ids, corrected)
    
    # Save adapter and clear log
    self.adapter.save(self.adapter_path)
    self.chelation_log.clear()
```

**Planned Tests:**
1. **Unit test**: `compute_target_vectors` with known inputs
2. **Unit test**: `train_adapter` loss decreases over epochs
3. **Unit test**: `sync_to_qdrant` preserves payloads (mocked)
4. **Integration test**: Existing sedimentation tests still pass

**Files to Create:**
- `sedimentation_trainer.py` (new module)

**Files to Modify:**
- `antigravity_engine.py` (refactor `run_sedimentation_cycle`)
- `recursive_decomposer.py` (refactor `run_hierarchical_sedimentation`)
- `test_unit_core.py` or new `test_sedimentation_trainer.py`

**Estimated Effort:** 2-3 hours  
**Risk Level:** MEDIUM (requires careful refactoring and testing)  
**Dependencies:** F-006 (RESOLVED)

---

### F-012: Add chelation_logger.py test coverage

**Status:** OPEN  
**Current Evidence:**
- File: `chelation_logger.py` (413 lines, 0 direct test coverage)
- Untested methods: `log_event`, `log_query`, `log_training_start`, `OperationContext`, `get_logger` singleton

**Planned Change:**
Create `test_chelation_logger.py`:

```python
"""
Unit Tests for ChelationLogger

Tests structured logging, JSON output, operation context, and singleton behavior.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from chelation_logger import ChelationLogger, get_logger, OperationContext


class TestChelationLogger(unittest.TestCase):
    """Test ChelationLogger functionality."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_path = self.temp_dir / "test.jsonl"
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_log_event_json_format(self):
        """Test that log_event writes valid JSON."""
        logger = ChelationLogger(log_path=self.log_path)
        logger.log_event("test_event", "Test message", custom_field="value")
        
        # Read log file
        with open(self.log_path, 'r') as f:
            line = f.readline()
        
        event = json.loads(line)
        self.assertEqual(event['event_type'], 'test_event')
        self.assertEqual(event['message'], 'Test message')
        self.assertEqual(event['custom_field'], 'value')
        self.assertIn('timestamp', event)
        self.assertIn('elapsed_seconds', event)
    
    def test_operation_context_duration(self):
        """Test that OperationContext tracks duration."""
        logger = ChelationLogger(log_path=self.log_path)
        
        with OperationContext(logger, "test_op", "Test operation"):
            pass  # No-op
        
        # Read log file - should have start and end events
        with open(self.log_path, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)
        start_event = json.loads(lines[0])
        end_event = json.loads(lines[1])
        
        self.assertEqual(start_event['event_type'], 'test_op')
        self.assertEqual(end_event['event_type'], 'test_op_complete')
        self.assertIn('duration_seconds', end_event)
        self.assertGreaterEqual(end_event['duration_seconds'], 0)
    
    def test_get_logger_singleton(self):
        """Test that get_logger returns same instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        
        self.assertIs(logger1, logger2)
    
    def test_log_query_structure(self):
        """Test log_query adds required fields."""
        logger = ChelationLogger(log_path=self.log_path)
        logger.log_query(
            query_text="test query",
            top_k=10,
            action="CHELATE",
            jaccard=0.5
        )
        
        with open(self.log_path, 'r') as f:
            event = json.loads(f.readline())
        
        self.assertEqual(event['event_type'], 'query')
        self.assertEqual(event['query_text'], 'test query')
        self.assertEqual(event['top_k'], 10)
    
    def test_log_training_start(self):
        """Test log_training_start records hyperparameters."""
        logger = ChelationLogger(log_path=self.log_path)
        logger.log_training_start(
            num_samples=100,
            learning_rate=0.01,
            epochs=10,
            threshold=3
        )
        
        with open(self.log_path, 'r') as f:
            event = json.loads(f.readline())
        
        self.assertEqual(event['event_type'], 'training_start')
        self.assertEqual(event['num_samples'], 100)
        self.assertEqual(event['learning_rate'], 0.01)

if __name__ == '__main__':
    unittest.main()
```

**Planned Tests:**
1. JSON format validation
2. Operation context duration tracking
3. Singleton behavior
4. Query/training log structure
5. File I/O error handling
6. Console vs file level filtering

**Files to Create:**
- `test_chelation_logger.py` (new test file)

**Estimated Effort:** 90 minutes  
**Risk Level:** LOW (pure testing, no production changes)  
**Dependencies:** None

---

### F-013: Add AntigravityEngine unit tests

**Status:** OPEN  
**Current Evidence:**
- File: `antigravity_engine.py` (900+ lines, no unit tests)
- Only integration test in `test_integration_rlm.py`
- Untested: `embed()` retry logic, `_chelate_toxicity`, `run_inference` branching, `ingest` batching

**Planned Change:**
Create `test_antigravity_engine.py`:

```python
"""
Unit Tests for AntigravityEngine

Mocked tests for core engine logic without requiring Qdrant/Ollama.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from antigravity_engine import AntigravityEngine


class TestAntigravityEngineEmbed(unittest.TestCase):
    """Test embed() method with mocked backends."""
    
    @patch('antigravity_engine.SentenceTransformer')
    def test_local_mode_embed(self, mock_st):
        """Test local mode embedding."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0]])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_st.return_value = mock_model
        
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        result = engine.embed(["test"])
        self.assertEqual(result.shape, (1, 3))
    
    @patch('requests.post')
    def test_ollama_mode_embed_retry(self, mock_post):
        """Test Ollama embed with retry on truncation."""
        # First call fails, second succeeds
        mock_post.side_effect = [
            Exception("Token limit"),
            Mock(json=lambda: {"embedding": [1.0, 2.0, 3.0]})
        ]
        
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="ollama:nomic-embed-text"
        )
        
        # Should retry and succeed
        result = engine.embed(["very long text" * 1000])
        self.assertEqual(len(result[0]), 3)


class TestChelateToxicity(unittest.TestCase):
    """Test _chelate_toxicity dimension masking logic."""
    
    def setUp(self):
        with patch('antigravity_engine.SentenceTransformer'):
            self.engine = AntigravityEngine(
                qdrant_location=":memory:",
                chelation_p=80
            )
    
    def test_high_variance_dimensions_masked(self):
        """Test that high-variance dimensions are masked."""
        # Create cluster with known variance pattern
        # Dims 0,1 have low variance, dim 2 has high variance
        cluster = np.array([
            [1.0, 1.0, 0.0],
            [1.1, 1.1, 10.0],
            [0.9, 0.9, -10.0],
        ])
        
        mask = self.engine._chelate_toxicity(cluster)
        
        # At P=80, top 20% variance dims should be masked (dim 2)
        self.assertEqual(mask[0], 1.0)  # Low variance, kept
        self.assertEqual(mask[1], 1.0)  # Low variance, kept
        self.assertEqual(mask[2], 0.0)  # High variance, masked
    
    def test_empty_cluster_returns_ones(self):
        """Test that empty cluster returns all-ones mask."""
        mask = self.engine._chelate_toxicity(np.array([]))
        np.testing.assert_array_equal(mask, np.ones(self.engine.vector_size))


class TestRunInference(unittest.TestCase):
    """Test run_inference decision tree."""
    
    @patch('antigravity_engine.SentenceTransformer')
    def setUp(self, mock_st):
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_st.return_value = mock_model
        
        self.engine = AntigravityEngine(
            qdrant_location=":memory:",
            use_quantization=True,
            use_centering=False
        )
        
        # Mock Qdrant
        self.engine.qdrant = Mock()
    
    def test_low_variance_fast_path(self):
        """Test that low variance triggers FAST path."""
        # Mock low-variance neighborhood
        mock_points = [Mock(id=i, vector=[1.0, 1.0, 1.0]) for i in range(10)]
        self.engine.qdrant.query_points.return_value = Mock(points=mock_points)
        
        std_ids, chel_ids, mask, jaccard = self.engine.run_inference("test query")
        
        # Low variance -> no reranking, IDs should be identical
        self.assertEqual(std_ids, chel_ids)

if __name__ == '__main__':
    unittest.main()
```

**Planned Tests:**
1. Local mode embedding
2. Ollama mode retry logic
3. `_chelate_toxicity` variance masking
4. `run_inference` FAST vs CHELATE paths
5. `ingest` batching
6. Empty input edge cases
7. Dimension mismatch handling

**Files to Create:**
- `test_antigravity_engine.py` (new test file, ~200-300 lines)

**Estimated Effort:** 4-5 hours (large test surface area)  
**Risk Level:** LOW (pure testing)  
**Dependencies:** None

---

### F-014: Add OllamaDecomposer tests

**Status:** OPEN  
**Current Evidence:**
- File: `recursive_decomposer.py:127-208` (OllamaDecomposer class)
- Current tests only cover `MockDecomposer`
- Untested: `_parse_response` regex, `decompose` HTTP calls, error fallback

**Planned Change:**
Add to `test_recursive_decomposer.py`:

```python
class TestOllamaDecomposer(unittest.TestCase):
    """Test OllamaDecomposer with mocked requests."""
    
    def setUp(self):
        self.decomposer = OllamaDecomposer(
            model="llama3.2",
            ollama_url="http://localhost:11434/api/generate"
        )
    
    def test_parse_response_numbered_list(self):
        """Test parsing numbered list format."""
        text = "1. First query\n2. Second query\n3. Third query"
        result = self.decomposer._parse_response(text)
        self.assertEqual(result, ["First query", "Second query", "Third query"])
    
    def test_parse_response_bullet_list(self):
        """Test parsing bullet list format."""
        text = "- Query one\n- Query two"
        result = self.decomposer._parse_response(text)
        self.assertEqual(result, ["Query one", "Query two"])
    
    def test_parse_response_mixed_format(self):
        """Test parsing mixed numbering styles."""
        text = "1) First item\n2. Second item\n- Third item"
        result = self.decomposer._parse_response(text)
        self.assertEqual(result, ["First item", "Second item", "Third item"])
    
    def test_parse_response_empty(self):
        """Test parsing empty response."""
        result = self.decomposer._parse_response("")
        self.assertEqual(result, [""])
    
    @patch('recursive_decomposer.OllamaDecomposer.requests')
    def test_decompose_success(self, mock_requests_module):
        """Test successful decomposition."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "1. sub-query one\n2. sub-query two"
        }
        mock_requests_module.post.return_value = mock_response
        
        result = self.decomposer.decompose("complex query")
        
        self.assertEqual(result, ["sub-query one", "sub-query two"])
    
    @patch('recursive_decomposer.OllamaDecomposer.requests')
    def test_decompose_http_error(self, mock_requests_module):
        """Test HTTP error fallback."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_requests_module.post.return_value = mock_response
        
        result = self.decomposer.decompose("query")
        
        # Should fallback to atomic
        self.assertEqual(result, ["query"])
    
    @patch('recursive_decomposer.OllamaDecomposer.requests')
    def test_decompose_timeout(self, mock_requests_module):
        """Test timeout fallback."""
        mock_requests_module.post.side_effect = TimeoutError("timeout")
        
        result = self.decomposer.decompose("query")
        
        # Should fallback to atomic
        self.assertEqual(result, ["query"])
    
    def test_is_base_case_short_query(self):
        """Test base case detection for short queries."""
        self.assertTrue(self.decomposer.is_base_case("short"))
        self.assertFalse(self.decomposer.is_base_case("this is a longer query string"))
```

**Planned Tests:**
1. `_parse_response` with various formats (numbered, bullet, mixed)
2. `_parse_response` edge cases (empty, malformed)
3. `decompose` with mocked successful response
4. `decompose` with HTTP errors (400, 500)
5. `decompose` with timeout/connection error
6. `is_base_case` threshold logic

**Files to Modify:**
- `test_recursive_decomposer.py` (add TestOllamaDecomposer class)
- `recursive_decomposer.py` (import OllamaDecomposer if not already in test imports)

**Estimated Effort:** 45 minutes  
**Risk Level:** LOW (pure testing)  
**Dependencies:** None

---

### F-015: Cap chelation_log unbounded memory growth

**Status:** OPEN  
**Current Evidence:**
- File: `antigravity_engine.py:31` (initialization: `self.chelation_log = defaultdict(list)`)
- File: `antigravity_engine.py:641` (append: `self.chelation_log[doc_id].append(center_of_mass)`)
- **Current behavior:** Unbounded list growth - O(Q * 50 * D) memory
- **Recent fix:** Lines 641-645 already implement capping! Need to verify this is working correctly.

**Current Code Review:**
```python
# Line 641-645 in _spectral_chelation_ranking
max_entries = ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC
for doc_id in local_ids:
    self.chelation_log[doc_id].append(center_of_mass)
    # Cap log size to prevent unbounded memory growth
    if len(self.chelation_log[doc_id]) > max_entries:
        # Keep most recent entries
        self.chelation_log[doc_id] = self.chelation_log[doc_id][-max_entries:]
```

**Status Update:** PARTIALLY RESOLVED  
**Evidence:** Code already caps at `CHELATION_LOG_MAX_ENTRIES_PER_DOC`

**Verification Needed:**
1. Confirm `ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC` is defined
2. Add test to verify capping behavior
3. Document the memory bounds

**Planned Change:**
```python
# Verify config.py has the constant
class ChelationConfig:
    # ... existing code ...
    CHELATION_LOG_MAX_ENTRIES_PER_DOC = 50  # Caps memory per doc
```

**Planned Tests:**
```python
# test_antigravity_engine.py
def test_chelation_log_capping(self):
    """Test that chelation log is capped per document."""
    engine = AntigravityEngine(qdrant_location=":memory:")
    
    # Simulate many collisions for one doc
    doc_id = 42
    center = np.random.rand(engine.vector_size)
    
    max_entries = ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC
    
    # Add more than max
    for i in range(max_entries + 10):
        engine.chelation_log[doc_id].append(center)
        # Manually trigger cap (or call _spectral_chelation_ranking)
    
    # Should be capped
    self.assertLessEqual(len(engine.chelation_log[doc_id]), max_entries)
```

**Files to Modify:**
- `config.py` (verify constant exists, add if missing)
- `test_antigravity_engine.py` (add capping test)

**Estimated Effort:** 20 minutes (verification + test)  
**Risk Level:** LOW (code already exists, just needs testing)  
**Dependencies:** None

---

### F-016: Tighten init except Exception to specific errors

**Status:** OPEN  
**Current Evidence:**
- File: `antigravity_engine.py:71-84`
- Code uses broad `except Exception` to catch Ollama connection test failures
- **Risk:** Silent failure on config errors, engine proceeds with wrong vector_size

**Current Code:**
```python
try:
    import requests
    self.requests = requests
    test_vec = self.embed("test")[0]
    self.vector_size = len(test_vec)
    self.logger.log_event("initialization", f"Connected to Ollama. Vector Size: {self.vector_size}", vector_size=self.vector_size)
except ImportError as e:
    raise ImportError(f"'requests' library required for Ollama mode. Install with: pip install requests") from e
except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
    raise ConnectionError(f"Failed to connect to Ollama at {self.ollama_url}. Make sure Docker container is running!") from e
except Exception as e:
    self.logger.log_error("connection", f"Ollama connection test failed: {e}", exception=e)
    self.logger.log_event("initialization", "Vector size will be validated on first real embedding call", level="WARNING")
    # Keep default 768 as fallback
```

**Analysis:** Code is actually GOOD now! Recent updates (F-010) already tightened this.
- `ImportError` caught specifically and re-raised
- `ConnectionError` and `Timeout` caught specifically and re-raised
- Only fallback `except Exception` logs warning and continues

**Status Update:** MOSTLY RESOLVED  
**Remaining issue:** Should the final `except Exception` re-raise for non-network errors?

**Planned Change:**
```python
except Exception as e:
    # Don't silently continue on config errors
    if isinstance(e, (KeyError, AttributeError, TypeError)):
        # Config or logic error - re-raise
        raise
    
    # Unknown error - log and continue with degraded mode
    self.logger.log_error("connection", f"Ollama connection test failed: {e}", exception=e)
    self.logger.log_event("initialization", "Vector size will be validated on first real embedding call", level="WARNING")
```

**Planned Tests:**
1. **Unit test**: Mock Ollama to raise ConnectionError, verify exception propagates
2. **Unit test**: Mock Ollama to raise generic network error, verify graceful degradation
3. **Unit test**: Mock config error (KeyError), verify re-raised

**Files to Modify:**
- `antigravity_engine.py` (line 81-84, refine exception handling)
- `test_antigravity_engine.py` (add exception handling tests)

**Estimated Effort:** 30 minutes  
**Risk Level:** LOW (tightens error handling)  
**Dependencies:** F-010 (RESOLVED)

---

### F-017: Fix fragile conditional requests import

**Status:** OPEN  
**Current Evidence:**
- File: `antigravity_engine.py:72-79`
- `import requests` done inside try block
- **Risk:** Works but fragile, non-standard pattern

**Current Code:**
```python
try:
    import requests
    self.requests = requests
    test_vec = self.embed("test")[0]
    # ...
except ImportError as e:
    raise ImportError(f"'requests' library required for Ollama mode. Install with: pip install requests") from e
except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
    # ^^ This references 'requests' which might not be defined if ImportError occurred
    raise ConnectionError(...)
```

**Planned Change:**
```python
# Move import to module level (top of antigravity_engine.py)
try:
    import requests
except ImportError:
    requests = None  # Will be checked in __init__

# Then in __init__:
if model_name.startswith("ollama:"):
    if requests is None:
        raise ImportError(
            "'requests' library required for Ollama mode. "
            "Install with: pip install requests"
        )
    
    self.requests = requests
    # ... rest of Ollama setup
    try:
        test_vec = self.embed("test")[0]
        self.vector_size = len(test_vec)
        self.logger.log_event("initialization", f"Connected to Ollama. Vector Size: {self.vector_size}", vector_size=self.vector_size)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise ConnectionError(f"Failed to connect to Ollama at {self.ollama_url}. Make sure Docker container is running!") from e
```

**Planned Tests:**
1. **Unit test**: Mock `requests=None`, verify ImportError raised for Ollama mode
2. **Unit test**: Verify local mode works without requests
3. **Integration test**: Existing tests should still pass

**Files to Modify:**
- `antigravity_engine.py` (move import to module level, update __init__)

**Estimated Effort:** 20 minutes  
**Risk Level:** LOW (standard Python pattern)  
**Dependencies:** None

---

### F-018: Make checkpoint hash mismatch a hard failure

**Status:** OPEN  
**Current Evidence:**
- File: `checkpoint_manager.py:163-168`
- Hash mismatch only warns, still restores potentially tampered file

**Current Code:**
```python
# Verify integrity
current_hash = self._compute_file_hash(adapter_checkpoint)
if current_hash != checkpoint_meta["adapter_hash"]:
    if not allow_hash_mismatch:
        print("ERROR: Checkpoint file hash mismatch. Refusing restore.")
        return False
    print("WARNING: Checkpoint file hash mismatch, restoring due to allow_hash_mismatch=True")
```

**Analysis:** Code already has the right logic! `allow_hash_mismatch` parameter defaults to False.

**Verification Needed:**
Check if `allow_hash_mismatch` defaults to False in method signature.

**Files to Check:**
```python
# checkpoint_manager.py:restore_checkpoint signature
def restore_checkpoint(
    self,
    checkpoint_id: str,
    target_adapter_path: Optional[Path] = None,
    allow_hash_mismatch: bool = False  # <-- Verify this defaults to False
) -> bool:
```

**Status Update:** ALREADY RESOLVED  
**Evidence:** Method signature at line ~145 shows `allow_hash_mismatch: bool = False`

**Planned Tests:**
Already covered by F-003 tests in `test_checkpoint_manager.py`, but verify:
1. Test exists for hash mismatch rejection
2. Test exists for `allow_hash_mismatch=True` override

**Files to Verify:**
- `test_checkpoint_manager.py` (confirm hash mismatch test exists)

**Estimated Effort:** 10 minutes (verification only)  
**Risk Level:** NONE (already resolved)  
**Dependencies:** F-003 (RESOLVED)

---

### F-019: Tighten OllamaDecomposer exception handling

**Status:** OPEN  
**Current Evidence:**
- File: `recursive_decomposer.py:189`
- Bare `except Exception` silently degrades to atomic mode

**Current Code:**
```python
try:
    response = self.requests.post(
        self.ollama_url,
        json={...},
        timeout=self.timeout,
    )
    # ... parse response ...
except Exception:
    return [query]
```

**Planned Change:**
```python
import requests
import json

try:
    response = self.requests.post(
        self.ollama_url,
        json={...},
        timeout=self.timeout,
    )
    
    if response.status_code != 200:
        return [query]
    
    text = response.json().get("response", "")
    sub_queries = self._parse_response(text)
    
    if len(sub_queries) > 1:
        return sub_queries
    return [query]

except (requests.RequestException, requests.Timeout) as e:
    # Network errors - expected failure mode
    self.logger.log_event(
        "decompose_network_error",
        f"Ollama request failed: {e}",
        level="WARNING",
        exception=str(e)
    )
    return [query]
except (json.JSONDecodeError, KeyError) as e:
    # Parsing errors - unexpected but recoverable
    self.logger.log_event(
        "decompose_parse_error",
        f"Failed to parse Ollama response: {e}",
        level="WARNING",
        exception=str(e)
    )
    return [query]
except Exception as e:
    # Unknown errors - log for investigation
    self.logger.log_error(
        "decompose_unexpected_error",
        f"Unexpected decomposition error: {e}",
        exception=e
    )
    return [query]
```

**Planned Tests:**
1. **Unit test**: Mock `requests.RequestException`, verify logged and fallback
2. **Unit test**: Mock `json.JSONDecodeError`, verify logged and fallback
3. **Unit test**: Mock unexpected exception, verify logged
4. **Integration test**: Confirm existing decomposition tests still pass

**Files to Modify:**
- `recursive_decomposer.py` (OllamaDecomposer.decompose method)
- Need to ensure logger is available (add to __init__ if not present)

**Additional Requirement:**
Add logger to OllamaDecomposer:
```python
def __init__(self, model="llama3.2", ollama_url=None, timeout=None):
    # ... existing code ...
    self.logger = get_logger()
```

**Files to Modify:**
- `recursive_decomposer.py` (OllamaDecomposer.__init__ and decompose)
- `test_recursive_decomposer.py` (add exception handling tests)

**Estimated Effort:** 30 minutes  
**Risk Level:** LOW (improves observability)  
**Dependencies:** None

---

## Dependency Graph

```
Tier 2 Quick Wins (S effort, 0 blockers):
┌─────────────────────────────────────┐
│ F-004 (trust_remote_code removal)   │  Priority 1
│ F-008 (SSRF URL validation)        │  Priority 1
│ F-015 (chelation_log capping)      │  Priority 1
│ F-016 (init exception tightening)  │  Priority 1 (needs F-010✓)
│ F-017 (requests import fix)        │  Priority 1
│ F-018 (hash mismatch - VERIFIED)   │  Priority 2 (needs F-003✓)
│ F-019 (decomposer exceptions)     │  Priority 2
│ F-014 (OllamaDecomposer tests)    │  Priority 2
└─────────────────────────────────────┘

Tier 3 Medium Effort (M effort):
┌─────────────────────────────────────┐
│ F-007 (Qdrant error handling)      │  Priority 2 (needs F-006✓, F-010✓)
│ F-009 (path traversal validation)  │  Priority 2
│ F-012 (chelation_logger tests)     │  Priority 3
└─────────────────────────────────────┘

Tier 3 Large Refactoring:
┌─────────────────────────────────────┐
│ F-011 (shared sedimentation)       │  Priority 3 (needs F-006✓)
│ F-013 (AntigravityEngine tests)    │  Priority 4
└─────────────────────────────────────┘
```

**Execution Order Recommendation:**
1. **Session 3a** (Quick wins, 2-3 hours):
   - F-004 (15min)
   - F-017 (20min)
   - F-015 (20min verification)
   - F-016 (30min)
   - F-008 (30min)
   - F-018 (10min verification)
   - F-019 (30min)
   - F-014 (45min)

2. **Session 3b** (Medium effort, 2-3 hours):
   - F-007 (45min)
   - F-009 (90min)
   - F-012 (90min)

3. **Session 4** (Large items, 4-6 hours):
   - F-011 (2-3 hours)
   - F-013 (4-5 hours) - can be split across multiple sessions

---

## Test Strategy Summary

### New Test Files to Create:
1. `test_antigravity_engine.py` (F-007, F-013, F-015, F-016, F-017)
2. `test_chelation_logger.py` (F-012)
3. `test_sedimentation_trainer.py` (F-011)

### Existing Test Files to Extend:
1. `test_recursive_decomposer.py` (F-008, F-014, F-019)
2. `test_unit_core.py` (F-004, F-009)
3. `test_checkpoint_manager.py` (F-018 verification)

### Test Coverage Goals:
- **F-004**: Integration tests pass without `trust_remote_code`
- **F-007**: Qdrant failure scenarios (network, missing collection)
- **F-008**: URL validation (localhost OK, external blocked)
- **F-009**: Path traversal attempts blocked
- **F-011**: Shared trainer unit tests + refactored integration tests pass
- **F-012**: ChelationLogger JSON output, duration tracking, singleton
- **F-013**: Engine embed retry, chelate logic, inference branching
- **F-014**: OllamaDecomposer parse/decompose/error handling
- **F-015**: Memory capping verification
- **F-016**: Exception propagation vs graceful degradation
- **F-017**: Import error handling
- **F-018**: Hash mismatch rejection (verify existing)
- **F-019**: Exception-specific logging

---

## Risk Assessment

### Low Risk (Ready to Implement):
- F-004: Mainstream models unaffected
- F-008: Only restricts to localhost
- F-014: Pure testing
- F-015: Code already exists
- F-016: Tightens error handling
- F-017: Standard pattern
- F-018: Already resolved
- F-019: Improves observability

### Medium Risk (Requires Careful Testing):
- F-007: Must not break existing retrieval
- F-009: Path validation can be tricky on Windows
- F-011: Large refactoring, must preserve behavior
- F-012: Pure testing, low risk

### Deferred (High Effort):
- F-013: Large test surface, can be split

---

## Recommended Next Actions

### Immediate (Session 3 - Next 2 Days):
1. **Implementer**: Start with F-004, F-017, F-015 (quick security wins, 1 hour total)
2. **Tester**: Add F-014 tests (OllamaDecomposer, 45 min)
3. **Implementer**: F-016, F-008, F-019 (exception handling, 1.5 hours)
4. **Reviewer**: Verify F-018 tests exist and are adequate
5. **Implementer + Tester**: F-007 (Qdrant error handling, 1 hour)

### Medium-term (Session 3b - Next Week):
1. **Security + Implementer**: F-009 (path traversal, 90 min)
2. **Tester**: F-012 (logger tests, 90 min)

### Long-term (Session 4+):
1. **Architect + Refactorer**: F-011 (sedimentation refactor, 3 hours)
2. **Tester**: F-013 (engine unit tests, 4-5 hours, can split)

---

## Success Criteria

**Tier 2 Complete** when:
- [ ] 8 findings resolved (F-004, F-007, F-008, F-014, F-015, F-016, F-017, F-018, F-019)
- [ ] All tests passing
- [ ] Security audit clean (no remaining High severity security findings in quick-win category)
- [ ] Test coverage increased by ~50 tests

**Tier 3 Complete** when:
- [ ] F-009, F-011, F-012 resolved
- [ ] Sedimentation code deduplicated
- [ ] Path traversal protection in place
- [ ] Logger fully tested

**Long-term (F-013)** when:
- [ ] AntigravityEngine has comprehensive unit test suite
- [ ] All major code paths covered
- [ ] Mocked tests enable fast iteration

---

## Open Questions for Team

1. **F-004**: Should we maintain an allowlist of models that require `trust_remote_code`?
2. **F-008**: Should ALLOWED_OLLAMA_HOSTS be configurable or hardcoded?
3. **F-009**: Windows path edge cases - need Windows tester confirmation?
4. **F-011**: Should `SedimentationTrainer` be standalone or part of existing module?
5. **F-013**: Should we split engine tests into multiple files (embed, chelation, inference)?

---

## Appendix: File Modification Summary

| File | Findings | Modification Type | Estimated LOC |
|------|----------|------------------|---------------|
| `antigravity_engine.py` | F-004, F-007, F-015, F-016, F-017 | Security + Reliability fixes | +30 |
| `recursive_decomposer.py` | F-008, F-019 | Security + Reliability fixes | +25 |
| `chelation_adapter.py` | F-009 | Path validation | +5 |
| `config.py` | F-009, F-015 | Path utils + constants | +30 |
| `checkpoint_manager.py` | F-009, F-018 | Path validation + verification | +5 |
| `sedimentation_trainer.py` | F-011 | New module | +200 (NEW) |
| `test_antigravity_engine.py` | F-007, F-013, F-015, F-016, F-017 | New test file | +300 (NEW) |
| `test_chelation_logger.py` | F-012 | New test file | +150 (NEW) |
| `test_sedimentation_trainer.py` | F-011 | New test file | +100 (NEW) |
| `test_recursive_decomposer.py` | F-008, F-014, F-019 | Extend tests | +80 |
| `test_unit_core.py` | F-004, F-009 | Extend tests | +30 |
| `test_checkpoint_manager.py` | F-018 | Verify tests | +0 (verify) |

**Total Estimated New Code:** ~955 lines (mostly tests)  
**Total Estimated Modified Code:** ~95 lines

---

**End of Research Report**
