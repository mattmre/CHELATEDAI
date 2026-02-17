# Architecture Decision: F-020..F-024 Remediation

**Date:** 2026-02-17  
**Cycle:** AEP-2026-02-13  
**Scope:** Medium-priority reliability/security fixes (Tier 4, batch 1)  
**Status:** Implementation-ready

---

## Overview

5 localized fixes addressing input validation gaps, prompt injection, callback safety, and numerical edge cases. Total effort: 5S (1 session). All findings unblocked.

---

## F-020: Qdrant URL/location validation gap

**Problem:** `qdrant_location` accepts arbitrary strings, crashes on `None` input.  
**Risk:** Unclear runtime errors, potential SSRF if misconfigured.

### Decision

Add validation helper at module level in `antigravity_engine.py`:

```python
from urllib.parse import urlparse

def _validate_qdrant_location(location):
    """Validate Qdrant location (URL or path)."""
    if location is None:
        raise ValueError("qdrant_location cannot be None")
    
    if isinstance(location, str) and location.startswith("http"):
        parsed = urlparse(location)
        if not parsed.hostname:
            raise ValueError(f"Invalid Qdrant URL: {location}")
    
    return location
```

**Touchpoint:** Lines 75-79 in `antigravity_engine.py` (`__init__`)  
**Apply:** Call `_validate_qdrant_location(qdrant_location)` before client init.

### Tests Required

File: `test_antigravity_engine.py` (new file or append to existing)

```python
def test_qdrant_location_none_rejected()
def test_qdrant_location_invalid_url_rejected()
def test_qdrant_location_valid_url_accepted()
def test_qdrant_location_local_path_accepted()
```

**Regression risk:** Low (additive validation)  
**Source:** SEC-008, REL-020

---

## F-021: Prompt injection in OllamaDecomposer

**Problem:** User query interpolated directly into LLM prompt without sanitization.  
**Risk:** Malicious queries can manipulate decomposition output structure.

### Decision

Add input sanitization and output validation in `recursive_decomposer.py`:

**1. Query sanitization (module-level helper):**
```python
import re

def _sanitize_query(query: str, max_length: int = 500) -> str:
    """Sanitize query for safe LLM prompt interpolation."""
    query = query[:max_length]
    
    forbidden_patterns = [
        r'ignore\s+previous\s+instructions',
        r'system:',
        r'assistant:',
        r'<\|.*?\|>',
    ]
    for pattern in forbidden_patterns:
        query = re.sub(pattern, '', query, flags=re.IGNORECASE)
    
    return query.strip()
```

**2. Output validation (instance method):**
```python
def _validate_subqueries(self, original_query: str, subqueries: List[str], 
                         max_subqueries: int = 6) -> List[str]:
    """Validate LLM output for safety."""
    subqueries = subqueries[:max_subqueries]
    
    original_words = set(original_query.lower().split())
    filtered = []
    for sq in subqueries:
        sq_words = set(sq.lower().split())
        if original_words & sq_words or len(sq_words) <= 2:
            filtered.append(sq)
    
    return filtered if filtered else [original_query]
```

**Touchpoints:**
- Line 160-166: Apply `_sanitize_query()` before prompt interpolation
- After `_parse_response()`: Apply `_validate_subqueries()` to parsed results

### Tests Required

File: `test_recursive_decomposer.py`

```python
def test_ollama_decomposer_sanitizes_injection_attempt()
def test_ollama_decomposer_limits_subquery_count()
def test_ollama_decomposer_validates_relevance()
```

**Regression risk:** Medium (changes LLM output filtering)  
**Mitigation:** Make thresholds configurable, preserve existing test fixture compatibility  
**Source:** SEC-006

---

## F-022: Callback functions without sandboxing

**Problem:** `remediate_fn` and `verify_fn` callbacks execute without timeout or exception wrapping.  
**Risk:** Buggy callbacks can hang orchestrator or crash with unhandled exceptions.

### Decision

**Windows-compatible approach:** Use signal-based timeout on Unix, graceful fallback on Windows.

Add safe callback wrapper in `aep_orchestrator.py`:

```python
import signal
from contextlib import contextmanager

@contextmanager
def _timeout(seconds: int):
    """Context manager for function timeout (Unix only)."""
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Callback exceeded {seconds}s timeout")
    
    if hasattr(signal, 'SIGALRM'):  # Unix
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:  # Windows - no timeout enforcement
        yield

def _safe_callback(self, callback: Callable, *args, 
                   timeout_sec: int = 30, 
                   expected_return_type: Optional[type] = None,
                   **kwargs) -> Any:
    """Execute callback with timeout and exception wrapping."""
    try:
        with self._timeout(timeout_sec):
            result = callback(*args, **kwargs)
        
        if expected_return_type and not isinstance(result, expected_return_type):
            self.logger.log_event(
                "callback_type_error",
                callback=callback.__name__,
                expected=expected_return_type.__name__,
                actual=type(result).__name__
            )
            return None
        
        return result
        
    except TimeoutError as e:
        self.logger.log_event("callback_timeout", 
                            callback=callback.__name__, 
                            error=str(e))
        return None
    except Exception as e:
        self.logger.log_event("callback_error",
                            callback=callback.__name__,
                            error_type=type(e).__name__,
                            error=str(e))
        return None
```

**Touchpoints:**
- Lines 617-631: `_remediate_finding()` - wrap `self.remediate_fn()` call
- Lines 677-681: `_verify_finding()` - wrap `self.verify_fn()` call

**Add to `__init__`:** `self.callback_timeout_sec = 30` (configurable)

### Tests Required

File: `test_aep_orchestrator.py`

```python
def test_remediate_fn_timeout_handled()
def test_remediate_fn_exception_handled()
def test_remediate_fn_wrong_return_type_handled()
def test_verify_fn_exception_defaults_to_false()
```

**Platform note:** Windows tests should verify graceful degradation (no timeout enforcement, but error handling still works).

**Regression risk:** Medium (changes callback execution semantics)  
**Mitigation:** Timeouts configurable per test; backward compatible for working callbacks  
**Source:** SEC-012

---

## F-023: Division by zero in target vector normalization

**Problem:** Zero-norm vectors produce NaN after normalization, corrupting training.  
**Risk:** Training diverges when all chelation targets are identical or empty.

### Decision

Add safe normalization helper (shared by both modules):

**Module-level function (both `antigravity_engine.py` and `recursive_decomposer.py`):**

```python
def _safe_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize vector, handling zero-norm case."""
    norm = np.linalg.norm(vec)
    if norm < eps:
        logger = get_logger()
        logger.log_event("zero_norm_target", 
                       vector_shape=vec.shape,
                       norm=float(norm))
        return np.zeros_like(vec)
    return vec / norm
```

**Touchpoints:**
- `antigravity_engine.py:400`: Replace `target_vec /= np.linalg.norm(target_vec)`
- `recursive_decomposer.py:509`: Replace `target = target / np.linalg.norm(target)`

### Tests Required

Files: `test_antigravity_engine.py`, `test_recursive_decomposer.py`

```python
# test_antigravity_engine.py
def test_sedimentation_handles_zero_norm_target()

# test_recursive_decomposer.py
def test_hierarchical_sedimentation_zero_variance_cluster()
```

**Regression risk:** Low (defensive edge case handling)  
**Source:** REL-015

---

## F-024: ChelationAdapter crashes on 1D input

**Problem:** Single-vector input (shape `[dim]`) crashes in `forward()` with dimension error.  
**Risk:** Direct adapter usage or batch-size-1 inference fails.

### Decision

Add dimension handling in `chelation_adapter.py` `forward()` method:

```python
def forward(self, x):
    """
    Forward pass with dimension handling.
    
    Args:
        x: Input tensor of shape [batch, dim] or [dim]
    
    Returns:
        Normalized output of shape matching input
    """
    # Handle 1D input
    input_was_1d = False
    if x.dim() == 1:
        input_was_1d = True
        x = x.unsqueeze(0)  # [dim] -> [1, dim]
    
    # Main forward pass (existing logic)
    x = normalize(x, dim=1)
    residual = x
    
    x = self.ln1(self.fc1(x))
    x = self.ln2(self.fc2(x))
    x = x + residual
    
    output = normalize(x, dim=1)
    
    # Restore original shape
    if input_was_1d:
        output = output.squeeze(0)  # [1, dim] -> [dim]
    
    return output
```

**Touchpoint:** Lines 38-47 in `chelation_adapter.py`

### Tests Required

File: `test_unit_core.py` (append to ChelationAdapter tests)

```python
def test_chelation_adapter_forward_1d_input()
def test_chelation_adapter_forward_2d_input()
def test_chelation_adapter_forward_batch_size_1()
def test_chelation_adapter_forward_invalid_dim()
```

**Regression risk:** Low (preserves 2D behavior)  
**API guarantee:** Output shape matches input shape  
**Source:** REL-007

---

## Implementation Order

Execute sequentially (no hard dependencies, but logical flow):

1. **F-020** (input validation) → Standalone
2. **F-021** (prompt injection) → Follows validation theme
3. **F-022** (callback safety) → Standalone
4. **F-023** (zero-norm) → Numerical stability
5. **F-024** (1D tensor) → Numerical stability

---

## PR Strategy

**Recommendation:** One PR per finding (5 PRs total)

### Branch naming convention:
```
pr/f020-qdrant-url-validation       → feature/aep-cycle-remediation-20260216
pr/f021-prompt-injection-guards     → pr/f020-qdrant-url-validation
pr/f022-callback-sandboxing         → pr/f021-prompt-injection-guards
pr/f023-zero-norm-guards            → pr/f022-callback-sandboxing
pr/f024-adapter-1d-input            → pr/f023-zero-norm-guards
```

Each PR merges to previous (sequential chain), final PR merges to feature branch.

**Rationale:**
- Easy independent review
- Granular rollback capability
- Clear commit history
- Independent merge approval

**Alternative (not recommended):** Combine F-023 + F-024 into single "numerical-stability" PR if time-constrained.

---

## Test Execution

### Baseline (before changes):
```bash
pytest --tb=short -v
# Expected: 383 passing tests
```

### Per-PR validation:
```bash
# Full suite after each PR
pytest --tb=short -v

# Specific test files
pytest test_antigravity_engine.py -v          # F-020, F-023
pytest test_recursive_decomposer.py -v        # F-021, F-023
pytest test_aep_orchestrator.py -v            # F-022
pytest test_unit_core.py::TestChelationAdapter -v  # F-024
```

### Final integration check:
```bash
pytest --tb=short -v
# Expected: ~399 passing tests (16 new)

pytest test_integration_rlm.py -v  # End-to-end validation
```

---

## Open Decisions

### F-022 Windows Timeout Strategy
**Options:**
- A) `threading.Timer` fallback (less precise)
- B) Skip timeout on Windows, log warning
- C) `multiprocessing` with timeout (heavier)

**Decision:** Option B (skip timeout on Windows, log warning)  
**Rationale:** Research prototype; simplicity over cross-platform parity. Document limitation.

### F-021 Validation Strictness
**Decision:** Start with basic pattern blocking + relevance heuristic.  
**Rationale:** Defense-in-depth approach; iterate if LLM manipulation observed in practice.

### F-023 Zero-Norm Behavior
**Decision:** Return zero vector (no-op in training) + log event.  
**Rationale:** Non-fatal behavior; observable for debugging; avoids crashing on edge case.

---

## Files Modified Summary

| File | Findings | New Lines | Tests Added |
|------|----------|-----------|-------------|
| `antigravity_engine.py` | F-020, F-023 | ~25 | 5 |
| `recursive_decomposer.py` | F-021, F-023 | ~40 | 4 |
| `aep_orchestrator.py` | F-022 | ~50 | 4 |
| `chelation_adapter.py` | F-024 | ~15 | 4 |
| **Total** | 5 findings | ~130 | **17 tests** |

---

## Rollback Strategy

Each PR independently revertible:
- **F-020:** Remove validation helper, revert to direct location check
- **F-021:** Remove sanitization/validation, revert to direct prompt
- **F-022:** Remove callback wrapper, revert to direct invocation
- **F-023:** Revert to inline normalization
- **F-024:** Revert to 2D-only forward pass

No cross-PR dependencies.

---

## Success Criteria

- [ ] All 5 findings marked RESOLVED in backlog
- [ ] 17 new tests added (all passing)
- [ ] Full test suite passes (no regressions)
- [ ] All 5 PRs merged to feature branch
- [ ] Session log updated
- [ ] No new TODOs/FIXMEs introduced

---

## References

- **Research artifact:** `research-2026-02-17-f020-f024-implementation.md`
- **Backlog:** `backlog-2026-02-13.md` (lines 254-291)
- **Original findings:** SEC-006, SEC-008, SEC-012, REL-007, REL-015, REL-020

---

**Prepared by:** Documentation Agent  
**Ready for:** Session 6 implementation
