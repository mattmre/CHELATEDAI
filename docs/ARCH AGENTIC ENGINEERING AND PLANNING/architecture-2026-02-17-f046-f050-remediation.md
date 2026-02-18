# Architecture Plan: F-046..F-050 Remediation

**Date:** 2026-02-17  
**Cycle:** AEP-2026-02-13  
**Scope:** F-046, F-047, F-048, F-049, F-050

---

## Goals

1. Implement all five findings in a behavior-safe way.
2. Keep one coherent PR per finding.
3. Preserve existing public APIs and pass the current test suite.

## Non-Goals

- No broad redesign of training pipelines beyond F-046 scoped extraction.
- No changes to retrieval math outputs.
- No unrelated cleanup.

---

## Dependency Graph

```text
F-046 (scoped engine decomposition)
  └─ independent of F-047..F-050 but touches core engine internals

F-047 (invert_chelation init) ─┐
F-048 (tracker status error)   ├─ independent quick wins
F-049 (falsy payload lookup)   ┤
F-050 (benchmark id handling)  ┘
```

Recommended implementation order:
1. F-046
2. F-047
3. F-048
4. F-049
5. F-050

---

## Detailed Design

### F-046: Scoped decomposition of `AntigravityEngine`

**Problem:** `AntigravityEngine` still owns chelation/gravity/spectral logic directly, making the class harder to reason about and test.

**Design:**
- Add a new module for extracted chelation internals, e.g. `antigravity_components.py`.
- Extract focused component class(es) for:
  - gravity neighborhood extraction orchestration
  - toxicity mask computation
  - spectral reranking computation
- Keep `AntigravityEngine` public methods unchanged:
  - `_gravity_sensor`
  - `_chelate_toxicity`
  - `_spectral_chelation_ranking`
  - `_cosine_similarity_manual`
- Implement these engine methods as thin wrappers/delegators where practical, preserving return types and side effects (`chelation_log`, logger events, and existing test monkeypatch behavior).

**Files:**
- `antigravity_engine.py`
- `antigravity_components.py` (new)
- `test_antigravity_engine.py` (only if delegation-specific assertions are needed)

---

### F-047: Initialize `invert_chelation` explicitly

**Problem:** `_chelate_toxicity` gates behavior via `hasattr(self, 'invert_chelation')`.

**Design:**
- In `AntigravityEngine.__init__`, initialize:
  - `self.invert_chelation = False`
- Update `_chelate_toxicity` to branch directly on `self.invert_chelation` (remove `hasattr` check).

**Files:**
- `antigravity_engine.py`
- `test_antigravity_engine.py` (add explicit default-state coverage)

---

### F-048: Validate `finding_id` in `AEPTracker.update_status`

**Problem:** `AEPTracker.update_status` indexes `self.findings[finding_id]` directly and raises raw `KeyError`.

**Design:**
- Add explicit existence check before lookup.
- Raise:
  - `ValueError(f"Finding '{finding_id}' not found in tracker.")`
- Keep behavior unchanged for valid IDs.

**Files:**
- `aep_orchestrator.py`
- `test_aep_orchestrator.py` (new test for helpful error)

---

### F-049: Preserve nested falsy values in payload lookup

**Problem:** Recursive payload search treats nested falsy values (`0`, `False`, `""`) as not-found due to truthiness check.

**Design:**
- In `benchmark_utils.find_payload`, replace recursive guard:
  - from `if res:`
  - to `if res is not None:`
- Keep `None` as the only not-found sentinel.

**Files:**
- `benchmark_utils.py`
- `test_benchmark_rlm.py` (update bug-documenting tests to expected fixed behavior)
- `test_benchmark_utils.py` (add/adjust coverage for nested falsy values)

---

### F-050: Standardize benchmark ID handling

**Problem:** ID handling across benchmark flows can mismatch due to mixed raw types (`int`, `str`, UUID), especially during mapping and fallback logic.

**Design:**
- Introduce canonical benchmark ID helper(s) in `benchmark_utils.py`, e.g.:
  - deterministic point-id normalization helper
  - canonical key conversion helper (`str(...)`) for mapping operations
- Update benchmark code paths to use consistent canonicalization:
  - `benchmark_rlm.py` (`map_predicted_ids`, ingestion ID generation path)
  - if needed for parity, `benchmark_evolution.py` ingestion/mapping path
- Preserve backward compatibility by always mapping back to `payload['original_id']` when present.

**Files:**
- `benchmark_utils.py`
- `benchmark_rlm.py`
- `benchmark_evolution.py` (if required)
- `test_benchmark_rlm.py`
- `test_benchmark_utils.py`

---

## Test Strategy

Targeted after each finding:
- F-046/F-047: `python -m pytest test_antigravity_engine.py -q`
- F-048: `python -m pytest test_aep_orchestrator.py -q`
- F-049/F-050: `python -m pytest test_benchmark_rlm.py test_benchmark_utils.py -q`

Session regression gate:
- `python -m pytest (Get-ChildItem -Name test_*.py) -q`

---

## PR Stack Strategy

Planned branches (one finding per PR):
- `pr/f046-engine-decomposition`
- `pr/f047-invert-chelation-init`
- `pr/f048-tracker-status-validation`
- `pr/f049-payload-falsy-fix`
- `pr/f050-benchmark-id-standardization`
- `pr/session10-tracking-docs`

Planned stacked chain:
`pr/f046-engine-decomposition` -> `pr/f047-invert-chelation-init` -> `pr/f048-tracker-status-validation` -> `pr/f049-payload-falsy-fix` -> `pr/f050-benchmark-id-standardization` -> `pr/session10-tracking-docs`

---

## Risks and Mitigations

- **F-046 delegation drift risk:** preserve exact method contracts and side effects; run existing engine tests unchanged.
- **F-049 behavior-change risk:** update tests from “known bug” assertions to fixed expectations in same PR.
- **F-050 cross-script consistency risk:** centralize canonicalization helper and use targeted tests for mixed ID types.

---

## Acceptance Checklist

- [ ] F-046 extracted component module added and engine behavior preserved.
- [ ] F-047 `invert_chelation` explicitly initialized and `hasattr` removed.
- [ ] F-048 missing finding ID raises helpful `ValueError`.
- [ ] F-049 nested falsy payload values return correctly.
- [ ] F-050 benchmark ID handling canonicalized and tested.
- [ ] Targeted suites and full regression pass.
- [ ] Session tracking docs updated for Session 10.
