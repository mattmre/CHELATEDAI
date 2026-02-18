# Research: F-051 through F-055 Implementation

Cycle ID: `AEP-2026-02-13`  
Session Date: `2026-02-18`  
Scope: Final unresolved tranche (`F-051`..`F-055`)

## Executive Summary
- Backlog state at start: `50 / 55` resolved.
- Target findings are all low-severity, small-effort items.
- Codebase review confirms each finding is still open and directly remediable with bounded edits.

## Finding Breakdown

### F-051 -- bare `except` in `benchmark_rlm.py::map_predicted_ids`
- Location: `benchmark_rlm.py` (`map_predicted_ids`, current broad `except Exception` fallback).
- Gap: broad handler catches programming errors and runtime infrastructure errors alike.
- Remediation direction: catch concrete retrieval exceptions only; allow programmer bugs (e.g. `AttributeError`) to propagate.
- Test impact: update fallback test to use specific retrieval exception; add propagation test.

### F-052 -- integration tests skip silently in CI
- Location: `test_integration_rlm.py` (module-level `HAS_SENTENCE_TRANSFORMERS`, class-level `@skipUnless`).
- Gap: skip condition is present but missing-package state is not explicitly surfaced as a warning in collection output.
- Remediation direction: emit explicit warning when dependency is unavailable; keep skip behavior unchanged.
- Test impact: behavior validation via targeted run and warning visibility in CI output.

### F-053 -- config validation/test coverage gaps
- Location: `config.py`, `test_unit_core.py`.
- Gap: no `validate_max_depth` helper; no `rlm`/`sedimentation` preset families despite project convention; `get_config()` path not covered.
- Remediation direction: add small, static config primitives and focused tests, consistent with existing config style.
- Test impact: add coverage for `validate_max_depth`, preset retrieval by type, invalid preset type, and `get_config()`.

### F-054 -- AEP tracker test gaps
- Location: `aep_orchestrator.py`, `test_aep_orchestrator.py`.
- Gap: `get_unresolved`, `EffortSize.weight`, `Finding.to_dict`, and discovery default fallback paths are not directly asserted.
- Remediation direction: tests only (no production behavior changes).
- Test impact: add narrow tests in existing test classes.

### F-055 -- query text log injection in console output
- Location: `chelation_logger.py::log_query` (query snippet in message and JSON field).
- Gap: raw query text is sliced but not sanitized for newline/control chars before logging.
- Remediation direction: sanitize query snippet (`\n`, `\r`, control chars) before message/metadata logging.
- Test impact: add logger tests asserting sanitized query fields.

## Recommended Execution Order
1. F-051 (exception boundary tightening)
2. F-052 (CI observability improvement)
3. F-053 (config primitives + tests)
4. F-054 (test-only coverage additions)
5. F-055 (log sanitization + tests)

## Target Validation Commands
```powershell
python -m pytest test_benchmark_rlm.py -q
python -m pytest test_integration_rlm.py -q
python -m pytest test_unit_core.py -q
python -m pytest test_aep_orchestrator.py -q
python -m pytest test_chelation_logger.py -q
python -m pytest (Get-ChildItem -Name test_*.py) -q
```
