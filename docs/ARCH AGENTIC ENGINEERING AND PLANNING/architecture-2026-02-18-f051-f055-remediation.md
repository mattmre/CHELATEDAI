# Architecture Plan: F-051 through F-055

Cycle ID: `AEP-2026-02-13`  
Session Date: `2026-02-18`  
Mode: Fresh-agent orchestration with per-finding implementation and validation

## Design Principles
- Keep changes surgical and behavior-safe.
- Prefer test-first additions where possible.
- Preserve existing public behavior unless explicitly targeted by the finding.
- Use one reviewable PR slice per finding.

## Per-Finding Remediation Design

### F-051 -- `map_predicted_ids` exception narrowing
**Production changes**
- Update `benchmark_rlm.py::map_predicted_ids` to catch specific retrieval exceptions instead of broad `Exception`.
- Keep fallback output unchanged (`canonicalize_id` passthrough) for caught retrieval failures.

**Tests**
- Update exception fallback test to raise a concrete retrieval exception type.
- Add test verifying programming errors are not swallowed.

### F-052 -- integration-skip observability
**Production changes**
- In `test_integration_rlm.py`, emit a warning in the dependency-missing branch (`ImportError`) so CI logs clearly explain skipped integration tests.
- Keep `@unittest.skipUnless` gating unchanged.

**Tests/validation**
- Targeted run of `test_integration_rlm.py` to verify unchanged pass/skip behavior.

### F-053 -- config validation + preset completeness
**Production changes**
- Extend `ChelationConfig` with:
  - `validate_max_depth(...)`
  - `RLM_PRESETS`
  - `SEDIMENTATION_PRESETS`
- Update `get_preset(...)` to support preset types: `chelation`, `adapter`, `rlm`, `sedimentation`; raise clear error for unknown type.

**Tests**
- Add tests covering:
  - `validate_max_depth` clamping
  - `get_preset` for `rlm` and `sedimentation`
  - invalid `preset_type`
  - `get_config()` default and preset paths

### F-054 -- AEP tracker coverage gaps
**Production changes**
- None expected.

**Tests**
- Add direct tests for:
  - `AEPTracker.get_unresolved()`
  - `EffortSize.weight`
  - `Finding.to_dict()`
  - discovery fallback defaults for invalid/missing severity/effort fields

### F-055 -- query snippet sanitization for logging
**Production changes**
- Add small sanitization helper in `chelation_logger.py`.
- Sanitize query snippet before composing query log message and `query_snippet` metadata field.

**Tests**
- Add tests in `test_chelation_logger.py` to verify newline/control-character sanitization in logged query metadata.

## Validation Matrix
```powershell
python -m pytest test_benchmark_rlm.py -q
python -m pytest test_integration_rlm.py -q
python -m pytest test_unit_core.py -q
python -m pytest test_aep_orchestrator.py -q
python -m pytest test_chelation_logger.py -q
python -m pytest (Get-ChildItem -Name test_*.py) -q
```

## PR Split Plan
1. `pr/f051-map-predicted-ids-exception-narrowing`
2. `pr/f052-integration-skip-observability`
3. `pr/f053-config-validation-and-presets`
4. `pr/f054-aep-tracker-coverage-gaps`
5. `pr/f055-log-query-sanitization`
