# Packaging Evaluation -- 2026-02-27

Session 22 housekeeping: evaluate current flat layout vs src/ layout for ChelatedAI.

## Current State

ChelatedAI uses a **flat file layout** with all `.py` files at the project root. There
is no `__init__.py`, no package directory, and no `src/` directory. The project is
configured as a PEP 621 project via `pyproject.toml` with explicit `py-modules` listing
under `[tool.setuptools]`.

### pyproject.toml Configuration

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "chelatedai"
version = "0.1.0"
requires-python = ">=3.9"

[tool.setuptools]
py-modules = [
    "aep_orchestrator",
    "antigravity_engine",
    "benchmark_comparative",
    # ... 22 modules total
]
```

### Current Module Count

- 22 production modules listed in py-modules
- 18+ test files (test_*.py) at project root
- 2 sweep scripts (run_sweep.py, run_large_sweep.py)
- 1 dashboard directory (dashboard/)
- 1 reference directory (rlm_reference/)
- 1 docs directory (docs/)

## Layout Options

### Option A: Keep Flat Layout (Current)

```
chelatedai/
    antigravity_engine.py
    chelation_adapter.py
    config.py
    ...
    test_unit_core.py
    test_convergence_monitor.py
    ...
    pyproject.toml
    requirements.txt
```

### Option B: src/ Layout

```
chelatedai/
    src/
        chelatedai/
            __init__.py
            antigravity_engine.py
            chelation_adapter.py
            config.py
            ...
    tests/
        test_unit_core.py
        test_convergence_monitor.py
        ...
    pyproject.toml
    requirements.txt
```

### Option C: Package Layout (no src/)

```
chelatedai/
    chelatedai/
        __init__.py
        antigravity_engine.py
        chelation_adapter.py
        config.py
        ...
    tests/
        test_unit_core.py
        test_convergence_monitor.py
        ...
    pyproject.toml
    requirements.txt
```

## Pros and Cons

### Flat Layout (Current)

**Pros:**
- Zero friction for research iteration -- edit and run directly
- No import path management or `__init__.py` maintenance
- All 529+ tests use direct module imports (`from antigravity_engine import ...`) that work without installation
- CI pipeline (`python -m unittest discover -s . -p "test_*.py"`) works without editable install
- pyproject.toml already handles the flat layout correctly via explicit py-modules listing
- Sweep scripts and dashboard can import modules directly
- No risk of "installed version vs local version" confusion during development
- Simpler mental model for contributors

**Cons:**
- Module namespace pollution if installed (`import config` could collide with other packages)
- No clear separation between production code, tests, and scripts
- `py-modules` list in pyproject.toml must be manually updated when adding new modules
- Cannot use relative imports (not applicable since there are no packages)
- Does not follow modern Python packaging best practices

### src/ Layout

**Pros:**
- Industry standard for distributable Python packages
- Forces installation for testing, catching packaging issues early
- Clean namespace (`from chelatedai.antigravity_engine import ...`)
- Clear separation of source, tests, and project metadata
- Prevents accidental import of local modules during testing

**Cons:**
- Requires `pip install -e .` before any test can run
- All 529+ tests would need import rewrites (`from chelatedai.antigravity_engine import ...`)
- All cross-module imports in 22 production files would need updating
- CI pipeline would need modification to install before testing
- Development friction increases significantly for a research prototype
- Sweep scripts and dashboard would need import path updates
- Adds complexity with no clear benefit for a non-distributed prototype

### Package Layout (no src/)

**Pros:**
- Namespaced imports without the overhead of src/
- Can still run tests without editable install (with path manipulation)
- Cleaner than flat layout for large projects

**Cons:**
- Same import rewrite burden as src/ layout
- "import chelatedai" could accidentally import local directory instead of installed package
- Still requires updating all cross-module imports

## Migration Cost Estimate

Moving to either src/ or package layout would require:

1. **Import rewrites in 22 production modules:** Every `from module_name import X` becomes `from chelatedai.module_name import X`. Estimated: ~100-150 import statements across all files.
2. **Import rewrites in 18+ test files:** Every `from module_name import X` and every `patch('module_name.get_logger')` mock path must be updated. Estimated: ~200-300 import/mock statements.
3. **CI pipeline update:** Add `pip install -e .` step before test execution.
4. **pyproject.toml restructure:** Replace `py-modules` with `packages = ["chelatedai"]` and `package-dir`.
5. **Sweep script and dashboard updates:** Import paths in run_sweep.py, run_large_sweep.py, and any dashboard backend code.
6. **Verification:** Full regression test suite must pass after migration.

**Total estimated effort:** 2-3 focused sessions. High risk of introducing subtle import bugs.

## Recommendation: Keep Flat Layout

**ChelatedAI is a research prototype, not a distributable library.** The flat layout is
the correct choice for the current project phase for the following reasons:

1. **Research velocity matters more than packaging purity.** The project is in active
   research iteration (22 sessions and counting). Any layout migration would consume 2-3
   sessions with zero feature value.

2. **The project is not distributed.** ChelatedAI is not published to PyPI, not installed
   by external users, and not imported as a dependency by other projects. The namespace
   collision risk is theoretical, not practical.

3. **The existing setup works.** pyproject.toml's `py-modules` list correctly declares
   all modules. `pip install -e .` works for editable development. CI passes. Tests run
   without installation via `unittest discover`.

4. **Migration cost is high relative to benefit.** Rewriting 300+ import statements across
   40+ files for a non-distributed prototype is not a good use of research time.

5. **The flat layout is explicitly documented.** CLAUDE.md, the architecture section, and
   multiple session logs document the flat layout as an intentional design choice. All
   contributors understand the convention.

### When to Reconsider

Migrate to src/ layout if and when any of these conditions become true:
- ChelatedAI is published to PyPI as an installable package
- External projects depend on ChelatedAI as an import
- The module count exceeds ~40 and navigation becomes difficult
- A sub-package hierarchy emerges naturally (e.g., chelatedai.adapters, chelatedai.benchmarks)

### Minor Improvement (Optional)

If desired, a lightweight organizational improvement can be made without changing the
layout: move test files into a `tests/` directory with a `conftest.py` or test runner
that adds the project root to `sys.path`. This separates tests from production code
while keeping the flat module layout intact. However, this is cosmetic and not recommended
as a priority.
