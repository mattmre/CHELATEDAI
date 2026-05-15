# CHELATEDAI REFACTORING PLAN — PHASE 3 & PHASE 4

Generated: 2026-04-22
Scope: Phase 3 (Architecture Refactoring & Modularity) + Phase 4 (Code Quality & Type Safety)

================================================================================
PHASE 3: ARCHITECTURE REFACTORING & MODULARITY
================================================================================

3.1 RECOMMENDED PACKAGE STRUCTURE
---------------------------------

Proposed layout (shallow, flat-as-possible):

  src/
  └── chelatedai/
      ├── __init__.py              # Public API: AntigravityEngine, ChelationConfig, create_adapter
      ├── engine/
      │   ├── __init__.py
      │   ├── antigravity_engine.py  # God object → core orchestrator (~400 lines)
      │   ├── ingestion.py           # ingest(), ingest_streaming(), payload building
      │   ├── retrieval.py           # run_inference(), _gravity_sensor(), _chelate_toxicity()
      │   ├── training.py            # run_sedimentation_cycle(), run_offline_distillation(),
      │   │                            # training loops, Kalman LR, convergence
      │   │                            # spectral_chelation_ranking()
      │   └── adaptive.py            # enable_adaptive_threshold(), _update_adaptive_threshold(),
      │                            # enable_online_updates(), enable_teacher_weight_scheduling()
      │
      ├── adapters/
      │   ├── __init__.py
      │   ├── base.py                # AdapterABC abstract base
      │   ├── mlp_adapter.py
      │   ├── procrustes_adapter.py
      │   ├── low_rank_adapter.py
      │   ├── bounded_adapter.py
      │   └── factory.py            # create_adapter()
      │
      ├── embedding/
      │   ├── __init__.py
      │   ├── backend.py            # create_embedding_backend()
      │   ├── quality.py            # embedding_quality.py
      │   └── vector_store.py       # create_vector_store(), QdrantVectorStore
      │
      ├── training/
      │   ├── __init__.py
      │   ├── sedimentation_trainer.py  # compute_homeostatic_target()
      │   ├── sedimentation_loss.py     # InfoNCE, hybrid, hard negative mining
      │   ├── teacher_distillation.py   # DimensionProjection, EnsembleTeacherHelper
      │   ├── teacher_weight_scheduler.py
      │   ├── kalman_lr_scheduler.py
      │   ├── checkpoint_manager.py
      │   ├── convergence_monitor.py
      │   ├── online_updater.py
      │   ├── dimension_mask_predictor.py
      │   └── noise_injection.py        # Extract from sedimentation_trainer if separate
      │
      ├── diagnostics/
      │   ├── __init__.py
      │   ├── stability_tracker.py
      │   ├── topology_analyzer.py
      │   ├── isomer_detector.py
      │   └── structural_health_report.py  # get_structural_health_report() moved here or into engine
      │
      ├── analysis/
      │   ├── __init__.py
      │   └── recursive_decomposer.py   # RecursiveRetrievalEngine
      │
      ├── cross_lingual/
      │   ├── __init__.py
      │   ├── distillation.py           # cross_lingual_distillation.py
      │   └── language_detector.py
      │
      ├── evaluation/
      │   ├── __init__.py
      │   ├── base.py                   # Shared BenchmarkRunner base class
      │   ├── beir_benchmark.py
      │   ├── comparative_benchmark.py
      │   ├── distillation_benchmark.py
      │   ├── evolution_benchmark.py
      │   ├── multitask_benchmark.py
      │   ├── rlm_benchmark.py
      │   └── utils.py                  # Shared benchmark utilities
      │
      ├── dashboard/
      │   ├── __init__.py
      │   ├── server.py                 # HTTP server, DashboardHandler
      │   ├── events.py                 # load_events(), summarize_events(), filter_events()
      │   └── html/                     # Static dashboard HTML
      │       └── index.html
      │
      ├── config/
      │   ├── __init__.py
      │   ├── constants.py             # Magic numbers → named constants (see 4.6)
      │   ├── presets.py               # get_preset(), load/save presets
      │   └── validation.py            # All validate_* methods moved here
      │
      ├── logging_config/
      │   ├── __init__.py
      │   └── logger.py                # chelation_logger.py, get_logger()
      │
      └── sweep/
          ├── __init__.py
          ├── run_sweep.py
          ├── run_large_sweep.py
          └── run_overnight_campaign.py
          # run_weight_refinement_campaign.py could live here too

  tests/
  └── chelatedai/
      ├── test_engine/
      ├── test_adapters/
      ├── test_training/
      ├── test_diagnostics/
      ├── test_evaluation/
      ├── test_dashboard/
      ├── test_config/
      └── test_cross_lingual/
      # test files mirror src/ structure; test_*.py at root kept for backward compat

  computational_storage_poc/
  └── (keep as-is, sibling to src/)  # OR move under src/computational_storage/

  rlm_reference/
  └── (keep as-is, read-only reference)

  docs/
  └── (keep as-is)


3.2 GOD OBJECT DECOMPOSITION — AntigravityEngine (1621 lines → ~400)
--------------------------------------------------------------------

The AntigravityEngine has 7 distinct responsibility clusters. Each moves to its own module.

Responsibility Cluster 1: EMBEDDING & INGESTION (lines 135-354)
  Current methods: embed(), _sanitize_ollama_text(), ingest(), ingest_streaming()
  → Extract to: src/chelatedai/engine/ingestion.py
  New class: IngestionPipeline
  Depends on: embedding_backend, vector_store, ChelationConfig
  Lines: ~220

Responsibility Cluster 2: RETRIEVAL & CHELATION (lines 356-442)
  Current methods: _gravity_sensor(), _chelate_toxicity(), get_chelated_vector()
  → Extract to: src/chelatedai/engine/retrieval.py
  New class: RetrievalEngine
  Depends: vector_store, dimension_mask_predictor (optional)
  Lines: ~90

Responsibility Cluster 3: TRAINING LOOPS (lines 923-1506)
  Current methods: run_sedimentation_cycle(), run_offline_distillation()
  → Extract to: src/chelatedai/engine/training.py
  New class: SedimentationTrainer (wraps the training loop)
  The training loop logic is duplicated between sedimentation and distillation.
  Create a shared _run_training_loop() helper.
  Lines: ~400

Responsibility Cluster 4: ADAPTIVE FEATURES (lines 444-591)
  Current methods: enable_adaptive_threshold(), disable_adaptive_threshold(),
    get_threshold_stats(), _update_adaptive_threshold()
  → Extract to: src/chelatedai/engine/adaptive.py
  New class: AdaptiveThresholdManager
  Lines: ~150

Responsibility Cluster 5: ENABLE-XXX CONFIGURATION METHODS (lines 594-780)
  Current methods: enable_convergence_detection(), enable_kalman_lr(),
    set_temperature(), set_sedimentation_loss(), enable_online_updates(),
    enable_teacher_weight_scheduling(), enable_stability_tracking(),
    enable_topology_analysis(), enable_isomer_detection()
  → These are thin wrappers. Convert to constructor parameters or a builder pattern.
  Keep as methods on AntigravityEngine but remove enable_ prefix:
    convergence_detection, kalman_lr, temperature, sedimentation_loss,
    online_updates, weight_scheduler, stability_tracker, topology_analysis, isomer_detection

Responsibility Cluster 6: SPECTRAL CHELATION & SIMILARITY (lines 857-921)
  Current methods: _cosine_similarity_manual(), _spectral_chelation_ranking()
  → _cosine_similarity_manual is unused (numpy has np.dot + norms already used elsewhere)
     REMOVE: _cosine_similarity_manual
  → _spectral_chelation_ranking stays on AntigravityEngine (called from run_inference)
     OR move to: src/chelatedai/engine/retrieval.py as a standalone function
  Lines: ~65

Responsibility Cluster 7: STRUCTURAL HEALTH REPORT (lines 782-855)
  Current method: get_structural_health_report()
  → Extract to: diagnostics module or keep as a method on AntigravityEngine
     since it coordinates between trackers. Keep on engine, simplify.

RESULT: AntigravityEngine becomes a ~400-line orchestrator that:
  - Creates IngestionPipeline, RetrievalEngine, SedimentationTrainer internally
  - Delegates to them
  - Coordinates cross-cutting concerns (logging, checkpointing, adaptive threshold)


3.3 MODULE BOUNDARIES & DEPENDENCY GRAPH REDESIGN
--------------------------------------------------

Dependency direction: engine → embedding, engine → training, engine → diagnostics
embedding → (numpy, torch, sentence-transformers)
training → (numpy, torch) [no engine dependency]
diagnostics → (numpy, topology_analyzer, isomer_detector)
evaluation → (engine, numpy) [import at runtime, not compile time]

Key principle: NO circular dependencies. Training modules cannot import engine.
Engine imports training, embedding, diagnostics — one-way flow.

Before: All flat, all can import all
After:
  chelatedai.engine          ← imports from all subpackages
  chelatedai.embedding       ← no chelatedai imports
  chelatedai.training        ← no chelatedai imports  
  chelatedai.diagnostics     ← no chelatedai imports
  chelatedai.analysis        ← imports chelatedai.engine
  chelatedai.cross_lingual   ← imports chelatedai.training
  chelatedai.evaluation      ← imports chelatedai.engine
  chelatedai.dashboard       ← no chelatedai imports (standalone)
  chelatedai.config          ← no chelatedai imports
  chelatedai.logging_config  ← no chelatedai imports


3.4 BENCHMARK INFRASTRUCTURE CONSOLIDATION
-------------------------------------------

Current state: 7 benchmark files (~125k total lines), all independent,
no shared base class, duplicated setup/teardown/evaluation logic.

Actions:
  a) Create src/chelatedai/evaluation/base.py:
     - Abstract BenchmarkRunner class with:
       * abstract run_benchmark() -> dict
       * shared setup_dataset(dataset_name)
       * shared evaluate(engine, queries, ground_truth, metrics)
       * shared generate_report(results, format)
       * shared save_results(results, path)
     - Common metric functions: ndcg_at_k, recall_at_k, mrr_at_k, hit_rate

  b) Create src/chelatedai/evaluation/utils.py:
     - Dataset loading utilities (moved from each benchmark file)
     - Result formatting helpers
     - Config preset resolution
     - Move benchmark_utils.py here

  c) Each benchmark file becomes a thin subclass:
     class BEIRBenchmark(BenchmarkRunner):
         def run_benchmark(self): ...  # BEIR-specific logic
     
     class ComparativeBenchmark(BenchmarkRunner): ...
     class DistillationBenchmark(BenchmarkRunner): ...
     class EvolutionBenchmark(BenchmarkRunner): ...
     class MultitaskBenchmark(BenchmarkRunner): ...
     class RLMBenchmark(BenchmarkRunner): ...

  d) Create a benchmark CLI entry point:
     chelatedai/benchmark.py → handles --dataset, --config, --output flags
     Or use the existing run_sweep.py as the sweep entry point.

  e) Consolidate test files:
     test_benchmark_utils.py → tests/chelatedai/evaluation/test_utils.py
     test_benchmark_beir.py → tests/chelatedai/evaluation/test_beir_benchmark.py
     etc.

  Estimated effort: 15-20 hours
  Priority: Medium (not blocking core functionality)


3.5 DASHBOARD SERVER EXTRACTION
--------------------------------

Current state: 928 lines, monolithic single file with:
  - Inline HTML (~500 lines of CSS/JS embedded in Python string)
  - HTTP handler with 5 API endpoints
  - JSONL event loading/parsing
  - Sweep results, test results, BEIR results endpoints
  - Auth middleware

Actions:
  a) Split into 3 files:
     src/chelatedai/dashboard/server.py:
       - DashboardHandler class
       - run_server() function
       - main() CLI entry point
       - ~250 lines
     
     src/chelatedai/dashboard/events.py:
       - load_events()
       - summarize_events()
       - filter_events()
       - ~120 lines
     
     src/chelatedai/dashboard/html/index.html:
       - Extract the inline HTML/CSS/JS into a real HTML file
       - ~400 lines
       - Serve from file, not inline string

  b) Remove get_inline_dashboard_html() fallback.
     The server should only serve from dashboard/html/index.html.

  c) Consider separating sweep/test/BEIR result endpoints into
     their own handler classes for SRP:
       SweepResultsController, TestResultsController, BEIRResultsController

  Estimated effort: 4-6 hours


3.6 COMPUTATIONAL STORAGE POC CLEANUP
--------------------------------------

Current state: computational_storage_poc/ directory at root level,
mixed with core modules. Contains RP2040/TinyUSB firmware POC code,
hardware evidence capture, emulation, and payload transport.

Recommendation: Move under src/ as a feature subpackage:

  src/chelatedai/computational_storage/
  ├── __init__.py
  ├── firmware_build.py
  ├── capture_hardware_evidence.py
  ├── emulation.py
  ├── payload_transport.py
  ├── usb_host_inference.py
  └── ...

This is a breaking change IF any root-level imports reference these files.
Check all test_*.py and other .py files for "from computational_storage_poc import"
or "import computational_storage_poc" references.

Migration:
  - Add backward-compat shim at root:
    computational_storage_poc/__init__.py → from chelatedai.computational_storage import *
  - Update all imports in test files
  - Update pyproject.toml package discovery

Estimated effort: 3-5 hours


3.7 RLM REFERENCE CODE CLEANUP
-------------------------------

Current state: rlm_reference/ directory at root (read-only cloned paper impl)
AND GITHUBCHELATEDAIrlm_reference/ (incorrectly named, likely duplicate/corrupt)

Actions:
  a) DELETE GITHUBCHELATEDAIrlm_reference/ — it's incorrectly named and likely
     a copy artifact. Confirm it's not needed.
  
  b) MOVE rlm_reference/ to docs/rlm_reference/ — it's reference material,
     not source code. This is consistent with CLAUDE.md stating:
     "rlm_reference/ -- Cloned RLM paper implementation (read-only, do not modify)"

  c) Add to .gitignore the build artifacts:
     __pycache__/, .ruff_cache/, build/, checkpoints/, adapter_weights.pt
     *.benchmark-backup-*.pt, chelation_debug.jsonl, sweep_results.json
     overnight_campaign_*.log, experiment_runs/, db_*/

Estimated effort: 1 hour


3.8 MIGRATION STRATEGY (Backward Compatibility)
-------------------------------------------------

CRITICAL: The existing import pattern is:
  from module import Class
  from antigravity_engine import AntigravityEngine
  from config import ChelationConfig

Migration must preserve these imports during a transition period.

Phase 3a — Skeleton packages (no breaking changes):
  1. Create src/chelatedai/__init__.py with:
     from ..antigravity_engine import AntigravityEngine  # re-export
     from ..config import ChelationConfig, get_config  # re-export
     from ..chelation_adapter import create_adapter, BoundedAdapter  # re-export
     
  2. Create stub __init__.py files in each subpackage.

  3. Use sys.path manipulation in __init__.py to find sibling modules:
     import sys, os
     sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Phase 3b — Module migration (maintain old imports):
  4. For each file moved, leave a shim at the old location:
     # antigravity_engine.py (shim)
     from chelatedai.engine.antigravity_engine import AntigravityEngine
     __all__ = ["AntigravityEngine"]
  
  5. Update imports in moved files to use chelatedai.* paths.

Phase 3c — Full migration (breaking):
  6. Remove root-level shims.
  7. Update pyproject.toml to use src/ layout.
  8. Run full test suite.

Phase 3d — Cleanup:
  9. Remove root-level test_*.py files (moved to tests/).
  10. Delete rlm_reference/ from root (moved to docs/).
  11. Delete GITHUBCHELATEDAIrlm_reference/ (artifact).

Total estimated effort for Phase 3: 40-60 hours
Recommended sprint allocation: 4-6 weeks (2 sprints)


3.9 BREAKING CHANGE MANAGEMENT
-------------------------------

Breaking changes introduced:
  - Import paths change: from antigravity_engine import → from chelatedai.engine import
  - Module locations change: root → src/chelatedai/
  - File renames: _cosine_similarity_manual() removed (unused)

Backward compatibility approach:
  - Keep root-level shim files during Phase 3a/3b
  - Add deprecation warnings in shim files:
    import warnings
    warnings.warn(
      "Importing from 'antigravity_engine' is deprecated. "
      "Use 'from chelatedai.engine import AntigravityEngine' instead. "
      "Shim will be removed in v2.0.",
      DeprecationWarning,
      stacklevel=2
    )
  - CI: Run both old import style and new import style tests
  - Document migration in CHANGELOG.md

No breaking changes required for:
  - Public API methods (AntigravityEngine interface stays the same)
  - ChelationConfig (class interface stays the same)
  - create_adapter() factory (same signature)
  - All test files (they will be migrated too)

================================================================================
PHASE 4: CODE QUALITY & TYPE SAFETY
================================================================================

4.1 TYPE ANNOTATION STRATEGY
-----------------------------

Python version target: 3.9+ (CI runs 3.9, 3.10, 3.11, 3.12)

Strategy: Use `from __future__ import annotations` at the top of every module.
This enables PEP 604-style union syntax (`X | None`) while maintaining 3.9 compat.

Priority order (by impact/criticality):
  1. config.py — Most imported file, high-impact
  2. antigravity_engine.py — God object, all consumers
  3. chelation_adapter.py — Factory pattern, widely used
  4. embedding_backend.py — Cross-cutting dependency
  5. vector_store.py — Cross-cutting dependency
  6. sedimentation_trainer.py — Training dependency
  7. teacher_distillation.py — Training dependency
  8. dashboard_server.py — Already well-typed
  9. All benchmark files — Low priority (script-like)

Type annotation patterns to use:

  # PEP 604 unions (with __future__ annotations)
  def method(self, value: str | None = None) -> dict[str, Any]: ...

  # Explicit Optional for clarity on public APIs
  from typing import Optional, Sequence, Mapping, Protocol, TypeVar, Generic
  
  T = TypeVar("T")
  
  class AdapterABC(Protocol[T]):
      def forward(self, x: torch.Tensor) -> torch.Tensor: ...
      def parameters(self) -> Iterator[torch.nn.Parameter]: ...

  # Protocol for duck typing (adapters)
  class HasParameters(Protocol):
      def parameters(self) -> Iterator[torch.nn.Parameter]: ...
      def train(self) -> None: ...
      def eval(self) -> None: ...
      def save(self, path: str) -> bool: ...

  # TypedDict for config/result dicts
  from typing import TypedDict
  class BenchmarkResult(TypedDict):
      dataset: str
      ndcg_at_10: float
      recall_at_10: float
      mrr: float

  # Literal for enum-like parameters
  from typing import Literal
  LossType = Literal["mse", "infonce", "hybrid"]
  TrainingMode = Literal["baseline", "offline", "hybrid"]
  AdapterType = Literal["mlp", "procrustes", "low_rank"]

Specific type annotations needed per module:

  config.py:
    - All validate_* methods already have type hints ✓
    - get_preset: (preset_name: str, preset_type: str) -> dict[str, Any]
    - load_from_file: already typed ✓
    - save_to_file: already typed ✓
    - get_config: (preset: str | None) -> dict[str, Any]
    - get_db_path: (task_name: str) -> Path
    - validate_safe_path: already typed ✓
    - sanitize_name: already typed ✓
    Add: TypedDict for preset return types

  antigravity_engine.py:
    - __init__: Add full type hints for all params
    - embed: (texts: str | list[str]) -> np.ndarray
    - ingest: (text_corpus: list[str], payloads: list[dict] | None) -> None
    - ingest_streaming: add full annotations
    - _gravity_sensor: (query_vec: np.ndarray, top_k: int) -> np.ndarray
    - _chelate_toxicity: (local_cluster: np.ndarray) -> np.ndarray
    - get_chelated_vector: (query_text: str) -> np.ndarray
    - run_inference: (query_text: str) -> tuple[list, list, np.ndarray, float]
    - run_sedimentation_cycle: Add full annotations
    - run_offline_distillation: already partially typed ✓
    - close: () -> None
    - __enter__: () -> AntigravityEngine
    - __exit__: (exc_type, exc_val, exc_tb) -> bool
    Add: Type alias for QueryResult = tuple[list[int], list[int], np.ndarray, float]

  chelation_adapter.py:
    - create_adapter: (adapter_type: AdapterType, input_dim: int, ...) -> HasParameters
    - Each adapter class: full method signatures

  benchmark files:
    - Minimal typing — these are script-like entry points
    - Focus on main() signatures and any reusable functions

Estimated effort: 20-30 hours
Priority: HIGH — enables mypy/Pyright static analysis


4.2 DOCSTRING STANDARDIZATION
-------------------------------

Format: Google style (most common in the codebase, most concise)

Coverage target: 80% of public methods/classes by end of Phase 4.

Google format:
  def method_name(self, param1: str, param2: int = 10) -> bool:
      """One-line summary.
      
      Longer description if needed. Explains the 'why' not the 'what'.
      
      Args:
          param1: Description of param1
          param2: Description of param2 with default value noted
      
      Returns:
          Description of return value
      
      Raises:
          ValueError: When condition X occurs
          FileNotFoundError: When condition Y occurs
      
      Examples:
          >>> obj.method_name("hello", 5)
          True
      """

Modules needing docstrings (by priority):

  HIGH PRIORITY (core API):
    1. antigravity_engine.py:
       - __init__ (already has docstring ✓)
       - embed: ✓ (has docstring)
       - ingest: ✓ (has docstring)
       - ingest_streaming: ✓ (has docstring)
       - run_inference: ✗ MISSING
       - run_sedimentation_cycle: ✗ MISSING
       - run_offline_distillation: ✗ MISSING
       - enable_adaptive_threshold: ✓ (has docstring)
       - enable_convergence_detection: ✓ (has docstring)
       - enable_kalman_lr: ✓ (has docstring)
       - set_temperature: ✓ (has docstring)
       - set_sedimentation_loss: ✓ (has docstring)
       - enable_online_updates: ✓ (has docstring)
       - enable_teacher_weight_scheduling: ✓ (has docstring)
       - get_structural_health_report: ✓ (has docstring)
       - close: ✗ MISSING
       Total: 4 methods missing docstrings
   
    2. config.py:
       - ChelationConfig class: ✓ (has docstring)
       - get_preset: ✓ (has docstring)
       - load_from_file: ✓ (has docstring)
       - save_to_file: ✓ (has docstring)
       - get_db_path: ✓ (has docstring)
       - get_config: ✓ (has docstring)
       - validate_safe_path: ✓ (has docstring)
       - sanitize_name: ✓ (has docstring)
       - All validate_* methods: ✗ MISSING (8 methods)
       Total: 8 methods missing docstrings

    3. chelation_adapter.py:
       - create_adapter: Check
       - Each adapter class __init__: Check
       - Each adapter class forward: Check
       Estimated: 10-15 docstrings needed

    4. embedding_backend.py:
       - create_embedding_backend: Check
       - EmbeddingBackend methods: Check
       Estimated: 5-8 docstrings needed

    5. vector_store.py:
       - create_vector_store: Check
       - QdrantVectorStore methods: Check
       Estimated: 5-8 docstrings needed

  MEDIUM PRIORITY (training/diagnostics):
    6. teacher_distillation.py: ~5-10 docstrings
    7. sedimentation_trainer.py: ~3-5 docstrings
    8. online_updater.py: ~3-5 docstrings
    9. stability_tracker.py: ~3-5 docstrings
    10. topology_analyzer.py: ~3-5 docstrings
    11. isomer_detector.py: ~3-5 docstrings

  LOW PRIORITY (scripts/benchmarks):
    12. All benchmark files: Minimal (main() function docstrings only)
    13. dashboard_server.py: Already well-documented ✓
    14. sweep scripts: Minimal

  Currently well-documented (✓):
    - dashboard_server.py: Full docstrings on all public functions ✓
    - config.py: Good class-level and method-level docs ✓
    - antigravity_engine.py: Good on most public methods ✓

Estimated effort: 10-15 hours


4.3 NAMING CONVENTION ENFORCEMENT
----------------------------------

Rule 1: FILE NAMES — snake_case only
  Current violations (none found — all root files are snake_case ✓):
    antigravity_engine.py ✓
    chelation_adapter.py ✓
    teacher_distillation.py ✓
    All test files follow test_<name>.py pattern ✓

Rule 2: FUNCTION/METHOD NAMES — snake_case only
  Current violations found:
    - _sanitize_ollama_text() ✓ (already snake_case)
    - _gravity_sensor() ✓ (already snake_case)
    - _chelate_toxicity() ✓ (already snake_case)
    - _cosine_similarity_manual() ✓ (already snake_case, but unused)
    - _spectral_chelation_ranking() ✓ (already snake_case)
    - _update_adaptive_threshold() ✓ (already snake_case)
    - _is_loopback_host() ✓ (already snake_case)
    - _is_api_authorized() ✓ (already snake_case)
    - _is_ai_request() — check for camelCase methods
    Total violations: 0 found (all snake_case)

Rule 3: CLASS NAMES — PascalCase
  Current violations:
    - AntigravityEngine ✓
    - ChelationConfig ✓
    - ChelationLogger ✓
    - All adapter classes ✓
    Total violations: 0

Rule 4: CONSTANT NAMES — UPPER_SNAKE_CASE
  Current violations:
    - Config class constants: ✓ (ALL UPPER_SNAKE_CASE)
    - Module-level globals in dashboard_server.py: ✓ (ALL UPPER_SNAKE_CASE)
    Total violations: 0

Rule 5: VARIABLE NAMES — snake_case
  Check for camelCase variables (common Python anti-pattern):
    grep -rn "[a-z][a-z][a-z]*[A-Z]" *.py | grep -v "test_" | grep -v "# "
    Look for patterns like: docId, eventName, topK, batchSize, etc.

Rule 6: BOOLEAN FLAGS — enable_ prefix or is_/has_ prefix
  Current patterns:
    - enable_xxx(): ✓ (enable_stability_tracking, enable_online_updates)
    - _enabled suffix: ✓ (_adaptive_threshold_enabled, _kalman_lr_enabled)
    These are acceptable conventions.

Enforcement via ruff:
  # pyproject.toml
  [tool.ruff.lint]
  extend-select = ["N"]  # Naming conventions (pep8-naming)
  
  [tool.ruff.lint.pep8-naming]
  ignore-names = ["X", "x", "T", "Q", "K"]  # Scientific/mathematical symbols
  # Or: extend-ignore = ["N801", "N802", "N803", "N806", "N816"]
  # if using camelCase in third-party libraries

  R801: Class names should use PascalCase
  R802: Function names should use snake_case
  R803: Argument names should use snake_case
  R804: Arguments shouldn't be `*args` (consider named)
  R805: Use `self` in single-argument method
  R806: Function return should have type annotation

Estimated effort: 2-4 hours (mostly ruff configuration + fixing violations)


4.4 CODE SMELL FIXES
---------------------

Smell 1: GOD OBJECT — antigravity_engine.py (1621 lines)
  Fix: See Phase 3.2 decomposition plan.
  Effort: 15-20 hours

Smell 2: MAGIC NUMBERS scattered throughout codebase
  Fix: See 4.6 (magic number extraction).
  Specific instances:
    - antigravity_engine.py line 113: `scalar=models.ScalarType.INT8`
    - antigravity_engine.py line 116: `quantile=0.99`
    - antigravity_engine.py line 163: `> ChelationConfig.OLLAMA_INPUT_MAX_CHARS`
    - antigravity_engine.py line 284: `range(batch_size)`
    - antigravity_engine.py line 619: `process_noise=0.1` (hardcoded default)
    - antigravity_engine.py line 619: `max_lr_ratio=2.0` (hardcoded default)
    - antigravity_engine.py line 859: `== 0` (zero comparison — acceptable)
    - antigravity_engine.py line 884: `max_entries = ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC` ✓
    - antigravity_engine.py line 1097: `max=ChelationConfig.NOISE_INJECTION_MAX_SCALE` ✓
    - antigravity_engine.py line 1166: `0.01` (regularization weight — MAGIC NUMBER)
    - antigravity_engine.py line 1184: `epoch % max(1, epochs // 2)` (logging interval)
    - config.py line 216: `0.0078` (comment, not code — acceptable)
    - Many benchmark files have hardcoded batch_size=100, top_k=10, etc.
  Effort: 3-5 hours

Smell 3: DUPLICATE TRAINING LOOPS
  Fix: run_sedimentation_cycle() and run_offline_distillation() share ~80 lines
  of nearly identical training loop code (optimizer setup, loss computation,
  Kalman LR update, convergence check, weight scheduling).
  Extract to _run_training_loop(input_tensor, target_tensor, optimizer, criterion,
  adapter, **loop_options) and call from both methods.
  Effort: 2-3 hours

Smell 4: INLINE HTML IN PYTHON STRING (dashboard_server.py)
  Fix: See 3.5 (dashboard extraction).
  Effort: 4-6 hours

Smell 5: EXCEPTIONS SWALLOWED WITH GENERIC except Exception
  Instances:
    - antigravity_engine.py line 1044: `except Exception as e:`
    - antigravity_engine.py line 1077: `except Exception as e:`
    - antigravity_engine.py line 1312: `except Exception as e:`
    - antigravity_engine.py line 1359: `except Exception as e:`
    - antigravity_engine.py line 1493: `except Exception as e:`
    - antigravity_engine.py line 1608: `except Exception as e:`
    - dashboard_server.py: multiple `except Exception as e:` in API handlers
  Fix: See 4.7 (error handling standardization).
  Effort: 3-4 hours

Smell 6: _cosine_similarity_manual() — UNUSED METHOD
  antigravity_engine.py line 857-863: _cosine_similarity_manual() is defined but
  never called. The code uses numpy vectorized operations instead.
  Fix: DELETE this method.
  Effort: 0.5 hours

Smell 7: `from __future__ import annotations` MISSING in most modules
  Fix: Add to all .py files that use `X | None` or `list[X]` type hints.
  Effort: 2-3 hours (automatable)

Smell 8: GLOBAL MUTABLE STATE in dashboard_server.py
  Lines 27-29: LOG_FILE_PATH, DASHBOARD_TOKEN, DASHBOARD_CORS_ORIGIN are global
  variables modified at runtime in run_server().
  Fix: Pass as parameters to DashboardHandler constructor or use a config object.
  Effort: 2-3 hours

Smell 9: BACKWARD COMPATIBILITY REFERENCE
  antigravity_engine.py line 104: `self.qdrant = self._vector_store`
  This creates two references to the same object. Keep it but document why.
  Effort: 0.5 hours (documentation only)

Smell 10: PRINT STATEMENTS IN VALIDATION METHODS
  config.py lines 730, 738, 747, 749, 758, 766, 774, 782, 789, 810:
  All validate_* methods use `print(f"WARNING: ...")` instead of logging.
  Fix: Replace with self.logger.warning() or logging.warning().
  Need to add a module-level logger to config.py or use get_logger().
  Effort: 2-3 hours

Estimated total effort for code smell fixes: 30-45 hours


4.5 LINTING CONFIGURATION
--------------------------

Current state: CI uses `ruff check .` with no specific configuration.
No pyproject.toml [tool.ruff] section exists (or it's minimal).

Required pyproject.toml additions:

  [tool.ruff]
  target-version = "py39"
  line-length = 120
  src = ["src"]
  
  [tool.ruff.lint]
  select = [
      "E",      # pycodestyle errors
      "W",      # pycodestyle warnings
      "F",      # Pyflakes (unused imports, undefined names)
      "I",      # isort (import sorting)
      "N",      # pep8-naming (naming conventions)
      "C90",    # mccabe (complexity)
      "B",      # flake8-bugbear (bug patterns)
      "UP",     # pyupgrade (Python 3.9+ syntax)
      "RUF",    # Ruff-specific rules
      "S",      # flake8-bandit (security)
      "T20",    # flake8-print (print statements)
      "TID25",  # flake8-tidy-imports (ban relative imports)
      "D",      # pydocstyle (docstrings)
      "A",      # flake8-builtins (shadowing builtins)
      "PIE",    # flake8-pie (unnecessary passes/returns)
      "T10",    # flake8-debugger (debug statements)
      "TCH",    # flake8-type-checking (TYPE_CHECKING imports)
      "ISC",    # flake8-implicit-str-concat
      "COM",    # flake8-commas
      "FA",     # flake8-future-annotations
      "PT",     # flake8-pytest-style (for test files)
  ]
  
  ignore = [
      "E501",   # line length (handled by line-length setting)
      "N801",   # Class name casing (we use PascalCase, skip if false positives)
      "N803",   # Argument name casing
      "N806",   # Variable name casing in functions
      "N816",   # Mixed case module names
      "D100",   # Missing docstring in public module (we have module-level docs)
      "D104",   # Missing docstring in public package
      "D203",   # 1 blank line before class docstring
      "D212",   # Multi-line docstring should start at line 1
      "S101",   # Use of assert (needed for tests)
      "T201",   # Use of print (needed in some scripts)
      "B008",   # Function call in argument defaults (acceptable in Config class)
      "B905",   # zip() without explicit strict=True (Python 3.10+)
  ]
  
  [tool.ruff.lint.per-file-ignores]
  "test_*.py" = [
      "S101",   # Allow assert in tests
      "T201",   # Allow print in tests
      "D100",   # No module docstring needed for tests
      "D101",   # No class docstring needed for test classes
      "D102",   # No method docstring needed for test methods
      "D103",   # No docstring needed for test methods
      "PT011",  # pytest.raises pattern flexibility
  ]
  "benchmarks/**/*.py" = [
      "T201",   # Allow print in benchmark scripts
      "D100",   # No module docstring needed
  ]
  "rlm_reference/**/*.py" = [
      "ALL",    # Ignore all linting for reference code
  ]
  
  [tool.ruff.lint.mccabe]
  max-complexity = 12  # Currently: AntigravityEngine methods have up to ~30
  
  [tool.ruff.lint.pydocstyle]
  convention = "google"
  # Or use "numpy" or "pep257" — google is recommended
  
  [tool.ruff.lint.isort]
  known-first-party = ["chelatedai"]
  force-single-line = false
  profile = "black"  # If using black formatter

Additional tool: pyright or mypy for type checking

  [tool.pyright]
  include = ["src/chelatedai"]
  exclude = ["**/test_*.py", "**/__pycache__"]
  typeCheckingMode = "basic"  # Start with "basic", upgrade to "strict"
  pythonVersion = "3.9"
  reportMissingImports = true
  reportMissingTypeStubs = false
  pythonPlatform = "All"

  OR for mypy:
  [tool.mypy]
  python_version = "3.9"
  warn_return_any = true
  warn_unused_configs = true
  disallow_untyped_defs = false  # Start permissive
  # Gradually tighten to:
  # disallow_untyped_defs = true
  ignore_missing_imports = true
  follow_imports = "silent"

Estimated effort: 1-2 hours (configuration) + ongoing (fixing violations)


4.6 MAGIC NUMBER EXTRACTION
----------------------------

Create: src/chelatedai/config/constants.py

  """Named constants for magic numbers used across the codebase."""

  from pathlib import Path
  from typing import Final

  # -- Training constants --
  ADAPTER_REGULARIZATION_WEIGHT: Final[float] = 0.01       # antigravity_engine.py:1166
  TRAINING_LOG_INTERVAL_FRACTION: Final[float] = 0.5       # antigravity_engine.py:1184 (epochs // 2)
  DEFAULT_KALMAN_PROCESS_NOISE: Final[float] = 0.1          # antigravity_engine.py:619
  DEFAULT_KALMAN_MIN_LR_RATIO: Final[float] = 0.1           # antigravity_engine.py:619
  DEFAULT_KALMAN_MAX_LR_RATIO: Final[float] = 2.0           # antigravity_engine.py:619
  DEFAULT_KALMAN_WINDOW_SIZE: Final[int] = 10               # antigravity_engine.py:619
  
  # -- Quantization constants --
  SCALAR_QUANTIZATION_QUANTILE: Final[float] = 0.99         # antigravity_engine.py:116
  SCALAR_QUANTIZATION_TYPE: Final[str] = "INT8"             # antigravity_engine.py:113
  
  # -- Embedding constants --
  OLLAMA_INPUT_MAX_CHARS: Final[int] = 10000                # antigravity_engine.py:163
  OLLAMA_TRUNCATION_LIMITS: Final[list[int]] = [6000, 2000, 500]  # config.py:108
  
  # -- Retrieval constants --
  SCOUT_NEIGHBORHOOD_SIZE: Final[int] = 50                 # alias for SCOUT_K
  TOP_K_RESULTS: Final[int] = 10                           # alias for TOP_K
  
  # -- Logging constants --
  QUERY_SNIPPET_LENGTH: Final[int] = 50                    # config.py:715
  CHELATION_LOG_MAX_ENTRIES_PER_DOC: Final[int] = 1000     # config.py:705
  
  # -- Batch/chunk constants --
  DEFAULT_BATCH_SIZE: Final[int] = 100                     # config.py:157
  UPDATE_CHUNK_SIZE: Final[int] = 100                      # config.py:698
  STREAMING_BATCH_SIZE: Final[int] = 100                   # config.py:701
  STREAMING_PROGRESS_INTERVAL: Final[int] = 10             # config.py:702
  
  # -- Scroll/pagination constants --
  SCROLL_PAGE_SIZE: Final[int] = 10000                     # antigravity_engine.py:1292
  
  # -- Noise injection constants --
  NOISE_INJECTION_BASE_SCALE: Final[float] = 0.05          # config.py:127
  NOISE_INJECTION_MAX_SCALE: Final[float] = 0.5            # config.py:128
  
  # -- Teacher distillation constants --
  TEACHER_BATCH_SIZE: Final[int] = 64                      # config.py:288
  ENSEMBLE_MAX_WORKERS: Final[int] = 4                     # config.py:293
  MAX_CORPUS_CHUNK: Final[int] = 10000                     # config.py:291
  
  # -- Dashboard constants --
  DEFAULT_DASHBOARD_HOST: Final[str] = "127.0.0.1"         # dashboard_server.py:861
  DEFAULT_DASHBOARD_PORT: Final[int] = 8080                # dashboard_server.py:861
  
  # -- HTTP constants --
  HTTP_OK: Final[int] = 200
  HTTP_UNAUTHORIZED: Final[int] = 401
  HTTP_NOT_FOUND: Final[int] = 404
  HTTP_SERVER_ERROR: Final[int] = 500

Migration: Replace all hardcoded instances with imports from constants.
  e.g., antigravity_engine.py line 1166:
    Before: loss = loss + 0.01 * reg_loss
    After:  loss = loss + ADAPTER_REGULARIZATION_WEIGHT * reg_loss

Note: Many "magic numbers" in config.py are ALREADY named constants.
The main issue is hardcoded numbers in antigravity_engine.py and benchmark files
that don't reference ChelationConfig.

Estimated effort: 5-8 hours


4.7 ERROR HANDLING STANDARDIZATION
------------------------------------

Define custom exception hierarchy:

  src/chelatedai/exceptions.py:

    class ChelatedAIError(Exception):
        """Base exception for all ChelatedAI errors."""
        pass

    class ConfigurationError(ChelatedAIError):
        """Invalid or missing configuration."""
        pass

    class EmbeddingError(ChelatedAIError):
        """Error during embedding generation."""
        pass

    class RetrievalError(ChelatedAIError):
        """Error during vector retrieval."""
        pass

    class TrainingError(ChelatedAIError):
        """Error during training/sedimentation."""
        pass

    class CheckpointError(ChelatedAIError):
        """Error during checkpoint save/load."""
        pass

    class VectorStoreError(ChelatedAIError):
        """Error interacting with vector store."""
        pass

    class DashboardError(ChelatedAIError):
        """Error in dashboard server."""
        pass

  Update exception handling in:
    - antigravity_engine.py: Replace `except Exception as e:` with specific exceptions
    - embedding_backend.py: Catch embedding-specific errors
    - vector_store.py: Catch Qdrant-specific errors → VectorStoreError
    - checkpoint_manager.py: Catch file I/O errors → CheckpointError
    - dashboard_server.py: Keep generic exceptions for HTTP handlers (already handled)

  Standard try/except pattern:
    try:
        result = risky_operation()
    except SpecificError as e:
        logger.error(f"Operation failed: {e}")
        raise  # Re-raise if caller needs to handle
    except (ExpectedError1, ExpectedError2) as e:
        logger.warning(f"Non-critical failure: {e}")
        return fallback_value  # Graceful degradation
    except Exception as e:
        logger.exception("Unexpected error during operation")
        raise  # Let it propagate for CI/debugging

  Ruff rules to enforce:
    B012: return inside try/except/finally (bad pattern)
    B017: @pytest.mark.xfail with AssertionError (test files)
    S110: try/except Exception (flag bare except)
    S112: try/except OSError (too broad)

  Replace `except Exception` with more specific catches:
    - Qdrant errors: `from qdrant_client.http.exceptions import ...`
    - Torch errors: `torch.nn.modules.ModuleError`
    - File I/O: `FileNotFoundError`, `OSError`, `PermissionError`
    - JSON: `json.JSONDecodeError`
    - Network: `requests.RequestException` (if using requests)

Estimated effort: 5-8 hours


4.8 LOGGING CONSISTENCY
-------------------------

Current state: chelation_logger.py provides get_logger() with JSON structured logging.
Most modules use `self.logger = get_logger()` but some use print() directly.

Actions:
  1. Ensure ALL modules use get_logger():
     grep -rn "print(" *.py | grep -v "test_" | grep -v "# " | grep -v "if __name__"
     Check for: print(f"WARNING:", print("Dashboard server starting...",
     print(f"  Host: {host}"), etc.

  2. Fix print() usages:
     a) config.py validation methods: Replace print(f"WARNING:...") with logger.warning()
        Need: Add `_logger = get_logger()` at module level in config.py
     b) dashboard_server.py: The print() in run_server() and log_message() are
        intentional for CLI output. Keep them but add logger.info() alongside.
     c) sweep scripts (run_sweep.py, run_large_sweep.py): These are CLI tools.
        print() is acceptable for user-facing output, but add logger.debug() for
        machine-readable output.

  3. Standardize log event names:
     Current event names in chelation_logger.py:
       - "initialization", "adapter_init", "collection_init"
       - "ingestion_start", "ingestion_progress", "ingestion_complete"
       - "sedimentation_start", "sedimentation_success", "early_stopping"
       - "adaptive_threshold_enabled", "adaptive_threshold_update"
       - "online_updates_enabled", "weight_scheduler_enabled"
       - "qdrant" (for errors)
     
     Add missing event names:
       - "distillation_teacher_targets"
       - "distillation_hybrid_targets"
       - "offline_distillation_start/completed"
       - "kalman_lr_enabled"
       - "training_started" (currently only log_training_start)
       - "training_complete" (currently only log_training_complete)
       - "vector_update"
       - "checkpoint_save/load"
       - "stability_tracking_enabled"
       - "topology_analysis_enabled"
       - "isomer_detection_enabled"

  4. Add structured logging for error events consistently:
     Before: self.logger.log_error("qdrant", f"Error: {e}")
     After:  self.logger.log_error("qdrant", "Qdrant operation failed",
                     error_type=type(e).__name__, message=str(e),
                     module="retrieval", context={"collection": self.collection_name})

Estimated effort: 3-5 hours


4.9 DEAD CODE REMOVAL
---------------------

Identified dead/unused code:

  1. _cosine_similarity_manual() — antigravity_engine.py:857-863
     Never called. Delete immediately.
     Effort: 0.5 hours

  2. Unused imports — check each file:
     grep -rn "^import " *.py | grep -v "test_" | grep -v "# "
     Look for: imports of modules/classes that are never referenced.
     Example: from collections import defaultdict (used in AntigravityEngine ✓)
     Example: from threading import Lock (used in AntigravityEngine ✓)

  3. benchmark_evolution.py (8846 lines) — smallest benchmark file
     Check if referenced by any test file or CI workflow.
     If test_benchmark_evolution.py doesn't exist, this is dead code.
     Action: Check test file existence and CI references.

  4. adapter_weights.benchmark-backup-*.pt files (3 files at root)
     These are benchmark artifact backups. Add to .gitignore.
     Not code — cleanup only.

  5. chelation_debug.jsonl (at root)
     Debug output file. Add to .gitignore.

  6. overnight_campaign_20260311-225408.log (at root)
     Campaign log file. Add to .gitignore.

  7. sweep_results.json (at root)
     Sweep output. Add to .gitignore.

  8. run_overnight_campaign.py (4438 lines)
     Check if used in CI or referenced anywhere. If not, dead code.

  9. run_weight_refinement_campaign.py (21738 lines)
     Large file — check if used in CI. If not, dead code.

  10. test_cpu_inference.py, test_sparse_cpu_inference.py, test_disk_llm_estimator.py,
      test_moe_reap.py, test_memory_compression.py, test_packed_graph.py,
      test_repo_graph_memory.py, test_integrated_repo_runtime.py,
      test_phase7_system_evaluation.py
      Small test files — check if they test actual modules or were abandoned.

  11. NUL file at root (shown in directory listing)
      This is likely a Windows artifact. Delete if real.

  12. Check for unused methods in config.py:
      - validate_max_depth(): Check if called anywhere
      - get_db_path(): Check if called anywhere
      - ensure_directories(): Check if called anywhere

Estimated effort: 3-5 hours


4.10 TECHNICAL DEBT BACKLOG
----------------------------

Priority: P0 (Critical — must fix before v2.0)

  [P0-01] AntigravityEngine god object (1621 lines)
    → Phase 3.2 decomposition. Blocks all other refactoring.
    Effort: 15-20 hours
    Risk: HIGH — breaking changes to core API

  [P0-02] Circular import risk during migration
    → When moving files to src/chelatedai/, training modules must not import engine.
    Effort: 5-10 hours (testing)

  [P0-03] .gitignore incomplete
    → Missing: *.pt, *.jsonl, *.log, *.json (output files), __pycache__/, build/
    Effort: 0.5 hours

Priority: P1 (High — should fix in v1.x)

  [P1-01] Magic number extraction
    → Create constants.py, replace hardcoded values in antigravity_engine.py
    Effort: 5-8 hours

  [P1-02] GITHUBCHELATEDAIrlm_reference/ cleanup
    → Delete incorrectly named directory, move rlm_reference/ to docs/
    Effort: 1 hour

  [P1-03] Print statements in config.py validation methods
    → Replace with get_logger().warning()
    Effort: 2-3 hours

  [P1-04] Inline HTML in dashboard_server.py
    → Extract to dashboard/html/index.html
    Effort: 4-6 hours

  [P1-05] Duplicate training loop code
    → Extract _run_training_loop() shared method
    Effort: 2-3 hours

  [P1-06] Generic except Exception in antigravity_engine.py (7 instances)
    → Replace with specific exception types
    Effort: 3-4 hours

Priority: P2 (Medium — fix when convenient)

  [P2-01] Type annotations across all modules
    → Add comprehensive typing
    Effort: 20-30 hours

  [P2-02] Docstring standardization to 80% coverage
    → Google-style docstrings on all public methods
    Effort: 10-15 hours

  [P2-03] Benchmark infrastructure consolidation
    → Shared BenchmarkRunner base class
    Effort: 15-20 hours

  [P2-04] Dead code removal
    → Remove _cosine_similarity_manual, unused imports, abandoned test files
    Effort: 3-5 hours

  [P2-05] Linting configuration enhancement
    → Comprehensive ruff + mypy/pyright setup
    Effort: 1-2 hours config + ongoing fixes

  [P2-06] Exception hierarchy
    → Create custom exception classes
    Effort: 5-8 hours

  [P2-07] Logging consistency
    → Eliminate all print() statements in non-CLI code
    Effort: 3-5 hours

Priority: P3 (Low — nice to have)

  [P3-01] Computational storage POC reorganization
    → Move under src/chelatedai/computational_storage/
    Effort: 3-5 hours

  [P3-02] Testing framework modernization
    → Consider migrating from unittest to pytest (long-term)
    Effort: 10-15 hours

  [P3-03] Package installation setup
    → pyproject.toml [project] dependencies, entry points
    Effort: 2-3 hours


================================================================================
EXECUTION ORDER & TIMELINE
================================================================================

Week 1-2: Foundation (P0 tasks)
  - [P0-03] .gitignore cleanup
  - [P0-02] GITHUBCHELATEDAIrlm_reference/ delete
  - rlm_reference/ → docs/rlm_reference/
  - Create src/chelatedai/ skeleton with __init__.py
  - Create subpackage stubs (all __init__.py files)

Week 3-4: God Object Decomposition (P0)
  - [P0-01] Extract IngestionPipeline from antigravity_engine.py
  - Extract RetrievalEngine from antigravity_engine.py
  - Extract SedimentationTrainer from antigravity_engine.py
  - Extract AdaptiveThresholdManager from antigravity_engine.py
  - Delete _cosine_similarity_manual() [P2-04]
  - Extract shared _run_training_loop() [P1-05]

Week 5-6: Module Migration (P0)
  - Move remaining files to appropriate packages
  - Create backward-compatibility shims at root
  - Update internal imports
  - Run test suite against new structure

Week 7-8: Code Quality (P1 tasks)
  - [P1-01] Magic number extraction → constants.py
  - [P1-03] Print → logging in config.py
  - [P1-04] Dashboard HTML extraction
  - [P1-06] Exception handling standardization
  - [P1-02] Dead code removal

Week 9-10: Type Safety (P2 tasks)
  - [P2-01] Type annotations (priority order: config → engine → adapter → others)
  - [P2-02] Docstring standardization
  - [P2-05] Linting configuration
  - [P2-06] Exception hierarchy

Week 11-12: Polish & Cleanup
  - [P2-03] Benchmark consolidation
  - [P2-07] Logging consistency
  - [P3-01] Computational storage reorganization
  - Full CI run on new structure
  - Documentation updates

Total estimated effort: 120-180 hours (6-9 weeks at 40h/week)
Parallelizable: Yes — multiple developers can work on different modules simultaneously.
