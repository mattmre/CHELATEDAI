# Changelog

All notable changes to ChelatedAI are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — live progress branch `feat/live-progress-tracker-20260606`

### Added

- **Liquified Lattice vision** — [docs/VISION_LIQUIFIED_LATTICE.md](docs/VISION_LIQUIFIED_LATTICE.md) names the north-star (self-annealing pools, shims, evidence DAG, disk scale) with claim boundaries and success metrics.
- **Phase II execution program** — [docs/ROADMAP_EXECUTION.md](docs/ROADMAP_EXECUTION.md) steps 9–17: Model-Scope close-out, SHIM DoD, annealing controller, evidence DAG, disintegration loop, drift experiment, GNN prototype, quant shim routing, disk pool slice.
- **Execution queue** — [docs/ROADMAP_EXECUTION.md](docs/ROADMAP_EXECUTION.md) Phase I defines core-first, single-track work (ML → infra → Model-Scope → E2E; SHIM deferred last).
- **Phase development loop** — `scripts/phase_development_loop.py` registers CORE-SLICE and SHIM-SLICE handlers, persists turn state under `artifacts/phase_loop/`, and emits `NEXT_AGENT_SLICE.json` for the next operator turn.
- **BHS 10-minute loop tooling** — `scripts/run_10min_priority_bhs_loop.py`, `scripts/loop_core_10m.sh`, `scripts/loop_10m.sh`, `scripts/chelated_loop_timer.py`, and `docs/loop_workers/` worker briefs.
- **SHIM evidence recorders** — `scripts/record_shim_inference_evidence.py`, `record_shim_prod_evidence.py`, `record_shim_tts_intercept_evidence.py`, `record_shim_engine_embed_evidence.py`, `record_shim_promoted_sip_evidence.py`, `record_shim_scheduler_evidence.py` write dated JSON under `artifacts/`.
- **SHIM research module** — `chelated_shim_research.py` with env-guarded `promoted_sip_apply()` and `promoted_registry_probe()`; promoted copies at `shim_node_promoted.py` and `shim_collapse_benchmark_extension_promoted.py`.
- **Five-worker shim gate** — `scripts/run_five_worker_shim_gate.py` for in-repo SHIM-CD-06 partial closure.
- **Shim verification script** — `scripts/verify_shim_development.sh` runs the full SHIM evidence + unittest gate.
- **Step runner** — `scripts/run_step_with_checks.sh` runs long primary commands with parallel companion checks (execution-queue policy).
- **AEP findings reports** — `reports/ARCH_AEP_REMEDIATION_FINDINGS*.md`, `reports/MERGE_READINESS_20260603.md`, and turn execution notes.
- **Tests** — broad SHIM suite (`tests/test_shim_*`), Model-Scope runtime/bridge coverage, `tests/test_learning_loop_e2e.py`, `tests/test_phase_development_loop_scheduler_handler.py`, `tests/test_run_10min_priority_bhs_loop.py`.

### Changed

- **Model-Scope stack** — `model_scope_runtime.py`, `model_scope_steering.py`, `model_scope_engine_bridge.py`, `qwen_scope_adapter.py`, and related tests hardened for pilot persistence and intervention caps.
- **Core engine seams** — `antigravity_engine.py`, `tts_pipeline.py`, `vector_store.py`, `steering_policy.py`, `self_healing_chelation.py` wired for optional SHIM preflight metadata (default OFF).
- **Infra / hygiene** — `benchmark_utils.py` adapter isolation, `sedimentation_loss.py` InfoNCE masking, `run_large_sweep.py` bounded persistence, `checkpoint_manager.py` and `.gitignore` backup patterns.
- **Operational docs** — `CLAUDE.md`, `docs/next-session.md`, and tracker pointer updated for operator reprioritization (core first, SHIM on hold).

### Findings (honest status)

- **Core queue steps 1–6 and 8** are implemented and regression-tested on this branch; step 7 (Model-Scope shadow pilot on real weights) remains fixture/integration-gated.
- **SHIM-CD-05 CLOSED** — production-path inference evidence under `CHELATED_SHIM_RESEARCH=1`.
- **SHIM-CD-01/02/06/08/09 OPEN (on hold)** — partial promoted SIP and registry probes exist; full substrate wiring and external scheduler proof remain deferred per execution queue.
- **SHIM-CD-03 OPEN** — MTP lookahead remains simulation-only (MockMTP).
- **SHIM-CD-07 OPEN** — BHS program score has not shown quantified lift on §77-83 metrics.
- **Phase loop** at turn 3171+ (2026-06-05): block flag `CLEAR`, next executable slice `SHIM-SLICE-SCHEDULER-06` (advisory while core queue active).

### Validation (2026-06-06)

- `python -m unittest discover -s tests -p "test_*.py" -q` — 107 tests OK
- `python -m unittest discover -s . -p "test_*.py" -q` — 2684 tests OK (10 skipped)
- `python scripts/check_block_flag.py` — PASS (CLEAR)

---

## 2026-06-05 — AEP Remediation Turn-9 Queue (continuation)

### Remaining Queue Track Completion

- Updated documentation truth surfaces (`CLAUDE.md`) to match active entrypoints and smoke paths (`python -m unittest tests.test_e2e_smoke`, `bash scripts/smoke.sh`, active `project.scripts`/`py-modules` set).
- Verified Model-Scope pilot persistence path with direct runtime/steering tests:
  - `tests/test_model_scope_runtime.py`
  - `test_model_scope_steering.py`
- Executed end-to-end learning-loop regression on fixture corpus (`tests/test_learning_loop_e2e.py`) covering ingest → sedimentation cycle → measurable metric/adaptation deltas.
- Kept SHIM rows open but on-hold per `docs/ROADMAP_EXECUTION.md`; no shim substrate resume until queue step 8 criteria are met and tracked.

### Validation Notes

- `python tests/test_model_scope_runtime.py` ✅
- `python test_model_scope_steering.py` ✅
- `python tests/test_learning_loop_e2e.py` ✅

## 2026-06-03 - AEP Remediation Turn-9 Track

### Core Remediation Sweep (completed in current branch)

- Fixed InfoNCE false-negative masking for duplicated sample IDs in `sedimentation_loss.py` (with regression tests).
- Hardened benchmark adapter isolation (`benchmark_utils.py`) for nested contexts and ensured checkpoint restores/cleanup on exceptions.
- Bound the large sweep persistence path (`run_large_sweep.py`) to avoid per-iteration full JSON rewrites.
- Added `run_large_sweep` to package `py-modules` in `pyproject.toml`.
- Updated operational guidance in `CLAUDE.md` and added dedicated sweep regression test (`test_run_large_sweep.py`).
- Added `test_benchmark_utils.py` coverage for `isolated_adapter_state`.

### Validation Notes

- `python -m unittest -q test_benchmark_utils.py` ✅ (28 tests)
- `python -m unittest -q test_run_large_sweep.py` ✅
- `python -m unittest -q test_sedimentation_loss.py` ✅ (36 tests)

---

## 2026-05-28 — SHIM unblock probe (on `main`)

- Sustained 10-minute zero-wall BHS loop scaffolding and first thin SIP probe design at `VectorSteerer` (PR #256 open on `feat/shim-cd01-unblock-first-probe-design`).
- `promoted_sip_apply()` partial wiring when both `CHELATED_SHIM_RESEARCH=1` and `CHELATED_SHIM_PROMOTED=1`.

## 2026-05-16 — BHS Scope B remediation wave (merged to `main`)

- PRs #249–#254 closed nine Carried Debt rows (CD-MOD-001 through CD-TTS-002).
- PR #248 closed CD-245-01 (BHS rubric entropy penalty).
- PR #247 wired `disk_llm_estimator` into dashboard; marked experimental comp-storage modules.
- PR #244 reconciliation foundation (BHS_OFFICIAL=55, operator override).

---

## 2026-01-06 - Phase 1, 2, 3 Complete

### Phase 1: Stabilization ✅

#### Critical Bug Fixes

**antigravity_engine.py**:
- Fixed duplicate `return` statement at line 160 (removed unreachable code)
- Added timeout protection (30s) to ThreadPoolExecutor for Ollama embeddings
- Added specific exception handling for Ollama connection errors
- Improved Ollama embedding retry logic with specific error messages
- Fixed bare `except` clause in sedimentation cycle batch updates
- Added `failed_updates` counter to track update failures
- Enhanced `_log_event()` with proper error handling and encoding

**benchmark_evolution.py**:
- Replaced hardcoded Windows path with cross-platform pathlib
- Integrated `ChelationConfig.get_db_path()` for portable database paths
- Fixed bare `except` clauses with specific exception types
- Added traceback printing for ingestion failures

**chelation_adapter.py** (user-modified):
- Already had dimension mismatch handling with try/except

#### Cross-Platform Support

**Created config.py**:
- `ChelationConfig` class with all hyperparameters centralized
- Cross-platform path handling using `pathlib.Path`
- Configuration presets (conservative/balanced/aggressive)
- Validation functions with clamping
- JSON config save/load functionality

### Phase 2: Robustness ✅

**checkpoint_manager.py** — backup/restore with SHA256 verification and `SafeTrainingContext`.

**chelation_logger.py** — structured JSON logging with specialized query/training/error methods.

### Phase 3: Observability ✅

**test_unit_core.py** — 21 unit tests for adapter, config, algorithms, and ID management.

### Documentation (2026-01-06)

- README.md, TECHNICAL_ANALYSIS.md, REFACTORING_PLAN.md initial pass.

### Statistics (2026-01-06)

- **Files Modified**: 3
- **Files Created**: 7
- **Tests Added**: 21 unit tests

### Breaking Changes

**None** — all changes backward compatible.

---

## Version History

### v0.2.0 - 2026-01-06
- Production-hardened with error recovery, cross-platform support, structured logging, configuration management.

### v0.1.0 - Initial prototype
- Core antigravity engine, chelation adapter, homeostatic learning proof-of-concept, MTEB benchmarks.