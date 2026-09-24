# Changelog

All notable changes to ChelatedAI are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — `49804ae3ddcebf4e4060fae95ba812331299f757` (`origin/main`). This heading is not a check that every sentence in this section was opened or re-run on that commit.

The old label `feat/live-progress-tracker-20260606` is PR #257, which is open and is not this commit.

- **2026-09-23 steps 1–3 re-check** — `docs/phase-i-steps-1-3-reverify-2026-09-23.md`. The distillation-target path uses `project_numpy` and does not train the projection. InfoNCE still treats other in-batch targets as negatives, so step 2 is not closed. `test_isolated_adapter_state.py` locks checkpoint restore. `.github/workflows/test.yml` uses `actions/setup-python` `cache: pip` and then installs the CPU torch wheel; there is no separate torch cache key.

### Added

- **Liquified Lattice vision** — [docs/VISION_LIQUIFIED_LATTICE.md](docs/VISION_LIQUIFIED_LATTICE.md) names the north-star (self-annealing pools, shims, evidence DAG, disk scale) with claim boundaries and success metrics.
- **Phase II execution program** — [docs/ROADMAP_EXECUTION.md](docs/ROADMAP_EXECUTION.md) steps 9–17. **Already on main, for merge commits ancestor-checked in [docs/status-corrections-2026-09-23.md](docs/status-corrections-2026-09-23.md):** #260, #277, #280, #284, #285, #286, #289, #290, and #291. Harness PRs #258–#276 were not re-checked in this pass. **On main via merged PR #292 (`c148f7b3c1e3858b461f6c08ddb3e58cd7e99a85`):** the H5 living-bank verdict (fail-closed negative; do not promote). Rung 13 post-bank prune/re-anneal remains on main, and detector-to-DAG prune is on main via merged PR #293 (`782ab62ddbaf6ab6c40085255e48ff9a21562d34`). Rung 17 is on main via merged PR #294 (`8e6e83b7c30bae34015ad12996314b5eea2d1c64`). Rung 15 is OPEN, not done, and not refused (no merged GNN-prototype PR in `gh pr list --state merged --search "GNN prototype"`). Rung 16 is not on this commit (PR #295 is open and not merged). The ROADMAP Status column is the source of truth for the table. The corrected rung-17 cell is in this change and is not on `49804ae3ddcebf4e4060fae95ba812331299f757` until the change merges.
- **Execution queue** — [docs/ROADMAP_EXECUTION.md](docs/ROADMAP_EXECUTION.md) Phase I defines core-first, single-track work (ML → infra → Model-Scope → E2E; SHIM deferred last).
- **Phase development loop** — `scripts/phase_development_loop.py` is not in this commit. This bullet does not create it.
- **BHS 10-minute loop tooling** — `scripts/run_10min_priority_bhs_loop.py`, `scripts/loop_core_10m.sh`, `scripts/loop_10m.sh`, `scripts/chelated_loop_timer.py`, and `docs/loop_workers/` are not in this commit.
- **SHIM evidence recorders** — `scripts/record_shim_inference_evidence.py`, `scripts/record_shim_prod_evidence.py`, `scripts/record_shim_tts_intercept_evidence.py`, `scripts/record_shim_engine_embed_evidence.py`, `scripts/record_shim_promoted_sip_evidence.py`, and `scripts/record_shim_scheduler_evidence.py` are not in this commit.
- **SHIM research module** — `chelated_shim_research.py`, `shim_node_promoted.py`, and `shim_collapse_benchmark_extension_promoted.py` are not in this commit.
- **Five-worker shim gate** — `scripts/run_five_worker_shim_gate.py` is not in this commit.
- **Shim verification script** — `scripts/verify_shim_development.sh` is not in this commit.
- **Step runner** — `scripts/run_step_with_checks.sh` is not in this commit.
- **AEP findings reports** — `reports/` is not in this commit (no `reports/ARCH_AEP_REMEDIATION_FINDINGS*.md` and no `reports/MERGE_READINESS_20260603.md`).
- **Tests** — `tests/test_shim_*` has no matches, and `tests/test_phase_development_loop_scheduler_handler.py` and `tests/test_run_10min_priority_bhs_loop.py` are not in this commit. `tests/test_learning_loop_e2e.py` is added by this change. It is a floor-tier dim-8 fixture. It is not MiniLM and not BEIR. It fails when the chelated ranked-id list does not change. Phase I is not complete. Steps 1–4 were not re-verified. `test_model_scope_runtime.py` and `test_model_scope_steering.py` are present at the repository root and were not re-run in this PR. `tests/test_model_scope_runtime.py` is not in this commit.

### Changed

- **Model-Scope stack** — `model_scope_runtime.py`, `model_scope_steering.py`, `model_scope_engine_bridge.py`, `qwen_scope_adapter.py`, and related tests hardened for pilot persistence and intervention caps.
- **Core engine seams** — `antigravity_engine.py`, `tts_pipeline.py`, `vector_store.py`, `steering_policy.py`, and `self_healing_chelation.py` are in this commit and do not contain `shim` or `SHIM`. The previous "optional SHIM preflight metadata" sentence is ahead of this tree.
- **Infra / hygiene** — `benchmark_utils.py` adapter isolation, `sedimentation_loss.py` InfoNCE masking, `checkpoint_manager.py` and `.gitignore` backup patterns. `run_large_sweep.py` appends each result as one JSONL line and writes the JSON array once at the end (`sweep_result_store.py`). It no longer calls `json.load` on the result array per iteration. `run_large_sweep` and `sweep_result_store` are in `pyproject.toml` `py-modules`. The CI torch install is documented in `docs/phase-i-steps-1-3-reverify-2026-09-23.md` (`cache: pip` plus the CPU wheel URL; no separate torch cache key).
- **Operational docs** — `CLAUDE.md`, `docs/next-session.md`, and tracker pointer updated for operator reprioritization (core first, SHIM on hold).
- **README** — frames Liquified Lattice as the primary research path with Phase I/II status; preserves road-course, gates, storage, and baseline findings as queued work on the books.

### Findings (honest status)

- **Core queue steps 1–6 and 8** are not all shown on this commit. Step 8's test `tests/test_learning_loop_e2e.py` is added by this change. It is a floor-tier dim-8 fixture. It is not MiniLM and not BEIR. It fails when the chelated ranked-id list does not change. Phase I is not complete. Steps 1–4 were not re-verified in this PR and are not declared done or not done. Step 7 was not re-verified in this PR. The per-iteration JSON rewrite is replaced by a JSONL append plus one array write at the end of the sweep. `run_large_sweep` and `sweep_result_store` are in `pyproject.toml` `py-modules`. The CI torch install is documented in `docs/phase-i-steps-1-3-reverify-2026-09-23.md` (`cache: pip` plus the CPU wheel URL; no separate torch cache key). This bullet does not say Phase I is complete.
- **SHIM-CD-05 CLOSED** — this changelog already marks that id CLOSED. `chelated_shim_research.py` is not in this commit. This edit does not re-open or re-close the id.
- **SHIM-CD-01/02/06/08/09** — these five ids stay not-closed here. `chelated_shim_research.py` and `shim_node_promoted.py` are not in this commit, so this bullet does not say promoted SIP or registry probes exist on this tree.
- **SHIM-CD-03** — not re-verified on this commit (`49804ae3ddcebf4e4060fae95ba812331299f757`). This note does not close the id and does not add a debt row.
- **SHIM-CD-07** — not re-verified on this commit (`49804ae3ddcebf4e4060fae95ba812331299f757`). This note does not close the id and does not add a debt row.
- **Carried-debt count:** those seven historical ids (SHIM-CD-01, SHIM-CD-02, SHIM-CD-06, SHIM-CD-08, SHIM-CD-09, SHIM-CD-03, SHIM-CD-07) are not rows in the `docs/next-session.md` Carried Debt table. `python3 scripts/check_block_flag.py` reports CLEAR and zero OPEN rows, so it does not count them. SHIM-CD-05 is the CLOSED id in this list. SHIM-CD-04 does not occur. This note does not close the seven ids and does not add debt rows.
- **Phase loop** — `scripts/phase_development_loop.py` is not in this commit, so the previous turn-3171+ / `SHIM-SLICE-SCHEDULER-06` note was not re-verified here. Block flag on this commit is `CLEAR` (`python3 scripts/check_block_flag.py`, zero OPEN rows).
- **H5 living-bank VERDICT (closed negative, 2026-07):** on the query-encoder-swap arena the living/annealed post-bank (C5) fails its preregistered gate (C5 must beat **both** C5s frozen-static and C5r one-shot router). SciFact: C5 == C5s (0.131135, bit-identical) and C5r one-shot 0.180862 beats living. NFCorpus: C5 == C5s (0.046389), living edges C5r (0.045817) but does not clear the dual gate. **LIVING BANK WINS = False on both datasets** — the living/annealed lifecycle adds nothing over a frozen static bank; a one-shot router is competitive-or-better. Do not promote the living bank. Sources: [docs/drift-recovery-post-bank-headtohead-results-2026-06.md](docs/drift-recovery-post-bank-headtohead-results-2026-06.md), [docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md](docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md).
- **H4 compound-cycles ablation (single-seed):** SciFact C4a seed 42 — `compound_cycles=False` Final NDCG@10 0.236297 vs `True` 0.005258 (~45× collapse toward the frozen floor). Compounding is a rejected design, not a recovery path; the idempotent one-shot fixed point is correct. Source: [docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md](docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md).
- **H2 swap re-run (post-H1-fix):** SciFact + NFCorpus query-encoder-swap campaigns re-run on the H1-fixed line; the committed manifests show C3a seed-42 baseline equals C0/C2/C2O/C4a (SciFact `0.7934593935363062`, NFCorpus `0.5550044219616403`; contamination was on NFCorpus; SciFact baselines already matched). Closes CD-H1-01 from those manifests. CD-A2-01 close rests on `config.swap_model` `all-mpnet-base-v2` plus run records with measured `swap_dim` 768. The 2026-09-22 rebase did not re-execute the GPU campaign. Sources: [docs/drift-recovery-swap-results-2026-06.md](docs/drift-recovery-swap-results-2026-06.md), [docs/drift-recovery-swap-nfcorpus-results-2026-06.md](docs/drift-recovery-swap-nfcorpus-results-2026-06.md).
- **Lattice rungs 10–14 apparatus:** SHIM/promotable-route substrate #284/#285/#286 (+ DoD correction #289, diagnostics observation-only); annealing controller #260 + post-bank temperature schedule #280; Evidence DAG schema #277; disintegration as post-bank prune/re-anneal #279 (+ builder/runtime/conditions #281/#282/#283); H3 #287/#288, H4 `compound_cycles` knob #291, and H5 head-to-head driver #290 are in the ancestor list below. Concept-drift harness #258–#266 and track-0 hygiene #267 were not re-checked in this pass. Ancestor-checked on this commit: #260 `f0c643ae7bf5a7c7ca612ceaa878b7cb15412540`, #277 `81e3614bec7d5e5b9cce57af8e7126c6b97775dc`, #279 `06f4f5cced82fb37547b1a7a36ee34f0dc9451f6`, #280 `9db098ea0d2f7df91b54d29e5ee1c6eb9ea65bb8`, #281 `65d3bbf1f3898ae5ef408570fce4bf688b13113b`, #282 `cc580302344f29149dd922447753b1d9701c4bb9`, #283 `7e8443f924593ba3afbef720fd3877f14715c1ab`, #284 `efe6d1be55b5c702eb3a1cccde3169feb04b6c34`, #285 `49b36046b6d59a554633f631924005130a05cd1f`, #286 `d8181f64161522eb39f8b0e1fd9e48aa0fa536b2`, #287 `44ded13eafb2b29805ac8fd9743452f14e4b51d1`, #288 `2ee132573b6b5ecc936be342000624b65b5e923f`, #289 `d1bfc0903d7de5c2b81d41a7d081e7da8502ad40`, #290 `6d3af3009c29b31ce72244c13b53c02170bfbd61`, and #291 `34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2`. Harness PRs #258–#276 were not re-checked in this pass. None of the commits listed in this bullet were re-executed. Rung **13** detector-to-DAG prune is on main via merged PR #293 (`782ab62ddbaf6ab6c40085255e48ff9a21562d34`); post-bank prune/re-anneal #279 (`06f4f5cced82fb37547b1a7a36ee34f0dc9451f6`) is also an ancestor. The H5 living-bank verdict is on main via merged PR #292 (`c148f7b3c1e3858b461f6c08ddb3e58cd7e99a85`). This bullet does not mark historical SHIM-CD rows CLOSED.
- **Placement on this commit (`49804ae3ddcebf4e4060fae95ba812331299f757`):** rung 13 detector-to-DAG prune is on main (merged PR #293, `782ab62ddbaf6ab6c40085255e48ff9a21562d34`). Rung 17 disk pool is on main (merged PR #294, `8e6e83b7c30bae34015ad12996314b5eea2d1c64`). Rung 15 is OPEN, not done, and not refused. Rung 16 is not on this commit (PR #295 is open and not merged). Do not call rung 15 done. Do not treat a rung 16 campaign result as on main.

### Validation (2026-06-06)

These lines are the 2026-06-06 log. They were not re-run on this commit (`49804ae3ddcebf4e4060fae95ba812331299f757`).

- `python -m unittest discover -s tests -p "test_*.py" -q` — 107 tests OK
- `python -m unittest discover -s . -p "test_*.py" -q` — 2684 tests OK (10 skipped)
- `python scripts/check_block_flag.py` — PASS (CLEAR)

---

## 2026-06-05 — AEP Remediation Turn-9 Queue (continuation)

### Remaining Queue Track Completion

- Updated documentation truth surfaces (`CLAUDE.md`) to match active entrypoints and smoke paths (`python -m unittest tests.test_e2e_smoke`, `bash scripts/smoke.sh`, active `project.scripts`/`py-modules` set).
- Model-Scope pilot persistence was not re-run in this PR:
  - `tests/test_model_scope_runtime.py` is not in this commit. `test_model_scope_runtime.py` is at the repository root and was not re-run.
  - `test_model_scope_steering.py` is present and was not re-run.
- `tests/test_learning_loop_e2e.py` is added by this change. It is a floor-tier dim-8 fixture. It is not MiniLM and not BEIR. It fails when the chelated ranked-id list does not change. Phase I is not complete. Steps 1–4 were not re-verified. The previous end-to-end learning-loop regression line is not a production retrieval result.
- The seven historical SHIM-CD names are not rows in `docs/next-session.md`. This note does not close them and does not add debt rows.

### Validation Notes

- `python tests/test_model_scope_runtime.py` — that path is not in this commit. `test_model_scope_runtime.py` is at the repository root and was not re-run.
- `python test_model_scope_steering.py` — the file is present and was not re-run.
- `python tests/test_learning_loop_e2e.py` — `tests/test_learning_loop_e2e.py` is added by this change. It is a floor-tier dim-8 fixture. It is not MiniLM and not BEIR. It fails when the chelated ranked-id list does not change. Phase I is not complete. Steps 1–4 were not re-verified. This note does not record a production retrieval result.

## 2026-06-03 - AEP Remediation Turn-9 Track

### Core Remediation Sweep

- Fixed InfoNCE false-negative masking for duplicated sample IDs in `sedimentation_loss.py` (with regression tests).
- Hardened benchmark adapter isolation (`benchmark_utils.py`) for nested contexts and ensured checkpoint restores/cleanup on exceptions.
- `run_large_sweep.py` appends each result to `{prefix}_results.jsonl` and writes `{prefix}_results.json` once at the end. The per-iteration `json.load` / `json.dump` of the whole array is gone. The CI torch install is documented in `docs/phase-i-steps-1-3-reverify-2026-09-23.md` (`cache: pip` plus the CPU wheel URL; no separate torch cache key).
- `run_large_sweep` and `sweep_result_store` are listed in `pyproject.toml` `py-modules`. An editable install with `--no-deps` can import `sweep_result_store`. Importing `run_large_sweep` in that environment stops at the first missing runtime dependency (`numpy` via `benchmark_evolution.py`). The CI torch install is documented in `docs/phase-i-steps-1-3-reverify-2026-09-23.md` (`cache: pip` plus the CPU wheel URL; no separate torch cache key).
- `test_sweep_result_store.py` covers the JSONL append, the one-time array migration, and a bad JSONL line. `test_run_large_sweep.py` is still not in this commit. This change does not edit `CLAUDE.md`.
- Added `test_benchmark_utils.py` coverage for `isolated_adapter_state`.

### Validation Notes

- `python -m unittest -q test_benchmark_utils.py` ✅ (28 tests)
- `python -m unittest -q test_run_large_sweep.py` — not runnable on this commit; the file is absent. `python -m unittest test_sweep_result_store.py` is the persistence regression added with the JSONL path.
- `python -m unittest -q test_sedimentation_loss.py` ✅ (36 tests)

---

## 2026-05-28 — SHIM unblock probe (on `main`)

- Sustained 10-minute zero-wall BHS loop scaffolding and first thin SIP probe design at `VectorSteerer` (PR #256 open on `feat/shim-cd01-unblock-first-probe-design`).
- `promoted_sip_apply()` does not occur in `*.py` on this commit. The 2026-05-28 wiring sentence is not code in this tree.

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