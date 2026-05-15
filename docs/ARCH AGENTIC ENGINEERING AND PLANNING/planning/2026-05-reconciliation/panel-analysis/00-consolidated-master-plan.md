# ChelatedAI — Consolidated Improvement Master Plan

**Produced:** 2026-04-04  
**Source:** 8 expert panel reports (685 raw findings), deduplicated and cross-referenced  
**Repository state:** Session 31 wrap / Phase 7 promoted baseline / 1082+ tests on main

---

## 1. Executive Summary

ChelatedAI is a well-tested, actively developed research prototype that has outgrown several early structural decisions. The core research is sound — the adaptive embedding correction concept is implemented, tested, and benchmarked across six datasets — but the repository now carries three categories of technical debt that require deliberate attention before the next growth phase.

**What is genuinely strong:**
- 1082+ passing tests with an honest CI matrix across Python 3.9–3.12
- Clean structured logging, checkpoint rollback, and `SafeTrainingContext`
- Well-documented research tracks with session-level provenance
- Phase 7 successfully promoted with measured quantitative baselines

**The three dominant problem clusters:**

**Cluster 1 — ML Correctness Gaps (most impactful).** Several training paths have substantiated correctness bugs: the `DimensionProjection` detaches its own computation graph (F-ML-003), making the projection layer permanently untrained; the `SedimentationInfoNCELoss` uses inter-document neighbors as hard negatives, actively pushing semantically similar documents apart (F-ML-065); the hierarchical sedimentation always uses MSE loss regardless of the configured loss function (F-ML-006); and the Kalman LR scheduler uses an inverted noise model (F-ML-062). These affect research validity, not just code quality.

**Cluster 2 — Infrastructure Fragility (most urgent for CI stability).** Three untracked `adapter_weights.benchmark-backup-*.pt` files confirm the `isolated_adapter_state()` cleanup is leaking on exception. The large sweep performs O(N²) JSON I/O that degrades progressively and may never complete. The CI torch cache miss downloads ~720 MB of PyTorch on every run.

**Cluster 3 — Architectural Debt (most important for future work).** `AntigravityEngine` is 1,621 lines managing 15+ distinct responsibilities. The flat 84-file root layout has no namespace isolation. The two retrieval subsystems (Qdrant and `DiskBackedRepoGraphMemory`) are diverging with no architecture decision record documenting which is the forward path.

**Most critical next steps (in order):**
1. Fix the `DimensionProjection` detach bug — projection is silently untrained
2. Fix the `SedimentationInfoNCELoss` false-negative contamination
3. Fix `isolated_adapter_state()` cleanup leak and add `.pt` backup pattern to `.gitignore`
4. Fix the O(N²) JSON write in `run_large_sweep.py`
5. Add 7 missing modules to `pyproject.toml py-modules`
6. Update `CLAUDE.md` and `CHANGELOG.md` — both are materially stale

---

## 2. Severity Dashboard

After deduplication across 685 raw findings from 8 panels:

| Severity | Count | Dominant Domains |
|----------|-------|-----------------|
| CRITICAL | 22 | ML Correctness (9), Performance (4), Testing (3), Documentation (3), Infrastructure (3) |
| HIGH | 87 | Code Quality (22), ML Correctness (18), Performance (20), Infrastructure (15), Testing (12) |
| MEDIUM | 96 | All domains, roughly even distribution |
| LOW | 81 | Documentation, DevOps, Performance micro-optimizations |
| **Total unique** | **286** | |

**Cross-panel overlap on highest-priority findings (same bug found by 2+ panels):**
- `mock_nvme.py` preloads to RAM: Panels 3, 7 (architecture contradiction for disk-first claim)
- 7 modules missing from `py-modules`: Panels 4, 8A
- `CHANGELOG.md` frozen: Panels 6, 8A
- No permissions block in CI: Panels 4, 8B
- Flat layout / no package structure: Panels 4, 7, 8A
- Adapter checkpoint contamination / backup leak: Panels 3, 4, 7
- O(N²) sweep JSON write: Panels 3 (V-01, M-01)
- Torch pip cache miss (~720 MB/run): Panels 4, 8B

---

## 3. Top 20 Most Critical Findings

| Rank | ID | Title | File | Severity | Effort |
|------|----|-------|------|----------|--------|
| 1 | F-ML-003 | `DimensionProjection` immediately detaches computation graph — projection never trains | `teacher_distillation.py:340` | CRITICAL | S |
| 2 | F-ML-065 | InfoNCE uses inter-document targets as hard negatives — pushes similar docs apart | `sedimentation_loss.py:42–56` | CRITICAL | M |
| 3 | F-ML-002 | DimensionProjection hidden-path init produces near-zero-squared output | `teacher_distillation.py:42–46` | CRITICAL | S |
| 4 | F-ML-001 | InfoNCE temperature=0.07 with small batches causes mode collapse | `sedimentation_loss.py:26,159` | CRITICAL | S |
| 5 | F-ML-021 | No experiment tracking — hyperparameters and loss curves not persisted | `sedimentation.py, trainer` | CRITICAL | L |
| 6 | F-ML-062 | Kalman filter gain formula is mathematically inverted (high variance → lower LR is backwards) | `kalman_lr_scheduler.py:84–92` | CRITICAL | M |
| 7 | V-01/M-01 | `run_large_sweep.py` O(N²) JSON read-modify-write: final writes touch ~735 KB per iteration | `run_large_sweep.py:126–134` | CRITICAL | S |
| 8 | V-02 | Sweep reuses single engine + Qdrant state across all 7,350 configs — results are not independent | `run_sweep.py`, `run_large_sweep.py` | CRITICAL | M |
| 9 | F-ML-006 | Hierarchical sedimentation always uses MSE loss, ignoring configured InfoNCE/hybrid | `sedimentation.py:134–135` | HIGH | S |
| 10 | OH-01 | No dependency lockfile — every CI run may silently resolve different package versions | `requirements.txt` | HIGH | S |
| 11 | V-03/OH-06 | `isolated_adapter_state()` backup files never cleaned on exception; 3 leak files confirmed | `benchmark_utils.py:125–137` | HIGH | S |
| 12 | JO-001 | `UnboundLocalError` risk: `final_loss` referenced outside `SafeTrainingContext` if `__enter__` raises | `antigravity_engine.py:1228` | HIGH | S |
| 13 | F-001 | No end-to-end integration test for core learning loop (ingest → sedate → verify improvement) | (no file — missing test) | CRITICAL | M |
| 14 | NC-01/EC-02 | 7 production modules missing from `pyproject.toml py-modules` — `pip install` silently breaks them | `pyproject.toml` | HIGH | XS |
| 15 | F-ML-040 | No held-out validation split in sedimentation training — direct data leakage from training targets | `sedimentation.py, trainer` | HIGH | M |
| 16 | F-007 | No gradient clipping in any training loop — InfoNCE at small temperature can cause weight explosion | `sedimentation.py:150`, `dimension_mask_predictor.py:182` | HIGH | S |
| 17 | Y-01 | Legacy block format pads every matrix to 512×512 — 99.5% storage waste still used in profiling | `block_graph.py:29–32` | CRITICAL | M |
| 18 | F-06 | `mock_nvme.py` preloads entire file into RAM — contradicts the disk-first program's core claim | `mock_nvme.py` | HIGH | M |
| 19 | OH-02 | GitHub Actions pinned by semver tag, not SHA — supply chain attack surface | `.github/workflows/*.yml` | HIGH | XS |
| 20 | F-02 | `CHANGELOG.md` frozen at 2026-01-06 — 105 PRs and 14 months of work undocumented | `CHANGELOG.md` | CRITICAL | M |

---

## 4. Quick Wins List (XS/S effort — complete in one session)

These 32 items are each under 2 hours of work individually. A focused 3-hour session could clear most of them.

### XS (< 30 minutes)

| ID | Action | File |
|----|--------|------|
| NC-01 | Add 7 missing modules to `py-modules` in `pyproject.toml` (`sedimentation_loss`, `kalman_lr_scheduler`, `isomer_detector`, `topology_analyzer`, `language_detector`, `cross_lingual_distillation`, `benchmark_beir`) | `pyproject.toml` |
| OH-06 | Add `adapter_weights.benchmark-backup-*.pt` to `.gitignore` | `.gitignore` |
| OH-08/EC-10 | `git rm nul` — remove tracked Windows NUL artifact | root |
| EC-07 | Add `experiment_runs/`, `db_scifact_evolution/`, `GITHUBCHELATEDAIrlm_reference/`, `.claude/`, `*.cspg` to `.gitignore` | `.gitignore` |
| F-13 | Fix `CLAUDE.md` path for `REFERENCES.md` — it is at root, not `docs/REFERENCES.md` | `CLAUDE.md` |
| LP-10/SM-09 | Add workflow concurrency cancel-in-progress block to both CI workflow files | `.github/workflows/*.yml` |
| OH-02 | Pin GitHub Actions to commit SHAs (or enable Dependabot for Actions updates) | `.github/workflows/*.yml` |
| F-11 | Update `CLAUDE.md` test count and stale date (currently "1082 as of 2026-03-12") | `CLAUDE.md` |
| MC-011 | Add a comment to `ChelationConfig.DEFAULT_TEACHER_MODEL` warning that same-model distillation is a no-op | `config.py:278` |
| F-ML-014 | Fix `KalmanLRScheduler` to use sample variance `np.var(..., ddof=1)` instead of population variance | `kalman_lr_scheduler.py:85` |
| D-15 | Add timestamp to sweep output filename to prevent overwrite (`sweep_results_{timestamp}.json`) | `run_sweep.py` |

### S (30 min – 2 hours)

| ID | Action | File |
|----|--------|------|
| F-ML-003 | Remove `.detach()` from `generate_distillation_targets` line 340 OR rename `project_tensor` to `project_numpy` to clarify graph-detached path | `teacher_distillation.py:340` |
| F-ML-006 | Consult `engine._sedimentation_loss_type` in `HierarchicalSedimentationEngine.train()` instead of hardcoding `MSELoss` | `sedimentation.py:134–135` |
| JO-001 | Initialize `final_loss`, `total_updates`, `failed_updates` before the `SafeTrainingContext` `with` block | `antigravity_engine.py:1228` |
| V-01/M-01 | Replace JSON read-modify-write in `run_large_sweep.py` with append-only JSONL writes | `run_large_sweep.py:126–134` |
| V-03 | Fix `isolated_adapter_state()` exception path to delete backup file in `finally` block | `benchmark_utils.py:125–137` |
| JO-019 | Fix `delete_checkpoint` to delete directory BEFORE removing metadata entry | `checkpoint_manager.py:198–238` |
| MC-018/MC-019 | Replace `print()` calls in `checkpoint_manager.py` with the project's structured logger | `checkpoint_manager.py:44,54,123,158,184,237,259` |
| RT-015 | Remove dead Python 2 `urlparse` fallback import in `_validate_url` | `recursive_decomposer.py:179–199` |
| JO-022/RT-003 | Replace falsy-check `or` defaults with explicit `is None` checks throughout `AntigravityEngine` constructor and `enable_*` methods | `antigravity_engine.py:603–606,715–719` |
| F-ML-001 | Add batch-size-aware temperature scaling to InfoNCE: `effective_temp = max(temperature, 1.0 / batch_size)` | `sedimentation_loss.py:26` |
| F-007 | Add `torch.nn.utils.clip_grad_norm_` (max_norm=1.0) to sedimentation and MaskPreTrainer training loops | `sedimentation.py:150`, `dimension_mask_predictor.py:182` |
| F-31 | Fix docstring header in `test_online_updater.py`, `test_dimension_mask_predictor.py`, `test_benchmark_comparative.py` — change `pytest` to `python -m unittest` | test files |
| F-ML-019 | Add `teacher_model_name != model_name` validation in `AntigravityEngine.__init__` with a `warnings.warn` when they match | `antigravity_engine.py` |
| RT-004/RT-005 | Fix `chelation_adapter.py` load() to use `try/except (FileNotFoundError, pickle.UnpicklingError, RuntimeError)` instead of `os.path.exists` + narrow `RuntimeError` catch | `chelation_adapter.py:79–86` |
| OH-03/B1 | Add `permissions: {contents: read}` block to both CI workflow files | `.github/workflows/*.yml` |
| LP-02 | Add `needs: [computational-storage-fundamentals]` to the emulation CI job | `.github/workflows/test.yml` |
| TE-03/SM-06/B8 | Add `timeout-minutes: 20` to all CI jobs | `.github/workflows/test.yml` |
| F-03/F-27 | Update test file headers to remove `pytest` references and fix `TestingAgent` class name to `QAAgent` | `test_aep_orchestrator.py` |

---

## 5. Phase 0: Critical Bugs (fix before any other work)

These bugs cause incorrect behavior, invalid research results, or silent data loss.

---

### P0-01: DimensionProjection permanently untrained
**Source:** F-ML-003  
**File:** `teacher_distillation.py:340`  
**Bug:** `generate_distillation_targets` calls `self._projection.project_tensor(teacher_tensor)` which returns a grad-carrying tensor, then immediately calls `.detach().numpy()`. The `project_tensor` docstring states it "keeps the projection in the computation graph so its parameters receive gradients" — but the detach defeats this. The `DimensionProjection` is forever an untrained identity approximation.  
**Fix:** If the projection is never intended to be trained through this path (per the Challenge C-03 clarification), rename `project_tensor` → `project_numpy` to signal detached semantics and remove the misleading docstring. If it was intended to be co-trained, remove the `.detach()` and add a separate optimizer for the projection.

---

### P0-02: InfoNCE false-negative contamination
**Source:** F-ML-065  
**File:** `sedimentation_loss.py:42–56`  
**Bug:** `SedimentationInfoNCELoss` uses all documents in the batch as negatives for each anchor. Documents that are semantically similar (legitimately near-neighbors) are treated as hard negatives, actively pushing them apart. This corrupts the semantics of the sedimentation correction.  
**Fix:** Either (a) filter same-cluster pairs from the negative set using the chelation log's cluster membership, or (b) switch to a margin-based loss that only penalizes pairs closer than the margin threshold.

---

### P0-03: InfoNCE temperature causes mode collapse at small batch sizes
**Source:** F-ML-001  
**File:** `sedimentation_loss.py:26,159`  
**Bug:** Default `temperature=0.07` was designed for large-batch (512–65K) contrastive learning. Sedimentation batches are typically 10–100 items. At temperature=0.07 with batch=10, softmax is near one-hot, so gradients are essentially zero except for the hardest pair in the batch. This causes training instability, not quality improvement.  
**Fix:** Add a `batch_size_aware` parameter (default True) that computes `effective_temperature = max(self.temperature, 1.0 / batch_size)`. Alternatively, document the recommended minimum batch size for the current default temperature.

---

### P0-04: Large sweep O(N²) JSON corruption risk
**Source:** V-01, M-01  
**File:** `run_large_sweep.py:126–134`  
**Bug:** On each of 7,350 iterations, the code reads the entire JSON results file, appends one entry, and writes back the whole file. By iteration 5,000 this is multiple seconds per write. At 7,350 iterations, total I/O is O(N²). Any interruption leaves a partial JSON write.  
**Fix:** Replace with append-only JSONL writes: `with open(results_file, 'a') as f: json.dump(entry, f); f.write('\n')`. Convert to full JSON on completion only.

---

### P0-05: Sweep engine/Qdrant state not reset between configurations
**Source:** V-02  
**File:** `run_sweep.py`, `run_large_sweep.py`  
**Bug:** Both sweeps reuse a single `base_engine` object across all configurations. The chelation log is cleared but Qdrant vector state, in-memory caches, and internal engine attributes persist. Benchmark results across configurations are NOT independent. This is a research validity issue.  
**Fix:** Create a fresh `AntigravityEngine` per sweep configuration using a fresh `qdrant_location=":memory:"`. The added cost per iteration is one ingestion pass; use the existing `isolated_adapter_state` pattern to share adapter weights.

---

### P0-06: UnboundLocalError in sedimentation training on SafeTrainingContext failure
**Source:** JO-001  
**File:** `antigravity_engine.py:1228`  
**Bug:** `final_loss`, `total_updates`, and `failed_updates` are referenced after the `SafeTrainingContext` block but are only assigned inside it. If `SafeTrainingContext.__enter__` raises (e.g., disk full when creating checkpoint), all three variables are unbound and a `NameError` fires, masking the original exception.  
**Fix:** Initialize all three variables to `None`/`0` before the `with` block: `final_loss = None; total_updates = 0; failed_updates = 0`.

---

### P0-07: CHANGELOG.md frozen — misrepresents project state
**Source:** F-02, EC-03  
**File:** `CHANGELOG.md`  
**Bug:** The last entry is 2026-01-06 covering Phase 1–3 only. 105 PRs and 14 months of work (sedimentation loss, Kalman LR, BoundedAdapter, disk-first Phase 1–7, BEIR evaluation, cross-lingual routing) are undocumented. The "Next Steps (Phase 4 - Pending)" section lists items that have been complete for 12+ months. This actively misleads any new reader about project state.  
**Fix:** Either adopt keep-a-changelog format with a retroactive v0.2.0 → v0.3.0 entry covering Sessions 7–31, or add a prominent banner: "This changelog was not maintained after 2026-01-06. See docs/ARCH AGENTIC ENGINEERING AND PLANNING/ for session-by-session history."

---

### P0-08: DimensionProjection near-zero-squared output for two-layer path
**Source:** F-ML-002  
**File:** `teacher_distillation.py:42–46`  
**Bug:** When `hidden_dim` is provided, two linear layers are initialized with `xavier_uniform_ * 0.001`. The composition of two near-zero matrices produces near-zero-squared output on the first forward pass, meaning the teacher provides effectively zero signal. No log entry flags this pathology.  
**Fix:** For the two-layer path, initialize only the final layer at near-zero (to avoid initial scale shock) while keeping the first layer at standard xavier initialization, or use a learnable scaling parameter initialized to 1.0.

---

### P0-09: No integration test for the core learning loop
**Source:** F-001 (Panel 05)  
**Bug:** The system's primary value claim — that sedimentation training improves retrieval quality — has zero end-to-end test coverage. Individual units are tested, but no test exercises `ingest → run_inference (baseline) → run_sedimentation_cycle → run_inference (post) → assert improvement`.  
**Fix:** Create `test_integration_core_learning.py` using an in-memory Qdrant engine with a tiny synthetic corpus (5–10 documents) where collapse is deliberately induced. Assert that post-sedimentation top-1 jaccard improves over baseline.

---

## 6. Phase 1–6 Roadmaps

### Phase 1: Quick Wins (1 coding session, ~3 hours)

See Section 4 for the full XS/S list. Priority order:
1. `.gitignore` updates (backup .pt, experiment dirs, nul, .claude)
2. `pyproject.toml` — add 7 missing modules
3. `CLAUDE.md` — fix REFERENCES.md path, update test count/date
4. Fix `isolated_adapter_state()` exception cleanup
5. Fix `UnboundLocalError` in sedimentation training
6. Add `timeout-minutes` and `permissions` block to CI workflows
7. Replace `print()` in `checkpoint_manager.py` with logger
8. Fix `run_large_sweep.py` O(N²) JSON write
9. Fix `delete_checkpoint` metadata-before-file-delete ordering
10. Fix `_validate_url` Python 2 dead code removal
11. Add `adapter_weights.benchmark-backup-*.pt` to `.gitignore`
12. Fix `test_*.py` headers that say `pytest` instead of `unittest`

---

### Phase 2: Sprint 1 — Research Validity (1–2 weeks)

Goal: Make benchmark results trustworthy and training numerically sound.

| ID | Item | File | Effort |
|----|------|------|--------|
| P0-02 | Fix InfoNCE false-negative contamination | `sedimentation_loss.py` | M |
| P0-03 | Add batch-size-aware temperature scaling | `sedimentation_loss.py` | S |
| F-ML-006 | Hierarchical sedimentation: respect configured loss type | `sedimentation.py:134` | S |
| F-ML-007 | Add gradient clipping to all training loops | `sedimentation.py`, `dimension_mask_predictor.py` | S |
| F-ML-040 | Add 20% held-out validation split to sedimentation training | `sedimentation.py` | M |
| F-ML-023 | Normalize adapter outputs before Qdrant upsert (training/serving parity) | `sedimentation.py:186–187` | S |
| F-ML-024 | Add lock around `load_teacher_model` lazy load in EnsembleTeacherHelper | `teacher_distillation.py:503–528` | S |
| F-ML-025 | Persist DimensionProjection layer alongside adapter checkpoint | `teacher_distillation.py:282–295` | M |
| F-ML-062 | Document Kalman LR as "gain-inspired" (not rigorous Kalman); reconsider sign convention | `kalman_lr_scheduler.py` | S |
| F-ML-036 | Replace Jaccard with rank-biased overlap (RBO) in isomer detection | `isomer_detector.py:53–72` | M |
| F-ML-069 | Fix EmbeddingQualityAssessor threshold direction (higher scale = less aggressive, wrong way) | `embedding_quality.py:89–107` | S |
| F-ML-041 | Deduplicate identical noise vectors in chelation log before training | `sedimentation.py:52–53` | S |
| V-02 | Fix sweep engine non-independence (fresh engine per config) | `run_sweep.py`, `run_large_sweep.py` | M |
| D-02 | Add p95/p99 and std deviation to all benchmark latency reporting | `benchmark_comparative.py` | S |
| F-09/F-010 | Add benchmark run to CI with NDCG assertion on synthetic dataset | `.github/workflows/test.yml` | M |

---

### Phase 3: Sprint 2 — Code Quality & Testing (2–3 weeks)

Goal: Close the most important correctness and test coverage gaps.

**Test coverage gaps (critical):**
| ID | Item | Effort |
|----|------|--------|
| F-001 | End-to-end learning loop integration test | M |
| F-004 | Regression test: `use_quantization=True` populates `chelation_log` | S |
| F-005/F-006 | Regression tests: Procrustes non-zero init, LowRank post-training movement | S |
| F-007 | Seed control for all numeric threshold assertions | S |
| F-015/F-016 | Engine-level tests for `enable_learned_masking`, `enable_stability_tracking` | S |
| F-032 | Test for same-model distillation no-op detection | S |
| F-024 | Tests for 9 untested `ChelationConfig` preset families | S |

**Code quality improvements:**
| ID | Item | File | Effort |
|----|------|------|--------|
| MC-007 | Extract duplicated training loop in `run_sedimentation_cycle` / `run_offline_distillation` into `_run_training_loop` helper | `antigravity_engine.py:1102–1228` | M |
| MC-020 | Fix `_retrieve_for_node`: replace ordinal ranks with real similarity scores | `recursive_decomposer.py:453` | S |
| JO-013 | Fix CUDA device mismatch: `torch.tensor(0.0, device=adapted_query.device)` | `online_updater.py:108` | S |
| PS-015 | Add `maxlen` cap to unbounded lists in `StabilityTracker` | `stability_tracker.py:30–44` | S |
| PS-016 | Fix `AntigravityEngine` ignoring `ChelationConfig.BOUNDED_ADAPTER_ENABLED` in constructor | `antigravity_engine.py:82–89` | S |
| MC-004 | Extract `get_structural_health_report` subsystem logic into `HealthProvider` interface | `antigravity_engine.py:836–854` | M |
| MC-006 | Split `run_sedimentation_cycle` into private methods for filtering, data preparation, training, sync | `antigravity_engine.py:923` | M |
| RT-013 | Fix `torch.cat(params)` to handle mixed-dtype params in `StabilityTracker` | `stability_tracker.py:93` | S |
| F-ML-055 | Replace full text as `LanguageDetector` cache key with MD5 hash | `language_detector.py:135` | S |
| PS-014 | Add sub-second collision guard to `create_checkpoint` timestamp-based IDs | `checkpoint_manager.py:92–93` | S |
| JO-017 | Count consecutive NaN epochs and stop training after N in `ConvergenceMonitor` | `convergence_monitor.py:54–59` | S |
| MC-019 | `CheckpointManager` should accept optional logger; stop using `print()` | `checkpoint_manager.py` | S |

---

### Phase 4: Sprint 3 — Infrastructure & DevOps (1 week)

| ID | Item | Effort |
|----|------|--------|
| OH-01 | Add `pip-compile` lockfile (`requirements-lock.txt`) updated by Dependabot | S |
| TE-01/SM-01 | Fix torch pip cache: add dedicated `actions/cache` step keyed on PyTorch version | S |
| B2/SM-02 | Add `coverage.py` reporting to CI; upload to Codecov; add badge to README | M |
| B5 | Add `pre-commit` config with ruff hook to prevent unlinted pushes | S |
| B6/OH-04 | Enable Dependabot for Python packages and GitHub Actions | XS |
| LP-03/TE-02 | Exclude computational-storage test files from the root unittest discovery to avoid triple execution | `test.yml` | S |
| VR-01 | Pin pico-sdk to specific release tag (e.g., `--branch 2.1.0`) instead of `--branch master` | `build_firmware.yml` | XS |
| LP-05 | Add `actions/cache` for pico-sdk clone and apt ARM toolchain | `build_firmware.yml` | S |
| B7/JL-04/JL-05 | Add GitHub issue and PR templates under `.github/` | XS |
| NC-02 | Reconcile `requirements.txt` vs `pyproject.toml` — move `requests`, `mteb`, `langdetect` to optional extras | M |
| JL-01/EC-A7 | Create minimal `CONTRIBUTING.md` covering setup, test run command, lint, PR format | S |
| SM-08/LP-04 | Add Windows runner to CI matrix (at least for core unit tests) | M |
| VR-04 | Embed git commit SHA in firmware binary via CMake CONFIGURE_FILE | M |
| NC-15 | Add `pip install -e .` step to CI to validate `py-modules` completeness | XS |

---

### Phase 5: Sprint 4 — Architecture Evolution (1–2 months)

These are structural improvements requiring explicit planning before execution.

**5A: Experiment tracking (HIGH impact)**  
Introduce a lightweight run registry (SQLite or append-only JSONL) recording: git commit SHA, full config snapshot, adapter checkpoint path, and result metrics per benchmark invocation. Even 150 lines of code would eliminate the "unauditable" problem. MLflow is standard but adds operational complexity; a minimal in-house logger is acceptable for research.  
*Files:* `run_sweep.py`, `benchmark_beir.py`, `benchmark_distillation.py`, new `experiment_registry.py`  
*Effort:* L

**5B: mock_nvme.py mmap replacement (HIGH impact for disk-first validity)**  
Replace the `self.flash_memory = file.read()` preload with `mmap.mmap` lazy access. Add latency simulation at the block granularity based on actual NVMe read latency profiles. This is the gate between "software correctness proof" and "honest disk-first performance claim."  
*File:* `computational_storage_poc/mock_nvme.py`  
*Effort:* M

**5C: AntigravityEngine God Object decomposition (HIGH architectural impact)**  
Extract the following collaborator classes, keeping the public `AntigravityEngine` interface stable as a facade:
- `EmbeddingService` — `embed()`, `ingest()`, `ingest_streaming()`  
- `SedimentationTrainer` — `run_sedimentation_cycle()`, `run_offline_distillation()`
- `DiagnosticsService` — `enable_stability_tracking()`, `enable_topology_analysis()`, `enable_isomer_detection()`, `get_structural_health_report()`
- `AdaptiveControlService` — threshold adaptation, Kalman LR, convergence monitoring  

*File:* `antigravity_engine.py` (1,621 lines)  
*Effort:* XL (do incrementally over multiple sessions)

**5D: ChelationConfig decomposition (MEDIUM impact)**  
Split the 974-line God Config into per-subsystem dataclasses (`SedimentationConfig`, `AdapterConfig`, `TeacherConfig`, `BenchmarkConfig`). Add a config version field and JSON serialization per subsystem. This enables reproducible experiment configs.  
*File:* `config.py`  
*Effort:* L

**5E: Architecture Decision Record for retrieval subsystem convergence**  
Write an ADR answering: (1) Does `DiskBackedRepoGraphMemory` replace Qdrant in the disk-first path? (2) Will `AntigravityEngine` ever use `DiskBackedRepoGraphMemory`? (3) Is the chelation adapter applicable to the disk-first inference path? Without this decision, work on either track may be wasted. Estimated time: 2 hours for the discussion, 1 hour to write.  
*Files:* new `docs/decisions/ADR-001-retrieval-subsystem.md`  
*Effort:* S (the decision meeting), M (implementing whichever direction is chosen)

**5F: computational_storage_poc package isolation**  
Add `computational_storage_poc/__init__.py` and convert all bare relative imports (`from block_graph import ...`) to relative imports (`from .block_graph import ...`). This resolves the CWD dependency bug (F-004 in Panel 07) and makes the PoC importable from the repo root.  
*Effort:* M

---

### Phase 6: Strategic Decisions (program-level)

These items require explicit decisions before work can begin. They are not implementation tasks.

| Decision | Current State | Options | Impact |
|----------|--------------|---------|--------|
| **Flat layout vs. package** | 84 files at root | (A) Stay flat, fix pyproject.toml sync with CI check. (B) Migrate to `chelatedai/` + `tests/` + `benchmarks/` layout over 2–3 sessions. | All import stability, test organization, distribution |
| **Experiment tracking tool** | None | (A) Minimal in-house JSONL registry. (B) MLflow (local server). (C) Weights & Biases free tier. | Research reproducibility, hyperparameter comparison |
| **CHANGELOG strategy** | Frozen since 2026-01-06 | (A) Retroactive summary entry covering Sessions 7–31, then keep-a-changelog going forward. (B) Formal deprecation pointing to session log archive. | Project discoverability, contributor trust |
| **Teacher model default** | `all-MiniLM-L6-v2` (same as student) | Change `DEFAULT_TEACHER_MODEL` to `all-mpnet-base-v2` (768-dim, different model). | Distillation correctness — same-model = zero signal |
| **Kalman LR keep/replace** | Inverted noise model but possibly works empirically | (A) Document as "gain-inspired" heuristic. (B) Replace with principled adaptive LR (e.g., AMSGrad, Adafactor). | Research integrity |
| **BoundedAdapter in production** | Config flag exists but engine constructor ignores it | Enable `BOUNDED_ADAPTER_ENABLED` path in engine constructor with test validation. | Feature completeness |
| **Version discipline** | Stuck at 0.1.0 through 105 PRs | (A) Bump to 0.2.0 now and adopt semver. (B) Explicitly document as pre-release research code. | Distribution clarity |
| **BEIR research-tier run** | Never executed — blocks preset refinement | Prerequisite: Phase 0 stability fixes and multi-hour run window. Gate Phase 5D config optimization on this. | Research validation completeness |

---

## 7. Domain Breakdown

### Code Quality (22 HIGH, 18 MEDIUM findings)
The dominant themes are: (1) the `AntigravityEngine` God Object with 30+ methods across 1,621 lines; (2) duplicated training loop code between `run_sedimentation_cycle` and `run_offline_distillation`; (3) `checkpoint_manager.py` bypassing the logging system entirely; (4) `embedding_backend.py` constructor making a live network call; (5) false-optional `None`-check patterns (`x or default` instead of `x is None`). The highest-urgency individual code quality fix is `MC-007` (extract duplicate training loop), which also fixes the loss-function propagation bug.

### ML Correctness (9 CRITICAL, 18 HIGH findings)
This is the most consequential domain. Five of the nine critical ML findings are bugs that produce silently incorrect training behavior: the projection detach (F-ML-003), the InfoNCE false-negative contamination (F-ML-065), the hierarchical sedimentation hardcoded loss (F-ML-006), the two-layer projection near-zero init (F-ML-002), and the missing gradient clipping (F-ML-007). The Kalman LR inverted model (F-ML-062) is mathematically incorrect but may still function as an empirical heuristic. The no-validation-split issue (F-ML-040) is a fundamental research methodology gap that affects all published benchmark numbers.

### Performance & Scale (4 CRITICAL, 20 HIGH findings)
The O(N²) sweep JSON write (V-01) and non-independent sweep configurations (V-02) are the most impactful. The `mock_nvme.py` preload contradiction (F-006, Panel 07) undermines the disk-first program's central claim. The `TopologyAnalyzer.build_bond_matrix` O(N²) full pairwise computation (F-ML-051) will OOM for any production corpus. The `DiskBackedRepoGraphMemory` full `nodes.jsonl` load into Python objects on every `__init__` (Y-02) is a scalability ceiling for the repo-memory feature.

### Infrastructure & DevOps (3 CRITICAL, 15 HIGH findings)
After challenge-phase adjustments: (1) no dependency lockfile — HIGH (downgraded from CRITICAL for research context); (2) GitHub Actions SHA pinning — MEDIUM (downgraded from CRITICAL, no CI secrets present); (3) 7 missing `py-modules` — HIGH; (4) torch double-install defeating pip cache — MEDIUM after challenge; (5) no workflow timeouts — MEDIUM. The pico-sdk `--branch master` reproducibility issue was downgraded to HIGH from CRITICAL for POC context.

### Testing (3 CRITICAL, 12 HIGH findings)
The absence of an end-to-end integration test for the core learning loop is the defining gap. The Session 29 bugs (chelation path skip, Procrustes init, low-rank double-suppression) each have incomplete regression coverage — the fixes could silently re-regress. The mock overuse in `test_antigravity_engine.py` means the engine's actual behavior under real computation is completely untested. `test_structural_health_report.py` uses `object.__new__` to bypass `__init__`, which is a structural test fragility.

### Documentation (3 CRITICAL, 10 HIGH findings)
The `CHANGELOG.md` is the most critical because it actively misrepresents project state to any reader. The wrong arXiv ID for the primary MRL citation (F-05) affects research credibility. `CLAUDE.md` omitting the entire 15-module disk-first substrate (F-04) will cause coding session agents to miss the new codebase surface. The `docs/INDEX.md` covering only Sessions 1–3 of 31+ completed sessions makes 28 session logs effectively undiscoverable.

### Architecture (2 CRITICAL, 8 HIGH findings after domain normalization)
The God Object (F-001) and flat layout (F-002) are structural foundations that constrain every other improvement. Neither is a quick fix, but both should be on the roadmap. The two diverging retrieval subsystems without an ADR (F-008) is the most urgent architectural decision: continued investment in either track without a documented decision risks wasted work.

---

## 8. Decision Log

Items requiring explicit program decisions before implementation work can begin.

| # | Decision Required | Urgency | Context |
|---|------------------|---------|---------|
| D-01 | **Retrieval subsystem convergence:** Will Qdrant and `DiskBackedRepoGraphMemory` converge, and if so which direction? | HIGH — blocks disk-first integration | Panel 07 F-008; two incompatible interfaces with no ADR |
| D-02 | **Package layout migration:** Flat root vs. `chelatedai/` package directory | MEDIUM — blocked by merge risk | Panel 07 F-002, Panel 08A EC-01; Devil's Advocate acknowledged research-context validity of flat layout |
| D-03 | **CHANGELOG approach:** Retroactive summary vs. formal deprecation pointing to session logs | MEDIUM — affects contributor trust | Panel 06 F-02, Panel 08A EC-03; changelog is 105 PRs stale |
| D-04 | **Default teacher model:** Change `DEFAULT_TEACHER_MODEL` from `all-MiniLM-L6-v2` (same as student) to `all-mpnet-base-v2` | HIGH — same-model = zero signal confirmed in Session 29 | Panel 02 F-ML-029, Panel 01 MC-011; behavior is a confirmed no-op |
| D-05 | **Experiment tracking tool selection:** In-house JSONL registry vs. MLflow vs. W&B | HIGH — blocks benchmark reproducibility | Panel 02 F-ML-021, Panel 07 F-005; no audit trail for any benchmark result |
| D-06 | **Kalman LR mathematical validity:** Keep as empirical heuristic or replace with principled adaptive LR | MEDIUM — affects research integrity claims | Panel 02 F-ML-062 vs F-ML-076; Devil's Advocate notes empirical validity may exist |
| D-07 | **InfoNCE asymmetry:** Current loss is asymmetric (L(output→target) only); document as intentional or add symmetric term | MEDIUM — affects theoretical MI bound | Panel 02 F-ML-061; asymmetric InfoNCE is theoretically valid for distillation |
| D-08 | **mock_nvme.py replacement timeline:** When does the disk-first claim require real mmap-backed lazy loading? | HIGH — architectural integrity gate | Panel 07 F-006; the 94.71% storage reduction metric is measured against dense format, not against actual disk reads |
| D-09 | **BEIR research-tier execution:** Gate and schedule the never-executed 7,350-configuration large sweep and BEIR research tier | HIGH — blocks weight preset optimization per roadmap | Panel 07 F-013, F-021; Phase 6 of weight refinement plan never ran |
| D-10 | **Version number discipline:** Bump to 0.2.0 and adopt semver, or explicitly mark as permanent pre-release | LOW — cosmetic but affects distribution signals | Panel 08A EC-04, Panel 08B NV-01 |

---

*End of consolidated master plan. This document supersedes individual panel reports for planning purposes; refer to panel reports for full finding detail and challenge/dissent rationale.*
