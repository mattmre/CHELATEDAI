# Panel of Experts Report 07: Architecture and Planning Review

**Date:** 2026-04-04
**Panel:** Architecture and Planning
**Repository:** ChelatedAI — adaptive vector search with self-correcting embeddings, disk-first CPU inference PoC
**Cycle phase:** Post Session-31 / Post Phase-7 promotion

---

## Panel Members

| Expert | Role | Lens |
|---|---|---|
| Dr. Isabelle Fontaine | Principal Architect (20 years) | System coherence, coupling, abstraction quality, scalability ceiling, architectural debt |
| Raj Krishnamurthy | Staff Engineer, Distributed Systems (15 years) | Fault tolerance, data flow, component boundaries, integration patterns, evolution paths |
| Dr. Nadia Al-Amin | Research Architect (13 years) | Research-to-production gap, prototype quality, architectural decisions that open vs close future options |
| Michael Okonkwo | Senior Architect (12 years) | Module dependencies, circular import risks, package structure, monolith vs service tradeoffs |
| Sophia Wren | Technical Program Manager (14 years) | Roadmap coherence, plan completeness, risk identification, decision dependencies, milestone clarity |
| Devil's Advocate | Contrarian Expert | Challenges every consensus finding |

---

## CONVENE — Panel Mandate

The panel was convened to find every architectural weakness, planning gap, roadmap risk, and strategic improvement opportunity in ChelatedAI at its current inflection point: the system has crossed from a pure retrieval-research prototype into a dual-track repository that simultaneously pursues adaptive embedding correction and a disk-first CPU inference program. The system has 84 Python files at root (no package structure), 1,082 tests, and an extensive but complex documentation surface spanning eight research tracks, a bespoke ARCH-AEP agentic workflow, and an active multi-phase infrastructure program.

---

## Executive Summary — Top 10 Architectural Findings

**F-001 [CRITICAL] AntigravityEngine is a God Object with 30+ methods across 1,621 lines.** It owns embedding, retrieval, training, teacher distillation, online updating, topology analysis, isomer detection, stability tracking, quantization, checkpoint management, and threshold adaptation all in one class. This makes it untestable in isolation, impossible to evolve independently, and a bottleneck for any future concurrency or distributed deployment.

**F-002 [CRITICAL] Flat file layout with 84 Python source files at root has no namespace isolation.** There are no packages, no `__init__.py`, and no submodule boundaries. Every import is a global name collision risk, circular import is undetectable until runtime, and the layout has no ceiling — it will worsen with every new module added.

**F-003 [HIGH] ChelationConfig is a God Config with 18 preset families, 974 lines, and no versioning.** Every subsystem reads config directly, there is no config schema validation, no migration path, and config incompatibility between saved model artifacts and new code is invisible.

**F-004 [HIGH] The computational_storage_poc sub-directory uses bare relative imports at runtime** (`from block_graph import BLOCK_SIZE`, `from cpu_backends import ...`). This only works if the CWD is `computational_storage_poc/` at execution time, creating a silent environment dependency.

**F-005 [HIGH] No experiment tracking or artifact registry exists.** Benchmark scripts write JSON files with no versioning, no cross-run comparison schema, and no connection between run configs and stored model weights. The "weight refinement campaign" was designed as a long batch workflow with no way to resume, correlate, or roll back.

**F-006 [HIGH] The disk-first CPU program has a 7-phase roadmap that is substantially complete on paper but the mock_nvme.py still preloads the entire file into memory.** The core claim of the program — disk-resident inference — is contradicted by the implementation of its own storage layer.

**F-007 [HIGH] The AEP orchestrator is a meta-layer built on top of a research prototype.** The `aep_orchestrator.py` implements a production-grade 7-phase remediation workflow (FindingStatus, EffortSize, priority scoring, concurrency protocol) that dwarfs the complexity it is meant to manage. The tooling complexity has grown to match the codebase complexity rather than reducing it.

**F-008 [MEDIUM] The retrieval track and the disk-first CPU track are architecturally diverging** with no integration plan. AntigravityEngine uses Qdrant for vector storage; the computational storage PoC uses its own DiskBackedRepoGraphMemory. These two retrieval subsystems have no shared interface, no migration path, and no documented decision about which one is the forward architecture.

**F-009 [MEDIUM] Feature flags are booleans in ChelationConfig class attributes, not a runtime feature-flag system.** Changing a feature flag requires editing source code or class-attribute override. There is no environment variable override, no per-experiment config, and no audit trail of which flags were active during a specific benchmark run.

**F-010 [MEDIUM] The revised roadmap document is marked "superseded" for the old audit but the phase-planning.md template is empty.** The AEP planning loop requires a populated phase-planning.md as the live planning record, but it contains only template placeholders. The current program has no active planning document pointing to current phase, current phase owner, or active risks.

---

## Full Findings List

### CRITICAL Severity

**F-001: AntigravityEngine God Object**
- File: `antigravity_engine.py` (1,621 lines, 30+ methods)
- The engine owns: embedding (embed, ingest, ingest_streaming), retrieval (_gravity_sensor, _chelate_toxicity, _spectral_chelation_ranking, run_inference), sedimentation training (run_sedimentation_cycle, run_offline_distillation), teacher distillation (teacher_helper, enable_teacher_weight_scheduling), online updates (enable_online_updates), structural diagnostics (enable_stability_tracking, enable_topology_analysis, enable_isomer_detection, get_structural_health_report), threshold adaptation (enable_adaptive_threshold, _update_adaptive_threshold), loss function management (set_sedimentation_loss), learning rate scheduling (enable_kalman_lr), convergence detection (enable_convergence_detection), temperature scaling (set_temperature), checkpoint management (checkpoint_manager), and vector store management.
- Impact: Any change to any one of these concerns requires reading and reasoning about 1,621 lines of interleaved logic. Unit testing is only possible by mocking away the entire engine. Concurrency is locked to per-engine single-thread access. Future specialization (e.g., read-only inference engine vs training engine) is impossible without major surgery.
- Recommended fix: Define 5-6 role-specific collaborator classes (EmbeddingService, RetrievalService, SedimentationTrainer, DiagnosticsService, AdaptiveControlService) with a thin AntigravityEngine facade delegating to them. This can be done incrementally by moving method groups out while keeping the public interface stable.

**F-002: Flat File Layout — 84 Python Source Files at Root**
- Files: all `*.py` at `D:/GITHUB/CHELATEDAI/` root
- The repository has 84 Python source files at root with zero package structure. Python's import system finds modules by name, so any new file named like an existing stdlib or third-party module will silently shadow it. There is no namespace isolation between the retrieval core, the benchmark harness, the research scripts, and the AEP orchestration layer.
- Naming collision risk: `sedimentation.py` and `sedimentation_trainer.py` exist at the same level. A new contributor adding `encoder.py`, `embedding.py`, or `model.py` would shadow well-known names.
- Discovery for testing is done with glob `test_*.py` at root, which means all 40+ test files are co-mingled with the 44 source files.
- Recommended fix: Introduce a `chelatedai/` package directory, separate `tests/`, `benchmarks/`, and `scripts/` directories, and a `computational_storage_poc/` package with its own `__init__.py`. The pyproject.toml `py-modules` list already enumerates all source files, showing the maintainer is aware of the problem but deferred it.

**F-003: ChelationConfig God Config — 18 Preset Families, 974 Lines**
- File: `config.py` (974 lines)
- The config class contains 18 named preset families: `CHELATION_PRESETS`, `CONVERGENCE_PRESETS`, `ADAPTER_TYPE_PRESETS`, `BOUNDED_ADAPTER_PRESETS`, `TEACHER_ENCODING_PRESETS`, `ADAPTER_PRESETS`, `RLM_PRESETS`, `ENSEMBLE_PRESETS`, `SEDIMENTATION_PRESETS`, `SEDIMENTATION_TUNED_PRESETS`, `BEIR_PRESETS`, `CROSS_LINGUAL_PRESETS`, `TEACHER_WEIGHT_SCHEDULE_PRESETS`, `ONLINE_UPDATE_PRESETS`, `TOPOLOGY_PRESETS`, `ISOMER_PRESETS`, `SEDIMENTATION_LOSS_PRESETS`, `KALMAN_LR_PRESETS`.
- There is no config schema (no dataclass or pydantic model per subsystem), no versioning (saved configs at one version of the code cannot be detected as stale), no environment variable override path, and no separation between "constants that never change" and "hyperparameters that get tuned per experiment."
- Impact: Benchmark runs cannot be fully reproduced without the exact source code version because config defaults are baked into class attributes. Any saved `adapter_weights.pt` file is silently tied to whatever config was active at training time.
- Recommended fix: Split config into per-subsystem dataclasses (e.g., `SedimentationConfig`, `AdapterConfig`, `TeacherConfig`), add a config version field, add JSON/YAML serialization per subsystem, and add environment variable overrides for the most commonly varied parameters.

**F-004: computational_storage_poc Bare Relative Imports — CWD Dependency**
- Files: `computational_storage_poc/integrated_repo_runtime.py`, `moe_reap.py`, `phase7_system_evaluation.py`, and all benchmark files
- These files import with bare names: `from block_graph import BLOCK_SIZE`, `from cpu_backends import NumpyInt8DynamicBackend`, `from packed_graph import ...`. This only works if Python's sys.path contains the `computational_storage_poc/` directory, which only happens if the user `cd`s into that directory or if the test runner adds it explicitly.
- The root-level test files (`test_integrated_repo_runtime.py`, `test_moe_reap.py`, etc.) work because the CI runner adds the correct path via `python -m unittest discover`. But any direct invocation from the repo root will fail with `ModuleNotFoundError`.
- Recommended fix: Add a `computational_storage_poc/__init__.py` (making it a package) and convert all internal imports to relative imports (`from .block_graph import BLOCK_SIZE`). Alternatively, add the package root to sys.path explicitly in a `conftest.py`-equivalent or in the CI step.

### HIGH Severity

**F-005: No Experiment Tracking or Artifact Registry**
- Files: `run_sweep.py`, `run_large_sweep.py`, `run_weight_refinement_campaign.py`, `benchmark_beir.py`, `benchmark_distillation.py`
- Each benchmark script writes JSON output to a path specified by the caller. There is no shared schema for benchmark results, no automatic linkage between result files and the exact config that produced them, no deduplication of runs, and no way to compare two runs on a shared dimension except by manually reading JSON files.
- The roadmap audit explicitly notes that `large_sweep_results.json` and `large_sweep_results.csv` have never been produced. The large sweep cannot be resumed without re-running from the start because there is no checkpoint-based resume protocol for the sweep driver.
- Recommended fix: Introduce a lightweight run registry (even a simple SQLite database or append-only JSONL file) that records commit hash, config snapshot, adapter path, and result metrics for every benchmark invocation. MLflow or Weights and Biases are standard tools for this; a minimal in-house version could be 100 lines.

**F-006: mock_nvme.py Preloads Entire File — Contradicts Disk-First Claim**
- File: `computational_storage_poc/mock_nvme.py`
- The feasibility memo explicitly identifies this: "mock_nvme.py loads the full file into memory at initialization, so it is not exercising real disk behavior." The Phase 7 promotion review promoted the baseline while acknowledging this gap. The storage read reduction metric (~94.71%) measures reduction against a dense block format, not against an actual disk read baseline.
- Impact: The core architectural claim of the disk-first program — that model inference can operate with a small DRAM working set by streaming weights from disk — cannot be honestly validated until the mock is replaced with real mmap-backed lazy loading.
- Recommended fix: Replace the `mock_nvme.py` preload with `mmap`-based lazy-read access. Add a latency measurement that accurately reflects SSD seek + read latency rather than memory-copy latency. This is identified in the feasibility memo's "highest-value changes" section but has not been implemented.

**F-007: AEP Orchestrator Complexity Exceeds the Complexity It Manages**
- File: `aep_orchestrator.py` (982 lines)
- The AEP orchestrator implements `Severity`, `FindingStatus` (6 states), `EffortSize` (3 sizes with numeric weights), `Finding` dataclass with 15+ fields, a `FindingTracker` with markdown and JSON export, 4 specialist agent classes, and an `AEPOrchestrator` with 7-phase workflow execution. This is a production-grade workflow engine embedded in a research repository.
- The `_impact_effort_score()` uses `len(self.impact)` (string length) as a proxy for impact magnitude. This is a known hack that has never been replaced with a real scoring heuristic.
- The parallel validation logic uses `ThreadPoolExecutor` with `as_completed`, but all specialist agents in practice just call `.validate(findings)` which is a no-op stub that returns `findings` unchanged. The parallelism is scaffolding without a real payload.
- Recommended fix: Simplify the orchestrator to a data-class-based tracker with a CLI for creating, updating, and querying findings. The full 7-phase automation is appropriate for a team environment but is overhead for a single-contributor research repo. Consider whether a well-structured Markdown tracker (as the ARCH-AEP docs describe) would serve the same purpose with less code.

**F-008: Two Incompatible Retrieval Subsystems with No Integration Plan**
- Files: `vector_store.py` (Qdrant-backed), `computational_storage_poc/repo_graph_memory.py` (DiskBackedRepoGraphMemory)
- AntigravityEngine uses `VectorStore` → `QdrantVectorStore` for embedding-based retrieval. The disk-first PoC uses `DiskBackedRepoGraphMemory` with its own embedding format (EMBEDDING_DIM=256, custom tokenization, numpy-backed mmap index). These two systems use different embedding dimensions, different storage formats, different query interfaces, and different scoring models (Qdrant cosine similarity vs hybrid lexical+graph+embedding scoring).
- There is no documented decision about whether these will converge, which one is the forward architecture for the retrieval function, or whether ChelatedAI will run both in parallel.
- Recommended fix: Write an Architecture Decision Record (ADR) that explicitly answers: (1) Is DiskBackedRepoGraphMemory a replacement for Qdrant in the disk-first path? (2) Will AntigravityEngine ever use DiskBackedRepoGraphMemory? (3) What is the interface contract between the chelation adapter and the disk-first retrieval path? Without this decision, further development in either direction risks wasted work.

**F-009: Feature Flags Are Class-Level Boolean Attributes**
- File: `config.py` lines 117-263
- Features are enabled/disabled by setting `ADAPTIVE_THRESHOLD_ENABLED = False`, `NOISE_INJECTION_ENABLED = False`, `CONVERGENCE_ENABLED = False`, `ONLINE_UPDATE_ENABLED = False`, `LEARNED_MASK_ENABLED = False`, `QUALITY_ASSESSMENT_ENABLED = False`, `BOUNDED_ADAPTER_ENABLED = False` etc. These are class-level attributes on `ChelationConfig`, which means: they cannot be overridden per-experiment without subclassing, they leave no audit trail in result files, and they cannot be varied across parallel runs without modifying source code.
- Additionally, the `enable_*` methods on AntigravityEngine (e.g., `enable_adaptive_threshold()`, `enable_stability_tracking()`) act as runtime feature flags but the config-level booleans are separate from the runtime-enable-method pattern. These are two different feature-flag mechanisms with no reconciliation.
- Recommended fix: Replace class-level boolean flags with a per-experiment feature config that is serialized alongside benchmark results. Add environment variable overrides for the most commonly varied flags.

**F-010: phase-planning.md Is Empty Template**
- File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/phase-planning.md`
- This file exists as a template with all fields blank (Cycle start date: empty, Cycle ID: empty, Backlog Summary: empty, etc.). The ARCH-AEP workflow designates this as the authoritative long-running planning record for the current cycle. Its emptiness means there is no currently active planning document tying the disk-first program's phases to the AEP tracking infrastructure.
- Impact: The current program is being run ad-hoc against task_plan.md rather than through the formal ARCH-AEP loop. This creates risk that the review and hardening steps (steps 3-4 in the program loop) are being skipped.
- Recommended fix: Populate phase-planning.md with the current cycle state: active phase (Phase 3B / MoE REAP or the optimization pass post Phase-7), scope lock, entry and exit gates, and backlog counts. Connect it to the revised roadmap document.

**F-011: AntigravityEngine.__init__ Has 9 Parameters with Complex Optional Combinations**
- File: `antigravity_engine.py` line 20
- The constructor signature has 9 parameters and complex interaction logic: `training_mode` (3 values) × `teacher_model_name` (optional) × `teacher_models` (optional list of tuples) × `use_quantization` × `use_centering` creates a combinatorial configuration space that is never fully validated at construction time. The docstring says "Stage 8 Engine" which is an internal development stage label that has leaked into the public API.
- Impact: Tests must construct engines in elaborate ways to exercise different code paths. Session 29 found a real bug where passing `use_quantization=False` silently disabled the chelation path.
- Recommended fix: Move to a builder pattern (`AntigravityEngineBuilder`) or accept a structured config object, removing the ambiguity about which parameter combinations are valid.

**F-012: The reranker in IntegratedRepoRuntime Is Hard-Coded, Not Trained**
- File: `computational_storage_poc/integrated_repo_runtime.py` lines 57-67
- `build_repo_runtime_reranker_matrices()` returns hard-coded weight matrices. The base weights (`[0.7, 0.5, 1.2, 0.8, 1.1, 0.4, 0.1, 0.3]`) are not derived from training data, not configurable, and not documented with a derivation rationale. The Phase 7 promotion review itself notes: "The reranker is deterministic and hand-authored, not trained."
- Impact: The hit-rate metrics (top-3 hit rate: 75%) cannot be trusted as evidence of real system quality because the reranker is tuned by inspection rather than by optimization.
- Recommended fix: Either train the reranker on a real query-relevance dataset or replace it with a principled linear model with explainable coefficients. At minimum, document why those specific weight values were chosen.

**F-013: Large Sweep Results Have Never Been Produced**
- Files: `run_large_sweep.py`, `large_sweep_results.json` (absent)
- The roadmap audit (2026-03-06) explicitly noted that `large_sweep_results.json` and `large_sweep_results.csv` have never been produced. This means the "Phase 6: Large Sweep Execution" from the weight refinement plan has never been executed. The plan for Phase 6 was gated on Phase 1-5 completing, which the audit says is done — but the sweep itself has not run.
- Impact: The current model presets (`SEDIMENTATION_TUNED_PRESETS`) are validated only on the standard sweep's 81-config search space. The large sweep was designed to cover 7,350 configurations. Without it, there is no evidence that the current presets are near-optimal.

**F-014: disk_llm_estimator.py Computes Theoretical Bounds Only**
- File: `computational_storage_poc/disk_llm_estimator.py`
- The estimator models upper bounds using ideal SSD bandwidth and statistical assumptions about FFN sparsity (2% of streamed weights fetched per token). There is no real measurement of any transformer inference workload, no real SSD bandwidth measurement, and no correction for DRAM latency, PCIe contention, or CPU pipeline stalls.
- Impact: The feasibility memo's numbers (e.g., "7B/4b: 105 tokens/sec on consumer Gen4") are theoretical best-case numbers, not measured performance. Planning the roadmap around these numbers without a real 1B-3B transformer benchmark creates a risk that the architecture is being validated by its own optimistic assumptions.

**F-015: No Integration Between the Chelation Adapter and the Disk-First Inference Path**
- Files: `chelation_adapter.py`, `computational_storage_poc/packed_graph.py`
- The chelation adapter (the core research artifact) lives entirely in the root-level retrieval runtime. The disk-first packed inference path in `computational_storage_poc/` operates on hand-authored MLP weight matrices with no connection to the chelation adapter's learned corrections. The two systems have never been integrated in any benchmark.
- Impact: It is unclear whether the main research contribution (adaptive chelation corrections) would be compatible with or beneficial in the disk-first inference path. The divergence between these two tracks risks the research contribution becoming irrelevant to the system's forward architecture.

**F-016: No Abstract Interface for the Adaptation Layer**
- Files: `chelation_adapter.py`, `online_updater.py`, `dimension_mask_predictor.py`
- There is no `AdapterProtocol` or `CorrectionLayer` abstract base class. The `create_adapter()` factory returns one of three concrete classes (`ChelationAdapter`, `ProcrustesAdapter`, `LowRankAffineAdapter`) that share no declared interface. The `BoundedAdapter` wrapper assumes duck-typing works correctly across all three. If a fourth adapter type is added, there is no protocol to check it against.
- Compare: `VectorStore` and `EmbeddingBackend` do have ABC definitions. The adapter layer, which is the system's core research artifact, does not.

**F-017: No Versioning Strategy for Saved Adapter Weights**
- Files: `config.py` (ADAPTER_WEIGHTS_PATH), `chelation_adapter.py` (save/load)
- The adapter saves and loads with a single path (`adapter_weights.pt`). There is no format version in the saved file, no check for compatibility between saved weights and the current model architecture, and no migration path. The config backup mechanism (`adapter_weights.benchmark-backup-*.pt`) generates backups with UUID-based names but no metadata linking the backup to the run configuration that produced it.
- Three backup files are present in the git status as untracked: `adapter_weights.benchmark-backup-808fc792cae04d3da85b5d394b3e26a2.pt`, `adapter_weights.benchmark-backup-f658c34e70e047e0a0faaab68c7b5837.pt`, `adapter_weights.benchmark-backup-fa0a54d79ce1477191b68ac54114458b.pt`. These are untracked, undocumented, and unauditable.
- Recommended fix: Add a metadata header to saved adapter files (config snapshot hash, adapter type, input dimension, training mode). Reject loads where the metadata hash does not match the current config.

**F-018: sedimentation_loss.py Is Imported Lazily Inside Method Bodies**
- File: `antigravity_engine.py` lines 1117, 1391
- `from sedimentation_loss import create_sedimentation_loss` is imported inside method bodies (`run_sedimentation_cycle` and `run_offline_distillation`) rather than at module level. This was presumably done to avoid a circular import, which is itself a symptom of the flat layout. The lazy import means that import errors from `sedimentation_loss.py` are not caught until the training path is exercised.

**F-019: Teacher Distillation with Same Teacher and Student Model Is a Silent No-Op**
- File: `teacher_distillation.py`, `benchmark_distillation.py`
- Session 29 identified that using the same model as both teacher and student means zero learning signal. This is a known bug that requires the user to remember to pass a different teacher model. There is no validation in `create_distillation_helper()` or `AntigravityEngine.__init__()` that warns when `teacher_model_name == model_name`.

**F-020: pyproject.toml Does Not Include computational_storage_poc in py-modules**
- File: `pyproject.toml`
- The `[tool.setuptools] py-modules` list enumerates 20 root-level source files but does not include any `computational_storage_poc` modules. This means the disk-first PoC code is not installable as part of the `chelatedai` package. Running `pip install -e .` does not make `block_graph`, `cpu_backends`, or `integrated_repo_runtime` importable from anywhere except from within the `computational_storage_poc/` directory.

**F-021: BEIR Tier "research" Is Defined but No research-tier Artifacts Have Been Produced**
- File: `benchmark_beir.py`, `config.py` (BEIR_PRESETS)
- The BEIR presets define "quick", "small", "medium", and "research" tiers. The roadmap audit notes that only small and medium tiers have been run. No research-tier output files exist. The weight refinement plan gates preset refinement on completing the BEIR research tier, which creates a dependency on an experiment that has never been run.

**F-022: No Cross-System Tests Connecting Retrieval and Disk-First Tracks**
- Files: root test files vs `computational_storage_poc/` test files
- None of the 14 root-level test files for the retrieval track test any interaction between `AntigravityEngine` and the disk-first `IntegratedRepoRuntime`. The two subsystems have independent test suites with no integration test verifying that a chelation-corrected embedding can be used as input to the disk-backed retrieval path.

**F-023: Dashboard Server Has No Authentication and Exposes Raw Event Logs**
- File: `dashboard_server.py`
- The dashboard serves event logs, sweep results, and BEIR results over an HTTP server with no authentication. While it only binds to localhost, any process on the machine can read the served data. If the server were ever accidentally started with a non-localhost bind, it would expose training metrics and data patterns without restriction.

**F-024: Topology Analyzer and Isomer Detector Are Experimental but Always Initialized Eagerly**
- Files: `topology_analyzer.py`, `isomer_detector.py`
- `enable_topology_analysis()` and `enable_isomer_detection()` accept kwargs but the underlying TopologyAnalyzer and IsomerDetector objects are created at enable-time with no lazy initialization. If these features are enabled at the start of a session and then not used, their memory overhead persists for the session lifetime.

**F-025: run_weight_refinement_campaign.py Does Not Check for BEIR Dataset Availability Before Starting**
- File: `run_weight_refinement_campaign.py`
- The campaign runner starts benchmark configurations sequentially without pre-checking that all required BEIR datasets are downloadable. A multi-hour run can fail 3 hours in when it reaches a dataset tier requiring network access that is unavailable. There is no dry-run mode.

### MEDIUM Severity

**F-026: sedimentation.py and sedimentation_trainer.py Have Overlapping Responsibilities**
- Files: `sedimentation.py`, `sedimentation_trainer.py`
- `sedimentation.py` defines `HierarchicalSedimentationEngine`. `sedimentation_trainer.py` defines `compute_homeostatic_target` and `sync_vectors_to_qdrant`. Both are involved in the sedimentation training loop but the split between them is not clean: `antigravity_engine.py` imports directly from `sedimentation_trainer`, not from `sedimentation`. The `HierarchicalSedimentationEngine` in `sedimentation.py` is a re-export from `recursive_decomposer.py` per the code comment, which is an unusual indirection.

**F-027: AntigravityEngine Has Two Feature-Flag Patterns That Are Not Reconciled**
- File: `antigravity_engine.py`
- Some features are enabled by calling `enable_X()` methods (topology, isomer detection, stability tracking, online updates). Others are enabled by passing parameters to the constructor (`use_centering`, `use_quantization`, `training_mode`). Others are enabled by calling setter methods (`set_temperature()`, `set_sedimentation_loss()`). There is no single surface for "what features are currently active in this engine instance."

**F-028: The ARCH-AEP Workflow Has No Automated Enforcement**
- Files: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/workflow.md`, `aep_orchestrator.py`
- The 7-phase workflow is extensively documented with 25+ files in the ARCH-AEP directory. However, all enforcement is manual: the orchestrator in `aep_orchestrator.py` relies on a human to call its methods in the correct order. There is no CI gate that checks whether a PR advances a tier before the previous tier is closed. There is no machine-readable tracker that CI can read to block a merge.

**F-029: Kalman LR Scheduler Is Wired Into AntigravityEngine But the Coupling Is Invisible from Outside**
- File: `antigravity_engine.py`, `kalman_lr_scheduler.py`
- `enable_kalman_lr()` is called from outside the engine and stores state inside the engine. But the Kalman state is then used inside `run_sedimentation_cycle()` and `run_offline_distillation()` without any visible coupling in the method signatures. A caller who reads `run_sedimentation_cycle(threshold, lr, epochs)` has no indication that the effective LR will be modulated by Kalman state.

**F-030: teacher_weight_scheduler.py Creates 5 Schedule Types That Map to 3 Functional Behaviors**
- File: `teacher_weight_scheduler.py`
- The 5 schedule types are constant, linear, cosine, step, and adaptive. The adaptive type depends on the loss value being passed at each step. In practice, only 3 distinct behavioral modes exist: static, decay, and loss-responsive. The 5-type taxonomy is finer than the system can validate, and the "adaptive" type requires the caller to pass loss values that may not be available in all training loops.

**F-031: No Protocol for Graceful Degradation When Qdrant Is Unavailable**
- File: `antigravity_engine.py`, `vector_store.py`
- The engine has a context manager (`__enter__`/`__exit__`) and a `close()` method, but no graceful degradation when Qdrant is unavailable (e.g., connection refused, disk full). The test suite uses `:memory:` Qdrant universally, so this failure mode is never exercised in tests.

**F-032: The "revised-roadmap" Document Is Marked as "Proposed Active" but Has No Owner or Status Date**
- File: `docs/revised-roadmap-disk-first-program-2026-03-28.md`
- The document says "Status: Proposed active program roadmap" without a confirmation date, a named program owner, or a record of who approved the transition from the old roadmap. In the orchestrator-briefing.md, it is listed as "the current active architecture-led reference plan" — but there is no scope-lock record connecting this to an active ARCH-AEP cycle.

**F-033: repo_graph_memory.py Uses a Home-Grown 256-Dimensional Hash Embedding**
- File: `computational_storage_poc/repo_graph_memory.py`
- The embedding function uses SHA-256 hashing of tokens to produce 256-dimensional embeddings. This is a deterministic hash-based embedding, not a semantic embedding. Semantic similarity between code snippets is not captured: two functions that do the same thing with different variable names will have completely different embeddings. The retrieval quality benchmarks (75% top-3 hit rate) are measured against this hash-based embedding, not against real semantic embeddings.
- Impact: The disk-first retrieval path's quality metrics are not comparable to the retrieval quality metrics from the Qdrant-backed path that uses real sentence-transformers embeddings.

**F-034: computational_storage_poc/retrieval_eval_suite.py Is an Orphaned Module**
- File: `computational_storage_poc/retrieval_eval_suite.py`
- This file exists in the untracked list (`git status`) but is not imported by any other file in the PoC directory and is not listed in the phase7_system_evaluation.py's import list. It is unclear whether it is a work-in-progress, a dead file, or an intended component that was never wired in.

**F-035: No Type Annotations on the Public Interface of AntigravityEngine**
- File: `antigravity_engine.py`
- The `run_inference()` method returns a tuple but its return type is not annotated. `embed()` returns `np.ndarray` but the type hint is absent. `ingest()` returns `None` implicitly. For a module with 30+ methods serving as the primary public interface, lack of type annotations significantly reduces IDE-assisted development safety and makes downstream type-checking impossible.

**F-036: Benchmark Utilities Have a BEIR Dataset Downloader That Is Not Documented as a Network Dependency**
- File: `benchmark_beir.py`, `benchmark_utils.py`
- The BEIR benchmarks silently download datasets from the internet if they are not present locally. This means CI runs that include BEIR benchmarks can fail intermittently due to network issues or rate limiting. The CI workflow does not mark these tests as network-dependent or provide a stub/mock for offline testing.

**F-037: No Documented Recovery Path for Corrupt Adapter Weights**
- File: `checkpoint_manager.py`, `chelation_adapter.py`
- The `CheckpointManager` with `SafeTrainingContext` includes SHA256 verification of checkpoint files. But if the adapter weights become corrupt (e.g., training crash mid-save), the only documented recovery is manual deletion of the weights file, which causes the engine to start from scratch. There is no last-known-good checkpoint retention policy beyond the UUID-named backups.

**F-038: The dashboard_server.py Reads Arbitrary File Paths Passed as Query Parameters**
- File: `dashboard_server.py`
- If the dashboard server reads file paths from HTTP query parameters to serve sweep results, there is a potential path traversal risk. Even for a localhost-only server, this is a security smell in a research codebase where tests and scripts may invoke the server with adversarial inputs.

**F-039: docs/INDEX.md and RESEARCH_TRACKS.md Are Partially Out of Sync**
- Files: `docs/INDEX.md`, `docs/RESEARCH_TRACKS.md`
- RESEARCH_TRACKS.md references `revised-roadmap-disk-first-program-2026-03-28.md` in Track 7, but the INDEX.md may not reflect the full set of documents added in Sessions 31-32. The phase7-promotion-review document exists but is not cross-referenced from RESEARCH_TRACKS.md.

**F-040: No Migration Path from the Flat Layout to a Package Structure**
- File: `pyproject.toml`, all root `*.py` files
- The packaging evaluation document (`docs/packaging-evaluation-2026-02-27.md`) exists, implying this was evaluated. But the current pyproject.toml still uses `py-modules` (flat module list) rather than `packages` (directory-based). There is no migration plan in any active planning document for completing the package restructure.

**F-041: AntigravityEngine Has No Read-Only or Inference-Only Mode**
- File: `antigravity_engine.py`
- The engine always initializes a `CheckpointManager`, always attempts to load adapter weights, and always creates a full training context. There is no lightweight "inference-only" mode that skips training infrastructure. This makes the engine heavier than necessary for production query serving.

**F-042: The "disable_adaptive_threshold" Method Exists but "disable_kalman_lr", "disable_stability_tracking" etc. Do Not**
- File: `antigravity_engine.py`
- The API is asymmetric. `enable_adaptive_threshold()` has a corresponding `disable_adaptive_threshold()`. But `enable_stability_tracking()`, `enable_topology_analysis()`, `enable_isomer_detection()`, `enable_online_updates()`, `enable_kalman_lr()`, and `enable_convergence_detection()` have no corresponding disable methods. Once enabled, these features cannot be turned off without reconstructing the engine.

**F-043: The AEP _impact_effort_score() Is a String-Length Heuristic**
- File: `aep_orchestrator.py` line 92
- `return len(self.impact) / self.effort.weight` — the impact score is literally the character count of the impact description string. A finding with a long verbose description outranks one with a short crisp description regardless of actual impact severity. This has never been replaced with a real heuristic despite the comment in the code acknowledging it is a proxy.

**F-040: packed_cpu_inference.py and cpu_backends.py Have Overlapping CPU Backend Abstractions**
- Files: `computational_storage_poc/packed_cpu_inference.py`, `computational_storage_poc/cpu_backends.py`
- Both files define CPU execution paths for packed models. The `cpu_backends.py` defines `CPUInferenceBackend` (ABC), `NumpyFloat32Backend`, and `NumpyInt8DynamicBackend`. The `packed_cpu_inference.py` appears to define a separate execution path. Without reading both files in full, the boundary between these two abstractions is unclear and risks duplication.

**F-044: The moe_reap.py Module Is in the PoC Directory but Has No Integration Test at Root Level**
- Files: `computational_storage_poc/moe_reap.py`, `test_moe_reap.py`
- `test_moe_reap.py` is in the root-level test suite but the module it tests (`moe_reap.py`) is inside `computational_storage_poc/`. The import will only work if the test runner adds `computational_storage_poc/` to sys.path, which the CI workflow does not explicitly do for the root-level test discovery step. This may cause intermittent test failures.

**F-045: No Component or Interface Documentation for the computational_storage_poc Subsystem**
- Files: `computational_storage_poc/` directory
- The SYSTEM_BLUEPRINT.md documents the top-level retrieval runtime well. The computational_storage_poc has only a README.md (modified, in git status). There is no equivalent system blueprint for the PoC subsystem documenting component roles, data flows between block_graph/packed_graph/repo_graph_memory/integrated_repo_runtime, or the expected execution order of phases 1-7.

**F-046: docs/ARCH AGENTIC ENGINEERING AND PLANNING Contains 25+ Files but No Master Index**
- Directory: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/`
- The orchestrator-briefing.md lists 26 files in this directory. There is no single machine-readable index of which documents are active vs archived vs template. The `tracker-pointer.md` points to the current tracker, but there is no equivalent "active-documents-pointer" that tells a new session which documents are currently relevant vs which are stale templates.

**F-047: The Convergence Monitor Is a Standalone Module but Is Only Used via AntigravityEngine**
- File: `convergence_monitor.py`
- `ConvergenceMonitor` is a standalone class but it is only instantiated inside `AntigravityEngine.enable_convergence_detection()`. The module-level API is never used independently. This creates a pattern where reusable classes exist as standalone files but are only ever accessed through the engine, making them harder to test in isolation.

**F-048: The cross_lingual_distillation.py Module Has No Benchmark That Validates Its Cross-Lingual Claims**
- File: `cross_lingual_distillation.py`, `test_cross_lingual_distillation.py`
- The cross-lingual distillation module has unit tests that validate code behavior but no benchmark that validates the empirical claim: that routing through language-specific teachers actually improves retrieval quality for non-English queries. The research track "Can teacher-guided correction generalize across languages?" remains unanswered by any measurement.

**F-049: No Formal Definition of "Semantic Collapse" as a Measurable Condition**
- The core problem the system solves is "semantic collapse in RAG systems." But there is no formal definition of semantic collapse that is operationalized as a metric. The chelation threshold and chelation log detect a proxy (variance below a threshold), not semantic collapse directly. Without a formal definition and a synthetic benchmark for collapse, it is impossible to claim the system "detects and fixes" the problem rather than detecting and reacting to a correlated signal.

**F-050: The RP2040 Hardware Evidence Gap Is Tracked as an Open Item but Has No Actionable Next Step**
- Files: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-index.md`, `docs/computational-storage-hardware-evidence-runbook.md`
- "Real RP2040 hardware evidence capture" has been listed as an open tracker item since Session 22. The retention policy notes a review date of 2026-04-05. This date has now passed (current date: 2026-04-04). The item has no assigned owner, no definition of what hardware count constitutes evidence, and no fallback plan if no RP2040 is acquired.

### LOW Severity

**F-051: Collection Name "antigravity_stage8" Is a Development-Stage Artifact Exposed in the Public API**
- File: `config.py` line 149
- `DEFAULT_COLLECTION_NAME = "antigravity_stage8"` is a hard-coded development stage label. This name appears in any Qdrant database created by the engine, making it impossible to distinguish development vs production collections by name convention.

**F-052: The pyproject.toml version Is "0.1.0" and Has Never Been Incremented**
- File: `pyproject.toml` line 7
- The version has been "0.1.0" since the initial packaging evaluation. With 1,082 tests, 84 source files, and 8 research tracks, the version number no longer reflects the system's maturity relative to its original scope. Semantic versioning is not enforced by any CI check.

**F-053: Docstring Quality Is Inconsistent Across Core Modules**
- The `AntigravityEngine.__init__` docstring says "Stage 8 Engine: Docker/Ollama Integration + Teacher Distillation" — an obsolete development-phase description. Some methods have no docstrings (`_gravity_sensor`, `_chelate_toxicity`). Others have detailed docstrings. A public-facing research prototype should have consistent docstrings on all public methods.

**F-054: The BEIR Presets Define a "quick" Tier That Only Tests SciFact**
- File: `config.py` (BEIR_PRESETS)
- The "quick" tier runs only SciFact. This means CI is effectively overfitting quick validation to a single dataset. The "quick" tier should include at least 2-3 diverse datasets to catch regressions that SciFact alone would miss.

**F-055: requirements.txt and pyproject.toml Dependency Lists May Diverge**
- Files: `requirements.txt`, `pyproject.toml`
- The project has both a `requirements.txt` (for direct pip install) and a `pyproject.toml` (for package install). These can diverge over time, especially for optional dependencies. The CI uses `requirements.txt`, but `pip install -e .` uses `pyproject.toml`. A dependency added to one but not the other will cause silent CI vs local environment differences.

**F-056: The benchmark_evolution.py File Has an Unclear Purpose**
- File: `benchmark_evolution.py`
- This file appears in the root but is not mentioned in the SYSTEM_BLUEPRINT.md, RESEARCH_TRACKS.md, or any docs. It is not listed in the pyproject.toml py-modules list, suggesting it is either a new file or was intentionally excluded from the package.

**F-057: The CHELATEDAI_integration_demo.py File in computational_storage_poc Is Not Integrated in CI**
- File: `computational_storage_poc/CHELATEDAI_integration_demo.py`
- This file is named as an integration demo but does not appear in any CI step. Its relationship to the Phase 7 evaluation is unclear. If it is a manual demo, it should be documented. If it is a test, it should be in the CI matrix.

**F-058: No Lint Rules for Computational Storage PoC Code**
- File: `.github/workflows/test.yml`, `ruff check .`
- `ruff check .` runs from the repo root and covers all Python files including `computational_storage_poc/*.py`. However, the PoC code was added more recently and may not have been reviewed for consistency with the lint rules applied to the rest of the codebase.

**F-059: The AEP Glossary Is Not Machine-Readable**
- File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/glossary.md`
- The glossary is a markdown file. Because it is prose, there is no way to automatically validate that finding IDs, severity values, or effort sizes used in trackers and backlogs match the glossary definitions.

**F-060: Weight Refinement Campaign Results Are Session-Specific, Not Version-Tagged**
- File: `docs/weight-refinement-campaign-results-2026-03-06-session28.md`
- The results file is named by session date rather than by commit hash or model version. If the same campaign is re-run on a different commit, there is no filename convention that prevents overwriting or confusion with the earlier run.

**F-061: The structural_health_report Uses Config-Driven Thresholds That Are Not Documented as Tuned**
- File: `antigravity_engine.py` (get_structural_health_report)
- The structural health report uses config-driven thresholds for persistent collapse, oscillation, topology cohesion, and isomer drift. The CLAUDE.md notes these are config-driven. But there is no documentation of how these threshold values were chosen or validated against real data.

**F-062: No Parallelization Barrier Analysis for Future Scale**
- Files: `antigravity_engine.py`, `online_updater.py`
- The engine uses a `threading.Lock` for adaptive threshold updates but does not protect the adapter weights from concurrent access. If multiple threads call `embed()` simultaneously (which could happen if the engine is used in a web service context), there is a race condition on adapter weight access during `torch.no_grad()` forward passes.

**F-063: The MoE REAP Implementation in moe_reap.py Has Never Been Evaluated Against a Real MoE Model**
- File: `computational_storage_poc/moe_reap.py`
- The REAP addendum clearly states that REAP is relevant for coding-oriented MoE models at 20B-1T parameter scale. The PoC implementation uses the existing toy block-graph format rather than a real MoE architecture. The pruning simulation is on synthetic experts, not on real expert weight matrices from a deployed MoE model.

**F-064: The storage_substrate_benchmark.py "94.71% Read Reduction" Is Against a Toy Format Baseline**
- File: `computational_storage_poc/storage_substrate_benchmark.py`
- The 94.71% read reduction is measured against the old dense 512x512 FP16 block format. It is not measured against a full model loaded naively into memory, not against a real NVMe read baseline, and not against a GGUF or similar real production format. The benchmark context is internally consistent but not externally comparable.

**F-065: No Semantic Versioning for the ARCH-AEP Workflow Specification Itself**
- File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/workflow.md`
- The workflow.md has been modified (git status shows M). There is no version history in the document itself. If a cycle was started against an earlier version of the workflow, there is no way to know which version of the workflow protocol governed that cycle's decisions.

**F-066: The "Devil's Advocate" Layer Has No Mechanism in the Current ARCH-AEP Workflow**
- The AEP workflow defines specialist agents (Architecture, Security, Testing, Performance, Reliability, Documentation, UX) but no Contrarian or Devil's Advocate role. The workflow's "dispute resolution" section handles disagreements but assumes disagreements arise naturally, not that a devil's advocate is systematically assigned to challenge each tier's decisions.

**F-067: Online Updater Has 3 Loss Functions with No Documented Interaction with Sedimentation Loss**
- Files: `online_updater.py`, `sedimentation_loss.py`
- The online updater supports `triplet_margin`, `infonce`, and `cosine_similarity` loss. The sedimentation trainer supports `mse`, `infonce`, and `hybrid`. There are two separate InfoNCE implementations (one for online updates, one for sedimentation training). It is undocumented whether these are intentionally separate or whether they were separately implemented without coordination.

**F-068: The Board of Backup Adapter Weights Has No Cleanup Policy**
- Files: `adapter_weights.benchmark-backup-*.pt` (3 files, untracked)
- Three backup adapter weight files are sitting as untracked in the working directory. The `computational_storage_poc/model.cspg` is also untracked. There is a documented retention policy for hardware evidence artifacts (`docs/computational-storage-retention-policy-2026-03-06.md`), but no equivalent retention policy for adapter weight backups.

**F-069: The pyproject.toml py-modules List Will Require Manual Updates for Every New Source File**
- File: `pyproject.toml`
- The `py-modules` list manually enumerates every source module. Adding a new source file requires also updating this list. This is a maintenance tax that will only grow as the codebase expands, and a new file omitted from the list will silently be absent from the installed package.

**F-070: No Performance Regression Baseline Exists for the Core Retrieval Path**
- Files: `benchmark_beir.py`, test suite
- There is no performance regression test that runs in CI. The tests validate correctness (behavior) but not performance (latency, throughput). If a future change to `AntigravityEngine.run_inference()` or the adapter forward pass doubles query latency, CI would not detect it.

---

## Challenge Phase

**DA challenges F-002 (Flat File Layout):** The flat layout is a deliberate research choice, not an oversight. In a research codebase where every file may be imported directly for experimentation, packages add indirection friction. Python's module system handles 84 files without issue. The real problem is import discipline, which can be enforced by convention without restructuring.

**Panel response:** Discipline scales with team size, not file count. At 84 files with zero namespace isolation, the risk of silent name shadowing is real and grows with every new file. The pyproject.toml already lists 20 modules manually, which is itself evidence that the flat layout is already creating maintenance overhead. The discipline argument would be stronger if there were any documented naming convention that prevents future collisions.

**DA challenges F-001 (God Object):** AntigravityEngine being large is not inherently bad. The fluent API (`engine.enable_X().enable_Y()`) is user-friendly and all features are coherently related to adaptive retrieval. Splitting it into 5 collaborator classes would require the caller to understand the collaboration protocol, which is more complex than a single entry point.

**Panel response:** The fluent API can be preserved with a thin facade over collaborators. The current structure fails unit testing (30 methods, 1,621 lines, every test must mock the whole engine). The Session 29 bugs (chelation path silent no-op, Procrustes dead zero init, low-rank double suppression) were all harder to diagnose precisely because the training logic is interleaved with retrieval logic in a single class.

**DA challenges F-007 (AEP Over-Engineering):** The AEP orchestrator exists because the project has accumulated significant technical debt across 31+ sessions. Without a formalized remediation workflow, bugs would go unresolved and scope would creep without accountability. The complexity of the orchestrator is a reflection of the complexity of managing a research project at this maturity level.

**Panel response:** The DA's point has merit for the documentation layer. But the Python `aep_orchestrator.py` file itself implements the workflow as executable code that is barely used programmatically — the `validate()` methods on specialist agents are stubs. The documentation layer (workflow.md, templates, tracker) is valuable; the Python class hierarchy that mirrors it adds maintenance cost without corresponding benefit.

**DA challenges F-006 (mock_nvme.py preloads):** The feasibility memo explicitly says the system cannot claim disk-resident inference today. Everyone knows this. The mock is a placeholder and the Phase 7 review correctly deferred production promotion. Calling this a critical finding overstates its urgency.

**Panel response:** The issue is that the Phase 7 benchmarks are being used to justify the roadmap's next phases (MoE/REAP integration, KV compression) while the fundamental storage measurement is against an in-memory mock. Phase 8+ decisions should not be made on metrics that measure memory-copy performance. The urgency is about keeping the research trajectory honest.

**DA challenges F-033 (Hash-Based Embeddings):** The repo_graph_memory uses hash embeddings because the disk-first path needs a compute-free, storage-stable embedding that works without model inference. Semantic embeddings require a loaded model, which contradicts the disk-first goal of minimal resident compute. The trade-off is explicit.

**Panel response:** The trade-off may be valid architecturally, but the 75% hit-rate metric is presented without qualification that it measures hash-similarity retrieval, not semantic retrieval. Users reading the Phase 7 promotion review may incorrectly compare this 75% against semantic retrieval baselines from AntigravityEngine.

---

## Dissent Log

**Dr. Nadia Al-Amin (research-to-production):** Findings F-043 (string-length scoring) and F-033 (hash embeddings) should both be elevated to HIGH severity. A research prototype that validates its metrics against trivially-weak baselines risks publishing misleading results. A system that uses string length as an impact proxy will systematically mis-prioritize findings. Both of these directly undermine research integrity.

**Michael Okonkwo (package structure):** F-002 should be CRITICAL with higher urgency than F-003. The God Config is bad, but a God Config in a properly structured package is containable. 84 files at root is an architectural anti-pattern that gets worse every time a new module is added, and the flat-module pyproject.toml is evidence the developers are already managing the symptoms.

**Raj Krishnamurthy (distributed systems):** F-062 (threading race condition on adapter weights) should be HIGH severity, not LOW. If the engine is ever used in a Flask or FastAPI web service (the dashboard_server already exists as a proof that HTTP serving is in scope), concurrent access to the adapter during inference creates a real correctness bug. The `torch.no_grad()` context manager does not protect against concurrent write access during training.

**Sophia Wren (program management):** F-050 (RP2040 hardware evidence gap) is understated as LOW severity. The retention policy review date (2026-04-05) has passed. If the item is not resolved with an explicit closure decision in the next session, it becomes an indefinitely-open tracker item consuming planning attention. A formal "close without hardware evidence, replace with emulation-only scope" decision would be cleaner than leaving it open.

**Dr. Isabelle Fontaine (architecture):** F-008 (two incompatible retrieval subsystems) should remain HIGH but the risk is more subtle than stated. The real issue is not just that two systems exist — it is that the chelation adapter (the core research contribution) has never been coupled to the disk-first retrieval path. If the system evolves toward disk-first retrieval as the forward path, the chelation correction work may become architecturally orphaned. This is a strategic risk to the project's research coherence.

---

## Feasibility Assessment and Architectural Decision Recommendations

### Decision 1: Package Restructure (F-002, F-004, F-020, F-040, F-069)

**Recommendation: Proceed in two steps.**
- Step 1 (low-risk): Add `computational_storage_poc/__init__.py` and convert internal imports to relative imports. This is a self-contained change with zero impact on the root-level modules.
- Step 2 (medium-risk, defer to next major session): Create a `chelatedai/` package directory and move core modules in. Keep the flat root as a backwards-compatibility shim for one cycle using re-exports. Update pyproject.toml to `packages = ["chelatedai"]`.

**Feasibility rating: High (Step 1 this session, Step 2 next major session).**

### Decision 2: AntigravityEngine Decomposition (F-001, F-011, F-016, F-027, F-041)

**Recommendation: Extract in three increments, preserving the public interface.**
- Increment 1: Extract training logic into a `SedimentationTrainer` class (separate from the existing `sedimentation_trainer.py`). This is the highest-value extraction because the training path is separate from the inference path.
- Increment 2: Extract diagnostic/monitoring logic into a `DiagnosticsService`.
- Increment 3: Extract teacher distillation into a standalone `DistillationSession` class.
- The `AntigravityEngine` facade remains the public API surface throughout.

**Feasibility rating: Medium (each increment is a 2-3 hour PR, requires careful testing).**

### Decision 3: ChelationConfig Decomposition (F-003, F-009, F-017)

**Recommendation: Replace preset dictionaries with per-subsystem frozen dataclasses.**
- Define `SedimentationConfig`, `AdapterConfig`, `TeacherConfig`, `TopologyConfig` as frozen dataclasses with default values.
- Keep `ChelationConfig.get_preset()` as a factory for backward compatibility.
- Add a `config_version` field to `AdapterConfig` that is serialized alongside adapter weights.

**Feasibility rating: High (additive change, backwards compatible).**

### Decision 4: Experiment Tracking (F-005, F-013, F-021, F-060)

**Recommendation: Implement a minimal append-only run registry.**
- Add a `RunRegistry` class (100-200 lines) that appends JSON records to `runs.jsonl`.
- Each record: commit hash, timestamp, config snapshot (dict), result metrics, adapter path.
- Have all benchmark scripts write to this registry automatically.
- No external dependency required (plain Python + JSON).

**Feasibility rating: High (1-2 hours, no dependency).**

### Decision 5: ADR on Retrieval Path Forward Architecture (F-008, F-015, F-033)

**Recommendation: Write an Architecture Decision Record within the next session that explicitly answers:**
1. Is `DiskBackedRepoGraphMemory` a replacement for, complement to, or parallel-track-only alternative to Qdrant + ChelationAdapter?
2. What is the integration interface between chelation corrections and disk-first retrieval?
3. When does hash-based embedding become an unacceptable quality floor?

**Feasibility rating: High (docs-only, 1 session).**

### Decision 6: mock_nvme Real mmap Implementation (F-006, F-014, F-064)

**Recommendation: Replace mock_nvme.py with an mmap-backed lazy reader before Phase 3B begins.**
- The Phase 3B (MoE/REAP) metrics will inherit the same problem as Phase 1-7 metrics if the storage mock is not replaced.
- A real mmap-backed reader requires 50-100 lines of Python and will immediately produce honest latency numbers.

**Feasibility rating: High (straightforward mmap Python, 1-2 hours).**

### Decision 7: Validate Teacher Distinctness at Construction Time (F-019)

**Recommendation: Add a warning in `create_distillation_helper()` when teacher and student model names are the same. Add a note in `AntigravityEngine.__init__()` if `teacher_model_name == model_name`.**

**Feasibility rating: Very High (2 lines of code, one test).**

### Decision 8: AEP Orchestrator Simplification (F-007, F-043, F-028)

**Recommendation: Separate the data model (Finding, FindingTracker) from the workflow automation (AEPOrchestrator class with 7-phase execution). The data model is well-designed and provides value as a structured tracker. The workflow execution layer's stub validation agents add complexity without value. Replace the parallel validation stub with a single synchronous validation call.**

**Feasibility rating: Medium (requires careful refactoring, risk of breaking tests).**

### Decision 9: RP2040 Hardware Evidence Formal Closure (F-050)

**Recommendation: In the next session, make a formal scope decision: either (a) close the item as "hardware evidence not collected; emulation-path validation is the permanent substitute" and archive the tracker item, or (b) acquire RP2040 hardware and set a concrete evidence-collection date. The retention policy review date has passed and the ambiguity should be resolved.**

**Feasibility rating: Very High (documentation decision only).**

### Decision 10: Add ChelationAdapter ABC (F-016)

**Recommendation: Define a `CorrectionLayer` Protocol or ABC with `forward()`, `save()`, `load()`, and `regularization_loss()`. Have all three adapter types register against it. Update `create_adapter()` return type annotation.**

**Feasibility rating: High (additive, 30-50 lines, no behavior change).**

---

## Architecture Evolution Roadmap

### Immediate (Next Session, Low-Risk Changes)

These changes are backwards-compatible, require no architecture surgery, and can be done in 1-3 hour PRs:

1. Add `computational_storage_poc/__init__.py` and convert its internal imports to relative imports. (F-004, F-020)
2. Add `CorrectionLayer` ABC for chelation adapters. (F-016)
3. Add teacher distinctness validation in `create_distillation_helper()`. (F-019)
4. Populate `phase-planning.md` with current cycle state and connect it to the revised roadmap. (F-010, F-032)
5. Add a `config_version` field to adapter save/load. (F-017)
6. Write RP2040 hardware evidence scope closure decision. (F-050)
7. Replace `mock_nvme.py` with an mmap-backed lazy reader. (F-006)

### Near-Term (2-3 Sessions, Medium Complexity)

8. Implement `RunRegistry` append-only experiment log. All benchmark scripts write to it. (F-005, F-013, F-060)
9. Write ADR: retrieval path forward architecture decision. (F-008, F-015, F-033)
10. Extract `SedimentationTrainer` from `AntigravityEngine` as first decomposition increment. (F-001)
11. Replace AEP `_impact_effort_score()` string-length heuristic with a real scoring model. (F-043)
12. Add adapter save/load metadata validation against current config. (F-017)
13. Add `AntigravityEngine.disable_kalman_lr()`, `disable_stability_tracking()`, `disable_topology_analysis()` for API symmetry. (F-042)

### Medium-Term (Major Session or Dedicated Refactor, Higher Risk)

14. Begin package restructure: move core modules to `chelatedai/` package. Provide re-export shims at root. Update pyproject.toml. (F-002, F-040, F-069)
15. Extract `DiagnosticsService` from `AntigravityEngine`. (F-001)
16. Per-subsystem config dataclasses replacing preset-dict sections. (F-003)
17. Add performance regression CI tests for the core retrieval path (latency budget checks). (F-070)
18. Run BEIR research-tier benchmarks; update presets based on results. (F-021)
19. Train the IntegratedRepoRuntime reranker on real relevance data. (F-012)

### Long-Term (Program-Level Strategic Decisions)

20. Semantic embedding integration into `DiskBackedRepoGraphMemory` (replace hash embeddings). (F-033)
21. Integrate the chelation adapter into the disk-first retrieval path — this is the critical research coherence work. (F-015)
22. Real 1B-3B transformer benchmark with CPU-only inference and mmap-backed weight loading. (F-014, F-006)
23. MoE/REAP integration with a real MoE model (Phase 3B of the disk-first program). (F-063)
24. KV cache compression via TurboQuant-like approach for long-context paths (Phase 6 expansion). (addendum)

---

## Findings Summary by Severity

| Severity | Count | Key Findings |
|---|---|---|
| CRITICAL | 4 | God Object engine (F-001), flat file layout (F-002), God Config (F-003), CWD-dependent imports (F-004) |
| HIGH | 21 | No experiment tracking (F-005), mock_nvme preloads (F-006), AEP over-engineering (F-007), dual retrieval systems (F-008), class-level feature flags (F-009), empty phase-planning (F-010), constructor complexity (F-011), hard-coded reranker (F-012), zero large sweep artifacts (F-013), theoretical-only estimator (F-014), zero chelation/disk-first integration (F-015), no adapter ABC (F-016), no weight versioning (F-017), lazy sedimentation_loss import (F-018), teacher=student silent no-op (F-019), PoC not installable (F-020), BEIR research tier never run (F-021), no cross-system integration tests (F-022), dashboard no auth (F-023), eager diagnostic init (F-024), campaign no dry-run (F-025) |
| MEDIUM | 30 | Sedimentation module split (F-026), dual feature-flag patterns (F-027), unenforced ARCH-AEP workflow (F-028), Kalman state hidden coupling (F-029), teacher schedule taxonomy (F-030), no Qdrant graceful degradation (F-031), roadmap no owner (F-032), hash-based embeddings (F-033), orphaned retrieval_eval_suite (F-034), no type annotations (F-035), silent BEIR network dependency (F-036), no corrupt weights recovery (F-037), dashboard path traversal risk (F-038), docs out of sync (F-039), no package migration plan (F-040), no inference-only mode (F-041), asymmetric enable/disable API (F-042), string-length heuristic (F-043), overlapping CPU backend abstractions (F-040b), moe_reap.py import path (F-044), no PoC system blueprint (F-045), AEP no active doc index (F-046), ConvergenceMonitor only via engine (F-047), cross-lingual no benchmark (F-048), no semantic collapse definition (F-049), RP2040 item past review date (F-050) |
| LOW | 20 | Stage-8 collection name (F-051), stale version number (F-052), inconsistent docstrings (F-053), SciFact-only quick tier (F-054), requirements vs pyproject drift (F-055), benchmark_evolution unclear (F-056), CHELATEDAI_integration_demo not in CI (F-057), no PoC-specific lint coverage check (F-058), non-machine-readable glossary (F-059), session-specific result filenames (F-060), undocumented health report thresholds (F-061), threading race on adapter weights (F-062), REAP on synthetic experts only (F-063), toy-format storage baseline (F-064), workflow.md has no version (F-065), no contrarian role in AEP (F-066), dual InfoNCE implementations (F-067), no backup retention policy (F-068), manual py-modules maintenance (F-069), no latency regression CI (F-070) |

**Total findings: 70**

---

## Risk Register — Top 5 Strategic Risks

**SR-1 [HIGH]: Research coherence risk.** The chelation adapter (the core research contribution) and the disk-first retrieval path are architecturally diverging with no documented integration intent. If the disk-first path becomes the forward retrieval architecture, the chelation work may have no integration point. This risks the research contribution being validated only in a retrieval context that the system is moving away from.

**SR-2 [HIGH]: Metrics reliability risk.** Three key benchmark metrics are measured against weak baselines: (1) storage read reduction vs. dense toy blocks; (2) CPU speedup vs. float32 (only 1.19x); (3) retrieval hit rate using hash-based embeddings. Published metrics may not be comparable to any real-world baseline, and planning decisions made on them may be systematically optimistic.

**SR-3 [MEDIUM]: Scaling ceiling.** The flat file layout and God Object engine have a scaling ceiling in the low hundreds of modules. At the current rate of growth (84 root files after 31 sessions), the repository will become unmanageable in another 20-30 sessions without structural intervention.

**SR-4 [MEDIUM]: Knowledge concentration.** The ARCH-AEP workflow and session log accumulate a significant amount of institutional knowledge in markdown files. There is no mechanism that ensures this knowledge is recoverable if the session log is corrupted, the docs directory is restructured, or a new contributor needs to understand the system without reading 60+ documentation files.

**SR-5 [LOW]: Compliance window for hardware claim.** The computational-storage hardware track has an open RP2040 evidence item whose review date has passed. If the repo's hardware claim (deterministic transport proof on RP2040) is ever externally scrutinized, the absence of physical hardware evidence against a claimed real-hardware target could be perceived as an unsupported claim, even though the scope was intentionally narrowed.

---

*Panel cycle completed: CONVENE → SOLO → CHALLENGE → CONVERGE → FEASIBILITY → DELIVER.*
*Report produced: 2026-04-04.*
*Output file: `docs/panel-analysis/07-architecture-planning.md`.*
