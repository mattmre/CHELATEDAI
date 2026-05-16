# Panel of Experts Report 06: Documentation Quality

**Review date:** 2026-04-04
**Repository:** ChelatedAI (D:/GITHUB/CHELATEDAI)
**Panel mandate:** Identify every documentation accuracy issue, gap, stale content, structural problem, and improvement opportunity across the full repository documentation surface.

---

## Panel Composition

| Expert | Background | Lens |
|--------|-----------|------|
| Dr. Sarah Osei | Technical Writer, 15 years | Accuracy, completeness, clarity, logical flow |
| Marco Pereira | Developer Advocate, 11 years | Developer experience, onboarding friction, API docs |
| Dr. Chen Wei | Research Documentation Specialist, 12 years | Methodology, reproducibility, citation accuracy |
| Alicia Thompson | Knowledge Management Expert, 13 years | Information architecture, discoverability, cross-referencing |
| Ivan Petrov | Open Source Contributor, 9 years | README quality, contribution guides, community docs |
| Devil's Advocate | Contrarian Expert | Challenges every consensus finding |

---

## CONVENE Phase

**Mandate:** Surface every documentation accuracy issue, gap, stale content, structural problem, and improvement opportunity across README.md, CLAUDE.md, docs/INDEX.md, docs/RESEARCH_TRACKS.md, docs/MODULE_GUIDE.md, docs/SYSTEM_BLUEPRINT.md, docs/README.md, REFERENCES.md, computational_storage_poc/README.md, docs/disk-resident-llm-feasibility-2026-03-28.md, docs/phase7-promotion-review-2026-03-29.md, docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md, and source-file docstrings.

**Scope boundary:** The panel reviews documentation as documentation. Test behavior, runtime correctness, and CI configuration are out of scope unless they are specifically documenting those topics.

---

## Executive Summary: Top 10 Findings

1. **CRITICAL — README.md "Current Research Status" is nine weeks stale.** The README still says "as of 2026-03-06" and does not mention the entire disk-first CPU and retrieval program (Phases 1–7), which was designed, implemented, benchmarked, and promoted between 2026-03-28 and 2026-03-29. Readers encounter a false picture of the repository's current state immediately.

2. **CRITICAL — CHANGELOG.md is frozen at 2026-01-06.** The changelog lists "Phase 4 – Pending" items that have long since been completed (streaming ingestion, web dashboard, CI/CD pipeline). It has not been updated through any of the 31+ sessions, leaving the version history section permanently at v0.2.0 with a placeholder v0.1.0 entry dated "2024-01-XX".

3. **HIGH — CLAUDE.md omits the entire new computational_storage_poc module surface.** Approximately 15 new files added in sessions 28–32 (`packed_graph.py`, `cpu_backends.py`, `sparse_cpu_inference.py`, `repo_graph_memory.py`, `integrated_repo_runtime.py`, `moe_reap.py`, `disk_llm_estimator.py`, `phase7_system_evaluation.py`, and their benchmarks) are not listed anywhere in CLAUDE.md's architecture section or dependency graph. This is the operator guidance file for future coding sessions and its silence on these modules is a navigation hazard.

4. **HIGH — REFERENCES.md has a wrong arXiv ID for the primary MRL citation.** References 1 and 2 both cite Matryoshka Representation Learning as arXiv:2602.03306, but the actual MRL paper by Kusupati et al. is arXiv:2205.13147. Reference 2 is also a duplicate of Reference 1 using different framing ("Dimension Selection") with the same wrong ID and the same authors. REFERENCES.md is dated "Last updated: 2026-02-21" and is missing citations for all Session 31 additions (InfoNCE / NT-Xent, LoRA-style low-rank init, GAM-RAG Kalman-gain LR, REAP, TurboQuant, LLM in a Flash, T-MAC, ReLU Strikes Back, SeedLM).

5. **HIGH — docs/INDEX.md session log table covers only Sessions 1–3** out of 31 completed sessions. Sessions 4–31 (25 additional session logs that physically exist in the `ARCH AGENTIC ENGINEERING AND PLANNING/` directory) are absent from the index, making the session archive undiscoverable through the primary index document.

6. **HIGH — docs/INDEX.md contains two broken file links.** The table references `ARCH AGENTIC ENGINEERING AND PLANNING/session-log-2026-02-13-impl.md` (Session 1) and `ARCH AGENTIC ENGINEERING AND PLANNING/research-f006-config-mapping.md`, `research-f010-logger-migration.md`, `research-f002-f003-test-plan.md`, and `proposed-trials-validation-round-2026-02-17.md` — none of which exist on disk. The INDEX will 404 for readers following those links.

7. **HIGH — MODULE_GUIDE.md does not cover the Phase 1–7 disk-first substrate.** `packed_graph.py`, `cpu_backends.py`, `packed_cpu_inference.py`, `sparse_cpu_inference.py`, `repo_graph_memory.py`, `integrated_repo_runtime.py`, `moe_reap.py`, `disk_llm_estimator.py`, `phase7_system_evaluation.py`, and their benchmark companions are all absent from the module guide. These are not minor utilities; they form the entire new research track documented in `revised-roadmap-disk-first-program-2026-03-28.md`.

8. **HIGH — CLAUDE.md reports "1082 tests passing on main as of 2026-03-12" but the date is now 2026-04-04.** Dozens of new test files were added in sessions after 31 (test_cpu_inference.py, test_disk_llm_estimator.py, test_integrated_repo_runtime.py, test_packed_graph.py, test_sparse_cpu_inference.py, test_moe_reap.py, test_repo_graph_memory.py, test_phase7_system_evaluation.py, test_memory_compression.py, and others). The stated count of 1082 is now materially understated and the date is stale.

9. **MEDIUM — antigravity_engine.py lacks a module-level docstring.** The file begins directly with import statements. It is the central runtime entry point and the most-read source file in the codebase, yet it has no module-level docstring describing purpose, key classes, or usage pattern. The `__init__` docstring is still labeled "Stage 8 Engine: Docker/Ollama Integration + Teacher Distillation," which refers to an internal development stage numbering that means nothing to readers not present during that session.

10. **MEDIUM — REFERENCES.md is located at the repository root but CLAUDE.md says it is at `docs/REFERENCES.md`.** This wrong path will cause confusion for any session that follows CLAUDE.md's "Reference Material" section.

---

## Full Findings List

### CRITICAL Severity

**F-01 — README.md "Current Research Status" is nine weeks stale**
File: `README.md` lines 120–129
Details: The section reads "As of 2026-03-06" and describes the current active work as being narrowly focused on real RP2040 hardware evidence capture and retention review. The entire disk-first CPU and retrieval program (Phases 1–7, spanning six weeks of implementation and benchmarking, yielding a promoted research baseline with measured results: 94.71% packed-read reduction, 61.98% sparse byte reduction, 75% top-3 hit rate) is completely absent. Any reader who consults the README to understand where the project stands will receive a picture that is factually wrong about the scope and status of current work.
Recommendation: Expand the status section to describe the disk-first CPU track with a brief summary of each completed phase and a pointer to the revised roadmap and Phase 7 promotion review.

**F-02 — CHANGELOG.md is permanently frozen at 2026-01-06**
File: `CHANGELOG.md`
Details: The CHANGELOG covers only the initial Phase 1–3 hardening pass. It still lists as "pending" items like streaming ingestion, web dashboard, MTEB benchmark expansion, and CI/CD pipeline — all of which were completed in subsequent sessions. The version history section shows only v0.2.0 (2026-01-06) and a placeholder v0.1.0 (2024-01-XX with no actual date). Sessions 3 through 31+ added hundreds of new features, adapters, loss functions, benchmarks, and an entirely new disk-resident track without a single CHANGELOG entry. The stated "Contributors" section contains only "Phase 1-3 refactoring: 2026-01-06".
Recommendation: Either adopt a keep-a-changelog-style format updated with each PR, or formally deprecate this file and redirect to the session-log archive. The intermediate state — a frozen CHANGELOG that implies completion at a much earlier point — is worse than no CHANGELOG at all.

**F-03 — CHANGELOG.md "Next Steps (Phase 4 - Pending)" are all implemented**
File: `CHANGELOG.md` lines 231–240
Details: Seven items are listed as pending: streaming batch operations, memory monitoring, web dashboard, MTEB benchmark expansion (FEVER, HotpotQA, NFCorpus), adaptive threshold learning, adapter weight compression, and CI/CD pipeline. All of these exist in the current codebase (`ingest_streaming`, `dashboard_server.py`, `benchmark_beir.py` with multi-dataset support, `enable_adaptive_threshold`, and `.github/workflows/`). A reader checking what work remains will conclude incorrectly that these are open deliverables.
Recommendation: Strike the entire "Next Steps (Phase 4 - Pending)" section or replace it with a "Completed" retrospective note pointing to the appropriate session logs.

---

### HIGH Severity

**F-04 — CLAUDE.md architecture section omits the entire disk-first substrate**
File: `CLAUDE.md` architecture section
Details: The "Core module dependency graph" in CLAUDE.md covers only the original retrieval runtime (`AntigravityEngine` and its direct dependencies). The 15+ modules that form the disk-first CPU and retrieval track (`packed_graph.py`, `cpu_backends.py`, `packed_cpu_inference.py`, `sparse_cpu_inference.py`, `repo_graph_memory.py`, `retrieval_eval_suite.py`, `integrated_repo_runtime.py`, `moe_reap.py`, `disk_llm_estimator.py`, `phase7_system_evaluation.py`, and all benchmark companions) are entirely absent. A coding session agent using CLAUDE.md as its navigation document will not know these files exist, which defeats the purpose of CLAUDE.md as operator guidance.
Recommendation: Add a second dependency graph or module inventory section for the computational storage and disk-first substrate.

**F-05 — REFERENCES.md contains a wrong arXiv identifier for MRL (primary)**
File: `REFERENCES.md` lines 10–11
Details: References 1 and 2 both cite Matryoshka Representation Learning with arXiv:2602.03306. The MRL paper by Kusupati et al. (NeurIPS 2022) is arXiv:2205.13147. The identifier 2602.03306 does not correspond to this paper. This error appears in the primary citation used to justify the dimension mask predictor design, which is a core research component.
Recommendation: Correct the arXiv ID to 2205.13147 and update the author attribution to Kusupati et al. (NeurIPS 2022).

**F-06 — REFERENCES.md Reference 2 is a duplicate of Reference 1 with a different title**
File: `REFERENCES.md` lines 13–15
Details: Reference 2 ("Information-Theoretic Dimension Selection") cites the same paper (same authors, same arXiv ID) as Reference 1. It is not a separate citation; it is a relabeled copy of Reference 1. This inflates the citation count from 16 unique citations to 17 while providing no additional reference value.
Recommendation: Remove Reference 2 entirely, or replace it with the appropriate information-theoretic dimension selection paper if one was intended.

**F-07 — REFERENCES.md is missing citations for all Session 31+ additions**
File: `REFERENCES.md`, dated "Last updated: 2026-02-21"
Details: The following techniques implemented in Sessions 28–31 have no corresponding citations: InfoNCE / NT-Xent loss (Oord et al., 2018 / Chen et al., 2020), LoRA-convention asymmetric init for `LowRankAffineAdapter` (Hu et al., ICLR 2022, cited in the code comment but absent from REFERENCES.md), GAM-RAG Kalman-gain adaptive LR (March 2026 — cited in `kalman_lr_scheduler.py` but absent from REFERENCES.md), REAP expert pruning (Cerebras, arXiv:2510.13999), TurboQuant (Google Research), LLM in a Flash (Apple, used as architectural template in disk-resident feasibility), T-MAC (CPU lookup kernel), ReLU Strikes Back / SparseGPT (FFN sparsity). The code docstrings cite these; the REFERENCES.md does not.
Recommendation: Update REFERENCES.md with all missing citations, correct the last-updated date, and establish a policy that new research features require a corresponding REFERENCES.md entry.

**F-08 — docs/INDEX.md session log table covers only 3 of 31 sessions**
File: `docs/INDEX.md` lines 57–64
Details: The session log table in INDEX.md lists Sessions 1, 2, and 3 and no others. In the ARCH AGENTIC ENGINEERING AND PLANNING directory there are 25 session log files covering sessions 3 through 31 (with gaps at sessions 29 and 30 — see F-23). Sessions 4–28, 31 are completely invisible to anyone navigating via the index. The INDEX.md table also lists only the first backlog file and three early research files, giving a severely incomplete picture of the process archive.
Recommendation: Extend the session log table to include all sessions through 31, or restructure the table to use a "see full list" pointer to a sub-index maintained in the ARCH directory.

**F-09 — docs/INDEX.md contains broken file links (5 missing files)**
File: `docs/INDEX.md`
Details: The following links resolve to files that do not exist:
- `ARCH AGENTIC ENGINEERING AND PLANNING/session-log-2026-02-13-impl.md` (Session 1 log — missing)
- `ARCH AGENTIC ENGINEERING AND PLANNING/session-log-2026-02-13-impl-2.md` (Session 2 log — missing)
- `ARCH AGENTIC ENGINEERING AND PLANNING/research-f006-config-mapping.md`
- `ARCH AGENTIC ENGINEERING AND PLANNING/research-f010-logger-migration.md`
- `ARCH AGENTIC ENGINEERING AND PLANNING/research-f002-f003-test-plan.md`
- `proposed-trials-validation-round-2026-02-17.md`
A reader following any of these links will receive a 404 or "file not found" error. The early session logs appear to have been either never committed or removed, but the INDEX retains dead links to them.
Recommendation: Audit all links in INDEX.md. Remove or update dead links. If early session logs were intentionally not committed, add a note explaining this.

**F-10 — docs/MODULE_GUIDE.md omits the Phase 1–7 disk-first substrate**
File: `docs/MODULE_GUIDE.md`, "Computational Storage And Drive-Node Research" section
Details: MODULE_GUIDE.md's computational storage section covers only the 9 original POC files (`block_graph.py`, `compiler.py`, `train_and_compile.py`, `mock_nvme.py`, `mock_array.py`, `CHELATEDAI_integration_demo.py`, `payload_contract.py`, `usb_host_inference.py`, `capture_hardware_evidence.py`). The following new files are entirely absent: `packed_graph.py`, `cpu_backends.py`, `packed_cpu_inference.py`, `sparse_cpu_inference.py`, `sparse_inference_benchmark.py`, `repo_graph_memory.py`, `repo_graph_memory_benchmark.py`, `repo_graph_memory_compression_benchmark.py`, `retrieval_eval_suite.py`, `integrated_repo_runtime.py`, `integrated_runtime_benchmark.py`, `integrated_runtime_compression_benchmark.py`, `moe_reap.py`, `moe_reap_benchmark.py`, `disk_llm_estimator.py`, `phase7_system_evaluation.py`, `storage_substrate_benchmark.py`. This is the largest single omission in the module documentation.
Recommendation: Add a new "Disk-First CPU and Retrieval Substrate" subsection to MODULE_GUIDE.md that inventories all Phase 1–7 files with their responsibilities.

**F-11 — CLAUDE.md test count is stale**
File: `CLAUDE.md` line 51
Details: CLAUDE.md states "1082 tests passing on main as of 2026-03-12." The date is now 2026-04-04 and numerous new test files exist (`test_cpu_inference.py`, `test_disk_llm_estimator.py`, `test_integrated_repo_runtime.py`, `test_packed_graph.py`, `test_sparse_cpu_inference.py`, `test_moe_reap.py`, `test_repo_graph_memory.py`, `test_phase7_system_evaluation.py`, `test_memory_compression.py`) that are absent from the representative test file list and push the actual test count well above 1082.
Recommendation: Update the count and date after each session, or replace the hardcoded count with a note that the count is tracked via CI badge.

**F-12 — CLAUDE.md representative test file list is incomplete**
File: `CLAUDE.md` lines 51–68
Details: The representative test files list 13 files and does not mention any of the Phase 1–7 test files or several other existing tests: `test_cpu_inference.py`, `test_disk_llm_estimator.py`, `test_integrated_repo_runtime.py`, `test_packed_graph.py`, `test_sparse_cpu_inference.py`, `test_moe_reap.py`, `test_repo_graph_memory.py`, `test_phase7_system_evaluation.py`, `test_memory_compression.py`, `test_vector_store.py`, `test_benchmark_utils.py`, `test_run_weight_refinement_campaign.py`, `test_integration_rlm.py`, `test_online_correction.py`, `test_sedimentation_trainer.py`, `test_convergence_monitor.py`, `test_checkpoint_manager.py`, `test_chelation_logger.py`, `test_sweep_presets.py`. This affects a coding session's ability to verify which test to run when working on a specific module.
Recommendation: Either extend the list to cover all test files grouped by module area, or replace the list with a link to the test matrix.

**F-13 — CLAUDE.md path for REFERENCES.md is wrong**
File: `CLAUDE.md` line 155
Details: CLAUDE.md states `docs/REFERENCES.md -- 17 research paper citations`. The file is actually located at the repository root as `REFERENCES.md`, not inside `docs/`. Any session agent following this path will look in the wrong directory.
Recommendation: Correct the path to `REFERENCES.md`.

**F-14 — antigravity_engine.py has no module-level docstring**
File: `antigravity_engine.py` lines 1–18
Details: The file begins directly with import statements. As the central engine and most-referenced module, the absence of a module-level docstring is a significant gap. The `__init__` docstring still reads "Stage 8 Engine: Docker/Ollama Integration + Teacher Distillation" — an internal development stage label that communicates nothing meaningful about the current system, which now encompasses 12+ capabilities including teacher scheduling, online updates, Kalman LR, contrastive loss, topology analysis, isomer detection, and structural health reporting.
Recommendation: Add a module-level docstring describing purpose, primary class, public API, and key design patterns. Update the `__init__` docstring to describe the current capability set without internal stage numbering.

**F-15 — computational_storage_poc/packed_graph.py, cpu_backends.py, sparse_cpu_inference.py, and integrated_repo_runtime.py have no module-level docstrings**
Files: `computational_storage_poc/packed_graph.py`, `computational_storage_poc/cpu_backends.py`, `computational_storage_poc/sparse_cpu_inference.py`, `computational_storage_poc/integrated_repo_runtime.py`
Details: These four files, which form the core of the Phase 1–7 substrate, begin with import blocks and code without any module-level docstring. `cpu_backends.py` defines the `CPUInferenceBackend` abstract base and `BackendResult` dataclass with no explanation of the backend contract. `integrated_repo_runtime.py` is the flagship Phase 5 output and the entry point to the integrated CPU + retrieval runtime, yet it has no module-level description of what it does or how to use it.
Recommendation: Add module-level docstrings for all four files following the style established in the well-documented modules (e.g., `topology_analyzer.py`, `online_updater.py`).

---

### MEDIUM Severity

**F-16 — docs/RESEARCH_TRACKS.md "Current state" for Track 7 does not reflect Phase 7 completion**
File: `docs/RESEARCH_TRACKS.md` lines 164–186
Details: Track 7 ("Disk-First CPU and Retrieval Program") says "The research basis now exists, but the implementation roadmap is only partially reflected in the older docs." As of 2026-03-29, Phase 7 was promoted as a research baseline. The implementation is complete through Phase 7 and Phase 3B (MoE / REAP branch). The current state description is now inaccurate in the direction of underselling actual completion.
Recommendation: Update Track 7 current state to reflect the Phase 7 promotion, list what phases are complete, and describe the remaining gap (production promotion deferred, per phase7-promotion-review).

**F-17 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/phase-planning.md is an unfilled template**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/phase-planning.md`
Details: phase-planning.md is described as the "single long-running planning record for the current ARCH-AEP cycle," but every field in the template is blank: cycle start date, orchestrator, PR range, cycle ID, phase goals, dependencies, remediation strategy, and decision log. This is a critical operational document that is perpetually empty, making it misleading rather than useful.
Recommendation: Either fill in the current cycle metadata, or explicitly mark the document as "no active cycle" with a pointer to the most recent completed session log.

**F-18 — docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md is not marked superseded**
File: `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md` lines 1–3
Details: The document has a banner: "> Superseded as the active program roadmap by revised-roadmap-disk-first-program-2026-03-28.md." This is good. However, README.md still points to this document in its "Current Research Status" section as the active audit reference, without flagging that it has been superseded. A reader who follows the README link will find the superseded document and may not notice the small banner text.
Recommendation: Update the README.md "Current Research Status" section link to point to the revised roadmap, with a parenthetical note about the older audit for historical context.

**F-19 — docs/INDEX.md "Analysis" section is nearly empty**
File: `docs/INDEX.md` lines 68–73
Details: The Analysis section lists only one document (`rlm-analysis.md`), which does not exist on disk (file is missing). The research analysis files for sessions 28, 31, and the disk-first track (`docs/disk-resident-llm-feasibility-2026-03-28.md`, `docs/disk-resident-llm-addendum-reap-turboquant-2026-03-28.md`, `docs/phase7-promotion-review-2026-03-29.md`, `docs/revised-roadmap-disk-first-program-2026-03-28.md`) are absent from this section, as are the session-scoped architecture and research documents in `docs/ARCH AGENTIC ENGINEERING AND PLANNING/`.
Recommendation: Expand the Analysis section to cover the disk-first track research documents and correct or remove the broken `rlm-analysis.md` link.

**F-20 — REFERENCES.md "Spectral Chelation" citation (Reference 3) lacks a specific paper**
File: `REFERENCES.md` lines 21–23
Details: Reference 3 for "Spectral Reranking Methods" is attributed only to "Standard spectral methods in information retrieval" with no specific author, paper title, venue, or year. This is a placeholder, not a citation. Spectral reranking has concrete literature (e.g., Laplacian eigenmaps for IR, spectral clustering of embedding spaces) that could be cited.
Recommendation: Replace the placeholder with a specific paper citation or remove the entry if no specific paper was used.

**F-21 — docs/disk-resident-llm-feasibility-2026-03-28.md assumes "LLM in a Flash" was the intended paper without verification**
File: `docs/disk-resident-llm-feasibility-2026-03-28.md` lines 7–14
Details: The document explicitly states "The user request referenced 'this paper' without attaching the paper in the session context. For this memo, the working assumption is that the target paper is LLM in a Flash." This assumption is never verified or flagged for follow-up. As a research document, this creates an unresolved uncertainty that could affect all conclusions drawn from the paper analysis if the assumption turns out to be wrong.
Recommendation: Add a prominently visible "NOTE: Paper assumption unresolved" block at the document header, or if the paper has since been confirmed, update the document to remove the conditional framing.

**F-22 — TECHNICAL_ANALYSIS.md is not flagged as deprecated / historical in the file itself**
File: `TECHNICAL_ANALYSIS.md`
Details: `docs/README.md` correctly maps `TECHNICAL_ANALYSIS.md` to `SYSTEM_BLUEPRINT.md` as its canonical successor. However, the file itself contains no deprecation notice. It still presents as current architectural documentation with API reference content for the `AntigravityEngine.__init__` that predates teacher distillation, online updates, Kalman LR, bounded adapters, and the structural health system. A reader discovering this file via search or link will read it as current and form a wrong mental model.
Recommendation: Add a header banner marking the file as historical, stating its date, and pointing to `docs/SYSTEM_BLUEPRINT.md` as the current architectural reference.

**F-23 — Session logs for Sessions 29 and 30 are absent from the archive**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/` (directory)
Details: The session log archive jumps from `session-log-2026-03-07-session28.md` directly to `session-log-2026-03-12-session31.md`. Sessions 29 and 30 are mentioned in the memory file and in CLAUDE.md as containing critical bug findings (chelation path bug, Procrustes init bug, low-rank double suppression, same-model distillation no-op), but no session logs for these sessions exist in the archive. These sessions produced research findings that remain referenced elsewhere without a primary source log.
Recommendation: If sessions 29 and 30 occurred but logs were not written, create retrospective summary logs or at minimum add a note in the archive index explaining the gap.

**F-24 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/glossary.md covers only ARCH-AEP process terms, not domain terms**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/glossary.md`
Details: The glossary contains 8 entries covering process concepts (finding, tier, blocked, deferred, etc.). There is no glossary for domain terms used throughout the codebase and research docs: chelation, sedimentation, homeostatic target, noise center, semantic collapse, spectral chelation, isomer (retrieval isomer), Jaccard similarity, NDCG, block-graph, payload contract, deterministic transport, sector 100, orthogonal Procrustes, skew-symmetric parameterization, Cayley transform, or any of the molecular metaphors (covalent bond, Van der Waals, etc.) used in `topology_analyzer.py`. A new reader encountering these terms has no single reference point.
Recommendation: Create a separate `docs/GLOSSARY.md` (or expand the existing glossary) covering domain terms with brief definitions. This would substantially reduce onboarding time for readers new to the project's vocabulary.

**F-25 — No CONTRIBUTING.md exists at the repository root**
File: Missing — `CONTRIBUTING.md`
Details: The repository has a SECURITY.md, CODEOWNERS, and LICENSE but no CONTRIBUTING.md. There are no documented guidelines for how to run tests before submitting changes, what the branch naming convention is, how PRs are structured (the convention of one coherent theme per PR is described in CLAUDE.md but not in any externally visible document), what lint standards apply, or how to set up the development environment. The rlm_reference subdirectory has its own CONTRIBUTING.md, which a newcomer might mistake for the repository's contribution guide.
Recommendation: Create a minimal CONTRIBUTING.md that covers: environment setup, running tests, branch naming, PR structure, lint (ruff), and the no-pytest rule.

**F-26 — README.md "Current Research Status" links to superseded roadmap document**
File: `README.md` line 129
Details: The README links to `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md` as the current evaluation plan. That document contains a header marking it as superseded by `revised-roadmap-disk-first-program-2026-03-28.md`. The README should link to the active program roadmap, not the superseded audit.
Recommendation: Update the README link to `docs/revised-roadmap-disk-first-program-2026-03-28.md` with a brief description of what changed.

**F-27 — Module guide does not list sedimentation.py**
File: `docs/MODULE_GUIDE.md`, "Distillation, Scheduling, And Adaptation" section
Details: `sedimentation.py` (HierarchicalSedimentationEngine) exists at the repository root and is referenced in CLAUDE.md's dependency graph (under `RecursiveRetrievalEngine`), but it does not appear in MODULE_GUIDE.md. This is a documented module that the module guide omits.
Recommendation: Add `sedimentation.py` to the appropriate section in MODULE_GUIDE.md with its responsibility description.

**F-28 — Module guide does not list benchmark_utils.py, run_overnight_campaign.py, or run_weight_refinement_campaign.py**
File: `docs/MODULE_GUIDE.md`
Details: Three utility/driver scripts that exist at the repository root are absent from MODULE_GUIDE.md: `benchmark_utils.py` (shared benchmark helpers and data loading — listed in CLAUDE.md's architecture section), `run_overnight_campaign.py` (overnight multi-phase evaluation runner, referenced in CLAUDE.md next-session notes), and `run_weight_refinement_campaign.py` (bounded campaign runner mentioned in CLAUDE.md and SYSTEM_BLUEPRINT.md).
Recommendation: Add these three files to the "Evaluation And Experiment Drivers" section of MODULE_GUIDE.md.

**F-29 — CLAUDE.md architecture diagram omits convergence_monitor.py from the main dependency graph**
File: `CLAUDE.md` lines 76–99
Details: The CLAUDE.md dependency graph shows `AntigravityEngine` depending on `convergence_monitor.py` via the text reference "`convergence_monitor.py` -- Phase 1: patience-based early stopping," but `convergence_monitor.py` does not appear as a node in the ASCII module graph. This is an inconsistency between the graph and the listed modules.
Recommendation: Add `convergence_monitor.py` to the AntigravityEngine dependency graph, or add a parenthetical note that it is used indirectly via the sedimentation cycle.

**F-30 — docs/SYSTEM_BLUEPRINT.md does not mention the disk-first CPU track**
File: `docs/SYSTEM_BLUEPRINT.md`
Details: SYSTEM_BLUEPRINT.md covers the "three major surfaces" of the repository but Surface 3 is described only as "a computational-storage proof-of-concept that tests drive-resident node execution." The entire disk-first CPU and retrieval program (packed artifacts, CPU inference substrate, sparse loading, repo graph memory, integrated runtime) is not mentioned anywhere in the blueprint. This is now inaccurate — there are four major surfaces.
Recommendation: Update the System Summary to mention the disk-first CPU substrate as a fourth surface, and add a subsection under "Main Runtime Areas" describing the Phase 1–7 stack.

**F-31 — docs/README.md (docs home) does not link to disk-first track docs**
File: `docs/README.md`
Details: The canonical docs home lists its canonical documents and recommended reading paths (runtime, research, storage). None of the disk-first CPU track documents appear in the canonical docs table or the reading paths: `disk-resident-llm-feasibility-2026-03-28.md`, `disk-resident-llm-addendum-reap-turboquant-2026-03-28.md`, `phase7-promotion-review-2026-03-29.md`, `revised-roadmap-disk-first-program-2026-03-28.md`. A reader who reads docs/README.md and follows the recommended reading paths will never encounter the primary disk-first research documents.
Recommendation: Add a "Understand the disk-first CPU track" reading path and update the canonical documents table.

**F-32 — CLAUDE.md "Key APIs" section omits new disk-first POC APIs**
File: `CLAUDE.md`, "Key APIs" section
Details: The Key APIs section covers `AntigravityEngine`, `create_adapter`, `DimensionProjection`, `EnsembleTeacherHelper`, `TeacherWeightScheduler`, `create_weight_scheduler`, `create_distillation_helper`, and `RecursiveRetrievalEngine`. It omits all APIs from the disk-first substrate: `DiskBackedPackedGraph`, `packed_cpu_inference`, `SparseChunkCache`, `DiskBackedRepoGraphMemory`, `MoEReapArtifact`, `DiskLLMEstimator`, and the integrated runtime entry point. A session agent that needs to use or test any of these will not find them referenced in CLAUDE.md.
Recommendation: Add a "Computational Storage and Disk-First APIs" subsection to the Key APIs section.

**F-33 — COMPLETION_SUMMARY.md contains stale "future development" language**
File: `COMPLETION_SUMMARY.md`
Details: COMPLETION_SUMMARY.md has a historical note from 2026-03-07 explaining that the "old Phase 4 and future development wording below is preserved as historical context only." However, the document still lists "Future Development (Phase 4-5)" checkboxes with open/pending states — specifically items like streaming support and memory monitoring that are now implemented. The historical note is good but easy to miss.
Recommendation: Increase the visibility of the historical note — make it a prominent banner at the top, not a buried inline paragraph. Alternatively, move the stale content to a clearly labeled "Historical Context" section.

**F-34 — PR_DESCRIPTION.md historical note is in the middle of the document**
File: `PR_DESCRIPTION.md`
Details: PR_DESCRIPTION.md contains "Deferred to Phase 4" items that are now implemented. The historical note from 2026-03-07 is present but appears after the summary stats block, not at the top. A skim reader will encounter the stale "Deferred" items before reaching the note.
Recommendation: Move the historical note to the top of the document as the first visible content.

**F-35 — REFACTORING_PLAN.md historical note is partially inlined**
File: `REFACTORING_PLAN.md`
Details: Like COMPLETION_SUMMARY.md and PR_DESCRIPTION.md, REFACTORING_PLAN.md has a historical context note from 2026-03-07 but it is buried inside the document. The "Phase 4: Performance" section with open checkboxes follows after the note, confusing readers about current status.
Recommendation: Same as F-34: move the historical note to the document header.

**F-36 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md describes Session 32 scope but appears to be a persistent document**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md`
Details: The next-session checklist references specific session 31 state ("Session 31 Fixes," "PRs #96-#103 all merged," "1082 tests passing") that is now historical. The document was designed as a "session handoff checklist" but has not been updated for the current session start. A new session that follows this checklist will act on outdated context — especially the instruction to "launch overnight campaign" which may have already been done.
Recommendation: Either update next-session.md to reflect the current state after each session, or make the document clearly dated and add a note saying "superseded by" a more recent handoff document.

**F-37 — docs/COMPUTATIONAL_STORAGE_DRIVE_NODES.md does not mention Phase 1–7 substrate**
File: `docs/COMPUTATIONAL_STORAGE_DRIVE_NODES.md`
Details: This document is the "canonical overview for the repo's hard-drive, SSD, and storage-node experiments." It describes the original block-graph format, mock NVMe path, storage-node array simulation, and deterministic transport contract but contains no mention of the Phase 1–7 disk-first CPU substrate. The packed artifact format, CPU inference substrate, sparse loading, repo graph memory, and integrated runtime all directly extend the research scope of this document.
Recommendation: Add a "Phase 1–7 Extension: Disk-First CPU Substrate" section that summarizes the new research track and cross-references the promotion review.

**F-38 — computational_storage_poc/README.md section ordering is confusing**
File: `computational_storage_poc/README.md`
Details: The README now has 8+ distinct sections in roughly the order they were added (Binary Format, then Packed Manifest, then CPU Substrate, then Sparse Runtime, then Repo Graph Memory, then Integrated Runtime, then Memory Compression, then MoE/REAP, then Usage, then USB/Emulation Payload Contract, then Current Scope Lock). The "Usage" section (with `python compiler.py`) appears after the Phase 4–6 sections but uses only the original compiler output. The "Current Scope Lock" section appears at the end but should arguably be a prominent header note. The document reflects accretion without reorganization.
Recommendation: Restructure the POC README into logical sections: (1) Overview and scope note, (2) Original block-graph format and binary spec, (3) Phase-by-phase substrate additions, (4) Usage commands, (5) Transport and firmware notes.

**F-39 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/glossary.md omits molecular-metaphor terms**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/glossary.md`
Details: The topology analyzer uses chemistry metaphors (covalent bond, hydrogen bond, Van der Waals bond) as first-class concepts with specific cosine-similarity thresholds. These terms appear throughout test files and research docs but are defined only inside `topology_analyzer.py`'s module docstring. No centrally visible glossary document explains them to readers who encounter them in session logs or findings.
Recommendation: Add bond classification definitions to the repository-level glossary or to a new `docs/GLOSSARY.md`.

**F-40 — No troubleshooting guide exists**
File: Missing
Details: The CHANGELOG.md contains a "Known Issues" section from 2026-01-06 with three issues (Ollama model not found, adapter dimension mismatch, memory usage). No current troubleshooting guide covers: test suite failures due to missing torch/sentence-transformers, Qdrant connection errors, CI branch policy issues (noted extensively in CLAUDE.md Git Workflow Notes but not in a user-facing document), RP2040 not found (no hardware), Windows raw-device path handling, or numpy/sklearn import behavior differences between Python versions.
Recommendation: Create a `docs/TROUBLESHOOTING.md` covering the most common failure modes documented across CLAUDE.md, CHANGELOG.md, and session logs.

**F-41 — docs/computational-storage-hardware-evidence-runbook.md is not verified current**
File: `docs/computational-storage-hardware-evidence-runbook.md`
Details: This runbook is referenced from multiple documents as the operator guide for RP2040 evidence capture. Given that the codebase has evolved substantially since its creation, it should be verified that the runbook still accurately describes the current `capture_hardware_evidence.py` interface, file paths, and validation steps. No "last reviewed" date is present.
Recommendation: Add a "Last reviewed" date field and confirm the runbook matches current `capture_hardware_evidence.py` usage.

**F-42 — CLAUDE.md does not document the Python 3.9 forward-annotation requirement for new computational_storage_poc modules**
File: `CLAUDE.md`
Details: CLAUDE.md's "Test Conventions" section notes that "Python 3.9 CI compatibility: If a module imported by tests uses `X | None` annotations, add `from __future__ import annotations`." The new POC modules (`packed_graph.py`, `cpu_backends.py`, etc.) use `from __future__ import annotations` correctly. However, CLAUDE.md does not explicitly call out these files or remind sessions that new computational_storage_poc modules imported by tests must follow this rule.
Recommendation: Extend the Python 3.9 compatibility note to explicitly mention that this applies to all test-imported files including the computational_storage_poc modules.

**F-43 — CLAUDE.md "Session 30 Research and Fixes" memory note references files not in the ARCH directory**
File: CLAUDE.md (via memory file)
Details: The memory file references `research-session30-findings.md`, `session30-strategic-research.md`, and `architecture-vision-dual-hemisphere.md` as files in memory. These are memory-layer entries, not files checked into the repository. A session agent that tries to `ls` or `Read` these files will not find them, which could cause confusion.
Recommendation: Clarify in CLAUDE.md (or memory) that these are memory-layer documents and not checked-in files.

**F-44 — docs/INDEX.md "Proposed Trials And Validation Round" link is broken**
File: `docs/INDEX.md` line 66
Details: The link `[Proposed Trials and Validation Round (2026-02-17)](proposed-trials-validation-round-2026-02-17.md)` resolves to a file that does not exist. This is a document from the early session archive that either was never committed or was removed.
Recommendation: Remove or update this link.

**F-45 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/orchestrator-briefing.md references docs/agentic-review-framework.md as a required input**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/orchestrator-briefing.md`, workflow.md
Details: The ARCH-AEP Workflow specification lists `docs/agentic-review-framework.md` as a required workflow input. This file does not exist in the repository. It is also referenced in the orchestrator narrative as "ingest `docs/agentic-review-framework.md`." This is a hard dependency of the ARCH-AEP workflow that has never been created.
Recommendation: Either create `docs/agentic-review-framework.md` with the content implied by its usage in the workflow, or update the workflow spec to reflect that this document was consolidated into `workflow.md` or the orchestrator briefing.

---

### LOW Severity

**F-46 — CHANGELOG.md version history has a placeholder date for v0.1.0**
File: `CHANGELOG.md` line 257
Details: The version history entry for v0.1.0 reads "2024-01-XX (Initial Prototype)" — a placeholder that was never filled in.
Recommendation: Either determine and fill in the actual date or remove the placeholder entry.

**F-47 — CLAUDE.md uses "Stage 8 Engine" terminology in a Key APIs example**
File: `CLAUDE.md`, Key APIs section
Details: The AntigravityEngine API description does not use "Stage 8" language, but the source code's `__init__` docstring still says "Stage 8 Engine." This internal staging terminology should be replaced with descriptive API documentation.
Recommendation: Addressed as part of F-14 (module docstring update).

**F-48 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/change-log.md content unclear**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/change-log.md`
Details: This document is described as a "scope/defer decision log" but the reviewer could not verify its current content state. If it remains in template form with no entries for the 31+ sessions, it is a gap in the audit trail.
Recommendation: Verify that scope/defer decisions made during sessions 24–31 are logged in this document.

**F-49 — README.md Documentation Guide section links to docs/README.md but not to MODULE_GUIDE.md directly**
File: `README.md` lines 178–187
Details: The Documentation Guide section correctly lists the five primary canonical docs and their purposes. However, it lists `docs/MODULE_GUIDE.md` as link item 3 but the surrounding text does not call attention to the fact that MODULE_GUIDE.md is now incomplete (see F-10). A reader following the guide may assume MODULE_GUIDE.md is comprehensive.
Recommendation: After F-10 is addressed, this becomes a non-issue. Until then, consider adding a note that the module guide is being updated to cover the disk-first substrate.

**F-50 — CODEOWNERS has only a single wildcard entry**
File: `CODEOWNERS`
Details: CODEOWNERS contains `* @mattmre`. This means every file in the repository is owned by a single person with no module-area ownership structure. For a multi-track research codebase with distinct subsystems (retrieval runtime, evaluation harness, computational storage, AEP process), this is workable but provides no granularity for review assignment.
Recommendation: While appropriate for a single-maintainer repo, consider adding subsystem-level ownership entries for future scalability, especially if the disk-first substrate track attracts contributors.

**F-51 — SECURITY.md does not describe the path-traversal security controls in config.py**
File: `SECURITY.md`
Details: `config.py` implements `validate_safe_path()` and `sanitize_name()` as documented security controls. SECURITY.md only describes vulnerability reporting and supported version policy. It does not mention any of the implemented security posture, which would help security reviewers understand what protections exist and what the attack surface is.
Recommendation: Add a brief "Security Architecture" section noting the path-traversal prevention in `validate_safe_path()` and any other implemented controls.

**F-52 — GITHUBCHELATEDAIrlm_reference directory name is malformed**
File: `GITHUBCHELATEDAIrlm_reference/` (root directory)
Details: There is a directory named `GITHUBCHELATEDAIrlm_reference` at the repository root. This appears to be a git submodule or clone of the RLM reference with a concatenated path as its name. CLAUDE.md refers to the RLM reference as `rlm_reference/`, which is a separately existing directory. The malformed `GITHUBCHELATEDAIrlm_reference` is not documented anywhere.
Recommendation: Determine whether this is an accidentally created directory (a git clone artifact), add a note in CLAUDE.md if it is intentional, or remove it if it is not.

**F-53 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/agent-learning.md is not linked from INDEX.md session section**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/agent-learning.md`
Details: The cross-session learnings file is listed in the orchestrator briefing but is not linked from the public-facing INDEX.md, making it undiscoverable to a reader who does not know to look inside the ARCH directory for it.
Recommendation: Add a link to `agent-learning.md` from the INDEX.md AEP section.

**F-54 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/README.md uses "minimum artifacts per cycle" language that conflicts with orchestrator-briefing**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/README.md` lines 31–37
Details: The README lists "Minimum artifacts per cycle" as: scope lock record, backlog file, tracker file, tracker pointer, backlog index entry, tracker index entry, verification log entries. The orchestrator-briefing lists the cycle close checklist with slightly different items and includes phase/cycle summaries. There is a minor inconsistency that could cause a session to miss required artifacts.
Recommendation: Align the minimum artifact list between README.md and orchestrator-briefing.md.

**F-55 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/workflow.md references a non-existent refinement-cycle input format**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/workflow.md` line 31
Details: The workflow inputs list "Latest refinement report (e.g., docs/refinement-cycle-YYYY-MM-DD.md)". No such document format or example exists in the repository. The refinement reports that actually exist use different naming conventions.
Recommendation: Update the example to match a real document pattern (e.g., the session logs or the roadmap audit document).

**F-56 — docs/INDEX.md session log table format is inconsistent with other tables**
File: `docs/INDEX.md` lines 57–65
Details: The session log table and the backlog entry use a 2-column format (link, description) while the Core Documentation and AEP sections use 2-column tables with different header labels ("Document", "Description"). The session section is also missing a proper table header, which means the pipe-table formatting may not render correctly in all Markdown viewers.
Recommendation: Add a consistent table header row to the session log section.

**F-57 — REFERENCES.md "Spectral Chelation" section description is circular**
File: `REFERENCES.md` lines 20–23
Details: The description reads "Standard spectral methods in information retrieval" and the relevance says "Validates center-of-mass centering approach used in spectral chelation." This is circular — it uses "spectral chelation" to describe a reference that justifies "spectral chelation." Without a specific paper, this entry provides no actual scholarly grounding.
Recommendation: See F-20.

**F-58 — No version badge or CI status badge in README.md**
File: `README.md`
Details: The README has no CI status badge, Python version compatibility badge, or license badge. Given that the repository has active CI and supports Python 3.9–3.12, a CI badge would provide readers with an immediate visual signal about the build state.
Recommendation: Add a GitHub Actions CI badge and a Python version compatibility badge to the README header.

**F-59 — docs/disk-resident-llm-feasibility-2026-03-28.md uses informal "Short Answer" framing**
File: `docs/disk-resident-llm-feasibility-2026-03-28.md` lines 24–45
Details: The document opens with a "Short Answer" section that presents a definitive architectural conclusion ("ChelatedAI can plausibly evolve into a disk-resident, CPU-executed inference path"). This is appropriate content but the informal framing ("Short Answer / Long Answer") is inconsistent with the academic memo style of other documents in the research track. The conditional framing (the assumed paper may be wrong) adds uncertainty that undercuts the "Short Answer" structure.
Recommendation: Retain the content but restructure as "Executive Summary" and "Technical Analysis" for consistency with peer documents.

**F-60 — docs/phase7-promotion-review-2026-03-29.md benchmark summary uses "about" approximations**
File: `docs/phase7-promotion-review-2026-03-29.md` lines 58–86
Details: The benchmark summary states values like "about 94.71%", "about 1.19x", "about 61.98%", "about 3.33 ms." Using "about" in front of numbers with 4+ significant figures is an internal inconsistency — either the precision is meaningful (and "about" is wrong) or the precision is not meaningful (and the number should be rounded). This is a minor stylistic issue but matters for research documentation rigor.
Recommendation: Remove the "about" qualifiers and present the values as measured results, or explicitly note measurement uncertainty if applicable.

**F-61 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-pointer.md content unknown**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-pointer.md`
Details: The tracker pointer is described as the "authoritative pointer to the active tracker file." Its current content was not reviewed in this panel, but given that no active ARCH-AEP cycle appears to be in progress and phase-planning.md is blank, there is a risk that the tracker pointer is stale (pointing to a tracker for a completed cycle or pointing to nothing).
Recommendation: Verify that tracker-pointer.md accurately reflects the current tracker state and update it if an active cycle exists.

**F-62 — No architecture decision records (ADRs) exist**
File: Missing
Details: The repository has made numerous significant architectural decisions that are not captured as formal ADRs: the choice to use orthogonal Procrustes vs. MLP vs. low-rank adapters, the decision to use Qdrant rather than other vector databases, the flat file layout choice, the selection of InfoNCE over MSE, the decision to scope the hardware claim to "deterministic transport proof," the disk-first CPU architecture direction. These decisions are scattered across session logs and research docs but not recorded in a structured format. When future contributors revisit these choices, the rationale may be difficult to reconstruct.
Recommendation: Create a `docs/decisions/` directory with lightweight ADRs for the most consequential architectural decisions. This does not require a full MADR template — even a simple markdown file with context, decision, and consequences is valuable.

**F-63 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/ has undocumented sub-session logs with impl-N naming**
File: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/` session logs
Details: Sessions 3–18 use naming like `session-log-2026-02-17-impl-3.md` through `session-log-2026-02-18-impl-17.md`. There is no session log named `session-log-*-impl-10.md` (between impl-9 and impl-11). The INDEX.md section lists only Sessions 1–3 and all later sessions (21–31) use `session-NN` naming. The transition between `impl-N` and `session-NN` naming is not explained anywhere, and the missing "session 10" file is not noted.
Recommendation: Add a naming convention note in the orchestrator-briefing or AEP README explaining the transition from `impl-N` to `session-NN` numbering. Note the absence of an impl-10 log.

**F-64 — docs/ARCH AGENTIC ENGINEERING AND PLANNING/research-2026-03-12-* files not listed in INDEX.md**
File: `docs/INDEX.md`; files: `research-2026-03-12-session31-contrastive-loss.md`, `research-2026-03-12-session31-projection-and-kalman.md`, `research-2026-03-12-session31-scaler-constrainer.md`
Details: These three Session 31 research documents exist in the ARCH directory (confirmed via git status) but are not listed in docs/INDEX.md. They document the key technical decisions for InfoNCE loss, DimensionProjection gradient behavior, and the BoundedAdapter scaler-constrainer design — important reference documents that are effectively invisible to anyone relying on the index.
Recommendation: Add these documents to the Session Logs and Research section of INDEX.md.

**F-65 — README.md module walkthrough for computational storage is incomplete relative to current codebase**
File: `README.md` lines 155–164
Details: The "Computational storage and drive nodes" subsection in the module walkthrough lists 7 files. It does not mention `compiler.py` (which is described in POC README as the entry point for generating artifacts), `CHELATEDAI_integration_demo.py`, `validation_config.py`, or any of the Phase 1–7 files. The walkthrough appears correct for the original 2026-03-06 state of the subsystem but not for the current state.
Recommendation: Extend the module walkthrough to include a brief mention of the Phase 1–7 substrate files, or add a pointer to the POC README for the complete inventory.

**F-66 — Missing quick-start example for the disk-first CPU substrate**
File: `README.md`, `docs/README.md`
Details: The README quick start covers: install, optional Ollama, run tests, run representative research entrypoints. The disk-first CPU substrate (which is now a promoted research baseline) has no entry in the quick start section. A researcher interested in the CPU inference path must discover the POC README separately and then navigate through it to find the relevant benchmark entry points.
Recommendation: Add a step "5. Run disk-first CPU substrate benchmarks" to the README quick start, with example commands for `storage_substrate_benchmark.py`, `cpu_inference_benchmark.py`, and `repo_graph_memory_benchmark.py`.

**F-67 — docs/hybrid-distillation-research.md and docs/distillation-experiment-protocol.md are not linked from INDEX.md**
File: `docs/INDEX.md`; files: `docs/hybrid-distillation-research.md`, `docs/distillation-experiment-protocol.md`
Details: These files exist in the docs directory and are cross-referenced from the roadmap audit document, but they are not listed in INDEX.md. They are part of the research support documentation for the distillation track and would be valuable to researchers following that thread.
Recommendation: Add them to the Analysis or Session Logs/Research section of INDEX.md.

**F-68 — docs/weight-refinement-campaign-results-2026-03-06-session28.md is in INDEX.md but lacks session context**
File: `docs/INDEX.md` line 21
Details: The weight refinement campaign results document is correctly listed in INDEX.md. However, the description reads only "Durable summary of the partial bounded campaign, recovered findings, and promotion guidance." There is no indication that this document describes work from Session 28 specifically, or that the campaign was partial due to infrastructure issues. A reader expecting complete campaign results may be confused.
Recommendation: Update the description to note "partial campaign due to benchmark infrastructure issues; see Session 28 log for context."

---

## Challenge Phase Log

**Challenge 1 (Devil's Advocate against F-02, F-03): "The CHANGELOG being frozen is intentional — the AEP session logs serve as the running changelog. Duplicating session-level change tracking into CHANGELOG.md is redundant."**

Panel response: Partially valid. The session logs are rich and detailed but are buried in the ARCH directory and are not a machine-readable or grep-friendly changelog format. The CHANGELOG.md still presents as current with "Phase 4 - Pending" items that are implemented, which actively misleads readers. The fix is either to formally deprecate CHANGELOG.md with a pointer to the session archive, or to add high-level changelog entries. Doing nothing and leaving contradicting information is the worst option.

**Challenge 2 (Devil's Advocate against F-25): "There is no external contributor base for this repository, so a CONTRIBUTING.md is cargo-cult documentation."**

Panel response: This is a research repository for which the primary contributor is the session agent (Claude Code). CONTRIBUTING.md is less about external contributors and more about encoding the conventions that must be followed by any future coding session — no pytest, ruff clean, unittest discover, Python 3.9 compatibility. These conventions exist in CLAUDE.md but not in any externally accessible document. As the project evolves and potentially attracts collaborators or reviewers, the absence of a contribution guide creates unnecessary friction.

**Challenge 3 (Devil's Advocate against F-07): "The code comments in kalman_lr_scheduler.py and chelation_adapter.py already cite the relevant papers inline. REFERENCES.md doesn't need to duplicate them."**

Panel response: Code comments and a research bibliography serve different audiences and purposes. A reader doing a literature review will consult REFERENCES.md; a reader doing code review will consult the inline comments. The two should stay in sync. Currently REFERENCES.md is 9+ citations behind the inline code references. The citation for LoRA (Hu et al., ICLR 2022) appears in chelation_adapter.py but not in REFERENCES.md, which explicitly claims to be the "formal attribution for research papers."

**Challenge 4 (Devil's Advocate against F-10 and F-04): "The new computational_storage_poc modules are untracked in git (per the git status), meaning they haven't been committed to main yet. It's premature to add them to documentation."**

Panel response: The git status shows many of these files as `??` (untracked), which means they exist locally. However, they are described as complete and promoted in `task_plan.md` and `docs/phase7-promotion-review-2026-03-29.md`. The documentation gap should be addressed when (not after) the code is merged. Pre-writing documentation for modules that are staged for merge is not premature — it is good practice.

**Challenge 5 (Devil's Advocate against F-62): "ADRs add process overhead without clear value in a single-maintainer research repo where the session logs already capture rationale."**

Panel response: ADRs do not need to be heavy. A 10-line file per decision is sufficient. The specific concern is that key architectural decisions (Qdrant over alternatives, Cayley parameterization for Procrustes, scope lock on hardware claims) are currently only reconstructable by reading through session logs in order. The value is not process overhead reduction; it is enabling someone to understand why the current architecture is the way it is without reading 31 session logs.

---

## Dissent Log

**Dissent 1 (Dr. Chen Wei on F-21): "The LLM in a Flash assumption should be an immediate high-severity finding, not a medium. If the paper assumption is wrong, every recommendation in that feasibility document is invalid. The document should be withdrawn or restructured until the paper is confirmed."**

Status: Maintained as MEDIUM. The document is explicitly conditional and contains accurate repo analysis regardless of the paper assumption. The risk is noted. However, since this is a research document rather than a product commitment, and since the repo-side analysis sections are paper-independent, the severity remains MEDIUM rather than HIGH.

**Dissent 2 (Ivan Petrov on F-50, CODEOWNERS): "Single-owner CODEOWNERS should be HIGH, not LOW. It creates a single point of failure for PR approvals and could block CI merges, which has already happened repeatedly per the git workflow notes in CLAUDE.md."**

Status: Noted. The CODEOWNERS content itself is not the problem — it correctly reflects the repo's single-maintainer structure. The merge-blocking issues documented in CLAUDE.md are a GitHub branch protection policy issue, not a CODEOWNERS content issue. Severity remains LOW.

**Dissent 3 (Marco Pereira on F-14): "The absent module docstring for antigravity_engine.py should be CRITICAL, not HIGH. This is the primary entry point. Every researcher's first action will be to read this file. No docstring is the worst possible onboarding experience."**

Status: Elevated to HIGH (from originally contemplated MEDIUM). Marco's argument is valid that this is the most-read file. However, the module does have substantial inline docstrings and comments within the `__init__` and key methods. The absence of a module-level docstring is significant but does not rise to CRITICAL given the presence of compensating inline documentation.

---

## Feasibility Assessment

| Finding | Effort | Impact | Priority | Owner |
|---------|--------|--------|----------|-------|
| F-01 README status update | S (1) | Critical | Immediate | Any session |
| F-02 CHANGELOG decision | S (1) | Critical | Immediate | Any session |
| F-03 Remove stale CHANGELOG items | S (1) | Critical | Immediate | Any session |
| F-04 CLAUDE.md disk-first modules | M (3) | High | Session start | Any session |
| F-05 REFERENCES.md MRL arXiv ID | S (1) | High | Immediate | Any session |
| F-06 Remove duplicate reference | S (1) | High | Immediate | Any session |
| F-07 REFERENCES.md new citations | M (3) | High | Next session | Any session |
| F-08 INDEX.md session log table | M (3) | High | Next session | Any session |
| F-09 INDEX.md broken links | S (1) | High | Immediate | Any session |
| F-10 MODULE_GUIDE.md disk-first | M (3) | High | Next session | Any session |
| F-11 CLAUDE.md test count | S (1) | High | After merge | Any session |
| F-12 CLAUDE.md test file list | M (3) | Medium | Next session | Any session |
| F-13 CLAUDE.md wrong REFERENCES path | S (1) | High | Immediate | Any session |
| F-14 antigravity_engine.py docstring | S (1) | High | Next session | Any session |
| F-15 POC module docstrings (4 files) | S (1) | High | Before merge | Any session |
| F-16 RESEARCH_TRACKS.md Track 7 | S (1) | Medium | Next session | Any session |
| F-17 phase-planning.md template | S (1) | Medium | Next session | Any session |
| F-18 README link to superseded doc | S (1) | Medium | Immediate | Any session |
| F-19 INDEX.md Analysis section | S (1) | Medium | Next session | Any session |
| F-20 REFERENCES.md spectral citation | S (1) | Medium | Next session | Any session |
| F-21 Feasibility paper assumption | S (1) | Medium | Immediate | Any session |
| F-22 TECHNICAL_ANALYSIS.md banner | S (1) | Medium | Next session | Any session |
| F-23 Sessions 29/30 log gap | M (3) | Medium | Retrospective | Any session |
| F-24 Domain glossary | M (3) | Medium | Next session | Any session |
| F-25 CONTRIBUTING.md | S (1) | Medium | Next session | Any session |
| F-26 README roadmap link | S (1) | Medium | Immediate | Any session |
| F-27 sedimentation.py in MODULE_GUIDE | S (1) | Low | Next session | Any session |
| F-28 Missing utils in MODULE_GUIDE | S (1) | Low | Next session | Any session |
| F-29 CLAUDE.md convergence_monitor | S (1) | Low | Next session | Any session |
| F-30 SYSTEM_BLUEPRINT.md disk-first | M (3) | Medium | Next session | Any session |
| F-31 docs/README.md disk-first paths | S (1) | Medium | Next session | Any session |
| F-32 CLAUDE.md Key APIs disk-first | M (3) | Medium | Next session | Any session |
| F-33–35 Stale doc banners | S (1) | Low | Next session | Any session |
| F-36 next-session.md update | S (1) | Medium | Each session | Any session |
| F-37 COMP_STORAGE_DRIVE_NODES update | M (3) | Medium | Next session | Any session |
| F-38 POC README restructure | M (3) | Low | Future | Any session |
| F-39 Topology bond glossary | S (1) | Low | Next session | Any session |
| F-40 Troubleshooting guide | M (3) | Medium | Future | Any session |
| F-41 Hardware runbook review date | S (1) | Low | Future | Any session |
| F-42 Python 3.9 note extension | S (1) | Low | Next session | Any session |
| F-43–44 Dead link cleanup | S (1) | Low | Next session | Any session |
| F-45 agentic-review-framework.md | M (3) | Medium | Future | Any session |
| F-46–47 Minor cleanups | S (1) | Low | Backlog | Any session |
| F-48–55 Low-severity items | S (1) | Low | Backlog | Any session |
| F-56–68 Low-severity items | S (1) | Low | Backlog | Any session |
| F-62 ADRs | L (5) | Medium | Future | Any session |

**Effort legend:** S = small (< 30 min), M = medium (30 min – 2 hr), L = large (> 2 hr)

---

## Documentation Improvement Roadmap

### Tier 1: Immediate (can be done in the next 30 minutes)

1. **F-09** — Remove broken links from INDEX.md (5 broken links)
2. **F-05 + F-06** — Correct MRL arXiv ID in REFERENCES.md; remove duplicate reference
3. **F-13** — Fix REFERENCES.md path in CLAUDE.md (`docs/REFERENCES.md` → `REFERENCES.md`)
4. **F-18 + F-26** — Update README.md "Current Research Status" to link to revised roadmap
5. **F-03** — Strike "Phase 4 - Pending" checklist from CHANGELOG.md or mark as historical

### Tier 2: Next Session (1–2 hours total)

6. **F-01** — Expand README.md Current Research Status to cover the disk-first program through Phase 7 promotion
7. **F-02** — Deprecate CHANGELOG.md with a header notice and pointer to session logs; alternatively add high-level entries for sessions 3–31
8. **F-04** — Add disk-first substrate module list to CLAUDE.md architecture section
9. **F-10** — Add "Disk-First CPU and Retrieval Substrate" section to MODULE_GUIDE.md
10. **F-11 + F-12** — Update CLAUDE.md test count and expand representative test file list
11. **F-14** — Add module-level docstring to antigravity_engine.py; update `__init__` docstring
12. **F-15** — Add module-level docstrings to the four undocumented POC files
13. **F-08** — Extend INDEX.md session log table through Session 31
14. **F-16** — Update RESEARCH_TRACKS.md Track 7 current state
15. **F-07** — Add missing citations to REFERENCES.md (LoRA, InfoNCE, GAM-RAG, REAP, LLM in Flash, T-MAC)

### Tier 3: Near-Term (addressable across 2–3 sessions)

16. **F-30** — Update SYSTEM_BLUEPRINT.md to describe 4 major surfaces
17. **F-31** — Update docs/README.md with disk-first reading path
18. **F-32** — Add disk-first APIs to CLAUDE.md Key APIs section
19. **F-24 + F-39** — Create `docs/GLOSSARY.md` covering domain and topology terms
20. **F-25** — Create minimal `CONTRIBUTING.md`
21. **F-22 + F-33–35** — Add prominent historical banners to TECHNICAL_ANALYSIS.md, COMPLETION_SUMMARY.md, PR_DESCRIPTION.md, REFACTORING_PLAN.md
22. **F-37** — Update COMPUTATIONAL_STORAGE_DRIVE_NODES.md with Phase 1–7 section
23. **F-36** — Establish a practice of updating next-session.md at session start/end
24. **F-19** — Expand INDEX.md Analysis section with disk-first research docs
25. **F-64** — Add Session 31 research documents to INDEX.md

### Tier 4: Future Backlog

26. **F-62** — Create `docs/decisions/` with lightweight ADRs for key architectural choices
27. **F-40** — Create `docs/TROUBLESHOOTING.md`
28. **F-38** — Restructure `computational_storage_poc/README.md` for logical flow
29. **F-23** — Write retrospective session logs for Sessions 29 and 30
30. **F-45** — Create `docs/agentic-review-framework.md` or update workflow references
31. **F-58** — Add CI status badge to README.md
32. **F-41** — Add last-reviewed date to hardware evidence runbook
33. **F-63** — Document the impl-N to session-NN naming transition in the AEP README
34. **F-66** — Add disk-first quick-start commands to README.md

---

## Summary Statistics

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High | 12 |
| Medium | 26 |
| Low | 27 |
| **Total** | **68** |

The critical and high findings center on four themes: (1) stale status documentation that misrepresents the current project state, (2) the entire disk-first CPU substrate being invisible to primary navigation documents, (3) citation inaccuracies in REFERENCES.md, and (4) broken links in the primary index. All Tier 1 items require only small edits and can be addressed in under 30 minutes combined.

---

*Report generated: 2026-04-04*
*Files reviewed: README.md, CLAUDE.md, CHANGELOG.md, REFERENCES.md, TECHNICAL_ANALYSIS.md, COMPLETION_SUMMARY.md, PR_DESCRIPTION.md, REFACTORING_PLAN.md, SECURITY.md, CODEOWNERS, pyproject.toml, docs/INDEX.md, docs/README.md, docs/SYSTEM_BLUEPRINT.md, docs/MODULE_GUIDE.md, docs/RESEARCH_TRACKS.md, docs/COMPUTATIONAL_STORAGE_DRIVE_NODES.md, docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md, docs/disk-resident-llm-feasibility-2026-03-28.md, docs/disk-resident-llm-addendum-reap-turboquant-2026-03-28.md, docs/phase7-promotion-review-2026-03-29.md, docs/revised-roadmap-disk-first-program-2026-03-28.md, docs/computational-storage-hardware-evidence-runbook.md, computational_storage_poc/README.md, docs/ARCH AGENTIC ENGINEERING AND PLANNING/README.md, docs/ARCH AGENTIC ENGINEERING AND PLANNING/orchestrator-briefing.md, docs/ARCH AGENTIC ENGINEERING AND PLANNING/workflow.md, docs/ARCH AGENTIC ENGINEERING AND PLANNING/glossary.md, docs/ARCH AGENTIC ENGINEERING AND PLANNING/phase-planning.md, docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md, antigravity_engine.py (first 100 lines), chelation_adapter.py (full docstring review), config.py (first 100 lines), and module-level docstrings for all 20+ Python source modules.*
