# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Brutal Honesty Convention (load-bearing — read first)

See `docs/conventions/brutal-honesty-rulebook.md` (v3.3 — v3.2 executable PR gates plus L13 soft-prose-claimed-as-mechanical and the v3.3 schema/prose/artifact drift validator. v3.2 baseline: Tier B independence enforcement via `BHS_*_AGENT` lines, severity caps on Tier B scores, `DEFERRED_SCOPE:` tracking, quantitative cycle definition, override structural barrier at BLOCKED, executable validators in `scripts/`).

**Premise**: Assume every implementation/completion claim is false until independently proven by runtime evidence. Tests existing, routes existing, docs saying complete, PR text saying complete, agent claims, and even self-attested brutal-honesty sections are NOT evidence. Evidence is: command output from the production code path, real UI/API behavior, persistence/state mutation, artifact/replay/checkpoint surviving a fresh checkout, or an independent reviewer who tried to disprove and failed.

**Five hard rules**:

1. **Evidence rule.** Every PR with a "complete" claim includes `EVIDENCE:` and `SMOKE:` lines pointing at runtime evidence — not tests.
2. **Visible means verified.** Incomplete code may live in the repo; it may NOT be presented as a working feature (no UI surfacing, no API doc, no release-note implication, no roadmap tick).
3. **Mandatory PR brutal-honesty section.** Every PR body ends with a `## Brutal Honesty` section disclosing stubs (L1), escape conditionals (L2), mocks in production paths (L3), partial implementations (L4), untested production paths (L5/L8/L12), broad-catch swallowing (L11), and any other §1 instance with `file:line`. Empty answers must be justified, not omitted (template in §4 of the rulebook).
4. **Adversarial cross-agent review.** Implementation slices are done by a fresh sub-agent; a second fresh sub-agent is given the diff + brutal-honesty section + EVIDENCE + smoke command + this rulebook, and is asked to **try to disprove** completion. Not verify — *disprove*.
5. **One deterministic smoke path per repo.** `scripts/smoke.sh` (or equivalent) is the release gate. **Two honest tiers per §1 Rule 5**: Floor (import + surface-check through production code paths — the v3.2 minimum) is acceptable IF the PR body's SMOKE: line names which tier was run AND any ceiling-tier gap appears as a Carried Debt entry. Ceiling (true end-to-end against a real fixture) is the target. Claiming "smoke passed" at ceiling-tier when only floor-tier was run is itself L4 partial-as-complete. If smoke fails, the release is not ready regardless of test count.

**Trigger phrases** (use these instead of "is this done?"): *"Be brutally honest." / "What did you fake to get here?" / "If I ran this on a fresh checkout right now, what would actually fail?" / "Tell me the smallest concrete thing that does NOT work yet." / "Show me the runtime evidence, not the test." / "If I disable this feature behind a flag, what visible behavior changes?"*

**The Remediation Loop (meta-component, §6 of the rulebook; v3.1 closes the carry-forward loophole)** — applies to this repo:
- **Tier A (per-PR)**: 5 iterations max. Implementer self-scores `BHS_SELF_DRAFT` after each. If self-score < 100 at iteration 5, the PR is **withdrawn or scope-reduced** — it does NOT ship at <100 hoping for carry-forward. Same gap surviving 2 iterations → escalate, don't loop more.
- **Tier B (pre-merge — THE OFFICIAL SCORER)**: one fresh adversarial agent assigns `BHS_TIER_B` independently. `BHS_OFFICIAL = min(BHS_SELF_DRAFT, BHS_TIER_B)`. **Hard merge gate: only PRs with `BHS_OFFICIAL = 100` may merge.** No "ships with caveats" path. Self-vs-Tier-B gap > 5 → automatic L4 score-gaming disclosure.
- **Tier C (cross-PR / session boundary)**: aggregate disclosures into `docs/next-session.md` "Carried Debt" with TTL = 1 cycle. Items not cleared in the next cycle flip the **block flag** to BLOCKED — the cycle after that forbids ALL new feature work until Carried Debt = 0. **No two-cycle slop.** Block flag is automatic from the math.
- **BHS scale (only 100 ships)**: 100 = merges · 90–99 = does not merge, back to Tier A · 70–89 = does not merge, reduce scope · 50–69 = does not merge, scaffold-only allowed · 0–49 = draft, not a PR.
- **Operator override**: rulebook can't prevent the operator from typing the merge command at <100; it makes the override visible (`OPERATOR_OVERRIDE:` line) and force-creates a top-priority carried-debt entry for the next cycle.
- **PR body required lines**: `BHS_SELF_DRAFT`, `BHS_SELF_DRAFT_AGENT`, `BHS_TIER_B`, `BHS_TIER_B_AGENT`, `BHS_TIER_B_SEVERITY`, `BHS_OFFICIAL`, `CARRY_FORWARD`, `DEFERRED_SCOPE`, `LOOP_ITERATIONS`, `OPERATOR_OVERRIDE` (per §4 template).
- **Why this is in CLAUDE.md, not in a skill or MCP server**: skills can fail to load, agents can spin up without MCP context, the convention can't. This file is auto-loaded every session — the convention is the meta-component.

**Lie taxonomy L1–L13** is in §1 of the rulebook; quote by number when calling out a failure. Speculating is itself a lie — *"I don't know"* is the correct answer when there is no evidence.

## Session Rules (enforced every session, no exceptions)

### Rule 1 — Full Implementation Only
Every slice, every phase, every PR must be **fully implemented**. No scaffolding, no stubs, no `pass`-body placeholders, no `TODO: implement` markers left in committed code. If a feature is too large for one PR, split it into genuinely shippable sub-slices — but each sub-slice must itself be complete and working, not a placeholder for future work.

### Rule 2 — PR on Completion
Every unit of completed work must land in a PR **before moving to the next task**. Do not accumulate dirty branches, uncommitted changes, or local-only work across sessions. The rule is: finish a slice → open PR → get it merged (or at minimum opened and green) → then start the next slice. Stale branches and deferred cleanup are not acceptable.

### Rule 3 — No Placeholder Data or Fake Metrics
Dashboards, frontends, and reporting surfaces must be wired to **real data pipelines**. Placeholder values, hard-coded demo numbers, mock metrics, and fake responses are forbidden in committed code. Every data field displayed to an operator must trace to an actual source artifact, live computation, or explicitly documented empty-state. The wiring from backend to frontend must be obvious and verifiable.

## EGV Research Integrity Rules

- A sealed freezer with zero eligible rows is a valid terminal research result.
  Report `NO_ADMISSIBLE_TRAINING_SET`; do not fabricate rows, an adapter, or a
  trained-versus-base comparison. Trained arms remain `UNEVALUATED`.
- Production training must reject an ineligible or undersized dataset before
  private staging, evaluator construction, CUDA checks, model loading, or output
  creation. Tests must prove the ordering, not merely the final exception.
- Keep private campaign seals distinct from public reproducer fields. Public
  reports may summarize privately verified facts only when they explicitly say
  which details are not independently reproduced by the public artifact.
- Public evidence must use canonical closed-schema records with exact byte
  pins, fail-closed semantic validation, and scans excluding prompts, generated
  source, task identities, credentials, endpoints, paths, host labels, and
  deployment topology.
- A local or dual-host implementation test is not evidence of live cross-host
  trainer/evaluator transport. Claim that boundary only after the exact route is
  executed and its receipt/artifact chain is independently verified.

## What This Project Is

ChelatedAI is a research prototype for adaptive vector search with self-correcting embeddings. It detects "semantic collapse" in RAG systems (where unrelated concepts get similar embeddings) and fixes it through dynamic dimension masking and neural adaptation.

## Dependencies & Setup

Dependencies are managed via `pyproject.toml` (PEP 621) and `requirements.txt`:

```bash
pip install -r requirements.txt
# Or install as editable package:
pip install -e .
```

Optional: Ollama for Docker-based embeddings (`ollama:nomic-embed-text` model prefix).

## CI Pipeline

GitHub Actions workflow at `.github/workflows/test.yml`:
- **Lint job:** `ruff check .` on Python 3.11
- **Test job:** `unittest discover` across Python 3.9, 3.10, 3.11, 3.12 matrix

GitHub Actions workflow at `.github/workflows/build_firmware.yml`:
- **Firmware build job:** builds the RP2040/TinyUSB computational-storage firmware when `computational_storage_poc/firmware/**` changes and uploads UF2/ELF/BIN artifacts

GitHub Actions workflow at `.github/workflows/test.yml` also includes:
- **Computational-storage emulation job:** validates emulator semantics on hosted CI without privileged FUSE

## Running Tests

All tests use Python `unittest` (not pytest). CI does **not** install `pytest`, so do not add `pytest` imports or pytest-only fixtures to `test_*.py`. Run via CI or locally:

```bash
# Run a single test file
python test_unit_core.py

# Run a specific test class or method
python -m unittest test_unit_core.TestChelationAdapter.test_adapter_forward_pass

# Run all test files (bash glob)
for f in test_*.py; do python "$f"; done

# Discover and run all tests
python -m unittest discover -s . -p "test_*.py" -v
```

**Representative test files (`1082` tests passing on `main` as of 2026-03-12):**
- `test_unit_core.py` - Core adapter variants, BoundedAdapter, DimensionProjection training
- `test_noise_injection.py` - Noise injection validation under `unittest`
- `test_online_updater.py`, `test_dimension_mask_predictor.py`, `test_stability_tracker.py`
- `test_benchmark_beir.py`, `test_benchmark_comparative.py`, `test_dashboard_server.py`
- `test_sedimentation_loss.py` - InfoNCE, hybrid loss, hard negative miner, factory
- `test_kalman_lr.py` - Kalman-gain adaptive LR, variance behavior, clamping, engine integration
- `test_computational_storage_poc.py` - block-graph parity, latency invariants, and real-data storage round-trip validation
- `test_computational_storage_payload.py` - deterministic trigger-sector payload, host decoding, and virtual-disk interception validation
- `test_computational_storage_hardware_evidence.py` - deterministic evidence capture and Windows raw-device path handling
- `test_computational_storage_emulation.py` - dependency-light emulator parity and file-image validation
- `test_cross_lingual_distillation.py`, `test_language_detector.py`
- `test_topology_analyzer.py`, `test_isomer_detector.py`, `test_structural_health_report.py`
- `test_teacher_distillation.py`, `test_teacher_weight_scheduler.py`, `test_aep_orchestrator.py`

**Environment-dependent tests:** `test_antigravity_engine.py`, `test_adaptive_threshold.py`, and `test_memory_optimization.py` require full `torch` + `sentence-transformers` installed. They pass in CI but may fail locally without those dependencies.

## Architecture

### Flat file layout

All `.py` files live at the project root. No packages, no `__init__.py`. Imports are direct module references (e.g., `from antigravity_engine import AntigravityEngine`).

### Core module dependency graph

```
AntigravityEngine (antigravity_engine.py)  <- central entry point
|-- embedding_backend.py       # F-045: "ollama:model" -> HTTP, else -> SentenceTransformer
|-- vector_store.py            # F-044: QdrantVectorStore abstraction
|-- chelation_adapter.py       # 4 adapter types via create_adapter() factory + BoundedAdapter wrapper
|-- config.py                  # ChelationConfig with presets + validation
|-- chelation_logger.py        # get_logger() -> JSON structured logging
|-- teacher_distillation.py    # Offline/hybrid modes + DimensionProjection + EnsembleTeacherHelper
|-- teacher_weight_scheduler.py # 5 schedule types: constant/linear/cosine/step/adaptive
|-- sedimentation_loss.py      # InfoNCE, hybrid, hard-negative mining loss functions
|-- kalman_lr_scheduler.py     # Kalman-gain adaptive LR for sedimentation training
|-- sedimentation_trainer.py   # Shared homeostatic target logic
|-- checkpoint_manager.py      # SafeTrainingContext with SHA256 verification
|-- convergence_monitor.py     # Phase 1: patience-based early stopping
|-- online_updater.py          # Phase 3: inference-time micro-updates
|-- dimension_mask_predictor.py # Phase 4: learned dimension masking
|-- embedding_quality.py       # Phase 4: per-doc quality with decay weighting
`-- stability_tracker.py       # Phase 5: structural health diagnostics

RecursiveRetrievalEngine (recursive_decomposer.py)
|-- Uses AntigravityEngine for retrieval
|-- sedimentation.py           # HierarchicalSedimentationEngine
`-- config.py, chelation_logger.py, checkpoint_manager.py

AEPOrchestrator (aep_orchestrator.py)  <- 7-phase agentic remediation workflow
`-- chelation_logger.py

Dashboard & Sweeping (dashboard_server.py, run_sweep.py, run_large_sweep.py)
`-- Serves live metrics to localhost:8080 and manages parameter grid search.

Evaluation & Analysis Modules
|-- benchmark_beir.py                # Multi-dataset BEIR benchmarking + report generation
|-- cross_lingual_distillation.py    # Language-aware teacher routing for distillation
|-- language_detector.py             # Lightweight language detection with caching/fallbacks
|-- topology_analyzer.py             # Topology snapshots, bond matrices, cluster connectivity
`-- isomer_detector.py               # Query-result isomer detection built on topology signals
```

### Key design patterns

- **Adapter factory:** `create_adapter("mlp"|"procrustes"|"low_rank", input_dim, bounded=False)` returns one of three adapter types, optionally wrapped in `BoundedAdapter` for INT8-safe corrections. All initialize near-identity (small weight std 0.001) to preserve base model quality.
- **Config presets:** `ChelationConfig.get_preset(name, type)` supports `chelation`, `adapter`, `convergence`, `adapter_type`, `rlm`, `sedimentation`, `sedimentation_tuned`, `ensemble`, `cross_lingual`, `teacher_weight_schedule`, `teacher_encoding`, `online_update`, `beir`, `topology`, `isomer`, `bounded_adapter`, `sedimentation_loss`, and `kalman_lr`.
- **Sedimentation loss:** `engine.set_sedimentation_loss("mse"|"infonce"|"hybrid")` switches loss function. InfoNCE uses batch contrastive alignment; hybrid combines MSE + InfoNCE.
- **Kalman-gain adaptive LR:** `engine.enable_kalman_lr(process_noise, min_lr_ratio, max_lr_ratio)` modulates learning rate based on loss variance — high variance lowers LR, low variance raises it.
- **Noise Injection:** Dynamically scaled noise injection during sedimentation training.
- **Embedding backend routing:** Model names prefixed with `ollama:` use the HTTP API; all others use local SentenceTransformers.
- **Teacher distillation:** `DimensionProjection` for teacher-student dim mismatches, `EnsembleTeacherHelper` for multi-teacher weighted averaging, `TeacherWeightScheduler` for 5 dynamic schedule types.
- **AEP data model:** `Finding` objects with `Severity` (CRITICAL/HIGH/MEDIUM/LOW), `FindingStatus`, and `EffortSize` (S=1, M=3, L=5). Tiered remediation processes Critical->High->Medium->Low with no skipping.
- **Structural health reporting:** `antigravity_engine.py` now uses config-driven thresholds for persistent collapse, oscillation, topology cohesion, and isomer drift.

## Test Conventions

- **Mock logger:** `patch('module_name.get_logger')` returning `MagicMock()` -- used in nearly every test file.
- **In-memory Qdrant:** `qdrant_location=":memory:"` for isolation.
- **Local model for tests:** `model_name="all-MiniLM-L6-v2"` (requires sentence-transformers).
- **Temp files:** Tests use `tempfile` for filesystem isolation with cleanup in `tearDown`.
- **Sklearn usage:** Keep core runtime code on numpy/torch. `scikit-learn` is only acceptable in isolated validation flows such as the computational-storage digits test track.
- **Python 3.9 CI compatibility:** If a module imported by tests uses `X | None` annotations, add `from __future__ import annotations` or use `Optional[...]`.

## Git Workflow Notes

- The repository branch policy may still show PRs as blocked even after all required checks are green. Session 23 required admin merges for `#80`, `#83`, and `#82`.
- On Windows, update long GitHub PR bodies from a real UTF-8 body file and then
  re-fetch and validate the live body. A PowerShell string pipeline can submit
  an empty body even when `gh pr edit` reports success.
- When a stacked base branch was deleted after merge, fetch `main` by itself
  before rebasing. Including the deleted branch in the same fetch can leave the
  local `origin/main` stale even though later commands continue.
- Linux descendant-process tests inside an ephemeral container need an init
  reaper so exited children do not remain visible as zombies. Mount the Docker
  integrity inputs only for the reviewed nested-sandbox canary; keep source
  read-only and do not classify a harness-misconfigured first run as a product
  defect.
- Session 27 required admin merges for `#90`, `#91`, `#92`, and `#93` even after all required checks passed.
- `gh pr merge` can fail if a local worktree is holding `main`. Before merging stacked PRs, remove/prune merged worktrees or switch them off `main`.
- The computational-storage split is complete on `main` as of 2026-03-06: `#86` landed the validation foundation, `#87` landed the payload transport path, and `#88` landed the session-wrap docs.
- Session 26 follow-up PRs `#90` (hardware evidence capture tool), `#91` (emulation CI), `#92` (transport scope lock), and `#93` (retention policy) are merged on `main` as of 2026-03-06. The dated retention review was executed and merged as PR `#108` on 2026-04-24; the remaining computational-storage follow-through is real hardware evidence capture on actual RP2040 hardware.
- Session 31 merged PRs `#96`–`#103` (weight refinement, docs cleanup, 4 features, session wrap). The adapter checkpoint contamination risk from Session 28 is resolved (isolated checkpoints per config).
- Session 32 ended with an explicit no-promotion outcome for preset/default changes. Do not treat `mlp` + `teacher_weight=0.3` as promotable until a second independent run plus multi-task/BEIR transfer checks confirm it. See `docs/weight-refinement-campaign-results-2026-04-25-session32.md`.
- Do not revive the stale computational-storage PR `#84` or the old `feat/session22-online-correction` branch line. If historical comparison is needed, use the local `backup/retired-*` refs instead.
- If no RP2040 device is attached, do not fabricate hardware evidence. Use `computational_storage_poc/capture_hardware_evidence.py` once actual hardware is available.
- Explicit Windows raw-device paths like `\\.\PhysicalDrive2` are valid inputs to `usb_host_inference.py` and `capture_hardware_evidence.py`; do not rewrite them into a second `PhysicalDrive` prefix.
- Do not treat unrelated removable USB storage as RP2040 evidence. Session 27's local probe only found a SanDisk removable drive, which was not used as a proxy.
- If a resumed benchmark campaign is no longer the active task, stop the live process instead of leaving it consuming CPU in the background.
- `ruff check` does not validate GitHub Actions YAML. Keep workflow-file review separate from Python lint.
- Local `git status` may show `?? .claude/`; that directory holds local worktree metadata and retired-branch artifacts and is not, by itself, a product-code diff.

**2026-05-15 Desktop Reconciliation:**
- Desktop machine (previously 23 commits behind) reconciled with laptop work on branch `reconciliation/2026-05-desktop-sync`.
- Merged `origin/main` (BHS v3.3 tooling + full Model-Scope + heavy remediation).
- All major laptop 10-phase planning artifacts (`FINAL_PLAN.md` family + `panel-analysis/`) landed into `docs/ARCH AGENTIC ENGINEERING AND PLANNING/planning/2026-05-reconciliation/`.
- New computational storage POC code (packed/CPU/sparse/repo-graph/MoE/REAP) moved to `feat/post-merge-comp-storage-substrate`.
- Real BHS v3.3 scripts now live in `scripts/` (`validate_pr_brutal_honesty.py`, `validate_v33_schema_drift.py`, `smoke_pipeline.py`, etc.), though full wiring into AEP is still in progress.
- Reconciliation reimplementation backlog created and being actively maintained.

## Reference Material

- `rlm_reference/` -- Cloned RLM paper implementation (read-only, do not modify)
- `docs/rlm-analysis.md` -- Analysis of RLM source code
- `docs/REFERENCES.md` -- 17 research paper citations
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/` -- Full AEP workflow documentation (60+ files)
- `docs/INDEX.md` -- Documentation index
