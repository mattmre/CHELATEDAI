# FINAL REFINED 10-PHASE IMPROVEMENT PLAN
## ChelatedAI Repository — Panel Consensus Edition

*Synthesized from 3 rounds of 20-expert panel analysis. This is the final refinement.*

---

## Panel Consensus Summary

After 3 iterations of analysis by 20 expert panels, the following themes emerged consistently:

| Theme | Consensus |
|---|---|
| **Security** | ML-specific attack vectors (weight poisoning, prompt injection) are critical; `__getattr__` bypass in vector_store.py is the #1 architectural flaw |
| **Testing** | Coverage gaps are real but fuzz testing is overkill; integration tests need shared fixtures first; research validity testing is missing |
| **Architecture** | Package reorganization is necessary but must preserve backward compatibility; dead code removal first, then restructuring |
| **Code Quality** | 7 custom exceptions is overkill (reduce to 4); mypy + ruff + isort are essential; MkDocs not Sphinx; magic numbers → config |
| **Training** | Optuna overkill for research prototype; JSONL not MLflow for experiment tracking; online safety → checkpoint-restore only |
| **Benchmarks** | JSONL not SQLite; pandas not A/B framework for stats; golden regression suite is Day 1 priority |
| **Documentation** | MkDocs with mkdocstrings; ARCH/ directory should be archived not reorganized; CONTRIBUTING.md is essential |
| **Computational Storage** | Firmware docs are genuinely useful; local build docs missing; hardware evidence capture only if hardware exists |
| **Performance** | ONNX export + embedding caching = high ROI; profiling/baselines essential first step; async migration too costly |
| **Monitoring** | Phase 10 to be eliminated entirely — existing dashboard_server.py, stability_tracker.py, chelation_logger.py provide all needed monitoring |

---

## Expert Recommendations

### 1. Security Architect — Supply Chain, Input Validation, Attack Surface
**Recommendation:** Fix `__getattr__` bypass in `vector_store.py` immediately (P0). Add input validation for model weights and config files. Document attack surface.
**Plan modification:** Elevates security items from Phase 3 to a standalone Phase 1 with P0 priority.
**Priority:** P0
**Effort:** 2-3 days (critical path)

### 2. Test Automation Engineer — Coverage, Fixtures, Integration Testing
**Recommendation:** Create shared fixtures first (`conftest.py`-equivalent helpers). Add integration test layer. Skip complex fuzz testing — use parameterized tests with edge-case payloads.
**Plan modification:** Consolidates testing work into Phase 2 with fixture creation as Phase 2 prerequisite.
**Priority:** P1
**Effort:** 5-7 days

### 3. Software Architect — Package Design, Decomposition, Migration
**Recommendation:** Phased migration: (1) `from __future__ import annotations` everywhere, (2) dead code removal, (3) flat → package with `__init__.py` re-exports preserving imports, (4) `antigravity_engine.py` decomposition via composition. Never break `import antigravity_engine`.
**Plan modification:** Makes package reorganization non-breaking via re-export pattern. Phase 3 becomes 4 sub-phases.
**Priority:** P1
**Effort:** 10-14 days

### 4. Python Typing Expert — Type Annotations, Mypy, Python 3.9 Compat
**Recommendation:** Add `from __future__ import annotations` to all files (enables PEP 604 style unions in 3.9). Run mypy with `--strict` on new code only (relax for legacy). Replace `X | None` with `Optional[X]` in public APIs.
**Plan modification:** Adds typing pass before package restructuring. MyPy as CI gate for new code only.
**Priority:** P1
**Effort:** 3-5 days

### 5. Training Systems Engineer — Loss Functions, LR Schedules, HPO
**Recommendation:** Extend `run_sweep.py` for hyperparameter search instead of adding Optuna. ExperimentRegistry → JSONL files (already project convention). Online safety = checkpoint-restore (not complex rollback). Calibration validation is a research tangent — defer.
**Plan modification:** Keeps HPO lightweight. Removes MLflow-level complexity. Phase 5 scoped to existing patterns.
**Priority:** P2
**Effort:** 3-4 days

### 6. Benchmark Engineer — BEIR, Statistical Testing, Reproducibility
**Recommendation:** Golden regression suite on Day 1 (P0). Use JSONL for benchmark results (already project convention). Use pandas for basic stats (not scipy for small samples). A/B framework unnecessary.
**Plan modification:** Elevates golden suite to Phase 1. Statistical testing simplified to pandas + descriptive stats.
**Priority:** P0 (golden suite) / P1 (general)
**Effort:** 2 days (golden suite) + 3-4 days (general)

### 7. Documentation Specialist — MkDocs, API Docs, Knowledge Management
**Recommendation:** MkDocs with mkdocstrings for API auto-generation. Archive `docs/ARCH AGENTIC ENGINEERING AND PLANNING/` to `archive/` or `docs/archive/` (don't reorganize). Create CONTRIBUTING.md. Define maintenance model: who updates docs on each PR.
**Plan modification:** MkDocs chosen over Sphinx. ARCH/ archiving is Phase 7 first step. CONTRIBUTING.md is Day 1.
**Priority:** P1
**Effort:** 3-5 days

### 8. Embedded Systems Engineer — Firmware, RP2040, TinyUSB
**Recommendation:** Firmware docs are genuinely useful — keep and enhance. Add local build docs for RP2040 (pio build, dfu-util flashing). Hardware evidence capture only valuable if hardware exists. ADRs overkill — single `docs/COMPUTATIONAL_STORAGE_DESIGN_DECISIONS.md` suffices.
**Plan modification:** Focus on documentation improvements, not firmware changes. Local build docs as Phase 8.
**Priority:** P2
**Effort:** 2-3 days

### 9. Performance Engineer — Profiling, Optimization, Caching
**Recommendation:** Profiling/baselines are essential first step (P1). ONNX export for embedding models = high ROI. Embedding caching to disk = high ROI. Skip async migration (zero user-facing value). Skip GPU quantization (hardware doesn't exist). Skip Prometheus/OpenTelemetry (local tool).
**Plan modification:** Profiling → Phase 9 Step 1. Caching and ONNX → Phase 9 Steps 2-3. Removes all production-grade observability from scope.
**Priority:** P1
**Effort:** 5-7 days

### 10. DevOps/CI Engineer — Workflows, Artifact Management
**Recommendation:** Add mypy to CI for new code. Add ruff rules (I = isort, UP = pyupgrade, B = bugbear). Add golden benchmark guard in CI (fail if golden results drift > 2%). CI workflow YAML review separate from lint.
**Plan modification:** CI enhancements distributed across Phases 1, 6, 9. Golden benchmark CI guard is P0.
**Priority:** P1
**Effort:** 2-3 days

### 11. ML Research Scientist — Research Validity, Convergence, Scientific Rigor
**Recommendation:** Add convergence validation tests (does adaptation actually improve embeddings on known collapse scenarios?). Add drift detection tests. Ensure random seeds are controlled in all experiments. Document reproducibility protocol.
**Plan modification:** Adds research validity testing to Phase 2 (testing) as research-specific test suite.
**Priority:** P1
**Effort:** 3-5 days

### 12. Code Quality Engineer — Linting, Refactoring, Code Smells
**Recommendation:** Consolidate 7 custom exceptions to 4: `ChelatedAIError` (base), `AdaptationError`, `EmbeddingError`, `ConfigError`. Add mypy to CI. Expand ruff rules. Use isort. Remove magic numbers → config.py (already partially done).
**Plan modification:** Exception consolidation is Phase 4. Magic numbers → config.py is Phase 4 Step 2.
**Priority:** P1
**Effort:** 2-3 days

### 13. Developer Experience Lead — Onboarding, Ergonomics, Tooling
**Recommendation:** CONTRIBUTING.md with setup instructions. `.devcontainer/` or `docker-compose.yml` for one-command local environment. Pre-commit hooks (ruff, isort, mypy). Simple `scripts/setup.sh` / `scripts/setup.bat`.
**Plan modification:** CONTRIBUTING.md → Phase 7. Pre-commit hooks → Phase 10 (Production Readiness Lite). Dev container → Phase 10.
**Priority:** P1
**Effort:** 3-4 days

### 14. Data Engineer — Data Pipelines, Storage, Caching
**Recommendation:** Embedding cache as JSONL with SQLite-free design (already project convention). Checkpoint integrity via SHA256 (already in `checkpoint_manager.py`). Experiment results → JSONL (already pattern). No new storage backends needed.
**Plan modification:** Confirms existing patterns are correct. No new data infrastructure. Phase 9 caching task uses JSONL.
**Priority:** P2
**Effort:** 1-2 days (caching implementation)

### 15. QA Engineer — Quality Gates, Regression, Golden Tests
**Recommendation:** Golden test suite as regression guard (P0). Coverage threshold at 70% (not 90% — research code is exploratory). Per-module test files map one-to-one with source modules. Integration tests separate from unit tests.
**Plan modification:** Golden suite → Phase 1 (Day 1). Coverage threshold documented in CONTRIBUTING.md.
**Priority:** P0
**Effort:** 2 days (golden suite)

### 16. Research Integrity Auditor — Reproducibility, Bias, Methodology
**Recommendation:** Document seed management protocol. Add "reproducibility checklist" to PR template. Ensure benchmark results include confidence intervals. Add model card for each adapter type.
**Plan modification:** Adds reproducibility checklist as Phase 7 documentation. Model cards → Phase 7.
**Priority:** P2
**Effort:** 2-3 days

### 17. ML Ops Engineer — Experiment Tracking, Model Lifecycle
**Recommendation:** JSONL experiment logs (already project convention). Simple versioning: `experiment_runs/YYYYMMDD_HHMMSS_<name>.jsonl`. No MLflow (overkill). Checkpoint lifecycle: auto-prune checkpoints older than 7 days. No model registry needed.
**Plan modification:** Confirms JSONL approach. Adds checkpoint pruning as Phase 5.
**Priority:** P2
**Effort:** 1-2 days

### 18. Security Researcher — Adversarial ML, Prompt Injection, Data Exfiltration
**Recommendation:** Add input validation for model weights (size, shape, dtype checks). Add prompt injection guard in retrieval path. Document data exfiltration risks in SECURITY.md. Add `--strict-config` flag to reject unknown config keys.
**Plan modification:** Merges with Security Architect recommendations in Phase 1. Prompt injection guard → Phase 1 P1.
**Priority:** P1
**Effort:** 2-3 days

### 19. Systems Reliability Engineer — Observability, Resilience
**Recommendation:** Existing `stability_tracker.py`, `chelation_logger.py`, `dashboard_server.py` provide all monitoring needed. Remove Sentry, runbooks, health checks, alerting, Prometheus from scope (assumes production infrastructure). Add simple health check endpoint to dashboard.
**Plan modification:** **ELIMINATES Phase 10 entirely.** Replaces with "Production Readiness Lite" (see below).
**Priority:** N/A (Phase 10 eliminated)
**Effort:** N/A

### 20. Technical Program Manager — Sequencing, Dependencies, Risk
**Recommendation:** Phase 1 and 6 (golden suite) are parallel Day 1 tasks. Architecture changes (Phase 3) must wait for typing pass (Phase 4a). Performance work (Phase 9) is independent. Testing (Phase 2) and documentation (Phase 7) can run in parallel.
**Plan modification:** Defines parallel workstreams and explicit dependency chain. Adds risk mitigations.
**Priority:** N/A (orchestration role)
**Effort:** N/A

---

## FINAL REFINED PLAN: 10 PHASES

### Phase 1: Security Foundation (P0 — Critical Path)
**Duration:** 3-5 days | **Owner:** Security Architect + Security Researcher

| # | Task | Expert | Priority | Effort |
|---|---|---|---|---|
| 1.1 | Fix `__getattr__` bypass in `vector_store.py` | Security Architect | P0 | 0.5d |
| 1.2 | Add model weight validation (size, shape, dtype) | Security Researcher | P0 | 0.5d |
| 1.3 | Add prompt injection guard in retrieval path | Security Researcher | P1 | 1d |
| 1.4 | Add `--strict-config` flag to reject unknown config keys | Security Researcher | P1 | 0.5d |
| 1.5 | Update SECURITY.md with attack surface documentation | Security Architect | P1 | 1d |
| 1.6 | **Golden regression suite** — baseline benchmark results | Benchmark Engineer / QA Engineer | P0 | 2d |
| 1.7 | CONTRIBUTING.md with setup instructions | DevEx Lead / Documentation | P1 | 1d |

**Success Criteria:**
- [ ] `__getattr__` bypass patched and tested
- [ ] Model weights validated on load
- [ ] Golden suite captures baseline metrics for all BEIR datasets
- [ ] CONTRIBUTING.md allows new dev to set up environment in < 10 minutes
- [ ] SECURITY.md documents known attack vectors and mitigations

**Risk Mitigation:**
- Golden suite baseline is captured BEFORE any other changes, so drift is measurable
- Security fixes isolated to dedicated branch, reviewed before merge

---

### Phase 2: Testing Deepening (P1)
**Duration:** 5-7 days | **Owner:** Test Automation Engineer + ML Research Scientist + QA Engineer

| # | Task | Expert | Priority | Effort |
|---|---|---|---|---|
| 2.1 | Create shared test fixtures (in-memory Qdrant, model helpers, temp dirs) | Test Automation Eng | P1 | 2d |
| 2.2 | Add research validity tests: convergence validation | ML Research Scientist | P1 | 2d |
| 2.3 | Add drift detection tests (embedding drift over time) | ML Research Scientist | P1 | 1d |
| 2.4 | Add integration test layer (multi-module workflows) | Test Automation Eng | P1 | 2d |
| 2.5 | Parameterized edge-case tests (replacing over-engineered fuzz) | Test Automation Eng | P1 | 1d |
| 2.6 | Enforce per-module test files mapping source one-to-one | QA Engineer | P2 | 1d |

**Success Criteria:**
- [ ] Shared fixtures reduce test duplication by 40%+
- [ ] Convergence tests verify adaptation improves embeddings on known collapse
- [ ] Drift detection tests catch unexpected embedding changes
- [ ] Integration tests cover full AEP workflow end-to-end
- [ ] Test count increases by 10-20% (research validity additions)

**Risk Mitigation:**
- Research validity tests run in CI but don't block on flaky convergence thresholds
- Integration tests use `:memory:` Qdrant for isolation

---

### Phase 3: Architecture Migration (P1)
**Duration:** 10-14 days | **Owner:** Software Architect + Python Typing Expert

**Pre-requisite:** Phase 4a (typing pass) must complete before Phase 3 restructuring.

| # | Task | Expert | Priority | Effort |
|---|---|---|---|---|
| 3a.1 | Add `from __future__ import annotations` to all .py files | Python Typing Expert | P1 | 1d |
| 3a.2 | Replace `X | None` with `Optional[X]` in public API signatures | Python Typing Expert | P1 | 1d |
| 3.1 | Dead code removal — identify and delete unused modules/functions | Software Architect | P1 | 2d |
| 3.2 | Create `chelatedai/` package with `__init__.py` re-exports | Software Architect | P1 | 2d |
| 3.3 | Verify all existing imports still work (backward compatibility) | Software Architect | P1 | 2d |
| 3.4 | Decompose `antigravity_engine.py` — extract concerns into modules | Software Architect | P1 | 4d |
| 3.5 | Update all test imports to use new package paths | Software Architect | P1 | 2d |

**Success Criteria:**
- [ ] All `import X` from flat layout still work via re-exports
- [ ] `python -c "from antigravity_engine import AntigravityEngine"` still succeeds
- [ ] `antigravity_engine.py` reduced from 1248 lines to < 400 lines (composition pattern)
- [ ] All 1082+ tests pass with new package structure
- [ ] No breaking changes to public API

**Risk Mitigation:**
- Re-export pattern preserves all existing imports — no test changes needed for phase 3.1-3.3
- Dead code analysis uses `ruff --select F401` plus manual review
- Phased merge: package structure first (3.2), then decomposition (3.4)

---

### Phase 4: Code Quality (P1)
**Duration:** 3-5 days | **Owner:** Code Quality Engineer

| # | Task | Expert | Priority | Effort |
|---|---|---|---|---|
| 4.1 | Consolidate 7 custom exceptions to 4: `ChelatedAIError`, `AdaptationError`, `EmbeddingError`, `ConfigError` | Code Quality Eng | P1 | 1d |
| 4.2 | Move remaining magic numbers to config.py | Code Quality Eng | P1 | 1d |
| 4.3 | Add mypy to CI (new code only, `--strict`) | Python Typing Expert | P1 | 1d |
| 4.4 | Expand ruff rules: I (isort), UP (pyupgrade), B (bugbear), SIM (simplify) | Code Quality Eng | P1 | 1d |
| 4.5 | Run isort across all files | Code Quality Eng | P2 | 0.5d |
| 4.6 | Remove unused imports (F401) | Code Quality Eng | P2 | 0.5d |

**Success Criteria:**
- [ ] Exception hierarchy documented and all imports updated
- [ ] Zero magic numbers in core logic (all in config.py)
- [ ] CI passes with mypy `--strict` on new code
- [ ] ruff check passes with expanded ruleset
- [ ] isort order consistent across all files

**Risk Mitigation:**
- Exception consolidation reviewed by all module owners
- Magic number extraction verified against tests

---

### Phase 5: Training Systems (P2)
**Duration:** 3-4 days | **Owner:** Training Systems Engineer + ML Ops Engineer

| # | Task | Expert | Priority | Effort |
|---|---|---|---|---|
| 5.1 | Extend `run_sweep.py` for hyperparameter grid search | Training Systems Eng | P2 | 2d |
| 5.2 | Replace ExperimentRegistry with JSONL experiment logs | ML Ops Engineer | P2 | 1d |
| 5.3 | Add checkpoint auto-prune (older than 7 days) | ML Ops Engineer | P2 | 0.5d |
| 5.4 | Online safety: checkpoint-restore (not complex rollback) | Training Systems Eng | P2 | 0.5d |

**Success Criteria:**
- [ ] `run_sweep.py` supports nested parameter grids
- [ ] Experiment results stored as JSONL files in `experiment_runs/`
- [ ] Old checkpoints auto-pruned without losing last 3 checkpoints
- [ ] Online update failsafe: checkpoint-restore on failure

**Risk Mitigation:**
- JSONL format matches existing project convention — no new dependencies
- Checkpoint pruning preserves safety net (last 3 always kept)

---

### Phase 6: Benchmark Infrastructure (P1)
**Duration:** 2-4 days | **Owner:** Benchmark Engineer + QA Engineer

| # | Task | Expert | Priority | Effort |
|---|---|---|---|---|
| 6.1 | **Golden regression suite** — baseline capture + drift detection | Benchmark Engineer | P0 | 2d |
| 6.2 | Add golden suite guard in CI (fail if drift > 2%) | DevOps Engineer | P0 | 0.5d |
| 6.3 | Use pandas for statistical analysis (not scipy for small samples) | Benchmark Engineer | P1 | 1d |
| 6.4 | Benchmark results → JSONL (already project convention) | Data Engineer | P2 | 0.5d |

**Success Criteria:**
- [ ] Golden suite captures baseline for all BEIR datasets
- [ ] CI fails if any golden result drifts > 2% from baseline
- [ ] Pandas-based analysis replaces manual statistical calculations
- [ ] All benchmark outputs use JSONL format

**Risk Mitigation:**
- Golden baseline captured in Phase 1 before any changes
- CI drift threshold (2%) is configurable, not hardcoded

---

### Phase 7: Documentation (P1)
**Duration:** 3-5 days | **Owner:** Documentation Specialist + Research Integrity Auditor

| # | Task | Expert | Priority | Effort |
|---|---|---|---|---|
| 7.1 | Install MkDocs + mkdocstrings, configure site | Documentation | P1 | 1d |
| 7.2 | **Archive `docs/ARCH AGENTIC ENGINEERING AND PLANNING/`** to `docs/archive/` | Documentation | P1 | 0.5d |
| 7.3 | Generate API docs via mkdocstrings | Documentation | P1 | 1d |
| 7.4 | Create model cards for each adapter type | Research Integrity | P2 | 1d |
| 7.5 | Add reproducibility checklist to PR template | Research Integrity | P2 | 0.5d |
| 7.6 | Define maintenance model (who updates docs on each PR) | Documentation | P1 | 0.5d |

**Success Criteria:**
- [ ] MkDocs site builds locally with `mkdocs serve`
- [ ] API docs auto-generated from docstrings
- [ ] ARCH/ files moved to archive (preserved but not in main docs tree)
- [ ] Model cards document each adapter's purpose, inputs, limitations
- [ ] PR template includes reproducibility checklist

**Risk Mitigation:**
- ARCH/ archive preserves historical value without cluttering active docs
- MkDocs auto-generation reduces ongoing maintenance burden

---

### Phase 8: Computational Storage Docs (P2)
**Duration:** 2-3 days | **Owner:** Embedded Systems Engineer

| # | Task | Expert | Priority | Effort |
|---|---|---|---|---|
| 8.1 | Add local build docs for RP2040 (pio build, dfu-util flashing) | Embedded Eng | P2 | 1d |
| 8.2 | Consolidate ADRs into single `docs/COMPUTATIONAL_STORAGE_DESIGN_DECISIONS.md` | Embedded Eng | P2 | 0.5d |
| 8.3 | Enhance existing firmware documentation | Embedded Eng | P2 | 1d |
| 8.4 | Document hardware evidence capture prerequisites | Embedded Eng | P2 | 0.5d |

**Success Criteria:**
- [ ] Local build instructions include pio build output path and dfu-util command
- [ ] Single design decisions document replaces scattered ADRs
- [ ] Hardware evidence capture only recommended when RP2040 hardware is attached

**Risk Mitigation:**
- No firmware code changes — documentation only
- Hardware evidence capture explicitly gated on hardware availability

---

### Phase 9: Performance (P1)
**Duration:** 5-7 days | **Owner:** Performance Engineer + Data Engineer

| # | Task | Expert | Priority | Effort |
|---|---|---|---|---|
| 9.1 | Profiling/baselines — identify hot paths | Performance Eng | P1 | 2d |
| 9.2 | Embedding caching to disk (JSONL-based) | Data Engineer | P1 | 2d |
| 9.3 | ONNX export for embedding models | Performance Eng | P1 | 2d |
| 9.4 | Remove async migration from scope (zero ROI) | Performance Eng | P0 (deletion) | 0.5d |
| 9.5 | Remove GPU quantization / mixed precision from scope | Performance Eng | P0 (deletion) | 0.5d |
| 9.6 | Remove Prometheus/OpenTelemetry from scope | Performance Eng | P0 (deletion) | 0.5d |

**Success Criteria:**
- [ ] Profiling report identifies top 3 hot paths in antigravity_engine
- [ ] Embedding cache reduces repeated lookup latency by 50%+
- [ ] ONNX export produces valid model file for each embedding backend
- [ ] Outdated scope items removed from all planning documents

**Risk Mitigation:**
- Profiling first ensures optimizations target actual bottlenecks
- Embedding cache uses existing JSONL convention
- ONNX export is opt-in (doesn't change default behavior)

---

### Phase 10: Production Readiness Lite (P2)
**Duration:** 2-3 days | **Owner:** Developer Experience Lead + DevOps Engineer

**This replaces the original "Monitoring" phase which was eliminated entirely.**

| # | Task | Expert | Priority | Effort |
|---|---|---|---|---|
| 10.1 | Create `.devcontainer/devcontainer.json` for one-command setup | DevEx Lead | P2 | 1d |
| 10.2 | Add pre-commit hooks (ruff, isort, mypy for new code) | DevOps Engineer | P2 | 1d |
| 10.3 | Add simple health check endpoint to dashboard_server | DevOps Engineer | P2 | 0.5d |
| 10.4 | Add `scripts/setup.sh` and `scripts/setup.bat` | DevEx Lead | P2 | 0.5d |

**Success Criteria:**
- [ ] `devcontainer.json` allows VS Code remote container setup
- [ ] Pre-commit hooks run linting/formatting on staged files
- [ ] Dashboard health check returns basic system status
- [ ] Setup scripts work on both Linux/macOS and Windows

**Risk Mitigation:**
- Pre-commit hooks are advisory (not CI gates) for Phase 10 launch
- Dev container is optional enhancement, not required for development
- Health check is informational only (no alerting infrastructure)

---

## Execution Timeline

### Parallel Workstreams

```
Week 1 (Day 1-5):
┌─────────────────────────────────────────────────────────────────┐
│ WORKSTREAM A (Critical Path):                                   │
│   Phase 1: Security Foundation (3-5d)                           │
│   Phase 6: Benchmark Infrastructure — golden suite (2d)         │
│   → These are parallel Day 1 tasks                              │
├─────────────────────────────────────────────────────────────────┤
│ WORKSTREAM B (Independent):                                     │
│   Phase 7: Documentation — MkDocs setup, ARCH/ archive (2-3d)   │
│   Phase 4: Code Quality — exception consolidation, ruff expand  │
│   → These run parallel with A                                   │
├─────────────────────────────────────────────────────────────────┤
│ WORKSTREAM C (Dependent on A):                                  │
│   Phase 2: Testing Deepening — fixtures, research validity (5-7d)│
│   → Can start after Phase 1 security fixes land                 │
└─────────────────────────────────────────────────────────────────┘

Week 2-3:
┌─────────────────────────────────────────────────────────────────┐
│ WORKSTREAM D (Sequential — depends on 3a typing pass):          │
│   Phase 4a: Typing Pass (2d)                                    │
│   Phase 3: Architecture Migration (10-14d)                      │
│   → MUST wait for typing pass to complete                       │
├─────────────────────────────────────────────────────────────────┤
│ WORKSTREAM E (Independent):                                     │
│   Phase 8: Computational Storage Docs (2-3d)                    │
│   Phase 9: Performance — profiling, caching, ONNX (5-7d)        │
│   → These run in parallel, independent of D                     │
├─────────────────────────────────────────────────────────────────┤
│ WORKSTREAM F (Can start anytime):                               │
│   Phase 5: Training Systems (3-4d)                              │
│   Phase 10: Production Readiness Lite (2-3d)                    │
│   → These are independent of all other phases                   │
└─────────────────────────────────────────────────────────────────┘

Total Duration: 4-5 weeks
Critical Path: Phase 1 → Phase 2 → Phase 4a → Phase 3
```

### Dependency Graph

```
Phase 1 (Security) ──────────────────→ Phase 2 (Testing)
                                              ↑
Phase 6 (Golden Suite) ───────────────────────┘
                                              ↓
Phase 4a (Typing Pass) ──────────────────────→ Phase 3 (Architecture)
                                              ↓
                                          All Tests Pass
                                              ↑
Phase 4 (Code Quality) ───────────────────────┘

Phase 7 (Documentation) ──────────────────────────────────────→ Independent
Phase 8 (Comp. Storage Docs) ─────────────────────────────────→ Independent
Phase 5 (Training) ───────────────────────────────────────────→ Independent
Phase 9 (Performance) ────────────────────────────────────────→ Independent
Phase 10 (Prod. Readiness) ──────────────────────────────────→ Independent
```

### Milestones

| Week | Milestone | Gates |
|------|-----------|-------|
| End of Week 1 | Security patched, golden baseline captured, docs skeleton ready | All Phase 1 tasks done, golden suite passing |
| End of Week 2 | Testing deepening complete, code quality improved, typing pass done | 1082+ tests passing, ruff + mypy clean |
| End of Week 3 | Architecture migration complete | All imports preserved, 1082+ tests passing |
| End of Week 4 | Performance improvements, training systems, docs complete | Profiling report, cache working, MkDocs live |
| End of Week 5 | Production Readiness Lite, final cleanup | Pre-commit hooks working, devcontainer ready |

---

## Consolidation Map

The following recommendations were consolidated from multiple experts:

| Consolidated Item | Experts Contributing | Final Location |
|---|---|---|
| Golden regression suite | Benchmark Eng, QA Eng, DevOps Eng | Phase 1 + Phase 6 |
| JSONL for data storage | Data Eng, ML Ops Eng, Benchmark Eng | Phase 5, Phase 6 |
| MkDocs over Sphinx | Documentation, Code Quality Eng | Phase 7 |
| ARCH/ archiving | Documentation, Tech Program Mgr | Phase 7 |
| ONNX + caching | Performance Eng, Data Eng | Phase 9 |
| Pre-commit hooks | DevOps Eng, Code Quality Eng, DevEx Lead | Phase 10 |
| Research validity tests | ML Research Scientist, QA Eng | Phase 2 |
| Security fixes | Security Architect, Security Researcher | Phase 1 |
| Typing pass before restructure | Python Typing Expert, Software Architect | Phase 4a → Phase 3 |
| Eliminate monitoring phase | Systems Reliability Eng, Tech Program Mgr | Phase 10 replaced |

---

## What Was Deliberately Excluded

| Item | Reason | Who Recommended |
|---|---|---|
| Sentry integration | Assumes production infrastructure | Systems Reliability Eng |
| Runbooks | Assumes production incident response | Systems Reliability Eng |
| Health checks with alerting | Assumes production monitoring | Systems Reliability Eng |
| Prometheus metrics | Assumes production metrics stack | Performance Eng, Systems Reliability Eng |
| OpenTelemetry tracing | Assumes distributed tracing | Performance Eng |
| Optuna HPO | Overkill for research prototype | Training Systems Eng |
| MLflow experiment tracking | Overkill, project uses JSONL | Training Systems Eng, ML Ops Eng |
| ADRs (multiple) | Overkill for project scale | Embedded Systems Eng |
| GPU quantization / mixed precision | Hardware doesn't exist | Performance Eng |
| Async migration | Zero user-facing value | Performance Eng |
| Complex fuzz testing | Over-engineered for current scope | Test Automation Eng |
| pytest adoption | CI uses unittest exclusively | Test Automation Eng |
| Production CI gates (strict) | Research code is exploratory | QA Eng, Tech Program Mgr |
| 90% coverage target | Unnecessary for research code | QA Eng |
| Calibration validation | Research tangent, not core | Training Systems Eng |
| Complex online rollback | Scope creep, checkpoint-restore suffices | Training Systems Eng |

---

## Success Criteria Summary

| Metric | Current | Target | Phase |
|---|---|---|---|
| `antigravity_engine.py` lines | 1248 | < 400 | Phase 3 |
| Custom exceptions | 7 | 4 | Phase 4 |
| Test count | 1082+ | 1180-1250 | Phase 2 |
| Magic numbers in core | Many | 0 | Phase 4 |
| ARCH/ in main docs | 90 files | Archived | Phase 7 |
| Golden baseline | None | Captured | Phase 1 |
| CI mypy check | None | New code only | Phase 4 |
| Ruff rules | Basic | Extended (I, UP, B, SIM) | Phase 4 |
| Embedding cache | None | JSONL-based disk cache | Phase 9 |
| ONNX export | None | Per backend | Phase 9 |
| CONTRIBUTING.md | None | Complete | Phase 1 |
| Dev container | None | VS Code ready | Phase 10 |
| MkDocs site | None | API auto-generated | Phase 7 |

---

## Risk Registry

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Package restructure breaks existing imports | High | Medium | Re-export pattern preserves all imports; verify with import smoke test |
| Golden baseline drifts due to non-deterministic embeddings | Medium | High | Use fixed random seeds; allow configurable drift threshold |
| Typing pass conflicts with existing code | Medium | Low | `from __future__ import annotations` is safe in all Python 3.9+ |
| Dead code removal deletes useful but unused code | Medium | Low | Conservative removal: only F401-confirmed unused imports |
| Test count growth slows development | Low | Medium | Cap research validity tests at reasonable number; don't chase 90% coverage |
| MkDocs setup conflicts with existing docs | Low | Low | MkDocs is additive; doesn't modify existing markdown files |
| Pre-commit hooks annoy contributors | Low | Medium | Start with advisory-only mode, enforce in CI later |
| Checkpoint pruning removes needed checkpoints | Medium | Very Low | Always preserve last 3 checkpoints regardless of age |
| ONNX export fails on custom embedding models | Low | Low | ONNX export is opt-in; doesn't affect default runtime |
| Architecture migration takes longer than estimated | High | Medium | Phased approach: typing → dead code → package → decomposition |

---

## Notes

- **Phase 4a** (typing pass with `from __future__ import annotations`) is a pre-requisite gate for Phase 3. It must complete before architecture reorganization begins.
- **Golden regression suite** is the highest priority technical deliverable (P0) because it protects against regressions across ALL other phases.
- **Phase 10** is fundamentally different from a "Monitoring" phase — it is "Production Readiness Lite" covering developer tooling (devcontainer, pre-commit hooks, setup scripts) only. All production-grade monitoring was eliminated.
- All phases are designed to be independently mergeable as separate PRs, enabling parallel review.
- CI always passes the existing 1082+ test suite. No phase proceeds if test count drops below 1000.
