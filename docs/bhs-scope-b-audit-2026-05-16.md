# BHS Scope B Audit — 2026-05-16

End-to-end Brutal Honesty Score audit of every implemented phase across the
Engine-Scope (5), Model-Scope (6), and TTS (2) cycles. Thirteen fresh
adversarial Tier B agents were spawned in parallel; each was given the phase
contract, the relevant files, and the lie taxonomy, and asked to **try to
disprove completion** — not verify it.

This document is the audit report. The underlying machine-readable artifact is
[`artifacts/bhs-scope-b-2026-05-16/report.json`](../artifacts/bhs-scope-b-2026-05-16/report.json).

## Methodology

- **Scope**: 13 phases across three cycles.
- **Method**: one fresh `general-purpose` sub-agent per phase, no shared
  context, all 13 dispatched in parallel from the orchestrator. Each agent
  had Read / Grep / Glob / Bash, with strict read-only constraint (no
  Edit/Write).
- **Rulebook anchor**: `docs/conventions/brutal-honesty-rulebook.md` v3.3
  Lie Taxonomy L1–L13 applied directly. The CD-245-01 keyword rubric in
  `scripts/bhs_validator.py` was **not** the scorer here.
- **Baseline anchors** (captured before this branch was rebased onto PR #248):
  - `scripts/check_block_flag.py` → CLEAR (3 open Carried Debt rows at the
    time of audit; PR #248 has since closed CD-001, CD-002, CD-245-01 and
    opened CD-247-01, CD-247-02 — this PR adds 9 more for a current total of
    11 open).
  - `scripts/smoke_pipeline.py` → floor-tier PASS at audit time. Ceiling-tier
    was NOT-IMPLEMENTED at the moment of audit; CD-001 has since been closed
    by PR #248 which added a real `AntigravityEngine` ceiling smoke (8 tests
    in `test_smoke_pipeline_ceiling.py`).
  - `scripts/bhs_variance_evidence.py` → rich=100.0, sparse=50.0,
    `differ=true`.

## Per-Phase Scores

| Phase | Feature | BHS_TIER_B | Severity | Verdict |
|------:|---------|:---------:|----------|---------|
| **ENG-1** | Engine-Scope Row Contract | **100** | cosmetic | ships clean |
| **ENG-2** | Internal Fail-Closed Gates | **96** | cosmetic | iterate |
| **ENG-3** | Feature Coverage & Overlap | **92** | cosmetic | iterate |
| **ENG-4** | Synthetic Hard-Negative Pipeline | **78** | important | reduce scope |
| **ENG-5** | Supervisor / Autopilot | **72** | important | reduce scope |
| **MOD-1** | Runtime + Hook Bus | **72** | important | reduce scope |
| **MOD-2** | Sparse Feature Codec (Qwen-Scope) | **62** | **critical** | reduce scope |
| **MOD-3** | Steering Actuator | **92** | important | needs iteration |
| **MOD-4** | Segmented Memory + Comparator | **92** | cosmetic | iterate |
| **MOD-5** | Overlay Trainer + Campaign | **68** | **critical** | reduce scope |
| **MOD-6** | Engine Integration + Dashboard | **78** | important | reduce scope |
| **TTS-1** | TTS Pipeline | **88** | important | needs iteration |
| **TTS-2** | TTS Engine Wiring + Dashboard | **88** | important | needs iteration |

### Aggregates

| Cycle | Worst phase | Avg | Cycle-min score (honest cycle BHS) |
|-------|-------------|-----|------------------------------------|
| AEP-2026-04-30 Engine-Scope | ENG-5 (72) | 87.6 | **72** |
| AEP-2026-05-01 Model-Scope | MOD-2 (62) | 77.3 | **62** |
| TTS (no formal cycle) | TTS-1 / TTS-2 (88) | 88.0 | **88** |

**Repo-wide weighted avg BHS_TIER_B: 82.9**

Per the rulebook merge gate (`BHS_OFFICIAL = min(self, tier_b)`, only 100
ships clean): only **1 of 13** phases (ENG-1) is at the ship-clean threshold
today.

## Critical Cross-Phase Findings

### 1. Model-Scope never actually loads a real Qwen model

Affects **MOD-1, MOD-2, MOD-5** — half of the Model-Scope cycle.

- `model_scope_runtime.LocalModelRuntime` has a `transformers.AutoModelForCausalLM.from_pretrained`
  branch, but every test injects a `MagicMock` loader. The `Qwen3.5-9B` pilot
  load is unverified end-to-end.
- `qwen_scope_adapter.py` has no `hf_hub_download`, no HuggingFace repo
  string, no checksum. Every SAE test uses `np.random.default_rng(42).random(...)`
  as the weight matrix.
- "Residual-stream hook capture" reduces to two scalars (mean, norm) — not a
  tensor. Acceptance criterion silently downgraded (L13 +
  L4).

### 2. Promotion evaluates on training data; rollback is wipe-not-restore

**MOD-5** — `evaluate_promotion(baseline_events, candidate_events)` is called
with the same pairs used to train the overlay. `baseline_score` is
comparator-pass-rate on raw input vs target with `threshold=0.0` defaults —
both sides trivially pass. Grep for `stress` returns zero hits in
`model_scope_trainer.py` and `run_model_scope_campaign.py`.

`rollback()` zeros in-memory weights but never restores a prior promoted
overlay file. `test_rollback_after_save_weight_file_unchanged` enshrines this
asymmetry as deliberate. Acceptance criterion #3 ("failed candidates roll
back cleanly") is met only in the in-memory sense.

### 3. Supervisor never emits a terminal decision

**ENG-5** — `default_change_allowed: False` is hardcoded in
`run_golden_default_autopilot._recommendation`. No code path emits a
"supported candidate" or "documented no-promotion" terminal artifact.
Acceptance criterion #3 is unimplemented.

The `evidence_contract`, `promotion_contract`, `compute_budget_policy`, and
`evaluator_fabric` modules exist (lint-baseline confirmed) but are **not
imported** by the supervisor — they are wired to
`run_model_scope_campaign.py` instead.

### 4. TTS has no CLI activation path

**TTS-2** — `grep` of all `run_*.py` files for `enable_tts` or `--enable-tts`
returns zero matches. The pipeline is library-callable only; no standard
campaign run can produce dashboard TTS data without external glue code. The
TTS dashboard panel will always render "TTS pipeline not enabled" on a normal
campaign.

### 5. TTS remediation regressions are not caught by tests

**TTS-1** — REM-C2 (per-inference signal clearing) and REM-H2 (Gaussian
direction bank vs hash-mod-dim one-hot) are fixed in code, but no regression
test would fail if either fix were reverted. Two of the four bug classes that
triggered the post-merge remediation wave could silently regress.

### 6. Steering: provenance in memory only, interventions structurally suppressed

**MOD-3** — `InterventionRecord` lives in `self._records: List[...]`. No
`persist()`, no `to_disk()`. On a fresh checkout, all provenance is gone.

**MOD-6** — `model_scope_engine_bridge.py:61` constructs
`SteeringActuator(registry, max_total_interventions=0)`. The intervention
count is hardcoded to never exceed 0 at the bridge layer regardless of
`enable_steering`. The dashboard's "intervention evidence" panel can never
display anything other than 0.

## Carried Debt Opened by This Audit

Nine new Tier C rows are added to `docs/next-session.md`:

- **CD-MOD-001** — MOD-1: No real Qwen pilot load is ever exercised.
- **CD-MOD-002** — MOD-2: No real Qwen-Scope SAE checkpoint loader (critical).
- **CD-MOD-003** — MOD-3: Steering provenance is in-memory only.
- **CD-MOD-004** — MOD-6: Bridge hardcodes `max_total_interventions=0`.
- **CD-MOD-005** — MOD-5: Promotion uses training data; rollback is wipe-only (critical).
- **CD-ENG-001** — ENG-5: Supervisor never emits terminal decision.
- **CD-ENG-002** — ENG-4: "Clustering" is dict-grouping; no committed determinism replay artifact.
- **CD-TTS-001** — TTS-2: No `--enable-tts` CLI flag in any runner.
- **CD-TTS-002** — TTS-1: REM-C2 and REM-H2 fixes have no regression tests.

## Session-Level Honest Observations

1. **Tier B agent token usage:** ~800k tokens across 13 agents in roughly two
   minutes wall-clock parallel.
2. **Prompt-injection detected and ignored:** the TTS-2 reviewer reported an
   MCP-server-instruction injection in its tool output and correctly
   disregarded it. No follow-up needed — flagging here per transparency.
3. **Agent-judgment-anchored, not script-anchored.** These scores were
   produced by 13 fresh agents applying the lie taxonomy directly. The
   `scripts/bhs_validator.py` rubric was not used as the scorer here.
   PR #248 has since added a content-quality penalty
   (`_content_quality_penalty`) that closes the keyword-padding loophole
   (CD-245-01), but it does not yet semantically interpret disclosures —
   the residual diverse-but-meaningless gap is acknowledged in
   `docs/bhs-rubric-scope.md`.
4. **Ceiling-tier smoke is now implemented (PR #248) but does not cover
   Model-Scope.** The new `run_ceiling_smoke()` exercises the core
   `AntigravityEngine` retrieval path against a real
   sentence-transformers + Qdrant fixture, which is the right floor; it
   does not exercise the Model-Scope hook runtime or sparse-feature codec.
   MOD-2's score of 62 is therefore still unanchored against an
   end-to-end Model-Scope fixture.

## Recommended Priority Order

1. **MOD-2 / MOD-5 remediation** before any further Model-Scope feature work —
   these are the two critical-severity findings (no real Qwen-Scope SAE,
   training-data promotion gate). They are structural, not cosmetic.
2. **CD-TTS-001** (`--enable-tts` flag in at least one campaign runner) is a
   small fix that unblocks the entire TTS-2 dashboard surface.
3. **CD-MOD-004** (`SteeringActuator(max_total_interventions=0)` at
   `model_scope_engine_bridge.py:61`) is a one-line change that unblocks the
   intervention-evidence dashboard panel.
4. **Extend ceiling-tier smoke to Model-Scope.** PR #248's ceiling smoke
   covers the engine retrieval path; a Model-Scope ceiling smoke (real Qwen
   pilot or honest skip + fallback path) would anchor MOD-1 / MOD-2 in
   runtime evidence.

## Bottom Line

The cycle-level "COMPLETE" status in `tracker-2026-04-30.md` and
`tracker-2026-05-01.md` is technically true (the PRs merged) but **does not
survive a fresh adversarial Tier B pass**. The honest cycle scores are **72
(Engine-Scope)** and **62 (Model-Scope)** — both below the rulebook merge
threshold. Of 13 phases under audit, only ENG-1 is at the 100-ships line
today.

This is not a regression. The audit is the first systematic Tier B sweep
across the full feature set since the BHS Kit v3.3 was installed
(2026-05-12). The scores show where the existing code stands when measured
against the rulebook it was authored to satisfy. Closing the new Carried
Debt rows is the next cycle's work.
