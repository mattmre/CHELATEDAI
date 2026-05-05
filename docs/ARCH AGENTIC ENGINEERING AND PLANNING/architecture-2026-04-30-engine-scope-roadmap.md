# Architecture — 2026-04-30 Engine-Scope Roadmap

## Goal
Turn the current lexical-only gate search into an internal-feature search surface that can support fail-closed control, coverage-aware evaluation, and targeted hard-negative generation for the golden-default campaign.

## Context
- The conservative retrieval default is still the only evidence-backed default.
- The corrected autopilot loop is now structurally sound, but it still learns mostly from lexical query features.
- The Qwen-Scope pattern suggests the missing layer is an internal feature interface that can be reused for diagnosis, gating, coverage analysis, and targeted data generation.

## Overall Assessment
APPROVED WITH MODIFICATIONS

The direction is sound, but the rollout should be sequenced around internal telemetry first, not around broader policy activation. The first deliverable is not a new default. It is a reusable Engine-Scope feature layer.

## Decision
Adopt an Engine-Scope architecture with five implementation phases:

1. Instrument internal engine-feature rows.
2. Train fail-closed reform and mask gates on those rows.
3. Add feature-footprint coverage analysis to campaign selection.
4. Add synthetic hard-negative mining and generation.
5. Integrate all of the above into the autonomous golden-default supervisor.

## In Scope
- Internal query/profile row emission from the road-course and autopilot flows
- Reusable feature schema covering action choice, attribution, control drift, and profile deltas
- Internal-feature reformulation and masking gates
- Feature-overlap and coverage reporting for task/window selection
- Synthetic hard-negative generation targeted at active-negative and active-neutral failures
- ARCH-AEP cycle artifacts for implementation tracking

## Out Of Scope
- Promoting a new retrieval default before the Engine-Scope loop survives holdout validation
- Reopening global threshold sweeps as the primary search path
- Persistent self-healing writes or any always-on unsafe actuator path
- Direct use of Qwen SAE checkpoints inside ChelatedAI

## Architecture Findings

### 1. Internal features are the missing control surface
- Verdict: accept
- Rationale: current learned gates are structurally correct but underpowered because they mostly observe lexical features instead of internal retrieval/control telemetry.

### 2. Coverage should move upstream of more data collection
- Verdict: accept
- Rationale: the corrected loop can now rotate windows, but it still cannot tell which new windows are redundant. Feature-footprint analysis should guide future collection.

### 3. Negative-targeted generation is higher value than broad positive steering
- Verdict: accept
- Rationale: current evidence and Qwen-Scope both indicate that inducing and learning from failure signatures is a better path than pushing more always-on positive interventions.

### 4. Live policy activation should remain fail-closed until internal gates survive holdout
- Verdict: accept
- Rationale: all new gates should begin in advisory or shadow mode, then graduate only after no active-negative blockers remain.

## Phase Plan

### Phase 1 — Engine-Scope Row Contract
- Objective: define and emit a pooled internal-feature row per query/profile/window.
- Estimated effort: M
- Core files:
  - `run_golden_default_autopilot.py`
  - `run_thousand_query_tuning.py`
  - `static_mask_probe.py`
  - new `engine_scope.py`
- Deliverables:
  - stable row schema
  - pooled artifact writer
  - row validation tests
- Acceptance criteria:
  - every evaluated query/profile emits a normalized Engine-Scope row
  - rows contain task, query id, offset, action, attribution summary, control diagnostics, fault class, and metric deltas
  - pooled artifacts can be reloaded without task-specific custom parsing

### Phase 2 — Internal Fail-Closed Gates
- Objective: train reform and mask gates from Engine-Scope rows instead of lexical-only features.
- Estimated effort: M
- Core files:
  - `learned_reformulation_gate.py`
  - `learned_mask_gate.py`
  - `query_reformulator.py`
  - `antigravity_engine.py`
- Deliverables:
  - internal-feature gate trainers
  - shadow/advisory policy mode
  - holdout validation reports
- Acceptance criteria:
  - gate trainers can ingest pooled Engine-Scope rows directly
  - policy configs remain fail-closed when evidence is sparse
  - no gate may advance without explicit positive delta and zero active-negative blockers

### Phase 3 — Feature Coverage And Overlap
- Objective: measure which task windows and query families add new information.
- Estimated effort: M
- Core files:
  - new `engine_scope_coverage.py`
  - `run_golden_default_autopilot.py`
  - `research_pathway_analyzer.py`
- Deliverables:
  - feature-footprint extractor
  - overlap and redundancy reports
  - coverage-aware next-window selector
- Acceptance criteria:
  - tasks and offsets can be compared by feature overlap
  - the campaign can identify redundant versus novel windows
  - the next collection slice can be selected by feature novelty rather than only round-robin order

### Phase 4 — Synthetic Hard-Negative Pipeline
- Objective: mine or synthesize queries that activate known failure features.
- Estimated effort: L
- Core files:
  - new `engine_scope_negatives.py`
  - `static_mask_probe.py`
  - `run_golden_default_autopilot.py`
- Deliverables:
  - failure-signature miner
  - replayable hard-negative dataset/artifact
  - stress validation path for reform/mask gates
- Acceptance criteria:
  - active-negative and active-neutral signatures can be clustered into reusable failure families
  - at least one synthetic or mined hard-negative artifact can be replayed deterministically
  - the validation harness reports gate behavior on this stress set separately from normal holdout

### Phase 5 — Supervisor Integration And Decision Loop
- Objective: make Engine-Scope the primary autonomous golden-default search loop.
- Estimated effort: M
- Core files:
  - `run_golden_default_autopilot.py`
  - `docs/qwen-scope-engine-mapping-2026-04-30.md`
  - ARCH-AEP cycle artifacts
- Deliverables:
  - updated supervisor phases
  - campaign manifest/report extensions
  - candidate promotion and no-promotion decision rules
- Acceptance criteria:
  - manifests record Engine-Scope coverage and gate outcomes
  - the supervisor can distinguish collection, gate-training, coverage-expansion, and hard-negative phases
  - the campaign can end with either a supported candidate or a documented no-promotion result

## First Three PRs

### PR 1
- branch: `aep/high/AEP-20260430-PR000-001/engine-scope-rows`
- scope:
  - add the Engine-Scope row schema and emitters
  - pool rows from reformulation, masking, and validation paths
  - add contract tests and artifact reload tests

### PR 2
- branch: `aep/high/AEP-20260430-PR000-002/internal-gates`
- scope:
  - extend learned reformulation and learned mask gates to internal Engine-Scope features
  - keep advisory-only rollout with explicit fail-closed behavior
  - add holdout survival tests

### PR 3
- branch: `aep/medium/AEP-20260430-PR000-003/coverage-analysis`
- scope:
  - add feature-footprint overlap/redundancy analysis
  - feed coverage reports into the supervisor's next-window selection
  - add deterministic overlap unit tests

## Phase Sequencing Recommendations
1. Do not start synthetic hard-negative generation before the Engine-Scope row contract exists.
2. Do not connect new gates to live action selection before they pass shadow-mode holdout validation.
3. Treat coverage analysis as a prerequisite for any large-scale new collection run after the next short smoke cycle.

## Feasibility Concerns
1. Attribution summaries may need light normalization before they are stable enough for pooled training.
2. Row schema bloat is a real risk; the first contract should stay compact and versioned.
3. Hard-negative generation can drift into fixture-building if it is not anchored to observed failure families.

## Risk Register
| Risk | Likelihood | Impact | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| Engine-Scope row schema grows too quickly and becomes hard to reuse | M | H | version the schema and start with a compact required field set | Codex |
| Internal-feature gates leak regressions when evidence is sparse | M | H | enforce advisory-only shadow mode and fail-closed defaults | Codex |
| Coverage metrics become expensive or noisy | M | M | compute on pooled artifacts first and gate runtime use behind sampling limits | Codex |
| Synthetic hard negatives overfit to toy failures | M | M | mine from real active-negative signatures before adding synthetic generation | Codex |

## Value Alignment Assessment
- Highest-value phase: Phase 1, because every later phase depends on a reusable internal row contract.
- Lowest value-to-effort ratio if done too early: Phase 4, because synthetic negatives without a stable feature layer risk becoming expensive noise.
- Recommended MVP scope:
  - Phase 1 complete
  - Phase 2 in advisory mode
  - Phase 3 reporting integrated
  - Phase 4 limited to mined negatives, not broad generation

## Rollback Strategy
- All new gates begin advisory-only and can be disabled by config.
- Engine-Scope emitters write additive artifacts and do not replace existing telemetry.
- Coverage-aware selection can fall back to the current deterministic rotation if overlap metrics misbehave.
- Synthetic hard-negative paths stay off by default until replay evidence is stable.

## Success Criteria
1. The next autonomous run can explain why a gate fired using internal Engine-Scope features, not just lexical heuristics.
2. The campaign can show which windows add new failure/feature coverage and which are redundant.
3. Hard-negative stress validation exists for at least one real failure family.
4. The final decision surface is either a holdout-surviving candidate or a clearly evidenced no-promotion report.
