# Architecture — 2026-05-01 Model-Scope Steering Roadmap

## Goal

Build a true model-hook steering layer for local open models that can capture internal activations, extract sparse features, apply bounded steering interventions, and support iterative training without depending on unsafe base-model mutation.

## Context

- `Engine-Scope` is now the engine-side precursor layer, not the final target.
- The current repo has adapter training, online updating, checkpointing, and advisory self-healing, but no direct activation-hook runtime.
- Official Qwen-Scope artifacts now provide a practical path for sparse-feature steering on verified Qwen targets.
- As of 2026-05-01, official `Qwen3.6` models exist with verified open-weight releases on 2026-04-15 (`Qwen3.6-35B-A3B`) and 2026-04-22 (`Qwen3.6-27B`), but the verified Qwen-Scope hook-and-feature path is still clearer on `Qwen3` and `Qwen3.5`.
- The best first serious pilot is `Qwen3.5-9B`, with lighter debug work on `Qwen3.5-2B` or `Qwen3-1.7B`.

## Overall Assessment

APPROVED WITH MODIFICATIONS

The direction is sound, but the rollout must begin with a hook runtime and bounded overlay promotion, not with direct persistent base-weight edits. The first deliverable is a reusable `Model-Scope` control plane, not a self-rewriting model.

## Decision

Adopt a `Model-Scope` architecture with six implementation phases:

1. Build a local model runtime and hook bus.
2. Add sparse feature extraction with Qwen-Scope-backed codecs.
3. Add bounded steering actuators and shadow-mode policy control.
4. Add segmented memory and expectation comparison.
5. Add iterative training, replay, and promotion gates for overlays.
6. Integrate the full stack into ChelatedAI runtime and evaluation harnesses.

## In Scope

- local `transformers`-based model runtime for supported pilot models
- residual-stream capture and normalized activation event records
- Qwen-Scope sparse feature extraction for supported Qwen targets
- fallback raw-activation or probe-based feature extraction for unsupported targets
- steering overlays, probe heads, feature-scaling policies, and LoRA-style updates
- segmented memory for working, episodic, expectation, and persistent artifacts
- replay, comparator, rollback, and promotion controls
- ARCH-AEP cycle artifacts for implementation tracking

## Out Of Scope

- immediate persistent mutation of base model weights
- opaque API-only serving paths such as Ollama as the primary hook runtime
- claiming model-family-agnostic support before the Qwen pilot is stable
- unbounded autonomous updates without replay and rollback evidence
- replacing Engine-Scope; it remains a useful engine-side layer

## Architecture Findings

### 1. The repo needs a new runtime, not just new policies
- Verdict: accept
- Rationale: current runtime surfaces can steer adapters and retrieval policy, but they do not expose the internal hidden-state surfaces needed for true model-scope control.

### 2. Qwen3.5-9B is the right first serious pilot
- Verdict: accept
- Rationale: it fits the requested approximate size class and has verified official Qwen-Scope support, which lowers the cost of feature extraction and steering experiments.

### 3. Persistent promotion must target overlays first
- Verdict: accept
- Rationale: direct weight editing is a later-stage research path. The first safe promotion surface is overlays, probes, steering vectors, replay families, and typed memory artifacts.

### 4. Segmented memory must be typed and bounded
- Verdict: accept
- Rationale: a single undifferentiated “memory” abstraction will collapse into context sprawl. Working, episodic, expectation, and persistent memory need distinct contracts.

### 5. Comparator and replay logic are mandatory, not optional
- Verdict: accept
- Rationale: without replay and expectation comparison, “self-improving” quickly becomes “self-drifting”.

## Phase Plan

### Phase 1 — Model Runtime And Hook Bus
- Objective: create a local model runtime that supports deterministic inference and activation capture.
- Estimated effort: M
- Core files:
  - new `model_scope_runtime.py`
  - new `model_hook_bus.py`
  - `embedding_backend.py`
  - `config.py`
- Deliverables:
  - explicit runtime loader for supported local models
  - hook registration and capture API
  - observation-only activation event writer
- Acceptance criteria:
  - a supported local Qwen pilot can be loaded without the Ollama path
  - residual-stream hook capture works for configured layers
  - activation events are versioned, reloadable, and test-covered

### Phase 2 — Sparse Feature Codec Layer
- Objective: convert hooked activations into reusable sparse feature events.
- Estimated effort: M
- Core files:
  - new `qwen_scope_adapter.py`
  - new `model_scope_features.py`
  - new `model_scope_artifacts.py`
- Deliverables:
  - loader for official Qwen-Scope SAE checkpoints
  - sparse feature extraction API
  - fallback raw/probe feature summaries for unsupported targets
- Acceptance criteria:
  - `Qwen3.5-9B` sparse features can be extracted from hooked residual states
  - the artifact schema records model, layer, token span, and active-feature summaries
  - unsupported targets can still emit reduced feature summaries without SAE support

### Phase 3 — Steering Actuator Layer
- Objective: define how features can influence inference without losing reversibility.
- Estimated effort: M
- Core files:
  - new `model_scope_steering.py`
  - new `steering_policy.py`
  - `checkpoint_manager.py`
- Deliverables:
  - shadow-mode steering policy configs
  - soft feature-scaling and suppression interfaces
  - intervention provenance and rollback metadata
- Acceptance criteria:
  - all steering policies are fail-closed by default
  - interventions can be replayed and disabled without mutating the base model
  - provenance records show what feature, layer, and policy caused an intervention

### Phase 4 — Segmented Memory And Expectation Comparator
- Objective: make model-scope adaptation stateful without becoming unbounded.
- Estimated effort: L
- Core files:
  - new `model_scope_memory.py`
  - new `expectation_comparator.py`
  - `self_healing_chelation.py`
- Deliverables:
  - typed memory stores for working, episode, expectation, and persistent state
  - comparator rules for baseline-vs-steered and overlay-vs-overlay evaluation
  - replay-set generation hooks
- Acceptance criteria:
  - memory segments have explicit schemas and retention bounds
  - expectation comparison is testable outside live generation
  - replayable episode bundles can be rebuilt from stored artifacts

### Phase 5 — Iterative Training And Promotion Loop
- Objective: train overlays and probe layers from replayed evidence and promote only under evidence.
- Estimated effort: L
- Core files:
  - new `model_scope_trainer.py`
  - new `run_model_scope_campaign.py`
  - `online_updater.py`
  - `chelation_adapter.py`
- Deliverables:
  - overlay-training loop
  - promotion gate for probes, LoRA modules, or steering artifacts
  - rollback and checkpoint integration
- Acceptance criteria:
  - training can update overlays without editing base weights
  - promoted artifacts beat baseline under replay and stress checks
  - failed candidates roll back cleanly

### Phase 6 — ChelatedAI Engine Integration
- Objective: connect Model-Scope to the existing engine, diagnostics, and evaluation surfaces.
- Estimated effort: M
- Core files:
  - `antigravity_engine.py`
  - `run_live_fire_diagnostics.py`
  - `benchmark_multitask.py`
  - `dashboard_server.py`
- Deliverables:
  - engine bridge for model-scope runtime
  - diagnostics and evaluation hooks
  - campaign reports spanning Engine-Scope and Model-Scope
- Acceptance criteria:
  - the engine can run with model-scope observation enabled
  - evaluation harnesses can compare baseline, engine-scope, and model-scope variants
  - dashboards and reports show intervention and promotion evidence

## First Three PRs

### PR 1
- branch: `aep/high/AEP-20260501-PR000-001/model-runtime-hook-bus`
- scope:
  - add the local model runtime and hook bus
  - support at least one pilot target in observation-only mode
  - add deterministic hook tests and artifact reload tests

### PR 2
- branch: `aep/high/AEP-20260501-PR000-002/qwen-scope-feature-codec`
- scope:
  - load Qwen-Scope SAE checkpoints for `Qwen3.5-9B`
  - emit sparse feature events from captured residual states
  - add fallback feature summaries for unsupported targets

### PR 3
- branch: `aep/high/AEP-20260501-PR000-003/steering-shadow-mode`
- scope:
  - add fail-closed steering policy configs
  - implement shadow-mode actuator plumbing and provenance
  - block persistent promotion until replay/comparator evidence exists

## Phase Sequencing Recommendations

1. Do not start with `Qwen3.6` as the primary hook pilot unless verified sparse-feature support is available for the exact target.
2. Do not allow persistent base-weight mutation in the first implementation cycle.
3. Do not build memory promotion before the comparator and replay surfaces exist.
4. Keep Engine-Scope intact; use it as the engine-side counterpart rather than deleting or bypassing it.

## Feasibility Concerns

1. Activation capture volume can grow too quickly; the first event schema should default to compact summaries.
2. SAE checkpoint handling adds model-family coupling; the design must include a fallback non-SAE path.
3. Feature steering may look cleaner on observation tasks than on live generation; shadow mode is required before promotion.
4. “Self-improving” language can overstate what the first cycle can safely do; the real first step is bounded overlay improvement.

## Risk Register
| Risk | Likelihood | Impact | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| Hook runtime is too tightly coupled to one model family | M | H | separate runtime, hook bus, and feature codec interfaces | Codex |
| Activation logging becomes too large for routine runs | H | M | compact event schema, layer filtering, token sampling, and replay subsets | Codex |
| Steering overlays look positive in isolated examples but drift under replay | M | H | require comparator, stress replay, and rollback before promotion | Codex |
| Memory segments become undifferentiated context sprawl | M | H | typed schemas, retention limits, and explicit promotion boundaries | Codex |
| Base-weight mutation pressure arrives too early | M | H | scope lock persistent updates to overlays and artifacts only | Codex |

## Value Alignment Assessment
- Highest-value phase: Phase 1, because no true model-scope work exists until the runtime and hook bus are real.
- Lowest value-to-effort ratio if done too early: Phase 4, because memory without comparator discipline becomes storage without control.
- Recommended MVP scope:
  - Phase 1 complete
  - Phase 2 complete for `Qwen3.5-9B`
  - Phase 3 in shadow mode
  - no persistent base-weight edits

## Rollback Strategy

- hook capture is additive and can be disabled by config
- steering policies begin in shadow mode and fail closed
- promoted artifacts are overlays and probe checkpoints, not base weights
- checkpoints and replay bundles gate every persistence path

## Success Criteria

1. A local Qwen pilot can run under ChelatedAI with deterministic layer hooks.
2. Sparse features can be extracted from those hooks and serialized for replay.
3. Shadow-mode steering can explain what it would have done and why.
4. Persistent promotion is limited to reversible artifacts and beats baseline under replay.
5. The final reports clearly separate observation, steering, memory, and promotion evidence.
