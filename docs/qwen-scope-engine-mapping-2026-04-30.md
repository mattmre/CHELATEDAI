# Qwen-Scope Engine Mapping

Date: 2026-04-30

## What Was Fixed

The live autopilot defects identified in the previous review are now fixed:

1. Reformulation collection now rotates across the full phase template set instead of repeatedly hitting only the first three windows. This change is in [run_golden_default_autopilot.py](../run_golden_default_autopilot.py).
2. Reformulation validation now uses explicit query offsets, so the holdout slice is no longer silently reusing the first judged queries from collection windows.
3. Recommendation logic now requires a real candidate to survive thresholds and blocker checks before escalating to wider validation.
4. Dedicated tests now cover the live `150 / 50` reformulation loop shape and the offset-aware validation path.

These fixes matter because any conclusion about a "golden default" is only useful if the search loop is actually exploring the intended surface and validating on clean holdouts.

## What Qwen-Scope Actually Contributes

The Qwen-Scope paper is not mainly about post-hoc explanations. Its main contribution is turning sparse internal features into a reusable development interface.

The paper uses the same feature layer for five practical jobs:

1. Steering: identify a behavior-linked internal feature, then amplify or suppress it at inference time.
2. Evaluation: use activated-feature coverage as a cheap proxy for benchmark redundancy and benchmark overlap.
3. Classification: build simple, transparent rule-based detectors from discriminative internal features.
4. Data synthesis: generate examples that are selected for activating target internal features, rather than only matching surface descriptions.
5. Training: add feature-targeted losses or feature-induced negative samples to make rare failures easier to suppress.

The most important result for this repo is not "we should use SAEs." It is this workflow pattern:

1. Find sparse internal signals that separate good and bad behavior.
2. Use those signals to build fail-closed interventions.
3. Use the same signals for coverage analysis and hard-example generation.
4. Prefer negative-failure targeting before attempting broad positive steering.

That last point matches what we already observed locally. Qwen-Scope explicitly reports that positive steering was not the productive RL path, while synthetic rare negative augmentation was.

## What This Means For ChelatedAI

Our current learned reformulation and learned mask gates are conceptually adjacent to Qwen-Scope, but they are still working from shallow query lexical features. That is useful for a first fail-closed baseline, but it is not the right long-term control surface.

The closer ChelatedAI analogue to SAE features is an internal engine feature layer built from the retrieval and control pipeline itself:

1. Query-level attribution rows.
2. Per-query action decisions.
3. Control diagnostics such as variance, jaccard drift, mask density, reformulation-changed flags, and ranking deltas.
4. Doc-level attribution or contribution signals.
5. Counterfactual profile deltas between baseline, guard, reform, and mask branches.

In other words, the paper suggests that the next useful system here is not "stronger lexical gating." It is an "Engine-Scope" layer: sparse, inspectable, reusable internal signals over the retrieval pipeline.

## Proposed Golden Path

The likely positive-default path now looks like this:

1. Treat internal engine telemetry as the primary feature space.
2. Discover discriminative features by contrasting positive queries against active-negative and active-neutral failures.
3. Train fail-closed gates on those internal features before trying broader always-on policy changes.
4. Use feature coverage to decide which tasks, offsets, and query families add new information, and which are redundant.
5. Generate synthetic hard negatives that explicitly activate known failure features.
6. Only attempt positive default promotion after a gate survives clean holdout without active-negative blockers.

This is a better fit than continued global threshold sweeps because the current problem is not parameter search depth. The problem is missing internal supervision about when an intervention should activate.

## Concrete Repo Mapping

The current repo already has most of the scaffolding needed for an Engine-Scope pass:

1. `query_reformulator.py` already computes lightweight pre-retrieval features and supports structured policy configs.
2. `learned_reformulation_gate.py` and `learned_mask_gate.py` already support pooled fail-closed classifier training.
3. `static_mask_probe.py` already emits per-query mask examples and can train simple conditional gates.
4. `run_golden_default_autopilot.py` already pools data, trains gates, validates them, and promotes only under evidence.

What is missing is the feature source. Right now the gates are mostly learning from lexical query shape. Qwen-Scope suggests moving that learning target one layer inward, toward internal decision variables and contribution traces.

## Next 1-2 Day Build Plan

### 1. Add an Engine-Scope row emitter

Emit one row per evaluated query/profile with:

1. query id, task, offset, seed, profile
2. baseline metric and candidate metric deltas
3. action type
4. reformulation changed count, variant count
5. jaccard drift, variance stats, mask density
6. attribution summary features such as top-doc contribution concentration and attribution entropy
7. fault class and promotion blocker labels

This becomes the pooled internal-feature dataset for all later work.

### 2. Add feature-coverage analysis for road-course data

Mirror the Qwen-Scope benchmark-overlap idea:

1. compute feature footprints per task, offset band, and query family
2. estimate which windows are redundant
3. prioritize windows with low overlap against the current pool

This should replace blind query accumulation with coverage-aware accumulation.

### 3. Train internal fail-closed gates

Train reform and mask gates on Engine-Scope rows first, not just lexical features. Keep the same fail-closed promotion logic:

1. no active-negative blockers
2. positive mean delta above threshold
3. repeatability on unseen offsets

### 4. Generate synthetic hard negatives

Qwen-Scope's strongest practical lesson is targeted negative generation. For this repo, that means:

1. identify failure-feature signatures for active-negative reformulation and masking cases
2. generate or mine more queries with those signatures
3. use them to stress the gates and the self-healing branch

### 5. Try targeted training only after the feature layer exists

If we later want an auxiliary loss or a self-healing objective, it should penalize known failure-feature activations rather than generic global changes. Without the feature layer, that training signal is still too blunt.

## Bottom Line

The paper is likely pointing at the right architecture pattern, but not because we should literally import Qwen's SAEs into this engine. The useful transfer is the control philosophy:

1. build an internal feature interface,
2. use it for diagnosis,
3. use it for fail-closed gating,
4. use it for coverage-aware evaluation,
5. use it for synthetic rare negative generation,
6. only then ask whether a positive default exists.

The immediate recommendation is to make the next autopilot pass "Engine-Scope first" rather than "more lexical gate sweeps."
