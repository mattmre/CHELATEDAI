# Model-Scope Steering Architecture

Date: 2026-05-01

## Executive Summary

ChelatedAI now has two distinct control surfaces:

1. `Engine-Scope`, which instruments and steers the retrieval engine.
2. `Model-Scope`, which does not exist yet and would instrument and steer the model itself.

The current codebase is strong on engine-side telemetry, adapters, online updater logic, checkpointing, and advisory self-healing. It does **not** yet have a true activation-hook runtime, a sparse feature interface over hidden states, or a bounded promotion loop for model-internal steering.

The official Qwen-Scope release makes that next step practical. As of **2026-05-01**, official Qwen-Scope artifacts are publicly visible for `Qwen3.5-9B`, `Qwen3-8B`, `Qwen3.5-2B`, `Qwen3-1.7B`, and larger variants. Official `Qwen3.6` models also exist now, with verified open-weight releases on **2026-04-15** (`Qwen3.6-35B-A3B`) and **2026-04-22** (`Qwen3.6-27B`), but the verified Qwen-Scope surfaces available for direct sparse-feature hooking are still on the `Qwen3` and `Qwen3.5` families. That makes `Qwen3.5-9B` the best first serious pilot for a local hook-and-steer build.

## What We Are Actually Building

The target is not another retrieval-only gate. The target is a model wrapper with five core abilities:

1. load a local open model under our own runtime rather than through an opaque API-only path
2. hook specific internal layers during inference
3. map those hidden states into sparse, reusable feature activations
4. apply bounded steering interventions at inference time and during offline training
5. persist only reviewed overlays, probes, memories, and steering artifacts until a stronger promotion loop exists

In repo terms, that means building a new `Model-Scope` layer that sits beside the current Engine-Scope layer and then bridging both into the existing ChelatedAI engine.

## Recommended Target Stack

### Primary pilot

- base model: `Qwen/Qwen3.5-9B`
- feature layer: `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50`
- runtime style: local `transformers`-based loading with explicit forward hooks
- update style: steering overlays, probe heads, LoRA-style adapters, and memory artifacts before any base-weight mutation

### Lightweight debug target

- base model: `Qwen/Qwen3.5-2B` or `Qwen/Qwen3-1.7B`
- purpose: fast local hook validation, smoke tests, and activation/feature pipeline debugging
- note: these are the smallest verified official Qwen-Scope-capable targets currently visible

### Stretch target

- base model: `Qwen/Qwen3.6-27B` or `Qwen/Qwen3.6-35B-A3B`
- purpose: later expansion after the hook bus and steering runtime are stable
- note: use raw residual capture first unless verified official SAE support for the exact target is available

## Architecture Shape

### 1. Base model runtime

This is a new runtime surface that owns tokenizer/model loading, device placement, generation requests, and reproducible inference configuration.

It should not be hidden behind the current `OllamaEmbeddingBackend` path because that path cannot expose internal activations safely or deterministically enough for this work.

### 2. Hook bus

This layer registers hook points and normalizes captured activations.

Required capabilities:

1. select layer indices and hook type
2. capture residual-stream tensors per token step
3. emit summaries or full captures depending on policy
4. support observation-only mode first
5. support reversible intervention handles later

### 3. Feature codec layer

This layer converts raw hidden states into actionable features.

It should support two families:

1. Qwen-Scope SAE feature activations for supported Qwen models
2. fallback raw-activation summaries and probe features for unsupported models

This is the boundary where “hidden state” becomes “steerable feature”.

### 4. Steering actuator layer

This layer decides how features affect inference.

Actuators should be staged:

1. observation only
2. advisory recommendations
3. soft steering via feature scaling, suppression, or routing bias
4. train-time objectives over adapters, probe heads, or LoRA modules

Direct base-weight mutation should remain out of the first implementation tranche.

### 5. Segmented memory layer

Your memory idea makes sense if it is turned into typed, bounded stores instead of a vague “persistent memory” concept.

The first useful split is:

1. working memory: per-request or per-turn transient state
2. episode memory: replayable traces, prompts, activations, outcomes, and comparator signals
3. persistent memory: promoted steering artifacts, probe checkpoints, feature dictionaries, replay families, and accepted overlays
4. expectation memory: target behaviors, forbidden regressions, and comparator baselines used to judge later runs

### 6. Expectation and comparator layer

This is the real “self-improving” boundary.

The system should compare:

1. expected behavior vs observed behavior
2. steered run vs baseline run
3. new overlay vs prior promoted overlay
4. new memory segment vs previous segment family

Nothing becomes persistent because it “seemed better once”. It becomes persistent only if the comparator says it beat the baseline under replay, stress, and rollback checks.

### 7. Promotion and rollback layer

The current repo already has useful building blocks here:

1. `checkpoint_manager.py`
2. `online_updater.py`
3. `chelation_adapter.py`
4. `self_healing_chelation.py`
5. `stability_tracker.py`

The new system should reuse those patterns but shift the object under management from “embedding adapter only” to “model-scope steering artifacts”.

## What Must Stay Bounded

If this system is called “autonomous”, it still needs hard boundaries:

1. time budget
2. step budget
3. memory budget
4. replay-set budget
5. promotion gate
6. rollback path

That is the right interpretation of your “bounded concurrency to time and effort” idea. The loop can search and adapt aggressively, but its consequences must stay reversible until promotion evidence is strong.

## Recommended Claim Boundary

The first implementation should claim:

1. local model hooking
2. sparse feature extraction
3. fail-closed steering overlays
4. segmented memory and replay
5. bounded iterative training and promotion

It should **not** initially claim:

1. autonomous safe base-model rewriting
2. always-on self-modifying behavior
3. unrestricted long-horizon persistent memory
4. model-family-agnostic sparse steering from day one

## Bottom Line

The next serious program should be called `Model-Scope`, not just another extension of Engine-Scope.

The best primary pilot is `Qwen3.5-9B` because it matches the desired local size class and has verified official Qwen-Scope sparse-feature support. The right first implementation is a hook bus plus sparse feature layer plus bounded steering overlays, with persistent promotion limited to reversible artifacts until the comparator and replay loop prove they are safe and useful.
