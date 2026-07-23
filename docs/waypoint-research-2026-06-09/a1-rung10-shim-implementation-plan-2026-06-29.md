# A1 — Rung 10 SHIM Substrate DoD + Rollback Test — Implementation Plan

**Date:** 2026-06-29 · **Status:** investigated + scoped (not yet built) · **Local-only.**
Grounded in a real read of the Model-Scope subsystem; written so A1 can be executed cleanly
(ideally with fresh context) without re-investigating.

## What rung 10 asks (from the goal narrative)
> "Move the shim seams out of env-only research guards into production-wired control planes
> with a rollback test." Gate: "quantization-survival check passes for a promoted shim route."
> Unlocks: hot-swappable, promotable correction routes — the substrate the bank routes over.

## What already EXISTS (do NOT rebuild — verified by reading the code)
- `model_scope_steering.py`:
  - `SteeringActuator.apply()` — applies a policy (SHADOW / SOFT_SCALE / SUPPRESSION) to a
    `SparseFeatureEvent`, emitting a full-provenance `InterventionRecord` (original + modified
    values, mode, caps, decline reasons). Persist/load to JSONL exists.
  - `rollback_feature_event(record, event)` — restores `record.original_values`; raises on an
    unapplied record. **Already well-tested** (`test_model_scope_steering.py:439–498`:
    round-trip, unapplied-raises, non-targeted-preserved, provenance-preserved). The per-event
    rollback PRIMITIVE is done.
  - `ModelScopeShadowSteerer.evaluate_capture()` returns a **hardcoded** `"rollback"` descriptor
    (`:139–143` — `off_switch=True`, `base_weights_mutated=False`,
    `disable_by_setting_deployment_mode="disabled"`). This is a static CLAIM, not an executable
    artifact tied to the actual interventions. **This is the seam A1 hardens.**
- `promotion_contract.py`: `evaluate_promotion_candidate(...)` — comparator-first, fail-closed
  gate over evidence bundles; already has `require_rollback_path` + `rollback_path` handling,
  but only checks the path is a **non-empty string** (`:138–140`) — it does not verify the
  rollback is actionable.
- `quantization_promotion_gate.py` — the quantization-survival gate (the rung-10 gate exists as
  a module; A1 must WIRE it to a promoted steering route, not author it from scratch — read it
  first).
- `steering_policy.py` — `ModelScopeSteeringPolicy`, `PolicyRegistry`, `PolicyStatus`
  (ACTIVE/DISABLED/BLOCKED), `SteeringMode`, `SteeringPolicyConfig`.

## The GAP A1 must close (the rung-10 DoD)
A **production-wired control plane** that takes a steering route from candidate → promoted →
live, with (a) a quantization-survival check on the promoted route, and (b) an executable,
verified rollback (not a hardcoded descriptor). Decomposes cleanly (like S2 did):

### A1a — Executable rollback plan (bounded, no GPU)
Add a `RollbackPlan` built from the actuator's applied `InterventionRecord`s + an
`execute_rollback(plan, events)` that replays `rollback_feature_event` over them, and have
`ModelScopeShadowSteerer` emit the REAL plan (replacing the hardcoded `"rollback"` dict at
`:139–143`). Test: apply a SOFT_SCALE/SUPPRESSION route, build the plan, execute it, assert
the store/event is byte-identical to pre-application (the "base_weights_mutated=False" claim
becomes a *proven* invariant, not a literal). Adversarial Tier-B → PR.

### A1b — Quantization-survival gate wired to a promoted route (read the gate first)
Read `quantization_promotion_gate.py`, then wire it: a promoted steering route must pass the
quantization-survival check before going live. Test: a route that survives quantization passes
the gate; one that does not is fail-closed rejected (mirrors `promotion_contract`'s
fail-closed style). Strengthen `promotion_contract`'s `require_rollback_path` from
"string present" to "rollback plan is actionable" (cross-check against A1a's plan shape).

### A1c — Production-seam un-guarding + DoD doc
Identify the env-only research guards (live-fire flags in the run scripts / safety modules) and
promote the validated seam to a production-wired control plane (default-safe: SHADOW unless a
promoted+quantization-survived+rollback-backed route says otherwise). Write the rung-10 DoD doc.
Gate: a promoted route passes quantization-survival AND has a verified rollback.

## Why this was NOT built tonight (honest)
A1 is a substantial integration across 3+ mature, safety-critical modules. The primitives are
done and well-tested; the rung-10 work is careful WIRING, where a speculative guess at the
integration shape would be an L4 (partial-as-complete) violation. The H5 load-bearing thesis
work (the post-bank apparatus, 7 PRs) was the priority and is complete; A1 is best executed with
fresh context against this plan. No A1 code was written, so nothing is presented as done.
