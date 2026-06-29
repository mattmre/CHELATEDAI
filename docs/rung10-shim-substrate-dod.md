# Rung 10 — SHIM Substrate: Definition of Done

The Liquified Lattice rung-10 substrate is the **promotable, rollback-backed steering route**
that the correction bank routes over. Rung 10 moves Model-Scope steering from an env/CLI
research guard (`run_live_fire_diagnostics.py --enable-model-scope` — a binary "on") to a
**production-wired control plane** where a route earns live mode by passing a promotion gate,
and any live route is reversible.

## DoD — a steering route may run LIVE in production iff:

1. **Quantization-survival.** The route's fitness GAIN survives quantization — its
   retained-gain ratio meets the `QuantizationPromotionGate` threshold. (Gains that vanish
   under quantization are fail-closed rejected.)
2. **Actionable rollback.** The route carries an **executable** rollback — an A1a
   `RollbackPlan` (not merely a string path) that, when executed, restores the exact
   pre-steering feature values. Steering operates only on `SparseFeatureEvent` activations and
   never mutates base model weights, so the rollback is lossless.
3. **Default-safe control plane.** A live mode (`SOFT_SCALE` / `SUPPRESSION`) is permitted only
   when (1) and (2) hold (the route is *promotable*); otherwise the route is **downgraded to
   `SHADOW`** (observe-only, no mutation). `SHADOW` is always allowed.

## How it composes (all landed, BHS 100)

| Piece | Module | Role |
|-------|--------|------|
| A1a | `model_scope_steering.py` (`RollbackPlan` / `build_rollback_plan` / `execute_rollback`) | executable, reverse-order rollback of applied interventions |
| A1b | `steering_route_promotion.py` (`evaluate_steering_route_promotion`) | fail-closed promotion gate: quantization-survival **AND** actionable rollback |
| A1c | `production_steering_control.py` (`resolve_production_mode`) | the production control plane: live mode iff promotable, else default-safe SHADOW |

A route's production mode is therefore: `resolve_production_mode(requested_mode,
evaluate_steering_route_promotion(...))`. Every gate is fail-closed; the default outcome of
anything unproven is `SHADOW`.

## Verification (runtime evidence, not just tests)

- A1a: `execute_rollback` restores the EARLIEST pre-steering value across chained interventions
  (reverse-order), proven against a forward-order fault. 6 tests + 56 existing steering tests
  green (additive).
- A1b: no false-positive promotion path; the quant gate's own reasons are surfaced; the strict
  `isinstance` guard rejects string-path impostors. 8 tests.
- A1c: a promoted route keeps its live mode; a route failing quantization (or with no decision)
  is downgraded to SHADOW. 6 tests, including end-to-end against a real A1b decision.

## Remaining operational step (NOT a code change in this slice — honest scope)

The substrate (the control-plane authority above) is complete and default-safe. The final
operational adoption — having `run_live_fire_diagnostics.py --enable-model-scope` consult
`resolve_production_mode(...)` for each route instead of treating the flag as an unconditional
"on" — is a one-line wiring change to that campaign script, deferred from this slice so it is
not rushed into a run path. Until then, live steering remains behind the existing research flag;
the production control plane is available for adoption and is the documented authority.
