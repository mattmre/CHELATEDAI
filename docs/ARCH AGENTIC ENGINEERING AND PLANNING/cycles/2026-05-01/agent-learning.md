# Agent Learning — 2026-05-01 Model-Scope Cycle

## Notes

- Treat `Engine-Scope` and `Model-Scope` as separate but complementary layers. Do not collapse them into one abstraction.
- The safest first promotion surface is not base weights. It is overlays, probes, steering vectors, replay families, and typed memory artifacts.
- A real hook runtime is a new subsystem, not a small extension of `embedding_backend.py`.
- The pilot-model choice should track verified hook and sparse-feature support, not just raw benchmark popularity.
- Keep Model-Scope hooks out of `EmbeddingBackend`; the embedding contract is too narrow for activation capture and steering.
- Keep recursive observation fail-closed. The current bridge suppresses nested captures during reformulation and adapter-routing recursion to avoid artifact duplication and state leakage.
- Compact feature summaries are the right default. Persist full raw activations only behind an explicit later policy because the diagnostics path and artifact footprint are optimized for bounded JSON artifacts.
- Shadow steering should be recorded before it is applied. The current `runtime_applied = false` boundary is a useful guardrail because it preserves replayability and lets later promotion logic compare “would steer” against “did steer”.
- Typed Model-Scope memory should store compact summaries plus artifact references, not raw tensor payloads inline. The current memory/comparator path stays bounded because it records replayable JSON summaries rather than full activations.
- Replay labels can be bootstrapped from expectation comparisons. That is enough to learn bounded advisory steering policies before any live actuator path exists.
- `CheckpointManager` is usable for early policy-file promotion, but it is still single-file oriented. Multi-artifact atomic promotion remains the next real risk once overlays, probes, and optimizer state need to move together.
- If `model_scope` is not threaded into `IntegratedDiagnosticsReport`, the richest saved benchmark and live-fire artifacts silently drop the new subsystem. Preserve that field explicitly in every report path that claims to be authoritative.
