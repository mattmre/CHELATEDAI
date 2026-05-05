# Change Log — 2026-05-01 Model-Scope Cycle

## Entries

- 2026-05-01 - opened a new Model-Scope cycle to separate true model-hook steering work from the earlier Engine-Scope precursor
- 2026-05-01 - selected `Qwen3.5-9B` as the primary pilot because official Qwen-Scope support is verified there and aligns with the desired local size class
- 2026-05-01 - deferred persistent base-weight mutation from the initial cycle and constrained promotion to overlays, probes, and typed memory artifacts
- 2026-05-01 - implemented Phase 1 Model-Scope foundation: local runtime, hook bus, versioned observation artifacts, and fail-closed engine integration
- 2026-05-01 - implemented Phase 2 sparse feature surface: official Qwen-Scope SAE checkpoint adapter plus fallback last-token activation feature summaries
- 2026-05-01 - implemented Phase 3 shadow steering surface: fail-closed steering policy contracts, advisory steering summaries in runtime artifacts, and engine diagnostics propagation with no runtime mutation
- 2026-05-01 - implemented Phase 4 Model-Scope state surfaces: typed segmented memory, expectation profiles, offline comparator rules, replay-bundle construction, and runtime expectation propagation
- 2026-05-01 - implemented Phase 5 bounded promotion surfaces: replay-trained shadow policies, checkpoint-backed policy promotion, and a local Model-Scope campaign runner
- 2026-05-01 - extended the reporting path so integrated diagnostics now preserve `model_scope` payloads instead of dropping them at the runtime boundary
- 2026-05-01 - declared `transformers` as an explicit direct dependency because Model-Scope now imports it intentionally rather than relying on transitive installation
