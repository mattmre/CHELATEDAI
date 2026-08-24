# Drift Recovery — H4 compound-cycles ablation — July 2026

> [!WARNING]
> **LEGACY_METRIC_LINEAGE_BLOCKED (2026-07-23).** Exact nDCG values,
> comparator ordering, and derived gates below are historical diagnostics, not
> accepted evidence, until corrected regeneration. See
> `docs/research/metric-lineage-repair-protocol-2026-07.md`.

Arena: `query_encoder_swap`. Task SciFact, condition **C4a** (supervised unbounded adapter),
seed 42, cycles 12, swap `all-MiniLM-L6-v2 → all-mpnet-base-v2`, correction 30 steps @ lr 0.01.

Question (H4 "make cycles compound"): does re-training the adapter from accumulated weights and
applying each correction to the already-mutated store (`compound_cycles=True`) improve recovery over
the idempotent one-shot fixed point (`compound_cycles=False`)?

| `compound_cycles` | Final NDCG@10 (seed 42) | Artifact |
|---|---:|---|
| False (idempotent fixed point) | **0.236297** | `experiment_runs/drift-recovery/h4-compound/C4a_compound0_seed42.json` |
| True (compounding) | **0.005258** | `experiment_runs/drift-recovery/h4-compound/C4a_compound1_seed42.json` |

**Result: compounding is catastrophically worse — recovery collapses ~45× (0.236 → 0.005), to near the
frozen floor.** This quantifies the paper §5.2 "overshoot" observation: applying each bounded
correction to the already-mutated store escalates the per-cycle correction norm cycle-over-cycle
(unstable drift, not convergence), so the idempotent re-initialization from a fixed seed against the
frozen pre-drift snapshot is the correct, stable design — not a failure to compound. This is a
single-seed ablation (seed 42); the direction is unambiguous but the exact magnitude is one seed.
