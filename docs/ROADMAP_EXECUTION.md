# Execution queue (operator priority)

**Effective:** 2026-06-06 (Phase II lattice program added)  
**North-star:** [VISION_LIQUIFIED_LATTICE.md](VISION_LIQUIFIED_LATTICE.md) — self-annealing RAG/DAG with steerable shims and disk-scale pools  
**Policy:** SHIM substrate rows (SHIM-CD-01, 02, 06, 08, 09) are **on hold / last** during Phase I. Core engine and research-validity work runs **one track at a time** until each step meets its exit criteria.

## Block flag

Merge gate: **CLEAR** for this sequence (`docs/next-session.md`). SHIM debts stay **OPEN** but **non-blocking** until the queue below finishes.

## Sequence (do not parallelize)

| Step | Track | Exit criteria |
|------|--------|----------------|
| 1 | **ML correctness — projection** | `DimensionProjection` trains (no erroneous `.detach()` on teacher path); regression test fails if reverted |
| 2 | **ML correctness — InfoNCE** | `SedimentationInfoNCELoss` does not use inter-doc false negatives; tests green |
| 3 | **Infra hygiene** | `isolated_adapter_state()` leak fixed; `*.pt` backup pattern gitignored; no stray backup files |
| 4 | **Sweep / CI cost** | `run_large_sweep.py` O(N²) JSON path fixed or bounded; CI torch cache documented |
| 5 | **Packaging** | Missing `py-modules` in `pyproject.toml` added; `pip install -e .` smoke passes |
| 6 | **Docs truth** | `CLAUDE.md` + `CHANGELOG.md` reflect current entrypoints and smoke paths |
| 7 | **Model-Scope integration** | Shadow steering pilot on Qwen3.5-9B (one policy, provenance persist) per architecture-2026-05-01 phases 4–6 slice |
| 8 | **E2E learning loop** | Single test: ingest → sedate → measurable metric delta on fixture corpus |
| **Last** | **SHIM substrate** | Resume SHIM-CD-01/02/06/08/09 only after step 8; restore `Blocking=YES` when re-entering shim program |

### Execution posture

- Track work one **step** at a time, but do not leave long checks idle: if any test command in a step runs longer than ~30 seconds, launch at least one independent, lower-cost validation in parallel (state checks, docs check, packaging import smoke, or debt table audits).
- Keep the operator queue explicit: no step is considered complete until the long-running validation and the parallel companion check both pass.
- Use this automation to enforce the policy during execution:
  - `bash scripts/run_step_with_checks.sh --always --threshold 30 --companion "python -m unittest tests.test_check_block_flag -v" -- python -m unittest -v tests/test_model_scope_runtime.py`
  - Replace the primary command for each queue step; add step-specific companions as needed (for example, a docs audit or focused evidence smoke) and keep each check independent of the long-running step.

## Automation

- Phase loop prefers **CORE-SLICE-*** handlers (low `priority` number) before **SHIM-SLICE-*** (priority ≥ 200).
- Ten-minute loops: `bash scripts/loop_core_10m.sh` (core track) instead of shim BHS loop unless explicitly requested.

## What we are not doing (while on this queue)

- No new doc-only shim cycles or 10-agent backlog slices.
- No claiming SHIM-CD **CLOSED** without full substrate DoD.
- No Model-Scope + full shim + EGGROLL at full throttle in parallel.
- No GNN layer or disk-pool integration until Phase II steps 12–14 are honestly closed (schema **#277 DONE**; drift apparatus **DONE**, but H2/H4/H5 quantitative evidence is `LEGACY_METRIC_LINEAGE_BLOCKED` and supports no promotion; disintegration **#279 PARTIAL** — do not start step 15 while claiming step 13 fully green without a chair re-scope).

---

## Phase II — Liquified Lattice program

**Starts after:** Phase I step 8 exit criteria met (and step 7 Model-Scope pilot closed or explicitly carried with TTL).  
**Vision doc:** [VISION_LIQUIFIED_LATTICE.md](VISION_LIQUIFIED_LATTICE.md)

Same one-track rule: finish each step before starting the next. Phase II does not reopen parallel doc-only shim cycles.

**Status snapshot (2026-07, git-verified):** rungs **9–12 DONE**; **13 PARTIAL** (post-bank
prune/re-anneal lifecycle only — not isomer/convergence → Evidence-DAG prune); **14 apparatus DONE**
while its H2/H4/H5 exact nDCG values, comparator orderings, and derived gates are
`LEGACY_METRIC_LINEAGE_BLOCKED`. H5 and compounding remain conservatively non-promoted pending
corrected regeneration; neither is a confirmed negative. The complete
`artifacts/legacy-ndcg-quarantine-index-v2.json` inventory fail-closes all 113 affected tracked
artifacts (89 raw JSONs, 8 aggregate JSONs, 6 plots, and 10 prose surfaces); the prior v1 inventory
is retained only as an incomplete predecessor. **15–17 OPEN**. Lattice apparatus PRs:
#260, #277, #279–#291. Next executable feature work is step **15** only after chair re-scopes or
closes the step **13** remainder — or **16/17** if the quant plane / disk pool is prioritized over
GNN.

| Step | Track | Status | Exit criteria |
|------|--------|--------|----------------|
| 9 | **Model-Scope shadow (close Phase I #7)** | **DONE** (fixture path; Model-Scope stack + #254 persist/cap tests) | One bounded steering policy on Qwen3.5-9B fixture; `persist_records` / `load_records` round-trip; bridge `max_total_interventions` exercised in test |
| 10 | **SHIM substrate DoD** | **DONE** (rung 10: #284 A1a / #285 A1b / #286 A1c; DoD honesty #289 — diagnostics observation-only; live routes overlap rung 16) | Promotable steering-route control plane, default-safe, with quant-survival + actionable rollback (`docs/rung10-shim-substrate-dod.md`). This is the rung-10 DoD, **not** a claim that historical SHIM-CD-01/02/06 rows are CLOSED. |
| 11 | **Annealing controller** | **DONE** (#260 engine controller + #280 post-bank schedule; schedule ownership split across two modules) | Temperature schedule(s) drive explore ↔ stabilize; unit tests prove high-T vs low-T behavior; engine wires the controller into sedimentation temperature; post-bank path uses `annealing_schedule` in the C5 lifecycle |
| 12 | **Evidence DAG schema** | **DONE** (#277 `evidence_dag.py` + validator + JSON schema; no GNN) | Typed graph contract over `build_attribution_pool.py` output (nodes: query/cluster/actuator; edges: retrieval/intervention links); JSON schema + validator; no GNN required yet |
| 13 | **Disintegration loop** | **PARTIAL** — post-bank prune/re-anneal DONE (#279 H5a, wired #281–#283); **NOT** isomer/convergence → Evidence-DAG edge prune | Delivered: fitness-gated prune + re-anneal on `SteeringPostBank` with lifecycle artifacts. Original exit named `isomer_detector` / `convergence_monitor` triggers on DAG edges/pool entries — **not implemented**. Re-scope exit to the post-bank mechanism, or implement that trigger. |
| 14 | **Concept-drift experiment** | **DONE apparatus** (#258–#266 harness, #267 track-0 hygiene; swap arena #268–#276; H3 #287/#288; H4 #291; H5 driver #290). Quantitative result lineage: **`LEGACY_METRIC_LINEAGE_BLOCKED`** | Campaign machinery and historical artifacts exist under `docs/drift-recovery-*.md`. The stored H2/H4/H5 nDCG values, comparator orderings, and gates are quarantined because IDCG used retrieved relevance rather than all positive qrels. H5 and compounding remain non-promoted; no accepted fail/win/rejection claim exists until corrected regeneration. |
| 15 | **GNN prototype** | **OPEN** (no merged PR; no PyG/DGL code — only a docstring forward-ref in `evidence_dag.py`) | Lightweight GNN over evidence DAG (PyG or DGL); only after steps 12–14 green; must beat flat-pool baseline on drift fixture or fail closed |
| 16 | **Quant-aware shim routing** | **OPEN** (pieces exist: `adapter_router`, `QuantizationPromotionGate`, route gate #285 — **not** integrated as one retrieval-fitness steering plane) | `adapter_router.py` + `QuantizationPromotionGate` integrated as steering plane; promotion requires quant survival + retrieval fitness |
| 17 | **Disk pool slice** | **OPEN** (no pool-shard parity via `block_graph`; the computational-storage POC is a different graph) | One precomputed pool shard readable via `computational_storage_poc/block_graph.py` with host parity check; documented in storage track docs |

### Phase II dependencies (do not skip)

```text
9 (Model-Scope) → 10 (SHIM DoD)
10 → 11 (annealing controller needs stable steering seams)
11 → 13 (disintegration needs anneal schedule)
12 → 15 (GNN needs DAG schema)
13 + 14 → 15 (GNN needs drift benchmark)
16 depends on 10 + 11
17 depends on 12 + block-graph parity (can run parallel to 15–16 only after 12)
```

### Phase II success metrics

See [VISION_LIQUIFIED_LATTICE.md#success-metrics-program-level](VISION_LIQUIFIED_LATTICE.md#success-metrics-program-level). Promotions require artifact evidence, not prose.
