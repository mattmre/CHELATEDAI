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
- No GNN layer or disk-pool integration until Phase II steps 12–14 complete (schema + drift experiment first).

---

## Phase II — Liquified Lattice program

**Starts after:** Phase I step 8 exit criteria met (and step 7 Model-Scope pilot closed or explicitly carried with TTL).  
**Vision doc:** [VISION_LIQUIFIED_LATTICE.md](VISION_LIQUIFIED_LATTICE.md)

Same one-track rule: finish each step before starting the next. Phase II does not reopen parallel doc-only shim cycles.

| Step | Track | Exit criteria |
|------|--------|----------------|
| 9 | **Model-Scope shadow (close Phase I #7)** | One bounded steering policy on Qwen3.5-9B fixture; `persist_records` / `load_records` round-trip; bridge `max_total_interventions` exercised in test |
| 10 | **SHIM substrate DoD** | SHIM-CD-01/02/06 production seams wired without env-only guards; rollback test; restore `Blocking=YES` on open SHIM rows when re-entering |
| 11 | **Annealing controller** | Single module owns temperature schedule (explore ↔ stabilize); wired to sedimentation + `online_updater` + ES entrypoint; unit test proves high-T increases perturbation, low-T reduces it |
| 12 | **Evidence DAG schema** | Typed graph contract over `build_attribution_pool.py` output (nodes: query/cluster/actuator; edges: retrieval/intervention links); JSON schema + validator; no GNN required yet |
| 13 | **Disintegration loop** | `isomer_detector` / `convergence_monitor` triggers prune of low-fitness graph edges or pool entries; re-anneal path records fitness before/after in artifact |
| 14 | **Concept-drift experiment** | Injected drift fixture (extend `chelatedai-synthetic-collapse` or road-course window); measurable recovery within N anneal cycles documented in `CHANGELOG.md` |
| 15 | **GNN prototype** | Lightweight GNN over evidence DAG (PyG or DGL); only after steps 12–14 green; must beat flat-pool baseline on drift fixture or fail closed |
| 16 | **Quant-aware shim routing** | `adapter_router.py` + `QuantizationPromotionGate` integrated as steering plane; promotion requires quant survival + retrieval fitness |
| 17 | **Disk pool slice** | One precomputed pool shard readable via `computational_storage_poc/block_graph.py` with host parity check; documented in storage track docs |

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
