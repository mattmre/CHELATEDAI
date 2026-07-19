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
- No GNN layer while calling rung 15 done. Schema **#277** is on main. Drift apparatus is on main. The H5 living-bank verdict landed on main in PR #292 (it was not on main before that merge) and it is a hard negative — do not promote the living bank. Disintegration already on main is **#279 PARTIAL** (post-bank prune/re-anneal only). Rung 13's detector-to-DAG prune is on main via merged PR #293, not introduced by this PR. Rung 17's disk-pool slice is on main via merged PR #294, not introduced by this PR. Rung 16's quant-aware routing plane is in this PR (#295), not already merged. Do not call rung 15 done. Do not call rung 16 already merged.

---

## Phase II — Liquified Lattice program

**Starts after:** Phase I step 8 exit criteria met (and step 7 Model-Scope pilot closed or explicitly carried with TTL).  
**Vision doc:** [VISION_LIQUIFIED_LATTICE.md](VISION_LIQUIFIED_LATTICE.md)

Same one-track rule: finish each step before starting the next. Phase II does not reopen parallel doc-only shim cycles.

**Status snapshot (2026-09-22, rebased onto `origin/main` `8e6e83b`):** rungs **9–12 DONE**
and already on main. Rung **14 apparatus** is already on main; the **H5 living-bank verdict is on
main via PR #292**, not introduced here (hard negative — do not promote the living bank). Rung **13**
post-bank prune/re-anneal is on main (#279). Rung 13's detector-to-DAG prune is on main via merged
PR #293, not introduced by this PR. Rung **17** (disk pool slice) is on main via merged PR #294,
not introduced by this PR. Rung **15 is not done**. Rung **16** (quant-aware routing) is **in this
PR (#295), not already merged** — an honest non-promotion (both preregistered arenas FAIL-CLOSED).
Do not call rung 15 done. Do not call rung 16 already merged. Historical SHIM-CD rows are not CLOSED.

| Step | Track | Status | Exit criteria |
|------|--------|--------|----------------|
| 9 | **Model-Scope shadow (close Phase I #7)** | **DONE** (fixture path; Model-Scope stack + #254 persist/cap tests) | One bounded steering policy on Qwen3.5-9B fixture; `persist_records` / `load_records` round-trip; bridge `max_total_interventions` exercised in test |
| 10 | **SHIM substrate DoD** | **DONE** (rung 10: #284 A1a / #285 A1b / #286 A1c; DoD honesty #289 — diagnostics observation-only; live routes overlap rung 16) | Promotable steering-route control plane, default-safe, with quant-survival + actionable rollback (`docs/rung10-shim-substrate-dod.md`). This is the rung-10 DoD, **not** a claim that historical SHIM-CD-01/02/06 rows are CLOSED. |
| 11 | **Annealing controller** | **DONE** (#260 engine controller + #280 post-bank schedule; schedule ownership split across two modules) | Temperature schedule(s) drive explore ↔ stabilize; unit tests prove high-T vs low-T behavior; engine wires the controller into sedimentation temperature; post-bank path uses `annealing_schedule` in the C5 lifecycle |
| 12 | **Evidence DAG schema** | **DONE** (#277 `evidence_dag.py` + validator + JSON schema; no GNN) | Typed graph contract over `build_attribution_pool.py` output (nodes: query/cluster/actuator; edges: retrieval/intervention links); JSON schema + validator; no GNN required yet |
| 13 | **Disintegration loop** | **ON MAIN via merged PR #293.** Post-bank prune/re-anneal is on main (#279 H5a, wired #281–#283). Detector-to-DAG prune (`evidence_dag_disintegration.py` + `EvidenceDAG.prune_edges` / `reanneal_edges`) landed in #293. Not introduced by this PR. | #293 scores Evidence-DAG edges from `isomer_detector` (per-query strength) and `convergence_monitor` (per-cluster summary), with fitness-gated prune, transactional re-anneal, a before/after fitness artifact, and detector provenance. Fail-closed: missing/unmatched/immature signals → neutral fitness (never prunes); non-sedimentation isomer mode is rejected. |
| 14 | **Concept-drift experiment** | **DONE apparatus** already on main (#258–#266 harness, #267 track-0 hygiene; swap arena #268–#276; H3 #287/#288; H4 knob #291; H5 driver #290). **H5 living-bank verdict is on main via PR #292, not introduced by this PR: FAIL / non-promoted** (SciFact + NFCorpus). Do not promote the living bank. | Injected drift + recovery campaigns under `docs/drift-recovery-*.md`. Living annealed post-bank (C5) does **not** beat the frozen static bank + one-shot router gate. Compounding (`compound_cycles=True`) is a rejected single-seed H4 result. |
| 15 | **GNN prototype** | **OPEN — not done.** No merged PR on main. | Lightweight GNN over evidence DAG (PyG or DGL); only after steps 12–14 are honestly closed; must beat flat-pool baseline on drift fixture or fail closed. Do not call rung 15 done. |
| 16 | **Quant-aware shim routing** | **IN THIS PR (#295), NOT ALREADY MERGED.** `quant_aware_routing.py` integrates `AdapterRouter`, `QuantizationPromotionGate`, retrieval fitness, 40/30/30 leakage isolation, plane-level paired CI, margin fallback, route-usage binding, and opt-in engine serving. GPU campaign: Arena A FAIL-CLOSED (SELECT delta 0.000000, CI [0.000000, 0.000000], quant pass 0.25); Arena B FAIL-CLOSED (delta -0.000899, CI [-0.026005, 0.024604], quant pass 0.75). Both passed REPORT multi-route binding, but neither SELECT gate. No plane is enabled. Honest non-promotion; do not call rung 16 already merged. | `adapter_router.py` + `QuantizationPromotionGate` integrated as steering plane; promotion requires quant survival + retrieval fitness. Do not call rung 16 already merged. |
| 17 | **Disk pool slice** | **ON MAIN via merged PR #294.** Not introduced by this PR. | One precomputed pool shard readable via `computational_storage_poc/block_graph.py` with host parity check; documented in `docs/rung17-disk-pool-slice.md`. PR #294 added `write_pool_shard` / `read_pool_shard` (float32 bytes carried through the FP16-native block format via byte-lane encoding), `verify_pool_shard_parity` (three-way SHA256 plus byte-identical vector bytes, including NaN payloads, and ids, fail-closed), and `retrieve_topk`. POC/EXPERIMENTAL, not wired into live retrieval. |

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
