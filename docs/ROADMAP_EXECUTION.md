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
- No GNN layer or disk-pool integration until Phase II steps 12–14 are honestly closed (schema **#277 DONE**; drift apparatus **DONE** with the H5 living-bank verdict a hard negative; disintegration **DONE** — detector-driven Evidence-DAG edge prune, rung-13 PR). Rungs 16 and 17 are now implemented on their feature branches; rung 15 is the remaining endgame feature.

---

## Phase II — Liquified Lattice program

**Starts after:** Phase I step 8 exit criteria met (and step 7 Model-Scope pilot closed or explicitly carried with TTL).  
**Vision doc:** [VISION_LIQUIFIED_LATTICE.md](VISION_LIQUIFIED_LATTICE.md)

Same one-track rule: finish each step before starting the next. Phase II does not reopen parallel doc-only shim cycles.

**Status snapshot (2026-07, git-verified):** rungs **9–13 DONE** (13 completed by the detector-driven
Evidence-DAG edge-prune loop, rung-13 PR); **14 apparatus DONE** with the **H5 living-bank question
closed as a hard negative** (LIVING BANK WINS = False on SciFact + NFCorpus); **16 DONE locally as
an honest non-promotion** (integrated quant-aware plane; both preregistered arenas FAIL-CLOSED; change
set/evidence not yet published); **17 DONE** (block-graph pool-shard read with host parity, rung-17
PR); **15 OPEN**. Lattice apparatus PRs: #260, #277, #279–#291 + rung-13 + rung-17. Remaining
executable feature work is **15 (GNN)** — the endgame program (see
`docs/waypoint-research-2026-06-09/panel/lattice-endgame-plan-2026-07-14.md`).

| Step | Track | Status | Exit criteria |
|------|--------|--------|----------------|
| 9 | **Model-Scope shadow (close Phase I #7)** | **DONE** (fixture path; Model-Scope stack + #254 persist/cap tests) | One bounded steering policy on Qwen3.5-9B fixture; `persist_records` / `load_records` round-trip; bridge `max_total_interventions` exercised in test |
| 10 | **SHIM substrate DoD** | **DONE** (rung 10: #284 A1a / #285 A1b / #286 A1c; DoD honesty #289 — diagnostics observation-only; live routes overlap rung 16) | Promotable steering-route control plane, default-safe, with quant-survival + actionable rollback (`docs/rung10-shim-substrate-dod.md`). This is the rung-10 DoD, **not** a claim that historical SHIM-CD-01/02/06 rows are CLOSED. |
| 11 | **Annealing controller** | **DONE** (#260 engine controller + #280 post-bank schedule; schedule ownership split across two modules) | Temperature schedule(s) drive explore ↔ stabilize; unit tests prove high-T vs low-T behavior; engine wires the controller into sedimentation temperature; post-bank path uses `annealing_schedule` in the C5 lifecycle |
| 12 | **Evidence DAG schema** | **DONE** (#277 `evidence_dag.py` + validator + JSON schema; no GNN) | Typed graph contract over `build_attribution_pool.py` output (nodes: query/cluster/actuator; edges: retrieval/intervention links); JSON schema + validator; no GNN required yet |
| 13 | **Disintegration loop** | **DONE** — detector-driven Evidence-DAG edge prune (rung-13 PR: `evidence_dag_disintegration.py` + `EvidenceDAG.prune_edges`/`reanneal`); post-bank prune/re-anneal also DONE (#279 H5a, wired #281–#283) | `isomer_detector` (per-query strength) + `convergence_monitor` (per-cluster summary) now score Evidence-DAG edge fitness and drive fitness-gated prune + transactional re-anneal, with a before/after fitness artifact and detector provenance. Fail-closed: missing/unmatched/immature signals → neutral fitness (never prunes); non-sedimentation isomer mode is rejected. The original exit criteria (detector-triggered DAG-edge prune with recorded before/after fitness) are met. |
| 14 | **Concept-drift experiment** | **DONE apparatus** (#258–#266 harness, #267 track-0 hygiene; swap arena #268–#276; H3 #287/#288; H4 #291; H5 driver #290). **H5 living-bank VERDICT: FAIL / non-promoted** (SciFact + NFCorpus) | Injected drift + recovery campaigns under `docs/drift-recovery-*.md`. Living annealed post-bank (C5) does **not** beat the frozen static bank + one-shot router gate. Compounding (`compound_cycles=True`) is catastrophic on the single-seed H4 ablation. |
| 15 | **GNN prototype** | **OPEN** (no merged PR; no PyG/DGL code — only a docstring forward-ref in `evidence_dag.py`) | Lightweight GNN over evidence DAG (PyG or DGL); only after steps 12–14 green; must beat flat-pool baseline on drift fixture or fail closed |
| 16 | **Quant-aware shim routing** | **DONE locally / NON-PROMOTED** — `quant_aware_routing.py` integrates `AdapterRouter`, `QuantizationPromotionGate`, retrieval fitness, 40/30/30 leakage isolation, plane-level paired CI, margin fallback, route-usage binding, and opt-in engine serving. GPU campaign: Arena A FAIL-CLOSED (SELECT delta 0.000000, CI [0.000000, 0.000000], quant pass 0.25); Arena B FAIL-CLOSED (delta -0.000899, CI [-0.026005, 0.024604], quant pass 0.75). Both passed REPORT multi-route binding, but neither SELECT gate. No plane is enabled. Change set/evidence still needs durable publication. | `adapter_router.py` + `QuantizationPromotionGate` integrated as steering plane; promotion requires quant survival + retrieval fitness |
| 17 | **Disk pool slice** | **DONE** — one precomputed retrieval pool shard written as a block-graph payload and read back via the real `block_graph.read_block` traversal with a host parity check (rung-17 PR: `computational_storage_poc/pool_shard.py`) | One precomputed pool shard readable via `computational_storage_poc/block_graph.py` with host parity check; documented in storage track docs. Delivered: `write_pool_shard`/`read_pool_shard` (float32 bytes carried losslessly through the FP16-native block format via byte-lane encoding), `verify_pool_shard_parity` (three-way SHA256 + byte-exact vectors + ids, fail-closed), and disk-vs-in-memory top-k equivalence. Grok Tier B 100 (100k random-float32 round-trip, 0 fails). POC/EXPERIMENTAL, not yet wired into live retrieval. |

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
