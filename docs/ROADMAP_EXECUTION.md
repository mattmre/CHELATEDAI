# Execution queue (operator priority)

**Effective:** 2026-06-06 (Phase II lattice program added)  
**North-star:** [VISION_LIQUIFIED_LATTICE.md](VISION_LIQUIFIED_LATTICE.md) — self-annealing RAG/DAG with steerable shims and disk-scale pools  
**Policy:** SHIM substrate rows (SHIM-CD-01, 02, 06, 08, 09) are **on hold / last** during Phase I. Core engine and research-validity work runs **one track at a time** until each step meets its exit criteria.

## Block flag

Merge gate: **CLEAR** on this commit. `python3 scripts/check_block_flag.py` reports zero OPEN carried-debt rows in `docs/next-session.md` (22 data rows, every Status cell starts with CLOSED). CHANGELOG names seven historical ids, not eight: SHIM-CD-01, SHIM-CD-02, SHIM-CD-06, SHIM-CD-08, SHIM-CD-09, SHIM-CD-03, and SHIM-CD-07. SHIM-CD-05 is already marked CLOSED in CHANGELOG. SHIM-CD-04 does not occur. Those seven names are not rows in the carried-debt table, so the checker does not count them. This sentence does not close them and does not add debt rows.

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
- Those commands are not runnable on this commit: `scripts/run_step_with_checks.sh`, `scripts/loop_core_10m.sh`, and `scripts/phase_development_loop.py` are absent. The example also names `tests/test_model_scope_runtime.py`, which is not in this commit. `test_model_scope_runtime.py` is at the repository root and was not re-run. `tests/test_check_block_flag.py` is present and was not re-run.

## What we are not doing (while on this queue)

- No new doc-only shim cycles or 10-agent backlog slices.
- No claiming SHIM-CD **CLOSED** without full substrate DoD.
- No Model-Scope + full shim + EGGROLL at full throttle in parallel.
- No GNN layer or disk-pool integration while calling rung 15 done or treating rung 16 as on this commit. Schema **#277** is on main (`81e3614bec7d5e5b9cce57af8e7126c6b97775dc` is an ancestor). Drift apparatus is on main. The H5 living-bank verdict landed on main in PR #292 (`c148f7b3c1e3858b461f6c08ddb3e58cd7e99a85`) and it is a hard negative — do not promote the living bank. Disintegration on main includes **#279** post-bank prune/re-anneal (`06f4f5cced82fb37547b1a7a36ee34f0dc9451f6`). Rung 13's detector-to-DAG prune is on main via merged PR #293 (`782ab62ddbaf6ab6c40085255e48ff9a21562d34`). Rung 17's disk-pool slice is on main via merged PR #294 (`8e6e83b7c30bae34015ad12996314b5eea2d1c64`). Rung 15 is OPEN, not done, and not refused. Rung 16 is not on this commit.

---

## Phase II — Liquified Lattice program

**Starts after:** Phase I step 8 exit criteria met (and step 7 Model-Scope pilot closed or explicitly carried with TTL). Step 8's test `tests/test_learning_loop_e2e.py` is added by this change. It fails when the ranked-id list does not change, and it does not use a production encoder. Phase I is not complete. The snapshot below records Phase II rungs that are already on main.  
**Vision doc:** [VISION_LIQUIFIED_LATTICE.md](VISION_LIQUIFIED_LATTICE.md)

Same one-track rule: finish each step before starting the next. Phase II does not reopen parallel doc-only shim cycles.

**Status snapshot (this commit `49804ae3ddcebf4e4060fae95ba812331299f757`, which is `origin/main`):** rungs **9–12** are on main (merge commits checked as ancestors: #260 `f0c643ae7bf5a7c7ca612ceaa878b7cb15412540`, #280 `9db098ea0d2f7df91b54d29e5ee1c6eb9ea65bb8`, #277 `81e3614bec7d5e5b9cce57af8e7126c6b97775dc`, #284 `efe6d1be55b5c702eb3a1cccde3169feb04b6c34`, #285 `49b36046b6d59a554633f631924005130a05cd1f`, #286 `d8181f64161522eb39f8b0e1fd9e48aa0fa536b2`, #289 `d1bfc0903d7de5c2b81d41a7d081e7da8502ad40`). Rung **14 apparatus** commits checked as ancestors include #279 `06f4f5cced82fb37547b1a7a36ee34f0dc9451f6`, #290 `6d3af3009c29b31ce72244c13b53c02170bfbd61`, and #291 `34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2`. The **H5 living-bank verdict is on main via merged PR #292** (`c148f7b3c1e3858b461f6c08ddb3e58cd7e99a85`; hard negative — do not promote the living bank). Rung **13** post-bank prune/re-anneal is on main (#279). Rung 13's detector-to-DAG prune is on main via merged PR #293 (`782ab62ddbaf6ab6c40085255e48ff9a21562d34`). Rung **15** is OPEN, not done, and not refused (no merged GNN-prototype PR; `gh pr list --state merged --search "GNN prototype"` returned no rows). Rung **16** is not on this commit (PR #295 is open; head `d2393942119a69a79ee30a1807eef5d5694906da` is not an ancestor). Rung **17** is on main via merged PR #294 (`8e6e83b7c30bae34015ad12996314b5eea2d1c64`). Do not call rung 15 done. Historical SHIM-CD names are not CLOSED by this snapshot.

| Step | Track | Status | Exit criteria |
|------|--------|--------|----------------|
| 9 | **Model-Scope shadow (close Phase I #7)** | **DONE** (fixture path; Model-Scope stack + #254 persist/cap tests) | One bounded steering policy on Qwen3.5-9B fixture; `persist_records` / `load_records` round-trip; bridge `max_total_interventions` exercised in test |
| 10 | **SHIM substrate DoD** | **DONE** (rung 10: #284 A1a / #285 A1b / #286 A1c; DoD honesty #289 — diagnostics observation-only). `docs/rung10-shim-substrate-dod.md` line 55 says introducing live routes is a future feature that overlaps rung 16. Rung 16 is not on this commit. | Promotable steering-route control plane, default-safe, with quant-survival + actionable rollback (`docs/rung10-shim-substrate-dod.md`). This is the rung-10 DoD, **not** a claim that historical SHIM-CD-01/02/06 rows are CLOSED. |
| 11 | **Annealing controller** | **DONE** (#260 engine controller + #280 post-bank schedule; schedule ownership split across two modules) | Temperature schedule(s) drive explore ↔ stabilize; unit tests prove high-T vs low-T behavior; engine wires the controller into sedimentation temperature; post-bank path uses `annealing_schedule` in the C5 lifecycle |
| 12 | **Evidence DAG schema** | **DONE** (#277 `evidence_dag.py` + validator + JSON schema; no GNN) | Typed graph contract over `build_attribution_pool.py` output (nodes: query/cluster/actuator; edges: retrieval/intervention links); JSON schema + validator; no GNN required yet |
| 13 | **Disintegration loop** | **ON MAIN via merged PR #293.** Post-bank prune/re-anneal is on main (#279 H5a, wired #281–#283). Detector-to-DAG prune (`evidence_dag_disintegration.py` + `EvidenceDAG.prune_edges` / `reanneal_edges`) landed in #293 and is already on this commit. This docs edit does not add that code. | #293 scores Evidence-DAG edges from `isomer_detector` (per-query strength) and `convergence_monitor` (per-cluster summary), with fitness-gated prune, transactional re-anneal, a before/after fitness artifact, and detector provenance. Fail-closed: missing/unmatched/immature signals → neutral fitness (never prunes); non-sedimentation isomer mode is rejected. |
| 14 | **Concept-drift experiment** | **DONE apparatus** already on main (#258–#266 harness, #267 track-0 hygiene; swap arena #268–#276; H3 #287/#288; H4 knob #291; H5 driver #290). **H5 living-bank verdict is on main via PR #292, not introduced by this PR: FAIL / non-promoted** (SciFact + NFCorpus). Do not promote the living bank. | Injected drift + recovery campaigns under `docs/drift-recovery-*.md`. Living annealed post-bank (C5) does **not** beat the frozen static bank + one-shot router gate. Compounding (`compound_cycles=True`) is a rejected single-seed H4 result. |
| 15 | **GNN prototype** | **OPEN — not done, not refused.** No merged GNN-prototype PR (`gh pr list --state merged --search "GNN prototype"` returned no rows). | Lightweight GNN over evidence DAG (PyG or DGL); only after steps 12–14 are honestly closed; must beat flat-pool baseline on drift fixture or fail closed. Do not call rung 15 done. Do not call rung 15 refused. |
| 16 | **Quant-aware shim routing** | **Not on this commit.** PR #295 is open and not merged (head `d2393942119a69a79ee30a1807eef5d5694906da` is not an ancestor). | `adapter_router.py` + `QuantizationPromotionGate` integrated as steering plane; promotion requires quant survival + retrieval fitness. This cell does not state a campaign result. Rung 16 is not on this commit. |
| 17 | **Disk pool slice** | **ON MAIN via merged PR #294** (`8e6e83b7c30bae34015ad12996314b5eea2d1c64`). | One precomputed pool shard readable via `computational_storage_poc/block_graph.py` with host parity check; documented in `docs/rung17-disk-pool-slice.md`. On this commit `computational_storage_poc/pool_shard.py` defines `write_pool_shard` (line 100), `read_pool_shard` (line 241), and `verify_pool_shard_parity` (line 286), plus `retrieve_topk` (line 331). The module stores float32 bytes as exact FP16 cells (`BYTE_ENCODING` at line 29; NaN byte-identity comment at lines 303–305; at lines 297–301, two `_sha256` calls and two `==` checks among three hash values (`manifest_hash == disk_hash` and `in_memory_hash == disk_hash`)). A `*.py` search finds those four names only in that module and `test_pool_shard_parity.py`. `docs/rung17-disk-pool-slice.md` lines 8–9 say this is not a production retrieval path. |

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
