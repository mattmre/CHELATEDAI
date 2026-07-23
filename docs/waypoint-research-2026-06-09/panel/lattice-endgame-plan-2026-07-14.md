# Liquified Lattice — endgame program plan (rungs 13, 15, 16, 17)

**Author:** Fable-5 chair. **Date:** 2026-07-14. **Branch base:** `origin/main` after PR #292.
**Execution engine:** grok-4.5 + codex-5.6-sol swarms via /workflows; each rung = its own PR with a
BHS body + a fresh Grok Tier B at the official-scorer gate (BHS_OFFICIAL = min(self, Tier B) = 100 to
merge). GPU campaigns on the 3090, serial.

## Goal narrative (the pivot)

The lattice's *corrector* question is answered, and the answer is "no": every annealing/living/bounded
mechanism this program tested — drift corrector, recoverability estimator, living post-bank — is
matched or beaten by a trivial baseline. So we stop asking "does an elaborate corrector win" and start
building **the substrate that makes the lattice a real, honest, disk-scale, self-pruning system**,
where the *only* thing that ever gets promoted is something that provably beats its trivial baseline.
That reframes the four open rungs from "hope a mechanism wins" to "build the machine, and let the
fail-closed gate be the product." Two of the four (16, 17) have **no hypothesis risk** — they are
engineering wins whose deliverable is a working, gated capability. One (13) is the genuine
disintegration loop the vision always wanted. One (15, GNN) is a real coin-flip that, given the arc, we
budget to likely fail-closed — and a clean fail-closed GNN is itself a publishable "graph learning does
not beat the flat pool here" result.

**North-star for the program:** a query flows through the engine → routes to the best *quant-surviving*
adapter that provably lifts retrieval (16) → over an evidence DAG that prunes its own low-fitness edges
via drift detectors (13) → optionally learned over by a GNN that must beat the flat pool or be
discarded (15) → served from a disk-resident pool shard with host parity (17). Every arrow is a gate.

## Honest-baseline discipline (load-bearing, applies to every rung)

Each rung names the **trivial baseline it must beat**, and **fails closed** (no promotion, honest
negative doc) if it does not. This is non-negotiable and is what makes the program honest given the
session's track record.

| Rung | Deliverable | Trivial baseline it must beat | Hypothesis risk |
|---|---|---|---|
| **13** | Drift-detector → Evidence-DAG edge prune (real disintegration loop) | Prune nothing (static DAG) must not degrade; prune must remove only low-fitness edges w/ recorded before/after | Low (mechanism, not a win claim) |
| **16** | Quant-aware routing **plane**: promote a route iff quant-survives AND lifts NDCG | **No-routing** (single global/frozen adapter) on the swap eval | None — gate is the product |
| **17** | One retrieval pool shard read via block-graph with host parity | In-memory pool == disk-read pool bit-for-bit; retrieval identical | None — parity is binary |
| **15** | Lightweight GNN over the evidence DAG on the drift fixture | **Flat-pool** baseline on the same fixture | High — likely fail-closed |

## Sequence (respects the ROADMAP dependency graph; one rung at a time)

Dependencies (from ROADMAP): 16 ← 10+11 (done); 17 ← 12 (done); 15 ← 12+13+14; 13 is the gate for 15.

### Phase 0 — Rung 13: real disintegration loop (unblocks 15) — SMALL
- **Decision (operator):** (a) re-scope exit criteria to the delivered post-bank prune, OR (b) implement
  the genuine isomer/convergence → Evidence-DAG edge prune. **Recommend (b)** — it is the loop the
  vision names and it is a bounded, testable slice.
- **Slice 13b:** wire `isomer_detector` / `convergence_monitor` signals to score Evidence-DAG edges;
  prune edges below a fitness threshold; record fitness before/after in an artifact; a re-anneal path
  that can re-add an edge if its cluster recovers. Add `EvidenceDAG.prune_edges(scorer, threshold)`.
- **Exit:** unit test proves (i) a low-fitness edge is pruned, (ii) a healthy edge is kept, (iii) prune
  is a no-op on an all-healthy DAG (baseline: pruning nothing), (iv) before/after fitness artifact.
- **Baseline/fail-closed:** if detector-scored prune removes edges that were actually high-fitness
  (measured by downstream retrieval), fail closed and keep the static DAG.

### Phase 1 — Rung 16: quant-aware routing plane (highest value) — MEDIUM, GPU
- **16a (integration):** a `QuantAwareRoutePromotion` that, for a candidate (centroid, adapter), calls
  `QuantizationPromotionGate.evaluate()` AND scores retrieval-fitness lift vs the no-route fallback on a
  held-out eval split; registers the route in `AdapterRouter` **only if** quant-survives AND lift > 0
  (dual gate). Leakage contract: eval split never used for adapter fit or threshold selection.
- **16b (engine plane):** `AntigravityEngine.enable_quant_aware_routing(...)` exposes the promoted
  routes as a live steering plane (opt-in, default OFF); provenance record per promotion.
- **16c (campaign):** on the query-encoder-swap arena, compare **routed plane vs no-route baseline vs
  single-global-adapter**; report NDCG lift + quant pass-rate. Fail closed if the routed plane does not
  beat no-route under the dual gate.
- **Exit:** ROADMAP rung-16 criteria — `adapter_router` + `QuantizationPromotionGate` integrated as one
  plane; promotion requires quant survival + retrieval fitness; campaign artifact + test.

### Phase 2 — Rung 17: disk pool slice (systems bet, no hypothesis risk) — MEDIUM
- **17a (encoder/reader):** encode one precomputed retrieval pool shard (a block of doc vectors + ids)
  as a block-graph payload via `build_graph_payload`; a `read_pool_shard(flash, offset)` that
  reconstructs the shard via `read_block`.
- **17b (parity + equivalence):** host parity check — SHA256(in-memory shard) == SHA256(disk-read
  shard); retrieval on the disk-read shard returns bit-identical top-k to the in-memory pool for a
  fixture query set.
- **17c (docs):** storage-track doc recording the one end-to-end read + parity result.
- **Exit:** ROADMAP rung-17 — one pool shard readable via `block_graph.py` with host parity check,
  documented. Fail closed if parity mismatches (that is a real bug, not a negative).

### Phase 3 — Rung 15: GNN prototype (highest risk, last) — MEDIUM/LARGE, GPU
- **15a (model):** a lightweight message-passing GNN (torch; PyG/DGL optional, prefer a ~50-line
  torch-native GraphConv to avoid a heavy dep) over the Evidence DAG that predicts per-edge/cluster
  drift-fitness (the label the drift fixture already produces).
- **15b (honest gate):** on the concept-drift fixture (rung 14 harness), the GNN must **beat the
  flat-pool baseline** (predict from node features alone, no graph) on the held-out fixture — else
  **fail closed** and ship the negative note. Preregister the metric + bar before running.
- **Exit:** ROADMAP rung-15 — GNN over evidence DAG; beats flat-pool baseline on the drift fixture or
  fails closed. Expectation (honest prior): ~50/50, leaning negative — a clean fail-closed is an
  acceptable, publishable deliverable, not a program failure.

## Per-rung PR / gate protocol (every rung)
1. Grok design red-team of the slice design BEFORE any GPU (catches unwinnable/rigged designs — it
   already saved us a GPU run this session).
2. Codex-5.6-sol builds the slice; fresh Grok reviews the build.
3. Chair applies fixes; runs the honest-baseline campaign; regression + block-flag + §4 validator green.
4. Fresh Grok Tier B official score; BHS_OFFICIAL = 100 to merge; one PR per rung.
5. next-session.md + ROADMAP Status column updated per rung (docs-truth discipline).

## Honest program-level expectation
- **16 + 17 will ship** as real gated capabilities (no hypothesis to lose) — the routing plane and the
  disk-pool read are the substrate wins.
- **13 will ship** as a real loop (mechanism, bounded).
- **15 is a coin-flip, budgeted to fail closed** — and we say so up front. If it wins, that is the one
  genuinely novel positive of the whole lattice arc; if it fails closed, that is an honest
  "graph-learning-does-not-beat-flat-pool-here" result.
- **Net:** after this program, Phase II rungs 9–17 are all either DONE or honestly-closed-negative, and
  the lattice is a complete, gated, disk-scale system — the north-star, built honestly, with every
  promotion earned against a trivial baseline.
