# Grok design red-team — Rung 16 quant-aware routing plane (BEFORE any GPU)

Red-team this design adversarially. The pattern this whole program has shown: elaborate mechanisms
lose to trivial baselines, and dual gates can be rigged to manufacture a win. Try to break the design
before we spend GPU. Read `adapter_router.py`, `quantization_promotion_gate.py`,
`antigravity_engine.py` (retrieval/eval path), and the swap-arena driver
(`run_drift_recovery_swap_campaign.py`) to ground the attack.

## Proposed design
Integrate the existing `AdapterRouter` (centroid-cosine route select) + `QuantizationPromotionGate`
(`evaluate()` → survives?) + a retrieval-fitness check into ONE promotion plane:
- **Candidate:** a (centroid, adapter) route mined from a cluster of the corpus.
- **Dual promotion gate:** register the route in `AdapterRouter` **iff** (i) the adapter SURVIVES the
  quantization gate AND (ii) it produces NDCG **lift > 0** vs the **no-route** fallback on a **held-out
  eval split** that was never used to fit the adapter or pick the threshold.
- **Serving:** `AntigravityEngine.enable_quant_aware_routing()` exposes the promoted routes; a query
  routes to its best centroid; if no promoted route beats no-route, the plane is empty (no-op).
- **Campaign (swap arena):** routed-plane vs **no-route baseline** vs **single-global-adapter**; report
  NDCG + quant pass-rate. Fail-closed: if the routed plane does not beat no-route under the dual gate,
  ship the honest negative.

## Attack these
1. **Is the baseline honest?** Should the plane beat **no-route** (frozen), **single-global-adapter**,
   or BOTH? Which is the trivial baseline that makes a win meaningful vs which one hands routing a rigged
   edge? (Prior: single-global is the harder, more honest baseline — routing must beat *tuning one
   adapter well*, not just beat doing nothing.)
2. **Leakage in "lift > 0".** How is the held-out eval split isolated from adapter fit AND from route
   selection AND from the quant threshold? Where could eval leak in (e.g., centroids mined on eval docs,
   or per-route threshold tuned on the same split it's scored on)? Name the exact leakage-safe partition.
3. **Multiple-comparisons rigging.** Promoting "any route with lift > 0 on the held-out split" over N
   candidate clusters is a garden-of-forking-paths: some route lifts by chance. What correction makes a
   promoted route's lift real (per-route CI excludes 0? Holm over the candidate family? a min lift
   magnitude AND min cluster size)?
4. **Does routing even bind?** Centroid-cosine routing may send ~all queries to one adapter (degenerate
   plane == single adapter). How do we prove the plane is genuinely multi-route and not a disguised
   single-global? (route-usage histogram; require ≥K distinct routes actually used on eval.)
5. **Quant gate meaning.** Does surviving `QuantizationPromotionGate` actually constrain anything, or
   will near-identity adapters trivially survive (making gate (i) vacuous)? Check the gate's real
   threshold.
6. **Is this winnable, or another dead end?** Given the arc, is there a real regime where a
   quant-surviving *routed* plane beats a single well-tuned global adapter — or will the global adapter
   always match it (making rung 16 a fail-closed like the others)? If it's unwinnable-by-construction,
   say so — a fail-closed rung 16 is still an honest DONE (the gate works, nothing promotes), but we
   should know before GPU.

## Deliver
`PROCEED / PROCEED-WITH-CHANGES / RESCOPE` + the single highest-value design change, the exact
leakage-safe eval partition, the anti-rigging correction for gate (ii), and the honest baseline set.
One-paragraph bottom line: is rung 16 a real capability win, a fail-closed-but-honest DONE, or
unwinnable-and-should-be-descoped?
