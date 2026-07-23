# Codex build — Rung 16: quant-aware routing plane (de-rigged per Grok design red-team)

Build the quant-aware routing plane the ROADMAP names, integrating `AdapterRouter` +
`QuantizationPromotionGate` + retrieval-fitness into ONE promotion plane. The Grok design red-team
returned PROCEED-WITH-CHANGES; the changes below are BINDING and are the whole point. Expect and accept
a fail-closed (empty plane) result on the default swap arena — that is still a valid rung-16 DONE. GPU.
Read `adapter_router.py`, `quantization_promotion_gate.py`, `antigravity_engine.py`,
`run_drift_recovery_swap_campaign.py` first.

## BINDING design (all from the red-team)
1. **Three-way seeded split** (the 2-way anchor/eval is NOT enough — rung 16 adds a second selection
   stage): **ANCHOR** ~40% (fit per-cluster adapters + the single-global adapter + mine centroids on
   ANCHOR docs only), **SELECT** ~30% (all promotion decisions: quant gate, plane formation, CI,
   thresholds — LOCKED after), **REPORT** ~30% (ONE frozen evaluation of the plane vs baselines; never
   touched by fit/promote/threshold).
2. **Baselines (preregister):** no-route (floor — plane must not lose materially), **single-global
   adapter, same budget/class/anchors (PRIMARY win condition — plane MUST beat this)**, C2O oracle
   (ceiling context only), single-best-route (report-only ablation for degeneracy).
3. **Promotion gate = plane-level, not per-route.** Fit routes on ANCHOR; form the full plane on SELECT
   under the serving rule; promote the plane iff a paired query-level bootstrap CI for
   `Δ = NDCG_plane − NDCG_single_global` **excludes 0** (lower bound > preregistered min_lift), AND quant
   survival holds for each used route. **BAN** "any route with lift>0 on the report split."
4. **Multi-route binding (else degenerate).** Instrument a route-usage histogram (p_k, entropy, n_used)
   on SELECT+REPORT; require **≥2 distinct routes each used on ≥10% of REPORT queries** or mark the plane
   DEGENERATE (== single adapter; cannot claim a routing win). Add an optional margin-fallback: apply a
   route only if `cos(q,c*) − cos(q,c_global) ≥ δ`, else fall back to global/no-route.
5. **Honest quant gate.** Pass no-route NDCG on the same split as `baseline_fitness`; set
   `minimum_fp32_gain` to a preregistered absolute NDCG floor (not the vacuous 0.0 default); FP32 and
   quant fitness measured through the same retrieval path (`simulate_int8` on adapted embeddings).

## Two arenas (default = expected fail-closed; multi-domain = the fair chance)
- **Arena A — default swap (SciFact, MiniLM→mpnet):** the honest floor test. Prior: routing likely
  fails to beat single-global (global drift). Report the dual-gate outcome; empty plane is fine.
- **Arena B — multi-domain mixed corpus:** pool docs from SciFact + NFCorpus + FiQA2018 into one store
  (cluster = domain), queries from all three, same MiniLM→mpnet swap. This is the regime the red-team
  named as routing's only fair chance (mode-specific, potentially anti-aligned residual drift). If the
  routed plane beats single-global here under the plane-level CI gate with multi-route binding, that is
  the genuine positive; if not, honest fail-closed. Keep the leakage-safe 3-way split within each arena.

## Deliverables
1. `adapter_router.py` (margin fallback + usage instrumentation) + a `quant_aware_routing.py`
   (promotion plane: ANCHOR/SELECT/REPORT, plane-level CI, honest quant gate) +
   `AntigravityEngine.enable_quant_aware_routing(...)` (opt-in, default OFF, provenance per promotion).
2. `prereg_rung16.md` + `.json` frozen BEFORE the REPORT split is touched: split seeds, baselines,
   min_lift, min cluster size, K/m binding thresholds, quant floor.
3. Campaign runner + `docs/rung16-quant-aware-routing-results-2026-07.md`: per-arena route-usage
   histogram, plane vs single-global (ΔNDCG + bootstrap CI), quant pass-rate, and the
   PROMOTED / FAIL-CLOSED / DEGENERATE verdict per arena.
4. `test_quant_aware_routing.py`: 3-way split disjointness, plane-level CI gate, multi-route binding
   detection, honest quant baseline, no eval leakage into fit/promote. Run it + `test_adapter_router`
   (no regression); paste results.

Be brutally honest. This is built to REFUSE promotion unless the plane genuinely beats a well-tuned
single-global adapter under a leakage-safe, multiple-comparisons-safe, multi-route-binding gate. A
fail-closed empty plane on both arenas is the expected, acceptable, honest outcome — do NOT relax any
gate to manufacture a promotion. If the plane wins on Arena B, report the exact CI + usage histogram as
the genuine positive.
