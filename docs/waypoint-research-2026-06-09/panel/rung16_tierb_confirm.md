# Grok Tier B — rung 16 confirmation re-score after the Arena B purity fix (commit 2c78195d)

You scored rung 16 (`5a3fbcc2`) **90 / PASS-WITH-FIXES**, severity **important**, with one important
defect and two cosmetic ones:

- **important:** the Arena B "fair-chance home turf" framing overclaimed — serve-time domain purity
  collapsed (FiQA 30%, NFCorpus 3.3%, SciFact 36.7%) and home-correct routes actually *beat* global
  (+0.0257) while cross-domain misroutes (−0.0309) dominated the loss. Fix required: add the
  domain×route purity table + home/cross/global Δ attribution, and rephrase so the conclusion is
  "the preregistered centroid-margin domain plane failed," not "domain structure was cleanly exercised."
- **cosmetic:** DEFERRED_SCOPE named only anti-aligned residual drift, understating the measured
  assignment bottleneck.
- **cosmetic:** frozen `used_adapters` vs current `retained_adapters` string drift (lineage note only).

Commit `2c78195d` (branch `lattice/rung16-routing-20260714`, cwd = repo root) applies the fix.

Confirm ONLY:
1. `docs/rung16-quant-aware-routing-results-2026-07.md` Arena B section now carries the purity table
   (30.0% / 3.3% / 36.7%) and the attribution table (home n=21 +0.0257; cross n=39 −0.0309; global
   n=30 0.0000; total n=90 −0.007383) — and that those numbers are correct against the frozen manifest
   rows (recompute them yourself, do not trust the doc).
2. The rephrased conclusion no longer claims Arena B cleanly tested domain routing, and explicitly
   states the binding constraint was route assignment, not specialist capacity.
3. The commit message names the DEFERRED_SCOPE amendment (oracle/domain-label routing ablation +
   domain-separable centroids under swap).
4. The FAIL-CLOSED verdict, the SELECT gate arithmetic, and all frozen artifacts are UNCHANGED by this
   commit (docs-only; no decision/metric/query-id/verdict field altered).
5. `python -m unittest test_quant_aware_routing test_adapter_router` still passes (expect 24 OK).

Then assign the final `BHS_TIER_B` for rung 16. If the important defect is genuinely closed and nothing
new is broken, say so and give the integer; if the fix is cosmetic-only or incomplete, score under 100
and name exactly what remains. Give `BHS_TIER_B`, `BHS_TIER_B_SEVERITY`, disposition, and a one-line
bottom line. Fresh-agent: you = grok-4.5; implementer/chair = Fable-5 (independent).
