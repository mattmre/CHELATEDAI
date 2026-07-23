# Grok Tier B — rung 16 quant-aware routing plane (commit 5a3fbcc2). Try to disprove the FAIL-CLOSED.

Official Tier B scorer for commit `5a3fbcc2` on branch `lattice/rung16-routing-20260714` (cwd = repo
root). The campaign reports **FAIL-CLOSED on both arenas** (the routing plane did not beat a
single-global adapter). Your job: try to DISPROVE that this is an honest, correctly-gated negative.

A false NEGATIVE is as bad as a false positive here: if the gate was mis-implemented such that a real
routing win was suppressed, that is a defect. Equally, if any de-rigging requirement was quietly
relaxed, that is a defect.

Read: `quant_aware_routing.py`, `run_quant_aware_routing_campaign.py`, `adapter_router.py`,
`prereg_rung16.md`/`.json`, `docs/rung16-quant-aware-routing-results-2026-07.md`, the manifest and the
per-arena selection-lock / REPORT-consumed JSONs, `test_quant_aware_routing.py`, `test_adapter_router.py`.

## Attack
1. **Is the negative real, or an implementation bug?** Verify the plane was actually constructed and
   served (routes registered, specialists trained, queries routed) — not silently empty/no-op, which
   would produce a fake FAIL-CLOSED. Check the route-usage histogram: were ≥2 routes genuinely used on
   REPORT? Arena A reports plane−global = −0.003332 (plane LOSES) — is that a real trained plane
   underperforming, or a broken/identity plane?
2. **Three-way split integrity.** Are ANCHOR / SELECT / REPORT genuinely disjoint by query id, and were
   centroids mined on ANCHOR docs only? Did anything (route promotion, quant threshold, hyperparameters)
   touch REPORT before the frozen evaluation? Recompute the split disjointness from the artifacts.
3. **Was the PRIMARY baseline honest?** The single-global adapter must have the SAME budget / class /
   anchors as the specialists (otherwise the plane was beaten by an unfairly-strong baseline, making the
   negative invalid). Verify parity.
4. **Gate arithmetic.** Recompute the plane-level paired bootstrap CI vs single-global from the frozen
   per-query scores in the artifacts. Does the reported CI [-0.051022, 0.041025] and delta -0.003332
   check out? Is the promotion rule as preregistered (CI excludes 0 / min lift), and not silently
   stricter than prereg (which would manufacture a fail-closed)?
5. **Provenance honesty.** The results doc discloses that selection locks were written before REPORT but
   later metadata-edited (raw prereg hash) and REPORT-consumed markers backfilled from the manifest, so
   mtimes are not contemporaneous proof. Is that disclosure adequate and accurate, or does it conceal an
   actual ordering violation (i.e., was REPORT read before the SELECT lock)? Check hashes.
6. **Arena B fairness.** Arena B (pooled SciFact+NFCorpus+FiQA) was meant to be routing's fair home
   turf. Were clusters actually domain-aligned (did routing have a real chance), or did the clustering
   collapse domains so the "fair chance" was never given? If the latter, the Arena B negative is weak
   evidence and should be labeled as such.
7. **BHS body honesty.** Any overclaim; is the disclosed artifact-ordering note honest; is the
   DEFERRED_SCOPE (anti-aligned residual drift) an honest scoping rather than an excuse?

Deliver `BHS_TIER_B` (integer), `BHS_TIER_B_SEVERITY` (none/cosmetic/important/critical),
PASS/PASS-WITH-FIXES/FAIL, a defect table (severity, file:line, fix), and a one-line bottom line: is
this an honest correctly-gated negative, a mis-gated false negative, or a concealed problem? Recompute
the numbers yourself; run the tests. Fresh-agent: you = grok-4.5, implementer = codex/Fable.
