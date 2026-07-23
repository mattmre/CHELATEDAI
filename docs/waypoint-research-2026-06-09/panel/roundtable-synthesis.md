# Expert roundtable synthesis — chelation reframe (2026-06-30)

Panel: **Grok 4.5** (novelty/prior-art), **Codex gpt-5.6-sol xhigh** (technical/math), **Claude Fable-5** (chair
+ 3rd panelist). Gemini blocked (Code-Assist tier deprecated). Full reviews: `out_grok.txt`, `out_codex.txt`.

## Convergence (all three agree)

| Question | Verdict |
|---|---|
| Is the "mis-benchmarked / adversarial by construction" reframe correct? | **Partly (scope), mostly rationalization.** Geometry of the *bound* is valid (α=0.05 → max 2.86° rotation; a doc-side-only detector cannot even *observe* a query-space rotation → unidentifiable). But "encoder upgrade is affine *by construction*" is **false** (Codex: both encoders are nonlinear in text; Procrustes' dot-product preservation is an *assumption* it estimates on 10k pairs, not a theorem; Drift-Adapter's MLP *beats* linear in the ordinary regime). |
| Drift-corrector chelation | **DEAD.** (Grok "dead enough"; Codex ~95% stop.) Bounded/unpaired posts lose to a one-line ridge map ~4.2×, the bound excludes the needed motion, and the failure was *also* ordinary ML (wrong objective: +59 pts from paired regression vs −5 from the bound). |
| Within-model-collapse chelation | **Untested, NOT vindicated. ~10–15% for a meaningful win.** "Unborn, on life support" (Grok) / "unsupported, not surviving evidence" (Codex). |
| Highest-EV product | **The honest negative + methodology paper**, built on Drift-Adapter. Codex: 65–75% publishable, 20–25% *exceptional*, <5% groundbreaking mechanism. |
| Next step | **ONE preregistered existence/crossover test with STRONG baselines + kill criteria, before any more corrector work.** |

## New material the panel produced that we did NOT have

1. **⭐ The "recoverability certificate" (Codex) — the standout new direction, and it is NOT chelation.**
   A calibrated estimator/theorem: from paired Gram-distortion + query-margin distributions, *predict before
   deployment* whether an affine adapter will recover 70/90/99% of retrieval. Builds on Procrustes theory +
   our unusually-severe benchmark. **~25% chance of an exceptional paper, 5–10% genuinely important.** Higher
   ceiling than any chelation path.
2. **Our "84% linear ceiling" is statistically UNSOUND as stated (Codex).** Ridge 84% vs MLP 81% = only
   **0.027 absolute NDCG, with NO confidence intervals** — the ordering isn't even established. "Ceiling"
   needs: anchor-count learning curves, bootstrap CIs, held-out reconstruction, a residual→ranking-margin
   bound. **Actionable paper fix: add 10k paired-query bootstrap CIs + a learning curve; downgrade "ceiling"
   to "observed plateau under this protocol."**
3. **isotropy ≠ retrieval quality (Codex).** "Is Anisotropy Truly Harmful?" (ACL 2023) + "On Isotropy
   Calibration" found ~no significant isotropy↔quality relationship. **This is a landmine under P4** — even if
   heterogeneous collapse exists, fixing it may not help retrieval.
4. **P1 is already partly done (Codex).** Drift-Adapter's appendix: metadata-*routed local adapters* improve
   synthetic heterogeneous-drift ARR 0.85→0.94. So even "local > global" is anticipated in basic form.
5. **P5 / capacity decomposition (Grok)** — split the unrecovered 16% into linear-residual / nonlinear /
   rank-capacity; if capacity dominates, *no* point map can reach oracle → kills P2. (Subsumed by Codex's
   recoverability certificate.)

## The decisive experiment (Codex's protocol, endorsed by chair)
Preregistered **Regime U (real upgrade) × Regime C (favorable synthetic heterogeneous collapse) crossover**,
BEIR Quora (522,931 docs; 5k dev / 10k test), 5 seeds, qrel-positives excluded from anchors, everything
through the learned map (no target substitution). Baselines that MUST be beaten: global ridge/Procrustes,
**CBIE (cluster-based isotropy), ZCA, all-but-the-top, local+global hubness scaling**. Kill criteria: a
"win" = ≥5 recovery pts AND ≥0.02 NDCG AND 95% simultaneous CI excludes 0 AND same sign in all 5 seeds.
Predicted: ridge 82–86%, no stable ≥5-pt local gain; Regime-C chelation 10–35% vs established methods 20–45%;
**~10–15% chance of the decisive chelation crossover.**

## Chair's decision
- The elaborate chelation corrector is dead in the drift regime (unanimous) and evidentially near-dead in the
  within-model regime. Do **not** rebuild the living-bank.
- **Ship the negative/methodology paper** (add the bootstrap CIs first — it closes the one real hole a
  reviewer will find). That's the ~65–75% sure thing.
- If we want a real shot at *exceptional*, the panel's own best idea is the **recoverability certificate**,
  not chelation. That's where the groundbreaking odds actually live.
- Run the crossover ONLY as the one honest, preregistered last test of the within-model claim — with the
  kill criteria binding. If it fails Regime C, chelation is dead in both regimes and that's the healthy outcome.
