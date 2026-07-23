# PRE-REGISTRATION (DRAFT — for Grok red-team before any GPU)
# HI-1: Does a gradient-free nonlinear geometric operator beat a strong linear map
#       *specifically in measurable unknown/OOD regions* of a frozen embedding cloud?

**STATUS: CUT — DO NOT RUN. Never frozen, no data touched, zero GPU spent.**

> Grok design red-team (2026-07-14, `out_HI1_design_redteam.txt`) returned **CUT**, and the chair
> accepts it in full. HI-1 is not "probably fails" — as written it is **statistically unable to produce
> a legitimate POSITIVE** and is **operator-structurally biased against its own treatment**. Four
> disqualifying findings:
>
> 1. **Guaranteed fail-closed by design (power).** Realistic REPORT-UNKNOWN n is ~9 (100-query split)
>    to ~27 (300-query). Extrapolating this program's own D1 paired contrast (half-width 0.055 at
>    n=60), the paired 95% half-width at that n is **~0.08–0.14**. Clearing the written `LB ≥ +0.01`
>    would need a true effect of **~0.09–0.15 absolute NDCG on the hard stratum alone** — larger than
>    rung-16's entire plane-vs-global effect. Requiring the same sign on *both* datasets multiplies the
>    underpowering. Same bug class as D1 needing ~817 queries for a 0.015 threshold.
> 2. **The primary stratum starves the treatment.** UNKNOWN is *defined* as low FIT support; diffusion
>    maps need local sampling density; Nyström extension into a low-density region of a FIT-built graph
>    is the textbook spectral failure mode. Since G is "nonlinear coordinate change + linear map," G
>    degenerates to a *worse ridge* exactly where the coordinate change is poorly identified. This is
>    the same *flavor* of protocol error as the β=0.10 screen — the design selects a regime that
>    defeats the method's own premise.
> 3. **The baseline was not the strongest linear map we already ship.** "Best of ridge/Procrustes" is
>    weaker than the repo's leakage-safe **full-FIT affine ridge (`Wx+b`)**, and G was to be given
>    anchors + graph-k + kernel bandwidth + #eigenvectors + Nyström + a second ridge in harmonic
>    coordinates against L's single λ — so any POSITIVE would have been a **capacity win, not a
>    geometry win**. Honest parity requires a CAL-chosen **reduced-rank affine ridge** whose rank grid
>    covers G's harmonic dimension.
> 4. **No outcome could falsify the motivating theory (the fatal one).** §0's "a failure is evidence
>    about the analogy, not a refutation of the compressed-sensing theorem," combined with §8's
>    pre-narrated NEGATIVE write-up, means both branches were pre-interpreted before running. Decision
>    value ≈ 0. This is a chair design error, not a reviewer quibble.
>
> **Replacement (if the wedge is still worth a coin-flip): the WITHIN-SPACE test.** In ONE frozen
> embedding space with known superposed / multi-dimensional features (synthetic, or Engels-style
> circular features), can a **gradient-free** harmonic / compressed-sensing-style readout beat a
> **linear probe** at recovering the feature? That attacks the *actual proven wedge* (Garg–Kleinberg–
> Peng quadratic gap) rather than a cross-space analogy, has far more statistical power than NDCG@10 on
> ~9 OOD queries, and is the only cell that is both winnable-in-principle and not already dominated by
> mini-vec2vec's linear-≥-nonlinear result.

**Original draft status (superseded):** DRAFT. Not frozen. No data touched. Red-team first, freeze second, run third.
**Author:** Fable-5 (chair). **Date:** 2026-07-14.
**Provenance:** the one GENUINELY-UNPURSUED cell identified by the 2026-07-14 deep-research pass
(106 agents, 23 sources, 25 adversarially-verified claims). See `out` of that run.

---

## 0. Why this experiment exists (and why it is probably going to fail)

Deep research established three things we must design around:

1. **The wedge is real.** A linear readout is *provably not* span-exhausting. Under superposition
   there is a proven **quadratic gap** between linear and nonlinear (compressed-sensing) decoding of
   k-sparse features — nonlinear needs `d=O(k log(m/k))`, linear needs `d=Õ(k² log m)` with a matching
   lower bound (Garg–Kleinberg–Peng 2026, arXiv:2602.11246). Corroborated by Anthropic superposition
   and causally-verified multi-dimensional (circular) features (Engels et al., ICLR 2025,
   arXiv:2405.14860). **So "a linear map can't read everything that's in the span" is true.**
2. **Nobody has exploited that wedge gradient-free.** Every demonstrated exploiter is gradient-trained
   (SAE, MP-SAE), and sparse dictionary learning is provably **non-identifiable** — it can hit zero
   reconstruction loss while recovering *zero* ground-truth features (arXiv:2512.05534).
3. **At our actual task, linear wins.** Unsupervised cross-space realignment of a frozen cloud:
   a *linear*, gradient-free method (mini-vec2vec, arXiv:2510.02348) **matches or exceeds** the
   nonlinear adversarial vec2vec. vec2vec's OOD-robustness claim was **refuted 0-3** in verification.

**Honest prior: this experiment most likely FAILS**, consistent with our own five consecutive
fail-closeds (drift corrector, recoverability estimator, home-turf preflight, H5 living bank, rung-16
routing plane). It is worth running *only* because it is the single untested cell and because it is
now possible to state it falsifiably. **A clean fail-closed is an acceptable, publishable outcome.**

### The scope gap (the biggest scientific risk — stated up front)
The *proven* wedge is about **feature readout within one space** (compressed sensing / probing).
Our target is **cross-space alignment after drift**. The mechanism transfers only **by analogy** and is
**not proven** for the alignment setting. HI-1 is therefore explicitly a test of the analogy, and a
failure is evidence about *the analogy*, not a refutation of the compressed-sensing theorem.

---

## 1. The single primary hypothesis (frozen)

> **H1.** In the subset of eval queries identified a-priori as **UNKNOWN** (by the frozen estimator in
> §3), a gradient-free nonlinear geometric operator **G** (§4) achieves higher NDCG@10 than the strong
> linear baseline **L** (ridge / orthogonal Procrustes), on a REPORT split never used for fitting,
> anchor selection, region estimation, or hyperparameter choice.

Everything else in this document is descriptive/secondary and cannot rescue a failed H1.

---

## 2. Arena, splits, and what is frozen

- **Arena:** the existing query-encoder-upgrade swap arena (MiniLM-L6 store, mpnet query encoder,
  seeded projection), which we already know produces a genuine, large recoverable gap. Datasets:
  SciFact (primary) + NFCorpus (replication). Reuse the frozen `EmbeddingPack` machinery so every
  method is scored through one identical NDCG implementation (harness parity ≤1e-12).
- **Three-way seeded query split (per dataset), disjoint by query id:**
  - **FIT** 40% — fit L; fit/parameterize G; mine anchors (from FIT docs only).
  - **CAL** 30% — calibrate the unknown-region estimator threshold; select ALL hyperparameters for
    both L and G; **locked after this**.
  - **REPORT** 30% — ONE frozen evaluation. Never touched before the lock is written.
- **Leakage contract:** eval-positive documents excluded from the anchor pool and from any fit index
  (reuse `leakage_safe_fit_idx`). Anchors mined on FIT docs only. No re-clustering on CAL/REPORT.
- **Freeze artifact:** `prereg_HI1.json` (hash-locked) written BEFORE any REPORT read; a
  REPORT-consumed marker written on first REPORT access; the runner refuses a second REPORT read for
  the same (prereg-hash, dataset) pair. (Lesson from rung 16: write the lock *and* prove ordering.)

---

## 3. The UNKNOWN-region estimator (fixed a-priori — this is what makes H1 falsifiable)

Without this the claim is unfalsifiable ("unknown region" could be defined post-hoc to fit results).
**Primary estimator (frozen): local density / k-NN epistemic proxy in the FROZEN store space.**

For each eval query q: `u(q) = mean cosine distance to its k=10 nearest FIT-split document vectors`
(computed in the frozen store space, before any correction). Higher `u` = less support = more UNKNOWN.

- **Threshold:** the `u`-value at the **70th percentile of CAL queries** (i.e. the top-30% least-
  supported). Chosen on CAL, then frozen. REPORT queries are labeled UNKNOWN/KNOWN by that frozen
  threshold — never re-tuned.
- **Pre-declared sanity gate (must pass or H1 is uninterpretable):** on CAL, the linear baseline L must
  perform **materially worse** on UNKNOWN than on KNOWN queries (NDCG gap ≥ 0.05). If L is equally good
  in both, there is no "unknown region" to win in, and HI-1 reports **REGIME-INVALID** rather than a
  verdict. (This directly tests your claim (a): "where the model already knows, there is no headroom.")
- **Secondary estimator (descriptive only, reported not decisive):** relative-representation
  **isometry residual** — the disagreement between anchor-cosine profiles pre/post drift. This is the
  point where "harmonic invariance breaks," and is the region your theory actually names.

---

## 4. The operator G (gradient-free, nonlinear, geometric) and the baseline L

**L (baseline, strong):** ridge and orthogonal Procrustes fit on FIT paired embeddings; λ selected on
CAL; the better of the two on CAL is the single frozen L. (We do NOT get to pick the weaker one.)

**G (the treatment), strictly gradient-free — no backprop, no SGD, no learned dictionary:**
1. **Anchor-relative encode:** represent every doc/query by its cosine profile to a fixed anchor set
   A (mined on FIT docs; |A| frozen on CAL) → basis-free, isometry-invariant coordinates
   (Moschella et al., ICLR 2023).
2. **Diffusion-harmonic embed:** build a k-NN affinity graph on the anchor-relative coordinates;
   compute the leading non-trivial eigenvectors of the normalized graph Laplacian (diffusion map,
   Coifman–Lafon) → the *harmonic* coordinates. This is closed-form (eigendecomposition), not trained.
3. **Out-of-sample extension:** Nyström extension to place drifted/unseen points in the harmonic basis
   without refitting — this is the "extend the known geometry into the unknown region" step.
4. **Align in harmonic space:** solve a closed-form (Procrustes/ridge) alignment **in the harmonic
   coordinates** rather than the raw space, then map back. The nonlinearity is entirely in the
   diffusion embedding; the alignment stays closed-form. Nothing is gradient-trained.

**Parameter parity (non-negotiable):** G and L get the same FIT data, the same CAL budget for
hyperparameter selection, and comparable effective parameter counts; both are selected on CAL only.

---

## 5. Frozen decision rule

Let Δ = NDCG@10(G) − NDCG@10(L), computed **on REPORT UNKNOWN queries only**, paired per query.

- **POSITIVE (H1 supported)** iff **all** hold:
  1. paired 10k-bootstrap 95% CI for Δ **excludes 0** and its lower bound ≥ **+0.01** absolute NDCG;
  2. Δ > 0 with the **same sign on both datasets** (SciFact and NFCorpus);
  3. G does **not** lose materially on REPORT **KNOWN** queries (Δ_known ≥ −0.005) — i.e. no robbing
     Peter to pay Paul;
  4. the §3 sanity gate passed (a real unknown-region headroom existed).
- **NEGATIVE (fail-closed)** otherwise — reported plainly, with the CI, as "a gradient-free nonlinear
  harmonic operator does not beat a strong linear map in the measurable unknown region of this arena."
- **REGIME-INVALID** if the §3 sanity gate fails (no headroom to contest) — not a verdict on H1.

No other outcome is a win. In particular: beating L on *all* queries but not on UNKNOWN is **not** a
win (it would just be a better global map, which is the thing we already know linear does well).

---

## 6. Pre-declared threats to validity

| Threat | Mitigation |
|---|---|
| "Unknown region" defined post-hoc | Estimator + threshold frozen on CAL before REPORT (§3) |
| G handicapped by bad anchors | Anchor count/selection tuned on CAL, same budget as L's λ |
| L handicapped (fake win) | L = best of ridge/Procrustes chosen on CAL; parity enforced (§4) |
| Multiple comparisons | ONE primary hypothesis, one metric, two datasets; everything else descriptive |
| REPORT peeking | hash-locked prereg + REPORT-consumed marker + runner refuses re-read |
| Scope-gap over-reading | §0 states failure is evidence about the analogy, not the theorem |
| Nyström instability in sparse regions | Pre-declared: if Nyström residual exceeds a CAL-set bound for >20% of REPORT UNKNOWN points, report **OPERATOR-UNSTABLE** rather than a verdict |

---

## 7. Cost and stop rule

CPU-feasible for the harmonic construction; GPU only for embedding the existing packs (already cached).
Estimated ≤2 GPU-hours total. **Stop rule:** if the §3 sanity gate fails on SciFact CAL, do not run
NFCorpus — report REGIME-INVALID and stop. If SciFact REPORT is NEGATIVE, still run NFCorpus once
(replication of a negative is cheap and strengthens the result), then stop.

---

## 8. What a NEGATIVE would mean (so we cannot move the goalposts later)

A negative here means: *in a measurable low-support region of a frozen retrieval space, closed-form
harmonic/anchor geometry does not extract recoverable structure that a ridge map misses.* Combined
with our five prior fail-closeds and the verified literature, that would make the practical case
essentially closed for gradient-free correction of frozen retrieval embeddings — and the honest
write-up becomes a sixth hazard for the methodology paper: **"basis-free geometry does not beat the
trivial linear map even where the model is provably unsupported."**
