# Gradient-Free Nonlinear Readout — operator design for the WITHIN-SPACE wedge test

**Author:** Fable (design agent). **Date:** 2026-07-19. **Status:** DESIGN — nothing frozen, no data touched, zero GPU spent.
**Provenance:** the replacement cell nominated by the HI-1 design red-team (`out_HI1_design_redteam.txt`, verdict CUT), which recommended abandoning the cross-space analogy in favour of a within-space test against the *actual* proven compressed-sensing wedge.

---

## 0. Verdict up front

> ## **CUT** the wedge test as a *"gradient-free nonlinear readout beats a linear probe"* experiment.
>
> Not because it is underpowered — **it is the first design in this program that is not power-limited at all** (§1). It is CUT because of a structural dilemma that no amount of protocol hygiene dissolves:
>
> **The condition that makes the comparison FAIR is the same condition that makes it UNINFORMATIVE.**
>
> - To be **fair**, both sides must know the dictionary `D` (§4). Given `D`, the strongest linear baseline is the *analytic Bayes-optimal linear functional*, and the comparison against an l1 decoder reduces to exactly the comparison Garg–Kleinberg–Peng already proved, with a matching lower bound. Running it re-derives a theorem. Decision value ≈ 0 — the same defect (pre-known outcome) that killed HI-1 on point 4.
> - To be **informative**, the operator must *find* `D` in a real embedding space. Gradient-free CS decoding cannot: it requires a known dictionary, and obtaining one requires dictionary learning, which is gradient-based **and** provably non-identifiable (arXiv:2512.05534 — zero reconstruction loss with zero ground-truth features recovered, 3/3200 empirically).
>
> **The gradient-free constraint and the real-space target are jointly unsatisfiable given the current state of the art.** That is the finding. It is a genuine, citable, zero-GPU result, and it is worth more than a seventh fail-closed.

**What survives:** one strictly-scoped, CPU-only, asymmetric screen (**W-0**, §7) that *can close the wedge program permanently but can never open it*. It is offered as optional and low-priority, not recommended-by-default. It is explicitly **not** capable of producing a "we beat linear" headline.

**What would change this verdict:** a gradient-free *and identifiable* dictionary construction at embedding scale. One candidate class exists and was not surfaced by the deep-research pass — overcomplete ICA / tensor decomposition (§6). It is not currently feasible at `d=384`, but it is the only door that is not closed by proof.

---

## 1. Power math FIRST (per the governing lesson)

The HI-1 lesson is: *derive the bar from the achievable n; never propose a bar and hope n cooperates.* Doing that here produces a surprising and load-bearing inversion.

Paired 95% half-width, `HW = 1.96·sd/√n`:

| n | sd=0.1 | sd=0.3 | sd=0.5 | context |
|---:|---:|---:|---:|---|
| 9 | 0.0653 | 0.1960 | 0.3267 | HI-1 REPORT-UNKNOWN, 100-query split |
| 27 | 0.0377 | 0.1132 | 0.1886 | HI-1 REPORT-UNKNOWN, 300-query split |
| 30 | 0.0358 | 0.1074 | 0.1789 | rung-16 Arena A REPORT |
| 60 | 0.0253 | 0.0759 | 0.1265 | D1 pack (observed ridge−MLP HW **0.055**) |
| 90 | 0.0207 | 0.0620 | 0.1033 | rung-16 Arena B REPORT |
| 500 | 0.0088 | 0.0263 | 0.0438 | |
| **2000** | **0.0044** | **0.0131** | **0.0219** | within-space, trivially achievable |
| 10000 | 0.0020 | 0.0059 | 0.0098 | |
| 50000 | 0.0009 | 0.0026 | 0.0044 | |

**The inversion.** In a within-space readout test the unit of analysis is a *(sample, feature)* recovery, not a query. `n` is a **free design parameter** — synthetic samples are generated, not harvested. At `n=2000` the paired half-width is ~0.013 even at a pessimistic `sd=0.3`; that is **6–15× tighter than HI-1's 0.08–0.14** and ~4× tighter than the D1 observed 0.055. Any effect above ~0.03 is detectable with margin, and precision can be bought arbitrarily by generating more samples.

**Consequence, stated plainly:** this design **cannot die of power**. Therefore it must be judged entirely on **construct validity** — and that is exactly where it dies. Because power is free, "we were underpowered" is unavailable as an excuse and "we ran it and won" is unavailable as evidence: at `n=50000` you will detect the theorem's gap at 4 decimal places, which tells you nothing you did not already know from the proof.

A design that cannot fail to detect its target effect is not an experiment. It is a very expensive unit test.

---

## 2. What the wedge actually is (so the operator can be judged against it)

Under superposition: `x = D z`, with dictionary `D ∈ ℝ^{d×m}`, `m ≫ d` (overcomplete), and `z ∈ ℝ^m` **k-sparse**. Task: recover coordinate `z_j` from `x`.

- **Linear readout.** Any linear probe computes `wᵀx = (Dᵀw)ᵀz = vᵀz` with `v = Dᵀw`. Since `v` is confined to `range(Dᵀ)`, a `d`-dimensional subspace of `ℝ^m`, and `m > d`, the target `e_j` is generically **not** in it. The residual `‖P_{range(Dᵀ)}e_j − e_j‖` is irreducible **interference**. This is precisely "a linear map cannot read everything in the span."
- **Nonlinear (CS) readout.** Solve `min ‖z‖₁ s.t. Dz = x`. Under RIP/incoherence with `k ≲ d/log(m/d)`, this recovers `z` **exactly**, hence `z_j` exactly. The sparsity prior annihilates the interference.

That is the whole wedge. **The nonlinearity is not capacity — it is the sparsity prior.** This single sentence is the discriminator used throughout §3: an operator that adds nonlinear capacity *without* using a sparsity prior is not a wedge exploiter, no matter how nonlinear it is.

---

## 3. Candidate operator evaluation

| # | Operator | Truly gradient-free? | Exploits the wedge, or just adds capacity? | Must know a priori | Verdict |
|---|---|---|---|---|---|
| 1 | **l1 / basis pursuit / LASSO** vs known `D` | **Yes, on the spirit; yes on the letter via LARS-homotopy.** Zero learned parameters. LARS/homotopy is exact and finite-step; coordinate descent uses closed-form soft-thresholding. *(ISTA/FISTA are proximal-**gradient** — excluded on the letter; do not use them.)* | **Exploits it.** This is literally the decoder in the theorem. Capacity is entirely in the sparsity prior; one free knob (λ) — exact parity with ridge's one λ. | **`D`**, plus `k` implicitly via λ | **PRIMARY** (conditional on §4/§5) |
| 2 | **Orthogonal Matching Pursuit** | **Yes, unambiguously.** Greedy, deterministic, closed-form least squares per step. Cleanest on the letter of the constraint; no solver-tolerance nondeterminism. | **Exploits it.** A genuine CS decoder with RIP/coherence guarantees, weaker constants than BP. | **`D` and `k` explicitly** (stopping rule) — a *stronger* assumption than λ | **FALLBACK** |
| 3 | **Kernel / spectral readout** (kernel ridge, closed-form eigendecomposition) | Yes (closed form). | **NO — this is the disqualifier.** Generic nonlinear function approximation with **no sparsity prior and no mechanism to identify which of `m>d` features are active**. It cannot defeat interference; it can only fit the conditional mean better, and at `d=384` it pays the curse of dimensionality to do so. This is "elaborate structure adds capacity" — the exact failure mode of the previous six fail-closeds. | kernel + bandwidth + λ (≥3 knobs) | **NOT a contender → repurpose as NEGATIVE CONTROL (§5)** |
| 4 | **Anchor-relative + diffusion-harmonic** | Yes. | **NO.** No sparsity prior; density-dependent; Nyström into sparse regions is the textbook spectral failure mode. Already CUT with reasons in the HI-1 red-team. | anchors, graph-k, bandwidth, #eigenvectors, Nyström recipe (≥5 knobs) | **EXCLUDED** — named explicitly so it is not silently revived |

### Recommendation

- **Primary operator: LASSO / basis pursuit against a known dictionary, solved by LARS-homotopy.** It is the only candidate that (a) is gradient-free in both letter and spirit, (b) exploits the *actual* wedge mechanism rather than adding capacity, and (c) has knob-parity with ridge (one λ vs one λ) so a win cannot be dismissed as a capacity win — the objection that would otherwise repeat the HI-1 point-3 finding.
- **Fallback: OMP.** Deterministic and reproducible-by-construction (an asset for a repo that hash-locks artifacts), at the cost of needing `k` explicitly.
- **Negative control: kernel ridge.** Nonlinear, capacity-rich, sparsity-blind.
- **Excluded: diffusion-harmonic.** Already adjudicated.

### Exactly what the operator is allowed to know

| Quantity | Allowed? | Rationale |
|---|:---:|---|
| Dictionary `D` | **Yes — and the linear baseline gets it too** | The load-bearing condition. See §4. |
| Sparsity level `k` | **Only via a CAL-selected λ** (primary) / **explicitly** (OMP fallback, and this must be disclosed as an advantage OMP holds over ridge) | Selecting λ on CAL is parity with selecting ridge's λ on CAL. Handing OMP the true `k` is *not* parity and must be labelled. |
| Code `z` / true support | **No** | Trivially rigged. |
| Any REPORT-split quantity | **No** | Standard freeze contract; λ and `k` selected on CAL only. |
| Noise level / SNR | **Only if the linear baseline also gets it** | Symmetry rule (§4). |

---

## 4. THE FAIRNESS QUESTION — is giving the operator the true dictionary rigged?

**The answer depends entirely on which linear baseline you compare against, and the honest answer is different for each.**

Define three baselines:

- **L0 — sample-fit ridge probe.** Fit `w` on training pairs `(xᵢ, z_{j,i})`. Realistic; carries estimation error.
- **L1★ — analytic Bayes-optimal linear probe.** Computed in closed form from `D` and `Σ_z`: `w★ = Σ_x⁻¹ E[x z_j]`, with `Σ_x = D Σ_z Dᵀ` and `E[x z_j] = D Σ_z e_j`. **Zero estimation error. This is the provable ceiling of the entire linear hypothesis class.**
- **L2 — kernel ridge.** Nonlinear, capacity-matched-or-better, sparsity-blind.

Now the verdict, per baseline:

| Comparison | Rigged? | Why |
|---|---|---|
| CS decoder (given `D`) **vs L0** | **YES — flatly rigged.** | L0 must *estimate from finite samples* the very object the decoder is handed. Any win conflates an informational advantage with a structural one. **Never report this as the headline.** |
| CS decoder (given `D`) **vs L1★** | **NO — not rigged.** | `L1★` is the exact optimum of the linear class *given complete knowledge of `D` and the generative distribution*. Both sides have full information. They differ in **hypothesis class only** — which is precisely and exclusively what the theorem is about. This is the strongest linear baseline that can exist; it cannot be accused of being under-fit, under-tuned, or under-parameterised. |
| CS decoder **vs L2** | Control, not a contest | If L2 also beats L1★, the "win" is capacity, not the wedge → result void (§5). |

### The fair information-parity condition (stated formally)

> **Both sides receive `D` and the generative distribution of `z`. Neither side receives `z` or the true support. The linear side is instantiated at its analytic optimum `L1★`, not at a sample-fit estimate. The nonlinear side gets exactly one free knob (λ), selected on CAL only, matching ridge's one λ. Any additional information given to one side (SNR, `k`) must be given to the other or explicitly disclosed as an asymmetry.**

Under that condition the comparison is **meaningful and not rigged**. Giving the decoder `D` is defensible precisely because **knowing `D` does not raise the linear probe's ceiling** — `L1★` already saturates it, and a sample-fit probe converges to `L1★` asymptotically. The decoder's advantage is not "it knows more"; it is "it is allowed to use a prior that no linear functional can express."

### …and this is exactly why the experiment dies

Having made the comparison fair, look at what it now is:

`L1★` is the optimal linear decoder; l1-BP is the optimal sparse decoder; the gap between them at given `(d, m, k)` **is the quantity Garg–Kleinberg–Peng proved, with a matching lower bound.** Constructing a RIP dictionary, planting k-sparse codes, and measuring that gap does not test the theorem — it *instantiates* it. At the `n=50000` precision available (§1) you would recover the predicted gap to four decimals.

**Fair ⟹ the result is a mathematical certainty. Informative ⟹ the operator is not constructible gradient-free.** There is no third cell. That dilemma — not power, not baseline strength, not protocol hygiene — is why this is a CUT.

---

## 5. Guards that would be mandatory *if* anything in this family were ever run

Retained because they are reusable and because their necessity is itself part of the argument.

1. **Capacity-artifact gate (hard, pre-declared).** If kernel ridge (L2) also beats `L1★` by a comparable margin, the result is **VOID — capacity artifact**, not a wedge confirmation. The wedge claim requires *sparsity-aware nonlinear wins **and** sparsity-blind nonlinear does not*. Without this gate, any "win" is indistinguishable from the six prior "elaborate structure ≈ more parameters" outcomes.
2. **Headline baseline is `L1★`, never L0.** L0 may be reported descriptively (it quantifies estimation error) but can never carry the claim.
3. **Knob parity ledger.** Publish a table of free parameters per method. Primary must be 1-vs-1 (λ vs λ). OMP's known-`k` is disclosed as an asymmetry in OMP's favour.
4. **Solver-class disclosure.** State the l1 solver explicitly. LARS-homotopy / coordinate descent qualify as gradient-free; **ISTA/FISTA do not** and would silently violate the constraint. This is a real L3-class trap: the most convenient LASSO implementations are proximal-gradient.
5. **Freeze contract.** Hash-locked prereg + REPORT-consumed marker + runner refuses re-read, per the rung-16 pattern.

---

## 6. The one loophole — and an honest accounting of it

The non-identifiability result (arXiv:2512.05534) is about **l1/reconstruction-based dictionary learning**. It does **not** cover a second class the deep-research pass did not surface:

> **Overcomplete ICA / method-of-moments tensor decomposition.** Fixed-point iteration and tensor power methods are **gradient-free**, and — unlike reconstruction-based dictionary learning — they carry **identifiability guarantees** for `m > d` under non-Gaussianity plus incoherence conditions.

This is the only path not closed by proof, and intellectual honesty requires naming it. It nonetheless does not rescue the design at this scale:

| d | 4th-order moment tensor entries (`d⁴`) |
|---:|---:|
| 384 (MiniLM) | 21,743,271,936 |
| 768 (mpnet) | 347,892,350,976 |
| 128 | 268,435,456 |
| 64 | 16,777,216 |

Identifiable overcomplete recovery leans on 3rd/4th-order moments. At `d=384` the 4th-order tensor has ~2.2×10¹⁰ entries — statistically and computationally out of reach without aggressive dimension reduction, and **reducing to `d≈64` to make the tensor tractable destroys the overcompleteness (`m > d`) that the whole test is about.** The reduction is self-defeating, not merely expensive.

**Honest status: not refuted, not feasible here.** If someone demonstrates gradient-free identifiable dictionary recovery at `d ≥ 384`, the within-space wedge test becomes worth running *and informative*, because the decoder would then have earned `D` rather than been handed it. That is the single concrete condition that flips this verdict.

---

## 7. W-0 — the only thing that survives (OPTIONAL, low priority, asymmetric)

Not a wedge test. An **applicability screen** on whether the wedge's preconditions plausibly hold in real embedding spaces at all.

**Question.** Do real frozen embeddings (MiniLM-L6 / mpnet, existing cached packs) show the *necessary conditions* for k-sparse structure in an overcomplete dictionary — or are they effectively Gaussian, in which case no CS decoder helps regardless of `D`?

**Method (CPU-only, no model training, gradient-free).**
1. Random-projection **excess kurtosis**: a sparse signal projected onto a random direction has positive excess kurtosis; an elliptical Gaussian has ~0. Cheap, distribution-level, requires no dictionary.
2. **Participation ratio / effective rank** of the embedding covariance.
3. Compare each against a matched-covariance Gaussian null (same `Σ`, resampled) — so the statistic is tested against the *right* null rather than an iid one.

**Frozen decision rule.**
- **KILL (decisive).** If real-embedding excess kurtosis is statistically indistinguishable from the matched-covariance Gaussian null, sparsity in *any* basis is implausible → **the entire gradient-free wedge program closes**, publishable as methodology hazard #7: *"the proven compressed-sensing wedge is inapplicable to frozen retrieval embeddings — its precondition is not met, independently of the decoder."*
- **NOT-KILL (weak).** High kurtosis is *consistent with* sparsity but **does not identify `D`** and therefore **does not license any downstream experiment** — because §0's dilemma is untouched.

**This screen is deliberately asymmetric: it can close the program, it can never open it.** State that up front so a NOT-KILL cannot later be narrated as encouragement. That asymmetry is the design's honesty guarantee — the failure mode it is built to prevent is precisely the HI-1 defect of pre-narrating both branches into a win.

**Cost.** ~2 CPU-hours, zero GPU, no contention with night GPU work. **Decision value** = `P(KILL) × value(permanently closing the wedge program)`. The NOT-KILL branch changes nothing. Run it only if closing the direction is worth two CPU-hours; **do not run it expecting to learn how to proceed.**

---

## 8. Brutal Honesty

- **This design produces no positive result under any branch.** CUT is the recommendation; the surviving screen (§7) can only kill. Anyone hoping this cell yields a "we beat linear" headline should stop here.
- **I did not run anything.** No code executed beyond the §1 arithmetic (`1.96·sd/√n` and `d⁴`). Every number attributed to prior work is quoted from repo artifacts: D1 half-width 0.055 at n=60 (`out_HI1_design_redteam.txt`); rung-16 Arena A `−0.003332`, CI `[−0.051022, 0.041025]`, and Arena B `−0.007383`, CI `[−0.027989, 0.012247]`, home-correct `+0.0257` (n=21) vs misrouted `−0.0309` (n=39) (`docs/rung16-quant-aware-routing-results-2026-07.md`); preflight 48 cells / max residual 0.033 vs 0.05 gate (session brief, corroborated by the 0.05 `minimum_residual_band` in `d2b_preflight/preflight.py`). No number here is estimated, extrapolated, or invented beyond the explicitly-labelled `HW = 1.96·sd/√n` table, whose `sd` values are illustrative scenarios, not measurements.
- **Citations are inherited, not independently verified by me.** Garg–Kleinberg–Peng (arXiv:2602.11246), Engels et al. (arXiv:2405.14860), non-identifiability (arXiv:2512.05534), mini-vec2vec (arXiv:2510.02348) are taken from the deep-research pass as recorded in the session brief and the HI-1 prereg. I fetched none of them in this task. If any is misattributed upstream, §2 and §6 inherit that error.
- **The §6 tensor argument is structural, not empirical.** `d⁴` is arithmetic; "statistically out of reach" is a qualitative sample-complexity argument I did not quantify. I am confident in the direction, not in a threshold. Do not cite a specific feasibility boundary from it.
- **The §4 `L1★` formula assumes a second-moment characterisation** (`w★ = Σ_x⁻¹E[x z_j]` is the optimal *linear* predictor under squared loss). That is correct for the linear class under L2 loss and is all the argument needs, but it is not "Bayes-optimal" over all predictors — the wording in §4 says *Bayes-optimal linear*, and that qualifier is load-bearing.
- **Strongest counter-argument to my own verdict**, stated so it is not buried: one could argue the *phase-diagram* question — "is the proven-asymptotic gap **operationally large** at embedding-realistic `(d, m, k, SNR)`, given anisotropic real covariance?" — is genuinely open, since GKP is an asymptotic complexity statement, not a claim about effect size at `d=384`. I considered making that the primary recommendation and rejected it: it still requires a planted `D` (so it inherits the §0 dilemma), its result would be about *synthetic* geometry with real covariance bolted on, and — decisively — **it cannot change any action**, because even a large measured gap leaves you unable to construct `D` gradient-free on real data. A study that cannot change an action is not worth the hours regardless of how open the question is. I flag it because a reasonable reviewer may weigh it differently, and because it is the strongest thing that can be said against CUT.
- **No stubs, mocks, escape conditionals, or partial implementations** — this deliverable is a design document; no production code path was touched.
