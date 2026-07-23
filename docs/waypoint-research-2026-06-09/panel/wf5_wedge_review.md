# Grok adversarial design review — the WITHIN-SPACE WEDGE TEST (whole design, pre-freeze, pre-GPU)

Read these four documents **in full** before answering. They are the complete current design of the
within-space wedge test:

1. `docs/waypoint-research-2026-06-09/panel/wedge-power-analysis.md`
2. `docs/waypoint-research-2026-06-09/panel/wedge-feature-construction.md`
3. `docs/waypoint-research-2026-06-09/panel/wedge-operator-design.md`
4. `docs/waypoint-research-2026-06-09/panel/wedge-baseline-design.md`

Supporting context you may consult if useful (do not treat as authoritative over the four above):
`panel/out_HI1_design_redteam.txt`, `panel/prereg-harmonic-invariance-HI1-draft.md`,
`panel/wedge_power_sim/` (the committed pilot simulation source),
`panel/wedge-preflight/`, `docs/rung16-quant-aware-routing-results-2026-07.md`.

---

## 0. Why you are being asked, and the standard you must apply

This cell exists **only because you CUT its predecessor.** The HI-1 pre-registration (cross-space
harmonic-invariance alignment) was killed by a design red-team on these grounds — apply the **same
standard, at the same severity**, to the wedge test:

- **HI-1 cut reason 1 — statistically guaranteed fail-closed.** Realistic n was ~9–27 REPORT-UNKNOWN
  queries against a paired half-width of 0.08–0.14, while the win bar was a +0.01 CI lower bound.
  The bar was set *before* the power math and the power math could never clear it.
- **HI-1 cut reason 2 — baseline weaker than the one already shipped.** "Best of ridge / orthogonal
  Procrustes" was a strawman; the honest opponent was leakage-safe full-fit affine ridge plus a
  capacity-matched reduced-rank affine ridge. Procrustes was the weak arm.
- **HI-1 cut reason 3 — fake parity.** The treatment had ≥5–6 free knobs (anchor count, mining rule,
  graph-k, kernel bandwidth, #eigenvectors, Nyström recipe, then a ridge *inside* harmonic coords)
  against a baseline with one λ. Any win would have been a capacity win, not a geometry win.
- **HI-1 cut reason 4 — structurally rigged-to-FAIL.** The primary stratum (UNKNOWN = low FIT support)
  starved the density-based operator of exactly the density it needed; Nyström extension into a
  density-selected sparse region is the textbook spectral failure mode.
- **HI-1 cut reason 5 — no outcome could falsify the motivating theory.** Both branches were
  pre-narrated (a NEGATIVE was pre-written as "methodology hazard #6"), so the study could not change
  an action. Decision value ≈ 0.

**Cutting a doomed experiment before GPU is a win.** You have now done it twice in this program (the
beta-sweep and HI-1) and both were correct. Do not soften this one to be agreeable, and equally do not
CUT reflexively to look rigorous — a false CUT of a genuinely decisive cheap test is also a failure.

Program prior you must weigh: **six consecutive fail-closeds** in which elaborate structure lost to a
trivial global linear baseline (drift corrector beaten ~4.2x; recoverability estimator inverted at
powered scale; sparse-local home-turf preflight admitted nothing; H5 living bank tied a frozen static
bank; rung-16 routing plane FAIL-CLOSED on both arenas). The honest prior is that this one also fails.

---

## 1. The four documents DISAGREE. Adjudicate, do not average.

You must resolve these explicitly. Do not paper over them.

- `wedge-power-analysis.md` → **PROCEED-WITH-CHANGES** (power passes by ~119x; re-scope Tier 0 to a
  30-minute harness preflight; real experiment is Tier 1/2).
- `wedge-operator-design.md` → **CUT**, on a structural dilemma: *fair ⟹ uninformative* (given `D`,
  the comparison reduces to the Garg–Kleinberg–Peng theorem with a matching lower bound) and
  *informative ⟹ not constructible gradient-free* (obtaining `D` needs dictionary learning, which is
  gradient-based and provably non-identifiable, arXiv:2512.05534).
- `wedge-feature-construction.md` → **BUILD** Route A only as a dictionary-uncertainty ladder A0→A3
  with A1 declared an unreportable positive control; **CUT** Route B (circular day-of-week); asserts
  the briefing conflates two different wedges.
- `wedge-baseline-design.md` → conditional; its recommendation **flips to CUT** if the primary metric
  is an accuracy delta at a single fixed `d` rather than a minimum-dimension ratio.

**Which of these is right?** Is there a coherent single design that survives all four objections
simultaneously, or do they collectively prove the cell is not runnable as conceived?

---

## 2. Specific contradictions between the docs that you must check

These are candidate L3/L4-class defects. Verify each against the actual documents and the committed
`wedge_power_sim/` source; report what you find, including if I have misread them.

1. **Solver class.** `wedge-operator-design.md` §3/§5 states that ISTA/FISTA are proximal-**gradient**
   methods and are **excluded on the letter** of the gradient-free constraint, requiring LARS-homotopy
   or coordinate descent. But `wedge-power-analysis.md` §0 runs its entire pilot with **FISTA, 250
   iterations**. Did the power analysis measure an operator the operator design disqualifies? Does
   switching to LARS/OMP change the measured effect, the sd, or the cost extrapolation?
2. **Which linear baseline was actually run.** `wedge-operator-design.md` §4 says CS-decoder vs a
   **sample-fit ridge probe (L0)** is *"flatly rigged"* and must never be the headline; only vs the
   analytic oracle **L1★** is legitimate. `wedge-baseline-design.md` independently ranks **oracle
   linear (GLS / whitened matched filter)** as the mandatory PRIMARY BAR. But the pilot in
   `wedge-power-analysis.md` used a **full-fit ridge trained on 20,000 labelled samples** — an L0.
   Is the headline effect (Δ ≈ 0.5326, "119x MDE") therefore measured against the baseline both other
   docs forbid? **What is Δ against the oracle linear baseline?** If nobody has computed it, say so
   plainly and state whether the whole power argument is unanchored until it is.
3. **Metric mismatch.** `wedge-baseline-design.md` says the primary metric must be a
   **minimum-dimension ratio**, and that an accuracy delta at fixed `d` re-imports HI-1's power
   problem. `wedge-power-analysis.md` sets the bar as an **accuracy delta (M1 support-recovery
   fraction) at fixed (d,m,k) = (128,4096,8)**. By the baseline doc's own stated trigger, does that
   flip the recommendation to CUT?
4. **Capacity-artifact gate.** The operator doc mandates a hard pre-declared VOID if sparsity-blind
   kernel ridge (L2) also beats the linear ceiling. Is that gate present in the power analysis's
   preregistered win conditions? If not, is a "win" distinguishable from the six prior
   "elaborate structure = more parameters" outcomes?

---

## 3. The four questions you must answer directly

### Q1. Is this test winnable-in-principle AND falsifiable AND fairly baselined?

Treat these as three separate gates; a design must pass all three.

- **Winnable-in-principle.** Is there a real outcome in which the treatment loses? The power analysis
  is candid that the effect is **119x the MDE** and that it *"could have written the result table from
  the thresholds alone"* — the interference heuristic predicted measured linear accuracy to within
  0.014. Is that a demonstration rather than an experiment? Note this is the **mirror image** of HI-1:
  HI-1 was so underpowered nothing could pass; this is so over-powered everything passes, and the
  power doc concedes the *null* cell's 0.005 effect is itself "statistically significant" at n=10,000.
- **Falsifiable.** Name the exact sentence in any of the four docs that no outcome could falsify, or
  confirm none exists. Is the pre-declared NEGATIVE branch pre-narrated the way HI-1's was (i.e. can
  you already write the write-up for either branch today)? Are the null-cell fail-closed
  (CI upper ≤ 0.05) and monotonicity conditions genuine risk, or conditions that cannot fail given
  the theory?
- **Fairly baselined.** Given §2.2 above — is the strongest linear opponent actually instantiated
  anywhere that a number has been measured? Name the single strongest linear baseline the treatment
  is obliged to beat, and state whether the current numbers beat it or are silent on it.

### Q2. Is the operator genuinely gradient-free?

Rule on both the **letter** and the **spirit**.

- FISTA/ISTA proximal-gradient vs LARS-homotopy vs coordinate descent vs OMP — which of these are
  honestly gradient-free, and does the distinction *matter scientifically* or is it terminological
  hygiene that changes no result? If it changes no result, say that; if it is load-bearing for the
  claim, say what breaks.
- Is "gradient-free" even the right constraint for what this program is trying to learn, or is it an
  inherited framing from the CUT'd HI-1 cell that no longer serves a purpose? If the constraint is
  doing no work, that is itself a finding.
- Does any arm smuggle in fitted parameters (λ selection, k via stopping rule, the ridge fit inside
  the pipeline) that make "zero learned parameters" false as stated?

### Q3. Is giving the operator the dictionary a cheat?

This is the crux. The operator doc argues it is **not** a cheat against `L1★` (both sides have full
information; they differ in hypothesis class only) — but that making it fair is *exactly* what makes
it a theorem re-derivation. Adjudicate:

- Is that dilemma real and unescapable, or is there a third cell the docs missed? Specifically
  assess the **dictionary-uncertainty ladder** (feature-construction A0→A3) and the measured claim
  that the wedge survives isotropic jitter to eps=0.4 but the required atom-cosine bar is **≥ ~0.98**.
  Is "estimate `D` gradient-free, then measure the wedge against *that*" a genuine escape from the
  dilemma, or does it inherit it?
- The power doc concedes isotropic Gaussian jitter is the **easy** perturbation and does not simulate
  the documented non-identifiability signature (merged / split / rotated / permuted atoms). Does the
  robustness result therefore carry any weight at all? Is a merge/split/rotate perturbation sweep
  decisive, or does it just relocate the same problem?
- Is the **coherence sweep** (Tier 0b) — the doc's own admission that a Gaussian dictionary is the
  maximally CS-favourable design with near-optimal RIP, and that real embedding dictionaries are
  coherent — the actual gate that determines everything downstream? Should it run *before* anything
  else, and can it alone CUT the line?
- Rule on the **W-0 applicability screen** (operator doc §7): deliberately asymmetric — can close the
  program permanently, can never open it, ~2 CPU-hours. Is a screen with only one informative branch
  worth running? Note the operator doc's own framing that decision value = P(KILL) × value(closing).

### Q4. Is the power math sound?

Audit the arithmetic and the assumptions, not just the conclusion.

- The paired normal-approximation formula `delta_MDE = (z_{alpha/2} + z_power) * sd_d / sqrt(n)`,
  Bonferroni `alpha = 0.05/6 = 0.00833`, `z = 2.6383`, worked example at n=10,000 giving 0.004468 —
  check it. Is Bonferroni over 6 cells the right correction given the cells are **not independent**
  (shared dictionary, nested dimensions, a preregistered monotone ordering)?
- `sd_d` is **measured under a large-effect alternative** (threat #3, disclosed). Is using it to set
  an MDE for a possibly-small true effect valid? Do the sd=0.30/0.50 sensitivity rows adequately
  bound this?
- Pilot is **n=1000, single seed (seed=1) per cell** (threat #4). Is that sufficient to fix `sd_d`?
- The eps=0.05/0.10 rows are **bit-identical** to eps=0 by construction (threat #5) — confirm this is
  expected rather than a harness bug, and that no conclusion leans on those rows.
- Is the **0.15 CI-lower-bound bar** defensible as "33.6x MDE80 and 30x the null-cell effect", or is
  it a number chosen because the pilot already clears it by 3.5x? Would you have set the same bar
  *before* seeing the pilot?
- Is the FLOP extrapolation (7.67 GFLOP/s measured on contended CPU → n=100,000 free on a 3090) sound,
  and does it still hold if FISTA is replaced by LARS-homotopy or OMP per Q2?

---

## 4. Deliver

Give a single verdict — **PROCEED / PROCEED-WITH-CHANGES / CUT** — plus:

1. **The single highest-value change.** One change, not a list. If the verdict is CUT, state what (if
   anything) should replace it, or state that the direction should close entirely.
2. **The strongest linear baseline this test is obliged to beat**, named concretely, and whether any
   measured number currently beats it.
3. **A ruling on gradient-free**: satisfied / violated / constraint is doing no work.
4. **A ruling on the dictionary**: cheat / not a cheat / not a cheat but therefore uninformative.
5. **A power verdict**: is the math correct, and is the bar honest or reverse-engineered from the
   pilot?
6. **Explicit adjudication of the four docs' conflicting verdicts** — which doc is right, and why the
   others are wrong.
7. **A one-paragraph bottom line**: is this worth the compute, or is it dominated by a known theorem
   / by the non-identifiability result, and should it be cut like HI-1 was?

Be brutally honest. Speculating is itself a failure — *"I don't know"* is the correct answer where
there is no evidence. If you cannot verify a claim against the documents or the committed pilot
source, say so rather than inferring it.
