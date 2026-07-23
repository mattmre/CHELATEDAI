# Feature construction for the within-space wedge test

Design doc, pre-registration input. Written 2026-07-19.
Successor to the CUT'd HI-1 prereg (`prereg-harmonic-invariance-HI1-draft.md`,
red-team verdict in `out_HI1_design_redteam.txt`), which recommended replacing the cross-space
alignment cell with a **within-space** readout test.

**Governing rule inherited from the HI-1 CUT: power math first, bar derived from it. A CUT is a win.**

All numbers marked *[measured]* in this document come from preflight runs I executed while writing
it, on this host (RTX 3090, torch 2.5.1+cu121, numpy 2.4.5). Scripts are checked in beside this doc at
`panel/wedge-preflight/`. Numbers marked *[extrapolated]* are scaled from the program's own D1 anchor
and are approximations, not measurements. Nothing here is quoted from a paper beyond what the briefing
asserted; specific claims about the cited papers are flagged **[re-verify before freeze]**.

---

## 0. Bottom line

| Question | Answer |
|---|---|
| **Route A (synthetic superposition)** | **BUILD — but only as the dictionary-uncertainty ladder (A0→A3), and only if A1 is declared an unreportable positive control.** As usually written ("show l1 beats a linear probe on a k-sparse code") it is a theorem re-demonstration with ~0 decision value. |
| **Route B (real encoder, circular day-of-week)** | **CUT.** It tests a *different* wedge from the compressed-sensing theorem; its headline claim ("a 1-D linear readout provably cannot represent the modular structure") is **empirically false** on a finite stimulus set *[measured, §4.2]*; and the version that repairs it is a one-line provable fact re-measured at 7B-model cost. |
| **Does B work as "reality check" for A?** | **No.** Not a weaker version of the same test — a different phenomenon. Passing B would say nothing about whether A's wedge is exploitable. |
| **The one open question worth GPU** | Not "does the wedge exist" (it does, and I reproduced it in minutes). It is: **does the wedge survive dictionary uncertainty at achievable estimator quality?** That has a sharp measured threshold and a cheap decisive test. |
| **Cost of the recommended ladder** | **< 3 GPU-hours, zero downloads, CPU-feasible.** Steps S1 and S2 are already done inside this document. |
| **Honest prior** | The dictionary-quality bar is **atom cosine ≥ ~0.98** *[measured]*, and the non-identifiability result (arXiv:2512.05534) predicts gradient-free estimators will miss it. Expect a seventh fail-closed — but a *cheap, definitive* one that closes the wedge line instead of leaving it as an alluring unexplored cell. |

---

## 1. There are TWO wedges, and the briefing conflates them

This matters more than any construction detail, because it determines whether Route B can serve as a
check on Route A.

**Wedge 1 — compressed-sensing / superposition (Garg–Kleinberg–Peng, arXiv:2602.11246).**
Features are stored in *superposition*: m ≫ d directions in R^d, activations k-sparse. Linear readout of
a given feature is corrupted by *interference* from the other active features. The quadratic gap
(nonlinear d = O(k log(m/k)) vs linear d = Õ(k² log m)) is a statement about **interference under
sparse superposition**. Route A instantiates exactly this.

**Wedge 2 — multi-dimensional / non-1-D features (Engels et al., arXiv:2405.14860).**
A feature is genuinely 2-D (a circle), and the *target function of it* is not an affine function of the
circular code. There is no superposition, no sparsity, no interference. The failure is that a periodic
code cannot affinely express a non-periodic (or differently-periodic) label. Route B instantiates this.

These share the words "linear can't read it out" and share nothing else. Wedge 2 requires no theorem —
it is the observation that `t` is not a sinusoid in `t`. **[re-verify before freeze]** the exact model,
layer, and causal-intervention protocol in arXiv:2405.14860 before citing specifics; I have not read
the paper in this session and will not invent layer numbers.

Consequence: **Route B cannot be "the reality check for Route A."** If the program wants a reality check
on the CS wedge, it must be a real-data version of the *CS* question (§6), not a circular-feature probe.

---

## 2. Power math, first

### 2.1 Route A is not power-limited — it is validity-limited

Route A has free n. *[measured]* At n_test = 4,000 samples × 4,096 features, the bootstrap 95% CI on
the paired per-feature AUC wedge is:

| cell | n_features | mean wedge | 95% CI | half-width |
|---|---|---|---|---|
| d=128, k=16, n_test=4000 | 4096 | +0.0570 | [+0.0560, +0.0579] | **0.0010** |
| d=128, k=16, n_test=1000 | 3994 | +0.0557 | [+0.0537, +0.0577] | **0.0020** |
| d=256, k=32, n_test=4000 | 4096 | +0.0567 | [+0.0561, +0.0574] | **0.0007** |

Half-width ~0.001 against effects of ~0.05. Power is a non-issue by two orders of magnitude. **This is
why the bar for Route A must be an effect-size bar and a fairness bar, not a significance bar** — with
free n, *everything* is significant, and "p < 0.001" would be a meaningless credential. The HI-1 failure
mode is inverted here: the risk is not a false negative from noise, it is a **false positive from a
rigged construction**.

### 2.2 Any real-data follow-on IS power-limited — and the numbers say full test sets only

Anchor: the program's own D1 paired-NDCG bootstrap, **half-width 0.055 at n = 60 queries** (from the
HI-1 red-team, sourced to the D1 ladder artifacts). Scaling 1/√n *[extrapolated]*, against real cached
query counts *[measured, `count_queries.py`, offline HF cache]*:

| dataset | test queries | ≈95% paired half-width | true effect needed for LB ≥ +0.01 |
|---|---|---|---|
| SciFact | **300** | 0.025 | 0.035 |
| NFCorpus | **323** | 0.024 | 0.034 |
| FiQA | **648** | 0.017 | 0.027 |
| ArguAna | **1406** | 0.011 | 0.021 |
| pooled (4 datasets) | **2677** | 0.008 | 0.018 |
| *HI-1's stratified UNKNOWN set* | *9–27* | *0.08–0.14* | *0.09–0.15* |

**The constructive finding:** HI-1 was underpowered because it *stratified down* to 9–27 queries, not
because the datasets are small. Evaluating on full test sets buys a factor of 5–10 in half-width and
makes wedge-scale effects (0.02–0.06) detectable. Any real-data arm must therefore be **whole-test-set,
no REPORT/UNKNOWN stratification.** If a design needs stratification to make its story work, it is the
HI-1 design again and should be cut again.

**The honest limit on that finding:** the synthetic wedge is measured in *per-feature detection AUC*;
the real-data bar is in *NDCG@10*. **There is no established conversion between them.** A +0.05 AUC
wedge does not license a +0.05 NDCG prediction, and I will not pretend it does. This is the single
biggest reason the real-data arm (§6) must stay *gated* behind the synthetic decision, not run in
parallel.

---

## 3. Route A — synthetic superposition

### 3.1 Exact construction

```
d       ambient dimension          (swept: 32, 64, 128, 256, 512)
m       number of features         (4096; require m >> d)
k       active features per sample (swept: 4, 8, 16, 32)
sigma   observation noise          (0.01)

F in R^{d x m}:  columns f_i ~ N(0, I_d), then normalized to unit norm.
                 => typical coherence |<f_i,f_j>| ~ 1/sqrt(d)  (almost-orthogonal)
support S:       exactly k indices, uniform without replacement, per sample
activations a:   a_i ~ Uniform[0.5, 1.5] for i in S, 0 otherwise  (POSITIVE, bounded away from 0)
observation:     x = F a + sigma * eps,   eps ~ N(0, I_d)
```

Positive, bounded-away-from-zero activations are deliberate: they match the non-negativity that makes
sparse coding well-posed and they keep the detection label unambiguous. Zero-mean Gaussian activations
would put mass arbitrarily close to 0 and make "is feature i active" a label with no signal at the
boundary — a self-inflicted noise floor.

### 3.2 Ground-truth label

Two labels, and **they are not interchangeable** — this is the trap that eats naive versions:

- **L-detect:** `y_i = 1{i in S}` — is feature i active? Metric: per-feature detection AUC.
- **L-magnitude:** `y_i = a_i` — how active is it? Metric: R².

### 3.3 Why a linear probe should fail — and how much it actually fails

The optimal linear readout of feature i suffers interference: `<w, x> = a_i + Σ_{j∈S, j≠i} a_j <w, f_j>`,
with interference variance ~ k·E[a²]/d. Below the CS phase transition, that term dominates.

**But the size of the failure depends entirely on the label.** *[measured, preflight 1 + 2]*, with
m = 4096, comparing the **population-optimal linear MMSE decoder** (fit with 20,000 free training
samples — i.e. the strongest possible linear readout, not a weak probe) against l1/FISTA with the true
dictionary:

| d | k | linear R² | CS R² | linear per-feature AUC | CS per-feature AUC |
|---|---|---|---|---|---|
| 64 | 8 | 0.011 | 0.868 | 0.947 | 0.983 |
| 128 | 16 | 0.022 | 0.984 | 0.944 | 1.000 |
| 256 | 32 | 0.048 | 0.998 | 0.945 | 1.000 |
| 128 | 8 | 0.023 | 0.998 | 0.983 | 1.000 |
| 512 | 32 | 0.100 | 0.999 | 0.984 | 1.000 |

**The R² wedge is ~45×. The AUC wedge is +0.017 to +0.057.** Both are honest; they measure different
things. Linear readout **recovers the ordering** of feature activity almost perfectly (AUC 0.94–0.98)
while **failing completely at magnitude** (R² 0.01–0.10), because the target is 99.6% zeros and the
linear estimator cannot resolve the magnitudes of the rare actives through the interference.

> **Design consequence, and the most likely way this experiment gets rigged to win:** reporting the R²
> wedge as "the wedge" would advertise a ~45× effect that is largely a **calibration artifact**. If the
> downstream consumer only needs *ranking* (retrieval does), the honest wedge is **+0.02 to +0.06 AUC** —
> the same order as rung-16's entire effect. Pre-register **both**, and pre-register **which one the
> claim is about**. Any write-up that leads with R² is L4 partial-as-complete.

### 3.4 The baseline must be a detector, not a regressor

*[measured, preflight 2]* I checked whether the AUC wedge is an artifact of scoring a
regression-fit readout on a detection task. Fitting a **per-feature linear detector** (ridge onto the
binary active/inactive target) instead:

| d | k | MMSE-fit per-feature AUC | **detector-fit per-feature AUC** | CS | wedge vs detector |
|---|---|---|---|---|---|
| 64 | 8 | 0.947 | 0.946 | 0.983 | **+0.0370** |
| 128 | 16 | 0.944 | 0.943 | 1.000 | **+0.0565** |
| 256 | 32 | 0.945 | 0.944 | 1.000 | **+0.0562** |
| 128 | 8 | 0.983 | 0.982 | 1.000 | **+0.0181** |
| 512 | 32 | 0.984 | 0.983 | 1.000 | **+0.0169** |

The wedge survives a properly-fit linear detector. Good — it is not a baseline artifact. Freeze the
detector-fit variant as the baseline anyway; it is the strongest linear arm and costs nothing.

### 3.5 The window must be located, not assumed — CS *loses* outside it

*[measured, preflight 1]* Below the CS phase transition, l1 decoding is **worse** than linear:

| d | k | linear AUC | CS AUC | wedge |
|---|---|---|---|---|
| 32 | 8 | 0.883 | 0.656 | **−0.228** |
| 64 | 16 | 0.878 | 0.702 | **−0.175** |
| 128 | 32 | 0.877 | 0.785 | **−0.092** |
| 128 | 16 | 0.942 | 1.000 | +0.057 |
| 256 | 32 | 0.944 | 1.000 | +0.056 |

The predicted window `k log(m/k) ≲ d ≲ k² log m` is real but the constants are not given by the theory.
The wedge is a narrow ridge just above the phase transition, and it is **negative** below it. **Sweep to
locate the window; never assume a cell.** A design that picks one (d, k) a priori has a ~50% chance of
picking a cell where the "nonlinear" arm loses by 0.2 AUC and generating a spurious negative.

### 3.6 The confound that makes plain Route A worthless — and the ladder that fixes it

**The oracle problem.** In §3.3–3.5 the l1 decoder is handed the ground-truth dictionary F. Route A as
usually written therefore proves only what arXiv:2602.11246 already proves. Its decision value is
**zero**: no outcome changes what the program does next. Running it and reporting it would be the
research equivalent of a passing smoke test presented as a feature.

**What is actually unknown** is whether the wedge survives when F must be *estimated* — and sparse
dictionary learning is provably non-identifiable (arXiv:2512.05534: zero reconstruction loss while
recovering zero ground-truth features, empirically 3/3200) **[re-verify the 3/3200 figure before
freeze]**. So the ladder:

| arm | dictionary | role |
|---|---|---|
| **A0** | — | Per-feature linear detector, 20k free training samples. **The bar.** |
| **A1** | true F | l1/FISTA. **Positive control ONLY.** May not be reported as a finding. If A1 does not beat A0 inside the located window, the harness is broken → REGIME-INVALID, fix and re-run. |
| **A2** | F̂, gradient-free (k-SVD / MOD / FastICA / NMF) | **The decision arm.** Matches the program's gradient-free framing (mini-vec2vec, arXiv:2510.02348). |
| **A3** | F̂, gradient-trained SAE | **Upper-bound reference, clearly labelled.** Violates the gradient-free framing on purpose: if even a trained SAE misses the quality bar, every gradient-free arm certainly does, and the line closes at once. Run A3 *first* as a kill-fast screen. |

### 3.7 The dictionary-quality bar is measurable — and I measured it

*[measured, preflight 2]* Perturbing the dictionary (`F̂ = normalize(F + ε·G)`) and sweeping ε:

| d, k | mean atom cosine to truth | linear detector AUC | CS AUC | wedge |
|---|---|---|---|---|
| 128, 16 | 1.000 | 0.943 | 1.000 | +0.0564 |
| 128, 16 | 0.995 | 0.943 | 0.999 | +0.0555 |
| 128, 16 | **0.981** | 0.943 | 0.989 | **+0.0459** |
| 128, 16 | **0.929** | 0.943 | 0.935 | **−0.0079** |
| 128, 16 | 0.782 | 0.943 | 0.844 | −0.0996 |
| 256, 32 | 0.981 | 0.944 | 0.999 | +0.0552 |
| 256, 32 | 0.929 | 0.944 | 0.964 | +0.0206 |
| 256, 32 | 0.781 | 0.944 | 0.867 | −0.0772 |

**The wedge dies between mean atom cosine 0.98 and 0.93.** That is a sharp, pre-registerable
sufficient-condition screen:

> **S3 screen (the whole experiment, compressed into one number):** on the synthetic instrument, does any
> estimator reach **mean atom cosine ≥ 0.98** against the known ground-truth dictionary? If no → the
> wedge is unexploitable without ground truth, and the line closes. Cost: minutes.

Two honesty notes on this threshold:
1. My perturbation model is **isotropic noise on atoms — optimistic**. Real dictionary-learning error
   includes atom splitting, merging, duplication and permutation ambiguity, which are worse than
   isotropic noise at equal cosine. **0.98 is a lower bound on the required quality.**
2. Atom cosine must be computed after **optimal bipartite matching** between F̂ and F (Hungarian on the
   |cosine| matrix), otherwise permutation alone will fake a failure.

### 3.8 Remaining confounds in Route A, and the mitigations

| # | Confound | Why it rigs the result | Mitigation (pre-register) |
|---|---|---|---|
| C1 | Generative model = decoder's assumed model (exact k-sparsity, exact dictionary) | Free win for l1 | Also run **approximate sparsity** (power-law-decaying activations, no exact zeros) and **correlated supports** (features co-occur in blocks). If the wedge only exists under exact k-sparsity, say so — that *is* the finding. |
| C2 | Feature frequency uniform | Real features are Zipfian; rare features are where linear should be worst | Add a Zipf-frequency arm |
| C3 | Target feature cherry-picked | Interference differs per feature | Average over **all** m features (as done above), never a hand-picked i |
| C4 | λ tuned on the test set | Classic | λ selected on a **separate validation draw** (done in the preflights); freeze the grid |
| C5 | Capacity mismatch | l1 gets m=4096 atoms; linear gets d params | Report it explicitly. It is not fixable — it is the theorem. But it means "nonlinear wins" ≠ "nonlinear is efficient" |
| C6 | No normalization/LayerNorm | Real encoder outputs are normalized | Add a unit-norm arm; cheap |
| C7 | Metric shopping between R² and AUC | §3.3 | Pre-declare the primary metric **before** looking |

### 3.9 Cost

Pure numpy/torch, no downloads, no external data. Every preflight in this document ran in **1–3 minutes**
on the 3090; the full ladder including dictionary learning on 20k×256 is **< 3 GPU-hours**, and is
CPU-feasible overnight. This is the cheapest decisive cell the program has had in this arc.

---

## 4. Route B — real encoder, circular feature

### 4.1 Exact construction (as it would have to be built)

```
model     an LLM with an accessible residual stream (Engels-style), NOT a sentence encoder
stimuli   day-of-week (7) or month (12) mentions, embedded in many surface CONTEXTS
          e.g. C templates x 7 days, C in the hundreds
extract   residual-stream activation at a chosen layer, at a chosen token position
label     t in {0..6} (day index)
claim     x(t) ≈ c + u·cos(2πt/7) + v·sin(2πt/7) + noise
readout   linear:    y_hat = <w, x> + b
          nonlinear: theta = atan2(<v,x>, <u,x>), then any function of theta
```

Ground truth: the day index `t`, known by construction from the prompt.

### 4.2 Why the "linear provably fails" claim is FALSE as usually stated

The briefing's framing — "a 1-D linear readout provably cannot represent the modular structure" — is
true of the *idealized rank-2 circle* and **false of any probe fit on a finite stimulus set**.

Seven stimuli in R^768 with any per-item variation are **affinely independent almost surely**, and an
affine functional can therefore interpolate *any* labeling of them exactly.

*[measured, preflight 4]* Best affine probe, fit and evaluated on 7 day-representations, dim = 768:

| off-plane per-item noise | target | max abs residual |
|---|---|---|
| 0.00 (exact rank-2 circle) | parity `t%2` | 6.07e-01 |
| 0.00 | `t mod 3` | 1.20e+00 |
| 0.00 | ordinal `t` | 2.00e+00 |
| **0.01** | parity `t%2` | **6.4e-05** |
| **0.01** | `t mod 3` | **1.1e-04** |
| **0.10** | parity `t%2` | **6.4e-07** |
| **0.10** | ordinal `t` | **2.0e-06** |

A whisper of off-plane noise, and the linear probe fits parity to machine precision. **A Route B design
that fits and scores a probe on the 7 (or 12) stimuli measures nothing.** This is exactly the class of
error that killed HI-1 — a protocol that decides the answer before the phenomenon does — and it is
easy to write by accident.

### 4.3 The repaired version — and why it is still not worth running

The repair is many contexts, ridge selected on held-out contexts, scored on further held-out contexts,
so per-item noise cannot be exploited.

*[measured, preflight 5]* Fair regularized linear probe vs the analytic rank-2 ceiling:

| target | pure-circle affine ceiling (RMSE) | fair ridge probe, held-out contexts | nonlinear atan2 readout | target sd |
|---|---|---|---|---|
| parity `t%2` | 0.4820 | **0.4820–0.4822** | **0.0000** | 0.4949 |
| `t mod 3` | 0.7958 | **0.7942–0.7956** | **0.0000** | 0.8330 |
| ordinal `t` | 1.1593 | **1.1598–1.1630** | **0.0000** | 2.0000 |

(across off-plane noise 0.05/0.20 and 20/100 training contexts; held-out day recovery from the atan2
readout was **1.0000** in every configuration)

The repaired design works perfectly and produces a gigantic, trivially-powered effect. **That is the
problem.** The linear probe lands *exactly* on an analytic ceiling I can compute in closed form from
the Fourier coefficients of the target against a 7-point circular code. The experiment's outcome is
known before it runs, to four decimal places, conditional only on "the circle exists in this model."
That is a **measurement of whether a known feature is present**, not a hypothesis test — decision value
approximately zero, at the cost of a ~15 GB model download and GPU time.

### 4.4 Route B's confounds, for completeness

| # | Confound |
|---|---|
| B1 | **Feature may simply not be there** in whatever model/layer is chosen — and then the null is ambiguous (feature absent vs. extraction too weak), the worst kind of negative |
| B2 | **Wrong substrate available locally.** This host has `bge-large-en-v1.5` (1024-d) and `nomic-embed-text-v1` (768-d) — *sentence encoders*. Circular day-of-week features were demonstrated in **LLM residual streams**; assuming they survive into a pooled sentence embedding is **unverified** and probably false |
| B3 | Surface-form / tokenization / frequency confounds across day names ("Monday" vs "Mon" vs "the 3rd") |
| B4 | Layer and token-position shopping — many layers × positions × pooling = a large garden of forking paths for a single binary claim |
| B5 | Plane recovery from class means is itself a fitted step and needs its own held-out discipline |

### 4.5 Cost

~15 GB model download (not currently cached on this host), then minutes of inference per layer × a
layer sweep. Call it 1–2 GPU-hours plus download and integration work — **more than the entire Route A
ladder**, for a result whose value is bounded above by "yes, the published feature reproduces."

### 4.6 Verdict on B: CUT

Different wedge (§1) · headline claim false as stated (§4.2) · repaired version is an analytic identity
re-measured (§4.3) · most expensive option · substrate not locally available · ambiguous null.

---

## 5. Recommendation

**Run Route A as the S1→S4 ladder. Cut Route B. Do not run them in parallel.**

| step | what | cost | decision |
|---|---|---|---|
| **S1** | Locate the (d,k,m) window; measure the wedge under a fair linear **detector** baseline, in **both** R² and AUC | **DONE in this doc** | Window found. Fair wedge = **+0.017…+0.057 AUC**; R² wedge is largely a calibration artifact |
| **S2** | Measure the dictionary-quality threshold | **DONE in this doc** | Wedge dies between atom cosine **0.98 and 0.93** |
| **S3** | **THE DECISION.** Can any estimator reach mean atom cosine ≥ 0.98 (Hungarian-matched) on the instrument? Run gradient-trained SAE **first** as a kill-fast upper bound, then gradient-free (k-SVD / MOD / FastICA / NMF) | ~1–3 GPU-hours | **If no → CLOSE the wedge line, write it up, done.** If yes → S4 |
| **S4** | *Gated on S3 only.* Real-encoder bridge (§6) | separate design + prereg | — |

Why this ordering is right: S3 is the only step whose outcome is unknown, it is the cheapest step, and
it is decisive in the negative. The program's repeated failure mode has been running the expensive
ambiguous thing before the cheap decisive thing.

### 5.1 Pre-registered decision rules for S3

- **Primary metric:** mean per-feature detection AUC, Hungarian-matched, averaged over all m features,
  inside the S1-located window. Declared **before** any A2/A3 run.
- **Secondary (descriptive only):** R². Explicitly labelled as magnitude-recovery, never as "the wedge."
- **Bar:** A2 (or A3) beats A0 by **≥ +0.02 AUC** with bootstrap LB > 0. The +0.02 floor is set by the
  *smallest fair wedge the oracle arm itself achieves* (+0.017 at d=512,k=32) — i.e. we require the
  estimated-dictionary arm to retain roughly the weakest oracle-grade advantage. With half-width ~0.001
  this bar is about effect size, not significance, by design.
- **Positive control:** A1 must clear the bar. If it does not, the run is **REGIME-INVALID** — fix the
  harness, do not report.
- **Negative control:** A2 with a **random** dictionary must lose. If a random dictionary wins, the
  metric is broken.
- **Kill criteria (declare and stop):**
  - No (d,k) cell where A1 clears the bar → the instrument does not instantiate the wedge → fix or CUT.
  - Best achievable matched atom cosine < 0.93 across all estimators → **CLOSE**, wedge unexploitable
    without ground truth.
  - Atom cosine in [0.93, 0.98] with wedge < +0.02 → **CLOSE**, with the measured curve as the evidence.
- **Anti-shopping:** one primary metric, one window (frozen after S1), one bar. Every additional
  sparsity model / frequency model / normalization arm is **descriptive** and reported as such.
- **Pre-narration ban (the HI-1 lesson):** the write-up for a negative is *"gradient-free dictionary
  estimation does not reach the quality the wedge requires; measured threshold X, achieved Y"* — a
  number, not a narrative. If the write-up can be drafted before the run with only blanks to fill,
  the blanks must be **numbers**, not adjectives.

---

## 6. What replaces Route B as the reality check (gated on S3, not authorized here)

If and only if S3 passes, the real-data question is the *CS* question, not the circular one:

> On a frozen real encoder, does l1 decoding over a gradient-free-estimated dictionary beat the fair
> linear probe on a label we actually care about?

Design constraints derived from §2.2:
- **Whole test sets, no stratification.** SciFact 300 / NFCorpus 323 / FiQA 648 / ArguAna 1406 *[measured]*,
  all cached locally. Pooled half-width ≈ 0.008 *[extrapolated]*.
- **Bar must be ≥ 0.02 NDCG** to clear the pooled half-width with margin; ArguAna or FiQA alone can carry
  a 0.021–0.027 effect.
- **The unresolved gap, stated plainly:** the synthetic wedge is in AUC, the real bar is in NDCG, and
  **no conversion between them is established**. So the real-data arm's effect size is genuinely unknown
  a priori — meaning its power analysis cannot be completed until S3 tells us the wedge survives at all.
  **This is why it must not be designed now.** Designing it now would be exactly the HI-1 error: writing
  a bar before knowing the achievable effect.
- **Baseline to beat:** the repo's leakage-safe full-fit affine ridge `Wx + b` with λ on CAL, plus a
  capacity-matched reduced-rank variant — as the HI-1 red-team specified. Not "best of ridge/Procrustes."

---

## 7. What would make me wrong

- **If the S3 screen is too strict.** The 0.98 threshold comes from *isotropic* atom perturbation. A real
  estimator's errors might be *structured* in ways that hurt l1 decoding less than isotropic noise at the
  same cosine (e.g. errors concentrated in directions orthogonal to the active support). If so, an
  estimator at cosine 0.95 might still work. **Mitigation:** in S3, don't stop at the cosine number —
  always also run the actual decode with F̂. The cosine screen is a cheap *predictor*, and the decode is
  the *measurement*. If they disagree, the decode wins and the screen is recalibrated.
- **If ranking is the wrong consumer.** I argued the AUC wedge (0.02–0.06) is the honest one because
  retrieval needs ranking. If a downstream consumer genuinely needs calibrated magnitudes (e.g. a gating
  or budgeting decision on feature strength), the honest wedge is the R² one (~45×) and the economics
  change completely. **This should be resolved before S3, by naming the consumer.**
- **If the real encoder is not in superposition at the k that matters.** Everything here assumes the
  real regime sits inside the window. If real encoders sit far above the phase transition (d ≫ k² log m),
  linear is already near-optimal and the wedge is irrelevant in practice regardless of S3.

---

## 8. Limitations of this document

1. All preflight numbers are from **simulations I wrote today**, single seed per cell, no multi-seed
   variance except the bootstrap in §2.1. They are calibration, not results. Any of them entering a PR
   or paper must be re-run multi-seed.
2. The `d=32, k=8` and similar sub-transition CS results depend on the λ grid; a wider grid or a
   different solver (OMP, homotopy) could change the negative-wedge magnitudes. The *sign* is robust
   (below the transition l1 cannot recover), the magnitude is not.
3. The D1 half-width anchor (0.055 at n=60) is inherited from the HI-1 red-team, not re-derived here,
   and the 1/√n scaling across datasets is an approximation that ignores per-dataset difficulty variance.
4. Paper-specific claims (Garg–Kleinberg–Peng bounds, Engels et al. protocol, the 3/3200
   non-identifiability figure, mini-vec2vec) are taken from the briefing and **must be re-verified against
   the sources before any of them appear in a prereg or paper**. I did not fetch any paper in this session.
5. ~~The FISTA implementation is unvalidated.~~ **Closed** *[measured, preflight 6]*: validated against
   `sklearn.linear_model.Lasso` (float64, tol 1e-12) in both the unconstrained and non-negative modes at
   λ ∈ {0.01, 0.06}. Objective gap ≤ **1e-10**, max coefficient difference ≤ **7.7e-05**. The nonlinear
   arm is not a source of false negatives.
6. The dictionary-learning estimators for S3 (k-SVD / MOD / FastICA / NMF / SAE) are **not implemented
   or benchmarked** anywhere in this document. S3's cost estimate (~1–3 GPU-hours) is therefore an
   estimate, not a measurement, and is the one number here most likely to be wrong.
