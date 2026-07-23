# Within-space wedge test — load-bearing power analysis

**Date:** 2026-07-19
**Status:** power math done FIRST, bar derived from it (the HI-1 lesson)
**Verdict:** **PROCEED-WITH-CHANGES** — power PASSES by ~119x, but Tier 0 as proposed is a
theorem demonstration with near-zero decision value. Re-scope: Tier 0 becomes a 30-minute
harness preflight; the real experiment is Tier 1/2. Details in §6.

All numbers below are **measured**, not asserted. Pilot simulation source is committed alongside
this doc in `panel/wedge_power_sim/`: `wedge_pilot.py` (6 cells), `wedge_control.py` (null cells),
`wedge_misspec.py` (dictionary sweep), `power2.py` + `final_numbers.py` (power arithmetic).
Reproduced with numpy 2.4.5 / scipy 1.16.3, CPU only, no GPU used.
Nothing here is fabricated or cited from memory.

---

## 0. What was actually simulated

One frozen space with **known** superposed features:

- Dictionary `A` (d x m), Gaussian, unit-norm columns.
- Ground truth `s`: exactly `k` active atoms per sample, values `U(0.5,1.5)` with random sign.
- Observation `x = A s (+ sigma * noise)`.
- **Linear arm (baseline, deliberately strong):** full-fit ridge `W: x -> s` trained on
  **20,000 labelled samples**, ridge lambda tuned over `{1e-6,1e-4,1e-2,1,1e2}` on a held-out
  1,000-sample validation split. Readout = top-k of `Wx`.
- **Nonlinear arm (treatment):** gradient-free FISTA / l1 basis pursuit against `A`,
  250 iterations, `lambda = 0.02 * max|A^T x|`. Readout = top-k of `|s_hat|`.

**Fairness note (load-bearing).** Both arms have full information. The linear probe is trained on
ground-truth labels and is the *optimal* linear map; the l1 arm is given the dictionary. Any gap is
therefore **fundamental** (interference / span-exhaustion), not informational. This is the one
respect in which this design is cleaner than HI-1's "best of ridge/Procrustes" strawman — the
baseline here cannot be accused of being under-fed. Both arms use the same top-k readout, so `k` is
not a treatment-only advantage. The baseline gets a tuned knob; the treatment's lambda was fixed a
priori and never swept.

---

## 1. Candidate metrics and their variance behaviour

| # | Metric | Type | Variance behaviour | Verdict |
|---|---|---|---|---|
| **M1** | per-sample **support-recovery fraction** `|supp ∩ supp_hat| / k` | paired, bounded [0,1] | **measured sd of the paired difference = 0.105–0.166** across all wedge cells | **PRIMARY** |
| M2 | per-sample **exact** support recovery (0/1) | paired binary, McNemar | discordance-driven; sd of paired diff 0.00–0.50 | secondary (confirmatory) |
| M3 | per-coordinate `R^2` / corr of `s_hat_j` vs `s_j` | continuous | lower variance than M1 | rejected: rewards magnitude, not identity — a method can win M3 while getting the support wrong |
| M4 | downstream retrieval NDCG | paired, per-query | **this is the metric that killed HI-1 and produced rung-16's 0.003–0.03 effects with wide CIs** | **explicitly NOT primary** — descriptive only |

M1 is chosen because it is (a) paired, killing between-sample variance, (b) bounded, so sd is
bounded and cannot blow up, (c) granular (k+1 levels, not 2), which is why its sd (~0.13) is roughly
4x smaller than M2's worst case (0.50), and (d) directly the quantity the theory makes a statement
about.

**M4 is de-recommended on purpose.** Re-importing NDCG as primary would re-import HI-1's failure
mode wholesale. It stays in the report as a descriptive secondary, with no bar attached.

---

## 2. Achievable n — the claim that n is FREE, verified and quantified

**Verified: TRUE for the synthetic tiers, and this is the structural advantage over HI-1.** HI-1's n
was ~9–27 because queries had to be *harvested* from a labelled corpus. Here samples are
*generated*, so n is a compute question, not an inventory question.

Measured FISTA cost (the dominant term; `flops = 4 * d * m * n * iters`):

```
measured: d=128, m=4096, n=1000, iters=250
          = 5.243e11 flops in 68.4 s  ->  7.67 GFLOP/s effective
          (contended CPU: three python workers sharing BLAS threads)
```

Extrapolating at that measured, deliberately pessimistic rate:

| n | flops | CPU @ 7.67 GF/s | 3090 @ ~15 TF/s fp32 |
|---|---|---|---|
| 1,000 | 5.24e11 | **68 s** (measured) | <1 s |
| 10,000 | 5.24e12 | **11.4 min** | ~0.3 s |
| 100,000 | 5.24e13 | 114 min | **~3.5 s** |

**Conclusion: n = 10,000 is free on CPU. n = 100,000 is free on the 3090.** Full pilot (6 cells,
n=1000) completed in ~22 min of contended CPU with no GPU at all.

For Tier 2 (real frozen embedding space) n is still large: an Engels-style generated prompt set with
programmatic labels (day-of-week, month) can be built at 100k items and encoded by MiniLM on a 3090
in ~1 minute. **n is not the binding constraint anywhere in this design.** That is the whole reason
this test is worth analysing where HI-1 was not.

---

## 3. Minimum detectable effect

**Method: paired normal-approximation power formula**, `delta_MDE = (z_{alpha/2} + z_{power}) * sd_d / sqrt(n)`.

Chosen over bootstrap because (a) the outcome is a bounded paired mean, so the CLT applies fast and
the normal approximation is if anything conservative at these n, (b) `sd_d` is *measured*, not
assumed, so the formula's only input is empirical, and (c) it is auditable arithmetic rather than a
resampling black box. A paired bootstrap was cross-checked conceptually via the half-width column
(`1.96 * sd_d / sqrt(n)`), which is the same quantity.

`alpha` is **Bonferroni-corrected over 6 preregistered cells**: `alpha = 0.05/6 = 0.00833`,
two-sided, so `z_{alpha/2} = 2.6383`. `z_{0.80} = 0.8416`, `z_{0.95} = 1.6449`, `z_{0.975} = 1.9600`.

### 3a. At the operating cell's measured sd (`sd_d = 0.1284`)

| n | 95% half-width | **MDE @ 80% power** | MDE @ 95% power |
|---|---|---|---|
| 200 | 0.01780 | **0.03159** | 0.03971 |
| 1,000 | 0.00796 | **0.01413** | 0.01776 |
| 10,000 | 0.00252 | **0.00447** | 0.00562 |
| 100,000 | 0.00080 | **0.00141** | 0.00178 |

Worked example, n=10,000: `(2.6383 + 0.8416) * 0.1284 / sqrt(10000) = 3.4799 * 0.1284 / 100 = 0.004468`.

### 3b. Uncorrected-alpha and worse-sd sensitivity (alpha = 0.05, `z = 1.96`)

| sd_d | n=200 | n=1,000 | n=10,000 | n=100,000 |
|---|---|---|---|---|
| 0.13 | 0.02575 | 0.01152 | 0.00364 | 0.00115 |
| 0.30 | 0.05943 | 0.02658 | 0.00840 | 0.00266 |
| 0.50 (worst case) | 0.09905 | 0.04430 | 0.01401 | 0.00443 |

Even at the theoretical worst case for a bounded paired binary difference (`sd = 0.5`), n=10,000
gives MDE80 = 0.014.

### 3c. M2 (exact recovery) via McNemar

Exact two-sided McNemar with all discordant pairs favouring one arm reaches p<0.05 at **6 discordant
pairs** (`2 * 0.5^6 = 0.03125`). In the pilot's operating cell the discordance was 1000/1000, all
one way. M2 is over-powered to the point of triviality; it is confirmatory only.

---

## 4. Theory translation — Garg–Kleinberg–Peng, and does it match measurement?

GKP (arXiv:2602.11246) predict a **quadratic** dimensional gap: nonlinear/l1 decoding of k-sparse
features needs `d = O(k log(m/k))`, linear accessibility needs `d = Õ(k^2 log m)`, with a matching
lower bound. Corroborating: Anthropic superposition work; Engels et al. ICLR 2025
(arXiv:2405.14860) causally-verified 2D circular features.

Concrete thresholds (nats):

| d | m | k | `2k ln(m/k)` (CS) | `k^2 ln m` (linear) | d / CS | d / linear | regime |
|---|---|---|---|---|---|---|---|
| 128 | 4096 | 8 | 99.8 | 532.3 | 1.28 | **0.24** | **wedge** |
| 96 | 4096 | 6 | 78.3 | 299.4 | 1.23 | **0.32** | **wedge** |
| 256 | 8192 | 12 | 156.6 | 1297.6 | 1.63 | **0.20** | **wedge** |
| 64 | 2048 | 5 | 60.2 | 190.6 | 1.06 | **0.34** | **wedge** |
| 512 | 4096 | 8 | 99.8 | 532.3 | 5.13 | 0.96 | transition |
| 1024 | 4096 | 8 | 99.8 | 532.3 | 10.26 | **1.92** | **null / control** |

Wedge cells sit **above** the compressed-sensing threshold and **far below** the linear one — exactly
where the theory says the gap must open.

### Translating the dimension gap into an expected accuracy gap

For the linear top-k readout, signal = 1.0 and per-coordinate interference has sd `sqrt(k/d)`, so
the expected worst distractor over `m-k` atoms is `sqrt(k/d) * sqrt(2 ln(m-k))`. When that ratio
approaches 1.0, roughly half the true atoms get outranked, predicting `lin_frac ≈ 0.5`.

**Prediction vs measurement (n=1000 per cell):**

| d | m | k | predicted max-distractor/signal | **predicted lin_frac** | **measured lin_frac** | measured l1_frac |
|---|---|---|---|---|---|---|
| 128 | 4096 | 8 | 1.020 | ~0.5 | **0.4863** | 1.0000 |
| 96 | 4096 | 6 | 1.020 | ~0.5 | **0.4228** | 1.0000 |
| 256 | 8192 | 12 | 0.919 | ~0.5 | **0.4830** | 1.0000 |
| 64 | 2048 | 5 | 1.091 | ~0.5 | **0.4758** | 0.9992 |
| 512 | 4096 | 8 | 0.510 | high | **0.9204** | 1.0000 |
| 1024 | 4096 | 8 | 0.360 | ~1.0 | **0.9950** | 1.0000 |

The theory's quantitative prediction lands within 0.014 of measurement at the operating cell. The
l1 side is deep in the Donoho–Tanner success phase (`delta = d/m = 0.031`, `rho = k/d = 0.0625`) and
achieves exactly 1.000.

### Named settings where the gap is LARGE and unambiguous

**(d, m, k) = (128, 4096, 8)** — primary. Also (96, 4096, 6), (256, 8192, 12), (64, 2048, 5).
Measured `Delta` (M1, l1 minus linear), n=1000:

| cell | Delta | sd_d | l1 exact | linear exact |
|---|---|---|---|---|
| (128, 4096, 8), sigma=0 | **0.5138** | 0.1220 | 1.000 | 0.000 |
| (128, 4096, 8), sigma=0.05 | **0.5326** | 0.1284 | 0.998 | 0.000 |
| (96, 4096, 6) | **0.5772** | 0.1522 | 1.000 | 0.000 |
| (256, 8192, 12) | **0.5170** | 0.1047 | 1.000 | 0.000 |
| (64, 2048, 5) | **0.5234** | 0.1655 | 0.998 | 0.001 |
| (512, 4096, 8) — transition | 0.0796 | 0.0792 | 1.000 | 0.444 |
| **(1024, 4096, 8) — null control** | **0.0050** | 0.0245 | 1.000 | 0.960 |

**Dose-response is monotone in `d/k^2 ln m` exactly as predicted: 0.514 → 0.080 → 0.005.**

---

## 5. The bar — derived from the power math, not proposed and hoped for

Operating point: **M1, n = 10,000, (d, m, k) = (128, 4096, 8), sigma = 0.05.**

- MDE80 (Bonferroni, 6 cells) at that n = **0.00447**.
- Measured pilot effect = **0.5326**, i.e. **119x the MDE**. 95% CI at n=10,000 = [0.5301, 0.5351].

**Preregistered win conditions (all three must hold):**

1. **Effect bar:** in each of the four wedge cells, the paired 95% CI **lower bound** on Delta(M1)
   must be **≥ 0.15**.
   *Justification:* 0.15 is **33.6x** MDE80 at n=10,000 and **30x** the measured null-cell effect
   (0.005), so it cannot be cleared by nuisance or by a subtly broken harness. Pilot CI lower bound
   is 0.5301 — clears the bar with **0.380 of headroom (3.5x)**.
2. **Null-cell fail-closed:** the control cell (1024, 4096, 8) must show CI **upper** bound ≤ 0.05.
   Measured at n=1000: CI = [0.00348, 0.00652]; upper bound 0.0065. Passes.
3. **Monotonicity:** `Delta(d=128) > Delta(d=512) > Delta(d=1024)`, preregistered ordering.

**Methodological note that the bar exists to enforce:** at n=10,000 the *null* cell's effect
(0.005) is itself "statistically significant" (CI [0.00452, 0.00548], excludes zero). **A p-value
bar would be meaningless here and would have manufactured a false positive in the control cell.**
Only an effect-size bar is honest at free n. This is the mirror-image trap of HI-1's — HI-1 was
underpowered so nothing could pass; this design is so over-powered that *everything* passes a
significance test.

---

## 6. VERDICT

### 6a. Power question, as asked: **PASS, overwhelmingly.**

There is a concrete configuration where the bar is clearly clearable and the predicted effect is far
above the MDE:

> **metric** M1 (paired support-recovery fraction) · **n** = 10,000 · **(d,m,k)** = (128, 4096, 8),
> sigma=0.05 · **bar** = 95% CI lower bound ≥ 0.15 · **MDE80** = 0.00447 · **measured effect** =
> 0.5326 · **headroom** 119x over MDE, 3.5x over the bar · **cost** ~11 min CPU or ~0.3 s GPU.

This is the exact structural inverse of HI-1, which needed a true effect of 0.09–0.15 against a
half-width of 0.08–0.14 at n≈9–27. Here the half-width is 0.0025 at n=10,000 because samples are
generated rather than harvested.

### 6b. The brutally honest part: **passing this hard is a red flag, not a result.**

An experiment whose effect is **119x its minimum detectable effect** is not measuring anything
uncertain. What §4 actually did was re-derive the Donoho–Tanner phase transition on a Gaussian
dictionary. The outcome was fully determined before any compute was spent; I could have written the
result table from the thresholds alone, and the interference heuristic predicted the measured linear
accuracy to within 0.014.

Shipping Tier 0 as "the wedge test" would be the same category error as the previous six
fail-closeds, just inverted: instead of elaborate structure losing to a trivial baseline, it is
elaborate structure **beating a baseline that theory guarantees must lose**. Decision value ≈ 0.
It must be labelled **harness validation**, not a finding, and it is not publishable on its own.

### 6c. Where the decision value actually is — and it is genuinely open

**Tier 1 — dictionary estimated, not given.** I ran the misspecification sweep (`wedge_misspec.py`,
n=1000, operating cell). The linear arm is unaffected by eps (it is trained on true labels), so it
is the fixed reference at `lin_frac = 0.4863`:

| eps | mean column alignment | l1_frac | l1 exact | **Delta(M1)** | sd_d |
|---|---|---|---|---|---|
| 0.00 | 1.0000 | 1.0000 | 1.000 | 0.5138 | 0.1220 |
| 0.05 | 0.9988 | 1.0000 | 1.000 | 0.5138 | 0.1220 |
| 0.10 | 0.9951 | 1.0000 | 1.000 | 0.5138 | 0.1220 |
| 0.20 | 0.9807 | 0.9994 | 0.995 | 0.5131 | 0.1227 |
| 0.40 | 0.9288 | 0.9100 | 0.441 | **0.4238** | 0.1436 |

The wedge is **robust to isotropic dictionary jitter** — it survives to eps=0.4 (column alignment
0.929) with Delta still 0.424, far above the 0.15 bar.

**But I must not overclaim this.** Isotropic Gaussian jitter followed by re-normalisation is the
*easy* perturbation: it preserves atom identity, atom count, and incoherence. It does **not**
simulate the documented failure mode — sparse dictionary learning is provably non-identifiable
(arXiv:2512.05534: zero reconstruction loss while recovering zero ground-truth features, 3/3200
empirically), whose signature is **merged, split, rotated, and permuted atoms**, not jittered ones.
Tier 1 as a real experiment must estimate the dictionary gradient-free and measure the wedge against
*that*. That question is genuinely open and worth the compute.

**Tier 2 — real frozen embedding space** with a programmatically generated label set (Engels-style
circular features: day-of-week, month). n remains free (~100k prompts, ~1 min encode on a 3090).
Effect size is **unknown**, so the bar must be set from the MDE — at n=10,000 that is 0.004–0.014
depending on sd, which is fine-grained enough to detect effects an order of magnitude smaller than
anything this program has chased. **This is the cell with actual decision value.**

### 6d. Preregistered external-validity risk that must be cleared BEFORE Tier 2 GPU

A **Gaussian dictionary is the maximally compressed-sensing-favourable design** (near-optimal RIP).
Real embedding dictionaries are coherent and structured, and coherence directly shrinks the l1
advantage. **Required Tier-0b preflight: a coherence sweep** (correlated/clustered atoms at
increasing mutual coherence). If the advantage dies at realistic embedding coherence, **Tier 2 is
CUT before any GPU is spent** — the same discipline that correctly killed the beta-sweep and HI-1.

### 6e. Recommendation

- **Tier 0:** run as a ~30 min CPU **harness preflight** with the null cell mandatory. Not a result.
- **Tier 0b:** coherence sweep. **Gate.** Cheap, CPU-only, and it can CUT the whole line early.
- **Tier 1:** gradient-free dictionary *estimation*, with non-identifiability-style perturbations
  (merge/split/rotate), not jitter. Genuinely open.
- **Tier 2:** real frozen space, generated-label attribute, bar set from MDE. The actual experiment.
- **Do not** make retrieval NDCG the primary metric anywhere in this line.

---

## 7. Threats to validity, disclosed

1. Gaussian dictionary is best-case for l1 (§6d) — unaddressed until the coherence sweep runs.
2. Exact `k` is given to both arms via the top-k readout. Symmetric, but unrealistic; Tier 2 has no
   known `k`.
3. `sd_d` is measured under a large-effect alternative. Under a small true effect it could differ;
   §3b's `sd = 0.30 / 0.50` rows bound this.
4. Pilot n=1000, single seed (seed=1) per cell. Sufficient for a power analysis (the quantity needed
   is `sd_d`, which is stable), **not** sufficient as a result. The real run needs multiple seeds.
5. The eps sweep reuses one dictionary and one test set, so the eps=0.05/0.10 rows are bit-identical
   to eps=0 (l1 hits 1.000 in all three, and the linear arm is fixed). That is expected, not a bug,
   but it means those rows carry no independent information.
6. FISTA lambda was fixed a priori and never tuned. This handicaps the treatment, so it is
   conservative — but it means the l1 numbers are a lower bound on the treatment arm, not a tuned
   optimum.
