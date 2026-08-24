# Conditional replacement, layered inversion, and subspace reverberation protocol

**Protocol ID:** `CHELATEDAI-CRSV-LIR-SRS-v0.1`

**Status:** `NON_CONFIRMATORY_METHOD_DEV`

**Freeze target:** no freeze exists; this document defines the work required
before a preregistration may be frozen

**Scientific claim status:** unconfirmed

**Novelty claim status:** unconfirmed

**Deployment or promotion status:** prohibited

## 1. Brutally honest finding

The repository contains a plausible missing research lane, not a discovery.
The narrow idea is that replacement compatibility may be local to the semantic
subdomain that produced or validated a replacement, and that apparently safe
single replacements may interact destructively when layered or reordered.
Static subspace alignment may further miss finite-horizon amplification and
loss of task-relevant signal.

Three claims must remain separate:

1. **CRSV-1 — conditional replacement-subdomain variance:** serving subdomain
   and replacement provenance interact after their main effects are removed.
2. **LIR-1 — layered inversion and reversal:** singleton replacement behavior
   is insufficient to predict multi-layer behavior; signed interactions and
   order effects predict held-out chain outcomes.
3. **SRS-1 — subspace reverberation:** frozen geometric and finite-horizon
   diagnostics add held-out predictive information beyond CRSV-1 and LIR-1.

The terms “onion,” “attunement,” and “reverberation” are mnemonic labels. They
are not evidence and must not appear as causal conclusions. “Harmonic” is
disallowed unless Section 7's mode/frequency requirements are met.

The current Python implementation and synthetic tests validate calculations
and failure behavior only. They do not validate CRSV-1, LIR-1, SRS-1, a
certificate, production safety, or a claim about the history of computing.

## 2. Frozen objects and notation

For independent group \(g\), serving semantic subdomain \(d\), replacement
provenance subdomain \(r\), replacement subset \(S\), and admissible order
\(\pi\):

- \(Y_{gdrS\pi}\) is the corrected retrieval outcome. The confirmatory default
  is binary nDCG@10 under the repository's repaired metric-lineage contract.
- \(R^*_{gd}\) is the frozen native-target ranking.
- \(R_{gdrS\pi}\) is the candidate ranking with score ties resolved by
  canonical item ID ascending.
- \(I_{gdrS\pi}\in[0,1]\) is normalized weighted Kendall inversion loss
  relative to \(R^*_{gd}\).
- \(B_{gdrS\pi}=I_{gdr\varnothing}-I_{gdrS\pi}\) is inversion benefit; positive
  is better.
- A **replacement provenance subdomain** is where the replacement was fit or
  validated. It is not the representation ABI's `compatibility_domain_id`.
- An **independence group** is the smallest unit that may be resampled or
  permuted independently. Repeated views of one query, document, subject,
  session, or source remain in one group.

The weighted inversion loss is

\[
I(R^*,R)=
\frac{\sum_{i<j} w_{ij}
\mathbf{1}\!\left[
(r_i^*-r_j^*)(r_i-r_j)<0
\right]}
{\sum_{i<j}w_{ij}},
\qquad w_{ij}\ge0.
\]

The item universe, weights, tie rule, native ranking identity, and metric
lineage must be frozen before any REPORT result is opened.

## 3. CRSV-1: exact hypothesis

Let

\[
\mu_{dr}=\mathbb{E}_g[Y_{gdr}],
\quad
\gamma_{dr}=
\mu_{dr}-\mu_{d\cdot}-\mu_{\cdot r}+\mu_{\cdot\cdot}.
\]

The normalized general interaction energy is

\[
E_\gamma=\frac{\|\gamma\|_F^2}{|D||R|}.
\]

In samples, it must not be estimated only by squaring the sample interaction
mean because that is positively biased under the null. The METHOD_DEV
implementation instead double-centers each group and uses a cross-group
U-statistic for this cell-mean energy. That estimator may be negative in finite
samples and is descriptive rather than a significance test.

When every serving label has exactly one predeclared corresponding
replacement-provenance label, define the home-correspondence effect

\[
H_Y =
\frac{1}{|D|}
\sum_{d\in D}
\left(
\mu_{dd}
-
\frac{1}{|D|-1}\sum_{r\ne d}\mu_{dr}
\right).
\]

The inversion analogue \(H_B\) substitutes inversion benefit \(B\) for
retrieval outcome \(Y\).

### CRSV-1 null and alternative

\[
H_{0,\mathrm{CRSV}}: H_Y\le0
\]

\[
H_{1,\mathrm{CRSV}}:
H_Y>0,\quad
\widehat H_Y\ge0.005\ \text{absolute binary nDCG@10},
\quad
H_B>0.
\]

CRSV-1 passes only if all of the following hold on untouched REPORT:

1. the one-sided 97.5% independence-group and subdomain-block bootstrap lower
   bound for \(H_Y\) is above zero;
2. the point estimate is at least 0.005 binary nDCG@10;
3. the inversion-benefit estimate has the same favorable sign;
4. the one-sided simultaneous 97.5% lower confidence bound for
   \(\min_d H_{Y,d}\) is above the preregistered noninferiority margin of
   \(-0.02\) binary nDCG@10; the minimum is recomputed inside every nested
   block-bootstrap replicate rather than taking the minimum of separate
   marginal intervals;
5. no ABI, metric-lineage, exchangeability, or independence-group gate fails.

### Conditional permutation boundary

A label-permutation test is permitted only when the experiment design, before
outcomes, establishes exchangeability within explicit blocks. Naturally
associated, separately trained, or differently scoped replacement identities
are not exchangeable by default. Labels move; outcome profiles do not.

The resulting p-value is finite-label randomization inference about the
observed replacement identities. It does not establish generalization to
unseen identities or subdomains. Exact enumeration is used when the blocked
factorial universe is no larger than 10,000. Otherwise, 10,000 seeded random
permutations use

\[
p=(b+1)/(B+1).
\]

An exact one-sided \(\alpha=0.025\) test requires at least 40 admissible label
assignments. Four globally exchangeable labels have only \(4!=24\), so even a
unique most-extreme observed assignment has \(p_{\min}=1/24=0.0417\). Two
three-label blocks have \(3!\times3!=36\) and are also insufficient. Blocks
must never be weakened to manufacture resolution; when the design-justified
universe is smaller than 40, permutation evidence is descriptive only.

If exchangeability cannot be justified, the permutation result is not
computed. The hierarchical held-out estimate remains the primary inference.

## 4. LIR-1: onion differentials without naive compounding

For ordered prefixes \(\pi_{1:k}\), define

\[
\delta_k =
I(\pi_{1:k-1})-I(\pi_{1:k}),
\]

so \(\delta_k>0\) is a favorable shell and \(\delta_k<0\) is a reversal.
Endpoint benefit telescopes:

\[
\sum_k\delta_k=I(\varnothing)-I(\pi).
\]

That identity is bookkeeping, not evidence of a layered mechanism. The
layer-specific quantities are:

- reversal mass
  \(\sum_k[-\delta_k]_+\);
- weighted reversal burden
  \(\sum_k q_k[-\omega_k]_+\), where
  \(\omega_k=\log(I_{k-1}+\epsilon)-\log(I_k+\epsilon)\);
- order spread, with tested/admissible order coverage reported;
- commutator distance for operators already proven to share one coordinate
  space;
- set interactions after order is separated from membership.

For an order-averaged set benefit

\[
\bar B(S)=
\frac{1}{|\Pi(S)|}
\sum_{\pi\in\Pi(S)}B(S,\pi),
\]

the Möbius coefficient is

\[
M(S)=
\sum_{A\subseteq S}
(-1)^{|S|-|A|}\bar B(A).
\]

For a pair,

\[
G_{ij}=
\bar B(\{i,j\})-\bar B(\{i\})-\bar B(\{j\})+\bar B(\varnothing).
\]

\(G_{ij}>0\) is superadditive, \(G_{ij}<0\) is subadditive/redundant, and a
near-zero value is additive only as a deterministic description. Scientific
classification additionally requires uncertainty and a practical-effect
threshold.

### LIR-1 null and alternative

Fit two frozen predictors using SELECT only:

- \(M_0\): serving and replacement main effects plus singleton benefits;
- \(M_1\): \(M_0\) plus pair Möbius terms, reversal summaries, admissible-order
  coverage, and normalized commutator terms.

On unseen three- and four-layer REPORT chains:

\[
H_{0,\mathrm{LIR}}:
\operatorname{MAE}(M_1)\ge\operatorname{MAE}(M_0),
\]

\[
H_{1,\mathrm{LIR}}:
\operatorname{MAE}(M_1)\le0.90\operatorname{MAE}(M_0).
\]

LIR-1 passes only if the relative MAE reduction is at least 10% and the
one-sided 97.5% block-bootstrap lower bound for
\(\operatorname{MAE}(M_0)-\operatorname{MAE}(M_1)\) is above zero. A separate
secondary endpoint tests whether \(M_1\) predicts any negative shell
\(\delta_k<0\) with REPORT AUROC at least 0.70.

The earlier H4 compound-cycle lane is not reusable evidence. Its susceptible
metrics remain `LEGACY_METRIC_LINEAGE_BLOCKED`.

## 5. SRS-1: subspace additives, perturbation, and atrophy

SRS-1 does not assume that static alignment is sufficient. It defines a frozen
feature family:

1. **Principal-angle attunement.** For orthonormal subspace bases \(U_i,U_j\),
   retain every singular value of \(U_i^\top U_j\), not only the maximum.
2. **Signed additive interference.** For residual additives
   \(\Delta_k\) in one declared coordinate space,

   \[
   \Xi =
   \left\|\sum_k\alpha_k\Delta_k\right\|_F^2
   -
   \sum_k\|\alpha_k\Delta_k\|_F^2.
   \]

   Negative \(\Xi\) measures destructive interference. Triangle-inequality
   slack is reported separately because orthogonal, noninteracting additives
   also produce slack.
3. **Noncommutativity.**

   \[
   K_{ij}=
   \frac{\|\Delta_i\Delta_j-\Delta_j\Delta_i\|_2}
   {\|\Delta_i\|_2\|\Delta_j\|_2}.
   \]

4. **Finite-horizon amplification.** For one explicitly declared
   discrete-time transition \(T\),

   \[
   G_H(T)=\max_{0\le h\le H}\|T^h\|_2.
   \]

   Horizon zero has gain one. Spectral radius below one does not exclude
   non-normal transient amplification.
5. **Exact ordered prefixes.** Sequential replacements use
   \(T_k\cdots T_2T_1\), not powers of their sum.
6. **Useful-signal atrophy.** For a SELECT-frozen useful-signal basis \(U_s\),

   \[
   A_s =
   1-
   \frac{\|X_{\mathrm{candidate}}U_s\|_F^2}
   {\|X_{\mathrm{native}}U_s\|_F^2}.
   \]

   The ratio fails closed below a preregistered native-energy identification
   floor. Gains are preserved as signed change and are not hidden by the
   atrophy truncation.

For residual additives, the transition is explicitly

\[
T(\alpha)=T_0+\sum_k\alpha_k\Delta_k.
\]

The sum of residuals must never silently stand in for \(T\). Spectral
diagnostics are unavailable for non-square maps or maps without a proven
common coordinate contract.

### SRS-1 null and alternative

Fit \(M_2\) by adding the frozen SRS feature family to \(M_1\), without
reselecting features after REPORT is opened.

\[
H_{0,\mathrm{SRS}}:
\operatorname{MAE}(M_2)\ge\operatorname{MAE}(M_1),
\]

\[
H_{1,\mathrm{SRS}}:
\operatorname{MAE}(M_2)\le0.95\operatorname{MAE}(M_1).
\]

SRS-1 passes only if REPORT relative MAE falls by at least 5% and the one-sided
97.5% block-bootstrap lower bound for
\(\operatorname{MAE}(M_1)-\operatorname{MAE}(M_2)\) is above zero. This is an
incremental prediction claim. Static coherence, a large commutator, or a
transient peak alone cannot pass SRS-1.

## 6. Meta-scale and perturbation sweep

METHOD_DEV may evaluate the frozen scale grid

\[
\lambda\in
\{0,\ 0.25,\ 0.50,\ 0.75,\ 1.00,\ 1.25,\ 1.50\}
\]

for

\[
T(\lambda)=T_0+\lambda\sum_k\alpha_k\Delta_k.
\]

At every nonzero scale, record spectral radius, \(G_H\), peak horizon, signed
interference, leave-one-additive perturbations, and useful-signal change.
Interior peaks are called **scale-local amplification**. They are not called
resonance until they reproduce on held-out groups and predict later inversion
or retrieval loss.

\(\lambda=0\) is a base-only control and must be special-cased by the future
scale-sweep runner. At zero, compute spectral and useful-signal quantities from
\(T_0\); raw signed interference and raw leave-one-additive changes are zero by
algebra. Normalized interference, component subspaces, triangle ratios, and
normalized commutators are `UNIDENTIFIED_AT_ZERO_SCALE`. The runner must not
call `additive_operator_diagnostics` with zero effective weights or substitute
invented zeros for those unidentified quantities.

Leave-one-additive perturbation is

\[
P_j =
G_H(T-\alpha_j\Delta_j)-G_H(T).
\]

The complete scale curve must be retained. Selecting only its largest peak is
prohibited.

## 7. Harmonic claim gate

“Harmonic” becomes admissible only if all of the following are frozen before
REPORT:

1. a repeated-step, approximately linear time-invariant interpretation of
   \(T\);
2. a sampling/step scale with physical or operational meaning;
3. a mode basis fixed without REPORT outcomes;
4. complex eigenmodes or a frequency-response basis with reproducible phase or
   frequency;
5. a mode-specific prediction of held-out amplification and useful-signal
   loss;
6. correction for every tested mode and frequency.

If \(T\) has eigenvalue \(\rho e^{i\theta}\), the candidate angular frequency
is \(\theta\) per declared step. This algebra is not evidence that the
replacement chain is an oscillator. Without the six gates, use “transient
mode” or “scale-local amplification.”

## 8. Exact staged experiment

### Phase A — METHOD_DEV calibration

- 4 serving semantic subdomains;
- 4 matched replacement-provenance subdomains;
- 4 replacement layers: encoder, projection/adapter,
  normalization/quantization, and compatible scoring/index layer;
- 3 admissible alternatives per layer;
- 3 mutation families;
- 3 independent training seeds;
- at least 50 independence groups per balanced cell;
- all valid singletons;
- all valid pairs in both orders;
- selected balanced triples and four-layer chains;
- deliberately wrong-domain replacements and identity/native controls;
- corrected metrics only.

Phase A may choose features, the useful-signal basis construction, admissible
orders, identification floors, and practical-effect thresholds. Every Phase A
group and every feature invented from it remains permanently METHOD_DEV.

### Phase B — frozen REPORT campaign

- 6 serving subdomains total: 4 SELECT and 2 untouched REPORT;
- 6 matched provenance subdomains with identities hidden until evaluation;
- 6 mutation families total: 3 SELECT and 3 untouched REPORT;
- the same 4 layers and 3 alternatives per layer;
- 3 independent seeds;
- at least 200 independence groups per required REPORT cell;
- all singletons and all valid pairs in both orders;
- a preregistered balanced incomplete block of triples and four-layer chains;
- at least 25% of REPORT chains containing a deliberately wrong-domain
  replacement;
- a prospective power simulation using SELECT dispersion for both the average
  CRSV effect and the simultaneous minimum-subdomain safety bound; the
  200-group floor does not replace that analysis.

Holdouts are simultaneous:

1. unseen serving subdomains;
2. unseen mutation families;
3. unseen high-order replacement combinations;
4. unseen training seeds for a replication slice.

Each chain must bind exact component revisions, weights, tokenizer,
query/document role, metric, dimension, normalization, dtype, compatibility
domain, corpus snapshot, qrels snapshot, ranking tie rule, and admissible order.
Invalid contract chains are not pooled as poor outcomes; they are protocol
violations.

## 9. Analysis and multiplicity

The hierarchy is fixed:

1. test CRSV-1 at one-sided \(\alpha=0.025\);
2. only if CRSV-1 passes, test LIR-1 at one-sided \(\alpha=0.025\);
3. only if LIR-1 passes, test SRS-1 at one-sided \(\alpha=0.025\).

This gatekeeping prevents a geometry story from surviving after the
conditional-locality or compositional-prediction premise fails. Secondary
endpoints are descriptive unless separately corrected.

Resampling is nested:

- serving subdomain;
- mutation family;
- training seed;
- independence group.

All repeated chain views of one group remain coupled. Empty or singleton
resampling strata, incomplete crossed cells, invalid metric lineage, and
unfrozen feature changes fail closed.

The worst-subdomain safety endpoint uses the bootstrap distribution of
\(\min_d H_{Y,d}\) directly. Separate per-domain marginal intervals are not
substituted for this simultaneous bound.

## 10. Prospective certificate — not implemented

A future compatibility certificate may be issued only after independent
replication. At minimum it must carry:

- exact input/output representation contracts for every prefix;
- ordered component IDs and content hashes;
- evidence payload hashes recomputed from immutable bytes;
- authenticated issuer and accepted evidence schema;
- issued-at, evaluated-at, expiry, and revocation state;
- serving/provenance scope and excluded subdomains;
- tested/admissible order coverage;
- CRSV, LIR, and SRS estimates with uncertainty;
- worst-domain and wrong-domain bounds;
- fail-closed disposition.

`validate_declared_replacement_path` currently checks only caller-declared
path shape, contract chaining, SHA-shaped metadata, state text, and expiry. It
does not fetch bytes, authenticate an issuer, or consult a revocation registry.
It is not a certificate validator.

The eventual architecture is analogous in spirit to proof-carrying systems:
the consumer verifies a scoped evidence object instead of trusting a bare
compatibility claim. This analogy is not a novelty claim.

## 11. Current implementation and validation

`crsv_experiment.py` implements:

- deterministic ranking and weighted Kendall inversion loss;
- signed onion differentials, endpoint contraction, and reversal burden;
- explicit-baseline Möbius and additivity calculations;
- order spread with coverage and commutator distance;
- principal-angle subspace diagnostics;
- declared-transition finite-horizon diagnostics including horizon zero;
- explicit `base + additives` interference diagnostics;
- exact ordered-prefix products;
- useful-signal atrophy with an identification floor;
- group-double-centered, cross-group interaction energy;
- exact or seeded blocked label permutation with mandatory exchangeability;
- recursive non-finite-output rejection.

`tests/test_crsv_experiment.py` currently has 31 adversarial tests. They cover
known inversion values, reversal preservation, subadditive and superadditive
interactions, noncommuting order effects, aligned/orthogonal subspaces,
asymptotically stable non-normal amplification, coherent/orthogonal/anti-aligned
additives, ordered products, false-atrophy prevention, overflow failure,
noise-unbiased interaction structure, and blocked permutation behavior.

These tests establish implementation behavior on constructed examples only.

The combined deterministic calibration executed on 2026-07-23 produced:

| Constructed diagnostic | Result |
|---|---:|
| Four-subdomain unbiased CRSV interaction energy | 0.1875 |
| Perfect four-label home effect | 1.0 |
| Exact four-label permutation p-value | 0.041666666667 |
| Onion endpoint inversion benefit | 0.3 |
| Onion reversal mass | 0.1 |
| Stable non-normal spectral radius | 0.8 |
| Stable non-normal finite-horizon peak gain | 8.212429054411 |
| Ordered prefix gains | 1.0, 2.0, 0.5 |
| Orthogonal additive triangle slack | 0.292893218813 |
| Orthogonal net destructive interference | 0.0 within floating tolerance |

The calibration intentionally contains engineered signals. It verifies that
the diagnostics distinguish endpoint benefit from reversals, asymptotic
stability from transient growth, ordered products from additive composition,
and orthogonality from destructive cancellation. It is not empirical support
for any hypothesis.

## 12. Prior-art boundary as of 2026-07-23

The ingredients are established:

- broader adjacent conditional-independence permutation work, which estimates
  \(X\mid Z\) and is not the blocked finite-label randomization implemented
  here:
  [Berrett et al.](https://arxiv.org/abs/1807.05405);
- exact testing under restricted/random permutation rules:
  [Mehta, Patel, and Wei](https://doi.org/10.1093/biomet/75.2.295) and
  [Hemerik and Goeman](https://link.springer.com/article/10.1007/s11749-017-0571-1);
- set interactions and Möbius transforms:
  [Grabisch and Roubens](https://doi.org/10.1016/S0020-0255(99)00099-7);
- learning and combining preferences:
  [RankBoost](https://www.jmlr.org/papers/v4/freund03a.html);
- representation similarity:
  [CKA](https://proceedings.mlr.press/v97/kornblith19a.html);
- backward/forward compatible representation learning:
  [BCT](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html),
  [FCT](https://openaccess.thecvf.com/content/CVPR2022/html/Ramanujan_Forward_Compatible_Training_for_Large-Scale_Embedding_Retrieval_Systems_CVPR_2022_paper.html),
  [BC-Aligner](https://arxiv.org/abs/2206.03040), and
  [FastFill](https://arxiv.org/abs/2303.04766);
- non-normal transient growth:
  [Trefethen](https://doi.org/10.1137/S0036144595295284), including a
  new 2026 inference treatment by
  [Saiprasad, Troude, and Sornette](https://arxiv.org/abs/2607.14786);
- task/model composition interference and sparse or orthogonal subspaces:
  [Localize-and-Stitch](https://openreview.net/forum?id=9CWU8Oi86d),
  [OSRM](https://aclanthology.org/2025.acl-long.1284/), and
  [task-driven LoRA subspace decomposition](https://arxiv.org/abs/2603.00191);
- evidence that functional model stitching can overstate informational
  similarity:
  [Smith, Mannering, and Marcu](https://proceedings.mlr.press/v267/smith25a.html);
- proof-carrying code:
  [Necula](https://doi.org/10.1145/263699.263712).

No single ingredient above is novel. A bounded search did not locate the exact
combination of:

1. typed replacement-chain contracts;
2. serving-subdomain by replacement-provenance interaction;
3. onion/set/order decomposition;
4. finite-horizon subspace diagnostics;
5. a scoped, expiring evidence object that predicts unseen valid chains.

That absence is not proof of novelty and is not a patent clearance. The only
potentially defensible future claim is the integrated conditional-replacement
certification architecture, and only if CRSV-1, LIR-1, and SRS-1 pass
prospectively and independently.

## 13. Reconditioning decisions

- If CRSV-1 fails, stop. There is no replacement-locality premise.
- If CRSV-1 passes and LIR-1 fails, report local compatibility but cut the
  compositional predictor and certificate claim.
- If LIR-1 passes and SRS-1 fails, keep interaction/order terms and cut
  reverberation, resonance, and harmonic language.
- If SRS-1 passes once, replicate on new subdomains and replacement families;
  do not claim history-changing novelty.
- Only after independent replication, broader patent/literature review, and a
  real authenticated evidence registry should a white paper or certificate
  implementation be considered.
