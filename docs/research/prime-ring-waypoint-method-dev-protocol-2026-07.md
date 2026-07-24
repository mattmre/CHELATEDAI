# Prime-ring waypoint routing protocol

**Protocol ID:** `CHELATEDAI-PRW-v0.1`

**Status:** `NON_CONFIRMATORY_METHOD_DEV`

**Scientific claim status:** unconfirmed

**Novelty claim status:** standalone phase-address novelty refuted; integrated
systems interaction unresolved

**Production integration:** prohibited

## 1. Distilled theory

The testable idea is not that a large prime dimension creates intelligence, or
that a rotating ring is itself a memory. It is a typed two-part address:

1. a cyclic synchronization carrier identifies a noisy subspace/type key and
   one shared phase;
2. a separate semantic payload identifies a waypoint inside the unlocked
   subspace.

Eight carrier layers are scored under one shared rotation. Their relative
phase offsets and a constrained polarity codebook form the typed key. A query
may rotate the complete stack without changing the selected type. Independent
rotation of every layer is a degeneracy control because it destroys the
cross-layer constraint.

This is the narrow computational translation of “onion rings that rotate to
unlock aligned memory.” It is a robust routing hypothesis, not a claim about
gravity, resonance, consciousness, or the history of computing.

### Exact prior-art identification

The phase-signature address is exactly representable as a
one-pulse-per-wavelength two-dimensional optical orthogonal code. Define

\[
B_\phi(l,j)=\mathbf 1[j=\phi_l].
\]

Then adding one global phase rotates every row by the same cyclic offset, and

\[
\langle B_\phi,R_aB_\psi\rangle
=|\{l:\phi_l+a=\psi_l\}|.
\]

The right-hand side is precisely the quotient overlap `kappa` used below.
Therefore the quotient, gauge fixing, shared-shift orbit, and overlap metric
are known OPPW 2D OOC structure. The Legendre carrier replaces each sparse
pulse with a dense bipolar synchronization sequence; because matched shifts
contribute `p` and mismatches contribute `-1`, its noiseless score is an affine
transformation of the same overlap count.

This exact isomorphism refutes a standalone new-address-mathematics claim. It
does not pre-answer whether the dense realization has a useful error, hardware,
or systems interaction tradeoff.

## 2. Non-negotiable separations

- The additive rotation group is `Z_p`. It controls cyclic alignment.
- The multiplicative group is `F_p^x`, of order `p - 1`. For `p = 4691`,
  `p - 1 = 2 * 5 * 7 * 67`.
- The factorization of `p - 1` is relevant only to a real Rader/CRT
  implementation. A generic FFT benchmark cannot validate it.
- The `Z_2` factor of `F_4691^x` does not manufacture eight bits. Eight
  polarity choices exist only because the architecture declares eight layers.
- A single shared Legendre ring, maximized over every rotation, cannot identify
  waypoints: all of its rotations are equivalent.
- Carrier state and semantic payload are separate. Carrier-only retrieval is a
  required degeneracy control.
- Queue occupancy is operational metadata, not “gravity.”
- Cyclic carrier state must never be inserted as a cycle in `evidence_dag.py`;
  the evidence DAG remains acyclic.

## 3. Mathematical object

Let `p` be an odd prime with `p mod 4 = 3`. Define the bipolar Legendre
carrier `c_p` by

\[
c_p(0)=1,\qquad
c_p(x)=
\begin{cases}
+1,&x\ne0\text{ is a quadratic residue mod }p,\\
-1,&\text{otherwise.}
\end{cases}
\]

For cyclic rotation `R_s`,

\[
\langle c_p,R_s c_p\rangle =
\begin{cases}
p,&s=0,\\
-1,&s\ne0.
\end{cases}
\]

This identity validates a synchronization code. It does not establish a
retrieval, cost, or learning benefit.

Use `L = 8` layers. Each subspace/type `t` has:

- a relative phase signature
  \(\phi_t=(\phi_{t1},\ldots,\phi_{tL})\in Z_p^L\).

Each observation also has one admissible orientation word
\(m=(m_1,\ldots,m_L)\in\mathcal M\subseteq\{-1,+1\}^L\). Orientation is a
constrained nuisance/pivot state; it is not semantic waypoint identity.
The family used to draw the planted observation mask is
`planted_mask_family`; the family searched by the decoder is
`decoder_mask_policy`. They are independent experimental factors. Candidate
templates are stored with canonical all-positive orientation, so a mask cannot
silently become an additional type code.

The phase signature is gauge-fixed with \(\phi_{t1}=0\). Signatures that differ
only by adding one constant to every coordinate are the same object:

\[
[\phi_t]=\{\phi_t+a\mathbf 1:a\in Z_p\}.
\]

For a query from type `t*`, one shared unknown rotation `tau`, and independent
bit-flip noise `e_l(i)`, the observed carrier layers are

\[
x_l(i)=e_l(i)m_l
c_p\!\left(i-\tau-\phi_{t^*l}\right).
\]

The joint carrier score is

\[
A_t(s,m)=\frac{1}{Lp}
\sum_{l=1}^{L}
\left\langle
x_l,\,
m_lR_{s+\phi_{tl}}c_p
\right\rangle .
\]

The carrier route is

\[
(\hat t,\hat\tau,\hat m)=\arg\max_{t,s,m\in\mathcal M} A_t(s,m).
\]

It unlocks only when both the winning score and the top-one/top-two type margin
exceed thresholds frozen on `SELECT`.

In the first simulator, each waypoint `w` has `L` normalized, ring-indexed
semantic payload planes \(u_{wl}\in\mathbb R^p\) and belongs to one type
`t(w)`. The payload values are separate from the bipolar carrier and are stored
in type-canonical coordinates. They undergo only the shared global transport,
not the carrier's type-relative phase signature. Applying the type-relative
signature to payloads would let a global payload-only scan recover the type and
would invalidate the claimed key/payload separation. After the carrier freezes
one shared shift, define

\[
B_w(\hat\tau)=
\frac{1}{L}\sum_{l=1}^L
\frac{\langle q_l,R_{\hat\tau}u_{wl}\rangle}
{\|q_l\|_2\|u_{wl}\|_2}.
\]

The waypoint is

\[
\hat w=\arg\max_{w:t(w)=\hat t}B_w(\hat\tau).
\]

The carrier may select a candidate subspace; it does not contain the waypoint
payload. A later implementation may replace the ring-indexed payload planes
with a separately declared semantic projection, but it may not use payload
values to estimate the carrier shift or encode the carrier's relative-phase
type key.

The first simulator stores each distinct collision-group payload tensor once as
`float64`. Actual serialized storage must therefore charge `8 * L * p` bytes
per distinct stored collision group, plus any explicit type-to-group mapping
bytes. A type-expanded `T * W * L * p * 8` count may be reported separately as
`logical_type_expanded_bytes`, but it is not an actual-storage denominator when
the identical tensors are deduplicated. Conversely, an implementation that
materializes type-specific copies must charge every materialized copy. A
hypothetical one-bit payload size may be reported only as
`theoretical_packed_bytes`; it is not an observed storage result. Every
campaign reports separately:

- logical carrier bits;
- deduplicated serialized carrier, phase-codebook, mask-codebook, payload, and
  mapping bytes;
- logical type-expanded bytes, clearly excluded from actual-cost claims unless
  those copies are materialized;
- actual NumPy array `nbytes` for stored and query arrays;
- batched decoder working-array bytes; and
- process peak resident memory when that measurement is available.

The primary efficiency denominator is total serialized method bytes, including
the payload bank and decoder auxiliaries. Carrier-only bytes are descriptive.

### Required equivariance

If every observed layer is rotated by the same `a`, type selection must remain
unchanged and the recovered phase must change by exactly `a mod p`.

### Required degeneracy

If every candidate is allowed an independent best shift for every layer, a
shared pure carrier loses the relative-phase constraint. Carrier-only waypoint
identity must then remain unidentifiable.

### Noiseless quotient-overlap bound

For planted type `t*`, define the favorable overlap of wrong type `t` as

\[
\kappa(t,t^*)=
\max_{a\in Z_p}
\left|
\{l:\phi_{tl}+a=\phi_{t^*l}\}
\right|.
\]

Because every nonmatching Legendre rotation has correlation magnitude one, a
wrong type has the conservative noiseless bound

\[
\max_{s,m} A_t(s,m)
\le
\frac{\kappa p+(L-\kappa)}{Lp}.
\]

The correct type scores one, so its guaranteed margin is at least

\[
\frac{(L-\kappa)(p-1)}{Lp}.
\]

For `L = 8`, `p = 4691`, and signatures with at most one favorable aligned
layer under any shared rotation, the wrong score is at most about `0.12519`
and the noiseless margin is at least about `0.87481`. This is an analytic
construction property. Noise robustness, efficient codebook construction, and
semantic utility remain experimental questions.

### Exact sparse-code isomorphism

For a phase signature \(\phi\), define the one-pulse-per-wavelength array

\[
B_\phi(l,j)=\mathbf 1[j=\phi_l].
\]

A shared phase offset is exactly a common cyclic column shift, and

\[
\langle B_\phi,R_aB_\psi\rangle
=
\left|\{l:\phi_l+a=\psi_l\}\right|.
\]

For all-positive orientation in the noiseless Legendre construction, if
\(k_a=\langle B_\phi,R_aB_\psi\rangle\), then the dense normalized score is
exactly

\[
A_{\mathrm{dense}}(a)
=
\frac{(p+1)k_a-L}{Lp}.
\]

Thus the dense Legendre decoder and sparse OPPW decoder have identical
noiseless rankings; the dense score is only an affine transform of sparse
collision count. The dense representation can survive only by showing a
noise, resource, or systems advantage under matched conditions. It cannot be
claimed as a new phase-address geometry.

### Exact binary-code decoding reduction

Let \(\mathcal H\) be the finite bank of flattened clean bipolar carrier
hypotheses \(h_{t,s,m}\in\{-1,+1\}^{Lp}\). Under the simulator's independent
bit-flip model,

\[
y=e\odot h_*,\qquad
\Pr(e_i=-1)=q<\tfrac12.
\]

Because every hypothesis has the same norm, maximizing correlation is exactly
maximum-likelihood nearest-codeword decoding for a binary symmetric channel.
For a fixed competitor \(h\), let

\[
d=d_H(h_*,h).
\]

Then

\[
\langle y,h_*\rangle-\langle y,h\rangle
=2\sum_{i:h_{*,i}\ne h_i}e_i,
\]

so the exact probability that the competitor ties or beats the truth is

\[
P_{\mathrm{pair}}(d,q)
=
\sum_{j=\lceil d/2\rceil}^{d}
{d\choose j}q^j(1-q)^{d-j}.
\]

For all-positive or known matching orientation and \(k\) phase-aligned layers,
the Legendre two-level autocorrelation gives

\[
d=\frac{(L-k)(p+1)}{2}.
\]

More generally, let \(J\) be the aligned layers, \(r=|J|\), let
\(b=m\odot m_*\), let \(u\) count negative entries of \(b\) inside \(J\), and
let \(v\) count negative entries outside \(J\). Then the exact distance is

\[
d=
up+(L-r-v)\frac{p+1}{2}+v\frac{p-1}{2}.
\]

For `L=8`, overlap at most one, and `p=4691`, the wrong-type lower bounds are:

- `none`: `16422`;
- `typed16`: at least `16418`;
- `free256`: at least `16415`.

Consequently,

\[
\Pr(\text{any wrong hypothesis ties or wins})
\le
\sum_{h\ne h_*}P_{\mathrm{pair}}(d_H(h_*,h),q)
\le
(|\mathcal H|-1)P_{\mathrm{pair}}(d_{\min},q).
\]

This is a code-distance/union-bound result, not evidence of a new memory
physics. The campaign must enumerate the actual structured bank distance
spectrum, including masks, and compare the analytic bound with simulation.
A useful theorem contribution would require a sharper structured spectrum or
joint-decoder result than standard binary-code/OOC bounds, not merely this
reduction.

At `p=4691`, `L=8`, overlap at most one, `T=16`, `typed16`, and bit-flip
rate `0.45`, the conservative wrong-type union/Chernoff bound is approximately
`1.66e-30`. This number is a model implication, not a measured error rate.
It makes a zero-error finite simulation an implementation falsifier only; it
cannot empirically validate a probability near `1e-30`.

## 4. Polarity policies

The exact policies are:

- `none`: one all-positive word;
- `shared`: the all-positive and all-negative words;
- `typed16`: eight Walsh/Hadamard rows and their negatives, giving 16
  predeclared words with controlled separation;
- `free256`: every eight-bit word, retained only as an adversarial control.

For layer correlations `a_l(s)`, free polarity maximization satisfies

\[
\max_{m\in\{-1,+1\}^8}\sum_l m_l a_l(s)
=\sum_l|a_l(s)|.
\]

It therefore creates a multiple-search advantage and can turn unrelated input
into an apparent unlock. It is not a default model.

## 5. Formal hypothesis registry

These identifiers are canonical for this lane. Bare `H3`-style identifiers are
forbidden because the repository already assigns conflicting meanings to
`H3`.

### PRW-0 — construction validity

All primality, Legendre autocorrelation, global-shift equivariance,
gauge-canonicalization, typed-mask, explicit-encoding byte-accounting, and CRT
round-trip invariants hold exactly.

Failure disposition: implementation invalid; no experiment is interpreted.

### PRW-1 — constrained phase synchronization

At a threshold calibrated without `REPORT` outcomes to a false-unlock rate no
greater than 1%, the joint shared-shift carrier has higher planted-type recovery
than:

1. independent per-layer phase maximization;
2. a matched-length random bipolar carrier;
3. an unrelated random-input control.

The primary estimand is paired true-unlock recall at the frozen false-unlock
constraint. Phase error is circular distance in `Z_p` for decoders that return
one shared shift. Phase recovery is `N/A`, not zero, for the independent-layer
control unless a separate phase estimator is preregistered.

The earlier 95%/90% calibration targets are withdrawn as too weak for this
idealized channel. For `L=8`, `p=4691`, codebook overlap at most one, a
`typed16` decoder, and bit-flip rate `0.45`, the code-distance union bound
predicts essentially perfect raw type recovery. The construction sanity gate
is therefore 100% raw type recovery through rate `0.45` in the frozen finite
campaign; any observed miss triggers implementation/noise-model investigation
rather than being presented as partial support. Rate `0.49`, unrelated inputs,
matched random carriers, and matched-resource controls are the informative
stress conditions. These are still synthetic checks, not confirmatory utility.

#### RB-1 pre-run control reconditioning (2026-07-24)

Before any expanded campaign, the control gate is frozen to require true-unlock
recall of at least `0.50` in addition to a frozen SELECT threshold and the
REPORT false-unlock confidence gate. This is a nontriviality floor, not a
support threshold: an always-locked decoder must not count as a closed control.
The dense primary route, native sparse OPPW route, and equal-channel-use
repeated-bit route must each clear it.

The `iid` carrier and repeated-bit paths use independent Bernoulli chip flips,
so the binary-symmetric-channel calculation remains applicable. The block and
burst stress paths retain exact global flip counts but are explicitly outside
that binomial model. Native OPPW uses symbol substitution and therefore cannot
enter a matched-noise superiority claim against the dense chip channel.

Control execution is refused when the conservative standalone or co-resident
NumPy-array estimate exceeds `512 MiB`, or when the estimated control work
exceeds `50,000,000` units. These estimates are not measured process RSS and do
not authorize the expanded campaign.

### PRW-2 — payload-preserving waypoint unlock

On payload collision groups spanning multiple types, phase routing plus
within-type payload retrieval must close at least 90% of the gap between:

- global payload-only retrieval; and
- direct known-type payload retrieval.

Carrier-only waypoint retrieval must fail whenever multiple waypoints share a
type. Direct known-type lookup is the cost oracle and must be reported. If
reliable type metadata already exists, a carrier has no established advantage.

### PRW-3 — constrained masks versus free search

On the same observations planted from `typed16`, and at matched true-unlock
recall, a `typed16` decoder must have a lower false-unlock rate than a
`free256` decoder. The exact free-mask factorization is also checked directly.
Masks planted from `free256` are a separate out-of-distribution stress
condition and cannot substitute for this primary comparison.

Failure disposition: cut cross-layer polarity from the proposed mechanism.

### PRW-4 — 4691-specific utility

The null is that `4691` has no independent utility advantage after carrier
family, bit budget, false-unlock threshold, and timing are controlled.

Lengths `4091`, `4096`, and `4691` are mandatory. `4091` is a prime,
`3 mod 4` Legendre control; `4096` is the hardware-friendly composite-length
control. Random bipolar carriers are tested at every length.

Define descriptive efficiency

\[
E=\frac{\mathrm{true\ unlock\ recall}-\mathrm{false\ unlock\ rate}}
{\mathrm{total\ serialized\ method\ bytes}\times\mathrm{mean\ latency}}.
\]

`PRW-4` survives METHOD_DEV only if `4691` improves median `E` by at least 5%
over the best matched control across all three frozen seeds without worsening
waypoint recall. A raw accuracy gain bought by 14.7% more carrier bits than
`4091` is not a pass.

### RADER-1 — separate performance hypothesis

An actual Rader implementation for prime-length cyclic correlation must beat
the generic library path on the same hardware, dtype, batch, and output
tolerance. The `2 * 5 * 7 * 67` factorization is not evidence until that code
and benchmark exist.

`RADER-1` is outside the first implementation slice.

### PRW-T1 — sharp structured-bank error conjecture

This is the only newly formulated mathematical conjecture retained after the
OPPW and binary-code reductions. For frozen overlap-one
Legendre-by-`RM(1,3)` banks, let \(A_{\min}\) be the number of nearest wrong
type/shift/mask states and let
\(\beta_q(d)=\Pr[\operatorname{Bin}(d,q)\ge\lceil d/2\rceil]\).
As \(p\) increases through primes congruent to `3 mod 4`, conjecture

\[
\Pr(\text{wrong-type bank error})
=
A_{\min}\beta_q(d_{\min})(1+o(1)).
\]

This requires fixing `q`, tie handling, transmitted-state averaging, and one
explicit bank sequence, then proving both

\[
\sum_{i<j,\ i,j\text{ nearest}}\Pr(E_i\cap E_j)
=o\!\left(A_{\min}\beta_q(d_{\min})\right)
\]

and

\[
\sum_{d>d_{\min}}A_d\beta_q(d)
=o\!\left(A_{\min}\beta_q(d_{\min})\right).
\]

Pairwise intersections becoming small one at a time is insufficient when the
number of competitors grows with `p`. The first tests enumerate exact
disagreement-set intersections at small primes, compare the ordinary union
bound with Hunter's spanning-tree correction, and use preregistered importance
sampling at larger primes. Failure of the ratio to stabilize, a non-negligible
farther-distance contribution, or persistent clustering of nearest events
kills `PRW-T1`. Even a proof would be a narrow decoder theorem, not a new
computing substrate.

### PRW-5 — later queue-conditioned correction

This hypothesis may be opened only if `PRW-1` through `PRW-4` survive.

Let `n_t` be causally available queue occupancy and `C_t` a frozen capacity.
The proposed operational score is

\[
S_t'=S_t-\lambda\log(1+n_t/C_t).
\]

Against `lambda = 0` and the existing static H5 centroid bank, a frozen
queue-aware route must reduce p95 queue delay or correction cost without more
than 0.5 percentage-point loss in held-out Recall@1 and without increasing
provenance violations. Queue state available only after routing is leakage.

## 6. Exact METHOD_DEV grid

Frozen seeds: `7`, `42`, `1337`.

Initial factors:

- lengths: `4091`, `4096`, `4691`;
- carrier: `legendre` where defined, and `rademacher`;
- layers: `1`, `8`;
- phase policy: `shared`, `independent`;
- planted mask family: `none`, `shared`, `typed16`, `free256`;
- decoder mask policy: `none`, `shared`, `typed16`, `free256`, crossed against
  the planted family on identical observations;
- bit-flip rates: `0.00`, `0.20`, `0.35`, `0.45`, `0.49`;
- payload noise: `0.00`, `0.10`, `0.25`, `0.50`;
- input class: planted, unrelated, non-cyclic perturbation;
- type count: `16`;
- waypoints per type: `8`.

`4096` is exercised end to end with a generic cyclic template and a frozen
Rademacher base carrier shared by all types within a seed. It is not passed
through the Legendre-only factory. Giving each type an independent random base
carrier is forbidden because that would inject an unmatched second type code.

The sparse OPPW baseline uses the identical phase codebook, type bank, planted
shift, masks, corruption groups, and unlock rule. It decodes the
one-pulse-per-layer representation directly. A campaign without this baseline
is `INCONCLUSIVE` for dense-carrier advantage.

The one-layer condition cannot test a relative-phase code and is a negative
control. Invalid combinations are recorded as structurally unavailable, not
silently omitted.

Every run records:

- type accuracy;
- phase exact accuracy and circular error;
- waypoint Recall@1 and Recall@k;
- carrier-only waypoint accuracy;
- true- and false-unlock rates;
- score and margin distributions;
- mask recovery where identified;
- mean and p95 latency;
- logical bits, total serialized bytes, actual array bytes, query bytes, and
  decoder working-array bytes;
- complete configuration and seed;
- failures and unavailable cells.

## 7. Data separation and thresholds

Synthetic generation is deterministic from the frozen seed but uses disjoint
`BANK`, `SELECT`, and `REPORT` random streams derived by domain-separated hashes
and disjoint group IDs. Thresholds are fitted only on `SELECT` planted and
unrelated examples.

The sole primary false-unlock cell is frozen as:

- length `4691`, Legendre carrier, `L=8`, shared phase;
- `planted_mask_family=typed16`, `decoder_mask_policy=typed16`;
- bit-flip rate `0.45`, payload noise `0.25`;
- `16` types and `8` waypoints per type; and
- seeds `7`, `42`, and `1337`.

For each seed, `SELECT` contains exactly `512` independent planted query groups
and `512` independent unrelated query groups. `REPORT` contains exactly `1024`
of each. A query group, not a corruption variant or candidate score, is the
sampling unit. Generation is streamed in fixed-size batches; no optional
stopping is permitted.

The three seeds are replicate streams within this one primary cell. Their
`SELECT` groups are pooled to freeze one threshold pair, which is then applied
once to the pooled `REPORT` groups. Per-seed summaries are diagnostics and
cannot each be promoted as separate primary cells.

The frozen threshold grid is:

- score threshold `-1.00, -0.95, ..., 1.00`;
- margin threshold `0.00, 0.01, ..., 1.00`.

From `SELECT`, retain threshold pairs whose one-sided 97.5% exact
Clopper-Pearson upper confidence bound on unrelated-query false unlock is at
most `0.01`. Choose the pair with maximum planted true-unlock recall; break
ties by lower false-unlock upper bound, then higher score threshold, then higher
margin threshold. If no pair qualifies, the cell fails closed. The selected
pair is applied once to `REPORT`.

The primary `REPORT` claim survives only if its one-sided 97.5% exact
Clopper-Pearson upper bound is at most `0.01`; point estimates alone never
satisfy the gate. With zero false unlocks this requires at least `368`
independent unrelated groups. Results from other cells are descriptive unless
a separate multiplicity-controlled promotion was frozen before reading their
`REPORT` outcomes.

A fast execution may use smaller counts only under the literal label
`SMOKE_NON_EVIDENTIARY`. Such a run checks wiring, determinism, and artifact
completeness but cannot pass PRW-1, PRW-3, or any `<=1%` false-unlock claim.
To exercise the downstream unlock path, it may freeze a separately labeled
`SMOKE_FROZEN_POINT_ESTIMATE_ONLY` threshold using the same grid and empirical
SELECT false-unlock rate. That threshold can never set the confidence gate,
promotion eligibility, or a scientific verdict.

`REPORT` is still `METHOD_DEV`: synthetic data and constructed collision groups
cannot establish usefulness on real retrieval. The labels mean only that the
implementation did not tune on its own scored rows.

All reported comparisons are paired by seed, type, waypoint, corruption draw,
and payload draw. Every tested length/carrier/mask cell is retained.

Bank evaluation is frozen: score every candidate type, choose the maximum with
lowest canonical type ID as the exact-tie rule, compute top-one minus top-two
margin, apply both frozen unlock thresholds, and only then score payloads within
the selected type. Candidate count, preprocessing, and batched query latency
are charged to the method. Latent synthetic truth is never passed to the
scorer.

## 8. Controls that can kill the story

1. **Direct type lookup:** if known metadata is cheaper and equally robust,
   the carrier has no operational role.
2. **Payload only:** tests whether carrier routing adds anything.
3. **Carrier only:** proves that synchronization is not semantic identity.
4. **Independent shifts:** exposes loss of the interlayer constraint.
5. **Random carriers:** tests whether Legendre structure matters.
6. **Equal-bit repeated layer:** tests whether eight layers add structure or
   merely repetitions.
7. **Free 256 masks:** quantifies false-unlock inflation.
8. **Non-cyclic corruption:** tests whether the claimed equivariance is
   specific rather than generic robustness.
9. **4091 and 4096:** test whether `4691` matters.
10. **Matched candidate and latency budgets:** prevent a larger search from
    masquerading as better memory.
11. **Matched repetition/energy code:** tests whether dense robustness is only
    the expected consequence of using roughly `(p +/- 1)/4` more Hamming
    distance than sparse OPPW.
12. **Block-correlated and burst flips:** deliberately violate iid noise while
    retaining the same marginal flip rate; any iid theorem claim must fail
    closed outside its declared channel.

## 9. Novelty falsifier

Close prior art covers every ingredient:

- circular convolution and phase binding in vector symbolic architectures;
- [residue hyperdimensional computing](https://arxiv.org/abs/2311.04872);
- [resonator-network factorization](https://arxiv.org/abs/2007.03748);
- [linear-code HDC](https://arxiv.org/abs/2403.03278);
- [quantized FHRR](https://arxiv.org/abs/2604.25939);
- [phase-associative memory](https://arxiv.org/abs/2604.05030);
- [multi-reference alignment](https://arxiv.org/abs/2007.11482);
- [multi-frequency group synchronization](https://arxiv.org/abs/2406.03424);
- [cyclically equivariant neural decoding](https://proceedings.mlr.press/v139/chen21w.html);
- cyclic-shift code families such as
  [two-dimensional optical orthogonal codes](https://arxiv.org/abs/2602.18864)
  and frequency-hopping sequences;
- the 1992 binary constant-weight cyclic construction that maps each symbol of
  a `p`-ary outer word to a cyclic shift of a length-`p` Legendre word, yielding
  the same unmasked distance
  `(outer Hamming distance) * (p + 1) / 2`
  ([Nguyen Q. A., Györfi, and Massey](https://doi.org/10.1109/18.135636));
- [Kronecker-rotation-product VSA cleanup](https://proceedings.mlr.press/v284/liu25b.html);
- product-key and modern Hopfield memories;
- load-balanced mixture-of-experts routing.

Therefore primes, rotations, phase, codebooks, subspaces, associative memory,
and queue-aware routing are not individually novel.

The only candidate research contribution is the complete fail-closed
composition:

> a constrained diagonal-quotient phase key that unlocks a separate semantic
> payload bank, then participates in causal load-aware routing and reversible,
> provenance-scoped waypoint correction.

Even that is a combination-novelty conjecture. A bounded search finding no
identical paper is not proof of novelty or patent clearance.

The phase-code portion itself is not combination-only or unresolved: it is both
an OPPW representation and a direct instance of the known Legendre-inner,
`p`-ary-outer cyclic-code construction. The eight Walsh rows and their
negatives are also the known 16-word biorthogonal/first-order Reed-Muller code
`RM(1,3)`.

Mandatory added baselines are therefore:

- sparse OPPW decoding with the identical phase codebook;
- a conventional low-correlation synchronization family where available;
- ordinary direct metadata and product-key routing;
- the existing random bipolar and length controls.

The full-factorial interaction is the decisive test. Let `V(A)` be held-out
utility for mechanism subset `A` drawn from synchronization, typed masks,
payload gating, queue state, and reversible bank updates. The highest-order
Möbius interaction is

\[
M(F)=\sum_{A\subseteq F}(-1)^{|F|-|A|}V(A).
\]

If the complete system is explained by constituent main effects and
lower-order interactions, the result is engineering integration. A new theory
becomes plausible only after a preregistered practical `M(F)` threshold has a
positive 97.5% held-out lower confidence bound, survives new datasets and
subdomains, and beats matched known architectures.

## 10. Reconditioning rules

- If `PRW-0` fails, repair arithmetic/code only.
- If `PRW-1` fails, cut the carrier.
- If `PRW-2` fails, cut “unlocking” language and retain at most a
  synchronization primitive.
- If `PRW-3` fails, cut polarity layers.
- If `PRW-4` fails, remove all special status from `4691`; retain the best
  ordinary length.
- If the direct type lookup dominates, state that the phase carrier is useful
  only where a noisy distributed key is independently justified.
- Do not open `PRW-5` until the isolated mechanism survives.
- Do not add a production router, modify the evidence DAG, or issue a white
  paper from synthetic METHOD_DEV evidence.
