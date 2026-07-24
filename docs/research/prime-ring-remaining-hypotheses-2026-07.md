# Remaining Prime-Ring / Onion-Lattice Hypotheses

Status: `NON_CONFIRMATORY_METHOD_DEV`

This document converts the remaining informal language into independent,
falsifiable mechanisms. It does not assert novelty, physical resonance,
gravity, quantum behavior, or production utility.

## 1. Shared mathematical vocabulary

Let each node \(v\) carry a cyclic fiber \(x_v\in\mathbb R^p\). A common
rotation acts diagonally:

\[
(a\cdot x)_v=R_a x_v,\qquad a\in\mathbb Z_p.
\]

An edge \((u,v)\) may carry a declared relative phase
\(g_{uv}\in\mathbb Z_p\). A cycle is consistent when its signed edge phases
sum to zero modulo \(p\). This is ordinary finite-group synchronization or
graph consistency; “bonding” and “mesh” are only visual analogies.

“Sector” means a declared subset of coordinates or Fourier bins. “Harmonic”
means a discrete Fourier coefficient. “Gravity” means a causal queue/load
penalty. “Living” means a reversible update to derived state with immutable
source evidence.

Ring length, embedding dimension, graph dimension, and number of layers are
separate factors. In particular, a length-4096 cyclic array is not a
4096-dimensional learned embedding unless an experiment explicitly makes it
one.

## 2. PRW-G1 — edge-coupled cyclic-fiber graph

### Mechanism

Replace the flat eight-layer bank with a graph \(G=(V,E)\). Score a candidate
assignment of node phases by

\[
S(\phi)=
\sum_{v\in V} s_v(\phi_v)
-\lambda_E\sum_{(u,v)\in E}
\rho\!\left(\phi_v-\phi_u-g_{uv}\right),
\]

where \(\rho(0)=0\) and \(\rho(a)>0\) for \(a\ne0\).

This is the exact version of the “right-angle mesh,” “sector pairing,” and
“protein-like edge bonding” ideas. A 3D drawing is not part of the hypothesis;
only graph topology and cycle constraints are.

### Prior-art boundary

Graph-relative phase estimation is an established synchronization problem, not
a new mathematical family. Singer's angular-synchronization formulation
recovers node phases from noisy relative offsets, and later compact-group work
explicitly includes finite cyclic groups and multiple representation channels:

- [Angular Synchronization by Eigenvectors and Semidefinite Programming
  (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3003935/)
- [Message-passing algorithms for synchronization problems over compact groups
  (2016)](https://arxiv.org/abs/1610.04583)
- [Random Multitype Spanning Forests for Synchronization on Sparse Graphs
  (2025)](https://epubs.siam.org/doi/full/10.1137/24M1649563)

`PRW-G1` can therefore test only a narrower systems interaction: whether
node-local PRW/OPPW carrier scores plus typed cyclic edge constraints produce a
matched-resource retrieval or correction benefit. The graph, cyclic group,
cycle-consistency objective, and any 3D visualization are not novelty
candidates.

### Hypothesis

At matched stored bytes, candidate count, channel uses, latency budget, and
false-unlock rate, edge consistency improves recovery of planted conjunctive
assignments over:

1. independent node decoding;
2. one global shared phase;
3. a non-cyclic product-key lookup;
4. the same graph with shuffled edge labels.

### Kill criteria

- Any gain vanishes after matching the candidate/search budget.
- Shuffled or zero edge labels perform equally well.
- Cyclic consistency reduces recall or raises false unlocks.
- The graph collapses algebraically to one global phase or independent nodes.

### First bounded test

Use \(p\in\{7,11,31\}\), at most 12 nodes, at most 18 edges, and exact or
dynamic-programming decoding only when the estimated peak is below 512 MiB.

### Bounded result — 2026-07-24

The exact tiny analyzer compares independent unary decoding, graph-coupled
decoding, and a deterministic structure-matched edge-label permutation
control over the same streamed \(p^{|V|}\) assignments. Thirteen focused tests
and an independent exhaustive check over all 511 nonempty labeled simple
graphs on three nodes at \(p=7\) passed.

A planted triangle can be uniquely recovered by the graph score while the
independent and shuffled-label controls fail, with a planted-margin gain of
five in the toy construction. That observation does **not** survive the
mechanism kill criterion: the connected, cycle-consistent difference graph is
exactly one global phase plus fixed node offsets. The result is therefore
`MECHANISM_COLLAPSES_TO_GLOBAL_PHASE`, not evidence for a new graph memory.

The follow-up implements that explicit comparator by deriving canonical
offsets \(o_0=0\) and searching the exact family
\(x_i(z)=z+o_i\pmod p\). It filters the same streamed \(p^{|V|}\) assignment
order while reporting its honest \(p\)-candidate denominator. In the helpful
\(p=7\) case, the comparator and graph-coupled decoder have identical selected
assignment, score, tie count, planted margin, and planted rank; every
graph-minus-global effect is zero. Contradictory or disconnected graphs refuse
the comparator. Fifteen focused tests and an independent brute-force audit of
all 511 nonempty labeled three-node graphs pass. This closes the bounded
graph-specific lane negatively.

## 3. PRW-H1 — multi-frequency phase signature

### Mechanism

For a frozen set of nonzero Fourier bins \(K\), represent each node by

\[
z_{v,k}=\widehat{x_v}(k)/|\widehat{x_v}(k)|,\qquad k\in K,
\]

when the magnitude is nonzero. Under rotation by \(a\),
\(z_{v,k}\mapsto e^{-2\pi i k a/p}z_{v,k}\).

This is the operational meaning of “harmonic,” “resonance,” and “polar
alignment.” It is ordinary phase synchronization. No physical resonance or
quantization claim is implied.

### Prior-art boundary

Multi-frequency group/phase synchronization is also an established named
family:

- [Message-passing algorithms for synchronization problems over compact groups
  (2016)](https://arxiv.org/abs/1610.04583)
- [Multi-Frequency Phase Synchronization
  (2019)](https://arxiv.org/abs/1901.08235)

Accordingly, neither multiple Fourier bins nor their phase-consistency
objective is a novelty candidate. The bounded experiment may only ask whether
a frozen, quantized, resource-matched bin signature interacts usefully with the
specific PRW carrier/type/mask construction.

Quantizer-origin averaging must also be separated from classical subtractive
dither. The relevant baseline conditions predate this work:

- [Dither Signals and Their Effect on Quantization Noise
  (Schuchman, 1964)](https://doi.org/10.1109/TCOM.1964.1088973)
- [Dithered Quantizers
  (Gray and Stockham, 1993)](https://doi.org/10.1109/18.256489)

A finite common-origin grid is therefore described below only as a nuisance
audit. It is not treated as independent stochastic dither.

### Hypothesis

A preregistered, small set of frequency bins improves recovery under a frozen
structured corruption family relative to time-domain cyclic correlation at
matched stored coefficients, arithmetic operations, and false-unlock rate.

### Controls

- random bins with the same count;
- magnitude-only coefficients;
- one frequency;
- all frequencies;
- time-domain correlation;
- randomized phase with preserved magnitudes.

### Kill criteria

- Performance follows only the number of searched bins.
- Magnitude-only or randomized phases match the proposed signature.
- Quantization destroys the effect at the frozen bit budget.

### Bounded result — 2026-07-24

The exact analyzer searches \(T p\) shared type/shift states for every
condition and streams every binary-symmetric-channel corruption pattern for
tiny banks. It uses immutable hard ceilings, caller-lowerable budgets, one
deadline across all controls, and no packed-storage or speed claim.

For \(p=7\), one type, one node, and all 128 corruption patterns, every
two-bin subset of the three nonconjugate bins has identical unquantized exact
accuracy at each of \(q\in\{.20,.35,.45\}\). Distinct bins outperform repeated
copies of one bin under noise, which is generic redundancy rather than
selected-frequency capacity or resonance. The stronger 8-bit check crossed all
three two-bin subsets with common quantizer-grid origins \(0,.25,.5,.75\). The
selected-bin ranking changes with origin; the largest within-subset accuracy
spread is about \(0.0083385070\).

The follow-up `PRW-H1Q` screen covers all ten two-bin subsets of the five
nonconjugate bins at \(p=11\), all 2,048 exact BSC patterns at \(q=.45\), the
unquantized baseline, and the same four common origins. A cached shared stream
preflights at 54,432 model-level bytes, 17,951,240 work units, and 102,400
condition evaluations under one 25-second deadline. Focused equivalence tests
match the public reference decoder's scores, winners, ties, margins, and
abstentions. The exact root-owned run completed in 22.359 seconds.

The ten pairs form two five-member multiplicative-relabeling orbits. Within
each orbit the unquantized accuracy is constant to floating precision: orbit A
has mean \(0.1202012759\) and spread \(4.16\times10^{-17}\); orbit B has mean
\(0.1231158929\) and spread \(6.94\times10^{-17}\). The between-orbit difference
is \(-0.0029146170\), an ordinary ratio-class distinction rather than evidence
for one privileged frequency pair. Across the common-origin grid, 30 of 45
pairwise orderings reverse, no pair strictly dominates all others at every
origin, and the maximum per-pair origin spread is \(0.0048568741\).

The common-origin diagonal is not closed under every multiplier/conjugation
action. A stronger per-bin \(A_4^2\) origin product would require 59,734,280
estimated work units and is refused above the immutable ceilings. Thus this
screen supports unquantized within-orbit equivariance and quantizer-origin
sensitivity only. It does not establish continuous-dither behavior. The
bounded result remains `GENERIC_REDUNDANCY_ONLY`; frequency selection, harmonic
resonance, address expansion, packed-storage savings, and production utility
remain unsupported.

## 4. PRW-D1 — prime ambient-dimension null

### Null hypothesis

Prime dimensionality has no independent ML benefit after matching useful
coordinates, stored bytes, model parameters, candidate count, and arithmetic
cost.

### Required comparison

Compare nearby dimensions such as 4091, 4096, and 4691 using the same learned
task and padding/projection policy. Ring-length autocorrelation and FFT timing
must be reported separately from embedding quality.

### Kill criterion for a prime-dimension claim

Any apparent gain disappears after matching capacity and compute, or the
power-of-two control achieves equal utility at lower latency.

No learned high-dimensional campaign is authorized in the resource-bounded
continuation.

## 5. PRW-C1 — multiplicative CRT payload pivot

### Mechanism

For nonzero indices of \(\mathbb Z_{4691}\), use a declared primitive-root
coordinate \(n\in\mathbb Z_{4690}\) and its CRT representation

\[
n\leftrightarrow(n\bmod2,n\bmod5,n\bmod7,n\bmod67).
\]

Apply multiplicative pivots only to arbitrary or learned payload coordinates.
The Legendre character on the nonzero subgroup is a mandatory degeneracy
control because multiplication produces only the original character or its
sign complement. The repository's bipolar carrier sets coordinate zero to
`+1`; multiplication fixes that coordinate, so the full carrier has one
fixed-coordinate exception to the global-sign rule.

### Prior-art boundary

Prime-length cyclic transforms and coprime-factor/CRT indexing are established
techniques:

- [Discrete Fourier Transforms When the Number of Data Samples Is Prime
  (Rader, 1968)](https://doi.org/10.1109/PROC.1968.6477)
- [The Interaction Algorithm and Practical Fourier Analysis
  (Good, 1958)](https://academic.oup.com/jrsssb/article/20/2/361/7027226)

The smooth factorization \(4691-1=2\cdot5\cdot7\cdot67\) can make a
prime-length implementation convenient. It does not, by itself, add memory
states, payload capacity, or a new transform.

### Hypothesis

Factor-addressable payload permutations reduce search or correction cost
relative to additive rotations and ordinary learned/permuted indexes at
matched bytes and candidate count.

### Kill criteria

- The effect is only a relabeling of a flat permutation.
- The \(2\)-factor contributes nothing beyond an explicit polarity bit.
- Additive or random permutations match utility at lower cost.

### Bounded result — 2026-07-24

On \(\mathbb F_p^\*\), writing \(x=g^n\) makes multiplication by a power of
\(g\) an ordinary cyclic shift of \(n\in\mathbb Z_{p-1}\). CRT coordinates are
a bijective reshape of that one shift. Allowing one shift per factor therefore
still gives exactly \(2\cdot5\cdot7\cdot67=4690\) shared pivot choices, not
additional states. For the quadratic character, the action reduces to exponent
parity. Under the repository's bipolar convention an odd pivot is
\(-L+2e_0\): every nonzero coordinate changes polarity and coordinate zero
remains fixed.

Pure layers of additive rotations and multiplicative pivots generate the
ordinary affine family \(x\mapsto ax+b\). Every such onion word has one
canonical affine normal form, so layer order does not create an additional
group. At \(p=4691\), the bounded executable witness estimates 3,310,582
Python-object bytes and 2,176,160 work units; these are not process RSS or a
timing benchmark. It verifies 4,690 CRT tuples and shared pivots, a Legendre
stabilizer/orbit of \(2345/2\), and the ordinary affine upper bound
\(4691\cdot4690=22,000,790\).

For eight independently selected layers, \(4690^8\) is a count of possible
control tuples requiring about 97.563 control bits. The tuples were not
enumerated, and neither payload-level independence nor distinguishable memory
states was demonstrated. The exact algebraic kill criteria trigger for flat
relabeling, no extra CRT states, Legendre parity with the fixed-zero exception,
and affine collapse. Learned-payload utility and implementation cost remain
untested, so a matched systems experiment—not new group mathematics—is the only
surviving lane.

### PRW-C1B — bounded conditional-replacement screen

A separate \(p=7\) screen exhausts all 128 binary payloads for programs of at
most four operations with two fixed address masks and constant replacement
bits. The representative program reduces to \(x\mapsto3x+6\) followed by one
last-write-wins overwrite. Disjoint overwrites commute; overlapping overwrites
show only ordinary last-write-wins order. The preflight is 99,072 estimated
bytes and 29,808 work units under a one-second deadline.

This closes only the static-mask lane. Payload-, query-, or layer-state-dependent
predicates and replacement functions were not tested. Any such dynamic
mechanism needs a separately frozen branch/search budget and controls matched on
predicate information, replacement count, mask size and overlap, and search
opportunity.

## 6. PRW-Q1 — causal queue potential

### Mechanism

For causally available queue occupancy \(n_t\) and frozen capacity \(C_t\),

\[
S'_t=S_t-\lambda_Q\log(1+n_t/C_t).
\]

This is the complete operational meaning of “dimensional gravity during
queuing.”

### Hypothesis

On held-out queue traces, the penalty improves completed correct retrievals per
unit latency without increasing false unlocks, starvation, or provenance
violations versus:

- no load penalty;
- round-robin routing;
- shortest-queue routing;
- capacity-only admission;
- the existing static bank.

### Preconditions

`PRW-1` through `PRW-4` must first survive matched scientific gates. Queue
routing cannot rescue an ineffective retrieval mechanism.

## 7. PRW-L1 — reversible living waypoint state

### Mechanism

Maintain immutable source evidence \(e\) and a versioned derived state
\(\theta_t\). Updates are accepted only when a frozen confidence/provenance
gate passes:

\[
\theta_{t+1}=
\begin{cases}
U(\theta_t,e_t), & G(e_t,\theta_t)=1,\\
\theta_t, & \text{otherwise}.
\end{cases}
\]

Every update records its parent version and supports exact rollback.

### Hypothesis

Against static, one-shot, global-update, and LoRA-style controls, the
reversible local update reduces correction steps or trainable parameters while
preserving retained-task utility, calibration, and provenance.

### Kill criteria

- Living and static states are numerically identical.
- One-shot correction matches or beats repeated updates.
- Repeated updates create cycles, forgetting, or provenance drift.
- Benefits disappear after matching update count and trainable parameters.

Adjacent repository evidence is already negative for living/static and
compound-correction variants, so this hypothesis begins with a skeptical
prior.

## 8. Dependency and promotion order

1. Validate matched PRW controls and finite code theory.
2. Test `PRW-G1` and `PRW-H1` only on bounded synthetic cases.
3. Treat pure `PRW-C1` permutation layering as algebraically closed; test only a
   separately declared nonlinear or runtime-conditional mechanism.
4. Run powered PRW mechanism tests only after resource review.
5. Open `PRW-Q1` and `PRW-L1` only after real held-out retrieval survives.
6. Treat `PRW-D1` as a null until a matched learned-dimension campaign exists.

No hypothesis may inherit support from a predecessor. Unit tests establish
implementation correctness only.

## 9. Current bounded execution boundary — 2026-07-24

The graph, selected-frequency, and pure permutation-layer lanes have produced
negative bounded results. `PRW-G1` collapses to the explicit
global-phase-plus-fixed-offset comparator on connected, cycle-consistent labels.
`PRW-H1` shows generic multi-bin redundancy but no selected-bin advantage:
unquantized \(p=7\) two-bin sets tie, while the \(p=11\) pairs split into two
ordinary multiplicative orbits and their quantized orderings remain
origin-sensitive. `PRW-C1A` shows that CRT pivots are flat exponent shifts and
pure rotation/pivot onion words are affine maps. `PRW-C1B` reduces fixed-mask,
constant-bit replacements to affine-plus-last-write-wins normal form.

The lower-rate raw-sanity aggregation path is implemented but has **not** been
scientifically executed. Its structural gate requires all twelve frozen
`(seed,q)` cells, canonical planted-IID REPORT lineage, SELECT/REPORT stream
separation, and a per-run HMAC authority that is never serialized. Rewriting a
failure and reconstructing every seal from serialized fields now fails
authentication. This prevents a retained artifact from certifying its own
mutated summaries; it is not a claim of security against arbitrary code already
executing inside the campaign process.

The complete current-tree bounded research suite passes 230/230 tests: the
prior 177 checks plus 53 focused RB-7 checks. Scoped Ruff, read-only AST parsing,
and whitespace validation also pass. No full campaign, real corpus, large-prime
transform/timing campaign, or retained evidence generation was run. Therefore:

- graph-specific and selected-frequency interpretations are killed only in the
  declared tiny domains;
- pure CRT/onion-permutation novelty is killed algebraically, while the
  static-replacement normal form closes only its declared \(p=7\) domain;
- the full raw-sanity criterion remains `UNEXECUTED_POWERED_EVIDENCE`;
- no result supports novelty, production utility, cheaper training, better
  RAG recall, or a high-prime dimensionality advantage.

The next safe discriminator is no longer another pure permutation or
common-origin sweep. It is either (a) a proof-first normal form for genuinely
runtime-dependent replacement predicates, with branch information and search
opportunity charged explicitly, or (b) a tiny held-out learned/nonseparable
payload comparison against flat and randomized controls after a fresh resource
review. `PRW-Q1` and `PRW-L1` remain blocked by their held-out retrieval
precondition.
