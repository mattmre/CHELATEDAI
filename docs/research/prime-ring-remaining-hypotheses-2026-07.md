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
selected-frequency capacity or resonance.

The stronger 8-bit check crossed all three two-bin subsets with quantizer-grid
origins \(0,.25,.5,.75\). The selected-bin ranking changes with origin; the
largest within-subset accuracy spread is about \(0.0083385070\). This kills a
selected-frequency interpretation of the small quantized differences in this
toy. The all-nonconjugate-bin diagnostic often scores higher but uses three
coefficients rather than two, so it is not a matched superiority comparison.
The bounded result is `GENERIC_REDUNDANCY_ONLY`; frequency selection, harmonic
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
Pure Legendre carriers are a mandatory degeneracy control because
multiplication produces only the original sequence or its sign complement.

### Hypothesis

Factor-addressable payload permutations reduce search or correction cost
relative to additive rotations and ordinary learned/permuted indexes at
matched bytes and candidate count.

### Kill criteria

- The effect is only a relabeling of a flat permutation.
- The \(2\)-factor contributes nothing beyond an explicit polarity bit.
- Additive or random permutations match utility at lower cost.

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
3. Test `PRW-C1` only if it is not algebraically redundant.
4. Run powered PRW mechanism tests only after resource review.
5. Open `PRW-Q1` and `PRW-L1` only after real held-out retrieval survives.
6. Treat `PRW-D1` as a null until a matched learned-dimension campaign exists.

No hypothesis may inherit support from a predecessor. Unit tests establish
implementation correctness only.

## 9. Current bounded execution boundary — 2026-07-24

The graph and selected-frequency lanes have both produced negative small-prime
results. `PRW-G1` collapses to the explicit global-phase-plus-fixed-offset
comparator on connected, cycle-consistent labels. `PRW-H1` shows generic
multi-bin redundancy, but no selected-bin advantage: unquantized two-bin sets
tie and eight-bit rankings reverse with quantizer-grid origin.

The lower-rate raw-sanity aggregation path is implemented but has **not** been
scientifically executed. Its structural gate requires all twelve frozen
`(seed,q)` cells, canonical planted-IID REPORT lineage, SELECT/REPORT stream
separation, and a per-run HMAC authority that is never serialized. Rewriting a
failure and reconstructing every seal from serialized fields now fails
authentication. This prevents a retained artifact from certifying its own
mutated summaries; it is not a claim of security against arbitrary code already
executing inside the campaign process.

The complete bounded research suite passes 177 tests. No full campaign, real
corpus, large-prime timing, or retained evidence generation was run. Therefore:

- graph-specific and selected-frequency interpretations are killed only in the
  declared tiny domains;
- the full raw-sanity criterion remains `UNEXECUTED_POWERED_EVIDENCE`;
- no result supports novelty, production utility, cheaper training, better
  RAG recall, or a high-prime dimensionality advantage.

The next safe discriminator is a preregistered `p=11`
quantizer-origin/dither null with an up-front work estimate. `PRW-C1` should
receive an algebraic flat-permutation/polarity-bit redundancy screen before any
implementation. `PRW-Q1` and `PRW-L1` remain blocked by their held-out retrieval
precondition.
