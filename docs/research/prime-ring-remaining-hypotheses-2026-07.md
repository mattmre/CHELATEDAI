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
