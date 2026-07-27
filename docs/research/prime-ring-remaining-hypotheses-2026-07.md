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

## 2A. PRW-G2 — irreducible orbit-coded factor/hypergraph mesh

### Status

`BOUNDED_P7_ALGEBRAIC_SCREEN_COMPLETE / STATIC_ADVANTAGE_NOT_ESTABLISHED`.
This is not a reopening of `PRW-G1`. Connected, cycle-consistent
fixed-difference graphs remain closed because their scores reduce to a global
phase plus fixed offsets. At `p=7`, constructed pair, three-way, frustrated,
and query-interaction fixtures survive the necessary residual/reducibility
checks; exact flat and factorized MAP agree; independent, shuffled-label,
wrong-grouping, and matched-random controls pass; and no static advantage is
established. `PRW-G2H` remains conditionally open because the current pairwise
auxiliary construction is only an upper bound and no minimum auxiliary cost,
compact orbit representation, or lower-work decoder advantage is proved.
Stage 4 is blocked until that auxiliary/candidate gate is specified; it is not
authorized merely by nonzero interaction residuals. `PRW-G2A` remains blocked.

### Mechanism and falsifiable claim

For candidate state \(h=(h_1,\ldots,h_n)\), use

\[
S(h\mid y,q)=\sum_v \ell_v(h_v;y_v,q)
             +\sum_{A\in\mathcal F}\psi_A(h_A;q).
\]

The admissible sublanes are:

- `PRW-G2P`: a pair factor with a nonzero double-centered interaction;
- `PRW-G2H`: an arity-three factor with a nonzero highest-order
  Möbius/ANOVA residual;
- `PRW-G2Q`: query conditioning that changes an interaction rather than only
  changing unary terms; and
- explicitly noisy, frustrated, or multiple-latent-state variants for which
  the `PRW-G1` reduction proof does not apply.

The null is that an information-matched flattened code or matched generic
factor model reproduces the decision, error, cost, and calibration behavior.
The alternative requires a reproducible gain after charging factor
descriptions, graph construction, oracle grouping, parameters, operations,
wall time, and RSS. Required controls are independent nodes, explicit
global-phase-plus-offset, flat same-information coding, shuffled factors,
wrong grouping, and random factors.

### First bounded gate

Run exact enumeration only after `PRW-JO1` and, if nondegenerate, `PRW-A1`.
Start with \(p\in\{7,11\}\), 3--6 nodes, factor arity at most 3, at most 8
factors, and at most 200,000 joint assignments. Refuse any cell predicted to
exceed 256 MiB peak memory, 25,000,000 scored operations, or 30 seconds. A
static `PRW-G2` result must survive before opening adaptive `PRW-G2A`.

The canonical hypotheses, controls, kill rules, and artifact contract are in
`docs/research/prime-ring-orbit-plank-mesh-protocol-2026-07.md`.

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

1. Retain the internally proved exact-Hamming `PRW-T1R` theorem and frozen
   50/50 analytic tie corollary, but resolve the failed current float-FFT
   production linkage through a declared numerical-tie contract and
   exact-Hamming/direct/FFT boundary parity. Do not infer finite decoder error
   from the leading-tail formulas.
2. Treat the reconditioned primary-source chart as bounded literature
   positioning only. It now includes the direct 1992 Legendre-inner
   construction; independent proof/construction-equivalence and specialist
   citation review remain required if publication novelty is pursued.
3. Move the common latent-perturbation/common-channel controls ahead of new
   mechanism claims so carrier comparisons share the same corruption law.
4. Record `PRW-JO1` stacking/dimensionality as closed by exact flat identity.
   Run known-design and unrestricted matched-work controls only for the
   optional ordinary constrained-code residue.
5. Before an A1 policy, compute a single full action-channel conjugacy over all
   truths and freeze whether actions are physically selectable plus their
   semantic/relabeling cost. Kill if actions are free global relabelings;
   otherwise compare with matched random, cyclic, greedy, entropy-gain,
   Chernoff/MaxEJS, and unrestricted actions.
6. Treat the `PRW-G2` algebraic screen as complete: necessary pair/hyper/query
   interactions exist, but static advantage is false.
7. Before Stage 4, prove or bound pairwise auxiliary minimality for every
   hyperfactor and name a concrete orbit-specific representation/decoder
   advantage. Run one information-fair discriminator only if that survives.
8. Open adaptive `PRW-G2A` only after a static factor survives; keep
   information acquisition separate from computation scheduling.
9. Then run the noisy/nonlinear quotient and broader transcript-inference
   campaigns with matched generic learners, parameters, FLOPs, data, and
   preregistered seeds.
10. Run the bounded real-corpus gate before any relational-mesh retrieval
    claim, and measure process-tree RSS plus checkpoint/kill/resume behavior
    before expanding the campaign.
11. Open `PRW-Q1` and `PRW-L1` only after real held-out retrieval survives.
    Treat `PRW-D1` as a null until a matched learned-dimension campaign exists.

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

## 10. Current bounded execution boundary — 2026-07-25 RB-9

This section supersedes the execution status in Section 9 without rewriting its
historical 2026-07-24 record.

The original 56-state nearest-only form of `PRW-T1` is falsified for the
tie-as-error wrong-type competitor-event union. Eight states at
\(d_p+4\), where \(d_p=(7p-1)/2\), contribute the positive limiting fraction
\([4q(1-q)]^2/7\) relative to the nearest term. The reconditioned theorem uses

\[
B_p(q)=56\beta_q(d_p)+8\beta_q(d_p+4)
\]

and proves

\[
\Pr\!\left(\bigcup_i E_i^{\ge}\right)=B_p(q)(1+o(1))
\]

for fixed \(q\in(0,1/2)\), primes \(p>7\) congruent to `3 mod 4`, and every
transmitted state in the frozen two-type/eight-layer `typed16` bank. The proof
uses the complete seven-shell spectrum, a linear distance gap after the 64
leading states, the five exact leading-pair classes, a strict joint
large-deviation rate, Bonferroni, and an explicit regular Hamming-isometry
action across all `32p` states.
Status is `PRW-T1R-TE-EVENT-UNION: PROVED_ASYMPTOTIC_WITHIN_FROZEN_MODEL`.
The production decoder's lowest-type-ID tie rule is resolved for canonical
type-zero truth: its strict-win leading term is asymptotically
`q/(1-q)` times the inclusive term, and correct-type competition is negligible.
Type-one truth receives the adverse tie policy, so the balanced/all-state
decoder theorem remains unresolved. The result is not licensed for
\(q_p\to1/2\), growing layer count, arbitrary masks/signatures, or other banks.
The formal proof note derives the state-equivariant event theorem; the remaining
decoder gap is the asymmetric type-ordering path and transmitted-type mixture,
not an event-union symmetry gap.

Two systems screens also ran:

- An exact modular-affine learner operating on raw `Z_7^4` coordinates
  recovered the planted gauge-null rule on unseen gauges and withheld quotient
  classes. Four independent rows are rank-deficient and the fifth exactly
  identifies the five affine parameters. This is algebraic solver verification
  in a favorable noiseless affine teacher/learner match, not evidence of
  generic or neural invariance discovery.
- A factorized Boolean learner given the correct causal groups reduced
  finite training-search work on planted nonlinear transcript tasks. The
  unrestricted flat learner chose a simpler direct-payload shortcut, while a
  causal-only flat counterfactual also achieved shifted accuracy `1.0`. The
  equal-complexity wrong-group factorization underperforms, so the remaining
  signal is specifically the supplied correct grouping. True no-shift and
  nuisance-remapping controls are now distinct and the run contract is hashed.
  The factorized and flat learners have the same padded storage and inference
  envelope. This is a candidate oracle structural-prior/search-cost signal
  only, not an accuracy, capacity, compression, latency, or
  inference-operation advantage.

The remaining work is ranked by information value:

1. Extend the tested canonical type-zero decoder corollary to type-one truth and
   a frozen balanced mixture, then test the complete implemented decoder path.
2. Build a fresh primary-source claim chart for the narrow `PRW-T1R` theorem.
   Until then, novelty is unknown rather than established.
3. Test noisy and nonlinear quotient tasks with a generic or neural learner,
   matched parameters/FLOPs/data, and at least ten preregistered seeds.
4. Test transcript inference with matched generic model classes, broader
   teacher families, statistical seeds, and a nonconstructed shift; charge the
   causal grouping as prior information.
5. Compare carrier/control families through a common latent perturbation or
   common binary channel. The present dense and sparse OPPW noise laws are not
   matched.
6. Run a bounded real-corpus pilot, beginning with cached SciFact and then
   NFCorpus, using frozen embeddings and exact, ANN, direct-metadata,
   product-key, random, payload-only, and OPPW controls at matched candidate,
   storage, and latency budgets.
7. Measure process-tree RSS and implement atomic content-addressed
   checkpoint/kill/resume fault injection before any broader campaign.

`PRW-Q1`, `PRW-L1`, queue-conditioned correction, and “living” waypoint state
remain blocked until the real-retrieval gate survives. Do not rerun the refused
11,520-cell grid, pure CRT/onion permutations, constructed quotient
enumeration, the same local Rader benchmark, a third paid predicate bit alone,
or a dense-versus-sparse comparison with mismatched corruption laws.

## 11. Bounded RB-10 execution — 2026-07-25

The bounded continuation separated four explicit hypotheses:

1. `PRW-JO1`: fixed joint-plank orbit-spectrum coding;
2. `PRW-A1`: posterior-guided plank acquisition;
3. `PRW-G2`: irreducible orbit-coded pair and hypergraph factors; and
4. `PRW-G2A`: adaptive factor acquisition after a static factor survives.

The frozen type-mixture decoder prerequisite and bounded primary-source claim
chart are complete. Three exact cells then executed sequentially under the
shared RB-10 artifact/resource contract:

- `PRW-JO1` at `p=11`, `K=2` reduces the aligned leading-shell multiplicity
  from `56` to `19`, but 15 restricted schedules tie, complementary and
  seeded-random controls match, and the stack is exactly the same longer code
  as its flattened concatenation. Known-design and unrestricted matched-cost
  controls remain open.
- The `PRW-A1` action gate finds ten distinct ordered fixed-label fingerprints,
  160 variable competitors, and crossovers in all 45 action pairs, while all
  ten sorted distance multisets are identical. A posterior policy remains
  blocked on semantic action/relabeling cost and exact generic controls.
- `PRW-G2` at `p=7` contains constructed pair, three-way, frustrated, and
  query-interaction residuals that do not collapse under the necessary
  algebraic tests. Exact flat and factorized MAP agree, mandatory algebraic
  controls match, and no static advantage is established. An information-fair
  noisy recovery test is the only authorized next mesh step.
- `PRW-G2A` remains blocked because no static mesh advantage exists.

The retained evidence is
`artifacts/method-dev/prime-ring/rb10-bounded-experiment-manifest.json` plus
its three content-bound stage artifacts. Measured child peak RSS stayed near
25 MiB, but the monitor is reactive rather than a hard kernel cap. These are
bounded synthetic mechanics, not novelty, application, training, retrieval,
cost, latency, or production evidence. The canonical protocol remains
`docs/research/prime-ring-orbit-plank-mesh-protocol-2026-07.md`.

## 12. Code-backed current boundary — 2026-07-26 RB-11

This section supersedes Section 11's current queue without rewriting the RB-10
execution history.

### Decoder/theorem boundary

The exact-Hamming proof note and 14 focused formula tests support
`PRW-T1R-TE-EVENT-UNION` only as
`CLOSED_INTERNAL_ANALYTIC_SCOPE`. The frozen type-zero strict, type-one
inclusive, and 50/50 leading formulas are
`ANALYTIC_TIE_COROLLARY_COMPLETE`.

They do not bind the current float-FFT production decoder. For each
`p in {11,19,31}`, both truth types, and all 56 leading competitors, a midpoint
query was built with equal integer Hamming distance to the two competing types.
Exact Hamming and direct dot-product scoring returned `336/336` ties. The
float-FFT dense scorer returned only `117/336` bit-exact ties; roundoff margins
up to `3.3306690738754696e-16` sometimes changed the winner. Therefore:

- `CURRENT_FLOAT_FFT_PRODUCTION_LINKAGE = FAILED`;
- the abstract event-union theorem is retained;
- a numerical-tie contract plus exact/direct/FFT boundary parity is required
  before production-decoder wording;
- finite class-error importance sampling and a nonuniform near-half bound
  remain separate validation tasks.

### JO1/A1/G2 boundary

- `PRW-JO1-STACKING-MECHANISM = CLOSED_BY_FLAT_IDENTITY`. All 38,610 checked
  schedule/representative/competitor comparisons have identical summed-stack
  and flattened Hamming distance. The `(1,5)` shell shaping is real but tied by
  15 schedules and matched by complementary/random controls. Only ordinary
  constrained-code comparison against missing known-design and unrestricted
  matched-work controls remains optional.
- `PRW-A1 = CONDITIONAL_OPEN`. Ten distinct fixed-label fingerprints and all
  45 action-pair crossovers are validated. Equal radial distance multisets show
  truth-local competitor permutations, but do not prove one global
  prior-preserving state permutation or channel isometry. Full action-channel
  conjugacy and real semantic action availability/cost are the next gate.
- `PRW-G2 = NECESSARY_INTERACTIONS_VALIDATED / STATIC_ADVANTAGE_FALSE`.
  Pair, strict three-way, and query-interaction residuals survive at `p=7` and
  bounded `p=11` unit checks. Exact flat and factorized decisions agree.
  `PRW-G2H` remains conditional only because
  `minimal_auxiliary_cost_proved=false`; auxiliary minimality and a concrete
  orbit-specific representation/decoder frontier precede Stage 4.
- `PRW-G2A` remains blocked. Do not substitute ordinary message scheduling for
  new-evidence acquisition; that is `PRW-G2S`.

### Authoritative remaining queue

0. Preserve the exact untracked source, tests, protocols, and four immutable
   JSON artifacts in Git and publish the branch.
1. Resolve the float-FFT numerical-tie contract and boundary parity.
2. Independently audit the T1R proof and known-construction equivalence if a
   paper is contemplated.
3. Run the A1 global conjugacy/semantic-cost gate.
4. Run the G2H auxiliary-minimality/named-candidate gate.
5. Optionally close the ordinary JO1 code controls.
6. Run noisy/nonlinear quotient, matched transcript, and common-channel
   carrier/control experiments.
7. Validate process-tree RSS and atomic checkpoint/kill/resume behavior.
8. Run the bounded flat real-corpus pilot before any relational mesh.

Do not rerun p=11 algebra alone, open G2A, scale primes/meshes, or revive
onion/CRT/harmonic claims before these discriminators. No current result
supports AI/RAG utility, lower training cost, production promotion, physical
dimensionality, or a new computing substrate.
