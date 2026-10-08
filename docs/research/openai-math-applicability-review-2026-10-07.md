# OpenAI math collection: applicability to CHELATEDAI

**Review date:** 2026-10-07. **Status:** `SOURCE_REVIEW / PROPOSED CONNECTIONS / NO PROOF OR EXPERIMENT REPLAY`.

**Source snapshot:** [openai/math at adc7f1241b42e322a6451854ab7e4b4c146bf78a](https://github.com/openai/math/tree/adc7f1241b42e322a6451854ab7e4b4c146bf78a). The companion [source receipt](openai-math-applicability-source-receipt-2026-10-07.json) records hashes of selected downloaded review inputs.

## Assessment

Several findings could strengthen mathematical contracts, expose impossible design targets, or improve evidence collection. None establishes that our current geometric, holographic, or virtual-expert mechanisms improve task quality or throughput. The most actionable connection is numerical-range control of a **fixed linear operator**, followed by finite-length codebook characterization and explicit memory/precision accounting.

Keep the current [portfolio order](portfolio-reorganization-and-big-winner-review-2026-09-30.md): validity reconciliation and executable controls, typed lineage authorization, IQ-01 composition-harm prediction, and IQ-04 temporal correction. These mathematical connections can support those lanes or bounded side investigations. They do not justify promoting IQ-03 or launching a new broad model campaign.

| Finding | Relevant pathway | Potential contribution | Recommendation |
| --- | --- | --- | --- |
| 325: complete Crouzeix inequality | IQ-01, fixed-linear slices of IQ-02/IQ-05, frozen virtual experts | Bounds amplification beyond eigenvalue checks; supports perturbation certificates | Highest priority mathematical connection; use the established bound first |
| 076: Littlewood flatness | RHPC / IQ-03 codebooks | Spectrum and autocorrelation objectives for finite codes | Characterize constructible codes before a learned experiment |
| 179: circulant Hadamard / Barker constraints | Binary cyclic holographic codes | Rules out a particular perfect-orthogonality target if the result is admitted | Record as a design constraint; approximate and complex codes remain candidates |
| 140: memory–sample tradeoffs | Local assimilation, streaming correction, compressed state | Makes precision and retained information explicit | Add a bit ledger; use its Gaussian task only as a scoped synthetic control |
| 266: computer-assisted MUB exclusion workflow | Audit discipline and prime-ring numerical evidence | Concrete source/binary/arithmetic/shard evidence contracts | Borrow the evidence structure, without adopting an unverified exclusion result |
| 130: sub-`n log n` exact Fourier circuits | Prime-ring FFT scoring | Asymptotic algebraic possibility | Park practical implementation; it does not repair floating-point tie parity |

## Review coverage and proof status

The pinned README describes 722 manuscripts in 372 families and explicitly distinguishes verification stages. I screened all 372 catalogue family entries, selected 22 candidate connections, read 13 family-specific Lean scope documents, and inspected selected manuscript sources from six families: 076, 130, 140, 179, 266, and 325. For eight challenges I also inspected the statement template, verification configuration, and selected solution endpoints: DirectCrouzeix, CompleteCrouzeix, AsymptoticallyMinimalLittlewood, LittlewoodFiniteFlatness, ExactFourier, CirculantHadamard, NoiselessRegression, and MUBSix.

This is an applicability and statement-scope review, not a review of all 722 proofs. The recursive GitHub tree response was truncated; it was not used as evidence of a complete dependency inventory. No local `lean` or `lake` command was available. I did not install dependencies, execute Comparator, audit every transitive dependency, replay numerical exclusions, run CHELATEDAI experiments, or open held-out fixtures.

The Comparator challenge files contain placeholder `sorry` declarations because they specify the comparison target. Those placeholders are not the released solution proof. The inspected JSON configurations point to separate solution modules and permit `propext`, `Quot.sound`, and `Classical.choice`. Reading those configurations and visible solution endpoints is weaker than successfully checking the complete pinned library and comparing its definitions.

Three statement mismatches matter immediately:

| Family | Manuscript/catalogue claim | Registered Lean scope at this snapshot |
| --- | --- | --- |
| 076 | Uniform two-sided ultraflat real-sign polynomials for all sufficiently large lengths | Asymptotically minimal upper maximum and one family's fixed-finite-`p` mean flatness; existential, without an effective rate or signing algorithm |
| 130 | An all-length exact Fourier algorithm below `n log n` | Arbitrarily small normalized circuit cost along an unbounded subsequence; no conditioning, bounded-coefficient, or bit-complexity guarantee |
| 266 | At most three mutually unbiased bases in complex dimension six | A weaker upper bound of five and supporting cancellation statements; no formalized exclusion of four arbitrary bases |

Before admitting a new result as a dependency, pin the intended statement and definitions, verify the applicable Comparator configuration and transitive proof, and separately establish that our operator satisfies its assumptions. A theorem can certify an invariant while leaving usefulness entirely empirical.

## 325: certify amplification for a fixed operator

The [registered scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/325.md) claims the sharp complete Crouzeix inequality, including finite complex matrices without a normality assumption. For a scalar polynomial, the relevant specialization is

\[
\|p(A)\|_2\le C\max_{z\in W(A)}|p(z)|,
\qquad W(A)=\{u^*Au:\|u\|_2=1\}.
\]

The new release claims `C=2`. An established usable fallback is `C=1+sqrt(2)` from [Crouzeix–Palencia, The numerical range as a spectral set](https://arxiv.org/abs/1702.00668). Start with that result while the new constant's proof is checked. A smaller certificate constant is not a speedup measurement.

This is relevant because eigenvalues alone can miss transient amplification. For example,

\[
A=\begin{pmatrix}1/2&10\\0&1/2\end{pmatrix}
\]

has spectral radius `1/2`, but sends the second unit vector to a vector of norm greater than ten. This example and the applications below are review derivations, not reported CHELATEDAI results.

For a frozen recurrence `h_next=A h+e`, bounds on `p(A)=A^t` control propagation of initial error and injected noise. If a certified enclosure of `W(A)` lies in `|z|<=r<1`, then `||A^t||<=C r^t`. An error bounded by `epsilon` at each step gives an accumulated bound `C epsilon (1-r^T)/(1-r)`, plus the initial-state term. The enclosure and arithmetic error must themselves be certified; sampled numerical-range points are not an outer enclosure. Many residual updates will fail this sufficient contraction test, which is an informative outcome.

**Mapping to our pathways:**

- **IQ-01:** introduce operator-norm or amplification descriptors only when available before the outcome and expressed in a proved common coordinate system. Compare their incremental prediction value against the existing M0/M1/M2 controls. A bound on hidden-state movement does not predict semantic collateral harm by itself.
- **IQ-02:** apply to a genuinely fixed linearized slice. Its scoped gate protects one coordinate complement, but the surrounding nonlinear update and cache do not acquire a convergence guarantee from this theorem.
- **IQ-05 / DAG:** fixed directed diffusion or communication matrices can be non-normal. Our symmetric-Laplacian candidate already has simpler ordinary controls; use numerical-range machinery only where the asymmetry creates a real gap.
- **Virtual MoE / LoRA:** freezing the mixture coefficients gives one linear residual operator. Polynomial reuse of that operator is in scope. Changing coefficients at every token produces a switching system, outside this single-operator argument.

Two individually stable matrices can form an unstable product: `A1=[[0,2],[0,0]]` and `A2=[[0,0],[2,0]]` are nilpotent, while `A2 A1=diag(0,4)`. Adaptive routing needs control of the full products, a common Lyapunov argument, or another appropriate stability condition. Nonlinear attention, normalization, changing inputs, and KV state must be included separately.

### A simpler certificate for the proposed virtual experts

For the [issue #106 proposal](https://github.com/mattmre/CHELATEDAI/issues/106#issuecomment-6029945796), use

\[
E_\alpha(h)=h+U\operatorname{diag}(\alpha)V^*h.
\]

At the **same input state**, elementary norm inequalities already prove

\[
\|E_\alpha(h)-E_{\hat\alpha}(h)\|_2
\le\|U\|_2\,\|\alpha-\hat\alpha\|_\infty\,\|V^*h\|_2.
\]

This does not require a new OpenAI theorem. It gives a precise target for speculative coefficient approximation or lower-precision assembly. Shared `V^*h` work is reusable only while the input is identical. If the proposal and reference diverge in hidden state, add a state-error propagation term; do not reuse this local inequality as a whole-trajectory guarantee.

If the downstream logits have a certified Lipschitz constant `L`, multiplying the bound by `L` controls their difference. A bound below half a certified reference top-two logit margin preserves that step's greedy argmax. It does not preserve an exact stochastic sampling distribution, prove semantic safety, or demonstrate net acceleration. Obtaining the reference coefficients, margin, enclosure, and verification may erase the compute saving. Learned confidence is not a certified error bound.

The most useful first question is whether a cheap certificate authorizes more useful work than an ordinary norm bound at the same verification cost. Report false certifications, conservative refusals, verification overhead, accepted work per second, and quality separately.

## 076 and 179: improve holographic code objectives and constrain impossible targets

The [Littlewood scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/076.md) is relevant to binary code spectra; the stronger claim is in the [October 5 ultraflatness manuscript](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Ultraflat-real-Littlewood-polynomials-October-5-2026/ultraflat-real-littlewood-polynomials.pdf). For signs `a_j` and

\[
P(z)=\sum_{j=0}^{N-1}a_jz^j,\qquad
C_k=\sum_{j=0}^{N-1-k}a_j a_{j+k},
\]

the standard Parseval identity gives

\[
\frac1{2\pi}\int_0^{2\pi}|P(e^{it})|^4dt
=N^2+2\sum_{k=1}^{N-1}C_k^2.
\]

Thus the fourth moment connects spectral flatness to total **aperiodic autocorrelation** sidelobe energy, and to binary merit factor `N^2/(2 sum C_k^2)`. This is a useful codebook characterization, not a compression or retrieval theorem.

The formalized existence statements provide no effective convergence rate or signing algorithm. They do not supply a usable code at our operating lengths. The stronger October 5 uniform-flatness manuscript is outside the stated formal scope. Extract a concrete finite construction before allocating implementation work; otherwise retain this as an objective rather than an available method.

For RHPC/IQ-03, separately measure maximum sidelobe, circular autocorrelation, cross-code coherence, route recovery, noise tolerance, and finite-precision behavior. Our complex circular products and role permutations do not inherit a real aperiodic result automatically. Low single-code sidelobes do not imply low interference among multiple codes. Order still requires role binding; vector superposition alone does not encode position.

The [circulant Hadamard scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/179.md) claims that real circulant sign Hadamard matrices exist only at orders one and four. If admitted, it rules out a larger perfectly orthogonal bank consisting of **all cyclic shifts of one binary sign sequence**. It does not rule out complex-phase Fourier bases, approximate binary codes, independent orthogonal atoms, or multiple codebanks. The formalized Barker consequence covers even lengths only.

These results are compatible: low asymptotic aperiodic sidelobe energy is different from exact zero periodic sidelobes at a finite length. Neither establishes learned-bank compression, decoding from deployment observations, or model utility. Preserve those three IQ-03 claims and all ordinary SVD/shared-basis/aligned/quantized controls at matched information and bytes.

## 140: make memory and precision part of the learning claim

The [Gaussian regression scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/140.md) includes a lower bound of order `d log(1/epsilon)` observations for a learner retaining at most `A d^2` bits, for fixed `A`, sufficiently large dimension, and success probability at least two thirds. The target is an unknown unit vector under a uniform prior; observations are independent noiseless Gaussian inner products, and the target error is angular.

The useful connection is accounting: a small number of real coordinates is not a bounded-memory claim until physical precision is specified. Count coefficients, U/V bases, rotations, residuals, indices, router state, optimizer state, caches, replay records, and precision metadata. Report both retained bytes and work needed to recover information.

The one-pass finite-memory model cannot silently replay discarded observations or retrieve them from disk. Pretrained information correlated with the target, adaptive/non-Gaussian observations, and retained external replay change the assumptions. This is not a general lower bound on RAG, LoRA, or the number of virtual experts.

A future synthetic Gaussian task could compare one-pass compressed state with retained-data replay at matched bits and sample counts. It would establish only a memory/precision control. It would not establish language-model correction or transfer utility, and need not become a separate flagship campaign.

## 266: borrow the evidence workflow

The [MUB scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/266.md) does not formally establish the headline upper bound of three. The [manuscript's execution section](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-maximum-number-of-mutually-unbiased-bases-in-dimension-six-September-24-2026/The-maximum-number-of-mutually-unbiased-bases-in-dimension-six-September-24-2026.pdf) nevertheless supplies useful patterns for our audit discipline: source and executable hashes, exact record formats, complete shard counts, provenance of every fallback input, and explicit arithmetic assumptions.

The described binary64 exclusions require controlled rounding, no fast-math reassociation or fused contraction, and specified elementary-operation semantics. A `pow` lowering assumption is checked against the actual binary; absence of a dynamic symbol alone is insufficient. Missing records and resource exhaustion leave a case unresolved rather than turning it into a negative certificate. I did not replay or independently authenticate the reported exclusion run.

For the prime-ring tie repair, require the exact implementation/build, the declared tie rule, complete fixture coverage, reference parity, and a receipt connecting them. The local portfolio records direct/exact scoring at `336/336` and float FFT parity at `117/336`; those are existing documented results, not rerun here. Better receipts alone do not repair the arithmetic. Dimension-six MUB bounds also do not impose a universal limit on our virtual-expert codebanks.

## 130: no practical FFT shortcut established

The [formal Fourier scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/130.md) counts exact complex scalar gates and allows arbitrary predetermined scalars. Its subsequential savings say nothing about finite precision, conditioning, memory traffic, or useful finite-length runtime.

The [all-length manuscript's construction](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/An-explicit-power-saving-for-the-exact-discrete-Fourier-transform-September-25-2026/main.pdf) uses an initial block parameter `m=10^6`, `2^71` arrays, and an exponent only about `2.1e-13` below one before additional logarithmic factors. These are asymptotic ingredients, not a feasible replacement benchmark. No practical crossover at our dimensions was established in this review.

Retain the exact/integer rescoring or explicit tolerance-and-canonical-tie contract proposed for prime-ring scoring. An exact algebraic DFT does not prove floating-point midpoint parity. Do not count this result as resolving the current FFT failure.

## Other screened connections

| Family | Possible connection | Why it is secondary |
| --- | --- | --- |
| 094: finite `L^p` embeddings | Representation compression | The selected new scope excludes `p=2`; distortion does not establish aligned coordinates, signed information, or semantics. Ordinary Euclidean projection is still a baseline. |
| 098: doubling compact sets without finite-dimensional bi-Lipschitz embeddings | Limits on universal lossless compression | Infinite-dimensional compact-set obstruction; it does not defeat a finite corpus or an already finite-dimensional bank. |
| 107: matrix multiplication exponent | Batched adapter work | Asymptotic arithmetic complexity is not a finite GPU kernel or a bytes-moved advantage. |
| 139: log-concave sampling | Stochastic router controls | Strong convexity, supplied minimizer, and exact oracle assumptions; the query model leaves arithmetic/bit costs unrestricted. |
| 229: tree reconstruction | Synthetic evidence propagation | Selected Lean scope gives reconstruction above `b lambda^2>1` for specified symmetric three-state trees. It does not establish non-reconstruction below that threshold or truth on copy-dependent evidence DAGs. |
| 251: group representation stability | Compatibility/coordinate changes | The strong-stability companion is outside the registered scope; our adapter maps have not been shown to satisfy its group-representation assumptions. |
| 328: fixed points in reflexive spaces | Recurrent correction | Fixed-point existence is not convergence, a rate, uniqueness, or correctness. Finite-dimensional ordinary tools already apply to much narrower models. |
| 089, 221, 234, 265, 281 | Graph metrics, spin systems, tensor networks, QAOA | Screened at catalogue level only. Their graph/topology, quantum, equilibrium, or thermodynamic-limit assumptions do not prove finite learned-adapter performance or hardware mixing time. |

For **RAG**, these findings mainly support representation/error contracts, code diagnostics, and memory controls; none proves semantic retrieval improvement. For **DAG**, operator amplification and carefully scoped tree probes are plausible tools; temporal correction still needs the existing bitemporal/copy-aware baselines. For **LoRA and virtual MoE**, local coefficient-error bounds are directly derivable, while nonlinear/switching trajectories and net speculative acceleration remain separate questions.

## Proposed follow-up, in dependency order

**Integration update, 2026-10-08:** the [MA0–MA7 packet](innovation-test-queue-2026-09-12/math-findings-integration-2026-10-07.md) and [24-case register](innovation-test-queue-2026-09-12/math-test-cases-2026-10-07.csv) now attach these follow-ups to current IQ packets, sources, shared admission requirements, portfolio schedule, issue #106's local plan, and the next-session handoff. These additions specify tests; they do not claim implemented harnesses or executed experiments.

1. **Write the operator and admissible claim.** Identify one fixed-linear update or same-input virtual-expert comparison; record lineage, metric, normalization, precision, coefficient availability, and cache state. Start with the elementary perturbation bound and established Crouzeix–Palencia constant. Refuse a semantic or whole-trajectory guarantee unless separately justified.
2. **Check whether certification can be economical.** Compare a standard norm bound with numerical-range control on DEVELOPMENT-only constructed operators, including non-normal and switching counterexamples. Count enclosure/verification costs and refuse certificates outside assumptions. A useful result must improve admissible work or harm prediction after those costs.
3. **Extract a finite code before testing a new codebook.** Require a construction at our lengths, then compare ordinary random, Fourier/phase, and existing prime-ring codes at matched bytes/work. Stop if the existence proof supplies no usable construction. Keep code metrics, route decoding, learned compression, and utility as separate endpoints.
4. **Adopt a complete bit ledger and numerical receipt.** Attach them to the relevant future protocol, especially virtual-expert speculation and the bounded prime-ring repair. Preserve all current gates and ordinary baselines.
5. **Admit a new theorem only after checking it.** If a tighter constant or a negative design result becomes necessary, reproduce the pinned Comparator check and inspect its definitions/assumptions. The source review itself has not discharged that dependency.

These are proposed additions to existing research planning. This review creates no new experimental result and makes no change to the frozen protocols, current block/debt state, or GitHub issue.
