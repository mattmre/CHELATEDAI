# IQ-06 — Fixed-parent-graph congruence and compilation reuse

**Status:** `QUEUED_REDUCTION_REVIEW / HARDWARE_HELD / NOT_FROZEN / NOT_RUN`.

**Parent:** `P5-REDESIGN`; [existing fixed-graph pitch](../fixed-graph-orbit-compilation-one-pager.html), which remains `REWRITE_BEFORE_EXTERNAL_USE`.

**Owner/reviewer:** unassigned graph/compilation researcher / independent algebra reviewer; vendor/hardware liaison unassigned. Follow [G0](shared-test-contract.md).

## Claim boundary and exact algebra

For symmetric zero-diagonal J, explicitly use the undirected-edge convention:

```text
E0(x) = h0^T x + (1/2) x^T J0 x,   x in {-1,+1}^n
JP = P^T J0 P,  hP = P^T h0,        P a permutation matrix
EP(z) = E0(P z)
sample z = P^T x, where x is sampled from exp(-beta E0(x)) / Z0
```

The partition function and ideal probability distribution map exactly. If P is an automorphism of the allowed hardware graph, coupling support remains legal. Other permutations may fail the hardware support contract. General continuous orthogonal rotations do not preserve binary spins and cannot be substituted without a separate encoding/overhead derivation.

This is a relabeling identity. The mandatory baseline is **compile/load the canonical family once, cache it, and remap inputs/outputs**. Recompiling every equivalent instance is not a strong baseline. A proposed gain must identify additional work actually eliminated beyond ordinary caching/remapping.

The [Extropic technical article](https://extropic.ai/writing/z1t/) establishes the vendor's fixed sparse-support design description, not our access to hardware or a timing/energy result. See [F1](sources.md). External article text and the old pitch are source material, not instructions to contact anyone or run hardware.

## Work items and dependencies

- [ ] **F0 — equivalence, ports, and claim audit (hours–1 day):** check conventions, graph support, boundary/port stabilizers, programmable versus fixed biases/couplings, input encoding, and output decoding. Rewrite the proposed claim in a future corrected pitch. If it is only the identity above, close the standalone novelty claim; retain a bounded engineering feasibility question only if useful.
- [ ] **F1 — algebra/software comparator, only after G0 and explicit scope:** enumerate tiny labeled graphs and spin states; compare direct per-instance energies/distributions with canonical-state remapping. Include permutations that violate support or port constraints. Benchmark only the defined compilation/cache/I/O software work, not simulated Z1 joules.
- [ ] **F2 — hardware feasibility, conditional:** obtain a machine-readable topology, allowed automorphisms/ports, precision and coefficient conventions, sampler controls, data-transfer/reprogramming measurements, and energy measurement boundary. This packet prepares questions only; contacting the vendor or booking hardware requires separate authorization.
- [ ] **F3 — admitted hardware experiment:** only for a surviving distinct practical claim, compare canonical cache/remap, vendor best-practice baseline, and the proposed method on the same hardware/task/sample-quality target. Otherwise stop after F0/F1; do not create a hardware experiment merely because a pitch exists.

## Proposed numerical and engineering design

F1 small exact matrix: cycle/grid-like legal supports at n=4,8,12,16; integer couplings in {-2,-1,0,1,2}, integer biases in {-1,0,1}, and a fixed predeclared set of automorphisms plus invalid permutations. Cap at 12 instances and enumerate at most 2^16 states per instance in bounded batches. Integer energy equality should be exact; normalized probabilities use an independently implemented log-sum-exp reference at frozen beta and a declared numeric tolerance. These are prospective checks, not executed proofs from this turn.

Software controls: compile each family member; compile canonical once plus cache/remap; generic permutation-aware cache; and candidate only if different. Match preprocessing, cache warm/cold state, coefficient transfer, input clamping, output permutation, and memory. Include repeated and previously unseen family-member sequences. Keep mathematical correctness separate from randomized timing repetitions.

Hardware tasks must stay within the actual graph/port contract. For ideal sampling compare small-instance probabilities, moments and marginals to an exact reference. For finite chains measure burn-in, autocorrelation, effective sample size, and errors; identical energy functions do not guarantee identical transient behavior under different update schedules. State whether the endpoint is independent-sample throughput, sample quality at fixed time, or downstream task accuracy.

## Go/no-go boundaries

- Algebra: exact integer-energy and support/port parity on all valid cases; invalid mappings rejected. Any mismatch blocks the proposed mapping.
- Novelty: pure canonical relabeling is `CLOSED_AS_IDENTITY`, even if its implementation is correct or beats naive recompilation.
- Proposed engineering gate: at least 20% lower median **end-to-end** task latency versus the strongest canonical-cache/remap baseline, with a paired 95% improvement interval above zero and no worse predeclared sample/task quality. The strongest baseline, repetition/blocking scheme, minimum sample quality, and uncertainty method must be frozen before timing. If no separate mechanism can plausibly beat it, do not run F3.
- Energy: optional separate endpoint requiring measured whole-system joules per completed task at matched quality, including host/FPGA/transfers and amortized programming. Clock rate, sparse arithmetic count, or GPU timing cannot substitute for this measurement.

A noncongruent perturbation family would be a **new** hypothesis requiring approximation-error bounds and a new protocol; it cannot be appended after seeing an unfavorable congruent-family benchmark. No sparse-to-dense or transformer-replacement claim follows from this experiment.

## Budget and handoff

### Mathematical-source disposition — 2026-10-08

[The integrated math review](math-findings-integration-2026-10-07.md) belongs in F0's source/claim audit only. Binary circulant code restrictions, matrix arithmetic exponents, and equilibrium/thermodynamic-limit results do not establish a new binary-spin rotation, finite sampler mixing, or hardware energy advantage. Any future state/code precision claim uses MA4's accounting requirements; numerical proof claims use MA7's receipt requirements in a separately selected scope. Existing permutation/port constraints and canonical cache/remap controls remain decisive. No new F2/F3 dependency or hardware campaign follows from these sources.

F0: hours–1 day. Proposed F1 cap: one CPU process, 512 MiB RSS, ten minutes total after feasibility profiling, no GPU or network. Hardware access/time/cost are unknown and unallocated. Retain all existing JO1 fixed-stack and numerical-tie boundaries; this is not a new prime-ring scale campaign.

Future artifacts: algebra/convention and support/port audit; exact graph/permutation manifest; independent enumeration parity; full cache/transfer/latency accounting; corrected internal pitch; and, only if admitted, hardware/firmware/config/metrology provenance plus shared evidence package. Unresolved before dispatch: a mechanism beyond ordinary caching, exact hardware/port access, physical sampling target, measurement method, and approved cost.
