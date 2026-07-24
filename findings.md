# Findings & Decisions

## 2026-07-24 RB-7 Preregistration and Resource Screen
- The next bounded Fourier discriminator is `PRW-H1Q`: at `p=11`, compare every
  matched two-bin subset over a declared quantizer-origin grid, then report the
  complete rank distribution, origin spread, and an origin-averaged diagnostic.
  The evaluation may not choose a favorable subset or origin after seeing the
  outcomes. A finite origin average is a grid-robustness diagnostic, not a
  claim of stochastic dither unless the dither distribution and seed are
  independently fixed.
- There are five nonconjugate bins at `p=11`, hence ten two-bin subsets.
  Four quantized origins plus one unquantized baseline produce 50 candidate
  cells before controls. Calling the existing exact comparator independently
  for all 50 would duplicate the same 2,048 BSC patterns and is not approved
  until an aggregate bytes/work/deadline preflight passes. A shared streamed
  enumeration is the preferred formulation.
- A read-only estimator probe makes that refusal concrete. Each independent
  cell estimates only 2,408 model-level peak bytes, but 2,820,096 work units,
  already above the frozen 2,000,000 default. The naïve 50-cell grid totals
  141,004,800 work units, above even the 50,000,000 hard ceiling. The bank
  itself is tiny (541 estimated peak bytes and 44 work units), so memory is not
  the problem; duplicated decoding work is. No exact pattern loop ran during
  this estimate.
- A dependency-free group-action check found that the ten unordered `p=11`
  two-bin subsets form two multiplicative-relabeling orbits of five:
  `{(1,2),(1,5),(2,4),(3,4),(3,5)}` and
  `{(1,3),(1,4),(2,3),(2,5),(4,5)}`. The batch should report within-orbit
  spread as a symmetry audit. This does not assume phase quantization preserves
  the unquantized relabeling symmetry; breaking it by origin is itself a useful
  artifact diagnostic.
- Hostile analysis shows why a common four-origin grid is not a
  symmetry-closed dither. A multiplier can map one selected frequency through a
  negative representative (complex conjugation) while leaving the other
  positive, which maps per-bin origins as `(alpha_1, alpha_2) ->
  (+/- alpha_1, +/- alpha_2)`. The common-origin diagonal cannot represent all
  such maps. The current upper-tie rounding rule can also break conjugation at
  exact half-step boundaries. Consequently, the four common origins test
  nuisance sensitivity only. A stronger invariant screen needs a declared
  per-bin origin product grid, explicit boundary-hit accounting, or continuous
  origin integration, all with a new aggregate resource preflight.
- The final cached shared-stream preflight estimates 54,432 model-level bytes
  and 17,951,240 work units for 2,048 patterns, 50 conditions, and 102,400
  cached condition evaluations under one 25-second deadline. It computes one
  template FFT, 2,048 query FFTs, and no public reference-decoder calls; focused
  tests match the reference scores, winners, ties, margins, and abstentions.
  The straightforward symmetry-closed `A4^2` extension would require 170
  conditions, 348,160 cached evaluations, and 59,734,280 work units, exceeding
  both the local 25,000,000 and upstream 50,000,000 hard ceilings, so that route
  is explicitly refused.
- The root-owned final exact run completed in 22.359 seconds beneath the
  unchanged deadline. Unquantized accuracy is constant to floating precision
  within each predicted orbit: orbit A mean `0.1202012759` with spread
  `4.16e-17`, orbit B mean `0.1231158929` with spread `6.94e-17`; A minus B is
  `-0.0029146170`. The between-orbit gap is ordinary ratio-class geometry, not
  a special pair or resonance.
- Across the four common quantizer origins, 30 of 45 pairwise orderings reverse
  and no pair strictly dominates every other pair at every origin. Maximum
  per-pair origin spread is `0.0048568741`; maximum absolute grid-average
  change from the corresponding unquantized value is `0.0019161556`.
  Origin `0.5` has up to `0.0306274866` boundary-near probability mass, while
  `0.0009287012` zero-spectrum mass is separated and correctly abstains.
  Therefore the bounded result supports within-orbit unquantized equivariance
  and quantizer-origin sensitivity, not selected-frequency advantage.
- The proof-first `PRW-C1A` screen is narrower than the earlier geometric
  language: on the nonzero field elements, write `x = g^n` for a primitive root
  `g`. Multiplication by `g^a` is then the additive shift
  `n -> n + a (mod p-1)`; CRT coordinates merely relabel that shift
  componentwise. For the mathematical Legendre character on nonzero elements,
  the same action depends only on exponent parity and therefore reduces to a
  polarity bit. The zero element is a separate fixed point.
- The repository carrier uses a deliberately bipolar convention with coordinate
  zero set to `+1`, whereas the number-theoretic character has value zero there.
  Therefore an odd-exponent pivot flips every nonzero carrier coordinate while
  leaving coordinate zero at `+1`: the exact full-array reduction is a polarity
  bit plus one fixed-coordinate exception, not a literal global sign.
- `PRW-C1A` can establish an exact redundancy or degeneracy result, but CRT
  factorization alone does not establish compression, faster retrieval, or a
  new matching primitive. Any surviving advantage must beat explicit flat
  permutation and polarity-only controls at matched search opportunity.
- Allowing one independently chosen shift in each CRT factor still gives only
  `2 * 5 * 7 * 67 = 4,690` tuples, in bijection with one ordinary exponent
  shift. It does not create an extra address dimension. Giving each of `L`
  layers an independent pivot does create `4690^L` states, but that expansion
  must be charged as `L * log2(4690)` control bits and a correspondingly larger
  search space; it is not free capacity from the factorization.
- A further proof-first reduction covers pure onion layering. Additive
  rotations `T_b` and multiplicative pivots `M_a` generate the ordinary affine
  group, with `M_a T_b M_a^-1 = T_(a*b)`. Every word containing only those
  coordinate permutations collapses to one map `x -> a*x+b`, so ordering more
  such layers cannot by itself create a new high-order memory. A mechanism can
  escape this reduction only by inserting a declared nonlinear or conditional
  replacement; that replacement must then be tested against balanced random
  and conjugated sector-mask controls rather than attributing its effect to
  CRT.
- The effective multiplicative orbit is payload-dependent. For the punctured
  Legendre character, the 2,345 quadratic-residue pivots form its stabilizer
  and the orbit has only two states. Under the repository's bipolar
  zero-coordinate convention those are still just the original array and the
  common nonzero-coordinate complement with zero fixed.
- For future replacement tests, CRT-axis shifts conjugate a sector replacement
  into the same replacement at a shifted sector label. Disjoint overwrites
  commute; overlapping overwrites can show ordinary last-write-wins effects.
  Any claimed layer-order benefit must therefore match mask size, mask overlap,
  condition bits, and random balanced/coset partitions, and must survive a
  change of primitive-root address convention.
- The exact `PRW-C1A` screen now passes a root-owned direct run in 0.144 seconds
  with 3,310,582 estimated Python-object bytes and 2,176,160 work units
  (neither is measured process RSS). The CRT tuple count and shared pivot count
  are both exactly 4,690; the factorization creates no additional states.
- Eight independently assigned layers have `4690^8 =
  234089935364620159344100000000` possible control tuples and require about
  97.563 control bits. Those tuples were counted, not enumerated, and no
  payload-level independence or distinguishability was established. The
  Legendre stabilizer/orbit is
  `2345/2`, and the affine rotation-plus-pivot family has the ordinary upper
  bound `4691 * 4690 = 22,000,790`.
- Four algebraic kill criteria trigger: flat exponent-permutation relabeling,
  no extra CRT states, Legendre parity plus the fixed-zero exception, and
  affine collapse of rotation/pivot words. The systems control does not
  trigger because learned-payload utility and implementation cost were not
  tested. This closes new-group/new-CRT/new-onion-order interpretations, not a
  possible learned-layout engineering benefit.
- The `PRW-C1B` p=7 conditional-replacement screen also ran directly within a
  99,072-byte/29,808-work estimate and one-second deadline. Its representative
  four-operation program reduces to `x -> 3*x+6` followed by one exact
  last-write-wins overwrite; the two effective masks each have size three and
  overlap once. All 128 binary payloads confirm the normal form. Disjoint
  overwrites commute; overlapping overwrites show only declared
  last-write-wins order. The seeded random control matches mask
  size/overlap and overwrite conjugacy, but does not claim full affine-program
  or cost/utility equivalence.
- This first replacement screen uses fixed address masks and constant
  replacement bits. It does not test predicates or replacement functions that
  depend on payload, query, or layer state, so it narrows only the static-mask
  lane and does not kill the broader runtime-conditional hypothesis.
- The final current-tree validation passes 16/16 quantizer-null tests, 23/23
  CRT/algebra tests, and 14/14 conditional-replacement tests; the three suites
  pass together at 53/53. The CRT suite includes hostile numeric-subclass and
  greater-than-64-bit input paths so arbitrary big integers are refused before
  modular arithmetic or work allocation. Scoped Ruff, read-only AST parsing,
  and whitespace validation are separate correctness checks, not scientific
  replication.
- Folding those 53 checks into the prior eight-module regression set gives
  230/230 passing tests in 31.561 seconds on the final current tree. This is a
  low-memory deterministic regression run; it does not repeat the exhaustive
  `p=11` grid or constitute powered evidence.
- The accepted `p=11` run is one exact tiny synthetic grid, not a statistical
  replicate, continuous-dither study, corpus run, campaign, or retained
  evidence artifact. The `p=4691` result above is an algebraic table/check, not
  a transform, retrieval, or timing benchmark.

## 2026-07-24 Resource-Bounded Return
- Live recheck before new edits found
  `codex/prime-ring-onion-method-dev` at
  `08e3c2aeeb5c645a3c95f08a14ac101b226f8c9d`, clean and eight commits ahead
  of `origin/main`. This is local branch state only; no publication claim is
  implied.
- The previous checkpoint already implemented RB-1 matched controls, RB-2
  finite intersection diagnostics, RB-3 correctness-first Rader transforms,
  and typed remaining hypotheses. It did not establish novelty, systems
  superiority, production value, or a scientific result.
- The highest-information safe additions are currently:
  1. fail-closed aggregation over every preregistered lower-rate raw-sanity cell
     rather than only the pooled `q=0.45` cell;
  2. a tiny graph-coupled cyclic-fiber test with a degree/edge-matched shuffled
     graph control;
  3. a tiny multi-frequency phase-signature test with matched energy and search
     opportunity controls.
- These additions can test whether apparent gains come from cross-node
  coherence or frequency diversity, rather than extra energy, extra
  hypotheses, or a favorable graph. They remain synthetic mechanism tests.
- Repository brutal-honesty rules require runtime evidence for completion
  claims. The current slice therefore distinguishes bounded unit/property
  evidence from a powered campaign or real retrieval result.
- The default WindowsApps PowerShell launcher again failed with access denied
  after restart. The stable bundled Node runtime is being used for bounded
  inspection and command execution; the failed launcher path will not be
  retried.
- The live file map contains no existing `prime_ring_graph_fibers.py` or
  `prime_ring_multifrequency.py`. The relevant reusable primitives are the
  validated carriers/rotations/correlations in `prime_ring_waypoint.py`, the
  immutable finite-analysis caps in `prime_ring_intersection.py`, and the
  matched-control aggregation in `run_prime_ring_waypoint_experiment.py`.
- The formal ledger already defines `PRW-G1` as finite-group synchronization on
  a graph and `PRW-H1` as normalized discrete-Fourier phase synchronization.
  New code must preserve those ordinary mathematical meanings and may not turn
  the visual "3D", "gravity", or "resonance" language into a physical claim.
- A fresh primary-source collision check substantially narrows both ideas:
  - Singer's 2011 angular-synchronization work already estimates node phases
    from noisy graph-relative offsets, including consistency over general
    compact groups.
  - Perry, Wein, Bandeira, and Moitra (2016) explicitly treat compact-group
    synchronization with multiple Fourier/representation channels.
  - Gao and Zhao (2019) explicitly call the construction
    "Multi-Frequency Phase Synchronization."
  - Current sparse-graph work also uses connection/edge rotations and spanning
    structures for angular synchronization.
- Therefore neither graph-coupled cyclic fibers nor multi-frequency phase
  synchronization is novel in isolation. A surviving contribution would have
  to be a narrower interaction—such as the exact OPPW/type-mask construction,
  a proved resource/error tradeoff, or a held-out retrieval/correction
  advantage under matched controls.
- New bounded analyzers should copy the established finite-analysis safety
  pattern: immutable hard ceilings, caller budgets that may only lower those
  ceilings, shape/work preflight before allocation, a monotonic deadline, and
  `unittest`-only deterministic oracles compatible with Python 3.9.
- The protocol deliberately uses only `q=0.45` for the pooled false-unlock
  threshold cell, while separately requiring 100% raw type recovery at every
  rate through `0.45`. The lower-rate fix must not create four primary
  thresholds or reuse `REPORT` outcomes for selection; it should be a separate
  construction/noise-model sanity aggregation over raw decoder outputs.
- `PRW-H1` has a stronger null than initially stated. For prime `p`, any one
  nonzero Fourier bin is invertible modulo `p` and already distinguishes every
  clean cyclic shift. Multiple bins cannot add noiseless address capacity; they
  can only add redundancy/robustness under corruption. For real signals, bins
  `k` and `p-k` are conjugates rather than independent observations.
- The new bounded Fourier harness therefore:
  - restricts selected bins to `1..(p-1)/2`;
  - searches exactly `type_count * p` shared type/shift states for every
    condition;
  - normalizes phase features to unit energy;
  - includes same-count random bins, repeated-single-bin, single-bin,
    magnitude-only, randomized-phase, and time-domain controls;
  - supports explicit 1-16 bit phase quantization, including an 8-bit check;
  - streams exact BSC patterns only for tiny state spaces.
- The first PRW-H1 suite passed 17 deterministic tests in 0.525 seconds. This
  proves bounded implementation properties only; no frequency-diversity
  advantage was established.
- A tiny exact `p=7`, one-type, one-node BSC sweep enumerated all 128 corruption
  patterns at `q={0,.20,.35,.45}` with only 928 estimated peak bytes and 83,584
  estimated work units per condition set:
  - unquantized bins `(1,2)` and the same-count random control `(1,3)` had
    identical exact accuracy at every rate;
  - two distinct bins beat a repeated copy of one bin by `0.26112`, `0.109243`,
    and `0.026715` probability at `q=.20`, `.35`, and `.45`;
  - this is evidence for generic redundancy in this toy channel, not for a
    special frequency selection, harmonic resonance, or new address capacity;
  - 8-bit quantization introduced small bin-set differences (at most about
    `0.00834` here), which is more plausibly a quantization/bin interaction than
    a selected-bin result and requires all-bin/all-seed controls before any
    interpretation;
  - time-domain correlation used a different coefficient budget, so its toy
    accuracy is diagnostic rather than a matched superiority comparison.
- The lower-rate closure implementation is structurally separate from threshold
  fitting. It selects exactly one retained completed cell per frozen
  `(seed,q)` at the canonical 4691/Legendre/8-layer/typed16/payload-0.25
  configuration, requires the expected raw REPORT group count and accuracy
  exactly `1.0`, rejects missing/duplicate IDs and promotion-enabled records,
  and feeds its fail-closed result into the dense and all-mandatory control
  gates. Hostile review then showed that this first verifier trusted mutable
  summaries: fabricated noncanonical IDs, coerced numeric types, near-matching
  rates, and REPORT-derived threshold metadata could still close the gate.
  Provenance hardening is therefore required before the implementation itself
  can be called fail closed. The powered campaign remains unexecuted, so no
  empirical gate is closed.
- Fresh hostile review confirmed the PRW-H1 DFT sign/roll law across 2,100
  noiseless cases, but found preflight, condition-cap, Python 3.9, tie-margin,
  and accounting/control defects. Reconditioning now:
  - refuses malformed shapes and caller-tightened byte/condition limits before
    control construction or content scans;
  - uses a Python 3.9-compatible population count;
  - forces tied winners to margin zero;
  - distinguishes logical coefficient slots, stored scalars, energy, and
    arithmetic matching;
  - exposes proposed/random-bin collisions; and
  - includes the previously missing all-nonconjugate-frequency control.
- The reconditioned PRW-H1 suite now passes 37/37 tests in 2.669 seconds with
  clean focused Ruff. Final hostile review reports no P0, P1, or P2 findings.
- The second PRW-H1 reconditioning also gives the whole multi-condition
  comparison one aggregate work/deadline budget, guards the exported random-bin
  helper before `np.arange`, renames "independent" bins to the accurate
  "nonconjugate" term, counts bank-build magnitude temporaries, and states that
  phase-bit tests still use float64/complex128 arithmetic with no packed
  storage or speed claim. The focused suite remains 21/21 green in 0.523
  seconds with clean Ruff.
- A stronger artifact-free PRW-H1 falsifier exhaustively covered all three
  two-bin subsets of the nonconjugate `p=7` bins, all 128 BSC patterns,
  `q={.20,.35,.45}`, and 8-bit quantizer-grid origins
  `{0,.25,.5,.75}`:
  - all three unquantized two-bin subsets have exactly the same accuracy at
    each rate;
  - the apparent 8-bit bin ranking changes when only the quantizer origin
    changes;
  - the largest within-subset origin spreads are `0.0032768`, `0.0076534555`,
    and `0.0083385070` at the three rates;
  - the proposed two-bin subset trails the three-bin all-nonconjugate
    diagnostic in 29 of 36 quantized cells, although that diagnostic has a
    larger coefficient budget and is not a matched superiority comparator;
  - each run is guarded at no more than 1,288 estimated model-level bytes and
    101,504 work units.
  This kills a selected-frequency or resonance interpretation for the toy
  result. The distinct-bin gain over repeated copies remains ordinary
  redundancy, and the small 8-bit differences are grid-origin artifacts or
  interactions until stronger evidence says otherwise.
- The bounded `PRW-G1` graph analyzer now includes that explicit
  one-global-phase-plus-fixed-offset comparator. In the helpful `p=7`
  triangle, its assignment, score, tie count, planted margin, and planted rank
  are exactly identical to the graph-coupled decoder; every graph-minus-global
  effect is zero. Fifteen tests, focused Ruff, and an independent exhaustive
  check of all 511 nonempty labeled simple graphs on three nodes pass. The
  earlier margin gain over independent/shuffled controls is fully explained by
  restricting the decoder to the fixed-offset global-phase family, not by a
  new graph-memory mechanism.
- Warm model-array allocation checks at the largest permitted PRW-H1 shape
  (`p=31`, 32 types, 12 nodes, 15 bins) now remain below their declared
  estimates: bank construction measured 532,903 bytes versus a 579,871-byte
  estimate, one decode measured 755,648 versus 790,448, and the eight-control
  comparison measured 826,096 versus 846,000. These are `tracemalloc`
  diagnostics for Python/NumPy allocations, not process RSS or a production
  memory benchmark.
- The lower-rate raw-sanity verifier now binds every accepted cell to a
  canonical cell ID and campaign/stream-contract digest, requires exact Python
  scalar types and exact frozen coordinates, enforces SELECT-only thresholds
  with zero REPORT rows consumed, and recomputes row counts, group/order
  digests, row digests, accuracy, and canonical trial truth from planted IID
  REPORT lineage. An independent adversarial probe that changed both the
  retained true and predicted types, then recomputed the row digest, initially
  exposed one remaining gap; canonical `_trial_truth` recomputation now rejects
  it with `raw_provenance_true_type_not_canonical`.
- A second hostile attack showed that those digests still self-attested: after
  rewriting a failed prediction, correctness, counts, accuracy, and row digest,
  an attacker could reconstruct every old seal from serialized fields and turn
  the gate into `COMPLETE_PASS`. The accepted design now creates one per-run
  authority before campaign traversal, uses a random 32-byte HMAC-SHA256 key to
  authenticate each cell/campaign/carrier/raw-provenance/count/accuracy tuple,
  and never serializes the authority ID, key, or seals.
- Replaying the exact reconstruction attack with an unkeyed digest now yields
  `INCOMPLETE_FAIL_CLOSED`,
  `live_source_seal_authentication_failed`, `contract_complete=False`, and
  `gate_pass=False`. A replacement authority, missing or misbound seals,
  mutated/missing carrier provenance, and a SELECT/REPORT tag collision also
  fail closed. Independent hostile re-review reports no P0, P1, or P2 under the
  explicit serialized-input-tampering boundary; arbitrary same-process code
  execution is not claimed as a protected boundary.
- The 12 seals are conservatively budgeted at 24,576 bytes and the authority at
  2,048 bytes, for 26,624 estimated live-attestation bytes. This estimate is
  scoped to the extra seal/authority objects, not process RSS, cells, or the
  serialized raw-provenance rows already retained for auditability.
- The final complete bounded runner suite passes 29/29 tests, and the combined
  eight-module research suite passes 177/177 tests in 28.233 seconds. Scoped
  Ruff, read-only AST parsing, and whitespace validation also pass. These use
  only tiny smoke fixtures, exact small-prime enumeration, and fabricated
  in-memory provenance records; they do not invoke `full_config()`, the
  11,520-cell campaign, 4091/4691 timing, real retrieval, or retained evidence
  generation. This validates gate behavior, not the raw-sanity scientific
  criterion itself.
- The best next low-risk discriminator is not more `p=7` parameter searching.
  It is a preregistered, resource-estimated `p=11` quantizer-origin/dither null
  followed by an algebraic `PRW-C1` redundancy screen. Neither may inherit
  support from the current negative graph/Fourier results.

## 2026-07-23 Prime-Ring Onion-Lattice Findings
- Live `origin/main` already contains thirteen lattice commits after the
  recovered brain-file-map branch:
  - H6 temperature schedule (`#280`);
  - H5 steering-post bank, post-bank construction/runtime/conditions, and
    head-to-head driver (`#279`, `#281`-`#283`, `#290`);
  - rollback-backed steering control and promotion routing (`#284`-`#286`);
  - H3 teacher-supervised correction and C3b wiring (`#287`-`#288`);
  - H4 compound-cycle control (`#291`).
- Those commits provide implementation surfaces and unit coverage, but the
  current audit has not yet found durable real H5 head-to-head wins or evidence
  that H4 compounding improves retrieval. Existing test prose explicitly says
  the H5 win question belongs to the campaign rather than the wiring test.
- The local ignored waypoint-research folder contains newer July planning,
  campaign evidence, paper drafts, novelty assessments, harmonic-invariance
  preregistration, and lattice endgame documents. These are research context,
  not automatically durable repository evidence.
- Canonical BCC-1 recovery is v9 on `codex/semantic-cache-coherence-h1` at
  `11d548bd`, with pack, manifest, result, protocol, prior-art manifest, runner,
  and 171-test recovery evidence. The v8 preformat directory is superseded and
  must not be used as the implementation base.
- The same branch contains later commit `2b0fa044`, whose subject says the H2
  reruns closed both `CD-H1-01` and `CD-A2-01`. Exact artifacts and flag
  synchronization still require live audit before accepting that disposition.
- July lattice endgame records state that the corrector question was answered
  negatively and the powered estimator path also failed: annealing, living,
  and bounded correctors matched or lost to trivial controls. This is a
  disconfirming result, not a foundation for the prime-ring hypothesis.
- The live H5 implementation explains at least one negative structurally:
  `C5` and static `C5s` construct the same deterministic bounded bank, route
  from the same pre-drift snapshot, and `C5` deterministically rebuilds pruned
  posts from the same anchors/seed before applying them. They should tie, while
  the existing strict gate requires `C5 > C5s`. That gate is unwinnable without
  a genuinely state-changing router or bank update.
- Hypothesis naming is already ambiguous: durable diagnostics use `H3` for the
  confirmed C2 oracle-re-embed result, while later PRs reuse `H3` for unrun C3b
  teacher distillation. New work must use a separate `PRW-*` registry and may
  not inherit any bare `H3` status.
- The merged H3-H6 additions are not complete campaign surfaces:
  - run-level correction-norm aggregation omits C3b and C5 variants;
  - bounded runtime C3b discards learning diagnostics and has no NDCG campaign;
  - H6 controls only post-bank pruning, not the roadmap's shared
    sedimentation/updater/ES schedule or high-temperature perturbation;
  - H4/H5 campaign knobs are absent from the CLI and default campaign;
  - their deferred GPU campaigns were not carried into `docs/next-session.md`.
- Despite their `feat(lattice)` labels, the merged H3-H6 code contains no
  mathematical lattice, DAG, sheaf, holonomy, ring, or 4691 mechanism. H5 is a
  flat centroid-to-correction dictionary. The prime/onion mechanism therefore
  has not been tested by those commits.
- `codex/crsv-onion-method-dev` at `59378814` is an unmerged local branch whose
  name and recovery placement indicate likely overlap with conditional
  replacement/subdomain variance and onion-layer work. Audit it before adding
  a second implementation.
- The block-flag script mechanically passes `CLEAR` but reports two open carried
  debts. Their June dates are older than the rulebook's one-cycle/five-day
  limit, so the declared flag may be stale. New feature implementation remains
  paused while exact debt disposition is reconstructed.
- After fast-forwarding to the recovered canonical baseline, commit `2b0fa044`
  supplies the real swap-path campaign evidence and closes both rows.
  `scripts/check_block_flag.py` now reports `CLEAR`, zero carried-debt rows, and
  `PASS`; isolated METHOD_DEV feature work is permitted.
- The campaign now runs on isolated branch
  `codex/prime-ring-onion-method-dev` at current `origin/main`; the user's
  `feat/brain-file-map-b0-b1` commit remains preserved on its original branch.
- `4691` is prime, `4691 mod 4 = 3`, and
  `4691 - 1 = 2 * 5 * 7 * 67`.
- The length-4691 bipolar Legendre sequence was checked exhaustively in the
  current session: zero-shift correlation is `4691`; every one of the 4,690
  nonzero additive shifts has correlation `-1`.
- `4091` is also prime and `3 mod 4`, has the same Legendre property, and is a
  mandatory hardware-friendly control.
- Post-restart full-size exact verification passed for both carriers:
  - `4091`: prime and all 4,090 nonzero autocorrelations equal `-1`;
  - `4691`: prime and all 4,690 nonzero autocorrelations equal `-1`;
  - `4690` factors exactly as `(2, 5, 7, 67)`;
  - `2` is a primitive root modulo `4691`.
- A full-size eight-layer quotient probe at `p=4691` produced:
  - correct shared-shift score `1.0`;
  - a wrong signature with at most one aligned layer scored
    `0.12481347260712003`;
  - observed margin `0.8751865273928799`;
  - conservative analytic wrong-score bound `0.12518652739287997`.
  This validates the noiseless separation construction, not semantic utility.
- There are two distinct singleton/binary ideas:
  - Rader separates `{0}` from the 4,690 nonzero field coordinates. This is one
    anchor scalar per channel, not inherently one bit.
  - `Z_4690 ~= Z_2 x Z_5 x Z_7 x Z_67`; the `Z_2` factor supplies one
    multiplicative polarity character per ring.
- Eight binary rings can therefore form an eight-bit cross-layer orientation
  mask with 256 configurations. Free search over all masks also creates a
  256-way false-match opportunity, so masks require type/provenance constraints
  or multiple-comparison-corrected thresholds.
- Additive phase recovery and multiplicative payload pivots are different:
  - additive shifts create the 4,691 ideal Legendre alignments;
  - multiplicative action on a pure Legendre carrier yields only the original
    pattern or its sign inverse;
  - CRT pivots must be tested on arbitrary/learned payloads.
- With primitive root `2`, a nonzero exponent has CRT coordinates
  `(n mod 2, n mod 5, n mod 7, n mod 67)`. Exact reconstruction is
  `n = 2345*n2 + 1876*n5 + 2010*n7 + 3150*n67 (mod 4690)`.
- Eight binary 4,691-bit planes occupy exactly 4,691 bytes only as a
  theoretical bit-packed representation. The current semantic payload API uses
  `float64`: one `8 x 4691` payload occupies 300,224 array bytes. Packed binary
  accounting cannot be applied to those real-valued payloads.
- The strongest existing-prior-art collision is Residue Hyperdimensional
  Computing plus resonator decoding. HRR/FHRR, Legendre/Paley codes,
  product-key memory, cellular sheaves, and low-rank adapters cover other
  individual components.
- The current defensible research target is the integration and causal
  interaction of typed lattice consistency, constrained cross-layer phase,
  queue-conditioned gating, and reversible local correction. It is not yet a
  novel theory.
- The diagonal-quotient phase signature has an exact known-code isomorphism.
  Map phase vector `phi` to a one-pulse-per-wavelength 2D binary array
  `B_phi(layer, position) = 1[position == phi[layer]]`. A global phase offset
  is a common cyclic column shift, and the PRW overlap `kappa` is exactly the
  shifted cross-correlation of two such arrays. This is the standard orbit and
  correlation structure of OPPW two-dimensional optical orthogonal codes.
- Consequently, standalone phase-address novelty is **refuted**. The dense
  Legendre carrier converts the same OPPW overlap count into a bipolar
  correlation score; it does not create a new address space.
- `typed16` is also known: the eight Walsh/Hadamard rows and their negatives
  are the 16-word first-order Reed-Muller/biorthogonal code `RM(1,3)` with
  parameters `[8,4,4]`.
- The surviving publishable questions are now limited to:
  - a genuinely new finite-sample false-unlock/error theorem for the joint
    dense decoder;
  - a matched-resource Pareto advantage over sparse OPPW and ordinary
    key-value routing;
  - a preregistered positive higher-order interaction that is not explained by
    the known components;
  - a rigorous systems result joining routing, queue control, and reversible
    provenance-scoped correction.
- Hostile implementation review found no P0 defect and independently confirmed
  the FFT shift convention, exact 4691 autocorrelation, quotient gauge,
  overlap bound, `typed16` search, and `free256` factorization. It nevertheless
  issued a NO-GO for PRW-1 through PRW-4 evidence until six P1 contracts close:
  true end-to-end 4096 control, truthful payload storage, crossed planted and
  decoder masks, fixed false-unlock sample/interval rules, distinct clean
  templates and corrupted query types, and a frozen bank argmax/margin/unlock
  evaluator.
- The protocol now freezes a single primary false-unlock cell, disjoint
  hash-derived `BANK`/`SELECT`/`REPORT` streams, exact SELECT/REPORT group
  counts, a fixed score-margin grid, a lexicographic threshold rule, and a
  one-sided 97.5% exact Clopper-Pearson upper bound. Runs below that contract
  are explicitly `SMOKE_NON_EVIDENTIARY`.
- Sparse OPPW decoding over the identical phase codebook is mandatory. Without
  it, any dense-carrier advantage is `INCONCLUSIVE`.
- A deeper reduction further narrows the theorem lane. Flattening every
  type/shift/mask carrier into a bipolar codeword makes dense PRW correlation
  exactly maximum-likelihood nearest-codeword decoding on a binary symmetric
  channel. A fixed competitor at Hamming distance `d` ties or wins exactly
  when at least `ceil(d/2)` of those differing coordinates flip.
- With known matching orientation and `k` aligned layers, the exact Legendre
  distance is `(L-k)*(p+1)/2`. The resulting pairwise error is an ordinary
  binomial tail, and bank error admits the standard distance-spectrum union
  bound. This means a generic finite-noise theorem is coding theory, not a new
  memory theory; only a sharper structured distance spectrum or joint-decoder
  result could remain mathematically interesting.
- An independent small exact check at `p=31`, `L=4`, `k=2`, and bit-flip rate
  `0.35` gave distance `32`, analytic tie-or-win probability
  `0.05784490084795206`, and empirical frequency `0.057965` over 200,000
  trials.
- A post-restart local generic NumPy cyclic-correlation microbenchmark
  (16 candidate types, 8 layers, 30 measured repetitions after warmup) found:
  - length 4091: median `24.863 ms`, p95 `27.657 ms`;
  - length 4096: median `5.351 ms`, p95 `6.859 ms`;
  - length 4691: median `28.347 ms`, p95 `39.702 ms`.
  On this generic CPU path, 4096 is about 5.30x faster than 4691. This is a
  bounded kernel timing, not end-to-end latency and not a Rader implementation,
  but it is direct negative pressure on `PRW-4`.
- The exact code-distance analysis makes the old 90%-at-0.45 target
  scientifically misleading. For `p=4691`, `L=8`, `kappa<=1`, `typed16`, and
  bit-flip rate `0.45`, the wrong-type minimum-distance lower bound is `16418`.
  Across 1,125,840 wrong type/shift/mask hypotheses, the Chernoff union bound is
  about `1.66e-30`. Under the ideal simulator, anything materially below
  perfect raw type recovery is evidence of a bug or violated model assumption,
  not encouraging partial success.
- A sparse OPPW code under the same per-coordinate BSC has distance only
  `2*(8-1)=14` and a large pairwise error at rate `0.45`. That apparent dense
  advantage is the ordinary consequence of expanding the code across roughly
  `p/4` more Hamming distance/energy; it is not a matched-resource memory
  breakthrough.
- The only credible new mathematical conjecture left is sharper and much
  narrower: for the frozen overlap-one Legendre-by-`RM(1,3)` bank, the total
  wrong-type error may asymptotically equal nearest-state multiplicity times
  the exact binomial pairwise tail. Testing it requires disagreement-set
  intersections, Hunter's spanning-tree correction, and importance sampling.
  The pairwise theorem itself is standard; only a proved sharp structured-bank
  asymptotic could be a new decoder result.
- Hostile runner review exposed and killed one tempting but invalid
  composition: applying the type-relative phase signature to the supposedly
  separate payload lets global payload-only search recover the type. Payload
  collision groups must remain in identical type-canonical coordinates and
  inherit only global transport; otherwise there is no carrier "unlock" to
  test.
- A pure shared Legendre ring cannot identify waypoints after maximizing over
  rotation because all rotations are equivalent. It can be a phase/synchrony
  carrier only; identity must live in a separate payload code or subspace.
- Searching all 256 eight-layer polarity masks is algebraically equivalent to
  summing per-layer absolute correlations. It is therefore a false-unlock
  control, not a legitimate default router.
- The existing evidence DAG rejects cycles by design. Any cyclic phase carrier
  must remain inside an experimental router; waypoint lifecycle, correction,
  pruning, and rollback should reuse the existing steering-post bank rather
  than weakening evidence-DAG invariants.
- The proposed prime-phase mechanism therefore has one especially useful
  falsification target: it must make a query-conditioned route decision that
  differs from the static bank while preserving immutable evidence and
  rollback. Merely wrapping the same deterministic posts in rotations cannot
  alter H5 outcomes.
- A later off-baseline Rung-16 campaign is an even closer prior lane:
  `5a3fbcc2` tested a quant-aware subdomain routing plane and failed its frozen
  SELECT gate in both arenas. Plane-minus-single-global SELECT NDCG was
  `0.000000` on SciFact and `-0.000899` on the mixed
  SciFact/NFCorpus/FiQA2018 arena; corresponding REPORT deltas were
  `-0.003332` and `-0.007383`. This is direct negative evidence against the
  assumption that more conditional subdomain routes inherently improve
  retrieval.
- Commit `7e6c9118` records the adjacent living/annealing results:
  - C5 living and C5s static are numerically identical on SciFact
    (`0.131135`) and NFCorpus (`0.046389`);
  - the one-shot C5r route beats living C5 on SciFact and is nearly tied on
    NFCorpus; and
  - the one-seed H4 compound-cycle ablation collapses NDCG from `0.236297` to
    `0.005258`.
  These runs do not test PRW phase coding, but they do falsify any inherited
  premise that layering, living updates, or repeated correction is beneficial.
- Consequently, a PRW key can earn a systems contribution only if its route
  changes real held-out decisions and beats the single-global, direct-metadata,
  static-bank, and one-shot controls. Cleaner indexing of an ineffective bank
  is not a retrieval gain.
- The original `feat/brain-file-map-b0-b1` branch remains preserved. Current
  METHOD_DEV edits are isolated on `codex/prime-ring-onion-method-dev` and are
  intentionally uncommitted while the runner and hostile review converge.
- The strongest newly located primary-source collision is Nguyen Q. A.,
  Györfi, and Massey (IEEE TIT, 1992): it maps each symbol of a `p`-ary outer
  word to a cyclic shift of a length-`p` binary Legendre word. For
  `p = 3 mod 4`, two unequal symbols contribute exactly `(p + 1) / 2` Hamming
  distance, so its unmasked distance is exactly the PRW identity
  `(L - r) * (p + 1) / 2`. This is more direct prior art than the OPPW
  isomorphism alone.
- No primary paper was found in the bounded audit that proves the exact
  `PRW-T1` ratio-one asymptotic for the frozen overlap-one
  Legendre-by-`RM(1,3)` bank. The defensible residual is narrow: freeze `q`,
  tie handling, transmitted-state averaging, and a bank sequence, then prove
  both aggregate nearest-event intersections and all farther-neighbor
  contributions are little-o of
  `A_min * beta_q(d_min)`. Standard pairwise tails, union bounds, and
  conditional "negligible intersections imply exactness" are not novel.
- The bounded exact-number literature audit found no application-specific
  published use of `4691`. That is not evidence that none exists. Its known
  properties remain generic: prime, `3 mod 4`, and a smooth `p - 1` useful only
  if an actual Rader implementation wins a benchmark.
- The final corrected smoke artifact is
  `artifacts/method-dev/prime-ring/prime-ring-waypoint-v1.json`:
  - `576/576` cells retained;
  - `255` completed and `321` structurally unavailable;
  - zero errors and zero false-unlock gate passes;
  - `SMOKE_NON_EVIDENTIARY`, `METHOD_DEV`, promotion and production disabled;
  - `PRW-1` through `PRW-3` inconclusive smoke, `PRW-4` inconclusive,
    `RADER-1` untested, novelty not established.
- The smoke's apparently perfect routed payload result at some cells is not a
  scientific result: there are only three REPORT planted groups per seed,
  thresholds are point-estimate-only, and identical cross-type payloads make
  payload-only Recall@K canonical-tie dependent. The artifact explicitly marks
  `K = 5 > T = 4` as not evidence of type retrieval.
- Required controls remain fail-closed and unimplemented: a native sparse OPPW
  observation with matched noise/energy, an equal-channel-use repeated-bit
  code, block/burst noise, process RSS, repeated hardware timing, and the full
  pooled 512/1024-per-class campaign.
- The actual sample resource accounting now distinguishes, for the
  `p=4691`, `L=8`, `T=4`, `W=4` cell, `4,803,584` logical type-expanded
  payload bytes from `1,200,896` deduplicated stored payload bytes. Actual-cost
  efficiency uses the deduplicated total, while FFT/product/routed temporary
  peaks are conservative componentized estimates rather than process-RSS
  measurements.
- Final live validation passed:
  - `49/49` PRW core, runner, and exact-reduction tests;
  - `31/31` adjacent CRSV adversarial tests;
  - focused Ruff checks clean;
  - stable Clopper-Pearson probes at `n=1536` and `n=3072` invert to CDF
    `0.025` within floating-point tolerance;
  - block-flag gate `CLEAR`, zero carried debt.
- Current verdict: there is no established groundbreaking computing result.
  The broad rotating-prime/onion story reduced to known code constructions and
  ordinary distance expansion. The only mathematically plausible new work is
  the very narrow `PRW-T1` intersection-spectrum theorem; the only plausible
  systems work is a matched-resource carrier-to-separate-payload gate that
  changes real held-out decisions. Neither has been validated.
- Continuation decision on 2026-07-24: resume implementation, exact
  small-prime analysis, and bounded microbenchmarks only. The powered campaign,
  large importance sampling, real-corpus campaigns, and full-size Rader timing
  remain held because they are unnecessary for the next correctness slices and
  could create avoidable memory pressure.
- The next useful work separates into three independent code lanes: matched
  sparse/repetition/noise controls, a resource-guarded `PRW-T1` finite
  intersection enumerator, and a correctness-first bounded Rader
  implementation. Remaining harmonic/polar/3D/queue ideas require typed
  hypotheses before code.
- The remaining informal concepts are now separated in
  `docs/research/prime-ring-remaining-hypotheses-2026-07.md`: graph-coupled
  cyclic fibers (`PRW-G1`), Fourier-phase synchronization (`PRW-H1`), the prime
  dimension null (`PRW-D1`), CRT payload pivots (`PRW-C1`), causal queue
  potential (`PRW-Q1`), and reversible living state (`PRW-L1`). None inherits
  support from the original ring construction.
- RB-2 now has a finite, resource-guarded implementation in
  `prime_ring_intersection.py`. It materializes only bounded small-prime
  Legendre-by-mask banks, computes exact finite BSC pair and pair-intersection
  probabilities, builds the Hunter maximum-spanning-tree correction, and can
  brute-force the true event union only when the full noise-pattern count fits
  the declared budget. It explicitly leaves the asymptotic claim false.
- Eight bounded RB-2 tests pass, including direct noise-pattern oracles for the
  binomial tail, pair-event intersection, and whole-bank union; no importance
  sampling or large-prime bank was run.
- RB-3 has a correctness-first Rader implementation and 13 small-prime tests
  from the independent lane. Only `p=7`, `11`, and `31` correctness paths were
  exercised; no 4091/4691 benchmark was launched.
- The initial RB-1/RB-2 implementations did not survive hostile review
  unchanged. Three RB-1 failures were corrected: fixed-weight noise had been
  mislabeled as an independent BSC, an always-off decoder could close pooled
  controls at zero recall, and resource accounting omitted live temporaries and
  co-resident ring-template arrays while checking too late. RB-2 likewise
  normalized arbitrary inputs before its nominal hard guard.
- RB-1 now uses independent Bernoulli chip flips for both the dense carrier and
  equal-channel-use repeated-bit control. Block and burst paths retain exact
  global counts but are explicitly outside the binomial/BSC calculation. Native
  OPPW remains a distinct symbol-substitution channel, so dense-versus-native
  superiority is not a matched-noise claim.
- Pooled control closure now requires a frozen threshold, the REPORT
  false-unlock confidence gate, and true-unlock recall of at least `0.50`.
  The dense primary path additionally requires 100% raw type accuracy in the
  pooled `q=0.45` cell. The protocol's stronger "through q=0.45" lower-rate
  sweep is not inferred from that endpoint and remains an explicit blocker.
- RB-1 standalone and harness-co-resident NumPy-array estimates are now
  preflighted before native/repeated control allocation under a hard 512 MiB
  ceiling and a 50,000,000-work-unit ceiling. These are conservative array
  estimates, not measured process RSS and not authority for a powered run.
- RB-2 now exposes the full
  `(d_i,d_j,|D_i intersection D_j|)` signature spectrum and nearest-event sum,
  distinguishes competitor tie-or-better unions from final decoder error, and
  accepts PRW-T1-specific labeling only for verified overlap-one,
  `RM(1,3)`, `p congruent to 3 mod 4` banks. Its 512 MiB and 120-second hard
  ceilings cannot be raised by a caller.
- Final bounded validation on the reconditioned tree passed:
  - 61 combined prime-ring core, Rader, and finite-intersection tests;
  - 16 runner tests in 12.299 seconds, using only tiny smoke fixtures and a
    temporary artifact test;
  - 31 adjacent CRSV tests and 4 prior prime-ring theory tests;
  - focused Ruff checks across all six new/changed Python implementation and
    test files.
- RB-3 ultimately has 14 focused tests (not 13) after resource-hostile review,
  plus exhaustive odd-prime property checks through 101 reported by the lane.
  No 4091/4691 transform or benchmark was run.
- The bounded implementation result is still not a breakthrough result:
  PRW-T1 remains a finite diagnostic rather than an asymptotic theorem, Rader
  has no full-size performance evidence, matched controls have no powered
  empirical result, and every novelty/production/promotion gate remains false.

---

## 2026-03-28 Disk-Resident LLM Feasibility Findings
- The paper referenced in the user request was not attached in-session. The closest primary-source fit to the request is `LLM in a Flash`, so the current feasibility pass uses that as the working assumption.
- ChelatedAI's current computational-storage path is still a transport and replay proof, not a transformer runtime.
- `computational_storage_poc/block_graph.py` uses fixed `512 x 512` FP16 dense blocks with zero padding, which is far too inefficient for realistic transformer storage.
- `computational_storage_poc/mock_nvme.py` preloads the full binary into RAM, so the current "NVMe" path does not measure real SSD behavior.
- A credible near-term architecture is SSD-resident compressed weights plus CPU execution, not end-to-end transformer compute inside a commodity SSD controller.
- `LLM in a Flash` is the right architectural template for the storage side:
  - keep attention weights resident
  - stream only a small active FFN slice
  - prefer larger contiguous reads and reuse windows
- `T-MAC` is the right architectural template for the CPU side:
  - execute low-bit kernels directly
  - avoid dequantize-then-matmul overhead
- The new estimator at `computational_storage_poc/disk_llm_estimator.py` shows:
  - dense full-model streaming from SSD is not viable
  - sparse flash-style streaming makes 7B-70B plausible
  - 405B is only marginal even on workstation-class hardware
- Highest-value repo improvements are:
  - manifest-driven quantized packing
  - real disk-backed reads
  - sparse FFN predictor and cache
  - CPU low-bit kernel path
  - transformer microbenchmark acceptance test

## 2026-03-28 Addendum Findings: REAP / TurboQuant / CPU-Disk Systems
- `REAP` is directly relevant for disk-first deployment of coding MoE models because it reduces the stored expert bank without destroying router behavior.
- `TurboQuant` is not a weight-footprint method; it is primarily a KV-cache and vector-search compression method.
- The strongest published CPU-only inference evidence found in primary sources is currently:
  - `T-MAC` for low-bit CPU inference
  - `bitnet.cpp` / `1-bit AI Infra` for native 1-bit CPU inference
  - `Gemma.cpp` as an official CPU runtime surface from Google
- The strongest published evidence for external memory replacing parametric scale is:
  - `kNN-LM`
  - `RETRO`
  - graph-structured retrieval for code generation such as PKG / GraphSkill
- The practical architecture answer is not "SSD instead of GPU" in isolation; it is "small CPU-native core + disk-backed memory and retrieval + optional disk-resident compressed weights."

## 2026-03-28 Revised Roadmap Findings
- The old roadmap framing was too passive: it assumed the main remaining work was evaluation, not a new architecture program.
- The roadmap now needs to be treated as a multi-phase program with explicit dependency ordering:
  - storage substrate
  - CPU inference substrate
  - compression branches
  - retrieval / graph memory
  - runtime integration
  - long-context compression
  - end-to-end promotion review
- ARCH-AEP needs a new mandatory loop for architecture-led work:
  - scope lock
  - implementation
  - ARCH-AEP review
  - code analysis / hardening
  - promote / defer
- The safest way to minimize scope drift is to prohibit cross-phase PRs and require explicit phase non-goals.
- Retrieval / graph memory should be developed as a first-class phase, not mixed into low-level storage or CPU kernel phases.

## 2026-03-29 Phase 1 Storage Substrate Findings
- The first safe implementation slice for `Phase 1` is compatible with the new roadmap:
  - add a manifest-driven packed artifact path
  - switch the mock NVMe path to real file-backed access
  - add a benchmark surface for size/read reduction
- `packed_graph.py` now demonstrates the right substrate direction:
  - explicit manifest
  - exact matrix-shape storage
  - no mandatory `512 x 512` zero padding
  - `mmap`-backed reads
- The legacy padded path remains intact, which keeps the phase bounded and avoids premature runtime integration.
- Moving `MockNVMeDrive` from eager file reads to `mmap` surfaced an important Windows-specific lifecycle concern: tests and call sites must explicitly close the mapping before tempdir cleanup.
- The new storage benchmark showed the intended direction clearly on the current test graph:
  - artifact size reduction: about `94.69%`
  - read-byte reduction: about `94.71%`
  - parity maintained (`max_abs_diff = 0.0`)
- The packed substrate is now wired into the standard compile helpers:
  - `compiler.py --format packed`
  - `train_and_compile.compile_model(..., artifact_format="packed")`
- This phase should stop here for now. The next phase should build on the new packed substrate rather than mixing in CPU kernel work immediately.

## 2026-03-29 Phase 2 CPU Inference Substrate Findings
- The correct next bounded slice after the packed storage substrate is a CPU execution seam that consumes packed artifacts without introducing sparse routing, MoE logic, or retrieval memory yet.
- `cpu_backends.py` now provides a minimal backend abstraction with two baseline implementations:
  - `NumpyFloat32Backend` as the numerical reference path
  - `NumpyInt8DynamicBackend` as a correctness-oriented low-bit baseline
- `packed_cpu_inference.py` proves that packed disk-backed artifacts can be executed through the CPU backend seam without falling back to the legacy padded graph path.
- The current int8 implementation is numerically close enough to serve as a Phase 2 substrate:
  - benchmark `max_abs_diff` is about `0.002349`
  - unit tolerance remained within the current test thresholds
- The current int8 implementation is not yet a performance win:
  - float32 latency was about `0.1624 ms`
  - int8 latency was about `0.3110 ms`
  - reported speedup was about `0.52x`
- That performance result is expected for this implementation shape because it dynamically quantizes both activations and weights on every call and still uses generic NumPy matmul kernels rather than packed low-bit kernels.
- ARCH-AEP promotion decision for this phase:
  - promote as a correctness and interface baseline
  - do not promote as a CPU-performance claim
- The clean dependency-preserving next steps are:
  - `Phase 2b`: packed quantized artifacts plus reusable scales and lower-overhead CPU kernels
  - `Phase 3A`: sparse/dense FFN selective loading on top of the packed storage and CPU seam

## 2026-03-29 Phase 2b Prequantized Artifact Findings
- The cleanest way to improve the initial CPU substrate was to keep the same packed container and add a second packed storage mode rather than inventing a separate artifact family.
- `packed_graph.py` now supports:
  - FP16 packed weights
  - INT8 packed weights with per-block scales
- `packed_cpu_inference.py` now exploits that storage mode by letting the int8 backend consume prequantized weights directly instead of requantizing them on each call.
- `compiler.py` and `train_and_compile.py` now expose `packed_int8` as a first-class artifact format, which keeps the compile surface aligned with the roadmap.
- Validation results for the current microbenchmark are good enough to promote this slice as the first low-bit performance baseline:
  - float32 latency: about `0.1818 ms`
  - dynamic int8 latency: about `0.5022 ms`
  - prequantized int8 latency: about `0.1524 ms`
  - prequantized int8 speedup vs dynamic int8: about `3.30x`
  - prequantized int8 speedup vs float32: about `1.19x`
  - prequantized int8 read bytes: `41600` vs `83200` for the FP16-packed path
  - prequantized int8 max absolute diff: about `0.002579`
- ARCH-AEP promotion decision for this phase:
  - promote the prequantized INT8 path as the current CPU baseline for packed artifacts
  - still treat it as a prototype kernel path, not a final answer for large-model CPU inference
- Remaining technical constraints:
  - activations are still dynamically quantized per call
  - execution still relies on generic NumPy integer matmul rather than packed low-bit kernels
  - no sparse FFN loading, MoE routing, or retrieval memory is included yet
- The next bounded choices remain:
  - `Phase 2c`: deeper CPU-kernel and activation-path optimization
  - `Phase 3A`: selective FFN loading / sparse runtime behavior

## 2026-03-29 Phase 3A Dense / Sparse FFN Findings
- The most bounded way to implement `Phase 3A` in the current POC is to keep the packed artifact format stable and add selective row-chunk reads for streamed blocks.
- `packed_graph.py` now exposes row-chunk readers for both:
  - dequantized float storage
  - prequantized INT8 storage
- `sparse_cpu_inference.py` implements the first selective-loading runtime:
  - resident-vs-streamed split via `stream_from_block`
  - activation-driven routing heuristic using nonzero rows
  - `SparseChunkCache` for chunk reuse across repeated calls
- The current runtime preserves exact linear equivalence by summing only the row chunks whose activations are nonzero, rather than approximating the output.
- The benchmark harness was tuned so the streamed FFN block materially dominates the byte budget; otherwise the always-resident first block hides the gain signal.
- Current Phase 3A benchmark results:
  - avg dense latency: about `0.2478 ms`
  - avg sparse latency: about `0.2253 ms`
  - dense bytes per token: `98304`
  - sparse bytes per token: `37376`
  - streamed byte reduction: about `61.98%`
  - cache hits: `28`
  - cache misses: `36`
  - max absolute diff: about `0.000806`
- ARCH-AEP promotion decision for this phase:
  - promote the selective-loading runtime as the baseline dense/sparse FFN prototype
  - keep the scope explicitly limited to row-chunk streaming on the current toy packed-graph path
- Remaining constraints:
  - routing is still a simple activation sparsity heuristic, not a learned predictor
  - the cache is a small in-memory prototype, not yet a full runtime policy
  - the benchmark is still a layer-execution harness, not transformer token generation
- The clean next bounded choices are now:
  - `Phase 3B`: MoE / REAP-compatible storage and execution
  - `Phase 4`: retrieval and graph memory substrate

## 2026-03-29 Phase 4 Retrieval / Graph Memory Findings
- The cleanest `Phase 4` slice is a standalone repo-memory substrate that does not couple to the inference runtime yet.
- `repo_graph_memory.py` now provides:
  - a disk-backed node / edge / embedding bundle
  - file and symbol nodes
  - local-import and containment edges
  - a memory-mapped embedding surface
  - a query API that blends vector, lexical, and graph signals
- The current embedding approach is intentionally lightweight:
  - stable hashed token embeddings
  - no dependency on external embedding services
  - good enough for local architecture validation
- `repo_graph_memory_benchmark.py` measures this layer independently of the model runtime and currently reports:
  - node count: `111`
  - edge count: `220`
  - ingest latency: about `93.65 ms`
  - average query latency: about `0.4328 ms`
  - top-1 hit rate: `75%`
  - top-3 hit rate: `75%`
- An ingestion filter was necessary to keep the benchmark honest:
  - benchmark files, tests, docs, and cache directories can otherwise dominate repo-local code retrieval with artificial lexical matches
- ARCH-AEP promotion decision for this phase:
  - promote the repo-memory layer as the current disk-backed code-memory substrate
  - keep it decoupled from execution until `Phase 5`
- Remaining constraints:
  - embedding quality is still hash-based rather than model-based
  - graph edges are limited to containment and local-import relationships
  - reranking is heuristic, not learned
- The clean next bounded choices are now:
  - `Phase 5`: integrate storage, CPU, sparse runtime, and repo memory into one runnable prototype path
  - `Phase 3B`: keep MoE / REAP as a separate parallel branch if a concrete MoE target is chosen

## 2026-03-29 Phase 5 Runtime Integration Findings
- The cleanest `Phase 5` slice is a single CPU-only repository Q&A style prototype rather than a fake token generator.
- `integrated_repo_runtime.py` now integrates:
  - the disk-backed repo-memory query surface
  - retrieval-result featureization
  - a packed INT8 reranker artifact
  - the sparse CPU execution path
- The current end-to-end runtime uses a deterministic reranker graph rather than a trained language model, which keeps the phase honest:
  - it proves orchestration and metric capture
  - it does not overclaim generative capability
- `integrated_runtime_benchmark.py` now measures the integrated path and currently reports:
  - average retrieval latency: about `1.3089 ms`
  - average inference latency: about `1.9976 ms`
  - average total latency: about `3.3251 ms`
  - queries per second: about `28`
  - average bytes read per query: `1152`
  - mapped bytes: `121313`
  - peak Python heap: about `55.91 KB`
  - top-1 hit rate: `50%`
  - top-3 hit rate: `75%`
- A code-aware tokenization fix was necessary in the repo-memory substrate so snake_case and CamelCase repository identifiers become searchable in a repo-local way.
- ARCH-AEP promotion decision for this phase:
  - promote the integrated repository-Q&A prototype as the current end-to-end local baseline
  - keep the claim boundary narrow: this is orchestration and retrieval-aware reranking, not a full disk-resident LLM
- Remaining constraints:
  - the reranker is hand-authored, not trained
  - the runtime still targets repository retrieval/reranking rather than free-form code generation
  - no KV-cache or memory compression exists yet
  - MoE / REAP remains intentionally separate
- The clean next bounded choices are now:
  - `Phase 6`: long-context and memory compression on top of the integrated baseline
  - `Phase 3B`: MoE / REAP path as a parallel branch if a concrete MoE target is selected

## 2026-03-29 Phase 6 Memory Compression Findings
- The cleanest `Phase 6` slice on the current baseline is compressed repo-memory embeddings rather than KV-cache work, because the runtime is still a retrieval/reranking prototype rather than a long-context generator.
- `repo_graph_memory.py` now supports:
  - float32 embedding storage
  - int8-compressed embedding storage with a saved scale
- The compression path was threaded through both the standalone repo-memory benchmark and the integrated runtime so memory savings and quality impact can be measured at two levels.
- Current repo-memory compression benchmark results:
  - float32 mapped bytes: `122880`
  - int8 mapped bytes: `30720`
  - mapped-byte reduction: about `75%`
  - float32 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 average query latency is slightly higher in the current path
- Current integrated-runtime compression benchmark results:
  - float32 mapped bytes: `123361`
  - int8 mapped bytes: `31201`
  - mapped-byte reduction: about `74.71%`
  - float32 average total latency: about `3.8535 ms`
  - int8 average total latency: about `4.0467 ms`
  - float32 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 peak Python heap is higher in the current implementation because query-time quantization introduces temporary allocations
- ARCH-AEP promotion decision for this phase:
  - promote the int8 repo-memory path as the current compression experiment baseline
  - do not claim it is fully optimized yet, because the latency and heap tradeoff still need review
- Remaining constraints:
  - compression currently targets the repo-memory embeddings only
  - no KV-cache compression exists because the runtime is not a long-context generator yet
  - query-time quantization overhead still leaves optimization headroom
- The clean next bounded choices are now:
  - `Phase 7`: end-to-end evaluation and promotion review
  - `Phase 3B`: MoE / REAP as a separate parallel branch

## 2026-03-29 Phase 7 Evaluation And Promotion Findings
- `phase7_system_evaluation.py` now aggregates the benchmark pack across storage, CPU, sparse runtime, repo memory, integrated runtime, and compression.
- The current benchmark thresholds all pass for the intended research-baseline scope:
  - storage reduction
  - CPU baseline
  - sparse runtime
  - repo memory
  - integrated runtime
  - standalone compression
  - integrated compression
- The current promotion call is:
  - promote as `research baseline`
  - do not promote as `production-ready`
- The strongest reason for the defer remains architectural scope, not a failing benchmark:
  - the system is still retrieval-and-reranking oriented
  - it is not a full generative runtime
  - it is not yet clearly simpler operationally than a GPU-backed alternative
- The promotion memo at `docs/phase7-promotion-review-2026-03-29.md` now records:
  - promote/defer decision
  - recommended presets
  - no-go conditions
  - next-branch recommendations
- The clean next bounded choices after the closed review are:
  - `Phase 3B`: MoE / REAP branch if a concrete target model exists
  - targeted optimization and evaluation expansion under the current research baseline

## 2026-03-29 Phase 3B MoE / REAP Branch Findings
- The cleanest post-review parallel branch was to keep MoE / REAP isolated from the integrated baseline and land it as a separate artifact/runtime seam.
- `moe_reap.py` now provides:
  - a disk-backed MoE artifact format
  - expert-bank metadata with preserved router ids
  - routed-expert CPU execution
  - a REAP-like pruning compatibility function based on expert weight norms
- The current benchmark proves the intended branch property:
  - full artifact bytes: `2025`
  - pruned artifact bytes: `1122`
  - artifact reduction: about `44.59%`
  - full bytes read: `704`
  - pruned bytes read: `384`
  - read reduction: about `45.45%`
  - full experts evaluated: `4`
  - pruned experts evaluated: `2`
- The branch is intentionally not integrated into the Phase 5 prototype yet.
- ARCH-AEP promotion decision for this branch:
  - promote as a valid parallel MoE / REAP compatibility branch
  - defer runtime integration until a concrete MoE target model is selected

## 2026-03-29 Targeted Optimization And Evaluation Expansion Findings
- The cleanest post-Phase-7 follow-up was a bounded optimization and benchmark-hardening pass on the existing research baseline, not more architecture churn.
- `repo_graph_memory.py` now scores int8-compressed embeddings in row chunks instead of materializing an `int32` copy of the full memory-mapped matrix for every query.
- `repo_graph_memory.py` also now:
  - favors precise path matches over helper-path spillover
  - returns unique paths instead of repeated file/symbol duplicates for the same path
- That change materially improved the compressed integrated-runtime heap profile:
  - previous int8 peak Python heap: about `124.55 KB`
  - current int8 peak Python heap: about `58.45 KB`
- `integrated_repo_runtime.py` now uses retrieval-dominant fusion for final ranking, so the packed reranker refines candidate order instead of overwhelming stronger retrieval evidence.
- `cpu_inference_benchmark.py` now uses warmup plus median-of-trials timing so the Phase 7 promotion call is not driven by one noisy microbenchmark sample.
- `retrieval_eval_suite.py` now defines a shared 10-query repo-local benchmark suite covering:
  - packed storage
  - sparse loading
  - CPU backends
  - training / compile flow
  - repo memory
  - integrated runtime
  - MoE / REAP
  - payload transport
  - emulation
  - disk sizing / feasibility estimation
- The standalone repo-memory benchmark now reports on that wider suite:
  - query count: `10`
  - top-1 hit rate: `100%`
  - top-3 hit rate: `100%`
- The integrated runtime benchmark now reports on the same wider suite:
  - query count: `10`
  - average total latency: about `3.13 ms`
  - top-1 hit rate: `100%`
  - top-3 hit rate: `100%`
- The int8 compression tradeoff is now clearer after the optimization pass:
  - mapped-byte reduction is still about `75%`
  - quality stayed unchanged on the shared suite
  - standalone int8 query latency is now slightly better than float32 on the shared suite
  - integrated int8 total latency remains higher than float32, so compression is still a memory win first and an end-to-end latency win second
- ARCH-AEP decision for this pass:
  - promote as a bounded optimization and evaluation-hardening pass
  - keep the recommendation unchanged: the stack remains a promoted research baseline, not a production-ready system

## Requirements
- Proceed on the active computational-storage follow-up items overnight.
- Use implementation-session style orchestration with research, architecture, implementation, and validation phases.
- Keep track of session log and agent work.
- Create a PR for each reviewable item so the user can review via PR workflow tomorrow.
- Do not jeopardize work running in other repos.

## Research Findings
- Latest active handoff is the computational-storage post-merge follow-up from Session 25.
- The active queue is limited to four items: hardware evidence, emulator-path CI decision, transport-path scope decision, and retention policy review.
- The old Session 21/22 “top 15” is historical and already completed.
- `.github/workflows/test.yml` currently runs global `unittest` plus a dedicated computational-storage fundamentals job, but neither job provides a distinct emulator-path gate.
- `.github/workflows/build_firmware.yml` separately validates RP2040 firmware compilation and artifact generation.
- `test_computational_storage_payload.py` validates the deterministic payload contract, virtual-disk trigger-sector injection, and host-reader decoding from a file path.
- `computational_storage_poc/usb_host_inference.py` is the host-side raw-sector reader for physical or file-backed devices.
- A local hardware probe for RP2040 / Pico / TinyUSB devices returned no present device, so authentic real-hardware evidence cannot be captured tonight unless a device appears later.
- `computational_storage_poc/emulation/fuse_block_emulator.py` already contains emulator semantics, but importing it in CI currently requires `fusepy`; the core read behavior can be extracted into a dependency-light module and tested directly.
- `computational_storage_poc/emulation/docker-compose.yml` requires privileged FUSE (`/dev/fuse`), which is a poor fit for a stable default GitHub Actions gate.
- The existing docs already state that firmware scope is transport correctness, not on-device digits inference, but that boundary can be made more explicit and centralized.
- Backup refs currently present locally include `backup/retired-*` refs plus older local backup refs; remote backup refs from February are also still present.
- The hardware-evidence prep implementation can be validated tonight with file-backed trigger-sector images, which gives strong confidence in tomorrow’s physical capture workflow without claiming hardware success.
- The emulator-CI implementation can stay branch-independent from the hardware-evidence branch by avoiding the optional host-reader verbosity refinement.
- A dedicated emulator job can remain lightweight because it only needs `numpy`, the virtual controller, and the host-reader path.
- The scope decision is best enforced through one canonical doc referenced by the POC and firmware docs, rather than duplicated prose in multiple places.
- The safest retention decision tonight is a timed manual review, not deletion, because the computational-storage follow-up is still active and the March rollback refs are fresh.
- The planning-with-files catchup helper path in the installed skill is stale on this machine (`.claude` path missing), so session recovery has to be done from repo files directly.
- PR `#90` had a real correctness gap during review: `resolve_drive_path()` on Windows rewrote already-formed device paths like `\\.\PhysicalDrive2` into malformed paths. The branch now preserves explicit device paths and has a regression test.
- Local hardware inspection on 2026-03-06 still shows no RP2040 / Raspberry Pi Pico / TinyUSB mass-storage device. The only removable USB disk currently visible is a SanDisk drive, so physical evidence remains blocked.
- PR `#94` should not be merged until after the implementation PRs land or it is refreshed, because its handoff state would otherwise become stale as soon as `#90`-`#93` merge.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Separate active roadmap from historical top-15 work | Prevents wasting time reopening already completed items |
| Use documentation artifacts to preserve “fresh agent” outputs | Maintains auditable orchestration in a single-session environment |
| Treat hardware evidence as blocked by actual device availability | Prevents fabricated or software-only claims being mislabeled as physical validation |
| Slice the work into four item PRs plus one wrap PR | Minimizes conflicts while still giving the user reviewable units tomorrow |
| Prefer a reusable evidence-capture tool over a one-off manual checklist | Lets hardware evidence be captured immediately once a device is available |
| Extend `usb_host_inference.py` with a path resolver and optional verbosity control | Keeps existing CLI behavior while making automation and tests cleaner |
| Keep the emulator-CI branch independent of the host-reader refinement | Avoids hidden coupling between review branches |
| Express the current transport boundary through a canonical decision doc | Makes future scope promotion auditable and reduces overstatement risk |
| Use a tiered retention policy with a dated manual review window | Protects rollback paths now while preventing indefinite artifact sprawl |
| Preserve explicit Windows device paths in `resolve_drive_path()` | Matches the documented CLI contract for hardware evidence capture and avoids mangling valid raw-device arguments |
| Treat current hardware evidence as blocked despite a visible USB disk | The visible removable storage is a SanDisk drive, not an RP2040/Pico-class target |
| Merge the session-wrap PR last | Keeps `next-session.md`, `tracker-pointer.md`, and `CLAUDE.md` aligned with post-merge reality |
| Refresh the existing wrap PR instead of opening a sixth PR | Preserves the existing review thread while replacing the stale pre-merge handoff with the accurate Session 27 state |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| No evidence yet that physical RP2040 hardware is available | Verify connected devices before attempting hardware evidence capture |
| `computational_storage_poc/fuse_fs.py` does not exist at the probed path | Inspect actual POC file layout before planning emulator-path CI changes |
| `resolve_drive_path()` mis-handled explicit Windows device paths in PR `#90` | Fix on branch and add regression coverage before merge |
| `ruff check` was mistakenly run against `.github/workflows/test.yml` | Restrict lint to Python files; use other tooling for YAML if needed |

## Resources
- `CLAUDE.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/session-log-2026-03-06-session25.md`
- `computational_storage_poc/README.md`
- `computational_storage_poc/firmware/README_FIRMWARE.md`
- `.github/workflows/test.yml`
- `.github/workflows/build_firmware.yml`
- `test_computational_storage_payload.py`
- `computational_storage_poc/usb_host_inference.py`

## Visual/Browser Findings
- None yet.

## 2026-03-06 Roadmap Audit Findings
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-index.md` still shows only `AEP-2026-03-06` as active, and that cycle is limited to real RP2040 hardware evidence capture plus the dated retention review.
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md` aligns with the tracker: no non-hardware engineering phase is listed as active.
- `README.md` says Phases 1-4 are complete.
- `REFACTORING_PLAN.md`, `COMPLETION_SUMMARY.md`, and `PR_DESCRIPTION.md` still contain older "Deferred to Phase 4" or "Future Development" language, so they cannot be treated as authoritative roadmap sources without checking the live code.
- The next audit step is to verify those older deferred items against the current codebase and experiment scripts before declaring the development roadmap complete.
- Live-code verification shows the previously deferred research features are already present:
  - `antigravity_engine.py` implements `ingest_streaming()` and `enable_adaptive_threshold()`.
  - `teacher_distillation.py` supports configurable `batch_size`, chunked encoding, and ensemble parallelism.
  - `cross_lingual_distillation.py` exists with language-aware teacher routing.
  - `online_updater.py` supports `triplet_margin`, `infonce`, and `cosine_similarity` losses plus diagnostics/scheduling.
  - `benchmark_beir.py`, `benchmark_multitask.py`, `dashboard_server.py`, `run_sweep.py`, and `run_large_sweep.py` all exist.
- Historical docs that still list missing features are stale relative to the code. Session 22 and Session 23 logs show the old top-15 implementation items were delivered and merged.
- The notable non-hardware gap is experimental execution, not implementation: `run_large_sweep.py` exists, but `large_sweep_results.json` and `large_sweep_results.csv` do not.
- `docs/phase4-experiment-protocol.md` is useful background for Phase 4 feature usage, but it is not a current post-development roadmap or weight-refinement plan.
- A new canonical current-state doc now exists at `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md`.
- `docs/INDEX.md` now links that audit/test-plan doc so future sessions do not need to reconstruct the conclusion from session logs.

## 2026-03-06 Experiment Campaign Findings
- Runtime prerequisites are currently available locally:
  - `torch 2.9.1+cpu`
  - `sentence_transformers 5.2.0`
  - `mteb 2.6.1`
  - `qdrant_client` import succeeded
  - `numpy 2.3.5`
- A `requests` dependency warning is emitted during imports, but it is not a hard blocker.
- Existing local experiment artifacts include `adapter_weights.pt` and `sweep_results.json`.
- No completed large-sweep artifact exists yet.
- `antigravity_engine.py` already exposes `enable_online_updates(...)`, so Phase 5 does not require new engine hooks.
- Historical docs indicate `benchmark_comparative.py` already has an `online_updates` configuration, which may be sufficient for online-ablation work without adding a brand-new benchmark script.
- `benchmark_comparative.py` and `benchmark_beir.py` were not safe to use as-is for real evaluation because their CLIs did not wire in a real `engine_factory`; they defaulted to dummy retrieval. This session patched them to use real engines from the CLI path.
- Real evaluation also required mapping Qdrant point IDs back to original document IDs. A shared `map_predicted_ids()` helper is now in `benchmark_utils.py`.
- `run_sweep.py` and `run_large_sweep.py` originally reused the shared SciFact Qdrant path and had no query cap. This session added:
  - `--max-queries`
  - `--db-path`
- The first bounded campaign failed Phase 1 because an orphaned `run_sweep.py` process from an earlier launch still held `db_scifact_evolution`. That stale process was terminated.
- The campaign runner now:
  - passes `-u` to child Python benchmark commands for unbuffered logging on future runs
  - snapshots/restores `adapter_weights.pt`
  - writes an on-disk manifest
  - uses UTF-8 for child processes
  - launches sweep phases against isolated per-run Qdrant directories
- The current active campaign run is `experiment_runs/weight-refinement-20260306-session28-isolated`.
- During the current isolated run, `phase1_standard_sweep.log` stayed quiet after initialization, but the child `run_sweep.py` process continued consuming CPU and writing to the isolated Qdrant folder, indicating active execution rather than an early crash.
- The worktree currently has uncommitted documentation/planning files from the roadmap audit:
  - `docs/INDEX.md`
  - `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- Local branch inventory includes many historical `backup/*`, `feat/*`, `feature/*`, and `pr/*` branches. They will be treated as historical context only unless this campaign needs recovery from them.
- The remaining contamination bug was broader than the runner-level snapshot/restore: `benchmark_distillation.py` reused the shared root adapter across baseline/offline/hybrid in a single process, `benchmark_multitask.py` could reuse it across tasks, and `benchmark_beir.py` inherited the same risk through `ComparativeTestbed`.
- The fix is now centralized in `benchmark_utils.isolated_adapter_state()`. `benchmark_comparative.py` uses it per configuration, `benchmark_distillation.py` uses it per mode, and `benchmark_multitask.py` uses it per task.
- `benchmark_distillation.py` now accepts `--max-eval-queries`, and `run_weight_refinement_campaign.py` passes the campaign query budget through to that phase.
- Targeted validation passed after the fix:
  - `python -m py_compile benchmark_utils.py benchmark_comparative.py benchmark_distillation.py benchmark_multitask.py run_weight_refinement_campaign.py test_benchmark_comparative.py`
  - `python -m ruff check benchmark_utils.py benchmark_comparative.py benchmark_distillation.py benchmark_multitask.py run_weight_refinement_campaign.py test_benchmark_comparative.py`
  - `python -m unittest test_benchmark_comparative.py test_benchmark_beir.py -v`
  - a synthetic real-engine smoke covering baseline/offline/hybrid, with the root adapter checksum unchanged before and after
- The previous run at `experiment_runs/weight-refinement-20260306-session28-isolated` was intentionally abandoned after the contamination diagnosis.
- The fresh clean relaunch is `experiment_runs/weight-refinement-20260306-session28-clean`.
- The clean run started at `2026-03-06T13:42:14` with wrapper PID `50404`, runner PID `86244`, and Phase 1 child PID `72276`.
- Early clean-run evidence:
  - `manifest.json` created successfully with `baseline_adapter_snapshot: null`
  - `phase1_standard_sweep.log` is updating normally
  - Phase 1 is ingesting SciFact into the per-run Qdrant path `experiment_runs/weight-refinement-20260306-session28-clean/qdrant/phase1_scifact_db`
- Completed short-run outputs before the interruption:
  - Phase 1 sweep winner: `learning_rate=0.01`, `threshold=1`, `noise_scale=0.2`, `epochs=5`, improving SciFact NDCG from `0.6289` to `0.6766` (`+0.0477`)
  - Phase 2 distillation produced no retrieval delta across teacher weights `0.3`, `0.5`, and `0.7`; baseline, offline, and hybrid all stayed at mean NDCG `0.7553`, while offline pretraining time ranged from `171.7s` to `216.3s`
  - Phase 3 multitask results were stable (`avg_jaccard=1.0`) but showed zero learning gain; aggregate NDCG was `0.6782` for the small suite and `0.6265` for the medium suite
  - Phase 4 BEIR small favored `baseline` and `random_mask_50pct` (`mean_ndcg_at_10=0.6839`) over `chelation` (`0.5745`) and `online_updates` (`0.5746`), with `online_updates` also incurring materially higher latency (`87.75ms` mean)
- The clean campaign then stranded during `phase4_beir_medium`:
  - existing log content stopped mid-SciFact `online_updates`
  - no `phase4_beir_medium.json`, `phase5_online_ablation.json`, or `SUMMARY.md` was written
  - `manifest.json` stayed frozen at `phase4_beir_small`
- `run_weight_refinement_campaign.py` now supports `--resume-run-dir`:
  - resume mode loads the existing manifest
  - recovers already-completed phases from existing output files
  - executes only missing phases
  - can continue into Phase 5 and launch Phase 6 without replaying the entire campaign
- Resume validation passed with:
  - `python -m py_compile run_weight_refinement_campaign.py test_run_weight_refinement_campaign.py`
  - `python -m ruff check run_weight_refinement_campaign.py test_run_weight_refinement_campaign.py`
  - `python -m unittest test_run_weight_refinement_campaign.py -v`
- Active resumed run state:
  - resume wrapper PID `35364`
  - resumed child PID `47792`
  - child command: `python -u benchmark_beir.py --tier medium --model sentence-transformers/all-MiniLM-L6-v2 --max-queries 50 --output ...\\phase4_beir_medium.json`
  - `logs/phase4_beir_medium.log` was recreated at `2026-03-06 23:39:53`, confirming the resume path re-entered the missing BEIR medium phase

## 2026-03-06 Backlog Triage Findings
- The current repo state still supports the earlier roadmap-audit conclusion: there is no unfinished product-implementation phase outside the computational-storage follow-through.
- `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md` is still aligned with `README.md`: the remaining non-hardware work is experiment execution, result analysis, and optional documentation cleanup.
- The strongest actionable non-test work tonight is cleanup of stale roadmap language in:
  - `REFACTORING_PLAN.md`
  - `COMPLETION_SUMMARY.md`
  - `PR_DESCRIPTION.md`
- Those files still describe deferred "Phase 4" work even though the corresponding capabilities are already present in the codebase, so they now create confusion rather than plan useful implementation work.
- No additional active engineering phase was found via repository-wide backlog-marker search; the remaining concrete product backlog is still:
  - real RP2040 hardware evidence capture when hardware is available
  - the dated retention review window on or after `2026-04-05`

## 2026-03-07 Overnight Orchestration Findings
- The resumed `phase4_beir_medium` process was still live after the priority shifted away from overnight execution. It was intentionally stopped to avoid burning CPU while the session moved to documentation and reviewable PR prep.
- Three isolated PRs were opened instead of mixing code, doc cleanup, and wrap artifacts:
  - PR `#96` `feat: harden weight refinement campaign recovery`
  - PR `#97` `docs: normalize stale roadmap documents`
  - PR `#98` `docs: wrap session28 overnight orchestration`
- PR `#96` contains:
  - real-engine evaluation wiring for comparative and BEIR entrypoints
  - adapter isolation to avoid cross-configuration checkpoint contamination
  - resume support for interrupted campaign runs
  - durable docs for the partial Session 28 campaign results
- PR `#97` converts stale "Phase 4" roadmap language in the old hardening docs into explicit historical notes instead of active backlog wording.
- PR `#98` records the overnight orchestration in the session log, verification log, phase summaries, next-session handoff, and `CLAUDE.md`.
- No test suite was rerun in the overnight session by user direction. Only lightweight non-test validation was recorded:
  - `python -m py_compile ...` for PR `#96`
  - `git diff --check` for PRs `#96`, `#97`, and `#98`
