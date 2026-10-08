# IQ-03 — Learned holographic composition under matched information and bytes

**Status:** `QUEUED_DEPENDENCY_REVIEW / STAGE_B_DRAFT / NOT_FROZEN / NOT_RUN`.

**Parent:** `P4-REPRESENTATION`; [RHPC Stage A — byte-preserved reference](../references/rhpc-stage-a-method-dev-protocol-2026-08-25.md). Official implementation and execution admission remain in the separately governed RHPC worktree.

**Owner/reviewer:** unassigned representation researcher / independent coding-and-ML reviewer. Follow [G0](shared-test-contract.md).

## Hypotheses and source boundary

H1: a shared-basis/rotation/residual representation stores **learned** expert updates more compactly at matched task quality than ordinary low-rank/quantized representations.

H2: a recurrent decoder recovers useful compositions from **deployment-available observations** more efficiently than same-information conventional decoders.

These are separate claims. A successful route decoder does not prove weight compression, and reconstructing a deliberately factorized matrix does not prove either learned-model claim. Read [resonator networks, linear HDC codes, and KnOTS](sources.md). Ordinary factorization and coordinate alignment are established.

The [2026-10-08 ResonanceDB comparison](../resonance-memory-prior-art-2026-10-08.md) adds direct phase-aware retrieval prior art to H0 and an exact real-coordinate, energy-calibrated control to H2. Seven constructed reduction checks are available; they establish score equivalence only. Match encoder-supplied phase/role information and total bytes/work before attributing any retrieval advantage to holographic scoring. The pictured RHPC field note maps to this parent lane; the paper does not establish its ordered-route factorization or learned expert compression.

Keep Stage A byte-unchanged: four stations, six experts, 1,296 constructed paths, 1,000-dimensional composite code, official IDs 7/11, and its existing gates. It constructs the observed code from the known route; that is a valid constructed sanity input, not evidence that a deployed encoder can obtain it. Its one-shot/smoothing controls lack that composite and therefore do not establish a same-information decoder advantage.

## Mathematical candidate and accounting

```text
W[l,e] ~= U[l] Q[l,e] D[l,e] Q[l,e]^T V[l]^T + R[l,e]
c(route) = product_l permutation_l(code[e_l])
```

Freeze dimensions, dtype, orthogonality parameterization, residual sparsity/index encoding, code precision, and order binding. Different station permutations are required to distinguish positions in a commutative product. Account for U/V, every Q/D/R, sparse indices, codebooks, positional maps, router/encoder, decoder, caches, and all precision metadata. A continuous complex coordinate is not a free unbounded-information symbol.

## Work items and dependencies

- [ ] **H0 — source and opportunity audit (2–4 days):** confirm Stage-A disposition from verified public evidence without rerunning or opening official results speculatively. Review full resonator and linear-code methods/code. Produce future `rhpc-information-budget.md` documenting exactly what each decoder observes.
- [ ] **H1 — learned-bank compression, after G0 and reviewed Stage B:** independently train a small expert bank on a declared compositional task; do not generate weights using the candidate decomposition. Start with four stations/six experts and small hidden dimensions, not a large MoE. Compare task-preserving compression across a frozen rate/distortion grid before adding holographic routing.
- [ ] **H2 — same-observation decoding:** first compare decoders on a separately admitted constructed METHOD_DEV fixture giving every decoder the identical composite, codebook, hints, and allowed iterations. This does not amend Stage A or count as learned utility. Then train an input-to-composition observation encoder on MODEL-TRAIN and run new held-out routes without true-route-derived composites or hints.
- [ ] **H3 — combined utility:** only if H1 and deployment-observation H2 survive, freeze the combined task/cost experiment with no oracle route at inference. Report whether the combined effect exceeds its components.
- [ ] **H4 — transfer:** new expert identities/task families and a separately scoped small LM/LoRA study. No automatic Qwen Stage C or public novelty claim.

Do not bypass RHPC official admission: the signed EGV terminal public-chain seal, bound restore receipt, clean exact source, explicit opt-in, and absent output directory remain required. Stage A is CPU-only and cannot stop/reconfigure the restored service. New Stage B must explicitly inherit or independently resolve applicable predecessor/resource requirements; a root-worktree scout is not an escape route.

## Controls and proposed data

Compression controls: uncompressed learned bank; per-expert truncated SVD; shared-basis factorization; ordinary aligned low-rank updates; quantized low-rank updates; candidate without rotation; candidate without residual. Match actual serialized bytes, not only nominal rank.

Decoder controls: one-shot router; beam/coordinate search; resonator-style decoder; exhaustive route search for small-instance accuracy ceiling; learned conventional decoder with equal observation access. Linear HDC codes use different code structure: compare their own valid decoder in a separately labeled whole-system information/bit-budget comparison, not by feeding them incompatible complex codes.

Proposed pilot: three independent bank-training seeds, with route IDs deterministically split 50/10/20/20 into disjoint MODEL-TRAIN/DEVELOPMENT/SELECT/REPORT partitions, and disjoint task instances in each. The exact rounded route counts and any template/family holdouts must be fixed in the manifest. A route split tests unseen combinations of known experts, not unseen experts. H4 needs new identities. Route accuracy is diagnostic when multiple routes implement the same function; functional task quality is primary.

## Metrics, gates, and falsifiers

For H1, proposed primary gate: at least **20% fewer total stored auxiliary bytes** than the strongest ordinary representation at task accuracy noninferior within 1 percentage point (95% paired interval), with total p95 inference latency at most 1.2x. Compare a SELECT-fixed operating point, not a REPORT-chosen frontier point.

For deployment-observation H2, proposed gate: at least **20% less total inference work** including observation encoding and decoding at task accuracy noninferior within 1 percentage point, with no p95 latency regression. Report exact route recovery, task outcome, coverage/false lock, cycles, reconstruction error, and all code/router memory. Timing without equivalent output quality is not an advantage.

Choose one primary stage comparison before confirmation; H1 and H2 positive claims need the shared multiplicity/uncertainty plan. Reject or park the candidate if classical matched controls erase the gain, decoder observations embed the held-out answer, quantization destroys the benefit, the learned encoder cannot produce useful observations, or extra storage/work was omitted. Stable repeated routes are not correctness; include dead-state and two-cycle controls.

## Budget and handoff

### Integrated mathematical tests — 2026-10-08

[MA0/MA3/MA4 and cases MA3-T1–T4, MA4-T1–T3](math-findings-integration-2026-10-07.md) extend H0's source/information audit and future H1/H2 tests. Require a constructible finite code before using Littlewood existence as a method; report aperiodic/circular sidelobes, cross-code coherence, and decoding under matched observations/precision. The binary circulant classification is an unverified dependency until admitted and excludes neither approximate nor complex codes. The bit ledger includes every U/V/Q/D/R allocation, sparse index, positional map, decoder/router/optimizer/cache, and replay record, with physical shared-state deduplication. Keep Stage A byte-unchanged and learned compression, deployment decoding, and H3 utility separate. Handoff adds finite-construction provenance, diagnostic/control tables, and the complete rate/distortion/byte ledger.

Estimate 2–4 workweeks to a useful learned/same-information discriminator and 8–16+ for stronger utility evidence, excluding predecessor delays. Start with the small bank and bounded 1,296-route space; calculate all allocated arrays/search expansions before admission. Use no larger grid to rescue a failed mechanism.

Future artifacts: Stage-A/source disposition; observation equivalence table; learned-bank/task provenance; exact compression byte ledger; code precision and decoding search budget; oracle-separated route records; complete rate/distortion results; and the shared evidence package. Unresolved before dispatch: learned task/encoder design, full-code comparator applicability, allocation caps, route/instance split precision, and independent Stage-B review.
