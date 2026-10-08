# Research priorities and testing schedule — 2026-09-12

<!-- innovation-test-queue-2026-09-12:start -->
## Detailed test-plan extension — 2026-09-12

### 2026-10-08 mathematical-test integration

The [MA integration packet](innovation-test-queue-2026-09-12/math-findings-integration-2026-10-07.md) and [24-case register](innovation-test-queue-2026-09-12/math-test-cases-2026-10-07.csv) attach the reviewed OpenAI findings to the existing schedule. Preparation covers MA0 source/theorem admission and MA7 receipt design. C0/C1, S0/S1, H0/H2, T0-selected controls, and W0/W1 receive the corresponding prospective bound, state, code, accounting, and dynamics specifications. Existing order, practical-effect endpoints, ordinary controls, and run gates are retained; these are specified future cases, not experimental results.

The six directions from the subsequent innovation/time assessment now have [prepared test packets and local queue entries](innovation-test-queue-2026-09-12/README.md), with [primary sources](innovation-test-queue-2026-09-12/sources.md), mathematical reductions, matched controls, proposed data/metrics, stop/go criteria, future task IDs, ownership roles, and time/budget assumptions. This supersedes the statement below that detailed completion tasks remain entirely for the next planning step, **only for these six directions**. No run date, named execution owner, compute allocation, or protocol freeze has been set.

The new prior-art review narrows generic novelty: SimMerge already predicts merge selection/order; AlphaEdit and KnOTS cover protected weight-update subspaces and coordinate alignment; GraphCON/GRAND/DataMUX/resonator work covers the broad dynamics/representation ingredients. Use the packets' stronger ordinary controls. These are literature findings, not new experimental results.

Clarification to the independence language below: CRSV, LIR, and SRS remain logically distinct, but the current source protocol's **confirmatory** hierarchy requires CRSV then LIR then SRS. Failed upstream gates stop that sequence; an independent follow-up requires a separately reviewed scope. Nothing in this schedule removes it. Likewise, the new temporal packet preserves every EK7 predecessor disposition and existing resource ceiling.

Research-value ranking does not override readiness order. Keep P0 and the independently gated QSCCI lane; prioritize composition as the first scientific candidate after admission, temporal correction as the recommended practical candidate, and scoped/holographic pilots conditionally. Wave and fixed-graph entries begin with reduction review and remain compute/hardware held.

**2026-09-17 addition:** the queue now also holds `IQ-07`, an [externally supplied non-autoregressive UI decision-engine proposal](innovation-test-queue-2026-09-12/07-nonautoregressive-decision-engine.md). Its supplied Phase-0 battery was reviewed and measured invalid (two of three gates unreachable as coded; the third passes only on constructed noise). It is queued for harness repair in the preparation slot. It maps into `P4`/`P3`/`P5` rather than opening a new lane, and its training and procurement plan is held. The "six directions" wording above covers only IQ-01 through IQ-06.

**2026-09-30 addition:** the [portfolio reorganization and big-winner review](portfolio-reorganization-and-big-winner-review-2026-09-30.md) reorders this schedule around one organizing thesis, *pre-execution compatibility*.

The new order is:
1. Stage 0: validity reconciliation with `main`.
2. Stage 1: an executable baseline ladder plus a development-only gap preflight.
3. Stage 2: the encoder-upgrade residual-gap test, ahead of any typed-authorization (TRA-1) campaign.
4. Stage 3: IQ-01.
5. Stage 4: IQ-04 *or* EGV-v2.

The review also corrects several states recorded on this branch:
- The metric-lineage repair never reached `main`. PR #292 merged without it, and `main` still computes the retrieved-list ideal ranking.
- PR #293 merged on `main` and needs a re-probe.
- PR #295 is still open.
- PR #308 is **merged**, not an open draft; this supersedes the EGV statement below.

The operator chose on 2026-09-30 to fix the prime-ring float-FFT tie contract rather than drop it. That fix waits for the operator's go. Nothing in this addition authorizes a run, a freeze or a PR mutation.
<!-- innovation-test-queue-2026-09-12:end -->

**Status:** `LOCAL PLANNING UPDATE / NOT AN EXECUTION AUTHORIZATION`.
**Scope:** Portfolio reprioritization begun 2026-09-04, extended on 2026-09-12 for the recurrent-looped transformer proposal. Detailed completion tasks, owners, estimates, run dates, and compute commitments are the next planning step.

## Direction and authority

Center the portfolio on **predictable, reversible adaptation with demonstrable task benefit**. Geometric and hardware ideas are possible mechanisms; they must beat ordinary alternatives before becoming architecture commitments.

This document governs the proposed order of work in the reviewed portfolio. It does not replace frozen protocols, change result classifications retrospectively, authorize REPORT access, or override [next-session's block flag](../next-session.md). That flag remains `BLOCKED` for expired `CD-MLR-01`, `CD-R13-01`, and `CD-R16-01`. Existing execution, resource, review, and publication gates remain in force.

Use readiness gates rather than invented calendar dates. The default recommendation is one admitted compute campaign at a time, with read-only analysis and permitted documentation/preparation alongside it. Actual capacity and concurrency require a fresh resource check and operator allocation; this update did not inspect live Sparks or cloud jobs.

## Ordered schedule

Scientific importance and run order differ: QSCCI is a small discriminator, not the flagship. The new transformer idea deserves early design work, not first access to a large training budget.

| Order / ID | Work | Entry condition | Output that earns the next step |
| --- | --- | --- | --- |
| 0 / `P0-VALIDITY` | Reconcile source contracts, repair affected metric lineage, retain valid negatives, and disposition carried debt. | Current checkout and exact approved scope identified; no broad feature expansion. | Auditable metric caller/artifact lineage, task solvability where relevant, explicit observation timing, preserved data splits, and applicable gate disposition. |
| 1a / `P1-QSCCI` | Freeze/review the new nuisance-rank discriminator before considering a bounded run. | Independent final draft review, exact source/fixture freeze, mechanical SELECT/REPORT barriers, and execution/resource authorization. | A valid result about distinct selector behavior and oracle-assisted controllability; negative results close the tested variant. |
| 1b / `P1-ARCH-RLT` | Specify and reduce the new scoped recurrent-state candidate; compare close prior work. | Documentation-only design scope. Training remains unapproved. | [Design study](recurrent-scoped-transformer-design-review-2026-09-12.md) converted later into one small falsifiable protocol, not a large combined architecture. |
| 2 / `P2-COMPOSITION` | CRSV home effects, LIR ordered composition, and incremental SRS diagnostics. | Valid measures and independent splits; real task/model evidence; predictor-availability contract; applicable approvals. | Separate verdicts on each hypothesis, unseen-chain prediction, uncertainty, and worst-domain collateral effects. |
| 3 / `P3-UTILITY` | Select one practical flagship: temporal correction/evidence kernel (recommended default) **or** EGV-v2. | Selection at the next planning step; chosen lane's own predecessors, baseline, source, solvability, and admission gates satisfied. | Held-out practical benefit against the strongest ordinary baseline at bounded cost. Do not launch both simply because both have infrastructure. |
| 3b / `P3-ARCH-PILOT` | Small learned scoped-update experiment, if its specification survives. | Its own reviewed protocol, ordinary recurrent baseline, matched-resource plan, and explicit run allocation. Imported CRSV/SRS claims need transfer validation. | Incremental utility at fixed depth before adding adaptive depth, memory, or representation machinery. This competes for a later bounded pilot slot, not a foundation-model budget. |
| 4 / `P4-REPRESENTATION` | RHPC and remaining prime-ring discriminators. | Same-information ordinary controls and each original protocol's mathematical/resource prerequisites. | A specific surviving representation benefit rather than recovery of a constructed answer or a change of basis. |
| 5 / `P5-REDESIGN` | BCC new-family design, localized assimilation cleanup, and fixed-graph pitch correction. | A new explicit mechanism and corrected claim boundary, not reuse of an exhausted confirmation set. | A coherent future protocol worth admitting; otherwise retain as design notes. |
| Separate engineering lane | Disk-first/quantization/resource scheduling. | Needed by an admitted scientific study or a separately scoped engineering request. | End-to-end bytes, latency, memory, and quality evidence; no promotion of engineering readiness into scientific utility. |

These are not dependencies on positive results across unrelated hypotheses. CRSV failure does not logically kill LIR or SRS; QSCCI failure does not logically kill a label-blind recurrent architecture. A non-retrieval diagnostic does not inherit every nDCG artifact-regeneration dependency. However, no lane may bypass the repository's applicable block/approval rules or its own frozen prerequisites. Within `P2`, establish the ordinary predictor ladder before testing the incremental geometric predictor.

## Workstream dispositions and decisive tests

### P0 — validity before claims

- Follow the [metric-lineage repair protocol](metric-lineage-repair-protocol-2026-07.md): IDCG uses all positive qrels; rankings must have canonical unique document IDs before evaluating retrieval metrics. A repaired local function is not proof that every caller or old artifact is repaired.
- Keep affected legacy quantitative retrieval claims quarantined until the corresponding artifacts are regenerated and traced to corrected callers. Do not choose new confirmatory thresholds from quarantined numbers.
- Freeze what every method observes and when. `QRELS_FREE` is not equivalent to `PRE_SEARCH`; a post-update diagnostic is not automatically a pre-update compatibility predictor.
- Separate execution correctness, solvability, mathematical sanity, learned mechanism evidence, real task benefit, and novelty. Each needs its own evidence.

### QSCCI — earliest small diagnostic, narrow conclusion

The [follow-up draft](qwen-scope-nuisance-rank-separation-preregistration-draft-2026-08-16.md) remains `REVIEW-DRAFT / NOT FROZEN / NOT AUTHORIZED FOR EXECUTION`. Its new 12 DEVELOPMENT / 24 SELECT / 24 REPORT base-row design is not authorized by this schedule. Existing v4 SELECT/REPORT data are exhausted for confirmation.

The [original intervention protocol](qwen-scope-chelated-causal-intervention-preregistration-2026-08-16.md) supplies an oracle sign to all compared interventions. This can test controllability and nuisance separation, not an autonomous correctness detector. Distinct selected directions and manipulation checks are prerequisites for the intended mechanistic comparison, not proof of task utility. A deployable label-blind policy would be a separate hypothesis and protocol.

### CRSV / LIR / SRS — scientific core

Use the [current composition protocol](crsv-onion-method-dev-protocol-2026-07.md), preserving its distinct claims and thresholds. CRSV asks about domain/provenance compatibility; LIR asks whether ordered combinations need more than singleton effects; SRS asks whether geometry adds predictive information beyond those ordinary terms.

Before running, specify when every pair, reversal, prefix, and trajectory feature is obtained. A held-out chain's measured damage cannot also be an input to its prospective prediction. Validation must hold out the relevant chain/task/domain units, not just rows derived from the same unit. Common coordinates are necessary for comparing operators, but do not establish useful prediction.

Retain the protocol's LIR 10% and incremental SRS 5% MAE-improvement gates and their uncertainty requirements; do not silently replace them with a favorable descriptive correlation. A future compatibility certificate is not implemented merely because metadata shape/hash/expiry checks exist. Authentication, byte-level verification, revocation, and calibrated scientific coverage are different obligations.

The 2026-09-12 literature review adds [recent finite-time recurrent-dynamics work](https://arxiv.org/html/2608.18222v1) to the comparator review for SRS. Scope differs from adapter compatibility, so this is not a disproof of SRS; it does narrow novelty claims about stability/transient diagnostics themselves. Any needed comparator amendment must be reviewed before freezing/running, not introduced after observing REPORT.

### Temporal correction / evidence kernel — recommended practical flagship

The [RB-13/RB-14 queue](evidence-kernel-masked-subplane-experiment-queue-2026-07.md) is a source of individually gated hypotheses, not evidence that a full evidence engine has passed. Prefer the correction/retraction use case because its failure costs and ordinary controls are concrete.

Compare with version-aware retrieval, provenance tracking, copy-aware truth maintenance, and matched flat metadata. Include imperfect provenance, copied sources, delayed or incorrect retractions, and repeated correction lifecycles. Keep every predecessor's disposition explicit; include only independently surviving components with nonzero ablation contribution.

Preserve EK7's existing declared endpoint and gate: at least 25% relative reduction in false/stale/unsafe promotion, at most two percentage points clean-recall loss, at most 1.5x p95 latency, and at most 2x auxiliary storage. This summary does not authorize skipping EK7's entry gates or CPU/memory caps. OBS1/COA1/CTX1/SPU1 remain mechanism/sanity work until a named policy beats appropriate ordinary controls on independent evidence.

### EGV-v2 — alternative flagship, source reconciliation first

The old v1 task required an evaluator-only 128-bit oracle unavailable from model-visible inputs. Treat the affected zero-performance results as **diagnostic of an unlearnable benchmark**, not a capability ranking. Admission failures remain admission evidence.

The amended local design is in [the integrated EGV worktree](D:/GITHUB/CHELATEDAI-EGV-AVO-INTEGRATED-20260830/docs/research/egv-avo-nemotron-recommissioning-preregistration-2026-08-25.md). It requires a new learnable corpus and blind production-sandbox solvability proof before task-bearing model calls. Then follow the minimal matched pilot before any trajectory collection or training.

As checked 2026-09-12, [PR #308, “docs(research): preregister EGV AVO/Nemotron recommissioning”](https://github.com/mattmre/CHELATEDAI/pull/308) remains an open draft at `8ce7feb987490e8d39479673cba18604e376568d`; it is not interchangeable with the amended local design. Resolve an exact reviewed execution source before admission. The amended persistence reset contract is within-task, not lifelong/cross-task memory. Model-bundle comparisons are not clean causal tests of parameter count. H6 trace-conditioned scheduling replay is not a measured model-utility gain.

### Recurrent transformer — new design lane, conditional small pilot

Use [ARCH-RLT-01](recurrent-scoped-transformer-design-review-2026-09-12.md). The upstream release is a design report, not a runnable validated model. Our candidate is scoped recurrent revision plus separately tested halting and versioned evidence; the combination is unproven.

First distinguish token recurrence from additional inner computation and make the complete state/cache contract explicit. Then compare a small learned scoped update with ordinary gated recurrence at fixed depth. Only surviving effects justify separate adaptive-depth and memory additions. Stable states can be wrong; a common rotation can be an equivalent parameterization; projection after dense computation does not remove its cost.

Do not start with the upstream illustrative 48+48-layer design, full-scale RL, or a combined prime-ring/RHPC/memory/adapter system. Relevant prior work already includes adaptive recurrence, learned memory, and stability-guided halting. The opportunity is a narrower, measured incremental advantage.

### RHPC and prime-ring — contained discriminators, no scaling escape

RHPC's [Stage-A protocol](D:/GITHUB/CHELATEDAI-RHPC1/docs/research/rhpc-stage-a-method-dev-protocol-2026-08-25.md) is constructed METHOD_DEV, not learned-model compression or utility evidence. It generates the observed composite from a known route; a future inference system still needs a deployable way to obtain that information. Compare decoders with the same observations and learned matrices at matched rank/bytes. Preserve its existing EGV terminal-seal/restore and Spark admission prerequisites; no alternate run is authorized here.

For prime-ring, follow the [remaining-hypotheses record](prime-ring-remaining-hypotheses-2026-07.md): resolve the exact/float midpoint-tie contract, A1 global conjugacy plus semantic acquisition cost, and G2-H auxiliary minimality/candidate-specific advantage before larger studies. JO1 fixed-stack equivalence remains closed. Larger primes, meshes, or a full expanded factorial do not resolve an unspecified mechanism.

### Redesign or hold

| Line | Current disposition | Requirement to re-enter testing |
| --- | --- | --- |
| BCC-1 | Current feature/policy family `CUT`; no confirmatory REPORT may be opened from it. | New genuinely pre-search features and query-role-correct bridge design, new derivation/data boundaries, corrected metrics, and the original cycle/stop limits. [Source](bcc1-preregistered-confirmatory-protocol-2026-07.md). |
| Localized assimilation / issue #106 | Historical “implementation ready” and numerical novelty claims are not current clearance. | Keep event-triggered adaptation, failure-fingerprint routing, and quantization-tier calibration as hypotheses. Replace unsupported physics/geometry claims with explicit objectives, ordinary baselines, and collateral-effect tests. [Historical plan](../ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/planning/2026-05-reconciliation/issue-106-localized-assimilation-research-plan.md). |
| Fixed-graph / Extropic pitch | `REWRITE BEFORE EXTERNAL USE`; hardware experiment on hold. | Automorphism-congruent Ising families are exact relabelings. Compare against compile-once caching plus I/O remapping; establish actual graph, ports, sampler fidelity, transfer/reprogramming cost, and binary-encoding overhead. The current [one-pager](fixed-graph-orbit-compilation-one-pager.html) is not revised or endorsed by this schedule. |
| Disk-first / quantization / scheduler | Engineering support, not a scientific-efficacy result. | Real cold/warm I/O, bytes moved, end-to-end latency, memory, and task-quality controls. [PoC scope](../../computational_storage_poc/README.md). |

Specific corrections to the localized-assimilation framing: scalar changes in optimizer speed do not establish trajectory curvature; orthogonal adapter coordinates do not imply rigid-body inertia dynamics; entropy of an eigenvalue distribution is not an MDL objective without a coding/data-fit model. [DoRA](https://arxiv.org/abs/2402.09353) decomposes magnitude and direction, not an implicit Stiefel constraint. [QLoRA](https://arxiv.org/abs/2305.14314) trains adapters through a frozen quantized base; “QLoRA quantizes the trained adapters” is not an accurate description of that method. These are interpretation corrections, not new experiments.

## Results that do not return to the active queue unchanged

| Result scope | Retained conclusion | What it does not establish |
| --- | --- | --- |
| RB-15 tested factorial | Frozen interaction/worst-case/hysteresis/settling gates failed. | Not a proof against every nonlinear or distributed mechanism. |
| JO1 fixed stack; BIL1 boolean union; SPU0 factorization | Retain their ordinary-equivalence/negative boundaries. | A new name or larger size does not create a new hypothesis. |
| VAR1 cross-phase result | Retain the tested null; keep any separately open smooth subproblem distinct. | No general cross-phase benefit. |
| QSCCI v4 | No selector-specific chelation advantage; confirmatory fixture exhausted. | Not proof of autonomous task correction, even where shared interventions beat random controls. |
| EGV-v1 affected tasks | Diagnostic-only unlearnable benchmark. | Not a model-capability negative and not permission to reuse its scores for v2 comparisons. |
| BCC current family | METHOD_DEV cut and role/feature-timing defects retained. | No retrospective creation of a held-out REPORT. |

Historical artifact bytes and protocols remain unchanged. Latest local status/dispositions take precedence over old queue-header summaries; contradictions must be resolved before execution, not by selecting the most favorable wording.

## Source snapshot and handoff

- Root worktree: `codex/prime-ring-onion-method-dev`, HEAD `65c9085cd048e8a7351a53e87666fdd5639e612b`, with pre-existing tracked and untracked changes preserved.
- Amended EGV worktree: HEAD `2a08a637663c4be8fc146e2221b2234679419410`; RHPC worktree: HEAD `305986427850dcb393a6078900201a7aa84c5b33`. A checkout HEAD alone does not freeze its working-file contents; future execution must bind the full reviewed payload.
- Upstream RLT source: `1bee93a9b01c21bea0c7a50ce3f6619f24731e19`, report dated 2026-09-12. External papers inform design and prior-art boundaries; their results were not replicated here.
- No held-out fixtures were read, no experiments or model calls were launched, no live resource reservations were made, and no external publication occurred in this update.

Next planning session: turn admitted priorities into dependency-ordered tasks with exact evidence deliverables, owner/reviewer, budgets, and stop conditions; choose the practical flagship and any later architecture-pilot allocation. Do not interpret this schedule as the detailed task list or as execution approval.
