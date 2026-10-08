# Innovation research test queue — 2026-09-12

**Queue ID:** `IQ-20260912`

**Status:** `PLANS_PREPARED / NOT_FROZEN / NOT_AUTHORIZED_FOR_EXECUTION`

**Scope:** The six directions in the innovation-versus-time review, plus `IQ-07`, an externally supplied proposal queued on 2026-09-17. This is the detailed planning extension of the [portfolio schedule](../research-priorities-and-testing-schedule-2026-09-12.md), not a second execution scheduler.

Start with the [shared test contract](shared-test-contract.md) and [primary-source register](sources.md). Each linked packet contains the hypothesis, mathematical boundary, controls, proposed data design, work items, endpoints, stop rules, and handoff artifacts. Checkboxes describe future work and are intentionally unchecked.

## Authority and ownership

The user's instruction authorizes gathering documentation, preparing test plans, and queueing them. No run, model download, protocol freeze, fixture creation/access, paid cloud allocation, new Codex task, scheduled automation, publication, push, or PR mutation is authorized here. The [current block/debt surface](../../next-session.md) remains authoritative. Resolve applicable gates; do not reinterpret a queue entry as permission.

Planning owner: current research-planning task. Future execution owner: an operator-assigned research implementer, currently **unassigned**. Protocol/statistical reviewer: a separate reviewer, currently **unassigned**. Resource and run-admission owner: operator. No assignment or deadline is implied by naming these roles.

## Dependency-ordered queue

`G0` refers to the shared contract's separate `G0-BUILD` and `G0-RUN` gates: authorized harness construction first; final source/data/analysis freeze and explicit resource/run admission only after the harness exists and is reviewed. Applicable source/metric/preservation/debt restrictions govern both. Documentation preparation can precede either; these are not software-enforced launch states.

| Queue item | Existing portfolio lane | Next queued work and dependency | Current execution disposition |
| --- | --- | --- | --- |
| `IQ-00` — validity and source readiness | `P0-VALIDITY` | Apply the shared contract; bind exact source, metrics, data access, reviewer, and measured resource plan. | `BLOCKED_ON_EXISTING_GATES`; no debt marked closed. |
| `IQ-01` — [composition prediction](01-composition-prediction.md) | `P2-COMPOSITION` | C0 timing/operator audit, then C1–C2 bounded METHOD_DEV after G0; C3 confirmation only under the preserved CRSV → LIR → SRS hierarchy. | `QUEUED_PROTOCOL_REVIEW`; highest scientific-value/time candidate, novelty unconfirmed. |
| `IQ-02` — [scoped recurrent correction](02-scoped-recurrent-correction.md) | `P1-ARCH-RLT`, then `P3-ARCH-PILOT` | S0 full-state contract first; S1 correctness and S2 learned scope pilot after G0. Imported IQ-01 predictors need separate transfer evidence. | `QUEUED_PROTOCOL_REVIEW`; full new-transformer training held. |
| `IQ-03` — [learned holographic composition](03-learned-holographic-composition.md) | `P4-REPRESENTATION` | H0 Stage-A/source disposition and same-information audit; H1/H2 only after G0 and a separately reviewed learned Stage-B contract. | `QUEUED_DEPENDENCY_REVIEW`; existing Stage-A official admission unchanged. |
| `IQ-04` — [temporal evidence correction](04-temporal-evidence-correction.md) | `P3-UTILITY`, RB-13/RB-14 | T0 predecessor map, T1 admitted component tests, then T2 survivor-only EK7 after every named disposition and G0. | `QUEUED_DEPENDENCY_REVIEW`; recommended practical flagship, not selected for a run. |
| `IQ-05` — [wave/diffusion multiplexing](05-wave-diffusion-multiplexing.md) | Design review; conditional IQ-02 ablation | W0 reduction/claim chart first. W1 numerical checks and W2 matched pilot require a surviving distinct mechanism, IQ-02-compatible harness, and G0. | `QUEUED_REDUCTION_REVIEW`; architecture/compute allocation held. |
| `IQ-06` — [fixed-graph congruence](06-fixed-graph-congruence.md) | `P5-REDESIGN` | F0 relabeling and port audit, then any admitted F1 algebra/software control. F2 hardware only for a surviving claim with vendor access and G0. | `QUEUED_REDUCTION_REVIEW`; external pitch and hardware experiment held. |
| `IQ-07` — [non-autoregressive UI decision engine](07-nonautoregressive-decision-engine.md) | `P4-REPRESENTATION` (RHPC), `P3-UTILITY`/RB-13-14 (kernels), `P5-REDESIGN` (chelation) | Y0 battery repair and ordinary-baseline restatement first; Y1–Y3 mechanism tests after G0-BUILD; Y4 integration after G0-RUN. | `QUEUED_HARNESS_REPAIR_REVIEW`; supplied Phase-0 battery **measured invalid** — two of three gates unreachable as coded. Training/procurement held. |

Priority is not a claim that every earlier scientific hypothesis must pass before unrelated design work can continue. Within IQ-01, however, the existing protocol's confirmatory statistical hierarchy is binding. Within IQ-04 and RHPC's official lane, named predecessor requirements are binding. A negative upstream result can require a newly scoped protocol; it cannot be erased by renaming a task.

## Suggested work order

1. **Preparation slot:** IQ-00 plus C0/S0/T0/H0/W0/F0. Read-only literature/algebra preparation and documentation may be organized together; do not launch the numerical checks listed in the packets during preparation.
2. **First scientific compute slot:** IQ-01 bounded METHOD_DEV, if admitted. Existing `P1-QSCCI` remains the earlier small discriminator where its own new-fixture gate is ready; this queue neither duplicates it nor authorizes its run.
3. **Practical slot:** IQ-04, only when its predecessor map is resolved. EGV-v2 remains the existing alternative; no EGV source/solvability decision is implied here.
4. **Conditional pilots:** IQ-02, then IQ-03, each admitted independently. IQ-05 can use a later ablation slot only if W0 identifies more than known mixing dynamics. IQ-06 usually ends at the reduction/baseline memo unless an actual new advantage is specified.
5. **IQ-07 harness repair:** belongs in the preparation slot, not a compute slot. Its source review is already complete and recorded with reproducible local arithmetic in [`07-review-evidence/`](07-review-evidence/RESULTS.md); the supplied battery is invalid as written, so Y0 must produce a corrected scorecard before Y1–Y3 are worth scheduling. Because IQ-07 reaches into the RHPC and evidence-kernel lanes, it must not be run as a parallel unreviewed copy of `P4-REPRESENTATION` or RB-13/RB-14 — the mapping table in the packet is binding.

**2026-09-30 reorder:** the [portfolio reorganization and big-winner review](../portfolio-reorganization-and-big-winner-review-2026-09-30.md) places two stages ahead of slot 2:
- **Stage 0:** validity reconciliation with `main`.
- **Stage 1:** an executable baseline ladder and gap preflight.
- **Stage 2:** an encoder-upgrade residual-gap test.

IQ-01 moves to Stage 3. IQ-04 remains the recommended practical candidate at Stage 4. IQ-07 stays salvage-only. IQ-05 and IQ-06 stay reduction memos. The packets themselves are unchanged.

Only one compute campaign at a time by default. Readiness can move a ready independent lane ahead of an unready one, but the operator must record the allocation. Nothing here reserves machines or installs an automatic scheduler.

## Mathematical findings integration — 2026-10-08

The [October 7 OpenAI math review](../openai-math-applicability-review-2026-10-07.md) is integrated through the [MA0–MA7 task packet](math-findings-integration-2026-10-07.md) and [24-case test register](math-test-cases-2026-10-07.csv). These extend existing lanes rather than opening another campaign. Cases are specified, not implemented or run.

| Parent | Integrated task | First applicable work |
| --- | --- | --- |
| IQ-00 / audit discipline | MA0, MA7 | Theorem/definition admission ledger and numerical-receipt design |
| IQ-01 C0/C1 | MA1 | Pre-outcome amplification descriptors with ordinary norm controls |
| IQ-02 S0/S1; issue #106 | MA2 | Same-state coefficient bounds, normalization, switching/cache refusal, verifier cost |
| IQ-03 H0/H2 | MA3, MA4 | Constructible finite-code diagnostics and a complete physical-bit ledger |
| IQ-04 T0 | MA5, optional | Explicit Gaussian/tree assumptions and replay/copying controls |
| IQ-05 W0/W1 | MA6, conditional | Fixed versus adaptive dynamics, only if W0's distinct mechanism survives |
| Prime-ring bounded repair | MA7 | Complete exact/direct/FFT tie coverage and arithmetic/build receipts |
| IQ-06 F0; IQ-07 Y0 | Related-work disposition / selected reuse | No new hardware campaign or rehabilitation of the invalid supplied battery |

MA0 and the specification part of MA7 belong in preparation. The September 30 portfolio order, G0-BUILD/G0-RUN, existing scientific endpoints, and original lane predecessors remain in force. New release results stay `NOT_CHECKED` until their specific proof/definition dependencies are verified. Current ordinary controls remain the initial comparators.

## Task dependency index

Arrows mean prerequisites, not automatic dispatch or a demand for positive results where a packet requires only a disposition. G0-BUILD/G0-RUN and external predecessor requirements still apply at the relevant stage. Optional S3/T4/W3 and held F2/F3 remain optional/held even when their predecessors finish. S4 needs S3 only for claims about an S3 addition; the scoped core may transfer independently.

<!-- iq-task-dependencies:start -->
```text
C0 -> C1 -> C2 -> C3 -> C4
S0 -> S1 -> S2 -> S3
S2 -> S4
H0 -> H1 -> H3 -> H4
H0 -> H2 -> H3
T0 -> T1 -> T2 -> T3 -> T4
W0 -> W1 -> W2
S1 -> W2
W0 -> W3
F0 -> F1 -> F3
F0 -> F2 -> F3
Y0 -> Y1 -> Y4
Y0 -> Y2 -> Y4
Y0 -> Y3 -> Y4
Y4 -> Y5
MA0 -> MA1
MA0 -> MA2
MA0 -> MA3
MA0 -> MA4
MA0 -> MA5
MA0 -> MA6
MA0 -> MA7
C0 -> MA1
S0 -> MA2
H0 -> MA3
H0 -> MA4
T0 -> MA5
W0 -> MA6
```
<!-- iq-task-dependencies:end -->

## Time-to-evidence map

Estimates are active work for one focused researcher with Codex and accessible small-model compute, **after readiness**, not measured runtimes or promised calendar dates. First-test effort is included in, not added to, the stronger-evidence range. Existing debt, predecessor implementation, external access, and independent review can dominate elapsed time.

| Item | First useful discriminator | Stronger controlled evidence if positive | Decision at the first limit |
| --- | --- | --- | --- |
| IQ-01 | 1–2 workweeks | 6–12 workweeks | Continue only if valid predictive improvement survives strong controls and feature-acquisition accounting. |
| IQ-02 | 1–2 workweeks | 6–12 workweeks | Continue only for task/collateral benefit; smaller updates or stable states alone do not qualify. |
| IQ-03 | 2–4 workweeks | 8–16+ workweeks | Stop if the advantage requires a truth-derived composite or disappears at matched total bytes. |
| IQ-04 | 1–2 workweeks once required components are ready | 4–8 workweeks after predecessor readiness | Stop if ordinary versioning/copy-aware repair matches the integration. |
| IQ-05 | 1–3 days for W0; any later pilot estimated separately | Not defensibly estimable before W0 | Park if no distinct mechanism survives reduction. |
| IQ-06 | Hours–1 day for F0 | Hardware timetable unknown | Close the exact-relabeling novelty claim; retain only justified implementation questions. |
| IQ-07 | 0.5–1 day for Y0 (already partly done by the review below) | 1–2 workweeks for Y1–Y3; Y4 1–2 further weeks | Stop any mechanism that does not beat its named trivial predictor. Y5 training stays held regardless of Y1–Y3 outcomes. |

Cross-family replication, specialist novelty review, and independently reproduced cost/quality gains are additional requirements for a breakthrough claim. No packet has a breakthrough probability or a promised discovery date.

## Required closeout per item

Each future owner returns: exact scope/source/fixture bindings; completed task IDs; complete result matrix including failed runs; uncertainty and cost tables; ordinary-baseline comparison; one of `INVALID`, `INCONCLUSIVE`, `NEGATIVE_IN_SCOPE`, `SURVIVES_METHOD_DEV`, or `SUPPORTS_FROZEN_ENDPOINT`; and the next permitted step. Existing protocols retain their native result labels, mapped without replacing their bytes.

A documentation packet is not runnable code or a frozen preregistration. Before dispatch, fill every admission field in the shared contract. Scope-specific missing decisions are explicitly listed in each packet.
