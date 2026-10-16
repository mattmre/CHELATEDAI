# ScaleMAS and correction-aware shared research — 2026-10-15

**Status:** `DOCUMENTATION_REVIEW / PROSPECTIVE_TEST_REQUIREMENTS / NOT_IMPLEMENTED / NOT_RUN`.

**Scope:** Preserve the paper comparison and the operator-requested update to [IQ-04](innovation-test-queue-2026-09-12/04-temporal-evidence-correction.md). The October 15 request authorizes these documentation changes and explicitly prohibits starting experiments. It does not grant G0-BUILD/G0-RUN, access to new fixtures, model calls, resource allocation, or a protocol freeze. Existing portfolio order, scientific negatives, and block/debt dispositions are unchanged.

## Assessment

ScaleMAS is a required coordination baseline for IQ-04. The strongest research question is whether correction-aware shared memory improves the reliability of the answer actually delivered while retaining useful reuse of prior work. This is a proposed incremental contribution over ordinary truth maintenance, not a claim of novelty for linked memory or a claim that our implementation already outperforms ScaleMAS.

The [current portfolio](portfolio-reorganization-and-big-winner-review-2026-09-30.md) remains the organizing plan: validity reconciliation and ordinary-baseline admission precede utility claims. This comparison strengthens IQ-04's control set; it does not create a parallel campaign or reopen closed mechanisms.

## Paper evidence and its limits

Primary source: Qi et al., [Scaling Multi-Agent Intelligence with Substrate-Centric Coordination, arXiv:2610.16120v1](https://arxiv.org/html/2610.16120v1), submitted October 12, 2026. Methods, evaluation, ablations, and implementation appendices were inspected; no result was replicated.

ScaleMAS combines immutable linked messages, asynchronous agents, bounded search/visit reads, and BM25 plus decaying citation weight. At eight agents, removing citation weighting lowers BrowseComp-Plus from 60.0% to 40.8% (§3.5). External tasks use Pass@N, not selected-answer accuracy (C.3). Contradiction Audit credits edited-snippet discovery without penalizing false positives (B.6). The 128-agent extension covers two tasks; theorem proving largely plateaus after sixteen (§3.4). Correction/refutation citations also increase the cited message's weight (§2.3; C.4). [Source](https://arxiv.org/html/2610.16120v1)

**Our interpretation:** attention and evidence eligibility must be evaluated separately. An obsolete claim may remain useful to retrieve for explanation or audit while being ineligible as current support. A heavily discussed error is a useful stress case, not proof that ScaleMAS necessarily returns an incorrect answer: readers may follow the correction successfully. Likewise, discovery scoring cannot establish deployed answer reliability without evaluating a fixed, non-oracle answer selector.

## Comparison with current CHELATEDAI research

| Surface | Verified source or current proposal | Implication for the comparison |
| --- | --- | --- |
| IQ-04 / EK7 | [Packet](innovation-test-queue-2026-09-12/04-temporal-evidence-correction.md) is `QUEUED_DEPENDENCY_REVIEW / NOT_FROZEN / NOT_RUN`; it already specifies temporal correction, copy-aware controls, and bounded collateral damage. | Closest scientific fit. Evaluate validity-aware use of shared research, not merely successful retrieval of a once-useful message. |
| Root evidence DAG | [evidence_dag.py](../../evidence_dag.py), lines 1–46 and 265–305, describes query/cluster/actuator attribution. `CORRECTED_BY` denotes a retrieval action. | Do not present this graph as an implemented factual correction/retraction service. |
| Separate EGV lifecycle ledger | local `CHELATEDAI-EGV-AVO-INTEGRATED-20260830/egv/ledger.py`, lines 1319–1450 and 1543–1552, appends correction/retraction rows and excludes invalid roots and declared dependents from the current admissible view. | Source-level lifecycle machinery is a reuse candidate. Trusted-writer inputs and declared dependencies do not independently establish factual truth, bitemporal/as-of retrieval, or model-state reversal. |
| Separate EGV retrieval | local `CHELATEDAI-EGV-AVO-INTEGRATED-20260830/egv/variation/retrieval.py`, lines 115–161, scans current-valid events and sorts by event ID without an explicit result/token cap. | Bounded shared retrieval is not already established by this implementation. Integration and its costs require separate evidence. |
| EGV-v2 evaluation | local `CHELATEDAI-EGV-AVO-INTEGRATED-20260830/docs/research/egv-avo-nemotron-recommissioning-preregistration-2026-08-25.md`, opening status and reset contract, requires source reconciliation, a learnable corpus, and blind solvability before execution. Persistence is within a task/seed coordinate. | Existing infrastructure is not a successful multi-agent utility experiment or cross-task learning result. Invalid affected v1 tasks cannot rank capabilities. |
| Typed lineage authorization | [Representation contracts](../../representation_space.py), especially `BridgeContract`, encode role/lineage compatibility requirements. | Conceptual alignment with checking eligibility before use; measured benefit remains a separate question. |
| RHPC / virtual experts / composition | [IQ-03](innovation-test-queue-2026-09-12/03-learned-holographic-composition.md) retains its own observation, compression, and utility boundaries. | Shared-message coordination supplies no evidence of learned expert compression, representation equivalence, or geometric benefit. |

These are source-inspection findings and planning statuses, not fresh runtime acceptance. The root checkout was `65c9085cd048e8a7351a53e87666fdd5639e612b`; the separate EGV checkout was `2a08a637663c4be8fc146e2221b2234679419410`. Both must be treated as working checkouts rather than execution freezes. The EGV references identify a separate local checkout, not files shipped by this branch. GitHub commit lookup during publication returned no commit for that EGV SHA; a fresh clone of this PR cannot independently inspect those source bodies. The recorded paths, line locations, and file hashes identify the local inspection without claiming public availability or portable reproduction.

Inspected-file SHA-256 bindings, captured October 15:

| File | SHA-256 |
| --- | --- |
| `evidence_dag.py` | `ddb01ede9416a8146a55b8b4edd9ce409ba99ac52528a86dfc9c2f0ce1eef574` |
| `representation_space.py` | `98ee084524c9da755184ce9351ced0fb40a20b25db9b2337d8533c4154c0f902` |
| Separate EGV `egv/ledger.py` | `7c4990471622f4e52256035154f04df3ee30ea1f5bd1f361ed7f0d04440c4c80` |
| Separate EGV `egv/variation/retrieval.py` | `b247ca5484fa8c854f135b56a848a2693a3ea9af82396c42c927e683ebb28383` |

## Proposed discriminating comparisons

The detailed requirements are in the [IQ-04 baseline and case specification](innovation-test-queue-2026-09-12/04-temporal-evidence-correction.md), which remains prose rather than an implemented launch gate. The following explains the scientific purpose of that update.

1. **Separate access, reuse, and validity.** Require ScaleMAS and its matched BM25-only ablation alongside ordinary version/provenance-aware memory, bitemporal/copy-aware truth maintenance, and the independently qualified IQ-04 candidate. Keep relation typing, validity filtering, and candidate components as separately identified treatments. Beating a weak citation-only comparator is insufficient if an ordinary composite matches the result.
2. **Manipulate popularity and validity independently.** Cross high/low reuse with valid/invalid claims while matching relevance and observation access. Add early widely reused errors followed by corrections; copied versus independent sources; delayed observations and as-of queries; erroneous retractions and reinstatement; incomplete dependencies, cycles, and unaffected claims. A correction should not require erasing the audit history.
3. **Score the delivered answer.** Freeze a common answer-selection rule without access to gold labels. Preserve stale-or-false admission as the sole EK7 primary endpoint. Report selected-answer accuracy, discovery success, exact contradiction-pair precision/recall, clean recall, abstention, collateral demotion, correction adoption latency, and calibration separately. Do not select the primary endpoint after observing outcomes.
4. **Match information and total work.** In fixed-message replay, arms receive identical candidate banks and observable metadata at each cutoff. A later admitted live-agent comparison matches exogenous evidence, model/agent budgets, and observation schedules; its independently generated message banks may differ and must be logged rather than shared across arms. Replay cannot establish live-agent utility. Gold labels and hidden source lineage remain evaluator-only. Count publication, retrieval, ranking, indexing, verification, repair, answer selection, and replay costs. A per-read cap alone is not an end-to-end cost result. Separate equal-total-budget and equal-wall-time questions instead of changing agent count and resources together.

The existing EK7 thresholds and all fifteen predecessor dispositions remain binding. The candidate must improve over the strongest admitted ordinary composite, with the specified clean-recall, latency, storage, and uncertainty constraints. Surrogate results establish only the tested surrogate mechanism; they cannot satisfy a claim about real cooperative agents. Any future model/multi-agent extension needs a separate measured resource contract compatible with the packet's existing limits.

## Handoff and unresolved admission work

Next documentation work is T0: record baseline source fidelity and availability, ordinary-control implementations, observable metadata, candidate components, a common selector, and the case/split mapping. Pin the actual baseline implementation and record every local adaptation before a future build/run admission. No runnable ScaleMAS revision was qualified in this review. A locally reconstructed or deterministic adapted baseline must be labeled accordingly, not described as an upstream reproduction.

No experimental code, model, fixture, or scientific output was created for this update. No external result was reproduced. The required comparisons are specified for future admission, not implemented, frozen, or executed.
