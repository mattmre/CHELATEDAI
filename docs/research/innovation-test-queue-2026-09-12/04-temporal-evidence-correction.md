# IQ-04 — Temporal evidence correction and bounded retraction

**Status:** `QUEUED_DEPENDENCY_REVIEW / NOT_FROZEN / NOT_RUN`.

**Parent:** `P3-UTILITY`; [RB-13/RB-14 queue](../evidence-kernel-masked-subplane-experiment-queue-2026-07.md).

**Owner/reviewer:** unassigned evidence-memory researcher / independent evaluator and state-integrity reviewer. Follow [G0](shared-test-contract.md).

## Hypothesis and novelty boundary

A survivor-only evidence-memory system reduces stale/false admissions after corrections at competitive recall, latency, and storage compared with ordinary version-aware, provenance-aware, copy-aware truth maintenance. The practical opportunity is error recovery; bitemporal storage and dependency tracking themselves are established. Read [Donto and dynamic copying/truth discovery](sources.md).

Separate three levels of reversibility:

```text
store view: V(valid_time, transaction_time) = replay(eligible immutable events)
derived claims: repaired state = recompute(changed dependencies, declared rules)
model state: corrected state = replay(corrected visible history, full checkpoint)
```

The first equality does not prove the other two. A prompt record's deletion does not remove its influence from KV/hidden state, and no local operation retracts already emitted text or external actions. Exact model replay requires compatible weights, caches, RNG and declared execution determinism. Keep model-state retraction as a separate future extension, not an EK7 claim by implication.

## Required predecessors — no new shortcut

Before EK7, retain explicit dispositions for **all 15** named source cards:

`PRW-EK0`, `PRW-EK1`, `PRW-EK2`, `PRW-EK3`, `PRW-EK4`, `PRW-EK5`, `PRW-EK6`, `PRW-BIL1`, `PRW-DDF1`, `PRW-RCM1`, `PRW-ISI1`, `PRW-SPU0`, `PRW-SPU1`, `PRW-VAR1`, `PRW-REV1`.

Require the separate RSS/checkpoint/kill-resume guard. A disposition may be a retained characterization, reduction, rejection, or independent component result; it need not be a positive mechanism result. A component enters the pilot only after its own independent confirmation and nonzero component-level ablation contribution. Freeze the component set before EK7 REPORT. T3 checks the already selected set's incremental integration benefit; it cannot select new components or drop losing ones after REPORT. Exclude optional RB-14 coalition/transport subcells unless their own entry gates and a reviewed pilot amendment admit them.

Retain the source dependency order: EK1 after EK0; EK2/EK3 after EK1; EK4 after EK3/BIL1; EK5 after EK1/EK2/BIL1; EK6 after EK3; DDF1 after EK2; RCM1 after EK2/BIL1; ISI1 after EK3; SPU1 after SPU0; REV1 after EK1/RCM1. Previously completed sanity tests do not automatically satisfy these candidate-survival gates.

`CD-R13-01` and `CD-R16-01` record unsafe rejected mutation/promotion implementations. Do not import those branches as reusable working components. Resolve exact ownership, immutable provenance, atomicity, and scope in a fresh authorized implementation lane before any such reuse.

## Work items and proposed fixture

- [ ] **T0 — readiness map (1–3 days):** bind each predecessor to its current exact artifact and disposition; mark missing evidence without opening new held-out fixtures. Select the minimum candidate component set and identify the strongest ordinary composite control. Resolve the required ScaleMAS baseline's implementation, fidelity, and resource admission under the [2026-10-15 amendment](#scalemas-and-correction-aware-comparisons--2026-10-15).
- [ ] **T1 — component qualification, after G0:** perform only the missing admitted predecessor studies in their own order and caps. Preserve the existing synthetic negatives and ordinary-equivalence results. Do not rebuild killed mechanisms to fill the composite architecture.
- [ ] **T2 — bounded temporal task:** after the full entry gate, create a separately frozen corpus of claim lifecycles with delayed observations, valid-time/transaction-time disagreement, correction then retraction, incorrect retractions, later reinstatement, copied/paraphrased sources, hidden common causes, missing dependency edges, cycles, and unaffected claims. Include the required ScaleMAS comparison and popularity-by-validity cases below. Proposed maximum: 5,000 claims, 25,000 events, 50,000 edges, and propagation horizon eight.
- [ ] **T3 — survivor-only ablations:** compare full selected candidate with removal of each included component, all on matched candidate sets, metadata, retrieved tokens, model calls, and history retention. Include the required ScaleMAS and BM25-only arms, ordinary repair controls, and separately labeled typed-relation comparisons below. This is evidence about marginal utility, not automatic proof of every component.
- [ ] **T4 — state-replay extension:** only under a separate protocol and resource allocation, compare full corrected-history recomputation, checkpoint-and-replay, and any proposed selective repair. Test cache contamination and repeated correction cycles. No model-state inverse is assumed.

Proposed T2 sampling: 300 DEVELOPMENT, 400 SELECT, and 800 REPORT lifecycle groups; all questions, paraphrases, copies, and temporal variants from a lifecycle stay together. Hold source lineages, entities, and generation templates apart across splits. Use a deterministic temporal/reference evaluator, independently reviewed for impossible or ambiguous questions. No generator-only true provenance is given to the candidate while withheld from controls; oracle lineage is an explicitly separate upper bound.

## Controls and endpoints

Controls: ordinary vector RAG; recency/version-aware RAG; bitemporal provenance ledger; that ledger plus copy-aware corroboration and ordinary truth-maintenance repair; flat index with identical metadata; full rebuild/replay; and component ablations. ScaleMAS-style bounded linked memory and its BM25-only ablation are additionally required for T2/T3, subject to the explicit fidelity and admission requirements below. Do not make destructive deletion the only comparator.

For this draft, propose **stale-or-false admission rate on post-correction factual queries** as EK7's single primary endpoint. Freeze the exact positive/negative definition, denominator, abstention treatment, and weighting before REPORT. Unsafe-action decisions are separate sandboxed secondary tests, not actual external actions and not a substitute primary endpoint selected after results.

Preserve EK7's numerical boundaries: at least 25% relative primary-error reduction against the strongest composite baseline; clean recall loss at most two percentage points; p95 latency at most 1.5x; auxiliary index storage at most 2x. Proposed confirmation additionally requires a positive paired 95% interval for absolute error reduction and a clean-recall noninferiority bound within its margin. Review this inference specification before freeze; it does not rewrite the source card.

Measure current/as-of accuracy, contradiction handling, calibration/risk-coverage, stale survivors, collateral demotions, repair work, correction latency, recall, storage, and process-tree RSS. Exact store/history/hash/replay invariants are correctness gates. For EK5's sparse-invalidation subclaim retain exact full-recompute parity, zero unsupported survivors/non-descendant demotions, and at most 25% of full-recompute node/edge work.

Stop if stronger nondestructive controls match the benefit, gains come from refusing nearly everything, candidates receive privileged metadata, unsupported claims survive missing-edge stress without explicit abstention/fallback, or no included component contributes independently. Baseline error zero or insufficient uncertainty resolution is inconclusive, not a 25% improvement.

## ScaleMAS and correction-aware comparisons — 2026-10-15

**Amendment status:** prospective documentation only; `NOT_FROZEN / NOT_RUN`. No baseline, fixture, selector, scoring implementation, replication, or mechanical admission gate is claimed by this addition. Neither `G0-BUILD` nor `G0-RUN` is granted. The [ScaleMAS comparison](../scalemas-temporal-evidence-comparison-2026-10-15.md) records the literature analysis and its limits; the source is [arXiv:2610.16120v1](https://arxiv.org/html/2610.16120v1), especially §§2–3.5 and Appendices B.6/C.3–C.4.

The proposed discriminator is whether collaboration can recover after a widely reused claim loses support: adopt the correction, repair dependent conclusions, and retain unaffected knowledge within matched resources. ScaleMAS supplies a relevant coordination baseline through immutable linked messages, asynchronous agents, bounded search/visit operations, and BM25 plus decaying citation weight. Its unsigned citation signal records attention: a correcting or refuting citation can also increase the cited message's weight. That is a reason to test correction behavior, not evidence that ScaleMAS necessarily produces stale answers. External-benchmark Pass@N and the paper's contradiction-discovery task do not establish the correctness of a selected delivered answer after a correction.

### Required arms and implementation fidelity

The **full ScaleMAS-style bounded linked-memory baseline is required in the T2/T3 design**. T0 must bind either a pinned upstream implementation or an independently reviewed faithful reproduction to exact code/configuration/model revisions and the paper version. Record departures in message schema/linking, asynchronous scheduling, search/visit caps, BM25, citation updates/decay, agent budgets, and stopping. No implementation pin or faithful reproduction has been admitted by this amendment.

An adapted local deterministic proxy must be labeled `SCALEMAS_ADAPTED_SURROGATE`, with its omissions listed. It can support a surrogate-only comparison after its own admission; it cannot be called a replication, satisfy the full multi-agent comparator, or establish a real multi-agent utility claim. Missing upstream code, unresolved reproduction fidelity, or an unaffordable faithful baseline blocks that comparison's admission. A surrogate-only scope reduction requires explicit review before freeze, keeps the full-baseline requirement visibly unresolved, and cannot silently substitute for it in a ScaleMAS-superiority claim.

| Required comparison arm | Purpose and boundary |
| --- | --- |
| Full ScaleMAS-style bounded linked memory | Retain immutable messages/links, bounded retrieval, asynchronous coordination, and the paper's BM25 plus decaying **unsigned** citation weight. Keep refuting/superseding citations subject to the original citation rule; do not silently repair the baseline. |
| Matched BM25-only linked-memory ablation | Remove citation weighting while retaining the same message/link store, access operations, budgets, and coordination policy. Isolate the effect of popularity-weighted access. |
| Ordinary vector/flat and recency/version-aware controls | Retain the existing controls with identical available evidence and metadata; distinguish temporal eligibility from retrieval ranking. |
| Ordinary bitemporal/provenance ledger plus copy-aware corroboration and truth-maintenance repair | Preserve the strongest ordinary composite comparator, including dependency repair and retention of obsolete records for audit. Established lifecycle machinery is not candidate novelty. |
| Typed-relation ordinary comparator | Add explicit support/refute/supersede treatment to an ordinary ledger/retrieval control, using the same observed relation metadata and extraction costs as the candidate. Freeze a separate signed-score or eligibility rule. This is an adapted comparator, not the original unsigned ScaleMAS arm. |
| Selected survivor-only candidate and component ablations | Evaluate only the T0-selected, independently qualified component set. Separate typed-relation effects from any additional candidate mechanism by comparison with the typed ordinary arm. |
| Full rebuild/replay and separately labeled oracle upper bound | Preserve full-recompute correctness/work comparisons. Generator-only truth or lineage is evaluator-only and may appear only in the segregated oracle arm, never in deployable method selection. |

Keep signed/typed and unsigned citation arms distinct in configuration, receipts, and results. If relation labels are inferred rather than supplied as common observable metadata, all compared methods receive the same permissible inputs and their inference cost is charged. No candidate-only truth labels, future corrections, hidden lineage, or authority annotations are permitted. The primary comparison remains against the strongest ordinary composite selected under the existing contract; required ScaleMAS and ranking ablations are additional comparisons, not a replacement primary endpoint chosen after results.

### Proposed lifecycle matrix

Vary popularity and factual validity independently, crossing popular/unpopular with valid/invalid claims while matching evidence availability and event timing. Freeze how popularity is produced and counted, and distinguish repeated copies from independent corroboration. Populate these prospective cases within the existing T2 corpus and split caps; this table is not a fixture manifest or evidence of implemented coverage.

| Case | Proposed contrast and measurement |
| --- | --- |
| Early popular error followed by correction | A false claim is widely reused before observable corrective evidence arrives; measure stale admissions, correction adoption, and repair latency while retaining obsolete records for audit. |
| Popular valid and unpopular valid/invalid controls | Hold truth independent of citation volume; measure whether correction handling preserves valid minority claims and unaffected popular claims. |
| Refutation and supersession citations | Compare citations that support, refute, or supersede a claim. Preserve unsigned weighting in the original arm and measure whether explicit relation handling changes delivered answers. |
| Delayed observations and as-of queries | Separate valid time from transaction/observation time; score only evidence available at each query's declared cutoff, including queries before and after a correction becomes observable. |
| Erroneous retraction and later reinstatement | Measure recovery from an incorrect correction/retraction as well as recovery from the original error; record collateral demotions, abstentions, and unnecessary rereading. |
| Copied versus independent corroboration | Match surface citation volume while varying observable copy relationships; retain hidden-common-cause stress without leaking generator-only lineage. |
| Missing dependency edges and cycles | Test bounded repair, unsupported survivors, explicit abstention/fallback, and full-recompute parity within the existing propagation horizon. |
| Unaffected claims during repeated corrections | Measure retention of valid unrelated conclusions, clean recall, collateral repair work, and repeated correction cost. |

All questions, copies, paraphrases, temporal variants, and popularity interventions for one lifecycle remain in that lifecycle's existing DEVELOPMENT/SELECT/REPORT group. Source lineages, entities, and generation templates remain split apart. Do not add held-out cells or increase the 300/400/800 lifecycle-group proposal to accommodate this amendment without a separately reviewed change.

### Delivered-answer evaluation and matched resources

Freeze one common deterministic final-answer selector, its permitted inputs, tie handling, stopping, and abstention/fallback rules before REPORT. The selector may use only evidence/messages available at the query cutoff; it must not use gold answers, evaluator-only provenance, future events, or per-case correctness to choose an agent's answer. Freeze any permitted selection/tuning using the existing DEVELOPMENT/SELECT boundaries. Log the selected answer and its supporting messages so selection errors can be distinguished from retrieval and repair failures.

The single primary endpoint remains **stale-or-false admission rate on post-correction factual queries**, with its original denominator/abstention freeze requirement. Add selected-answer accuracy as a secondary outcome and report Pass@N only as a separately labeled discovery diagnostic. For contradiction tasks, freeze canonical claim-pair IDs, temporal eligibility, and negative pairs; report exact contradiction-pair precision and recall, including false-positive contradictions, separately from discovery success. Also retain current/as-of accuracy, abstention/risk-coverage, clean recall and unaffected-claim retention, correction latency, repair work, and the existing integrity gates. These measures do not authorize a second primary endpoint.

Match evidence access, candidate sets, source/relationship metadata, model/tokenizer, final-answer selector, observation schedule, retained history, retrieval limits, and aggregate token/tool/model-call budgets across comparable arms. Count search and visit operations, message creation, indexing, relation extraction, verification, dependency repair, replay, failed attempts, storage, and wall time; do not treat bounded per-call retrieval as a bound on total collaboration cost. A real multi-agent comparison additionally freezes agent count/concurrency and scheduling allowances, or explicitly reports any unmatched resource axis without making a matched-resource claim.

Separate fixed-message replay from any later admitted live-agent comparison. Replay uses identical frozen candidate/message banks and observable metadata at each query cutoff; it isolates retrieval, eligibility, and repair behavior and cannot establish live-agent utility. A live-agent extension matches exogenous evidence inputs, observable source metadata, models, agent/resource budgets, and cutoff schedules, while each arm produces its own messages independently. Log the size and content of each arm's resulting message/candidate bank. Do not force identical endogenous trajectories or expose one arm to another arm's messages, future outputs, or later corrections to manufacture matched banks.

The existing 25% relative primary-error reduction, two-percentage-point clean-recall margin, 1.5x p95 latency, 2x auxiliary storage, inference specification, and all 15 predecessor dispositions remain unchanged. The initial deterministic surrogate remains **one process at a time under 1 GiB**, with the existing preflight/checkpoint and stricter predecessor limits. This amendment allocates no agents, GPU, model service, model download, or additional compute for an experiment. Faithful multi-agent integration requires its own reviewed resource admission; hidden service memory or a proxy result cannot satisfy that requirement.

Future handoff additions are the baseline source/fidelity record and any explicit scope reduction; separate unsigned/signed configurations; matched-budget and observation ledgers; frozen selector and pair-scoring specifications; lifecycle/popularity manifests; and per-group selected-answer, contradiction, abstention, correction, and resource results. These are artifacts to create only after the relevant admission, not current deliverables. No experiment commands are introduced here.

## Budget and handoff

### Integrated mathematical tests — 2026-10-08

[MA0/MA5 and cases MA5-T1–T3](math-findings-integration-2026-10-07.md) add optional T0-selected controls for independent Gaussian streaming measurements and specified three-state reconstruction trees. Default disposition is `RELATED_ONLY` until an applicable question is selected; these are not extra EK7 prerequisites. Copies, common causes, replay, caches, or a different observation/prior model invalidate an automatic theorem transfer. Preserve all 15 predecessor dispositions and bitemporal/copy-aware controls. Any T4 state extension uses MA4's complete retained-bit/replay ledger and MA7's numerical receipts. Handoff adds the selected/related-only disposition, assumptions, copying/independence checks, and retained information/replay costs.

Estimate 1–2 workweeks for the useful T2 pilot **after** component readiness; 4–8 for stronger practical evidence. Missing predecessors are additional work, not covered by that estimate.

Preserve existing card-specific ceilings: EK1 is under 512 MiB/five minutes per seed; common RB-13 limits and stricter card limits apply; EK7 is one process at a time under 1 GiB after preflight/checkpoint testing. Start with a deterministic surrogate. Do not hide model memory in an uncounted service to fit the cap; a real model exceeding it needs a separate reviewed resource contract. No GPU or model download is required for Waves 0/1.

Future artifacts: 15-card predecessor map; exact candidate component selection; immutable lifecycle/event corpus manifests; independent evaluator; post-correction/per-lineage results; complete ablations; replay/hash integrity checks; and shared evidence/resource package. Unresolved before dispatch: actual surviving components, primary labeling contract, uncertainty/power, exact safe source, and whether/when a real-model extension is justified.
