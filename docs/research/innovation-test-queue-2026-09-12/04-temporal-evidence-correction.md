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

- [ ] **T0 — readiness map (1–3 days):** bind each predecessor to its current exact artifact and disposition; mark missing evidence without opening new held-out fixtures. Select the minimum candidate component set and identify the strongest ordinary composite control.
- [ ] **T1 — component qualification, after G0:** perform only the missing admitted predecessor studies in their own order and caps. Preserve the existing synthetic negatives and ordinary-equivalence results. Do not rebuild killed mechanisms to fill the composite architecture.
- [ ] **T2 — bounded temporal task:** after the full entry gate, create a separately frozen corpus of claim lifecycles with delayed observations, valid-time/transaction-time disagreement, correction then retraction, incorrect retractions, later reinstatement, copied/paraphrased sources, hidden common causes, missing dependency edges, cycles, and unaffected claims. Proposed maximum: 5,000 claims, 25,000 events, 50,000 edges, and propagation horizon eight.
- [ ] **T3 — survivor-only ablations:** compare full selected candidate with removal of each included component, all on matched candidate sets, metadata, retrieved tokens, model calls, and history retention. This is evidence about marginal utility, not automatic proof of every component.
- [ ] **T4 — state-replay extension:** only under a separate protocol and resource allocation, compare full corrected-history recomputation, checkpoint-and-replay, and any proposed selective repair. Test cache contamination and repeated correction cycles. No model-state inverse is assumed.

Proposed T2 sampling: 300 DEVELOPMENT, 400 SELECT, and 800 REPORT lifecycle groups; all questions, paraphrases, copies, and temporal variants from a lifecycle stay together. Hold source lineages, entities, and generation templates apart across splits. Use a deterministic temporal/reference evaluator, independently reviewed for impossible or ambiguous questions. No generator-only true provenance is given to the candidate while withheld from controls; oracle lineage is an explicitly separate upper bound.

## Controls and endpoints

Controls: ordinary vector RAG; recency/version-aware RAG; bitemporal provenance ledger; that ledger plus copy-aware corroboration and ordinary truth-maintenance repair; flat index with identical metadata; full rebuild/replay; and component ablations. Do not make destructive deletion the only comparator.

For this draft, propose **stale-or-false admission rate on post-correction factual queries** as EK7's single primary endpoint. Freeze the exact positive/negative definition, denominator, abstention treatment, and weighting before REPORT. Unsafe-action decisions are separate sandboxed secondary tests, not actual external actions and not a substitute primary endpoint selected after results.

Preserve EK7's numerical boundaries: at least 25% relative primary-error reduction against the strongest composite baseline; clean recall loss at most two percentage points; p95 latency at most 1.5x; auxiliary index storage at most 2x. Proposed confirmation additionally requires a positive paired 95% interval for absolute error reduction and a clean-recall noninferiority bound within its margin. Review this inference specification before freeze; it does not rewrite the source card.

Measure current/as-of accuracy, contradiction handling, calibration/risk-coverage, stale survivors, collateral demotions, repair work, correction latency, recall, storage, and process-tree RSS. Exact store/history/hash/replay invariants are correctness gates. For EK5's sparse-invalidation subclaim retain exact full-recompute parity, zero unsupported survivors/non-descendant demotions, and at most 25% of full-recompute node/edge work.

Stop if stronger nondestructive controls match the benefit, gains come from refusing nearly everything, candidates receive privileged metadata, unsupported claims survive missing-edge stress without explicit abstention/fallback, or no included component contributes independently. Baseline error zero or insufficient uncertainty resolution is inconclusive, not a 25% improvement.

## Budget and handoff

### Integrated mathematical tests — 2026-10-08

[MA0/MA5 and cases MA5-T1–T3](math-findings-integration-2026-10-07.md) add optional T0-selected controls for independent Gaussian streaming measurements and specified three-state reconstruction trees. Default disposition is `RELATED_ONLY` until an applicable question is selected; these are not extra EK7 prerequisites. Copies, common causes, replay, caches, or a different observation/prior model invalidate an automatic theorem transfer. Preserve all 15 predecessor dispositions and bitemporal/copy-aware controls. Any T4 state extension uses MA4's complete retained-bit/replay ledger and MA7's numerical receipts. Handoff adds the selected/related-only disposition, assumptions, copying/independence checks, and retained information/replay costs.

Estimate 1–2 workweeks for the useful T2 pilot **after** component readiness; 4–8 for stronger practical evidence. Missing predecessors are additional work, not covered by that estimate.

Preserve existing card-specific ceilings: EK1 is under 512 MiB/five minutes per seed; common RB-13 limits and stricter card limits apply; EK7 is one process at a time under 1 GiB after preflight/checkpoint testing. Start with a deterministic surrogate. Do not hide model memory in an uncounted service to fit the cap; a real model exceeding it needs a separate reviewed resource contract. No GPU or model download is required for Waves 0/1.

Future artifacts: 15-card predecessor map; exact candidate component selection; immutable lifecycle/event corpus manifests; independent evaluator; post-correction/per-lineage results; complete ablations; replay/hash integrity checks; and shared evidence/resource package. Unresolved before dispatch: actual surviving components, primary labeling contract, uncertainty/power, exact safe source, and whether/when a real-model extension is justified.
