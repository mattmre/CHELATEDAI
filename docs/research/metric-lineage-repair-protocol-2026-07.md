# Metric-lineage repair protocol

**Protocol ID:** `CHELATEDAI-MLR-1`

**Date:** 2026-07-23

**Status:** corrected BCC-1 local implementation and
`drift_recovery_metrics.ndcg_at_k` wrapper tested; general benchmark API
migration and legacy regeneration not yet executed

## 1. Defect and formal result

For query \(q\), let \(R_q\) be the complete set of documents with positive
relevance, let \(d_1,\ldots,d_k\) be the retrieved ranking, and define binary
gain \(g_i=\mathbf 1[d_i\in R_q]\). Correct binary nDCG@\(k\) is

\[
\operatorname{nDCG@k}(q)=
\frac{\sum_{i=1}^{k}g_i/\log_2(i+1)}
{\sum_{i=1}^{\min(k,|R_q|)}1/\log_2(i+1)}.
\]

The legacy helper receives only the retrieved gain vector and uses
\(m_q=\sum_i g_i\) in the ideal denominator:

\[
\widehat{\operatorname{nDCG}}_{\mathrm{legacy}}(q)=
\frac{\sum_{i=1}^{k}g_i/\log_2(i+1)}
{\sum_{i=1}^{m_q}1/\log_2(i+1)}.
\]

Because \(m_q\le\min(k,|R_q|)\), the legacy denominator is never larger than
the correct denominator. Therefore:

- the legacy score is at least the correct score whenever its numerator is
  positive;
- it is strictly larger when at least one positive is retrieved but not every
  ideal top-\(k\) positive is retrieved; and
- it is equal when the numerator is zero or
  \(m_q=\min(k,|R_q|)\).

Minimal falsifier: with ranking `["a", "x"]` and positives `{"a", "b"}`,
legacy nDCG@2 is `1.0`; correct nDCG@2 is
`0.6131471927654584`.

## 2. Formal repair hypotheses

### MLR-1A — implementation correctness

For every finite ranking, positive-qrel set, and cutoff in the bounded
property-test universe, the replacement implementation equals an independent
definition computed from the equations above, uses all positive qrels for
IDCG, and applies deterministic score-descending/document-ID-ascending ranking
ties.

### MLR-1B — lineage completeness

Every emitted retrieval metric artifact is accepted only if it content-binds:

- metric name and cutoff;
- gain rule;
- ideal-ranking population;
- query inclusion rule;
- qrels snapshot SHA-256;
- implementation SHA-256;
- ranking tie-break;
- input/result binding; and
- caller/runner revision.

An artifact missing any field is `LEGACY_METRIC_LINEAGE_BLOCKED`, even when its
numeric value happens to be unaffected.

### MLR-1C — regeneration impact

For every quarantined lane, corrected regeneration either:

1. reproduces its direction, gate, and uncertainty result within a frozen
   tolerance;
2. changes the quantitative result but not its conservative non-promotion; or
3. changes a decision, in which case the old decision is superseded and the
   lane requires a fresh preregistered evaluation before promotion.

No stored legacy value may select a new threshold, method, or regeneration
scope.

## 3. Exact caller inventory boundary

Known susceptible paths:

- `benchmark_comparative.py`;
- `benchmark_rlm.py`;
- `benchmark_distillation.py` (duplicated implementation);
- `benchmark_evolution.py`;
- `retrieval_fitness_evaluator.py`;
- `run_drift_recovery_experiment.py`;
- `run_phase_c_eval.py`;
- `run_road_course_campaign.py`;
- `run_thousand_query_tuning.py`; and
- the wrapper in `drift_recovery_metrics.py`.

`synthetic_collapse_benchmark.py` is a known safe structural exception because
it defines exactly one relevant document per query. That exception must still
bind its metric lineage before a future result is accepted.

The same hostile check found an adjacent metric-integrity failure:
`benchmark_utils.mean_average_precision_at_k(["a", "a"], {"a"}, 2)` returns
`2.0`. Recall de-duplicates and MRR is unaffected by this exact duplicate, but
AP/MAP trusts ranking multiplicity and can exceed its mathematical maximum.
The migration boundary therefore includes one canonical unique-ranking
validator shared by nDCG, AP/MAP, recall, MRR, and future retrieval metrics;
each metric must additionally retain its own complete relevance population.

The inventory is code-lineage based. Files and artifacts are not quarantined
merely because they contain the text `nDCG`, and they are not cleared merely
because a stored mean looks plausible.

## 4. Separate-slice execution plan

1. **Freeze the repair surface.** Record exact callers, tests, artifact
   producers, qrels loaders, and downstream status documents at one Git SHA.
2. **Introduce an unambiguous API.** The core function accepts ranked document
   IDs plus the complete positive-qrel set. Do not retain a public overload
   that can infer IDCG from top-\(k\) retrieved gains.
3. **Prove the boundary.** Add:
   - the two-positive/one-retrieved regression above;
   - no-positive rejection or explicitly typed exclusion;
   - zero-hit behavior;
   - more-than-\(k\) positives;
   - duplicate document-ID rejection;
   - AP/MAP and every other bounded metric remaining in [0, 1];
   - graded-to-binary conversion tests;
   - canonical score-tie tests; and
   - bounded randomized comparison with an independent reference equation.
4. **Migrate every caller.** Each caller must supply the complete qrels for the
   query, declare its inclusion rule, and emit the metric contract. Remove the
   duplicated distillation implementation.
5. **Run caller-specific negative controls.** The single-positive synthetic
   benchmark must remain bit-identical. A multi-positive fixture must change
   in the predicted direction. Missing qrels or incomplete candidate-only
   relevance must fail closed.
6. **Regenerate by dependency order.** Start with smallest deterministic unit
   artifacts, then D1/D2, H4/H5, swap, Phase C, calibration/knob, road-course,
   tuning/evolution, and finally aggregate status surfaces. Never overwrite a
   legacy artifact; write a new lineage/version and a supersession map.
7. **Recompute decisions from frozen rules.** Re-evaluate gates, intervals,
   option envelopes, and comparator ordering without retuning. Record
   `REPRODUCED`, `QUANTITATIVE_CHANGE_NO_DECISION_CHANGE`, or
   `DECISION_CHANGED_REQUIRES_NEW_PROTOCOL`.
8. **Fresh hostile review.** Attempt incomplete-qrels substitution,
   candidate-only IDCG, duplicate-ID inflation, ranking-tie drift, query
   omission, stale artifact reuse, and status-document drift.
9. **Publish only an exact-tree result.** Tests, generated artifacts,
   supersession map, and status surfaces must all bind the same commit.

## 5. Current bounded execution

The BCC-1 METHOD_DEV builder implements the corrected equation locally and has
unit coverage for the minimal falsifier, positive qrels outside the sampled
corpus, and canonical document-ID ties. Its manifest binds the implementation,
gain, cutoff, all-positive-qrels IDCG population, qrels digest, query inclusion
rule, pack/qrels binding, and tie-break.

The current tree also repairs `drift_recovery_metrics.ndcg_at_k`, whose existing
signature already receives both ranked IDs and the complete relevant-ID set.
Focused regressions cover the minimal two-positive/one-retrieved falsifier,
ideal retrieval, zero-hit and empty-positive behavior, more-than-\(k\)
positives, duplicate ranked IDs, and invalid cutoffs.

These corrections do not clear any legacy lane. The ambiguous
`benchmark_utils.ndcg_at_k` gain-vector API, its direct callers, duplicated
distillation implementation, and repo-wide artifact regeneration remain the
separate P1 migration because they require a common scientific API change and
a supersession campaign.

## 6. Stop conditions

Stop and fail closed if:

- a caller cannot supply complete positive qrels;
- a historical qrels snapshot or runner revision cannot be reconstructed;
- regenerated artifacts would overwrite the only legacy evidence;
- a decision-changing result has no untouched evidence for a new protocol; or
- any authoritative status surface still presents a quarantined value as
  confirmed.

The repair can restore evidence integrity. It cannot, by itself, establish the
correctness of an experimental design, routeability, novelty, or deployment
safety.
