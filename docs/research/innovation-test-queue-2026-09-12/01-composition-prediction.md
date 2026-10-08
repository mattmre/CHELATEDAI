# IQ-01 — Prospective composition and collateral-error prediction

**Status:** `QUEUED_PROTOCOL_REVIEW / NOT_FROZEN / NOT_RUN`.

**Parent:** `P2-COMPOSITION`; [CRSV/LIR/SRS source protocol](../crsv-onion-method-dev-protocol-2026-07.md).

**Owner/reviewer:** unassigned research implementer / independent methods reviewer. Follow [G0](shared-test-contract.md).

## Hypothesis and novelty boundary

Preserve three distinct questions: domain/provenance interaction (CRSV), extra predictive value from combination/order effects (LIR), and incremental geometric/finite-horizon information (SRS). The potentially useful extension is calibrated prediction of collateral errors on unseen valid compositions at lower total cost than evaluating each composition. A prevention policy is a later intervention, not evidence supplied by a good predictor.

Read [A1–A3 and B6](sources.md). SimMerge already predicts merge configurations; KnOTS already aligns LoRA coordinates; PermDoRA reports limits of static geometry in its setting. The generic ideas are not novel. A narrower claim must name the distinct object, information access, target, transfer population, and measured advantage.

## Mathematical and information contract

For candidate outcome Y and frozen predictors M0, M1, M2, retain the source ladder:

```text
M0 = domain/provenance main effects + singleton effects
M1 = M0 + allowed pair/order features
M2 = M1 + allowed SRS features
relative_MAE_gain(A -> B) = 1 - MAE(B) / MAE(A)
```

Zero/near-zero baseline MAE makes a relative-improvement target unidentified; predeclare the handling and report absolute errors. Do not manufacture improvement with an epsilon chosen after evaluation.

First specify whether a composition is a checkpoint merge, an ordered application `F_j(F_i(h))`, or a typed replacement at different layers. Simple additive weight updates commute: `W + A + B = W + B + A`. Reordering their addition is not a nonlinear order mechanism. Invalid layer/ABI reorderings are contract violations, not difficult model examples.

| Feature | Required availability | Leakage trap |
| --- | --- | --- |
| Component norms, principal angles, declared commutators | Before combination; one proved coordinate system | Equal tensor shapes do not establish aligned coordinates. |
| Functional similarity | Approved unlabeled individual-component probes; all probe work charged | Using held-out answers or a candidate chain's task score. |
| Pair/singleton outcome summaries | Only explicitly permitted, disjoint calibration observations | Measuring the target REPORT chain/prefix and calling it a predictor. |
| Reversal, useful-signal loss, transient summaries | Either training-fitted estimates or deployment-available probes with defined timing | Post-outcome explanations relabeled as prospective features. |

Resolve the key open design issue before freeze: for unseen replacement families, measured pair outcomes may not exist. A **zero-shot** transfer claim cannot use them. A **calibrated** transfer study may allow new-family calibration pairs only in a separately declared split with charged cost and an amended transfer claim. Do not silently change the original unseen-family protocol to obtain these features.

## Work items and dependencies

- [ ] **C0 — claim/timing audit (2–3 days):** classify composition objects, construct the full feature-availability table, compare SimMerge's full methods/code, and specify any necessary protocol amendment. Output `composition-claim-chart.md` and `feature-availability.csv` as future artifacts.
- [ ] **C1 — correctness and feasibility, after G0/C0:** inspect/reuse `crsv_experiment.py` and `tests/test_crsv_experiment.py` without treating existing synthetic tests as scientific evidence. Add future adversarial tests for commuting controls, invalid contracts, wrong qrels, train/REPORT contamination, missing features, nonfinite values, and coupled-group resampling.
- [ ] **C2 — bounded real-task METHOD_DEV:** use corrected retrieval with four serving/provenance domains and the source's valid identity/singleton/pair/triple/four-layer structure. Begin with a logged balanced subset only if separately admitted as a scout; a subset is not completion of source Phase A. Full Phase A retains its three alternatives/layer, three mutation families, three seeds, and at least 50 groups per balanced cell.
- [ ] **C3 — confirmation:** freeze source Phase B or a separately reviewed amendment, including six domains (four SELECT/two untouched REPORT), six mutation families (three SELECT/three REPORT), required chain coverage, at least 200 groups per required REPORT cell, and prospective power analysis. Preserve one-sided 97.5% CRSV → LIR → SRS gatekeeping. No confirmation is scheduled here.
- [ ] **C4 — calibrated decision/control extension:** only after predictive evidence, separately test accept/abstain or scoped suppression against ordinary thresholding, routing, and full candidate evaluation. Hold out new component identities and a new model family; do not claim transfer from new-query splits alone.

## Controls, endpoints, and decisions

Controls: identity/native; wrong-domain components; M0; regularized predictor on ordinary functional/structural probes; M1; M2; and exhaustive candidate evaluation as a costed reference. Match tuning opportunities. A SimMerge-inspired adapted baseline is labeled adapted, not an exact reproduction. Freeze which baseline is the primary competitor before REPORT.

Preserved confirmation gates:

- CRSV: home effect at least 0.005 absolute binary nDCG@10; favorable inversion sign; one-sided 97.5% lower bound above zero; simultaneous worst-domain lower bound above -0.02; all validity gates pass.
- LIR: at least 10% relative MAE reduction over M0 and one-sided 97.5% block-bootstrap lower bound of the absolute improvement above zero. Negative-shell AUROC at least 0.70 is the source's secondary endpoint, not a replacement primary outcome.
- SRS: at least 5% incremental MAE reduction over M1 with its specified positive lower bound. Geometric correlation alone does not pass.

Also report baseline-relative prediction MAE, worst-domain errors, risk/coverage, calibration, and total probe/selection cost. Proposed *additional* novelty discriminator: at least 10% relative MAE reduction versus the strongest ordinary probe predictor at no greater total selection cost, with a positive paired improvement interval. This requires pre-freeze comparison/multiplicity review; passing the original ladder alone cannot establish superiority to newly identified prior art.

If CRSV fails, stop that confirmatory sequence; if LIR fails, cut its predictive/certificate claim; if SRS fails, retain surviving ordinary terms without a geometric story. Missing exchangeability, leaked outcomes, bad metrics, or incomplete required cells yield invalid/inconclusive evidence, not a valid negative. Independent downstream research requires a new reviewed scope, not bypassed hierarchy.

## Budget and handoff

### Integrated mathematical tests — 2026-10-08

[MA0/MA1 and cases MA1-T1–T3](math-findings-integration-2026-10-07.md) extend C0's timing/coordinate audit and C1's correctness backlog with fixed-operator amplification descriptors, a valid outer numerical-range enclosure, ordinary norm controls, and wrong-lineage/post-outcome refusal. [OM1/OM2](sources.md#mathematical-findings-integration--2026-10-07) supply the source boundary. C2/C3 may use a new descriptor only after its acquisition cost, availability, feature-selection opportunity, and protocol amendment are reviewed. Preserve M0/M1/M2 and the CRSV → LIR → SRS hierarchy; a tighter mathematical bound alone does not pass any predictive endpoint. Handoff adds the MA task/case IDs, theorem-admission ledger, certificate slack/refusals, and descriptor cost.

Plan 1–2 workweeks to a first useful METHOD_DEV discriminator, not the entire Phase-B matrix. Stronger evidence: roughly 6–12 workweeks after readiness. Profile a scout before admitting the full matrix; use the shared proposed compute ceiling only if sufficient. No oversized campaign is authorized to satisfy these estimates.

Future handoff: source/metric bindings; operator and availability chart; split/group and chain-coverage manifest; full M0/M1/M2/ordinary-control predictions; separate CRSV/LIR/SRS verdicts; uncertainty; probe-cost ledger; and all shared evidence artifacts. Unresolved before dispatch: exact datasets/components, unseen-family feature availability, amended comparator hierarchy, power, and measured resource limits.
