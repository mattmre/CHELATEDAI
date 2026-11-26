# W22-08 Capture–Recapture Audit Residual Pilot: preregistration

- Queue item: `W22-08` (wave slot W22B, CPU only, optional and non-blocking). The queue entry is in
  `D:\GITHUB\CHELATEDAI\docs\research\wave22_queue_additions_2026-11-25.json`.
- Branch `pv7/w23b-w22-08-capture-recapture`, based on `origin/pv7/w16-core` at `a3ebf8ca`.
- Written: 2026-11-26. This file is committed **before** any defect list is extracted and before any estimate is
  computed. If anything in the analysis departs from this file, the report must name the departure.

## What I had seen before writing this

So that the reader knows how blind this preregistration is, here is everything I had read so far:

- the queue entry and the W22 handoff row for W22-08;
- `artifacts/pv7_w23a_seal/W23A_DISPOSITION.md` in full, and the first ~9 kB of `w22_02_claim_coverage_audit.md`;
- the headers, verdict paragraphs, and `#NN` mention counts of the council logs listed below.

From those headers I already know some things about independence. The two PV7 final judges (Carbon, Fable)
reconciled or cited the PV7 peer cross-validation memos. Each PV7 peer memo covered a disjoint slice of
#47–#64. The round-0 council brief pre-seeded all three reviewers with the same list of prior defects.

I have **not** read the bodies of the finding sections closely, and I have not counted findings for any list.

## Question

Can existing, independent, multi-model audit logs bound the number of residual undetected defects in the bundles
of `docs/research/COMPLETE_64_BUNDLE_EVIDENCE_SCORECARD.json`, using capture–recapture estimators?

## Unit and canonical defect ID

A **defect** is a statement by an auditor that something in an audited bundle is wrong, unsupported, overstated,
missing, or must be disclosed or fixed. That covers the bundle's evidence, code, receipt, verifier, registry row,
scorecard row, or headline text.

These do **not** count as defects:

- confirmations, such as "hash matches" or "holds";
- strategy or roadmap advice;
- proposals for new experiments that are not tied to an existing flaw;
- restatements of a defect that the auditor says was already fixed before their audit.

Severity is ignored. Blockers, majors, minors and nits all count.

The canonical ID is `B<ordinal>:<class>`:

- `<ordinal>` is the scorecard ordinal (1–64), or `XC` for a cross-cutting defect inside the family's shared scope
  that names no single bundle (for example a shared loader or verifier).
- A finding that names several bundles expands to one ID per named bundle.
- Several findings by the same auditor that map to the same ID collapse into one capture.
- A finding that clearly holds two separable defects may be split into two IDs. Every split is logged.

The class comes from this closed set, frozen here:

| Class | Meaning |
|---|---|
| C01 | Verifier or gate does not load or check the evidence it is credited with (replays a drycheck or fixture instead of the live receipt, or re-reads stored fields instead of recomputing). |
| C02 | Verifier, CLI or gate is a no-op or vacuous (exit 0 without verifying; for example, no `__main__`). |
| C03 | Status, tier, label, or headline overstates the evidence (fixture or synthetic presented as live or wild, or wrong denominator inflating a claim). |
| C04 | Status or text understates the evidence, or lags it (stale status or stale hash). |
| C05 | A failed or missed gate or endpoint is undisclosed, ignored, or reported as a pass. |
| C06 | Undisclosed tuning, configuration change, or in-sample selection. |
| C07 | Provenance gap (host, hardware, commit, code-tracked state, or digest missing or wrong). |
| C08 | Metric or comparator asymmetry (unfair or cost-unmatched baseline, or a metric that depends on an implementation choice). |
| C09 | Control or baseline arm invalid, missing, or miswired. |
| C10 | Spec, code, or doc drift (thresholds, loaders, or prose disagree with code or artifact). |
| C11 | Incomplete input, inventory, sample, or coverage (subset, partial inventory, missing rows or arms). |
| C12 | Test gap (vacuous test, or the test misses the production path). |
| C13 | Fabricated, simulated, or synthesized number in an evidence path presented as measured. |
| C14 | Statistical procedure error or deviation (wrong test, resample count, multiplicity, or CI). |
| C15 | Leakage or independence violation (split leakage, shared state between oracle and system under test, label or oracle consultation). |
| C16 | Correctness bug in code or math not covered above. |
| C17 | Other. Free text, and the reason must be given. |

Every finding record must carry:

- the source path and line number;
- a verbatim quote of at least 20 characters that occurs on that line;
- the ordinal(s);
- the class;
- a one-line alignment note.

The runner **mechanically verifies** each quote against the source file. A record whose quote does not match is
dropped and counted as an extraction error.

## Candidate list families

The scopes below are declared from each round's brief or header. A list enters a family's estimating set only
if both of these hold:

1. Its declared bundle scope **equals** the family's shared scope. Findings outside the shared scope are dropped
   and the drops are logged.
2. **Procedural independence:** the list's own text does not show that its author read, reconciled, adjudicated,
   or was briefed with another list in the same set. A list that fails this check is removed before kill rule K1
   is applied, and the quote that shows the dependence is logged. A round brief that seeded every reviewer with
   the same prior defect list counts as dependence for all of them.

| Family | Lists | Declared scope (to be confirmed with quotes during extraction) | Role |
|---|---|---|---|
| F-PV7 | `pv7-post-live-xval-grok47-xhigh`, `pv7-post-live-xval-sonnet55-high`, `pv7-final-judge-carbon4-med`, `pv7-final-judge-fable52-med` (`pv7-final-judge-r2-fable52-med` is a same-model follow-up: it is excluded and that is logged) | #47–#64, split into strata S_A = {47–55, 61–64} and S_B = {56–60} | **Primary.** This is the post-v7 scorecard scope. |
| F-R5 | `panel-r5-{astra-ultra, claude-opus-ultrathink, grok-xhigh, neon-xhigh}` (the synthesis is excluded because it is derived) | Draft v3, 27 bundles | Secondary |
| F-W8 | `wave8-r1-{grok47-xhigh, neon-high, opus55-high, sol62-xhigh}` | The 5 v5.0 Part VI bundles | Secondary |
| F-W4 | `wave4-r1-{grok47-xhigh, neon-high, opus55-high, sol62-xhigh}` | Wave-4 bundles under audit | Secondary |
| F-W14 | `wave14-code-xval-{grok47-xhigh, neon4-high, sol62-high, sonnet55-high}` | Bundle #46's pre-live D11 runner and formal code | Secondary. Code audit, before the live run. |
| F-W12 | `wave12-xval-*` | Each bundle covered by ≤2 lists | Excluded by K1 by construction. Shown for completeness. |
| F-W15 | `wave15-postgpu-xval-{neon4, opus55}` | Bundle #46 | Excluded by K1: only 2 lists. |
| F-R0 | `r0-{astra, carbon, neon}` | Flagship code and whitepaper at `b37e6a55` | Excluded under the procedural-independence rule: the shared brief pre-seeded every reviewer with the prior Astra defect list. Its scope is also pre-scorecard code and prose, not scorecard rows. |
| W23A | `artifacts/pv7_w23a_seal/*` | W22-02 covers the 31 `verified-sealed` bundles (ordinals ≤43). W22-03 covers #41, #45, #46, #53 and #62. The disposition nits cover #54 and #62. | No family has equal scope, so W23A cannot enter any estimating set. It is used as an **external, later miss check** on F-PV7 (see V2). |
| R1/R2 breakthrough memos | — | — | Not on disk. The handoff names thread IDs only. Recorded as unavailable. |

## Estimators

Every family is analysed separately. F-PV7 is analysed separately for each stratum.

- **Pairwise** (every pair in the estimating set):
  - Lincoln–Petersen `N = n1·n2/m`;
  - Chapman `N = (n1+1)(n2+1)/(m+1) − 1`, with Seber's variance and a normal 95% CI;
  - overlap `m / min(n1, n2)`;
  - Jaccard.
- **Multi-list** (k ≥ 3):
  - **Chao2** incidence lower bound. `f0 = ((k−1)/k)·f1²/(2 f2)`. When `f2 = 0`, use
    `((k−1)/k)·f1(f1−1)/2`. The 95% upper bound comes from Chao's (1987) log-normal interval.
  - **Poisson log-linear models**, fitted by IRLS on the 2^k − 1 observed capture-history cells:
    - the independence model;
    - one model for each single two-way interaction;
    - the predicted count of the all-zero cell.

    A model that does not converge, or that predicts a non-finite count, is reported as such.

## Kill rules

These are applied per family (per stratum for F-PV7). Each is a pass/fail gate.

- **K1 (lists).** Fewer than 3 lists remain in the estimating set after the scope and procedural-independence
  filters. Outcome: **KILL**.
- **K2 (overlap).** Any pair in the estimating set has `m / min(n_i, n_j) > 0.80`. Outcome: **KILL**.
- **K3 (dependence sensitivity).** **FAIL** if either of these holds:
  - (a) Across the independence model and every single-two-way-interaction model, the predicted undetected count
    `f0` is non-finite, or `max f0 / max(min f0, 1) > 3`.
  - (b) Under assumed positive list dependence θ = 2, applied as `f0(θ) = θ · f0_independence`, the residual
    exceeds `S_obs`, the number of distinct defects observed. In that case the bound is uninformative.

  Estimates at θ ∈ {1, 2, 4} are always reported.
- **K4 (held-out miss validation).** For each list `i` in the estimating set:
  - compute the Chao2 95% upper bound on the residual using only the other `k−1` lists;
  - compare it with `u_i`, the number of distinct defects that only list `i` found.

  **FAIL** if `u_i` exceeds that upper bound for any `i`.

A family **PASSES** only if K1–K4 all pass.

**Overall verdict.**

- **ADVANCE** (the method is viable on existing logs) if at least one family passes. The report must then say
  which bundles, and which audit date, the bound applies to.
- **KILLED** if no family passes.

F-PV7 is the family that answers the post-v7 scorecard question. A pass by a secondary family shows only that the
method works on that older scope.

## Reported checks that do not gate the verdict

- **V2 (W23A later-miss check, F-PV7).** Count the W23A defects on S_A and S_B bundles whose canonical IDs are not
  in the union of the F-PV7 lists. Compare that count with each stratum's residual estimates.
- **As-if-independent numbers.** For any family that fails K1 only because lists were removed as dependent, the
  estimators are also computed on all lists. Those numbers are labelled **non-gating**.

## Deliverables

All under `artifacts/pv7_w23b_w22_08/`:

- `extraction/*.json`: per-family finding records with quotes;
- `w22_08_results.json`;
- `W22_08_REPORT.md`, which includes the full aligned defect table.

Code:

- `capture_recapture_audit.py`;
- `scripts/run_w22_08_capture_recapture.py`;
- `tests/test_capture_recapture_audit.py` (unittest only).

Extraction of F-W14 may be done by a separate sub-agent using this exact schema and taxonomy. If so, the report
discloses it.
