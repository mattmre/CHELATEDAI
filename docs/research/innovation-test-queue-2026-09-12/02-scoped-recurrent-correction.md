# IQ-02 — Learned scoped recurrent correction

**Status:** `QUEUED_PROTOCOL_REVIEW / NOT_IMPLEMENTED / NOT_FROZEN / NOT_RUN`.

**Parent:** `P1-ARCH-RLT` then `P3-ARCH-PILOT`; [ARCH-RLT-01 design](../recurrent-scoped-transformer-design-review-2026-09-12.md).

**Owner/reviewer:** unassigned research implementer / independent architecture reviewer. Follow [G0](shared-test-contract.md).

## Hypothesis and mathematical boundary

At fixed refinement depth, a learned bounded subspace gate improves correct targeted revisions while reducing errors on unaffected queries relative to ordinary gated low-rank recurrence. The base computation, available evidence, training budget, and evaluator are matched. This tests activation/state control, not simultaneous weight editing, retrieval redesign, or a new foundation model.

```text
U^T U = I_r
P(h,e) = U diag(g(h,e)) U^T,    0 <= g_i <= 1
h_next = h + eta P(h,e) (G_theta(h,e,cache) - h),    0 <= eta <= 1
(I - U U^T)(h_next - h) = 0
```

Soft P is a contraction, not generally an idempotent projector. The last identity protects one update's coordinate complement only. Later attention/FFN/readout and caches can still change unrelated behavior. It is not a semantic-safety or convergence theorem.

Controls `g=0` and `r=d,U=I,g=1` must reduce to identity and ordinary relaxed recurrence. A fully conjugated change of coordinates must preserve function within declared numeric tolerance. Dense G followed by projection still pays for dense G.

Prior-art requirements: [AlphaEdit, ActAdd, KnOTS, RLT and recurrence sources](sources.md). RLT already proposes gated and low-rank feedback. AlphaEdit belongs in a separate weight-editing comparison if that scope is later introduced; it is not mislabeled an activation-only baseline.

## Proposed smallest learned test

Use a tiny causal recurrent model, initially at most 10 million trainable parameters, on a solvable key/value state-update task: interleave assignments, distractor updates, explicit corrections, and questions about both corrected and unaffected keys. Pair each correction episode with unchanged-key queries. Add a second rule-composition family only after the first controlled pilot survives.

Proposed generation design: up to one million MODEL-TRAIN tokens; 300 independent DEVELOPMENT episodes, 400 SELECT episodes, and 800 REPORT episodes, each expanded into related questions kept in the same group. Use disjoint entity/template/generation seeds; train on short update chains and predeclare longer-chain and late-question strata. These are planning sizes, not frozen data or a power guarantee. Prove solvability with a deterministic interpreter and an ordinary learned baseline before assigning model failure to the new mechanism.

Learn bases/gates on MODEL-TRAIN only, select rank/step size on SELECT under the shared trial cap, and freeze all weights for REPORT. Use fixed depth first (choose one of K=1,2,4 on SELECT with matched opportunities). No label-oracle sign, test-time weight training, adaptive halting, evidence kernel, holographic code, or RL in this first experiment.

## Work items and dependencies

- [ ] **S0 — full-state contract (2–3 days):** specify gate placement relative to normalization/attention/FFN, tensor axes, basis parameterization, token position versus refinement step, all cache reads/writes, reset scope, gradients, pending-token handling, and replay semantics. Output future `scoped-state-contract.md`.
- [ ] **S1 — correctness, after G0/S0:** identity/full-rank reductions; causal masking; no future-token access; prompt-split equivalence; episode reset isolation; no repeated commitment of one token into KV; correct gradients; deterministic full-state checkpoint/replay. Failure blocks training.
- [ ] **S2 — learned fixed-depth pilot:** train/compare arms below with identical data and tuning opportunity. First profile a small DEVELOPMENT shard and calculate the full cost. No full-scale upstream RLT architecture or external checkpoint download by default.
- [ ] **S3 — one addition at a time:** if S2 survives, separately preregister either adaptive halting or explicit corrected-history replay. Halting compares fixed depth, simple margin/entropy/residual-change rules, and established recurrent controls; full-state replay compares ordinary checkpoint/replay. Neither inherits a pass from S2.
- [ ] **S4 — transfer:** new task family, unseen compositions/lengths, then an explicitly selected small existing LM with frozen revision and license. IQ-01-derived controls need deployment-timing and transfer validation; they are not a prerequisite for the simpler S2 study.

## Arms, endpoints, and stop/go rules

Primary arms: ordinary gated low-rank recurrence; learned scoped recurrence; fixed/random shared basis with the same rank; constant gate; and the identity/full-rank reductions as sanity controls. Include a simple trained low-rank residual controller with comparable parameter budget. Report parameter-matched and compute-matched views if one comparison cannot satisfy both.

Proposed primary endpoint: **collateral error rate on unaffected-key queries**. Proposed advancement gate on REPORT: at least 20% relative reduction versus the strongest SELECT-chosen ordinary arm, a 95% paired-group interval for absolute error reduction entirely above zero, and all of:

- targeted correction accuracy noninferior within 1 percentage point, using the relevant confidence bound;
- overall task accuracy noninferior within 1 percentage point;
- p95 total episode latency no more than 1.2x and peak auxiliary memory no more than 1.2x that control;
- no causality, reset, cache, label-access, or invalid-number failure.

These thresholds are proposed minimum-useful-effect choices, not established literature constants. If the baseline has zero collateral errors or the fixture is too easy, return `UNIDENTIFIED/INCONCLUSIVE`; redesign DEVELOPMENT only, not REPORT. Smaller activation movement, lower disagreement, more loops, or a correct final answer after unlimited retries cannot pass.

Report update norm and complementary-subspace change as manipulation checks; stable-but-wrong states, prompt injection-like untrusted evidence strings, confidently incorrect corrections, and unrelated-query regressions are explicit stress cases. Evidence text is data, not authority to change the harness or policies.

## Budget and handoff

### Integrated mathematical tests — 2026-10-08

[MA0/MA2 and cases MA2-T1–T5](math-findings-integration-2026-10-07.md) extend S0/S1 with same-input coefficient-error bounds, live-adapter bias/normalization and epsilon branches, switching products, diverged hidden/KV/RNG state, and output-verification boundaries. Include verifier/reference computation in the cost of any [issue #106 virtual-expert proposal](https://github.com/mattmre/CHELATEDAI/issues/106#issuecomment-6029945796). A fixed-linear certificate does not extend the one-update coordinate identity into semantic safety or trajectory correctness. Any speculative or adaptive pilot is a separately reviewed addition after the existing state/correctness gates. Handoff adds certificate/refusal traces, retained-state bytes, recomputation, full rollback, and measured accepted work per second alongside task/collateral outcomes.

Target 1–2 workweeks to the tiny controlled pilot; 6–12 for stronger evidence if positive. Use at most four configurations per trainable arm and three seeds; admit only a measured matrix within the shared proposed resource ceiling. A later real-LM experiment needs a new allocation.

Future artifacts: full state/cache specification; oracle/task-solvability tests; parameter/config/training ledger; paired per-episode target/collateral outcomes; matched-cost tables; reductions and replay traces; and shared manifest/verifier outputs. Unresolved before dispatch: exact tiny architecture, split generator, operational tolerance, strongest-control selection, precision analysis, and hardware profile.
