# Recurrent scoped-state transformer: design review

<!-- innovation-test-queue-2026-09-12:start -->
## Test-packet follow-through — 2026-09-12

The proposed fixed-depth scoped update now has a [prepared IQ-02 test packet](innovation-test-queue-2026-09-12/02-scoped-recurrent-correction.md); the wave/diffusion suggestion has a separate [IQ-05 reduction-first packet](innovation-test-queue-2026-09-12/05-wave-diffusion-multiplexing.md). Both are queued for review, not implemented, frozen, or authorized to run.

The [expanded source register](innovation-test-queue-2026-09-12/sources.md) adds AlphaEdit/ActAdd/KnOTS to the editing/steering/alignment boundary and GraphCON/GRAND/Cross-stitch/DataMUX to the communication boundary. Weight editing, activation control, cooperating branches, and packed independent inputs must remain separate experiment types. The main first test still changes one mechanism at fixed depth; the proposed numeric gates are reviewable planning choices, not measured effects.
<!-- innovation-test-queue-2026-09-12:end -->

**Date:** 2026-09-12. **Working study ID:** `ARCH-RLT-01`.
**Status:** `DESIGN HYPOTHESIS / NOT IMPLEMENTED / NOT FROZEN / NO RUN AUTHORIZATION`.

## Recommendation

Worth a small, falsifiable architecture study; not yet worth a new foundation-model training campaign. Our strongest candidate contribution is **selective state revision with measurable collateral effects and retractable evidence**, not recurrence, rotation, or memory lookup by themselves.

The question is: at a matched compute budget, can scoped recurrent updates improve held-out compositional task accuracy or correction quality over ordinary gated recurrence plus ordinary versioned memory? A successful narrow result would be useful even if it becomes a module for an existing model rather than a new transformer family.

## What the linked design actually supplies

Reviewed [upstream commit `1bee93a`](https://github.com/yifanzhang-pro/recurrent-looped-tranformer/tree/1bee93a9b01c21bea0c7a50ce3f6619f24731e19), including all 16 pages of its [technical report](https://github.com/yifanzhang-pro/recurrent-looped-tranformer/blob/1bee93a9b01c21bea0c7a50ce3f6619f24731e19/Recurrent_Looped_Transformer.pdf). The tree contains a README, two PDFs, a figure, and a website; no runnable model, tests, checkpoints, or benchmark package. No license file was present in this snapshot; code reuse/licensing must be checked if an implementation appears. No upstream code was run.

RLT combines a causal encoder, global encoder-derived KV, and a recurrent decoder carrying its output and layerwise sliding-window KV across tokens. Its tied configuration reuses compatible encoder/decoder weights. Temporal depth grows with sequence length; this is not extra thinking before one output token. The report explicitly leaves quality and speed improvements unmeasured. Preserve its complete-state and current-parameter replay contract, rather than treating the final hidden vector as the whole model state. These are design specifications, not independently replicated results.

RLT already specifies gated feedback and mentions low-rank feedback projection. Adding a gate or low-rank matrix alone would not distinguish our design; those belong in its ordinary baseline.

## Closest prior work and what it rules out claiming

| Primary source | Relevant overlap | Consequence for our study |
| --- | --- | --- |
| [Feedback Transformer, 2020](https://arxiv.org/abs/2002.09402) | Higher-level past representations feed future computation. | Temporal feedback is established. |
| [Universal Transformers, 2018](https://arxiv.org/abs/1807.03819) | Shared recurrent computation and adaptive per-position halting. | Weight sharing and adaptive depth alone are not our novelty. |
| [Recurrent-depth latent reasoning / Huginn, 2025](https://arxiv.org/abs/2502.05171) | Additional latent iterations without requiring additional text tokens. | Compare against an existing recurrent-depth approach, not just a shallow transformer. |
| [The Recurrent Transformer, 2026](https://arxiv.org/abs/2604.21215) | Layerwise recurrent KV and an exact, memory-traffic-aware execution algorithm. | Its tiling result does not automatically apply to RLT's different full-decoder dependency. |
| [Fixed-Point Reasoners, 2026](https://arxiv.org/abs/2606.18206) | Residual scaling and convergence-based adaptive halting. | Stability-controlled adaptive looping already has direct prior work. |
| [Think Shallow, Solve Deep, 2026](https://arxiv.org/html/2608.18222v1) | Finite-time dynamics, margin-based answer stability, fixed-point objectives, and limits of deeper inference. | Include this recent preprint in the stability baseline review; do not relabel its mechanism as ours. Its reported results have not been replicated here. |
| [Titans, 2025](https://arxiv.org/abs/2501.00663) | Learned test-time memory alongside attention. | A memory update during inference is not a new category. |
| [Engram, 2026](https://arxiv.org/abs/2601.07372) | Conditional lookup complements neural computation. | Static lookup can be a control or a complementary component, not a straw-man opponent. |

This is a targeted prior-art scan, not an exhaustive novelty or patent assessment. Do not assign novelty percentages. These comparisons also do not establish that every related paper solves our specific collateral-damage or evidence-retraction task.

## Minimal mathematical candidate

Separate token position \(t\) from optional latent-refinement step \(k\). Carry token-to-token state, but initially test only **one** new mechanism: a scoped residual update. Additional inner iterations are a separate extension to RLT, with separate cache semantics and cost.

For hidden state \(h_{t,k}\in\mathbb R^d\), a shared basis \(U\in\mathbb R^{d\times r}\), and a gate computed only from presently available information:

\[
U^\top U=I_r,\qquad
P_{t,k}=U\,\operatorname{diag}(g_{t,k})\,U^\top,
\quad g_{t,k}\in[0,1]^r,
\]

\[
h_{t,k+1}=h_{t,k}+
\eta_{t,k}P_{t,k}
\left[G_\theta(h_{t,k},e_t,\mathcal M_t,C_{t-1})-h_{t,k}\right],
\qquad 0\le\eta_{t,k}\le1.
\]

Here \(G_\theta\) proposes a new state, \(e_t\) represents the observed token prefix, \(\mathcal M_t\) is visible evidence memory, and \(C_{t-1}\) denotes the required prior caches. This is a proposed hidden-state update, **not yet a complete transformer specification**. The implementation contract must define where this update sits relative to attention, normalization, FFNs, and cache construction.

What is mathematically available:

\[
(I-UU^\top)(h_{t,k+1}-h_{t,k})=0,
\qquad \|P_{t,k}\|_2\le1.
\]

The single update does not alter the complement of the selected parent subspace. A soft gate makes \(P\) a positive-semidefinite contraction, not generally an idempotent projector; binary gates give a projector. This does **not** prove semantic isolation, safe outputs, stable attention caches, convergence of the residual map, or preservation across later layers. Those are separate tests.

Interpretation of "congruent parent interfaces": shared coordinates could let modules describe and restrict their updates in a common space. But an invertible change of coordinates with the entire transition and readout conjugated consistently is only a reparameterization. Benefit must come from a useful restriction, learned routing, communication cost, or inductive bias that survives ordinary low-rank/gated controls. A phase or prime-ring label does not supply that evidence.

Two mandatory reductions are \(r=d, U=I, g=\mathbf1\), which yields ordinary relaxed recurrence, and zero gate, which makes this update the identity. Computing a dense \(G_\theta\) and projecting afterward does not save that dense computation; a compute-saving claim needs actual skipped work and measured total cost.

## How our research can contribute, conditionally

| Existing line | Candidate contribution | Boundary |
| --- | --- | --- |
| CRSV / LIR | Predict which scoped module updates and ordered combinations are compatible. | Adapter/retrieval results do not automatically transfer to hidden-state recurrence; validate the transfer. |
| SRS | Use trajectory diagnostics to decide whether another refinement is worth its cost. | Compare to simple entropy/margin, residual-change, and established halting controls. Stable-but-wrong is a failure. |
| RB-13 / RB-14 temporal evidence | Track provenance, versions, corrections, and retractions separately from transient state. | Versioned storage is an ordinary baseline; a retraction does not undo hidden-state contamination automatically. |
| Localized assimilation / routed adapters | Train or activate a bounded scope with collateral-effect measurements. | Separate activation routing from weight adaptation; first prototype must not change both at once. |
| RHPC / spinning-hologram representation | Later test compact composition/addressing at shared interfaces. | Defer until it beats a conventional same-information decoder on learned, not constructed, representations. |

These are sources of hypotheses, not proven building blocks. No positive QSCCI oracle-controllability result can stand in for a label-blind routing or halting policy.

## Missing contracts that matter before implementation

1. **What computation is being repeated?** Start with token recurrence and a fixed refinement count. If adding \(K_t\le K_{\max}\) inner steps, distinguish temporary per-step activations from committed token KV. One token must not accidentally become several historical positions. Freeze exactly which state/caches each refinement reads and what is committed after it.
2. **What can each decision observe?** A gate may use the consumed prefix, available memory metadata, and diagnostics from completed steps to choose the next step. It may not use a test answer, future tokens, or the eventual outcome of a held-out trajectory. Record availability as `PRE_TOKEN`, `PRE_NEXT_REFINEMENT`, or `POST_OUTCOME`; only the first two are deployment inputs.
3. **What does reversible mean?** Deleting a memory record reverses a store edit, not every downstream computation. Exact removal of its influence requires a valid earlier full-state checkpoint and replay of the declared corrected history, including KV and metadata; already emitted outputs remain emitted. Weight updates require version-compatible reconstruction. Begin with explicit replay, not an unproved inverse.
4. **How is training faithful?** Specify sequence resets, causal masking, loss masking, full-state gradients, checkpointing, and any truncation. Long nonlinear recurrence serializes decoder work and lengthens credit assignment. Do not quietly use detached/stale caches while reporting full-history training. RL is a later study, not the first prototype.
5. **What is the memory budget?** Bounded local decoder KV does not bound global encoder KV or training activations. Measure time-to-first-token, total task latency, peak memory, bytes moved, and controller/replay cost. A constant block count is not constant attention cost.
6. **What distinguishes an algorithmic benefit?** Test held-out compositions and lengths, changing evidence, irrelevant distractors, and wrong-but-confident states. Include independent tasks with different semantics; repeated seeds do not create new task families. Next-token training and latent state alone do not establish systematic reasoning.

Also vary required computation while keeping input length fixed, and include late-arriving questions or constraints. Work done before a question arrives is not automatically useful reasoning about that question; a longer token history must not be mistaken for extra task-conditioned refinement.

For the first study, learn the shared basis and gates on training data, choose configurations only on the declared selection split, and freeze model weights for REPORT. Specify and cost the orthonormal-basis parameterization. Do not fit the basis to REPORT activations or introduce test-time weight adaptation under an unchanged protocol.

## Smallest useful experimental path — not a frozen protocol

| Stage | Question and controls | Advance / stop |
| --- | --- | --- |
| A: specification and correctness | Define full state; test causality, prompt-split equivalence, reset isolation, pending-token handling, gradient paths, and replay. Demonstrate the identity/ordinary-recurrence reductions. | Stop before training on any unresolved state or equivalence defect. |
| B: small learned scope test | Learn a small ordinary recurrent baseline on a solvable compositional/state-update task. Compare it with the scoped update at fixed depth; include ordinary gated low-rank updates and fixed/random basis controls. Use the same visible information. | Advance only on held-out task benefit or reduced collateral error at a declared cost/quality constraint; not merely smaller state changes. |
| C: independent additions | Add adaptive halting or versioned evidence memory **one at a time**. Compare to established halting and ordinary provenance/versioned-memory controls. | Stop each component independently if its gain disappears after accounting for its information, parameters, or cost. |
| D: interaction and transfer | Factorial ablation of surviving components on new task families, longer compositions, and a modest real-language task. | Only reproducible incremental benefit justifies a combined architecture and a larger training proposal. |

Report both parameter-matched and compute-matched comparisons when they cannot be satisfied simultaneously. Equal loop counts are not equal FLOPs. Account for training tokens, tuning trials, failed attempts, full inference/replay work, wall time, and memory. Choose one primary outcome, a minimum useful effect, uncertainty method, independent sampling unit, seed plan, and collateral/cost limits **before** a confirmatory run. Those choices belong in the next protocol/task-planning stage; no threshold is frozen here.

Do not append unlimited loops until a test example becomes correct. A halting policy must be selected without REPORT outcomes, have a hard cap and fallback, and be evaluated on tasks not used to select it. SRS diagnostics are candidate predictors, not a correctness oracle.

## Decision and schedule placement

`ARCH-RLT-01` is high priority for design and a bounded discriminator; full-model training is on hold. Use the [portfolio schedule](research-priorities-and-testing-schedule-2026-09-12.md). It is not a replacement for metric repair, existing protocol gates, or the independent CRSV/LIR/SRS questions. It can be designed without waiting for those hypotheses to pass, but must independently validate any mechanisms imported from them.

Possible outcomes: an ordinary gated-recurrence result; a useful scoped-update or correction module; or, only after broader replication, a defensible new combination. There is presently no evidence for "much better," a general reasoning breakthrough, or revolutionary hardware efficiency.
