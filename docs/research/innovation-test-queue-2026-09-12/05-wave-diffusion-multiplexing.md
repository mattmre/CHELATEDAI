# IQ-05 — Wave/diffusion control as a conditional transformer ablation

**Status:** `QUEUED_REDUCTION_REVIEW / COMPUTE_HELD / NOT_FROZEN / NOT_RUN`.

**Parent:** design lane; possible later [IQ-02](02-scoped-recurrent-correction.md) ablation, not a separate new-model campaign.

**Owner/reviewer:** unassigned dynamics researcher / independent numerical-and-ML reviewer. Follow [G0](shared-test-contract.md).

## Hypothesis and scope choice

The candidate question is whether bounded, input-conditioned communication among parallel representations improves task quality or reduces interference beyond ordinary learned mixing at matched total cost. It is not whether a differential equation can describe a network.

Distinguish two experiments before proceeding:

- **Branch communication:** several branches cooperate on one task; diffusion/wave dynamics mix their states.
- **Data multiplexing:** several independent inputs share an encoded forward pass and are demultiplexed; throughput and cross-input leakage are the endpoints.

Default is branch communication because it matches the proposed inference-control idea. Data multiplexing remains a separately labeled optional branch; no result transfers automatically between them. Read [GraphCON, GRAND, Cross-stitch, and DataMUX](sources.md). Each supplies established ingredients/controls, not proof that the exact proposed combination exists.

## Checkable candidate and reductions

For aligned branch states Z and symmetric nonnegative edge weights A, let L=D-A:

```text
Z_next = (I - eta L) Z
0 <= eta <= 1 / max_i D_ii     (identity if the maximum is zero)
```

This gives convex row mixing for that step and preserves a constant branch signal. Input-dependent weights do not automatically make the complete nonlinear system contractive. Repeated mixing may erase useful differences; branch consensus is not correctness or independent corroboration.

A separate wave-controller candidate has state (a,v):

```text
a_dot = v
v_dot = b(h) - gamma v - K a
K = omega0^2 I + c^2 L
delta_h = U diag(epsilon tanh(a)) U^T h
```

For fixed symmetric positive-definite K and gamma>0, the unforced continuous energy `E=(v^T v+a^T K a)/2` obeys `E_dot=-gamma v^T v`; input adds `v^T b`. Adaptive K adds another derivative term. None of this proves discrete integrator stability, useful task behavior, or safety of the surrounding transformer. The second-order model is a first-order recurrence on (a,v); rewriting it is not escape from matrix algebra.

Common parent interfaces require an explicit learned/fixed coordinate alignment, fit without REPORT. Equal shape is insufficient. Count alignment work and information loss. A fully conjugated basis rotation is only reparameterization.

## Work items and entry gates

- [ ] **W0 — reduction and novelty chart (1–3 days):** specify whether the proposal is diffusion, an oscillator controller, or packed-input multiplexing; derive its exact discrete update and match it to ordinary recurrence/mixing. Read the closest full methods/code. State one distinction expected to produce measurable benefit. If no distinction survives, return `PARKED_AS_KNOWN_MECHANISM` without numerical experimentation.
- [ ] **W1 — bounded numerical checks, only after G0/W0 survival:** nonnegative/symmetric weight constraints, zero-edge/zero-gate reductions, constant-signal preservation, causal feature access, bounded update, step-size failure tests, disconnected graphs, dead states, forced oscillation, non-normal transient growth, and missing-coordinate-contract rejection. Proposed CPU cap: four branches, dimension at most 64, horizon at most 32, one process, 512 MiB, five minutes per configuration.
- [ ] **W2 — matched branch pilot:** reuse an admitted IQ-02-compatible harness with two/four branches on partial-information compositional tasks, distractor/conflict conditions, and private branch facts. The harness must be correct; a positive IQ-02 scientific result is not a logical prerequisite. Training/execution needs a separate allocation.
- [ ] **W3 — optional multiplexing study:** only if specifically selected at W0 under a separate protocol, compare packed inputs with ordinary batching and DataMUX-style packing. Vary two/four inputs per pack; keep per-input task quality, total work, hardware, and batch scheduling comparable. Test cross-input contamination; do not mix independent confidential user workloads in this research fixture.

## Controls, data, and decisions

For W2: no communication; arithmetic averaging; learned cross-stitch-style mixing; ordinary cross-attention; first-order diffusion; augmented-state gated recurrence with the **same (a,v) state size**; and the candidate wave controller. Match branch count, trainable parameters, state bytes, available observations, total training effort, and total inference work. Discrete numerical checks alone do not establish neural usefulness.

Use fresh group-disjoint MODEL-TRAIN/DEVELOPMENT/SELECT/REPORT instances with the IQ-02 size/budget proposal as an upper planning envelope, not a reused exposed REPORT. Train all branches on the same declared task opportunity; fit alignment on MODEL-TRAIN. Include delayed constraints and conflicting but plausible signals; extra branches must not receive extra correct labels unavailable to controls.

Proposed W2 primary gate: at least two percentage points of task-accuracy improvement over the strongest SELECT-chosen ordinary communication control, paired 95% improvement interval above zero, unaffected-query accuracy noninferior within one percentage point, total p95 latency at most 1.1x, and memory at most 1.2x. Freeze the finite candidate/configuration set before confirmation; multiple candidate families need multiplicity control or separate fresh confirmation.

For W3, proposed gate is at least 20% higher completed-input throughput at accuracy noninferior within one percentage point versus ordinary batching and the selected packing control, without worse p95 per-input latency or contamination. This is a separate throughput claim, not a W2 success criterion.

Stop if simple mixing or a same-size first-order recurrent controller matches the effect, the integrator alone accounts for the difference, branches collapse to one wrong answer, extra compute explains the improvement, or observation/alignment advantages are unmatched. Preserve RB-15's prior nonlinear-factorial negative; this packet does not relabel or rerun it.

## Budget and handoff

### Integrated mathematical tests — 2026-10-08

[MA0/MA6 and cases MA6-T1–T3](math-findings-integration-2026-10-07.md) extend W0/W1 only after W0 identifies a surviving distinction. Compare fixed symmetric ordinary mixing, fixed directed nonnormal amplification, switching products, and discrete/adaptive scope refusals. Numerical-range bounds require a certified outer enclosure; fixed-point existence or a continuous energy identity cannot certify a nonlinear discrete controller. The symmetric candidate keeps its simpler ordinary stability controls. Handoff adds fixed-versus-adaptive operator identity, enclosure/bound slack, discretization assumptions, refusal traces, and total numerical costs; task utility still requires W2's existing endpoint.

Only W0 has a present planning estimate. A later W2/W3 pilot may take 1–2 workweeks, but is not allocated and cannot be estimated credibly until the mechanism/harness is fixed. Full architecture proof time is unknown. Use the shared finite compute proposal only after profiling; no scaling to rescue a failed ablation.

Future artifacts: exact discrete equations; state/alignment and information contracts; prior-art/equivalence chart; stability/causality checks; matched branch or pack matrix; all task/collateral/leakage/cost records; and shared manifest. Unresolved before dispatch: selected branch, distinct mechanism, integrator, task, alignment, and measured limits.
