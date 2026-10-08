# RHPC Stage A constructed-mechanism protocol

**Protocol ID:** `CHELATEDAI-RHPC-STAGE-A-v1`

**Status:** `FROZEN_NOT_OFFICIALLY_RUN`

**Scientific status:** `UNCONFIRMED`

**Novelty status:** `UNCONFIRMED`

**Promotion eligibility:** `false`

## Claim boundary

This protocol tests a deterministic constructed mechanism only. It asks whether
one implementation can represent expert matrices as shared bases plus
per-slice orthogonal rotations and sparse residuals, bind an ordered four-
station route into a fixed-width Fourier-holographic code, recover a perturbed
route through recurrent factorization, and reject standing cycles or dead
apparent stability.

A passing result does **not** establish learned-model efficacy, compression
utility, general MoE or LoRA transfer, language-model quality, production
readiness, physical optics, novelty, or patentability. Stage B (a learned small
MoE) and Stage C (Qwen micro-LoRA transfer) are separate future protocols and
are not authorized by this document.

## Frozen construction

The fixture has four ordered stations and six expert choices per station,
creating `6^4 = 1,296` possible paths. Every station has a 32 by 32 constructed
expert bank with:

- shared orthonormal left and right bases of rank 16;
- four rank-4 slices;
- deterministic block-orthogonal rotations built from signed/permuted 4 by 4
  Walsh-Hadamard matrices;
- distinct diagonal spectra; and
- six deterministic sparse residual entries per expert.

The complete expert representation is

```text
W[l,e] = U[l] Q[l,e] D[l,e] Q[l,e]^T V[l]^T + residual[l,e]
```

The no-rotation control retains the exact shared bases and residual budget but
projects the rotated core onto a diagonal shared-basis spectrum. The
no-residual control retains the exact shared bases, rotations, and spectra but
removes the sparse residual. These controls test whether both additions are
mechanically load-bearing on the constructed matrices.

## Ordered holographic path representation

Six deterministic 1,000-dimensional complex root-of-unity code vectors are
generated from SHA-256. Station position is represented by a distinct
invertible affine coordinate permutation. A route `(e0,e1,e2,e3)` is encoded
as the elementwise product of the four position-permuted expert codes.

The position permutations are load-bearing: multiplying unpermuted codes would
be commutative and would lose the assignment between an expert symbol and its
station. The orderless control uses the same code dimension, codebook, number
of factors, and arithmetic budget but removes the station permutations.

## Recurrent resolver and lock

Each trial begins with router hints. One quarter of trials start on the true
path; the other three quarters contain exactly one low-margin wrong station.
The resolver updates stations from lowest to highest hint margin. At each
station it unbinds the current estimates for the other three factors, applies
the inverse position permutation, and performs deterministic nearest-codebook
cleanup. Ties select the lower expert ID.

A route may lock only when all of the following hold:

1. the same candidate path is observed for two successive completed updates;
2. real phase alignment with the observed composite is at least `0.85`;
3. circular coherence is at least `0.85`; and
4. no non-unit cycle has been observed.

Each trace also reports `phase_residual = 1 - circular_coherence`. Resonant
attenuation is the initial phase residual minus the final phase residual; it is
positive only when recurrent updates reduce the composite phase error.

Revisiting a prior non-adjacent candidate tuple is a standing cycle. It returns
`CYCLE` and can never be reclassified as `LOCKED`. A constant dead trace is
also evaluated; repeated state alone cannot lock unless reconstruction
alignment and coherence pass.

## Frozen run matrix

Official run IDs are exactly `7` and `11`. Each run contains 256 deterministic
trials and at most eight recurrent updates per trial. The composite code is
perturbed by deterministic sign flips on exactly one twentieth of its
coordinates.

Each run reports:

- exact/full, no-rotation, and no-residual matrix reconstruction error;
- recurrent, one-shot, smoothing, orderless, and dead-control path behavior;
- path accuracy, lock coverage, lock precision, false-lock rate, mean lock
  iterations, cycle rate, and mean phase-residual attenuation;
- an explicitly injected two-cycle classification check; and
- every trial's operational specifier chain and lock trace.

The smoothing control repeats the initial one-shot route for the same number of
observations. It is not allowed to inspect the holographic composite.

## Frozen constructed-mechanism gates

A run receives `SURVIVES_CONSTRUCTED_MECHANISM_GATES` only if every gate passes:

1. maximum exact matrix reconstruction relative error `<= 1e-12`;
2. mean no-rotation relative error `>= 0.01`;
3. mean no-residual relative error `>= 0.001`;
4. recurrent full-path accuracy `>= 0.95`;
5. recurrent accuracy exceeds one-shot accuracy by at least `0.25`;
6. recurrent accuracy exceeds orderless-control accuracy by at least `0.25`;
7. recurrent lock coverage `>= 0.90`;
8. recurrent lock precision `>= 0.99`;
9. recurrent false-lock rate `<= 0.01`;
10. dead-control lock coverage `<= 0.01`; and
11. recurrent mean phase-residual attenuation `>= 0.50`; and
12. the injected two-cycle is classified `CYCLE`, never `LOCKED`.

Failure of any gate produces `DOES_NOT_SURVIVE_CONSTRUCTED_MECHANISM_GATES`.
Thresholds may not be changed after either official run is opened. Both run IDs
must survive before Stage A can be described as replicated on this fixture.
The dual-Spark acceptance matrix runs both IDs on both machines; matching run
IDs must produce byte-identical canonical result files across hosts.

## Phase and execution boundary

Local implementation work may run `DEVELOPMENT` previews. A preview is not an
official observation and cannot publish an official result directory.

`SPARK_OFFICIAL` mode requires all of the following:

- explicit CLI and environment opt-in;
- a clean exact-source checkout;
- the prior EGV campaign's exported public projection;
- successful closed public cryptographic replay with a terminal public-chain
  seal;
- the terminal seal's bound signed restore receipt; and
- an absent output directory.

The public projection verification is the mechanical handoff condition. A
chat message, branch name, process absence, or unsigned completion file is not
sufficient.

Stage A is CPU-only and must not stop, restart, reconfigure, or otherwise
mutate the restored DeepSeek service. It performs no network access and creates
no GPU context.

## Artifact contract

Each official result directory contains exactly:

- `rhpc-stage-a.json`: the canonical closed result; and
- `manifest.json`: protocol, source, environment, predecessor-completion,
  result size, and SHA-256 bindings.

The runner writes and fsyncs both files in a hidden sibling staging directory,
verifies the staged set, and atomically renames it into an absent final target.
The verifier rejects extra files, unsafe members, links, noncanonical JSON,
unknown fields, digest or size changes, contradictory gates/status, and a
result that does not exactly regenerate from its frozen configuration.

## Interpretation

- Two surviving official runs establish only deterministic constructed-
  mechanism replication.
- A failure is retained as a valid negative and narrows the design.
- Stage B requires its own learned-model protocol, route-disjoint
  MODEL-TRAIN/DEVELOPMENT/SELECT/REPORT data, matched compression budgets, and
  frozen efficacy endpoints.
- Stage C requires a separate Qwen micro-LoRA protocol and cannot reuse an
  exhausted REPORT fixture.
