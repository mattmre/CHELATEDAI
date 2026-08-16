# RB-15 distribution-by-nonlinearity factorial protocol

**Protocol ID:** `PRW-RCM1-NLN-FACTORIAL-001`

**Frozen:** 2026-08-15, before any output from this protocol was observed

**Status:** `PREREGISTERED_NOT_OFFICIALLY_RUN`
**Claim boundary:** dependency-light synthetic mechanism contrast only;
scientific value, AI/ML transfer, production utility, and novelty remain
`UNCONFIRMED`

## 1. Question and estimand

Stage A observed a descriptive distributed-sidecar difference, but it omitted a
distributed linear control. This protocol asks only:

> On one fixed five-node forced path, is the difference between single and
> distributed placement materially larger with Duffing attachments than with
> linear attachments when total physical coefficients are matched?

For endpoint fundamental transfer gain `G`, lower is better. The frozen
difference-in-differences estimand is

`I = mean[(G_single,Duffing - G_distributed,Duffing)
          - (G_single,linear - G_distributed,linear)]
     / mean[G_single,linear]`.

This is a distribution-by-nonlinearity interaction on a constructed fixture. It
does not identify a general nonlinear-neutraliser advantage.

## 2. Exact 2 x 2 construction

All cells use NumPy float64, fixed-step classical RK4, a five-node path
Laplacian plus `0.20 I`, unit host masses, host damping `0.04 I`, forcing at
zero-based node 0, measurement at node 4, and forcing amplitude `0.20`.

| Distribution | Nonlinearity | Attachment specification |
|---|---|---|
| single | linear | node 2: `(mass=.40, k=.40, c=.064, beta=0)` |
| distributed | linear | nodes 1 and 3, each `(mass=.20, k=.20, c=.032, beta=0)` |
| single | Duffing | node 2: `(mass=.40, k=.40, c=.064, beta=1.00)` |
| distributed | Duffing | nodes 1 and 3, each `(mass=.20, k=.20, c=.032, beta=.50)` |

Thus every factor cell has total auxiliary mass `.40`, total linear stiffness
`.40`, total relative damping `.064`, and, within each nonlinearity level,
total cubic stiffness `0` or `1.00`. Distribution changes sidecar count and
therefore auxiliary state count (two versus four); that structural difference
is disclosed and is not called a free capacity match.

The no-sidecar five-node host is retained as a report-only reference. It is not
part of the factorial estimand or a pass gate.

## 3. Fixed IDs, grids, and initial conditions

- Run IDs are exactly `7` and `11`. No pseudorandom generator is used.
- Run 7 host displacement is `[.010, -.005, .002, 0, 0]`; run 11 is `-.8`
  times that vector. Host velocities are zero.
- Each attachment starts at the displacement of its attachment node and with
  zero velocity, so every initial local mismatch is exactly zero.
- Frequencies are the 17 exact float64 values from `linspace(.70, 1.50, 17)`.
- Each cell runs 60 forcing periods at 200 RK4 steps per period. The final 20
  periods are fitted by float64 least squares to the fundamental and third
  harmonic.
- The official campaign consists of two separate run directories. It must not
  overwrite or modify Stage-A artifacts.

## 4. Phase and continuation policy

Each configuration has two independent branches:

1. `forward`: frequencies in ascending order;
2. `reverse`: frequencies in descending order.

At the first cell of each branch, state is initialized from the run-specific
initial condition. Later cells warm-start from the preceding cell in that same
branch. State is never shared across configurations, directions, or run IDs.
Local integration time resets to zero at every frequency cell; forcing is
exactly `0.20 sin(omega t)`, so forcing phase is zero at every cell. This reset
is intentional and is part of the estimand, not an undisclosed confirmatory
choice.

## 5. Frozen endpoints

All endpoint calculations use all 17 frequencies and both directions.

### Aggregate endpoint

- `aggregate_interaction_ratio`: the estimand `I` above.
- `distributed_duffing_mean_gain_ratio`: mean distributed-Duffing gain divided
  by mean single-Duffing gain.

### Worst-case endpoint

- `worst_case_distributed_duffing_excess_ratio`: the maximum over frequency and
  direction of `(G_distributed,Duffing - G_single,Duffing)`, divided by the
  aggregate mean single-linear gain.

### Hysteresis endpoint

- For each configuration and frequency, compute
  `abs(G_forward-G_reverse) / max((G_forward+G_reverse)/2, 1e-12)`.
- `maximum_relative_hysteresis` is the maximum over the complete factorial.

### Numerical-settling endpoint

- For each cell, divide the absolute difference between final-half and
  preceding-half fundamental amplitudes by
  `max(final-half amplitude, 1e-12)`.
- `maximum_relative_settling_delta` is the maximum over all factorial cells.

Third-harmonic gains and attachment-local mismatch amplitudes/phases/effective
stiffness are retained descriptively. They are not substituted for a frozen
endpoint after execution.

## 6. Design preflight, runtime admission, and enforced guards

The runner is CPU-only and creates no child process, model, corpus, network
request, or GPU context. Design preflight must fail closed unless all of these
hold:

- protocol constants and the four factor configurations match this document;
- each configuration has finite, nonnegative parameters and exactly matched
  totals within `1e-15` absolute tolerance;
- NumPy float64 has eight-byte width;
- the output directory is absent or empty and is not a symlink;
- the frozen campaign size is exactly 136 factorial cells per run
  (`4 configurations x 2 directions x 17 frequencies`) plus 34 report-only
  no-sidecar control cells, for 170 total cells, 2,040,000 RK4 steps, and
  8,160,000 right-hand-side evaluations.

After design preflight, a separately reported runtime-admission check must show
at least one completed resource sample, a nonempty measurement method, and the
explicit initial current and peak self RSS below `512 MiB`; a passed design
preflight does not imply passed runtime admission. Each run has a cooperative hard
computation deadline of 300 seconds and a hard self-process RSS ceiling of
`512 MiB`. Both are checked before every cell and every 256 integration steps.
Breach raises a typed failure, stops further integration, and still permits the
runner to atomically retain a failure artifact. These are self-process/
cooperative guards, not an OS-enforced process-tree sandbox.

## 7. Controls and falsifiers

Controls are the single-linear, distributed-linear, and report-only no-sidecar
branches. The matched distributed-linear cell is load-bearing: without it, no
nonlinearity interaction may be reported. The two run IDs are deterministic
initial-condition sensitivity checks, not statistical replicates.

A run **survives the frozen synthetic gates** only if all are true:

1. every cell is finite and completes its exact step count;
2. `aggregate_interaction_ratio >= .05`;
3. `distributed_duffing_mean_gain_ratio <= .95`;
4. `worst_case_distributed_duffing_excess_ratio <= .10`;
5. `maximum_relative_hysteresis <= .15`;
6. `maximum_relative_settling_delta <= .05`;
7. no deadline or RSS guard is breached.

Otherwise the run is `FACTORIAL_KILLED_ON_FROZEN_SYNTHETIC_GATES`, with every
failed gate retained. Promotion requires both run IDs to survive without any
post-output threshold revision. Even two survivors establish only synthetic
fixture consistency; they do not open Stage B, which remains blocked on the
dependencies named in `docs/next-session.md`.

## 8. Artifact contract

Each run writes only to its dedicated empty directory:

- `factorial.json`: canonical strict JSON, including protocol, design preflight,
  configurations, cells, endpoints, gates, failures, and resource use;
- `manifest.json`: canonical strict JSON duplicating claim/status fields and
  recording SHA-256 plus byte count for `factorial.json`.

The complete two-file result set is written and `fsync`-verified in a sibling
staging directory, then published by one atomic directory rename into an absent
or verified-empty final destination. A failed staging write is removed without
leaving a partial final result or blocking a clean rerun. Verification rejects
noncanonical JSON, unknown or escaping paths, symlinks, digest/size mismatch,
protocol/status mismatch, incomplete schemas, contradictory status/gate/guard/
failure predicates, branch/endpoint disagreement, missing endpoints, and
invalid resource-guard fields. No Stage-A filename or directory is accepted.
The atomic directory rename is the publication commit point. A valid committed
result is idempotently returned on retry without re-executing the factorial. If
parent-directory `fsync` reports an error after that rename but the final set
still passes full verification, publication returns committed success rather
than raising while leaving a valid result that would block recovery.

The verifier reconstructs the resource-guard trace from the frozen branch
order and any terminal checkpoint. A successful cell contributes exactly 49
samples (start, 47 step checkpoints from step 0 through step 11776, and
complete). Thus a survivor has exactly 8,332 samples: one runtime-admission
sample, 170 cells times 49 samples, and one finalization sample. A deadline
breach is tested before RSS sampling, so its failed check is not counted; an
RSS-ceiling or measurement-method breach is counted because that sample was
observed. Only the first terminal resource breach is possible. Finalization is
reachable only after all 170 finite cells and endpoints exist, and reported
wall time cannot precede the last guard timestamp. Method-change evidence must
retain both the previously accepted and newly observed nonempty method names;
such a change is impossible on the first admission sample.

Continuation is direction-local: a nonfinite state makes that cell and every
later frequency in the same configuration/direction sweep nonfinite; recovery
is possible only at the already-frozen reset before the next direction or
configuration. Failure records preserve execution history rather than set
membership. They contain exactly one numerical record per nonfinite cell in
cell order, followed by at most one terminal resource record. If and only if
all cells and the final resource check pass, frozen-gate records follow once
each in the gate order declared above. A resource failure at finalization is
therefore the only failure record even though endpoints and gates have already
been computed.
