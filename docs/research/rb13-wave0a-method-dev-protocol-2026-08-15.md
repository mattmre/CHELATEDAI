# RB-13 Wave 0A frozen method-development protocol

**Protocol ID:** `CHELATEDAI-RB13-WAVE0A-v8`  
**Frozen:** 2026-08-16, before any successful official scientific output  
**Scope:** `PRW-EK0`, `PRW-BIL1`, `PRW-SPU0`, and exact-small `PRW-VAR1` only  
**Claim boundary:** deterministic synthetic/current-contract method development;
not production, utility, scientific-confirmation, or novelty evidence.

V1 through v6 were never officially executed. V7 was officially attempted,
and failed closed during the first EK0 infrastructure stage: sampled
process-tree RSS was 590,852,096 bytes against a 570,822,656-byte sampled
backstop. The quarantine contains a `COMPLETE` EK0 stage artifact for seed
1301, but there is no valid or published campaign result; campaign status is
not defined because no campaign manifest was published. The retained v7
failure evidence records `status=INVALID_RUN` and
`failure_category=RESOURCE_OR_DEADLINE`; it is not a scientific result. V8 resource calibration uses
only the retained resource-failure measurements above. The quarantined EK0
stage metrics were not inspected or used. V8 changes only EK0 resource
calibration: its aggregate modeled reservation and sampled backstop become
768 MiB. Every fixture, seed, endpoint, null, and scientific gate is unchanged.

V7
sizes invalid-run detail against the final canonical UTF-8 JSON byte budget,
records whether it was truncated, and retains a SHA-256 identity of the full
detail without retaining its full content. Evidence serialization or
quarantine errors never replace the original triggering exception. V6 put
every operation after partial-directory creation inside the quarantine
transaction, reserves a separate bounded invalid-run evidence allowance,
enforces evidence-capable budget minima, and records the execution platform.
V5 removed
the false implication that EK0 can perform import-inclusive admission before
Torch is imported. The parent instead reserves an EK0 aggregate ceiling
before launch; the worker performs its measured admission only after
the import. V4 described
RSS observation as 50 ms sampled, best-effort telemetry and distinguishes
fixed expected-witness verification from artifact-independent regeneration
that shares implementation code. V3 included EK0's
actual transitive Torch import in its preflight and ceiling, refreshed parent
RSS before every child, and added frozen-witness verification.
V2 had superseded v1 by pinning the canonical
preregistration digest in code, verifying child artifacts before publication,
retaining structured invalid-run evidence, and correcting SPU0/VAR1 reporting.
The official runner refuses protocol bytes
that do not match its embedded SHA-256, regardless of caller input. Each card receives its own artifact and
disposition. A result from one card cannot promote another card.

## 1. Immutable execution contract

- CPU-only execution intent; BIL1, SPU0, and VAR1 use the Python standard
  library plus NumPy. EK0 exercises the actual current memory-store import
  path, whose transitive dependencies include Torch. One stage
  child at a time. The runner hides CUDA devices and caps numeric-library
  threads, but those environment controls do not prove absence of network,
  model, or GPU access. The implemented cards contain no such calls.
- EK0 receives a 768 MiB aggregate modeled stage reservation and sampled RSS
  backstop. Its worker budget is the remainder after subtracting freshly
  measured parent RSS, preserving the unchanged 768 MiB campaign total and
  default. BIL1 and SPU0 have 256 MiB stage ceilings and VAR1 has a
  768 MiB stage ceiling. The common runner refuses a configured budget above
  2 GiB and defaults to a 768 MiB campaign ceiling.
- Official IDs are `SELECT` and `REPORT`. `SELECT` uses seeds
  `[1301, 1303, 1307]`; `REPORT` uses disjoint seeds `[2309, 2311, 2333]`.
  Tests use only fixture IDs prefixed `UNIT-` and seed `17`.
- Serialized budgets use plain integers for byte ceilings and a plain float
  for the stage deadline. Artifact verification reconstructs `Wave0ABudget`
  and rejects values outside its frozen bounds before result verification.
  The configured modeled-memory minimum is 16 MiB and per-artifact output
  minimum is 16 KiB. Invalid-run evidence has its own fixed 16 KiB per-file
  allowance rather than consuming a stage artifact's output allowance. Detail
  truncation is computed from canonical JSON encoded bytes, including escaping
  of control and non-BMP characters, so the final newline-terminated evidence
  file is at most 16 KiB.
- Default official sizes: BIL1 has six events and one four-node cyclic graph; SPU0 uses
  `d=32`, `K=5`, and 48 vectors; VAR1 uses `n=12` and four frozen cells. EK0
  uses two-entry bounded stores and temporary directories only.
- For EK0, the parent-before-launch 768 MiB aggregate modeled reservation is
  the primary preventive gate. The allocation-free dependency precheck confirms only that
  Python can locate the Torch and memory-store module specifications; it does
  not import them and does not predict import RSS. The worker must import Torch
  before it can measure its post-import baseline or perform its own modeled
  admission. That import allocation therefore occurs before the worker can
  check it, and the parent sampler cannot prevent an OOM before its first
  observation. Official DGX execution still requires a separate host-capacity
  preflight. Other card allocations use conservative modeled preflight.
  The parent also samples RSS approximately every 50 ms
  as a best-effort backstop. On Linux, each sample covers the parent and the
  worker process group/descendants; on Windows it covers the parent and direct
  worker only, not an arbitrary descendant tree. A sampled overage terminates
  the worker, but short-lived touched-memory overages between samples may evade
  observation. Sampled peak RSS is telemetry, not proof of a hard memory cap.
  Before allocation, each card records dimensions, operations, and a
  conservative byte estimate. A refused preflight produces no result artifact.
- Each stage is written to a temporary sibling, flushed, fsynced, and atomically
  replaced. The manifest is written last and contains protocol, config, stage,
  and file SHA-256 digests. An incomplete directory is not a campaign result.
- Once a partial campaign directory exists, parent RSS acquisition, platform
  detection, configuration, worker execution, and publication are all inside
  the failure transaction. Any caught post-creation failure attempts a bounded
  `INVALID_RUN` write and then atomically renames the partial directory to a
  quarantine sibling. If either evidence serialization or quarantine itself
  unexpectedly fails, that secondary failure is suppressed so the original
  campaign exception is re-raised.
- Artifact status is `COMPLETE`; scientific and novelty
  statuses are always `UNCONFIRMED` in this wave.
- Protocol, destination, or argument rejection before a partial directory is
  created is an invocation rejection, not a started run, so it does not create
  `INVALID_RUN` evidence. After a partial directory exists, worker, resource,
  deadline, and publication failures retain structured `INVALID_RUN` evidence.
- The frozen work deadline excludes a separate five-second cleanup grace used
  only to kill/reap processes and flush quarantine evidence.
- EK0 and BIL1 verification compares artifacts with fixed expected contract
  witnesses defined separately from their result generation. SPU0 and VAR1 use
  artifact-independent regeneration, but generation and verification share
  numerical implementation helpers; a systematic bug in shared code can affect
  both and escape this check. These checks detect artifact tampering, not an
  independently implemented scientific replication.
- Successful stage artifacts, manifest entries, and campaign configuration
  record `platform.system()` as `Linux` or `Windows`. Monitor booleans must
  match that recorded allowlisted value, so verification is portable across
  operating systems. The value is unkeyed and self-attested runtime metadata,
  not cryptographic host attestation; a fully resealed artifact set can lie
  about both the platform and its flags.

## 2. `PRW-EK0` current memory-contract characterization

### Fixture and observations

Use only fresh `ModelScopeMemoryStore` instances with every segment capped at
two entries and fresh temporary `PersistentMemory` directories. Feed minimal
synthetic artifacts with fixed prompt hashes. Observe, without changing the
classes: FIFO overflow, in-place annotation, `record_observation(promote=True)`,
`promote_episode` with missing status, repeated file-backed saves, deletion,
restart, and the fixed interleaving `save(a), save(b), save(a), delete(b)`.

### Null and disposition

Null: the stores already provide append-only bitemporal,
provenance-preserving, gate-enforced semantics. Classify each observation as
`CURRENT_CONTRACT`, `BUG`, or `MIGRATION_GAP`. The null survives only if there
is no destructive overflow/overwrite/delete, annotation appends a version,
every promotion path requires an affirmative gate, and restart/as-of history
is available. Otherwise route to an explicit compatibility/migration design;
do not mutate production behavior in this card.

## 3. `PRW-BIL1` four-state merge audit

### Fixture and controls

Events contain fixed IDs, lineage IDs, polarity (`support` or `refute`), valid
interval, transaction time, and active/retracted status. Fixtures include
duplicates sharing a lineage, independent opposing lineages, permutations,
corrections, valid-time changes, and cycles. Candidate state is the pair
`(has_support, has_refutation)` derived from unique active lineages, yielding
`UNKNOWN`, `SUPPORTED`, `REFUTED`, or `CONFLICTED`.

Controls are scalar `support-refutation`, last-write-wins category,
order-dependent paired booleans, and paired-boolean set union. Monotone graph
propagation unions lineage facts over successors until no state changes.

### Frozen endpoints

Primary endpoints are unknown/conflict confusion, permutation invariance,
duplicate-lineage idempotence, fixed-point convergence and equality across
node/edge orders, current/as-of oracle agreement, false promotion, and counted
operations. Pass requires all candidate invariants. Any failure kills the
candidate semantics. If paired-boolean set union matches every semantic result
with no higher information requirement, disposition is
`REDUCE_TO_PAIRED_BOOLEAN_UNION`, not a separate mechanism claim.

## 4. `PRW-SPU0` fixed-stack flattening guard

Generate seeded Gaussian square factors and vectors in float64. Compare lazy
ordered application, the same stored factorized-flat control, and the explicit
dense product. Compare outputs at absolute/relative tolerance `1e-10`, pairwise
Euclidean distances at absolute and relative tolerance `1e-10`, and stable
nearest-neighbor ranks exactly. Relative error is absolute error divided by the
maximum absolute reference value; a zero reference denominator yields zero
only for exact equality and infinity otherwise. Report matrix rank, separate
multiplication/addition operation estimates, and stored
bytes for dense and factorized forms.
Accounting separates dense-map construction/amortization, per-vector lazy,
factorized and dense application, three distance/square-root passes, a ranking
comparison upper bound, rank-list storage, and an SVD/matrix-rank workspace and
work upper bound. These are conservative accounting units, not an exact count
of vendor BLAS/LAPACK instructions. Preflight covers the declared arrays and
upper-bound work scopes.

Any mismatch is classified as arithmetic, state dependence, nonlinearity,
gating, or implementation error. Equivalence passes the guard and routes any
resource difference to ordinary factorization. It is not evidence for
dimensional or subplane novelty.

## 5. `PRW-VAR1` exact-small variational audit

Each frozen cell is an attractive binary Potts objective

`sum_i unary_i*x_i + lambda*sum_(i,j) |x_i-x_j|`.

For `n<=20`, enumerate every binary mask as the oracle. Compare: projected
smooth quadratic relaxation plus threshold rounding; projected proximal-TV
subgradient plus rounding; an exact s-t min-cut implementation where the
attractive-Potts assumptions hold; and deterministic marginal greedy. Ties use
lexicographically smallest mask. Feasibility means a binary mask of length
`n`. Downstream recall is selected-positive recall against the frozen relevant
set, with empty relevant sets defined as 1.0.

Report objective and integrality gaps, Jaccard with the oracle, feasibility,
recall, and runtime. Seed variance is computed only across corresponding cell
IDs after all seeds in a campaign are present; within-seed variance across
different cells is forbidden. The cross-phase summary keeps SELECT and REPORT
labels separate and explicitly marks both as synthetic, non-independent-label
method-development fixtures. The null survives only when smooth rounding
matches the exact objective and downstream recall in every SELECT and REPORT
cell. A positive gap routes Euler-Lagrange-style relaxation away from the hard
mask optimizer; tuning after REPORT is forbidden. Graph cut is an established
control and must reproduce the exact optimum in every applicable cell.

## 6. Reporting and kill rules

All cells and failures are retained. SELECT may choose no hyperparameters:
the step counts, learning rates, thresholds, and tie rules above are fixed.
REPORT is a disjoint reproducibility check, not a search. A crash, digest
mismatch, deadline breach, nonfinite value, preflight refusal, graph-cut/oracle
mismatch, or missing artifact makes the card `INVALID_RUN`; it cannot be
silently dropped. Negative/null/category-error outcomes are successful
experimental dispositions and must not be reconditioned into positive claims.
