# Matched held-out campaign runtime boundary

`egv.experiment.heldout` implements the frozen analysis slice from ADR-0001. It
plans exactly 192 A-H trajectories and 36 correction-shock trajectories, uses
balanced deterministic execution order, rejects coordinate substitutions, and
resumes from atomic per-coordinate records with a canonical hash-chain index.
Its analysis resamples only paired task-seed blocks and applies the preregistered integrity,
estimability, correction, efficacy, cost, and ordered disposition rules.

## What must be supplied by the two-Spark runtime

The generic scheduler deliberately does not assume a model-server or evaluator
transport. `run_pending_coordinates` requires three injected callables plus a durable
`CoordinateOperationStore`:

1. A coordinate runner performs the actual candidate trajectory under the
   coordinate's frozen model, adapter, prompt, policy, evaluator, and resource
   digests.
2. An evaluator-owned result verifier authenticates the frozen evaluator
   identity, verifies every signature and required receipt, performs private
   ledger-only and public cryptographic replay, checks isolation, and returns an
   evaluator-signed `egv-heldout-signed-result-v1` envelope around the closed
   `egv-heldout-result-v1` record.
3. An evaluator-owned reconciler examines any prior `DISPATCHING`, unindexed
   `COMPLETED`, or `QUARANTINED` operation before another external dispatch.
   Its signed decision must be `COMPLETED`, `NOT_EXECUTED`, or `UNKNOWN`.
   `UNKNOWN` remains quarantined and is never rerun.

The runner's output is never journaled directly. Boolean integrity fields are
facts produced by the trusted verifier; the scheduler cannot turn a model or
runner self-report into cryptographic evidence. Every admitted envelope binds
the exact coordinate, execution profile, result payload, receipt-collection
root, ledger head, evaluator digest, key ID, and protocol digest. The journal
verifies the Ed25519 signature against the public key frozen into that protocol
before atomically admitting it. A production integration must keep the verifier
and signing key on the evaluator Spark.

`HeldoutJournal.append` is also exposed for evaluator-side recovery and import.
Callers using it directly must provide the same signed envelope. The journal
verifies its signature and exact bindings, but the receipt collection and
hidden evaluator materials stay evaluator-private. The durable operation store
assigns one stable idempotency key per coordinate. A crash after dispatch leaves
`DISPATCHING`; only a signed reconciliation may make that coordinate safe to
retry. Journal records are atomically replaced individual objects, and an
atomically replaced index binds their canonical hash chain, avoiding torn JSONL
admission. The scheduler holds one cross-process campaign lock across recovery,
dispatch, journal admission, and the terminal operation transition. Journal
append and per-coordinate operation transitions also have their own
cross-process transaction locks, so independently started scheduler processes
cannot both dispatch the same absent coordinate. Every scheduler, operation,
command-build, and verifier-run admission reconstructs the sealed protocol and
rejects in-memory coordinate or binding substitution.

## Public and private outputs

`analyze_heldout_campaign` returns aggregate arm counts, paired intervals, gate
estimability, the complete gate vector, and the ordered disposition. It contains
no per-coordinate IDs, held-out task addresses, schedule, fixture IDs, or
execution-profile digests. Campaign IDs supplied for publication must be
pseudonymous and public-safe. Protocol construction enforces the bounded
`egv-campaign-<hex>` form.

`build_private_campaign_record` requires all 228 signed envelopes plus the
canonical evaluator-signed restoration receipt. It re-verifies both, recomputes
the aggregate analysis rather than accepting a caller-provided report, and
binds exact-result and restoration digests. It is private evidence and must not
be copied into the public repository as a live artifact.

## Sealed Qwen production integration

`egv.experiment.production` supplies the reviewed integration named
`qwen-heldout-production-v1`. It is a local, artifact-byte-bound execution path;
it does not claim a network topology or whole-process source isolation. The
production deployment manifest records the machine-readable isolation scope
`variation-router-only-v1` and binds the exact protocol, public trainer package,
base-model and adapter manifests, Variation evaluator manifest/key/command,
held-out evaluator service/command, Python executable, Qwen revision, source
commit, device, dtype, token budget, command timeout, and input/output byte
ceilings. The exhaustive model and adapter manifests remain responsible for
every checkpoint/adapter file beneath their respective roots.

Only the Variation receipt router is admitted through pre-import source
verification and an isolated/no-site interpreter. The trainer/current process
and held-out evaluator command are byte-bound, but their imported EGV and
third-party runtime code remains operator-trusted; the declarative source
commit does not prove those processes executed reviewed source. Consequently,
this integration must not be described as wholly content-carried,
source-isolated, or source-reviewed until equivalent launchers and source
manifests are implemented for those two execution boundaries.

The runtime loads the frozen base Qwen checkpoint and a separately loaded copy
with the sealed LoRA adapter. Adapter arms must carry the loader-issued adapter
attestation, and its applied state digest must equal the protocol's
`trained_model_digest`. Base arms cannot carry an adapter; adapter arms cannot
fall back to the base generator. Both use the real `BoundedCandidateLoop`, the
source-only response contract, and the exact remote Variation evaluator.

For each main coordinate, the trainer writes an evaluator-replayable private
bundle containing the exact Variation report, candidate-source bytes used for
the measured token count, canonical ledger export, receipt root, and ledger
head. The held-out evaluator reconstructs the ledger, verifies every Variation
receipt signature and chain, repeats private-ledger and public-receipt promotion
decisions, re-tokenizes the exact candidate source, and derives the result again
before signing it. Shock bundles bind the sealed block snapshot, runtime
journal, durable per-attempt operation records, candidate-source bytes,
correction/policy events, receipts, and ledger export. These bundles are private
evaluator inputs, not public research artifacts.

Every coordinate owns an independent receipt chain. The production Variation
command is therefore the reviewed content-addressed receipt router, not the
single-state `variation evaluator-once` command. It caches an exact operation
response for the common six-attempt PRE prefix, snapshots each signed receipt
head, and forks policy-specific POST chains from that immutable head. A durable
pending evaluator operation is kept quarantined after a crash. The path-free
router manifest binds the exact launcher bytes, service-manifest identity,
private-configuration digest, full source commit, every EGV Python source-file
digest, and the exact Python/bootstrap executable digests. The extensionless
POSIX launcher enters Python with isolated/no-site flags, rejects any preloaded
EGV module, and compiles freshly rehashed admitted source bytes through a
source-only importer. Explicit runtime import roots are private configuration;
they are never discovered from an environment variable or `.pth` startup file.
The operation directory is assembled off-path and atomically renamed only after
its exact request and receipt-anchor snapshot are durable, so an interruption
before publication leaves no poisoned operation root. The manifest also binds
the closed
`content-addressed-receipt-chain-fork-v1` capability. Admission rejects an
ordinary single-state command, a `.py` launcher that would bypass the shebang,
or a self-asserted/unreviewed router.

The trainer records `BEGIN_PENDING` before sending evaluator `BEGIN`. Reissuing
the same BEGIN is request-idempotent, so a crash at that boundary cannot cause a
model rerun. Completed execution is persisted before `VERIFY`; recovery sends
the exact observation through VERIFY and then obtains signed `COMPLETED`
reconciliation. A begun operation without durable execution evidence reconciles
as `UNKNOWN`, is quarantined by the scheduler, and is never retried.

## Bounded command-line workflow

The `heldout` command group exposes only the implemented local/content-bound
seams:

```text
python -m egv.cli heldout prepare ...
python -m egv.cli heldout freeze-evaluator-service ...
python -m egv.cli heldout freeze-production-deployment ...
python -m egv.cli heldout evaluator-once ...
python -m egv.cli heldout run ...
python -m egv.cli heldout finalize ...
python -m egv.cli variation write-receipt-router-command ...
python -m egv.cli variation freeze-evaluator-service ...
python -m egv.cli variation freeze-receipt-router-manifest ...
```

- `prepare` deterministically freezes the protocol and the exact public
  trainer inputs/source bundle from an evaluator-owned seed. It refuses to
  replace any frozen output. Frozen JSON, seed, and key inputs must be bounded
  single-link regular files whose entire path traverses no symlink or reparse
  ancestor. If the multi-output publication fails, already published members
  are moved out of their public paths into uniquely named recoverable rollback
  quarantine directories. Rollback never unlinks an output pathname; a file
  another process substituted at that path is preserved with its exact file
  identity and restored without replacing any newer occupant. Quarantines are
  private operator evidence and must be reviewed before manual removal.
- `freeze-evaluator-service` writes the path-free identity manifest derived
  from that exact protocol.
- `freeze-production-deployment` admits every explicit file/root listed above,
  verifies semantic model/adapter/evaluator/router bindings, and refuses to
  overwrite its path-free deployment manifest. No password, host name, private
  topology, or authentication material belongs in that manifest.
- `variation write-receipt-router-command` writes the deterministic private
  extensionless POSIX launcher from explicit absolute evaluator paths, EGV
  package root, full source commit, Python executable, bootstrap executable,
  and one or more runtime import roots. On the Sparks the bootstrap executable
  is the fixed `env` binary used only to split the exact `-I -S` shebang; both
  it and the selected Python executable are byte-bound in the router manifest.
  Create the launcher before freezing the Variation service because that
  service binds the command digest. Then
  `freeze-receipt-router-manifest` binds the completed service and command.
  The launcher contains evaluator-local paths and must remain private; only its
  digest and the path-free routing manifest belong in review artifacts.
- `evaluator-once` accepts one bounded JSON command on standard input. It
  requires an evaluator private key and a name from the built-in closed
  evaluator-integration registry. User-controlled module imports are rejected.
  Selecting `qwen-heldout-production-v1` additionally requires the deployment
  manifest and every explicit artifact/root argument; no environment variable
  or module path supplies missing configuration. The command provides no
  network transport.
- `run` resumes the canonical schedule and requires explicit local
  names from the built-in runner, verifier, and reconciler registries.
  Production requires `qwen-heldout-production-v1` for all three names together,
  plus the sealed deployment/artifact arguments (including
  `--variation-receipt-router-manifest`), runtime root, and dispatch root.
  Mixing production and fixture/sentinel names fails closed.
- `finalize` refuses partial evidence. It writes a private record only after
  all 228 signed outcomes and a matching signed restoration receipt verify.

These commands do not discover hosts, open SSH connections, restore services,
or publish a package. An operator-controlled wrapper may carry canonical
command/response bytes over an approved channel, but live cross-node execution
must be demonstrated and recorded separately. The digest-pinned executor proves
only the exact command bytes it launched; it does not prove where that command
ultimately transported work.

On Windows systems without long-path support, use short explicit
`--runtime-root` and `--dispatch-root` paths; full coordinate and run identities
can exceed the legacy path limit. This portability note does not affect the
Linux Spark target.

The current local evidence consists of unit tests for protocol scheduling,
runtime derivation, shock recovery, content-bound verifier behavior, analysis,
receipt-chain branching/cache/quarantine behavior, and CLI fail-closed behavior.
Trainer wall time remains monotonic trainer telemetry; evaluator seconds are
reconstructed from signed effect receipts. Exact pass counts belong in the
immutable PR/test record after the complete tree is rerun; this document does
not turn fixture passes into live campaign evidence.

## Claim boundary

Passing unit tests proves deterministic scheduling, closed-schema ingestion,
command/manifest substitution resistance, crash/quarantine behavior, and
reproducible metric/disposition calculations on fixtures. It does not prove
that any Qwen trajectory ran, that an evaluator signature verified in a live
campaign, that the two Sparks transported work, that LoRA improved held-out
performance, or that any research gate passed. Those claims require
all 228 exact live coordinates, authenticated receipts and replay evidence, a
sealed adapter and protocol, independent review, and a matching signed
restoration receipt.
