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
the evaluator revision derived from that closed Variation manifest, held-out
evaluator service/command, its explicit `python-json-v1` execution mode, Python
executable, Qwen revision, source commit,
device, dtype, token budget, command timeout, and input/output byte ceilings.
The closed deployment schema is `egv-heldout-production-deployment-v2`; v2 adds
the explicit held-out evaluator execution-mode binding and is intentionally not
accepted as the earlier draft v1 shape.
Admission proves the stored evaluator revision still matches the exact pinned
service-manifest bytes and protocol service digest; verdict projections cannot
select their own revision. The exhaustive model and adapter manifests remain
responsible for every checkpoint/adapter file beneath their respective roots.

The monotonic execution deadline is captured before native process setup and
covers request writing, response reading, and evaluator execution. Python cannot
preempt a blocking operating-system process-creation, executable-open, or Job
assignment call; a native setup call that returns after the deadline is rejected
as a timeout before request I/O begins. This is a fail-closed late-return check,
not a claim of hard wall-clock preemption inside native setup. Standard input is
asynchronous and output/error streams are bounded, so a command that never reads
its request cannot prevent the execution timeout from being recognized. Cleanup
uses one five-second monotonic bound to terminate and reap the containment
boundary and join I/O workers, including after a successful parent exit, and
fails closed if any worker remains alive. On
Windows, the process is created suspended, assigned to a kill-on-close Job
Object, and only then resumed; every exit path terminates that job. If Job
assignment fails, the still-suspended unassigned child is killed directly and
reaped within the cleanup deadline, and incomplete kill/reap cleanup fails
closed. Native startup failures remain primary while every secondary tree,
kill, reap, stream-close, and owned-handle-close failure is retained as cleanup
evidence; owned Windows handle closes are checked rather than assumed. On POSIX,
every exit path terminates the admitted
wrapper and descendants that remain in its new session/process group. The
leader is re-anchored with `waitid(..., WNOWAIT)` immediately before each
`killpg` and is not reaped until afterward; loss of that child anchor skips
`killpg` rather than signaling a potentially recycled process-group ID. The
reviewed POSIX command contract therefore forbids `setsid`/`setpgid` or another
daemonizing escape; this path does not claim containment of a hostile process
that deliberately leaves the admitted group.

The held-out evaluator mode is Python-only and never derives from a filename
suffix or shebang: identical admitted command bytes execute through the same
manifest-bound Python interpreter whether their private pathname ends in
`.py` or has no extension. On Linux, the verified interpreter bytes are copied
into a size/digest-verified anonymous executable, sealed against writes and
size changes, and executed through its inherited descriptor. On Windows, a
deny-write/delete handle remains held from verification through process
creation. The held-out evaluator's admitted runtime imports remain an explicit
operator-trusted boundary; it is not launched with isolated/no-site flags.

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
the measured token count, canonical raw prompt/model/contract-response bytes,
canonical ledger export, receipt root, and ledger head. The held-out evaluator
reconstructs the ledger, verifies every Variation receipt signature and chain,
repeats private-ledger and public-receipt promotion decisions, re-tokenizes the
exact candidate source, reparses successful source contracts, and requires a
recorded response-contract failure to reproduce the same closed, bounded,
outer-to-inner exception chain before signing. This deterministic replay binds
the classification to trainer-supplied private raw bytes; it does not prove
that those bytes originated from the model. Prompt-integrity failures preserve
only rendered bytes whose digest differs from the expected prompt. The other
failure shapes are exact: `PROMPT_RENDER` preserves no raw artifacts,
`MODEL_GENERATION` preserves only the exact rendered prompt, and
`RESPONSE_CONTRACT` preserves rendered, decoded, and contract bytes.
`PROMPT_RENDER` and
`MODEL_GENERATION` transport/runtime exceptions have no output from which the
exception can be independently reproduced; their closed verification mode is
therefore `TRAINER_ATTESTED_RUNTIME_FAILURE`, with available raw bytes and stage
shape validated but no claim that the evaluator reproduced the runtime error.
The chain-bearing private generation record, write-ahead intent, main/shock
evidence containers, and signed observation envelope are v2 schemas. Prior v1
campaign evidence is deliberately non-resumable rather than being silently
reinterpreted under this stronger verifier contract.
Shock bundles apply the same raw-generation rules while binding the sealed block
snapshot, journal, durable per-attempt operation records, candidate-source
bytes, correction/policy events, receipts, and ledger export. These bundles and
raw generations are private evaluator inputs, not public research artifacts or
public result fields.

Ledger evidence crosses the evaluator boundary as closed canonical JSONL. Every
raw record must have the exact export schema, and replay must reproduce the input
byte-for-byte; unknown fields, duplicate records, and normalization-only variants
are rejected. Signed receipt events must preserve the global signed-chain order.
Shock replay additionally reconstructs the deterministic global event sequence:
the six PRE operations precede corrected-premise/correction/commit/policy, and
the POST run and every POST operation event follow policy activation. Ledger
`created_at` values remain hash-bound observed provenance; they are not claimed
as independently authenticated wall-clock time.

Every coordinate owns an independent receipt chain. The production Variation
command is therefore the reviewed content-addressed receipt router, not the
single-state `variation evaluator-once` command. It caches an exact operation
response for the common six-attempt PRE prefix, snapshots each signed receipt
head, and forks policy-specific POST chains from that immutable head. A durable
pending evaluator operation is kept quarantined after a crash. The path-free
router manifest binds the exact launcher bytes, service-manifest identity,
private-configuration digest, full source commit, every EGV Python source-file
digest, and the exact Python/bootstrap executable digests. Production does not
execute the extensionless command through its shebang or bootstrap. The gateway
opens and rehashes the manifest-bound Python executable, keeps that exact file
identity open/locked through process creation, and invokes it explicitly as
`python -I -S <content-addressed-router>`. On POSIX the launch path is the open
descriptor under `/proc/self/fd` or `/dev/fd`, preserved with `pass_fds`; on
Windows a read-only handle denies write and delete sharing until `CreateProcess`
has opened the verified pathname. The router then rejects any preloaded EGV
module and compiles freshly rehashed admitted source bytes through a source-only
importer. Explicit runtime import roots are private configuration;
they are never discovered from an environment variable or `.pth` startup file.
The operation directory is assembled off-path and atomically renamed only after
its exact request and receipt-anchor snapshot are durable, so an interruption
before publication leaves no poisoned operation root. The manifest also binds
the closed
`content-addressed-receipt-chain-fork-v1` capability. Admission rejects an
ordinary single-state command, a `.py` command that would escape the reviewed
extensionless-router contract, or a self-asserted/unreviewed router.

This executable-identity boundary closes atomic pathname replacement between
verification and launch. It does not claim protection from same-user process
injection, debugger access, or in-memory tampering after process creation.

The trainer records `BEGIN_PENDING` before sending evaluator `BEGIN`. Reissuing
the same BEGIN is request-idempotent, so a crash at that boundary cannot cause a
model rerun. Completed execution is persisted before `VERIFY`; recovery sends
the exact observation through VERIFY and then obtains signed `COMPLETED`
reconciliation. A begun operation without durable execution evidence reconciles
as `UNKNOWN`, is quarantined by the scheduler, and is never retried.

Model generation has its own narrower write-ahead boundary. `STARTED` is
persisted before the non-idempotent model call and is the only state allowed to
finish an already durable generation intent without calling the model again.
Once an operation is `GENERATED`, `GENERATION_FAILED`, or `COMPLETE`, its private
generation record, raw artifacts, and any trajectory/materialized ledger facts
are validate-only: missing or conflicting evidence fails closed and is never
recreated during reconciliation, terminal bundle materialization, or
verification. A main coordinate that consumes its exact 12-slot generation
budget produces evaluator-verifiable `BUDGET_EXHAUSTED` evidence instead of
stranding the campaign. The terminal bundle merges successful generations and
source-contract failures by exact attempt index: evaluated rejected candidates
retain their signed receipts and effects, while failed generations invent no
candidate or receipt. In correction-shock coordinates, a failed generation is
a durable, non-promoted attempt with no candidate, signed verdict, or effect; it
still counts toward the bounded attempt/cost window, while receipt denominators
count only attempts that actually reached the frozen evaluator.

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
  and one or more runtime import roots. The bootstrap and selected Python
  executable remain byte-bound in the router manifest, but the production
  gateway does not trust or execute the bootstrap/shebang path: it launches the
  bound Python explicitly with `-I -S` under the verified open/locked identity.
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
