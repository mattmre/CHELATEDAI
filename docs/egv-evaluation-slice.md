# EGV Evaluation slice

This slice is the runnable, bounded Evaluation contract. It is deliberately
CPU-only and does not load a model, access a hosted service, change networking,
read credentials, access DeepSeek, or claim OpenShell support.

## Commands

```text
python -m egv evaluation freeze --output <new-empty-directory>
python -m egv evaluation smoke --json
python -m egv evaluation smoke --json --output <new-empty-directory>
```

`freeze` is write-once at the selected output directory. Given the same
evaluator-private seed, regeneration is deterministic and byte-identical. A
fresh CLI freeze creates a fresh evaluator-private seed; it never accepts a
seed value on the command line. It produces:

- 36 Python micro-repositories with the exact 20 train / 8 dev / 8 heldout
  distribution and the frozen family manifest;
- five immutable prompt IDs and the `egv-sft-row-v1` protocol metadata;
- a trainer view containing train and dev source only;
- a public view containing closed summary metadata only; and
- an evaluator-private view containing held-out source and hidden expected
  values.

The public `frozen/` view contains only the closed data summary, prompt
manifest, and protocol;
the complete data manifest, evaluator seed, hidden manifests, and private
`candidate/src/task.py` files are under `evaluator-private/`. The `freeze`
command does not create or claim to freeze the evaluator receipt journal; that append-only
journal is created by the running evaluator process during evaluation. The
root `artifact-manifest.json` lists only
`frozen/`, `trainer/`, and `public/` files and is scanned after it is finalized.

The smoke compares two freezes from the same private seed byte-for-byte, scans
trainer/public artifacts after final manifests for held-out IDs, private
material, credentials, and topology, and runs the real Docker isolation
adapter against the exact evaluator-private seed and held-out candidate
artifacts from the first freeze. It binds the evaluated corpus manifest digest
to the freeze digest and publishes only source digests and match evidence, not
the seed or private paths. The candidate receives one read-only content-addressed source
artifact through an ephemeral Docker volume and one opaque JSON input. It has
no hidden fixtures, host paths, Docker socket, credentials, evaluator key, or
ledger. Production controls cover hidden read/list/stat/chdir/readlink,
rename/link/unlink/truncate/chmod, process, network, same-UID signalling,
mount/umount, real libc `open`/`openat` attempts against the present read-only
source inode, ctypes socket/system escape, `userfaultfd`/`bpf`, PID/user
identity, host-path, credential, and source-artifact absence/mutation probes.
Hidden paths are reported as `ABSENT`; live-inode probes must return a real
`-1` with `EPERM`/`EROFS`. The AST validator is a production admission and
decision precondition for the narrow pure-return contract, not an isolation
boundary. Docker is the isolation boundary only: its return code, stdout, and
low-level probe output are untrusted evidence, and Docker does not authenticate
the result. After AST admission, the evaluator-private hidden oracle is the
decision authority. The local AST/audit/subprocess helper is test-only and
cannot substitute for Docker.

Docker is configured by `EGV_DOCKER_IMAGE` and `EGV_DOCKER_IMAGE_ID`; the
configured image must already be cached and match the pinned SHA-256 ID. No
image pull or network access is attempted. The frozen production limits are
network `none`, read-only root, `cap-drop=ALL`, no-new-privileges, uid/gid
`65534:65534`, 64 processes, 128 MiB memory, and a 16 MiB noexec bounded
`/tmp`. The candidate wall-clock timeout is exactly 2 seconds and stdout is
bounded at 65,536 bytes. `RESOURCE_BOUND` held-out tasks use a separate frozen
64 MiB cgroup ceiling, map an actual OOM to `RESOURCE_LIMIT` with the
`LIMIT_REACHED` resource bucket, and use the distinct `OUTPUT_LIMIT` bucket and
status when stdout exceeds its ceiling. No logical port or network listener
exists in this bounded path. The Docker seccomp profile is deny-default and
the post-load candidate filter removes EGV filesystem, process, network,
mutation, topology, memfd, `userfaultfd`, and `bpf` capabilities.

The Docker PID 1 runner keeps the authentication token and status classifier in
its supervisor interpreter, then forks and `execve`s a sanitized second-stage
Python bootstrap. That bootstrap forks a separate filtered worker, loads only
the content-addressed source and opaque input, denies `exit`/`exit_group` plus
the other candidate escape syscalls, and gives the worker only a bounded result
channel. The worker reports evidence and blocks; the bootstrap parent
terminates/reaps it and alone emits final JSON/status. The channel, frames,
stdout, and status are not trusted decision inputs: a candidate that shares
the worker process could otherwise forge them.

Therefore the trusted controller enforces the explicit `pure-return-v1`
candidate contract before Docker execution. The contract is a positive Python
AST allowlist for generated pure functions only: no imports, ambient
capability lookup, frame/FD introspection, process/network APIs, direct output,
dynamic calls, or dunder access. A source outside that contract is a bounded
`PROTOCOL_VIOLATION` and cannot be promoted, even if an untrusted low-level
sandbox probe emits the expected JSON or status. The hidden evaluator then
compares the captured content-addressed result bytes with its evaluator-private
oracle; only that trusted comparison plus contract validation can produce
`PASS`; Docker itself does not authenticate that result. The low-level Docker adapter defaults to the same contract. Its
explicitly named `untrusted-adversarial-v1` mode is used only by negative
controls to exercise kernel denials. Its result type structurally marks
evidence-only completion as `PROTOCOL_VIOLATION` with
`decision_eligible: false`; it cannot return `PASS` or enter the public
decision path. The mode is therefore not an alternative authority backend
and is never used by the controller.

A per-run handshake and host-verified sentinel authenticate filter/setup
failure. Candidate-controlled `os._exit(1)`, `os._exit(44)`, `SystemExit`, raw
exit syscalls, unknown nonzero statuses, forged JSON, forged sentinels, and
direct worker-channel frames are bounded runtime/protocol evidence; only an
authenticated runner-origin filter/setup failure may produce exit 44 and
`INFRASTRUCTURE_LOSS`.

It then runs the evaluator controller in a separate spawned process from the
trainer SQLite writer. The evaluator owns an ephemeral Ed25519 key and
canonical JSONL journal; only the public key/key ID leave that process, the
trainer is the only SQLite writer, an immutable `sqlite3.connect` audit denial
is installed at evaluator startup, ledger-writer co-hosting is attempted and
denied in the production path, and evaluator IPC is restricted to signed
`ingest_receipt`.

The smoke output's `runtime_tier` is
`ceiling-docker-evaluation-fixture`. That is the actual tier. It is not a
dual-Spark campaign, a model evaluation, a hosted-model result, or an
OpenShell result. The output must retain the explicit limitations, the
verified Docker isolation report, and the `openshell.claimed: false` status.

`INTERNAL_ERROR` is reserved for evaluator infrastructure loss. Candidate
syntax, runtime, timeout, resource, authority, protocol, and wrong-output
results remain their own bounded diagnostics and are never relabeled as
infrastructure loss. Every private/public `INTERNAL_ERROR` receipt carries
`task_family`, `normalized_public_locus`, `public_rule_id`, and an exact
recomputed incident-bound `failure_family_root`; SFT rows enforce the same
five-component canonical identity.

## Scope boundary

The public CLI still recognizes the later `preflight`, service-control,
training, variation, packaging, and restore names only to fail closed. This
slice does not start Variation, Training, or Campaign work. The administrative
block-flag check for this bounded implementation is run separately with
`python scripts/check_block_flag.py`; it is currently `CLEAR`, so no block
override is asserted. The operator-approved scope is this bounded Evaluation
slice only; any policy-required override record must remain outside public
artifacts. Later mutating/model phases remain
unavailable and fail closed. `--allow-provisional` is a public CLI option only
for explicitly verifying an unsealed public projection.
