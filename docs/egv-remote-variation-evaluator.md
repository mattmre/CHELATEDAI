# Independent Variation evaluator bridge

## Status and boundary

Production Variation can use an independently administered evaluator process,
including a second DGX Spark, without committing a hostname, address,
credential, private key, or hidden task input. The original same-host gateway
remains available; production accepts only its exact class or the exact sealed
remote gateway class.

The transport is a generic executable boundary. An operator-owned executable
may use SSH, herdr, a scheduler, or another transport, but its bytes are SHA-256
pinned. The repository does not record an infrastructure-specific wrapper.

## Freeze the trust manifest

Run this on the evaluator machine after its key, private seed, pinned Docker
image, and operator transport command exist:

```text
python -m egv variation freeze-evaluator-service \
  --campaign-id <frozen-campaign-id> \
  --model-digest <sha256> \
  --evaluator-revision <revision> \
  --evaluator-seed <private-seed-file> \
  --public-key <evaluator-public-key> \
  --command <operator-owned-command> \
  --output <private-service-manifest.json>
```

The freeze verifies the locally cached Docker image and records only digests
and public task bindings. It includes frozen `train` and `heldout` records and
rejects `dev`. Seed paths, hidden inputs, expected outputs, host identity,
transport arguments, and private key material are not serialized.

The pinned command reads one canonical JSON request from standard input and
returns one canonical JSON response. A typical evaluator-owned command invokes:

```text
python -m egv variation evaluator-once \
  --service-manifest <private-service-manifest.json> \
  --evaluator-seed <private-seed-file> \
  --private-key <evaluator-private-key> \
  --workspace <private-sandbox-workspace> \
  --state-root <private-durable-state-directory>
```

Operational paths belong in protected evaluator configuration, not this
repository or a public evidence bundle. Changing the launcher requires a newly
frozen manifest.

## Closed protocol

The generator sends candidate bytes and digest, public candidate/task bindings,
declared authority/locus, frozen campaign/model/protocol/policy/data/image and
service digests, and the authoritative next receipt sequence/head. The client
explicitly discards `opaque_input` and `candidate_source_path` before encoding.

The evaluator regenerates its corpus from its local private seed, resolves the
opaque input locally, verifies the public task binding, and executes through the
exact evaluator controller, local controller gateway, Docker candidate sandbox,
and candidate-execution authority policy. It returns a closed result, an
Ed25519-signed private receipt suffix anchored to the requested ledger head,
and an Ed25519 signature over the complete response.

The caller rehashes the command immediately before invocation, executes an
immutable content-addressed copy, enforces the 600-second and 1 MiB response
bounds, verifies the response signature, complete receipt chain, key,
sequence/head, all frozen bindings, and result-versus-receipt consistency. Only
then is the whole suffix appended in one SQLite transaction. Replay,
stale-head races, timeouts, wrong keys, altered commands/manifests, digest
substitution, malformed receipt order, and response tampering fail without
partial receipt ingestion.

## Evidence boundary

`tests/test_egv_variation_remote.py` exercises the real subprocess client,
content-pinned command copy, closed request, signature/receipt verification,
private-input/path exclusion, atomic rollback, replay rejection, wrong keys,
stale/tampered bindings, timeout, and train/heldout/dev split rules. The
evaluator-side Docker route is production code but requires the operator's
actual pinned image and private artifacts; dependency-light tests do not claim
a remote Spark or Docker execution occurred.
