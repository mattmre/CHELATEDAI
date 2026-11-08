# EGV/AVO recommissioning live attestation gateway

Status: engineering control plane implemented; live model deployment not yet
commissioned by this artifact.

## Purpose and claim boundary

The gateway supplies the production-side HTTP contract expected by
`AttestedLoopbackService`. It gives a controller an offline-capable way to
authorize one immutable model profile, runtime-configuration digest, service
signing key, generation-receipt key, and opaque service nonce. A separately
provisioned service signer then attests the exact controller launch and serves
that observation from a numeric-loopback-only gateway.

This component does not train or evaluate a model. Its tests use a local fake
upstream and contain no corpus prompts. Passing them proves the gateway's
closed HTTP and signature contracts, not model quality, GPU residency,
cross-node transport, or a live container/image identity.

## Authority separation

1. An external provisioner creates and protects the controller, service, and
   generation-receipt Ed25519 keys. The gateway never generates keys.
2. `issue-launch` loads the private deployment config, controller private key,
   service public key, and generation-receipt public key. It writes one new
   canonical `egv-trusted-launch-receipt-v2` record with atomic no-overwrite
   semantics.
3. `serve` loads only the controller public key, service private key, trusted
   launch record, and the same private deployment config. It rejects any key,
   nonce, model profile, or runtime digest mismatch before binding.
4. The live gateway freezes the upstream health and model-list response bytes.
   Every public read and every chat request rechecks those exact bytes. Chat is
   checked both before and after forwarding. The gateway permanently latches
   closed after any snapshot failure or drift.

The public launch and service-observation records contain no bind address,
upstream address, filesystem location, or host label. The private runtime
preimage includes the exact bind and upstream origins plus all gateway limits;
only its SHA-256 digest crosses into the signed public launch.

## Private configuration

The input is one closed `egv-recommissioning-gateway-config-v1` JSON object.
It contains:

- `bind_host`: numeric loopback only;
- `bind_port`: explicit nonzero port;
- `upstream_origin`: normalized numeric-loopback HTTP origin;
- `upstream_timeout_ms`: bounded request and socket timeout;
- `max_request_bytes` and `max_response_bytes`: each at most 4 MiB;
- `max_concurrency`: 1 through 64;
- `service_nonce`: externally provisioned `nonce_` plus 256 bits of lowercase
  hexadecimal entropy;
- `model_profile`: the exact closed
  `egv-recommissioning-model-service-v3` public manifest.

The configuration is private because it contains deployment topology. Do not
commit it or attach it to public evidence.

## Entry points

After installation, issue one launch record:

```text
egv-recommissioning-gateway issue-launch \
  --config <private-config> \
  --controller-private-key <controller-key> \
  --service-public-key <service-public-key> \
  --generation-receipt-public-key <generation-public-key> \
  --output <new-public-launch-record>
```

Then start the gateway under a supervised service manager:

```text
egv-recommissioning-gateway serve \
  --config <private-config> \
  --controller-public-key <controller-public-key> \
  --service-private-key <service-key> \
  --trusted-launch-receipt <public-launch-record>
```

Private key files must contain exactly 32 raw Ed25519 bytes. On POSIX systems,
group or world permission bits are rejected. Public keys may be raw Ed25519 or
PEM. Key and config symlinks are rejected. The launch output parent must exist,
and an existing output is never replaced.

The gateway exposes only three exact reads and one exact write route:

- health snapshot;
- single-model list snapshot;
- signed EGV service observation;
- chat completions proxy.

All other routes or methods fail. Transfer encoding, content encoding,
redirects, environment proxies, noncanonical chat JSON, mismatched model IDs,
oversized bodies, and non-loopback peers fail closed. HTTP access logging is
disabled so request material is not written by this process.

## Remaining live-deployment gates

The controller must obtain independent evidence for the facts that an HTTP
model API cannot prove by itself:

- the upstream process was launched from the pinned engine image and exact
  runtime arguments;
- the loaded weights, tokenizer assets, chat template, precision, and cache
  configuration match the signed model profile;
- the service and generation private keys are isolated from the controller and
  from each other;
- the supervised process, filesystem mounts, and network namespace prevent an
  untrusted process from impersonating the frozen loopback upstream;
- an exact live deployment survives a fresh client connection and a bounded
  non-corpus control before any task-bearing request is authorized.

Until those external facts are collected and independently reviewed, this is a
working loopback trust-chain implementation but not a `GO_TASK_BEARING` receipt
and not evidence of live distributed execution.
