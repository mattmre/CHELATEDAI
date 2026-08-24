# Matched held-out campaign runtime boundary

`egv.experiment.heldout` implements the frozen analysis slice from ADR-0001. It
plans exactly 192 A-H trajectories and 36 correction-shock trajectories, uses
balanced deterministic execution order, rejects coordinate substitutions, and
resumes from atomic per-coordinate records with a canonical hash-chain index.
Its analysis resamples only paired task-seed blocks and applies the preregistered integrity,
estimability, correction, efficacy, cost, and ordered disposition rules.

## What must be supplied by the two-Spark runtime

The module deliberately does not contain a model-server or evaluator transport.
`run_pending_coordinates` requires three injected callables plus a durable
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
admission.

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

## Claim boundary

Passing unit tests proves deterministic scheduling, closed-schema ingestion,
fail-closed auditing, and reproducible metric/disposition calculations on
fixtures. It does not prove that any Qwen trajectory ran, that an evaluator
signature verified in a live campaign, that the two Sparks transported work,
that LoRA improved held-out performance, or that any research gate passed. Those claims require
all 228 exact live coordinates, authenticated receipts and replay evidence, a
sealed adapter and protocol, independent review, and a matching signed
restoration receipt.
