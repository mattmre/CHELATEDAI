# EGV Slice 2 evidence core

This slice implements the evidence substrate described by ADR-0001 and the
corresponding replay sections of the dual-Spark runbook. It is intentionally
offline: it does not start or stop services, inspect the private restore
inventory, access hidden tests, use credentials, load hosted models, or alter
the active DeepSeek workload.

## Production entry points

```python
from egv import EvidenceLedger, ReceiptSigner, InMemoryProjection

ledger = EvidenceLedger("ledger.sqlite", blob_root="private")
```

`EvidenceLedger` is the only writable SQLite handle. A file-backed writer lock,
WAL, `synchronous=FULL`, foreign keys, immutable-table triggers, and a ledger
hash chain protect the authoritative history. Read-only handles use SQLite
`mode=ro` plus `query_only`. `append_correction` and `append_retraction` add
history; `event_disposition`, `candidate_disposition`, and
`current_valid_events` derive the current view through dependency reachability.

`export_jsonl()` emits canonical JSONL. `EvidenceLedger.replay_jsonl()` imports
that export into a fresh database and verifies the same event hashes and head.
Checkpoint records are included after the event records and replay with
campaign/durable-event/hash validation plus conflicting-ID detection. Payloads
above the 16 KiB inline limit require a configured content-addressed private
blob root; missing or mismatched blob bytes fail integrity verification.

`ReceiptSigner` creates Ed25519 receipts. `ReceiptJournal` is a durable,
append-only evaluator journal; `EvidenceLedger.ingest_receipt()` verifies the
signature, sequence, previous receipt hash, and idempotency key before the
atomic ledger append. The first valid evaluator key is pinned to the ledger;
mixed-key deliveries are rejected before commit. Invalid or conflicting
deliveries are quarantined.

`InMemoryProjection` is an explicit dependency-free test/replay projection.
`QdrantProjection` is selected explicitly and raises a clear
`OptionalDependencyError` when `qdrant-client` is unavailable; it never falls
back to memory. Rebuilds derive point IDs and payloads only from a committed
ledger snapshot and enqueue failed Qdrant writes for later rebuild.

`PublicProjection`, `PublicEventChain`, and
`PublicCryptographicVerifier` implement the closed candidate/dependency,
receipt-envelope, lifecycle-event, strict restore-receipt, and terminal-seal
schemas. The verifier calculates each disposition from signed facts and the
corrected dependency graph before comparing it with the signed recorded
disposition. Its report is explicitly **public cryptographic decision replay**:
it does not rerun hidden tests, prove that an effect physically occurred, or
reconstruct private content from a digest.
Public exports contain only the closed pseudonymous IDs, digests, signatures,
and allowlisted public blob roles; evaluator credentials, IPC tokens, local
paths, and private topology are not public fields.

## Safe CLI paths

```text
python -m egv smoke --json
python -m egv smoke --json --two-process
python -m egv status --ledger ledger.sqlite
python -m egv export --ledger ledger.sqlite --output ledger.jsonl
python -m egv replay --input ledger.jsonl --output replay.sqlite --json
python -m egv verify-public --projection bundle --public-key bundle/receipts/evaluator-public-key.pem --protocol-digest <sha256>
```

`python -m egv smoke --json` is an explicitly **floor-tier synthetic** smoke:
it runs the ledger, signed receipt, public sealed replay, memory projection, and
ledger export/replay paths against a deterministic fixture. It reports that
Qdrant, a real campaign, and service paths were not exercised. The future
campaign/service/model phases named by the runbook are present as fail-closed
command names but are outside Slice 2; they perform no host or service action.

The `--two-process` variant adds a bounded CPU-only integration fixture. A
trainer process owns the writable ledger and authenticated Unix socket; a
separate evaluator process owns only its append-only receipt journal and sends
signed receipts through the evaluator-restricted `ingest_receipt` IPC method.
The trainer materializes verdict/effect rows after receipt ingestion. The pilot
mechanically denies evaluator SQLite connections and verifies that non-receipt
IPC is rejected. It uses deterministic fixtures and does not load a model,
start a service, inspect restore inventory, access credentials, or touch the
active DeepSeek workload.
