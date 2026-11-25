# chelated-kernel

Zero-services Python wheel for tenant-partitioned index wrap, a stratified ledger, and streaming suppression. Bundle #49. Version 1.0.0. MIT license.

Runtime dependencies are `numpy` and `cryptography`. The wheel does not bundle benchmark scripts, tests, or repository modules. `chelated_kernel/*.py` stays under 2,500 physical lines.

This package does not call a model. `llm_write_tokens` stays 0. The cold-start gate runs 36 deterministic local sessions (3 families x 4 fixtures x 3 seeds) and records `live_model_calls = 0`. Those sessions are a local smoke of the wheel. They are not live model calls.

The audit signing key is derived with SHA-256 from the tenant, policy, and disclosure mode. It is an audit key, not an HSM key.

## Public API

`chelated_kernel.__all__` is exactly:

- `wrap_index`
- `EventLedger`
- `wrap_stream`
- `negotiate_capability`

```python
from chelated_kernel import EventLedger, negotiate_capability, wrap_index, wrap_stream
```

`negotiate_capability.UnsupportedCapabilityError` is the typed failure for anything outside the wheel.

## wrap_index

```python
wrapped = wrap_index(
    backend,
    old_dim=384,
    new_dim=256,
    unlabeled_anchors=anchors,
    tenant_id="tenant-a",
    profile="enterprise_search",
)
hits = wrapped.search(query, k=10)
```

`enterprise_search` is the only index profile. It fits a rectangular or square orthogonal map from unlabeled paired anchors (ZeroMu) and stores an 8-byte little-endian sidecar per document: float32 centered norm, then float32 CSLS radius. TRA-1 requires 32 fit anchors and 16 validation anchors, route confidence and its 95% bootstrap lower bound at or above 0.70, and 1,000 bootstrap resamples. Labeled anchors, non-finite values, degenerate clouds, and a failed orthogonal check are rejected.

Backends may be a dict, a callable `{"op": "list", "tenant_id": ...}`, a SQL object with `execute`, a Qdrant-style `scroll`/`search`, or a FAISS-style `reconstruct`/`search`/`ntotal`. SQL listing uses a bound parameter:

```sql
SELECT id, embedding, tenant_id
FROM chelated_vectors
WHERE tenant_id = %s
ORDER BY id ASC
```

`search` has no tenant override. Rows from any other tenant are dropped and are not counted as authorized reads.

## EventLedger

```python
ledger = EventLedger(policy="sec_audit", tenant_id="tenant-a")
ledger.ingest(envelope)
rows = ledger.as_of(valid_time, tx_time)
ledger.verify_chain()
certificate = ledger.non_nda_certificate()
```

Policies: `sec_audit`, `cve_vex`, `gdpr_erasure`, `dpu_fabric`, `agent_pool`.

Envelopes:

- CloudEvents 1.0 (`specversion`, `type`, `id`, `source`, `data`). `com.chelated.sec.8k` with item `4.02` supersedes the named prior claim. `com.chelated.dpu.flow` with `register == 11` is disputed. `com.chelated.agent.assertion` records an agent-pool claim. `com.chelated.stream.suppression` is audit-only.
- W3C PROV `prov:entity` and `wasDerivedFrom`. A positive cycle of non-base derived claims is unfounded and becomes withdrawn.
- OSV (`id`, `affected`) and CSAF VEX (`document`, `vulnerabilities` / `product_status`). Equal-authority affected versus not-affected claims are disputed. Unlinked repeats of the same id and status collapse to one echo cluster.
- Git `commit` (7-64 hex characters) and `symbols`.
- CDC `op` (`c`, `u`, `d`), `table`, and `pk`. An update supersedes the prior row. A delete erases it. `gdpr_erasure` replaces payloads with `{erased: 1}`.

`as_of(valid_time, tx_time)` recomputes truth without deleting history. Wire values are `current`, `withdrawn`, and `disputed`. `CITES_AUDIT` keeps the citing claim current and does not restore or defeat the cited claim. A lower-authority attack is blocked and does not withdraw the target. `tx_time` is the ledger's integer counter.

`non_nda` receipts are 56 signed int64 values, an Ed25519 signature, and a hash chain. The certificate text is decimal integers plus hex. `non_nda_ip_report()` reports `dirty_fields`, `checked_fields` (56), `payload_leaks`, and `ratio`.

## wrap_stream

```python
guard = wrap_stream(
    token_iter,
    ledger=ledger,
    restricted_spans=[("do", "not", "emit")],
    profile="strict_holdback",
    tenant_id="tenant-a",
)
for token in guard:
    consume(token)
```

Profiles are `strict_holdback` and `async_armed`. `strict_holdback` keeps a window of `max(span length)` tokens before the first yield. `async_armed` yields a token that is not a prefix of a restricted span immediately, and drops the span once it completes. Neither profile yields a completed restricted span. The ledger records `com.chelated.stream.suppression` with the span SHA-256 only. Empty spans are rejected. A span dict whose `tenant_id` is a different tenant is ignored. The ledger tenant must match.

## negotiate_capability

`negotiate_capability(name, **kwargs)` returns a `CapabilityGrant` for a supported name. These names always raise `UnsupportedCapabilityError`:

- `bare_metal_p4_tcam_asic_dataplane`
- `physical_weight_erasure_unlearning`
- `hdd_servo_mcu_firmware`
- `cross_tenant_raw_join`

Unknown names fail closed. Keyword arguments that request a cross-tenant join fail closed even on a supported name.

## Tenant id

`tenant_id` matches `[A-Za-z0-9._:-]{1,64}`. Any other string, including a SQL fragment, raises `TenantIsolationError` before a query or an ingest.

## Build

From this directory:

```bash
python -m pip wheel --no-deps --no-build-isolation .
```

The Bundle #49 verifier is `scripts/run_pv7_w17_6d_sdk_coldstart_gate.py`. It builds the wheel, counts lines, writes a dependency SBOM, checks the license allowlist, plants a two-tenant canary, checks the unsupported-capability canary, and blocks private imports.
