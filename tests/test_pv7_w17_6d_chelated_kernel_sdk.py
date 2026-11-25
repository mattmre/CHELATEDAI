"""Runtime checks for the chelated-kernel wheel and the Bundle #49 cold-start gate."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "packages" / "chelated_kernel"


def _load_sdk():
    sys.path.insert(0, str(PKG))
    sys.path.insert(0, str(ROOT))
    import chelated_kernel as kernel
    from chelated_kernel import (
        EventLedger as event_ledger_cls,
        negotiate_capability as negotiate,
        wrap_index as wrap_idx,
        wrap_stream as wrap_strm,
    )
    from chelated_kernel.errors import (
        EnvelopeError as envelope_error,
        ProfileError as profile_error,
        ReceiptVerificationError as receipt_error,
        StreamGuardError as stream_error,
        TenantIsolationError as tenant_error,
        Tra1VerificationError as tra1_error,
    )
    from chelated_kernel.event_ledger import RECEIPT_SPEC as receipt_spec
    from chelated_kernel.index_wrapper import LIST_VECTORS_SQL as list_sql
    from scripts.run_pv7_w17_6d_sdk_coldstart_gate import (
        CANARY_A as canary_a,
        CANARY_B as canary_b,
        count_loc as count_lines,
        main as gate_main,
        synthetic_pair_anchors as pair_anchors,
    )
    from scripts.verify_non_nda_ip_clean import (
        FORBIDDEN_METHOD_PATTERNS as forbidden_patterns,
        scan_text_for_ip_leaks as scan_leaks,
    )

    return (
        kernel,
        event_ledger_cls,
        negotiate,
        wrap_idx,
        wrap_strm,
        envelope_error,
        profile_error,
        receipt_error,
        stream_error,
        tenant_error,
        tra1_error,
        receipt_spec,
        list_sql,
        canary_a,
        canary_b,
        count_lines,
        gate_main,
        pair_anchors,
        forbidden_patterns,
        scan_leaks,
    )


(
    chelated_kernel,
    EventLedger,
    negotiate_capability,
    wrap_index,
    wrap_stream,
    EnvelopeError,
    ProfileError,
    ReceiptVerificationError,
    StreamGuardError,
    TenantIsolationError,
    Tra1VerificationError,
    RECEIPT_SPEC,
    LIST_VECTORS_SQL,
    CANARY_A,
    CANARY_B,
    count_loc,
    main,
    synthetic_pair_anchors,
    FORBIDDEN_METHOD_PATTERNS,
    scan_text_for_ip_leaks,
) = _load_sdk()


class _Tokens:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.reads = 0

    def __iter__(self) -> _Tokens:
        return self

    def __next__(self) -> str:
        if self.reads >= len(self.tokens):
            raise StopIteration
        token = self.tokens[self.reads]
        self.reads += 1
        return token


class _SqlBackend:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE chelated_vectors (id TEXT, embedding BLOB, tenant_id TEXT)")
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def add(self, doc_id: str, vector: list[float], tenant_id: str) -> None:
        blob = np.asarray(vector, dtype="<f8").tobytes()
        self.conn.execute(
            "INSERT INTO chelated_vectors (id, embedding, tenant_id) VALUES (?, ?, ?)",
            (doc_id, blob, tenant_id),
        )

    def execute(self, sql: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
        self.calls.append((sql, params))
        return list(self.conn.execute(sql.replace("%s", "?"), params))


class _Qdrant:
    def __init__(self, points: list[dict[str, object]]) -> None:
        self.points = points
        self.filters: list[object] = []

    def scroll(self, scroll_filter: object, limit: int, offset: object) -> tuple[list[dict[str, object]], None]:
        self.filters.append(scroll_filter)
        return self.points, None

    def search(self, query: object, k: int) -> list[object]:
        return []


class _Faiss:
    def __init__(self, rows: list[tuple[str, str, list[float]]]) -> None:
        self.rows = rows
        self.ntotal = len(rows)
        self.ids = [row[0] for row in rows]
        self.tenant_ids = [row[1] for row in rows]
        self.search_calls = 0
        self.reconstructed: list[int] = []

    def search(self, query: object, k: int) -> tuple[list[list[float]], list[list[int]]]:
        self.search_calls += 1
        return [[0.0]], [[0]]

    def reconstruct(self, index: int) -> list[float]:
        self.reconstructed.append(index)
        return self.rows[index][2]


class KernelSdkTests(unittest.TestCase):
    def test_public_surface_and_loc(self) -> None:
        self.assertEqual(
            chelated_kernel.__all__,
            ["wrap_index", "EventLedger", "wrap_stream", "negotiate_capability"],
        )
        self.assertLess(count_loc(ROOT), 2500)
        self.assertEqual(len(FORBIDDEN_METHOD_PATTERNS), 56)

    def test_square_and_rectangular_recovery(self) -> None:
        for old_dim, new_dim in ((8, 8), (8, 6), (6, 8)):
            anchors, docs, queries = synthetic_pair_anchors(7, old_dim, new_dim)
            backend = {f"d{index}": row for index, row in enumerate(docs)}
            wrapped = wrap_index(
                backend,
                old_dim=old_dim,
                new_dim=new_dim,
                unlabeled_anchors=anchors,
                tenant_id="tenant-a",
            )
            hits = wrapped.search(queries[0], k=3)
            self.assertEqual(hits[0].doc_id, "d0", (old_dim, new_dim))
            self.assertEqual(hits[0].tenant_id, "tenant-a")
            self.assertGreater(wrapped.contract.confidence, 0.99, (old_dim, new_dim))
            self.assertGreaterEqual(wrapped.contract.confidence_lcb, 0.70)
            self.assertEqual(wrapped.contract.rectangular, old_dim != new_dim)
            self.assertLess(wrapped.contract.ortho_error, 1e-5)
            self.assertEqual(wrapped.contract.sidecar_bytes_per_doc, 8)
            self.assertEqual(wrapped.contract.llm_write_tokens, 0)
            self.assertEqual(len(wrapped.sidecar_bytes()), 8 * len(docs))
            self.assertEqual(len(wrapped.sidecar_bytes("d0")), 8)
            self.assertEqual(wrapped.unauthorized_reads, 0)

    def test_tra1_rejections(self) -> None:
        anchors, docs, queries = synthetic_pair_anchors(9, 8, 8)
        with self.assertRaises(Tra1VerificationError) as short:
            wrap_index({"d0": docs[0]}, old_dim=8, new_dim=8, unlabeled_anchors=anchors[:10], tenant_id="tenant-a")
        self.assertEqual(short.exception.code, "fit_count")
        constant = [([0.0] * 8, [0.0] * 8) for _ in range(64)]
        with self.assertRaises(Tra1VerificationError) as flat:
            wrap_index({"d0": docs[0]}, old_dim=8, new_dim=8, unlabeled_anchors=constant, tenant_id="tenant-a")
        self.assertEqual(flat.exception.code, "degenerate_anchors")
        broken = list(anchors)
        broken[0] = ([float("nan")] * 8, anchors[0][1])
        with self.assertRaises(Tra1VerificationError) as nan:
            wrap_index({"d0": docs[0]}, old_dim=8, new_dim=8, unlabeled_anchors=broken, tenant_id="tenant-a")
        self.assertEqual(nan.exception.code, "non_finite")
        labeled = list(anchors)
        labeled[0] = {"old": anchors[0][0], "new": anchors[0][1], "label": 1}
        with self.assertRaises(Tra1VerificationError) as labels:
            wrap_index({"d0": docs[0]}, old_dim=8, new_dim=8, unlabeled_anchors=labeled, tenant_id="tenant-a")
        self.assertEqual(labels.exception.code, "labeled_anchor")
        wrapped = wrap_index({"d0": docs[0]}, old_dim=8, new_dim=8, unlabeled_anchors=anchors, tenant_id="tenant-a")
        with self.assertRaises(Tra1VerificationError):
            wrapped.search([0.0, 1.0, 2.0], k=2)
        with self.assertRaises(TenantIsolationError):
            wrapped.search(queries[0], k=2, tenant_id="tenant-b")
        with self.assertRaises(TenantIsolationError):
            wrap_index({"d0": docs[0]}, old_dim=8, new_dim=8, unlabeled_anchors=anchors, tenant_id="tenant' OR 1=1")
        with self.assertRaises(ProfileError):
            wrap_index({"d0": docs[0]}, old_dim=8, new_dim=8, unlabeled_anchors=anchors, profile="missing")

    def test_backend_isolation(self) -> None:
        anchors, docs, queries = synthetic_pair_anchors(4, 8, 8)
        foreign = list(docs[0])
        foreign[0] = 777777.25
        sql = _SqlBackend()
        for index, row in enumerate(docs):
            sql.add(f"d{index}", row, "tenant-a")
        sql.add(CANARY_B, foreign, "tenant-b")
        wrapped = wrap_index(sql, old_dim=8, new_dim=8, unlabeled_anchors=anchors, tenant_id="tenant-a")
        self.assertEqual(sql.calls[0][0], LIST_VECTORS_SQL)
        self.assertEqual(sql.calls[0][1], ("tenant-a",))
        self.assertNotIn("tenant-a", sql.calls[0][0])
        hits = [hit.doc_id for hit in wrapped.search(queries[0], k=10)]
        self.assertEqual(hits[0], "d0")
        self.assertNotIn(CANARY_B, hits)
        self.assertEqual(wrapped.unauthorized_reads, 0)
        self.assertEqual(wrapped.rejected_foreign_records, 0)

        class _HostileSql(_SqlBackend):
            def execute(self, sql: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
                self.calls.append((sql, params))
                return list(self.conn.execute("SELECT id, embedding, tenant_id FROM chelated_vectors ORDER BY id"))

        hostile = _HostileSql()
        for index, row in enumerate(docs):
            hostile.add(f"d{index}", row, "tenant-a")
        hostile.add(CANARY_B, foreign, "tenant-b")
        hostile_wrapped = wrap_index(hostile, old_dim=8, new_dim=8, unlabeled_anchors=anchors, tenant_id="tenant-a")
        hostile_hits = [hit.doc_id for hit in hostile_wrapped.search(queries[0], k=10)]
        self.assertNotIn(CANARY_B, hostile_hits)
        self.assertGreater(hostile_wrapped.rejected_foreign_records, 0)
        self.assertEqual(hostile_wrapped.unauthorized_reads, 0)

        def listing(request: dict[str, object]) -> list[dict[str, object]]:
            self.assertEqual(request, {"op": "list", "tenant_id": "tenant-a"})
            rows = [{"id": f"d{index}", "vector": row, "tenant_id": "tenant-a"} for index, row in enumerate(docs)]
            rows.append({"id": CANARY_B, "vector": foreign, "tenant_id": "tenant-b"})
            return rows

        called = wrap_index(listing, old_dim=8, new_dim=8, unlabeled_anchors=anchors, tenant_id="tenant-a")
        called_ids = [hit.doc_id for hit in called.search(queries[0], k=10)]
        self.assertNotIn(CANARY_B, called_ids)
        self.assertEqual(called.unauthorized_reads, 0)
        self.assertFalse(np.any(called._docs == np.float64(777777.25)))

        points = [
            {"id": f"d{index}", "vector": row, "payload": {"tenant_id": "tenant-a"}}
            for index, row in enumerate(docs)
        ]
        points.append({"id": CANARY_B, "vector": foreign, "payload": {"tenant_id": "tenant-b"}})
        qdrant = _Qdrant(points)
        q_wrapped = wrap_index(qdrant, old_dim=8, new_dim=8, unlabeled_anchors=anchors, tenant_id="tenant-a")
        self.assertEqual(qdrant.filters[0], {"must": [{"key": "tenant_id", "match": {"value": "tenant-a"}}]})
        self.assertNotIn(CANARY_B, [hit.doc_id for hit in q_wrapped.search(queries[0], k=10)])

        faiss_rows = [(CANARY_B, "tenant-b", foreign)]
        faiss_rows.extend((f"d{index}", "tenant-a", row) for index, row in enumerate(docs))
        faiss = _Faiss(faiss_rows)
        f_wrapped = wrap_index(faiss, old_dim=8, new_dim=8, unlabeled_anchors=anchors, tenant_id="tenant-a")
        self.assertEqual(faiss.search_calls, 1)
        self.assertNotIn(0, faiss.reconstructed)
        self.assertNotIn(CANARY_B, [hit.doc_id for hit in f_wrapped.search(queries[0], k=10)])
        self.assertEqual(f_wrapped.unauthorized_reads, 0)
        self.assertNotIn(CANARY_A, json.dumps([hit.doc_id for hit in f_wrapped.search(queries[0], k=10)]))

    def test_sec_bitemporal_supersede(self) -> None:
        ledger = EventLedger(policy="sec_audit", tenant_id="issuer")
        ledger.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.sec.filing",
                "id": "rev-1",
                "source": "issuer",
                "time": 1000,
                "data": {"statement": "revenue"},
            }
        )
        ledger.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.sec.8k",
                "id": "8k-1",
                "source": "issuer",
                "time": 2000,
                "data": {"item": "4.02", "supersedes": "rev-1"},
            }
        )
        early = {row["claim_id"]: row["truth"] for row in ledger.as_of(1500, 1)}
        late = {row["claim_id"]: row["truth"] for row in ledger.as_of(2500, 2)}
        self.assertEqual(early, {"rev-1": "current"})
        self.assertEqual(late["rev-1"], "withdrawn")
        self.assertEqual(late["8k-1"], "current")
        self.assertEqual(ledger.llm_write_tokens, 0)

    def test_vex_dispute_and_echo(self) -> None:
        ledger = EventLedger(policy="cve_vex", tenant_id="secops")
        ledger.ingest({"id": "OSV-1", "affected": [{"package": {"name": "libfoo"}}], "time": 10})
        ledger.ingest({"id": "OSV-1", "affected": [{"package": {"name": "libfoo"}}], "summary": "echo", "time": 11})
        clusters = ledger.echo_clusters(100, 10)
        self.assertTrue(any(len(cluster) == 2 for cluster in clusters))
        before = ledger.as_of(100, 10)
        self.assertTrue(all(row["truth"] == "current" for row in before))
        self.assertEqual(sum(1 for row in before if row["collapsed"]), 1)
        ledger.ingest(
            {
                "document": {"title": "vex", "category": "csaf_vex"},
                "vulnerabilities": [{"cve": "OSV-1", "product_status": {"known_not_affected": ["prod-a"]}}],
                "time": 12,
            }
        )
        after = ledger.as_of(100, 10)
        self.assertTrue(after)
        self.assertTrue(all(row["truth"] == "disputed" for row in after))
        self.assertTrue(any(row["collapsed"] for row in after))

    def test_cites_audit_does_not_restore(self) -> None:
        ledger = EventLedger(policy="sec_audit", tenant_id="issuer")
        ledger.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.sec.filing",
                "id": "rev-1",
                "source": "issuer",
                "time": 1000,
                "data": {"statement": "revenue"},
            }
        )
        ledger.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.sec.8k",
                "id": "8k-1",
                "source": "issuer",
                "time": 2000,
                "data": {"item": "4.02", "supersedes": "rev-1"},
            }
        )
        ledger.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.audit.note",
                "id": "aud-1",
                "source": "auditor",
                "time": 3000,
                "data": {"CITES_AUDIT": "rev-1", "note": "seen"},
            }
        )
        rows = {row["claim_id"]: row for row in ledger.as_of(4000, 10)}
        self.assertEqual(rows["aud-1"]["truth"], "current")
        self.assertTrue(rows["aud-1"]["audit_only"])
        self.assertEqual(rows["rev-1"]["truth"], "withdrawn")
        self.assertNotIn("CITES_AUDIT", json.dumps(rows["aud-1"]["payload"]))

    def test_prov_cycle_is_withdrawn(self) -> None:
        ledger = EventLedger(policy="sec_audit", tenant_id="prov-tenant")
        ledger.ingest({"prov:entity": {"id": "seed", "base": True}, "time": 1})
        ledger.ingest(
            {
                "prov:entity": {"id": "cycle-a", "base": False},
                "wasDerivedFrom": {"generatedEntity": "cycle-a", "usedEntity": "cycle-b"},
                "time": 5,
            }
        )
        ledger.ingest(
            {
                "prov:entity": {"id": "cycle-b", "base": False},
                "wasDerivedFrom": {"generatedEntity": "cycle-b", "usedEntity": "cycle-a"},
                "time": 6,
            }
        )
        rows = {row["claim_id"]: row["truth"] for row in ledger.as_of(20, 10)}
        self.assertEqual(rows["seed"], "current")
        self.assertEqual(rows["cycle-a"], "withdrawn")
        self.assertEqual(rows["cycle-b"], "withdrawn")

    def test_gdpr_cdc_git_dpu_agent_and_ceiling(self) -> None:
        private = EventLedger(policy="gdpr_erasure", tenant_id="privacy")
        private.ingest({"op": "c", "table": "subjects", "pk": "1", "after": {"name": "alice-private"}, "time": 1})
        private.ingest({"op": "d", "table": "subjects", "pk": "1", "before": {"name": "alice-private"}, "time": 2})
        erased = json.dumps(private.as_of(10, 10))
        self.assertNotIn("alice-private", erased)
        self.assertNotIn("alice-private", private.non_nda_certificate())
        self.assertGreater(private.gdpr_redactions, 0)
        kept = EventLedger(policy="sec_audit", tenant_id="auditco")
        kept.ingest({"op": "c", "table": "subjects", "pk": "1", "after": {"name": "kept-name"}, "time": 1})
        kept.ingest({"op": "d", "table": "subjects", "pk": "1", "before": {"name": "kept-name"}, "time": 2})
        kept_rows = {row["claim_id"]: row for row in kept.as_of(10, 10)}
        self.assertEqual(kept_rows["cdc:subjects:1:c"]["truth"], "withdrawn")
        self.assertEqual(kept_rows["cdc:subjects:1:d"]["truth"], "current")
        self.assertIn("kept-name", json.dumps(kept_rows["cdc:subjects:1:d"]["payload"]))
        early = {row["claim_id"]: row["truth"] for row in kept.as_of(1, 1)}
        self.assertEqual(early["cdc:subjects:1:c"], "current")

        git = EventLedger(policy="sec_audit", tenant_id="source")
        git.ingest({"commit": "abcdef1234567890", "symbols": ["parse_header", "write_row"], "time": 1})
        git_rows = {row["claim_id"]: row for row in git.as_of(10, 5)}
        self.assertEqual(git_rows["git:abcdef1234567890"]["truth"], "current")
        self.assertEqual(git_rows["sym:parse_header"]["kind"], "git_symbol")
        self.assertEqual(git_rows["sym:parse_header"]["truth"], "current")

        fabric = EventLedger(policy="dpu_fabric", tenant_id="fabric")
        fabric.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.dpu.flow",
                "id": "flow-11",
                "source": "dpu",
                "time": 1,
                "data": {"register": 11},
            }
        )
        fabric.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.dpu.flow",
                "id": "flow-1",
                "source": "dpu",
                "time": 2,
                "data": {"register": 1},
            }
        )
        fabric_rows = {row["claim_id"]: row["truth"] for row in fabric.as_of(10, 5)}
        self.assertEqual(fabric_rows["flow-11"], "disputed")
        self.assertEqual(fabric_rows["flow-1"], "current")

        pool = EventLedger(policy="agent_pool", tenant_id="pool")
        pool.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.agent.assertion",
                "id": "pool-1",
                "source": "pool",
                "time": 1,
                "data": {"statement": "keep"},
            }
        )
        pool.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.agent.assertion",
                "id": "pool-2",
                "source": "pool",
                "time": 2,
                "data": {"supersedes": "pool-1"},
            }
        )
        pool_rows = {row["claim_id"]: row["truth"] for row in pool.as_of(10, 5)}
        self.assertEqual(pool_rows["pool-1"], "withdrawn")
        self.assertEqual(pool_rows["pool-2"], "current")

        ceiling = EventLedger(policy="sec_audit", tenant_id="issuer2")
        ceiling.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.sec.filing",
                "id": "stable",
                "source": "issuer2",
                "time": 10,
                "data": {"statement": "stands"},
            }
        )
        ceiling.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.sec.8k",
                "id": "weak",
                "source": "attacker",
                "time": 20,
                "authority": 10,
                "data": {"item": "4.02", "supersedes": "stable"},
            }
        )
        ceiling_rows = {row["claim_id"]: row["truth"] for row in ceiling.as_of(30, 5)}
        self.assertEqual(ceiling_rows["stable"], "current")
        self.assertGreater(ceiling.receipts[-1].integers[RECEIPT_SPEC.index("ceiling_blocks")], 0)
        with self.assertRaises(TenantIsolationError):
            ceiling.ingest(
                {
                    "specversion": "1.0",
                    "type": "com.chelated.sec.filing",
                    "id": "other",
                    "source": "x",
                    "tenant_id": "tenant-b",
                    "data": {},
                }
            )
        with self.assertRaises(EnvelopeError):
            ceiling.ingest({"hello": "world"})

    def test_receipts_are_integer_chain_and_ip_clean(self) -> None:
        ledger = EventLedger(policy="sec_audit", tenant_id="receipts")
        ledger.ingest(
            {
                "specversion": "1.0",
                "type": "com.chelated.sec.filing",
                "id": "one",
                "source": "receipts",
                "time": 5,
                "data": {"statement": "filed"},
            }
        )
        self.assertEqual(ledger.verify_chain(), 1)
        receipt = ledger.receipts[0]
        self.assertEqual(len(receipt.integers), 56)
        self.assertTrue(all(isinstance(number, int) for number in receipt.integers))
        self.assertEqual(ledger.llm_write_tokens, 0)
        report = ledger.non_nda_ip_report()
        self.assertEqual(report["ratio"], "0/56")
        self.assertEqual(report["dirty_fields"], 0)
        self.assertEqual(report["payload_leaks"], 0)
        self.assertEqual(scan_text_for_ip_leaks(ledger.non_nda_certificate()), [])
        tampered = (receipt.integers[0] + 1,) + receipt.integers[1:]
        object.__setattr__(receipt, "integers", tampered)
        with self.assertRaises(ReceiptVerificationError):
            ledger.verify_chain()

    def test_stream_holdback_and_capability_negotiation(self) -> None:
        async_source = _Tokens(["safe", "NO", "LEAK", "tail"])
        async_ledger = EventLedger(policy="sec_audit", tenant_id="stream-a")
        async_guard = wrap_stream(
            async_source,
            ledger=async_ledger,
            restricted_spans=[("NO", "LEAK")],
            profile="async_armed",
            tenant_id="stream-a",
        )
        self.assertEqual(next(async_guard), "safe")
        self.assertEqual(async_source.reads, 1)
        self.assertEqual(list(async_guard), ["tail"])
        self.assertGreaterEqual(async_guard.suppressed_spans, 1)
        self.assertEqual(async_ledger.llm_write_tokens, 0)
        self.assertNotIn("NO", json.dumps(async_ledger.as_of(100, 10)))
        self.assertGreaterEqual(async_ledger.verify_chain(), 1)

        strict_source = _Tokens(["safe", "NO", "LEAK", "tail"])
        strict_ledger = EventLedger(policy="sec_audit", tenant_id="stream-b")
        strict_guard = wrap_stream(
            strict_source,
            ledger=strict_ledger,
            restricted_spans=[("NO", "LEAK")],
            profile="strict_holdback",
            tenant_id="stream-b",
        )
        self.assertEqual(next(strict_guard), "safe")
        self.assertGreaterEqual(strict_source.reads, 3)
        self.assertEqual(list(strict_guard), ["tail"])

        foreign_ledger = EventLedger(policy="sec_audit", tenant_id="stream-a")
        foreign = wrap_stream(
            iter(["safe", "NO", "LEAK"]),
            ledger=foreign_ledger,
            restricted_spans=[{"tokens": ["safe"], "tenant_id": "stream-b"}, ("NO", "LEAK")],
            profile="async_armed",
            tenant_id="stream-a",
        )
        self.assertEqual(list(foreign), ["safe"])
        with self.assertRaises(StreamGuardError):
            wrap_stream(iter(["a"]), ledger=foreign_ledger, restricted_spans=[()], tenant_id="stream-a")
        other = EventLedger(policy="sec_audit", tenant_id="stream-b")
        with self.assertRaises(TenantIsolationError):
            wrap_stream(iter(["a"]), ledger=other, restricted_spans=[("a",)], tenant_id="stream-a")

        grant = negotiate_capability("event_ledger", tenant_id="tenant-a", limit=3)
        self.assertEqual(grant.name, "event_ledger")
        self.assertEqual(grant.tenant_scope, "tenant-a")
        for name in (
            "bare_metal_p4_tcam_asic_dataplane",
            "physical_weight_erasure_unlearning",
            "hdd_servo_mcu_firmware",
            "cross_tenant_raw_join",
        ):
            with self.assertRaises(negotiate_capability.UnsupportedCapabilityError) as ctx:
                negotiate_capability(name)
            self.assertEqual(ctx.exception.capability_name, name)
        with self.assertRaises(negotiate_capability.UnsupportedCapabilityError):
            negotiate_capability("not_a_real_capability")
        with self.assertRaises(negotiate_capability.UnsupportedCapabilityError):
            negotiate_capability("wrap_index", cross_tenant_raw_join=True)
        with self.assertRaises(ImportError) as blocked:
            importlib.import_module("chelated_kernel._hidden_channel")
        self.assertIn("private import blocked", str(blocked.exception))
        self.assertNotIn(CANARY_A, (PKG / "README.md").read_text(encoding="utf-8"))
        self.assertNotIn(CANARY_B, (PKG / "README.md").read_text(encoding="utf-8"))


class GateTests(unittest.TestCase):
    def test_coldstart_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["--out", tmp, "--repo-root", str(ROOT)])
            report = json.loads((Path(tmp) / "gate_report.json").read_text(encoding="utf-8"))
        failed = [row for row in report["session_records"] if not row["passed"]]
        summary = {
            "code": code,
            "gate_pass": report["gate_pass"],
            "loc": report["loc"],
            "wheel": report["wheel"],
            "license_problems": report["license_problems"],
            "ip_leaks": report["ip_leaks"],
            "ast_violations": report["ast_violations"],
            "canary": report["canary"],
            "private_import_blocked": report["private_import_blocked"],
            "repo_import_blocked": report["repo_import_blocked"],
            "sessions_passed": report["sessions_passed"],
            "per_family_passed": report["per_family_passed"],
            "cross_tenant_leaks": report["cross_tenant_leaks"],
            "unauthorized_reads": report["unauthorized_reads"],
            "failed": failed[:2],
        }
        self.assertEqual(code, 0, json.dumps(summary, indent=2)[:6000])
        self.assertTrue(report["gate_pass"])
        self.assertEqual(report["execution"], "deterministic-local-smoke")
        self.assertEqual(report["live_model_calls"], 0)
        self.assertEqual(report["sessions"], 36)
        self.assertGreaterEqual(report["sessions_passed"], 30)
        self.assertEqual(report["cross_tenant_leaks"], 0)
        self.assertEqual(report["unauthorized_reads"], 0)
        self.assertEqual(report["canary"]["accuracy"], 1.0)
        self.assertTrue(report["wheel"]["deterministic"])
        self.assertTrue(report["wheel"]["members_ok"])
        self.assertEqual(report["license_problems"], [])
        self.assertLess(report["loc"], 2500)


if __name__ == "__main__":
    unittest.main()
