"""Unit and integration coverage for the Slice 2 EGV evidence core."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

from egv.canonical import GENESIS_HASH, canonical_json, content_id, digest_for, failure_family_root
from egv.errors import (
    DependencyCycleError,
    IdempotencyConflictError,
    IntegrityError,
    LedgerBusyError,
    LedgerError,
    OptionalDependencyError,
    ProjectionError,
    PublicReplayError,
    PublicSchemaError,
    ReceiptConflictError,
    ReceiptVerificationError,
)
from egv.ipc import LedgerClient, LedgerWriterService
from egv.ledger import EvidenceLedger, INLINE_PAYLOAD_LIMIT, RETRACTED, STALE_DEPENDENT
from egv.projection import (
    InMemoryProjection,
    QdrantProjection,
    create_projection,
    isolated_collection_name,
    qdrant_available,
    qdrant_point_uuid,
)
from egv.public import (
    PublicCryptographicVerifier,
    PublicEventChain,
    PublicProjection,
    build_public_restore_receipt,
    load_public_projection,
    public_candidate_digest,
    public_dependency_id,
    public_dependency_set_digest,
    validate_public_candidate,
    verify_public_receipt,
)
from egv.receipts import ReceiptJournal, ReceiptSigner, receipt_hash, verify_receipt


class LedgerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="egv-test-")
        self.root = Path(self.tempdir.name)
        self.ledger = EvidenceLedger(
            self.root / "ledger.sqlite",
            blob_root=self.root / "private",
            clock=lambda: "2026-08-21T00:00:00Z",
        )

    def tearDown(self) -> None:
        self.ledger.close()
        self.tempdir.cleanup()

    def add_event(self, event_type: str, subject: str, value: object = 1) -> dict:
        return self.ledger.append_event(event_type, {"value": value}, subject_id=subject)

    def add_campaign_run_candidate(self, requested_authority: str = "NONE") -> str:
        self.ledger.create_campaign(
            "campaign-1",
            protocol_hash=digest_for("protocol"),
            source_commit="commit-1",
            model_revision="model-1",
            data_manifest_hash=digest_for("data"),
            evaluator_hash=digest_for("evaluator"),
            policy_hash=digest_for("policy"),
            seed_set=[1, 2],
            created_at="2026-08-21T00:00:00Z",
        )
        self.ledger.create_run(
            "run-1",
            campaign_id="campaign-1",
            arm="D",
            task_id="task-1",
            seed=1,
            parent_checkpoint=None,
            start_state="READY",
            host_role="spark_trainer",
            software_manifest_hash=digest_for("software"),
            created_at="2026-08-21T00:00:00Z",
        )
        self.ledger.append_candidate(
            "candidate-1",
            campaign_id="campaign-1",
            run_id="run-1",
            task_id="task-1",
            parent_candidate_id=None,
            mutation_family="PURE_FUNCTION",
            patch_hash=digest_for("patch"),
            requested_authority=requested_authority,
            prompt_hash=digest_for("prompt"),
            model_hash=digest_for("model"),
            adapter_hash=digest_for("adapter"),
        )
        return "candidate-1"

    def make_receipt(self, signer: ReceiptSigner, *, sequence: int, previous: str, decision: str = "PASS", idem: str = "idem-1") -> dict:
        return signer.sign_receipt(
            {
                "receipt_type": "VERDICT",
                "campaign_id": "campaign-1",
                "run_id": "run-1",
                "task_id": "task-1",
                "candidate_id": "candidate-1",
                "request_id": f"request-{idem}",
                "candidate_artifact_digest": digest_for("artifact"),
                "protocol_digest": digest_for("protocol"),
                "policy_digest": digest_for("policy"),
                "evaluator_digest": digest_for("evaluator"),
                "decision": decision,
                "diagnostic_enum": "PASS" if decision == "PASS" else "WRONG_OUTPUT",
                "resource_bucket": "UNDER_25",
                "exit_status_class": "SUCCESS",
            },
            sequence=sequence,
            previous_receipt_hash=previous,
            idempotency_key=idem,
        )


class TestCanonicalContentAddressing(unittest.TestCase):
    def test_canonical_json_and_hash_are_stable(self) -> None:
        self.assertEqual(canonical_json({"b": 2, "a": 1}), '{"a":1,"b":2}')
        self.assertEqual(digest_for({"a": 1, "b": 2}), digest_for({"b": 2, "a": 1}))
        self.assertNotEqual(content_id("evt", {"a": 1}), content_id("evt", {"a": 2}))

    def test_unsafe_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            canonical_json({"values": {1, 2}})
        with self.assertRaises(ValueError):
            canonical_json({"number": float("nan")})

    def test_failure_family_root_uses_ordered_public_tuple(self) -> None:
        self.assertEqual(
            failure_family_root("PURE_FUNCTION", "WRONG_OUTPUT", "module:function", "rule-1"),
            digest_for(["PURE_FUNCTION", "WRONG_OUTPUT", "module:function", "rule-1"]),
        )

    def test_internal_failure_root_is_incident_bound(self) -> None:
        first = failure_family_root("PURE_FUNCTION", "INTERNAL_ERROR", "module:function", "rule-1", infrastructure_incident_id="incident-a")
        second = failure_family_root("PURE_FUNCTION", "INTERNAL_ERROR", "module:function", "rule-1", infrastructure_incident_id="incident-b")
        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            failure_family_root("PURE_FUNCTION", "INTERNAL_ERROR", "module:function", "rule-1")


class TestAppendOnlyLedger(LedgerTestCase):
    def test_writer_lock_and_read_only_handle(self) -> None:
        second = None
        try:
            with self.assertRaises(LedgerBusyError):
                second = EvidenceLedger(self.root / "ledger.sqlite")
            readonly = EvidenceLedger(self.root / "ledger.sqlite", mode="read_only")
            try:
                with self.assertRaises(LedgerError):
                    readonly.append_event("NO", {})
                self.assertEqual(readonly.connection.execute("PRAGMA query_only").fetchone()[0], 1)
            finally:
                readonly.close()
        finally:
            if second is not None:
                second.close()

    def test_sqlite_triggers_reject_update_and_delete(self) -> None:
        event = self.add_event("OBSERVATION", "subject-1")
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.connection.execute("UPDATE events SET disposition='REJECTED' WHERE event_id=?", (event["event_id"],))
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.connection.execute("DELETE FROM events WHERE event_id=?", (event["event_id"],))

    def test_duplicate_append_is_idempotent_and_conflict_is_rejected(self) -> None:
        event = self.ledger.append_event("OBSERVATION", {"value": 1}, subject_id="subject-1", idempotency_key="one")
        duplicate = self.ledger.append_event("OBSERVATION", {"value": 1}, subject_id="subject-1", idempotency_key="one")
        self.assertEqual(event, duplicate)
        with self.assertRaises(IdempotencyConflictError):
            self.ledger.append_event("OBSERVATION", {"value": 2}, subject_id="subject-1", idempotency_key="one")
        self.assertEqual(self.ledger.verify_integrity()["event_count"], 1)

    def test_large_payload_uses_private_content_addressed_blob(self) -> None:
        payload = {"text": "x" * (INLINE_PAYLOAD_LIMIT + 1)}
        event = self.ledger.append_event("LARGE", payload, subject_id="large")
        self.assertEqual(event.get("payload"), payload)
        digest = event["blob_digest"]
        self.assertIsNotNone(digest)
        blob = self.root / "private" / "blobs" / "sha256" / digest[:2] / digest[2:4] / digest
        self.assertTrue(blob.exists())
        original_blob = blob.read_bytes()
        self.assertEqual(self.ledger.verify_integrity()["event_count"], 1)
        blob.unlink()
        with self.assertRaises(IntegrityError):
            self.ledger.verify_integrity()
        blob.write_bytes(b"tampered")
        with self.assertRaises(IntegrityError):
            self.ledger.verify_integrity()
        blob.write_bytes(original_blob)
        blob.chmod(0o444)
        self.assertEqual(blob.stat().st_mode & 0o777, 0o444)
        self.assertEqual(self.ledger.export_jsonl(), self.ledger.export_jsonl())

    def test_large_payload_replays_and_preserves_content_address(self) -> None:
        payload = {"text": "y" * (INLINE_PAYLOAD_LIMIT + 1)}
        event = self.ledger.append_event(
            "LARGE_TEXT",
            payload,
            subject_id="large-text",
            blob_media_type="text/plain",
        )
        replayed = EvidenceLedger.replay_jsonl(
            self.ledger.export_jsonl(),
            self.root / "large-replayed.sqlite",
            blob_root=self.root / "large-replayed-private",
        )
        try:
            replayed_event = replayed.events()[0]
            self.assertEqual(replayed_event["blob_digest"], event["blob_digest"])
            self.assertEqual(replayed_event["payload"], payload)
            self.assertEqual(replayed_event["blob_media_type"], "text/plain")
            self.assertEqual(self.ledger.export_jsonl(), replayed.export_jsonl())
        finally:
            replayed.close()

    def test_public_blob_export_is_allowlisted_and_excludes_private_roles(self) -> None:
        public_event = self.ledger.append_event(
            "PUBLIC_BLOB",
            {"text": "p" * (INLINE_PAYLOAD_LIMIT + 1)},
            subject_id="public-blob",
            blob_visibility="public-eligible",
            blob_role="public-metrics",
        )
        private_event = self.ledger.append_event(
            "PRIVATE_BLOB",
            {"text": "s" * (INLINE_PAYLOAD_LIMIT + 1)},
            subject_id="private-blob",
        )
        output = self.root / "public"
        manifest = self.ledger.export_public_blobs(output)
        self.assertEqual([entry["digest"] for entry in manifest], [public_event["blob_digest"]])
        self.assertTrue((output / "blobs" / "sha256" / public_event["blob_digest"][:2] / public_event["blob_digest"][2:4] / public_event["blob_digest"]).exists())
        self.assertFalse((output / "blobs" / "sha256" / private_event["blob_digest"][:2] / private_event["blob_digest"][2:4] / private_event["blob_digest"]).exists())

    def test_correction_retraction_and_dependency_staleness_are_derived(self) -> None:
        premise = self.add_event("PREMISE", "premise")
        child = self.add_event("DERIVED", "child")
        unrelated = self.add_event("UNRELATED", "unrelated")
        self.ledger.append_dependency(premise["event_id"], child["event_id"], edge_type="DEPENDS_ON")
        replacement = self.add_event("PREMISE", "replacement", 2)
        self.ledger.append_correction(
            premise["event_id"],
            replacement["event_id"],
            reason_code="EVALUATOR_RULE_CORRECTED",
            correction_source="FROZEN_EVALUATOR",
        )
        self.assertEqual(self.ledger.event_disposition(premise["event_id"]), RETRACTED)
        self.assertEqual(self.ledger.event_disposition(child["event_id"]), STALE_DEPENDENT)
        self.assertEqual(self.ledger.event_disposition(replacement["event_id"]), "OBSERVED")
        self.assertEqual(self.ledger.event_disposition(unrelated["event_id"]), "OBSERVED")
        self.assertIn(child["event_id"], self.ledger.stale_dependents(premise["event_id"]))
        self.assertNotIn(child["event_id"], {event["event_id"] for event in self.ledger.current_valid_events()})

    def test_retracting_promoted_candidate_by_candidate_id_closes_all_aliases(self) -> None:
        self.add_campaign_run_candidate()
        signer = ReceiptSigner(b"\x07" * 32)
        verdict_receipt = self.make_receipt(signer, sequence=1, previous=GENESIS_HASH, idem="alias-verdict")
        self.ledger.ingest_receipt(verdict_receipt, signer.public_key)
        self.ledger.append_verdict(
            "alias-verdict-row",
            candidate_id="candidate-1",
            correctness=True,
            performance={"score": 1},
            hidden_test_set_hash=digest_for("alias-hidden"),
            evaluator_revision="alias-evaluator",
            receipt_id=verdict_receipt["receipt_id"],
            signed_receipt_hash=receipt_hash(verdict_receipt),
        )
        candidate_event_id = self.ledger.connection.execute(
            "SELECT event_id FROM candidates WHERE candidate_id=?", ("candidate-1",)
        ).fetchone()[0]
        self.assertEqual(self.ledger.candidate_disposition("candidate-1"), "PROMOTED")
        self.ledger.append_retraction(
            "candidate-1",
            reason_code="PREMISE_RETRACTED",
            retraction_source="FROZEN_PROTOCOL",
        )
        self.assertEqual(self.ledger.event_disposition("candidate-1"), RETRACTED)
        self.assertEqual(self.ledger.event_disposition(candidate_event_id), RETRACTED)
        self.assertEqual(self.ledger.candidate_disposition("candidate-1"), RETRACTED)
        self.assertNotIn(candidate_event_id, {event["event_id"] for event in self.ledger.current_valid_events()})

    def test_non_read_candidate_requires_signed_effect_receipt(self) -> None:
        self.add_campaign_run_candidate(requested_authority="EXECUTE_CANDIDATE")
        signer = ReceiptSigner(b"\x08" * 32)
        authority = signer.sign_receipt(
            {
                "receipt_type": "AUTHORITY",
                "campaign_id": "campaign-1",
                "run_id": "run-1",
                "task_id": "task-1",
                "candidate_id": "candidate-1",
                "candidate_artifact_digest": digest_for("artifact"),
                "protocol_digest": digest_for("protocol"),
                "policy_digest": digest_for("policy"),
                "evaluator_digest": digest_for("evaluator"),
                "request_id": "alias-authority",
                "decision": "ALLOW",
            },
            sequence=1,
            idempotency_key="alias-authority",
        )
        verdict = self.make_receipt(signer, sequence=2, previous=receipt_hash(authority), idem="alias-verdict-effect")
        self.ledger.ingest_receipt(authority, signer.public_key)
        self.ledger.ingest_receipt(verdict, signer.public_key)
        self.ledger.append_verdict(
            "effect-required-verdict",
            candidate_id="candidate-1",
            correctness=True,
            performance={"score": 1},
            hidden_test_set_hash=digest_for("effect-required-hidden"),
            evaluator_revision="effect-required-evaluator",
            receipt_id=verdict["receipt_id"],
            signed_receipt_hash=receipt_hash(verdict),
        )
        self.assertEqual(self.ledger.candidate_disposition("candidate-1"), "ABSTAINED")
        effect = signer.sign_receipt(
            {
                "receipt_type": "EFFECT",
                "campaign_id": "campaign-1",
                "run_id": "run-1",
                "task_id": "task-1",
                "candidate_id": "candidate-1",
                "candidate_artifact_digest": digest_for("artifact"),
                "protocol_digest": digest_for("protocol"),
                "policy_digest": digest_for("policy"),
                "evaluator_digest": digest_for("evaluator"),
                "request_id": "effect-required",
                "decision": "ALLOW",
                "normalized_action_hash": digest_for("effect-action"),
                "sandbox_id": "effect-sandbox",
                "started_at": "2026-08-21T00:00:00Z",
                "finished_at": "2026-08-21T00:00:00Z",
                "exit_status_class": "SUCCESS",
                "output_digest": digest_for("effect-output"),
                "environment_diff_digest": digest_for("effect-environment"),
            },
            sequence=3,
            previous_receipt_hash=receipt_hash(verdict),
            idempotency_key="effect-required",
        )
        self.ledger.ingest_receipt(effect, signer.public_key)
        self.ledger.append_effect_receipt(
            "effect-required",
            candidate_id="candidate-1",
            identity="effect-evaluator",
            normalized_action_hash=effect["normalized_action_hash"],
            decision="ALLOW",
            policy_hash=effect["policy_digest"],
            sandbox_id=effect["sandbox_id"],
            started_at=effect["started_at"],
            finished_at=effect["finished_at"],
            exit_status_class=effect["exit_status_class"],
            output_hash=effect["output_digest"],
            environment_diff_hash=effect["environment_diff_digest"],
            signature=effect["signature"],
            receipt_id=effect["receipt_id"],
        )
        self.assertEqual(self.ledger.candidate_disposition("candidate-1"), "PROMOTED")

    def test_dependency_cycle_is_rejected(self) -> None:
        first = self.add_event("A", "a")
        second = self.add_event("B", "b")
        self.ledger.append_dependency(first["event_id"], second["event_id"], edge_type="DEPENDS_ON")
        with self.assertRaises(DependencyCycleError):
            self.ledger.append_dependency(second["event_id"], first["event_id"], edge_type="DEPENDS_ON")

    def test_export_replay_is_byte_for_byte_deterministic(self) -> None:
        self.add_campaign_run_candidate()
        self.ledger.add_checkpoint(
            "checkpoint-1",
            campaign_id="campaign-1",
            last_completed_phase="candidate-created",
            projection_generation=digest_for("projection"),
            artifact_manifest_hash=digest_for("artifacts"),
            created_at="2026-08-21T00:00:00Z",
        )
        exported = self.ledger.export_jsonl()
        replay_path = self.root / "replayed.sqlite"
        replayed = EvidenceLedger.replay_jsonl(exported, replay_path, blob_root=self.root / "replayed-private")
        try:
            self.assertEqual(exported, replayed.export_jsonl())
            self.assertEqual(self.ledger.ledger_head_hash(), replayed.ledger_head_hash())
            self.assertEqual(self.ledger.verify_integrity(), replayed.verify_integrity())
        finally:
            replayed.close()

    def test_checkpoint_idempotency_detects_conflicting_content(self) -> None:
        self.add_campaign_run_candidate()
        self.ledger.add_checkpoint(
            "checkpoint-conflict",
            campaign_id="campaign-1",
            last_completed_phase="phase-a",
            projection_generation=digest_for("projection-a"),
            artifact_manifest_hash=digest_for("artifacts-a"),
            created_at="2026-08-21T00:00:00Z",
        )
        with self.assertRaises(IdempotencyConflictError):
            self.ledger.add_checkpoint(
                "checkpoint-conflict",
                campaign_id="campaign-1",
                last_completed_phase="phase-b",
                projection_generation=digest_for("projection-a"),
                artifact_manifest_hash=digest_for("artifacts-a"),
                created_at="2026-08-21T00:00:00Z",
            )


class TestReceiptsAndJournal(LedgerTestCase):
    def test_private_receipts_reject_closed_enum_escape(self) -> None:
        self.add_campaign_run_candidate()
        signer = ReceiptSigner(b"\x06" * 32)
        with self.assertRaises(ReceiptVerificationError):
            signer.sign_receipt(
                {
                    "receipt_type": "VERDICT",
                    "campaign_id": "campaign-1",
                    "run_id": "run-1",
                    "task_id": "task-1",
                    "candidate_id": "candidate-1",
                    "request_id": "raw-status",
                    "decision": "ERROR",
                    "exit_status_class": "DOCKER_RUNTIME_FAILED",
                },
                sequence=1,
                previous_receipt_hash=GENESIS_HASH,
                idempotency_key="raw-status",
            )

    def test_internal_receipts_require_incident_and_bound_failure_root(self) -> None:
        signer = ReceiptSigner(b"\x0b" * 32)
        base = {
            "receipt_type": "VERDICT",
            "campaign_id": "campaign-1",
            "run_id": "run-1",
            "task_id": "task-1",
            "candidate_id": "candidate-1",
            "request_id": "internal-receipt",
            "decision": "ERROR",
            "diagnostic_enum": "INTERNAL_ERROR",
        }
        canonical = {
            "task_family": "PURE_FUNCTION",
            "normalized_public_locus": "module:function",
            "public_rule_id": "rule-public-1",
        }
        incident = "incident-1"
        exact_root = failure_family_root(
            canonical["task_family"],
            "INTERNAL_ERROR",
            canonical["normalized_public_locus"],
            canonical["public_rule_id"],
            infrastructure_incident_id=incident,
        )
        four_tuple_root = digest_for(
            [canonical["task_family"], "INTERNAL_ERROR", canonical["normalized_public_locus"], canonical["public_rule_id"]]
        )
        invalid_receipts = [
            {**base, **canonical, "infrastructure_incident_id": incident},
            {**base, **canonical, "failure_family_root": exact_root},
            {**base, **canonical, "infrastructure_incident_id": incident, "failure_family_root": four_tuple_root},
            {**base, **canonical, "infrastructure_incident_id": incident, "failure_family_root": digest_for("mismatch")},
        ]
        missing_canonical = {**base, **canonical, "infrastructure_incident_id": incident, "failure_family_root": exact_root}
        missing_canonical.pop("public_rule_id")
        invalid_receipts.append(missing_canonical)
        for index, invalid in enumerate(invalid_receipts):
            with self.assertRaises(ReceiptVerificationError):
                signer.sign_receipt(
                    invalid,
                    sequence=1,
                    previous_receipt_hash=GENESIS_HASH,
                    idempotency_key="internal-invalid-" + str(index),
                )
        valid = signer.sign_receipt(
            {
                **base,
                **canonical,
                "infrastructure_incident_id": incident,
                "failure_family_root": exact_root,
            },
            sequence=1,
            previous_receipt_hash=GENESIS_HASH,
            idempotency_key="internal-valid",
        )
        self.assertEqual(verify_receipt(valid, signer.public_key), receipt_hash(valid))
        noncanonical = dict(valid)
        tail_alias = {"A": "B", "Q": "R", "g": "h", "w": "x"}
        noncanonical["signature"] = (
            noncanonical["signature"][:-1] + tail_alias[noncanonical["signature"][-1]]
        )
        with self.assertRaisesRegex(ReceiptVerificationError, "canonical"):
            verify_receipt(noncanonical, signer.public_key)

    def test_public_internal_receipts_require_incident_and_bound_failure_root(self) -> None:
        signer = ReceiptSigner(b"\x0c" * 32)
        common = {
            "campaign_id": "campaign-public-internal",
            "run_id": "run-public-internal",
            "task_id": "task-public-internal",
            "candidate_id": "candidate-public-internal",
            "candidate_artifact_digest": digest_for("artifact"),
            "protocol_digest": digest_for("protocol"),
            "policy_digest": digest_for("policy"),
            "evaluator_digest": digest_for("evaluator"),
            "public_candidate_record_digest": digest_for("candidate-record"),
            "public_dependency_set_digest": digest_for("dependencies"),
            "receipt_type": "VERDICT",
            "request_id": "internal-public",
            "decision": "ERROR",
            "diagnostic_enum": "INTERNAL_ERROR",
            "task_family": "PURE_FUNCTION",
            "normalized_public_locus": "module:function",
            "public_rule_id": "rule-public-1",
        }
        incident = "incident-public"
        exact_root = failure_family_root(
            common["task_family"],
            "INTERNAL_ERROR",
            common["normalized_public_locus"],
            common["public_rule_id"],
            infrastructure_incident_id=incident,
        )
        four_tuple_root = digest_for(
            [common["task_family"], "INTERNAL_ERROR", common["normalized_public_locus"], common["public_rule_id"]]
        )
        invalid_receipts = [
            {**common, "infrastructure_incident_id": incident},
            {**common, "failure_family_root": exact_root},
            {**common, "infrastructure_incident_id": incident, "failure_family_root": four_tuple_root},
            {**common, "infrastructure_incident_id": incident, "failure_family_root": digest_for("mismatch")},
        ]
        missing_canonical = {**common, "infrastructure_incident_id": incident, "failure_family_root": exact_root}
        missing_canonical.pop("public_rule_id")
        invalid_receipts.append(missing_canonical)
        for invalid in invalid_receipts:
            with self.assertRaises(PublicSchemaError):
                signer.sign_public_receipt(
                    invalid,
                    public_sequence=1,
                )
        valid = signer.sign_public_receipt(
            {
                **common,
                "infrastructure_incident_id": incident,
                "failure_family_root": exact_root,
            },
            public_sequence=1,
        )
        self.assertTrue(valid["signature"])
        self.assertEqual(verify_public_receipt(valid, signer.public_key), digest_for(valid))
        forged = dict(valid, failure_family_root=four_tuple_root)
        with self.assertRaises(PublicSchemaError):
            verify_public_receipt(forged, signer.public_key)

    def test_ed25519_chain_journal_and_idempotent_ingest(self) -> None:
        self.add_campaign_run_candidate()
        signer = ReceiptSigner(b"\x02" * 32)
        first = self.make_receipt(signer, sequence=1, previous=GENESIS_HASH, idem="idem-1")
        second = self.make_receipt(signer, sequence=2, previous=receipt_hash(first), idem="idem-2")
        self.assertEqual(verify_receipt(first, signer.public_key), receipt_hash(first))
        journal = ReceiptJournal(self.root / "receipts.jsonl", signer.public_key)
        journal.append(first)
        journal.append(first)
        journal.append(second)
        self.assertEqual(journal.verify()["count"], 2)
        self.ledger.ingest_receipt(first, signer.public_key)
        self.ledger.ingest_receipt(first, signer.public_key)
        self.ledger.ingest_receipt(second, signer.public_key)
        self.assertEqual(self.ledger.verify_receipt_chain(signer.public_key)["receipt_count"], 2)
        self.assertEqual(self.ledger.status()["receipt_count"], 2)

    def test_conflicting_idempotency_and_bad_chain_are_quarantined(self) -> None:
        signer = ReceiptSigner(b"\x03" * 32)
        first = self.make_receipt(signer, sequence=1, previous=GENESIS_HASH, idem="idem-1")
        journal = ReceiptJournal(self.root / "receipts.jsonl", signer.public_key)
        journal.append(first)
        conflicting = signer.sign_receipt(
            {**{key: value for key, value in first.items() if key not in {"signature", "receipt_id", "sequence", "previous_receipt_hash", "signing_key_id", "schema_version", "idempotency_key"}}, "receipt_type": "VERDICT", "request_id": "different-request"},
            sequence=1,
            previous_receipt_hash=GENESIS_HASH,
            idempotency_key="idem-1",
        )
        with self.assertRaises(ReceiptConflictError):
            journal.append(conflicting)
        self.ledger.ingest_receipt(first, signer.public_key)
        bad = signer.sign_receipt(
            {"receipt_type": "VERDICT", "campaign_id": "campaign-1", "run_id": "run-1", "task_id": "task-1", "candidate_id": "candidate-1", "request_id": "bad", "candidate_artifact_digest": digest_for("artifact"), "protocol_digest": digest_for("protocol"), "policy_digest": digest_for("policy"), "evaluator_digest": digest_for("evaluator"), "decision": "PASS"},
            sequence=2,
            previous_receipt_hash=GENESIS_HASH,
            idempotency_key="bad-chain",
        )
        with self.assertRaises(ReceiptVerificationError):
            self.ledger.ingest_receipt(bad, signer.public_key)
        self.assertGreaterEqual(self.ledger.status()["quarantine_count"], 1)
        self.assertTrue(self.ledger.status()["quarantined"])

    def test_retracted_receipt_cannot_leave_candidate_promotable(self) -> None:
        self.add_campaign_run_candidate()
        signer = ReceiptSigner(b"\x05" * 32)
        verdict_receipt = self.make_receipt(signer, sequence=1, previous=GENESIS_HASH)
        self.ledger.ingest_receipt(verdict_receipt, signer.public_key)
        self.ledger.append_verdict(
            "verdict-retracted",
            candidate_id="candidate-1",
            correctness=True,
            performance={"score": 1},
            hidden_test_set_hash=digest_for("hidden"),
            evaluator_revision="evaluator-1",
            receipt_id=verdict_receipt["receipt_id"],
            signed_receipt_hash=receipt_hash(verdict_receipt),
        )
        self.assertEqual(self.ledger.candidate_disposition("candidate-1"), "PROMOTED")
        self.ledger.append_retraction(
            verdict_receipt["receipt_id"],
            reason_code="INVALID_RECEIPT",
            retraction_source="FROZEN_PROTOCOL",
        )
        self.assertEqual(self.ledger.candidate_disposition("candidate-1"), STALE_DEPENDENT)

    def test_mixed_evaluator_key_is_quarantined_before_chain_commit(self) -> None:
        self.add_campaign_run_candidate()
        first_signer = ReceiptSigner(b"\x09" * 32)
        second_signer = ReceiptSigner(b"\x0a" * 32)
        first = self.make_receipt(first_signer, sequence=1, previous=GENESIS_HASH, idem="key-one")
        second = self.make_receipt(second_signer, sequence=2, previous=receipt_hash(first), idem="key-two")
        self.ledger.ingest_receipt(first, first_signer.public_key)
        with self.assertRaises(ReceiptVerificationError):
            self.ledger.ingest_receipt(second, second_signer.public_key)
        self.assertEqual(self.ledger.status()["receipt_count"], 1)
        self.assertEqual(self.ledger.verify_receipt_chain(first_signer.public_key)["receipt_count"], 1)
        self.assertGreaterEqual(self.ledger.status()["quarantine_count"], 1)


class PublicFixture:
    def __init__(self, *, stale: bool = False, include_effect: bool = True) -> None:
        self.signer = ReceiptSigner(b"\x04" * 32)
        self.campaign_id = "campaign-public"
        self.run_id = "run-public"
        self.task_id = "task-public"
        self.protocol_digest = digest_for("public-protocol")
        self.evaluator_digest = digest_for("public-evaluator")
        self.policy_digest = digest_for("public-policy")
        self.artifact_digest = digest_for("public-artifact")
        self.candidate_id = "candidate-public"
        self.candidate = {
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "arm": "D",
            "attempt_index": 1,
            "candidate_id": self.candidate_id,
            "parent_candidate_id": None,
            "candidate_artifact_digest": self.artifact_digest,
            "model_digest": digest_for("public-model"),
            "adapter_digest": digest_for("public-adapter"),
            "prompt_template_digests": [digest_for("public-template")],
            "mutation_family": "PURE_FUNCTION",
            "normalized_public_locus": "module:function",
            "requested_authority": "EXECUTE_CANDIDATE",
            "declared_public_evidence_ids": [],
            "public_dependency_ids": [],
        }
        self.projection = PublicProjection()
        self.projection.add_candidate(self.candidate)
        self.candidate_digest = public_candidate_digest(self.candidate)
        self.dependency_digest = public_dependency_set_digest([])
        common = {
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "candidate_artifact_digest": self.artifact_digest,
            "protocol_digest": self.protocol_digest,
            "policy_digest": self.policy_digest,
            "evaluator_digest": self.evaluator_digest,
            "public_candidate_record_digest": self.candidate_digest,
            "public_dependency_set_digest": self.dependency_digest,
        }
        authority = self.signer.sign_public_receipt({**common, "receipt_type": "AUTHORITY", "request_id": "authority-public", "decision": "ALLOW"}, public_sequence=1)
        verdict = self.signer.sign_public_receipt(
            {**common, "receipt_type": "VERDICT", "request_id": "verdict-public", "decision": "PASS", "diagnostic_enum": "PASS", "resource_bucket": "UNDER_25", "exit_status_class": "SUCCESS"},
            public_sequence=2,
            previous_public_receipt_digest=receipt_hash(authority),
        )
        effect = None
        if include_effect:
            effect = self.signer.sign_public_receipt(
                {
                    **common,
                    "receipt_type": "EFFECT",
                    "request_id": "effect-public",
                    "decision": "ALLOW",
                },
                public_sequence=3,
                previous_public_receipt_digest=receipt_hash(verdict),
            )
        self.projection.add_receipt(authority)
        self.projection.add_receipt(verdict)
        if effect is not None:
            self.projection.add_receipt(effect)
        restore = build_public_restore_receipt(
            {
                "campaign_id": self.campaign_id,
                "logical_service_set_id": "svcset-public",
                "logical_service_ids": ["svc-001"],
                "private_inventory_digest": digest_for("inventory"),
                "service_definition_set_digest": digest_for("definitions"),
                "model_set_digest": digest_for("models"),
                "configuration_set_digest": digest_for("configuration"),
                "executable_or_image_set_digest": digest_for("executables"),
                "expected_service_count": 1,
                "restored_service_count": 1,
                "health_check_count": 1,
                "health_pass_count": 1,
                "all_health_checks_passed": True,
                "smoke_input_digest": digest_for("smoke-input"),
                "smoke_output_digest": digest_for("smoke-output"),
                "smoke_matches_baseline": True,
                "restoration_outcome": "RESTORED",
            },
            self.signer,
        )
        self.projection.set_restore_receipt(restore)
        chain = PublicEventChain(self.signer, campaign_id=self.campaign_id, protocol_digest=self.protocol_digest, evaluator_digest=self.evaluator_digest)
        if stale:
            premise_candidate = {
                **self.candidate,
                "candidate_id": "candidate-premise",
                "candidate_artifact_digest": digest_for("public-premise-artifact"),
                "requested_authority": "NONE",
                "public_dependency_ids": [],
            }
            premise_digest = public_candidate_digest(premise_candidate)
            premise_authority = self.signer.sign_public_receipt(
                {
                    "campaign_id": self.campaign_id,
                    "run_id": self.run_id,
                    "task_id": self.task_id,
                    "request_id": "authority-premise",
                    "candidate_id": premise_candidate["candidate_id"],
                    "candidate_artifact_digest": premise_candidate["candidate_artifact_digest"],
                    "protocol_digest": self.protocol_digest,
                    "policy_digest": self.policy_digest,
                    "evaluator_digest": self.evaluator_digest,
                    "public_candidate_record_digest": premise_digest,
                    "public_dependency_set_digest": public_dependency_set_digest([]),
                    "receipt_type": "AUTHORITY",
                    "decision": "ALLOW",
                },
                public_sequence=1,
            )
            premise = {
                "parent_id": premise_candidate["candidate_id"],
                "child_id": self.candidate_id,
                "edge_type": "EVIDENCE_USE",
                "insertion_receipt_id": premise_authority["receipt_id"],
            }
            self.candidate["public_dependency_ids"] = [public_dependency_id(premise)]
            self.projection = PublicProjection()
            self.projection.add_candidate(premise_candidate)
            self.projection.add_candidate(self.candidate)
            self.projection.add_dependency(premise)
            self.candidate_digest = public_candidate_digest(self.candidate)
            common["public_candidate_record_digest"] = self.candidate_digest
            common["public_dependency_set_digest"] = public_dependency_set_digest([premise])
            authority = self.signer.sign_public_receipt({**common, "receipt_type": "AUTHORITY", "request_id": "authority-public", "decision": "ALLOW"}, public_sequence=2, previous_public_receipt_digest=receipt_hash(premise_authority))
            verdict = self.signer.sign_public_receipt({**common, "receipt_type": "VERDICT", "request_id": "verdict-public", "decision": "PASS", "diagnostic_enum": "PASS", "resource_bucket": "UNDER_25", "exit_status_class": "SUCCESS"}, public_sequence=3, previous_public_receipt_digest=receipt_hash(authority))
            effect = None
            if include_effect:
                effect = self.signer.sign_public_receipt(
                    {
                        **common,
                        "receipt_type": "EFFECT",
                        "request_id": "effect-public",
                        "decision": "ALLOW",
                    },
                    public_sequence=4,
                    previous_public_receipt_digest=receipt_hash(verdict),
                )
            self.projection.add_receipt(premise_authority)
            self.projection.add_receipt(authority)
            self.projection.add_receipt(verdict)
            if effect is not None:
                self.projection.add_receipt(effect)
            chain = PublicEventChain(self.signer, campaign_id=self.campaign_id, protocol_digest=self.protocol_digest, evaluator_digest=self.evaluator_digest)
            chain.append_retraction(run_id=self.run_id, task_id=self.task_id, subject_id=premise_candidate["candidate_id"], reason_code="PREMISE_RETRACTED", effective_after_attempt=1, authorizing_public_receipt_id=authority["receipt_id"])
            chain.append_recorded_disposition(run_id=self.run_id, task_id=self.task_id, candidate_id=premise_candidate["candidate_id"], disposition="STALE_DEPENDENT", candidate_digest=premise_digest, dependency_digest=public_dependency_set_digest([]))
            chain.append_recorded_disposition(run_id=self.run_id, task_id=self.task_id, candidate_id=self.candidate_id, disposition="STALE_DEPENDENT", candidate_digest=self.candidate_digest, dependency_digest=public_dependency_set_digest([premise]))
            self.projection.events = chain.events  # type: ignore[misc]
            self.projection.restore_receipt = restore
            seal = chain.seal(run_id=self.run_id, task_id=self.task_id, receipt_head_digest=receipt_hash(effect or verdict), candidate_records=[premise_candidate, self.candidate], dependency_records=[premise], restore_receipt=restore)
            self.projection.events.append(seal)
        else:
            chain.append_recorded_disposition(
                run_id=self.run_id,
                task_id=self.task_id,
                candidate_id=self.candidate_id,
                disposition="PROMOTED" if include_effect else "ABSTAINED",
                candidate_digest=self.candidate_digest,
                dependency_digest=self.dependency_digest,
            )
            self.projection.add_event(chain.events[-1])
            self.projection.add_event(chain.seal(run_id=self.run_id, task_id=self.task_id, receipt_head_digest=receipt_hash(effect or verdict), candidate_records=[self.candidate], dependency_records=[], restore_receipt=restore))


class TestClosedPublicReplay(unittest.TestCase):
    def test_public_replay_computes_signed_promotion(self) -> None:
        fixture = PublicFixture()
        report = PublicCryptographicVerifier(fixture.signer.public_key, protocol_digest=fixture.protocol_digest, evaluator_digest=fixture.evaluator_digest).verify(
            candidates=fixture.projection.candidates.values(),
            dependencies=fixture.projection.dependencies.values(),
            receipts=fixture.projection.receipts.values(),
            events=fixture.projection.events,
            restore_receipt=fixture.projection.restore_receipt,
        )
        self.assertTrue(report.valid)
        self.assertEqual(report.decisions[fixture.candidate_id], "PROMOTED")
        self.assertEqual(report.to_dict()["replay_type"], "public cryptographic decision replay")

    def test_public_schema_is_closed_and_terminal_seal_is_required(self) -> None:
        fixture = PublicFixture()
        extra = dict(fixture.candidate)
        extra["disposition"] = "PROMOTED"
        with self.assertRaises(PublicSchemaError):
            validate_public_candidate(extra)
        with self.assertRaises(PublicReplayError):
            PublicCryptographicVerifier(fixture.signer.public_key, protocol_digest=fixture.protocol_digest, evaluator_digest=fixture.evaluator_digest).verify(
                candidates=fixture.projection.candidates.values(),
                dependencies=fixture.projection.dependencies.values(),
                receipts=fixture.projection.receipts.values(),
                events=fixture.projection.events[:-1],
                restore_receipt=fixture.projection.restore_receipt,
            )

    def test_public_replay_rejects_recorded_disposition_mismatch(self) -> None:
        fixture = PublicFixture()
        event = dict(fixture.projection.events[0])
        event["recorded_disposition"] = "REJECTED"
        # The signature is intentionally not repaired: a public verifier must
        # reject the tampered signed record before trusting it.
        fixture.projection.events[0] = event
        with self.assertRaises((PublicReplayError, ReceiptVerificationError, PublicSchemaError)):
            PublicCryptographicVerifier(fixture.signer.public_key, protocol_digest=fixture.protocol_digest, evaluator_digest=fixture.evaluator_digest).verify(
                candidates=fixture.projection.candidates.values(),
                dependencies=fixture.projection.dependencies.values(),
                receipts=fixture.projection.receipts.values(),
                events=fixture.projection.events,
                restore_receipt=fixture.projection.restore_receipt,
            )

    def test_public_replay_derives_stale_dependency_from_correction(self) -> None:
        fixture = PublicFixture(stale=True)
        report = PublicCryptographicVerifier(fixture.signer.public_key, protocol_digest=fixture.protocol_digest, evaluator_digest=fixture.evaluator_digest).verify(
            candidates=fixture.projection.candidates.values(),
            dependencies=fixture.projection.dependencies.values(),
            receipts=fixture.projection.receipts.values(),
            events=fixture.projection.events,
            restore_receipt=fixture.projection.restore_receipt,
        )
        self.assertEqual(report.decisions[fixture.candidate_id], "STALE_DEPENDENT")

    def test_public_stale_traversal_only_follows_ancestors(self) -> None:
        dependencies = [
            {"parent_id": "parent", "child_id": "child", "edge_type": "DEPENDS_ON"},
            {"parent_id": "parent", "child_id": "sibling", "edge_type": "DEPENDS_ON"},
        ]
        parent = {"candidate_id": "parent", "public_dependency_ids": []}
        child = {"candidate_id": "child", "public_dependency_ids": []}
        sibling = {"candidate_id": "sibling", "public_dependency_ids": []}
        self.assertFalse(PublicCryptographicVerifier._candidate_stale("parent", parent, dependencies, {"child"}))
        self.assertTrue(PublicCryptographicVerifier._candidate_stale("child", child, dependencies, {"child"}))
        self.assertFalse(PublicCryptographicVerifier._candidate_stale("sibling", sibling, dependencies, {"child"}))
        self.assertTrue(PublicCryptographicVerifier._candidate_stale("child", child, dependencies, {"parent"}))

    def test_public_non_read_candidate_requires_effect_receipt(self) -> None:
        fixture = PublicFixture(include_effect=False)
        report = PublicCryptographicVerifier(
            fixture.signer.public_key,
            protocol_digest=fixture.protocol_digest,
            evaluator_digest=fixture.evaluator_digest,
        ).verify(
            candidates=fixture.projection.candidates.values(),
            dependencies=fixture.projection.dependencies.values(),
            receipts=fixture.projection.receipts.values(),
            events=fixture.projection.events,
            restore_receipt=fixture.projection.restore_receipt,
        )
        self.assertEqual(report.decisions[fixture.candidate_id], "ABSTAINED")

        fixture = PublicFixture()
        candidate = fixture.candidate
        receipts_without_effect = [
            receipt
            for receipt in fixture.projection.receipts.values()
            if receipt["receipt_type"] != "EFFECT"
        ]
        without_effect = PublicCryptographicVerifier._compute_dispositions(
            [candidate], [], receipts_without_effect, set()
        )
        with_effect = PublicCryptographicVerifier._compute_dispositions(
            [candidate], [], list(fixture.projection.receipts.values()), set()
        )
        self.assertEqual(without_effect[candidate["candidate_id"]], "ABSTAINED")
        self.assertEqual(with_effect[candidate["candidate_id"]], "PROMOTED")

    def test_public_pseudonymous_ids_reject_topology_values(self) -> None:
        fixture = PublicFixture()
        restore_payload = {
            key: value
            for key, value in fixture.projection.restore_receipt.items()
            if key not in {"schema_version", "receipt_id", "signing_key_id", "signature"}
        }
        restore_payload["logical_service_set_id"] = "svc-set-x7q2"
        restore_payload["logical_service_ids"] = ["svc-x7q2"]
        signed_restore = build_public_restore_receipt(restore_payload, fixture.signer)
        self.assertEqual(signed_restore["logical_service_ids"], ["svc-x7q2"])

        topology_values = [
            "127.0.0.1",
            "2001:db8::1",
            "example.invalid",
            "localhost:8080",
            "https://example.invalid",
            "/srv/egv",
            "5432",
            "host-01",
            "qdrant-prod",
            "spark-worker-1",
            "service-001",
        ]
        for topology_value in topology_values:
            with self.subTest(topology_value=topology_value):
                invalid_restore = dict(restore_payload)
                invalid_restore["logical_service_ids"] = [topology_value]
                with self.assertRaises(PublicSchemaError):
                    build_public_restore_receipt(invalid_restore, fixture.signer)

        invalid_candidate = dict(fixture.candidate)
        invalid_candidate["candidate_id"] = "127.0.0.1"
        with self.assertRaises(PublicSchemaError):
            validate_public_candidate(invalid_candidate)

        invalid_candidate = dict(fixture.candidate)
        invalid_candidate["public_dependency_ids"] = ["5432"]
        with self.assertRaises(PublicSchemaError):
            validate_public_candidate(invalid_candidate)

    def test_public_projection_jsonl_round_trip_is_deterministic(self) -> None:
        fixture = PublicFixture()
        fixture.projection.set_public_key(fixture.signer.public_key_pem)
        first_dir = Path(tempfile.mkdtemp(prefix="egv-public-one-"))
        second_dir = Path(tempfile.mkdtemp(prefix="egv-public-two-"))
        try:
            fixture.projection.export(first_dir)
            loaded = load_public_projection(first_dir)
            loaded.set_public_key(fixture.signer.public_key_pem)
            loaded.export(second_dir)
            public_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in second_dir.rglob("*")
                if path.is_file() and path.name != "evaluator-public-key.pem"
            )
            self.assertNotIn("egv-dual-smoke-auth-token", public_text)
            self.assertNotIn("sqlite", public_text.lower())
            for relative in (
                "ledger/public-candidates.jsonl",
                "ledger/public-dependencies.jsonl",
                "ledger/public-events.jsonl",
                "receipts/public-signed-envelopes.jsonl",
                "restore/public-restore-receipt.json",
                "projection-manifest.json",
            ):
                self.assertEqual((first_dir / relative).read_bytes(), (second_dir / relative).read_bytes())
        finally:
            import shutil

            shutil.rmtree(first_dir)
            shutil.rmtree(second_dir)


class TestProjectionAndIPC(LedgerTestCase):
    def test_memory_projection_rebuild_is_deterministic_and_validates_hits(self) -> None:
        first = self.add_event("OBSERVATION", "one", "alpha")
        projection = InMemoryProjection(collection_name=isolated_collection_name("campaign", "D", "run"))
        manifest_one = projection.rebuild(self.ledger, lambda payload: [float(len(str(payload["value"]))), 1.0], embedding_model_revision="embed-v1")
        point = next(iter(projection.points.values()))
        self.assertTrue(projection.validate_hit(self.ledger, point))
        manifest_two = projection.rebuild(self.ledger, lambda payload: [float(len(str(payload["value"]))), 1.0], embedding_model_revision="embed-v1")
        self.assertEqual(manifest_one, manifest_two)
        self.assertEqual(point.payload["source_event_id"], first["event_id"])

    def test_projection_reserves_failure_root_validation_for_internal_errors(self) -> None:
        common = {
            "value": "failure-root",
            "task_family": "PURE_FUNCTION",
            "diagnostic_enum": "WRONG_OUTPUT",
            "normalized_public_locus": "module:solve",
            "public_rule_id": "rule-projection",
        }
        exact = failure_family_root(
            common["task_family"],
            common["diagnostic_enum"],
            common["normalized_public_locus"],
            common["public_rule_id"],
        )
        case_number = [0]

        def rebuild(payload: dict) -> dict:
            case_number[0] += 1
            ledger = EvidenceLedger(
                self.root / ("projection-root-{}.sqlite".format(case_number[0])),
                blob_root=self.root / ("projection-root-blobs-{}".format(case_number[0])),
            )
            try:
                ledger.append_event("VERDICT", payload, subject_id="projection-root")
                projection = InMemoryProjection(collection_name="projection-root")
                return projection.rebuild(
                    ledger,
                    lambda value: [1.0, 0.0],
                    embedding_model_revision="embed-v1",
                )
            finally:
                ledger.close()

        ordinary_failure = rebuild(common)
        self.assertEqual(ordinary_failure["point_count"], 1)

        with self.assertRaises(ProjectionError):
            rebuild({**common, "failure_family_root": exact})
        with self.assertRaises(ProjectionError):
            rebuild({**common, "failure_family_root": None})
        with self.assertRaises(ProjectionError):
            rebuild({**common, "infrastructure_incident_id": "incident-not-internal"})
        with self.assertRaises(ProjectionError):
            rebuild({**common, "diagnostic_enum": "INTERNAL_ERROR"})

        valid = rebuild(
            {
                **common,
                "diagnostic_enum": "INTERNAL_ERROR",
                "infrastructure_incident_id": "incident-projection-root",
                "failure_family_root": failure_family_root(
                    common["task_family"],
                    "INTERNAL_ERROR",
                    common["normalized_public_locus"],
                    common["public_rule_id"],
                    infrastructure_incident_id="incident-projection-root",
                ),
            }
        )
        self.assertEqual(valid["point_count"], 1)

        with self.assertRaises(ProjectionError):
            rebuild({**common, "failure_family_root": digest_for("forged-root")})
        with self.assertRaises(ProjectionError):
            rebuild({key: value for key, value in {**common, "failure_family_root": exact}.items() if key != "public_rule_id"})
        incident = "incident-projection-root"
        internal_root = failure_family_root(
            common["task_family"],
            "INTERNAL_ERROR",
            common["normalized_public_locus"],
            common["public_rule_id"],
            infrastructure_incident_id=incident,
        )
        with self.assertRaises(ProjectionError):
            rebuild(
                {
                    **common,
                    "diagnostic_enum": "INTERNAL_ERROR",
                    "infrastructure_incident_id": incident,
                }
            )
        with self.assertRaises(ProjectionError):
            rebuild(
                {
                    **common,
                    "diagnostic_enum": "INTERNAL_ERROR",
                    "infrastructure_incident_id": incident,
                    "failure_family_root": digest_for(
                        [
                            common["task_family"],
                            "INTERNAL_ERROR",
                            common["normalized_public_locus"],
                            common["public_rule_id"],
                        ]
                    ),
                }
            )
        root_for_a = failure_family_root(
            common["task_family"],
            "INTERNAL_ERROR",
            common["normalized_public_locus"],
            common["public_rule_id"],
            infrastructure_incident_id="incident-a",
        )
        with self.assertRaises(ProjectionError):
            rebuild(
                {
                    **common,
                    "diagnostic_enum": "INTERNAL_ERROR",
                    "infrastructure_incident_id": "incident-b",
                    "failure_family_root": root_for_a,
                }
            )
        internal = rebuild(
            {
                **common,
                "diagnostic_enum": "INTERNAL_ERROR",
                "infrastructure_incident_id": incident,
                "failure_family_root": internal_root,
            }
        )
        self.assertEqual(internal["point_count"], 1)

    def test_ingest_receipt_nested_payload_is_rebuilt_and_root_checked(self) -> None:
        self.add_campaign_run_candidate()
        signer = ReceiptSigner(b"\x2a" * 32)
        incident = "incident-nested-projection"
        root = failure_family_root(
            "PURE_FUNCTION",
            "INTERNAL_ERROR",
            "module:solve",
            "rule-projection",
            infrastructure_incident_id=incident,
        )
        receipt = signer.sign_receipt(
            {
                "receipt_type": "VERDICT",
                "campaign_id": "campaign-1",
                "run_id": "run-1",
                "task_id": "task-1",
                "candidate_id": "candidate-1",
                "request_id": "request-nested-projection",
                "candidate_artifact_digest": digest_for("artifact"),
                "protocol_digest": digest_for("protocol"),
                "policy_digest": digest_for("policy"),
                "evaluator_digest": digest_for("evaluator"),
                "task_family": "PURE_FUNCTION",
                "diagnostic_enum": "INTERNAL_ERROR",
                "normalized_public_locus": "module:solve",
                "public_rule_id": "rule-projection",
                "infrastructure_incident_id": incident,
                "failure_family_root": root,
                "decision": "ERROR",
                "resource_bucket": "UNDER_25",
                "exit_status_class": "INFRASTRUCTURE_LOSS",
            },
            sequence=1,
            previous_receipt_hash=GENESIS_HASH,
            idempotency_key="nested-projection",
        )
        event = self.ledger.ingest_receipt(receipt, signer.public_key)
        projection = InMemoryProjection(collection_name="nested-receipt-projection")
        manifest_one = projection.rebuild(
            self.ledger,
            lambda payload: [1.0, float(len(payload.get("diagnostic_enum", "")))],
            embedding_model_revision="embed-v1",
            campaign_id="campaign-1",
        )
        manifest_two = projection.rebuild(
            self.ledger,
            lambda payload: [1.0, float(len(payload.get("diagnostic_enum", "")))],
            embedding_model_revision="embed-v1",
            campaign_id="campaign-1",
        )
        self.assertEqual(manifest_one, manifest_two)
        receipt_points = [
            point for point in projection.points.values() if point.payload["source_event_id"] == event["event_id"]
        ]
        self.assertEqual(len(receipt_points), 1)
        self.assertEqual(receipt_points[0].payload["failure_family_root"], root)

        dummy_wrapper_event = self.ledger.append_event(
            "RECEIPT",
            {"receipt": receipt, "receipt_hash": digest_for("wrapper"), "failure_family_root": digest_for("dummy")},
            campaign_id="campaign-1",
            run_id="run-1",
            task_id="task-1",
            subject_id="candidate-1",
        )
        rebuilt = projection.rebuild(
            self.ledger,
            lambda payload: [1.0, float(len(payload.get("diagnostic_enum", "")))],
            embedding_model_revision="embed-v1",
            campaign_id="campaign-1",
        )
        self.assertEqual(rebuilt["point_count"], manifest_one["point_count"] + 1)
        dummy_point_id = content_id(
            "point",
            {
                "event_id": dummy_wrapper_event["event_id"],
                "payload_hash": dummy_wrapper_event["payload_hash"],
                "embedding_model_revision": "embed-v1",
            },
        )
        self.assertEqual(projection.get(dummy_point_id).payload["failure_family_root"], root)

        forged_nested = dict(receipt, failure_family_root=digest_for("dummy-nested-root"))
        self.ledger.append_event(
            "RECEIPT",
            {"receipt": forged_nested, "receipt_hash": digest_for("forged-wrapper")},
            campaign_id="campaign-1",
            run_id="run-1",
            task_id="task-1",
            subject_id="candidate-1",
        )
        with self.assertRaises(ProjectionError):
            projection.rebuild(
                self.ledger,
                lambda payload: [1.0, float(len(payload.get("diagnostic_enum", "")))],
                embedding_model_revision="embed-v1",
                campaign_id="campaign-1",
            )

    def test_qdrant_is_explicitly_optional(self) -> None:
        if qdrant_available():
            self.skipTest("environment supplies qdrant-client; optional-missing branch is not applicable")
        with self.assertRaises(OptionalDependencyError):
            create_projection(backend="qdrant", collection_name="egv-test", location=":memory:")

    def test_qdrant_location_dispatch_is_explicit(self) -> None:
        class FakeQdrantClient:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        with patch(
            "egv.projection._require_qdrant",
            return_value=(FakeQdrantClient, object(), lambda: True),
        ):
            cases = (
                (":memory:", {"location": ":memory:"}),
                ("http://qdrant.example.invalid:6333", {"url": "http://qdrant.example.invalid:6333"}),
                ("https://qdrant.example.invalid", {"url": "https://qdrant.example.invalid"}),
                (self.root / "qdrant-local", {"path": str(self.root / "qdrant-local")}),
            )
            for location, expected in cases:
                with self.subTest(location=location):
                    projection = QdrantProjection(location=location, collection_name="egv-dispatch")
                    self.assertEqual(projection.client.kwargs, expected)
            with self.assertRaises(ProjectionError):
                QdrantProjection(location="ftp://qdrant.example.invalid", collection_name="egv-dispatch")

    @unittest.skipUnless(qdrant_available(), "qdrant-client is not installed")
    def test_qdrant_rebuild_uses_deterministic_uuid_and_exact_replacement(self) -> None:
        event = self.add_event("OBSERVATION", "qdrant-subject", "alpha")
        projection = create_projection(
            backend="qdrant",
            collection_name=isolated_collection_name("campaign", "D", "qdrant-run"),
            location=":memory:",
        )

        def embedding(payload: dict) -> list[float]:
            return [float(len(str(payload["value"]))), 1.0]

        try:
            manifest_one = projection.rebuild(
                self.ledger,
                embedding,
                embedding_model_revision="embed-v1",
            )
            self.assertEqual(manifest_one["point_count"], 1)
            self.assertEqual(
                self.ledger.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                1,
            )
            first_rows, _ = projection.client.scroll(
                collection_name=projection.collection_name,
                limit=10,
                with_payload=True,
                with_vectors=True,
            )
            self.assertEqual(len(first_rows), 1)
            first_hit = first_rows[0]
            canonical_id = first_hit.payload["canonical_point_id"]
            self.assertEqual(canonical_id, manifest_one["point_ids"][0])
            self.assertEqual(str(uuid.UUID(str(first_hit.id))), str(first_hit.id))
            self.assertEqual(str(first_hit.id), qdrant_point_uuid(canonical_id))
            self.assertEqual(first_hit.payload["source_event_id"], event["event_id"])
            self.assertEqual(first_hit.payload["source_payload_hash"], event["payload_hash"])
            self.assertTrue(projection.validate_hit(self.ledger, first_hit))

            models = projection._models_import()
            projection.client.upsert(
                collection_name=projection.collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=[0.0, 1.0],
                        payload={"rogue": True},
                    )
                ],
                wait=True,
            )
            self.assertEqual(projection.client.count(projection.collection_name, exact=True).count, 2)

            manifest_two = projection.rebuild(
                self.ledger,
                embedding,
                embedding_model_revision="embed-v1",
            )
            self.assertEqual(manifest_two, manifest_one)
            self.assertEqual(projection.client.count(projection.collection_name, exact=True).count, 1)
            second_rows, _ = projection.client.scroll(
                collection_name=projection.collection_name,
                limit=10,
                with_payload=True,
                with_vectors=True,
            )
            self.assertEqual([str(row.id) for row in second_rows], [str(first_hit.id)])
            self.assertNotIn("rogue", second_rows[0].payload)
            self.assertTrue(projection.validate_hit(self.ledger, second_rows[0]))

            wrong_id_hit = type("Hit", (), {"id": str(uuid.uuid4()), "payload": second_rows[0].payload})()
            self.assertFalse(projection.validate_hit(self.ledger, wrong_id_hit))
            wrong_event_payload = dict(second_rows[0].payload)
            wrong_event_payload["source_payload_hash"] = digest_for("tampered")
            wrong_event_hit = type("Hit", (), {"id": second_rows[0].id, "payload": wrong_event_payload})()
            self.assertFalse(projection.validate_hit(self.ledger, wrong_event_hit))
        finally:
            projection.delete()

    @unittest.skipUnless(qdrant_available(), "qdrant-client is not installed")
    def test_qdrant_persistent_path_rebuild_and_reopen_is_deterministic(self) -> None:
        event = self.add_event("OBSERVATION", "qdrant-persistent-subject", "persistent")
        location = self.root / "qdrant-local"
        collection_name = isolated_collection_name("campaign", "D", "qdrant-persistent-run")
        first = create_projection(
            backend="qdrant",
            collection_name=collection_name,
            location=location,
        )
        reopened = None
        first_closed = False

        def embedding(payload: dict) -> list[float]:
            return [float(len(str(payload["value"]))), 1.0]

        try:
            manifest_one = first.rebuild(
                self.ledger,
                embedding,
                embedding_model_revision="embed-v1",
            )
            self.assertEqual(manifest_one["point_count"], 1)
            first_rows, _ = first.client.scroll(
                collection_name=collection_name,
                limit=10,
                with_payload=True,
                with_vectors=True,
            )
            self.assertEqual(len(first_rows), 1)
            first_hit = first_rows[0]
            self.assertEqual(first_hit.payload["canonical_point_id"], manifest_one["point_ids"][0])
            self.assertEqual(first_hit.payload["source_event_id"], event["event_id"])
            self.assertEqual(first_hit.payload["source_payload_hash"], event["payload_hash"])
            self.assertTrue(first.validate_hit(self.ledger, first_hit))
            first.client.close()
            first_closed = True

            reopened = create_projection(
                backend="qdrant",
                collection_name=collection_name,
                location=location,
            )
            self.assertEqual(reopened.client.count(collection_name, exact=True).count, 1)
            persisted_rows, _ = reopened.client.scroll(
                collection_name=collection_name,
                limit=10,
                with_payload=True,
                with_vectors=True,
            )
            self.assertEqual([str(row.id) for row in persisted_rows], [str(first_hit.id)])
            self.assertEqual(persisted_rows[0].payload["source_event_id"], event["event_id"])

            models = reopened._models_import()
            reopened.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=[0.0, 1.0],
                        payload={"rogue": True},
                    )
                ],
                wait=True,
            )
            self.assertEqual(reopened.client.count(collection_name, exact=True).count, 2)
            manifest_two = reopened.rebuild(
                self.ledger,
                embedding,
                embedding_model_revision="embed-v1",
            )
            self.assertEqual(manifest_two, manifest_one)
            self.assertEqual(reopened.client.count(collection_name, exact=True).count, 1)
            second_rows, _ = reopened.client.scroll(
                collection_name=collection_name,
                limit=10,
                with_payload=True,
                with_vectors=True,
            )
            self.assertEqual([str(row.id) for row in second_rows], [str(first_hit.id)])
            self.assertEqual(second_rows[0].payload["canonical_point_id"], manifest_one["point_ids"][0])
            self.assertEqual(second_rows[0].payload["source_event_id"], event["event_id"])
            self.assertEqual(second_rows[0].payload["source_payload_hash"], event["payload_hash"])
            self.assertTrue(reopened.validate_hit(self.ledger, second_rows[0]))
        finally:
            if reopened is not None:
                reopened.delete()
                reopened.client.close()
            elif not first_closed:
                first.delete()

    def test_authenticated_single_writer_ipc(self) -> None:
        socket_path = self.root / "writer.sock"
        with LedgerWriterService(self.ledger, socket_path, "token", evaluator_auth_token="evaluator-token"):
            client = LedgerClient(socket_path, "token")
            event = client.append_event(event_type="IPC", payload={"value": 1}, subject_id="ipc")
            self.assertEqual(event["event_type"], "IPC")
            with self.assertRaises(LedgerError):
                LedgerClient(socket_path, "wrong").append_event(event_type="IPC", payload={"value": 2}, subject_id="ipc2")
            with self.assertRaises(LedgerError):
                LedgerClient(socket_path, "evaluator-token", role="evaluator").append_event(
                    event_type="IPC_FORBIDDEN", payload={"value": 3}, subject_id="ipc3"
                )


class TestCLISmoke(unittest.TestCase):
    def test_cli_smoke_is_real_and_repeatable(self) -> None:
        command = [sys.executable, "-m", "egv", "smoke", "--json"]
        first = subprocess.run(command, check=True, capture_output=True, text=True)
        second = subprocess.run(command, check=True, capture_output=True, text=True)
        first_payload = json.loads(first.stdout)
        second_payload = json.loads(second.stdout)
        self.assertEqual(first_payload, second_payload)
        self.assertEqual(first_payload["smoke"], "PASS")
        self.assertEqual(first_payload["runtime_tier"], "floor")
        self.assertFalse(first_payload["qdrant_exercised"])
        self.assertFalse(first_payload["campaign_path_exercised"])
        self.assertTrue(first_payload["public_replay"]["terminal_seal_verified"])
        self.assertEqual(first_payload["ledger"]["event_count"], first_payload["ledger"]["replay"]["event_count"])

    def test_cli_two_process_cpu_smoke_proves_writer_boundary(self) -> None:
        command = [sys.executable, "-m", "egv", "smoke", "--json", "--two-process"]
        payload = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual(payload["two_process"]["mode"], "two-process-cpu-only")
        self.assertFalse(payload["two_process"]["evaluator_used_sqlite"])
        self.assertEqual(payload["two_process"]["candidate_disposition"], "PROMOTED")
        self.assertEqual(payload["two_process"]["trainer_process_exit"], 0)
        self.assertEqual(payload["two_process"]["evaluator_process_exit"], 0)
        self.assertEqual(payload["two_process"]["evaluator_sqlite_connect_calls"], 0)
        self.assertFalse(payload["two_process"]["evaluator_used_sqlite"])
        self.assertEqual(payload["two_process"]["evaluator_ipc_methods"], ["ingest_receipt"])
        self.assertTrue(payload["two_process"]["evaluator_non_receipt_ipc_rejected"])

    def test_cli_verify_public_projection_path(self) -> None:
        fixture = PublicFixture()
        fixture.projection.set_public_key(fixture.signer.public_key_pem)
        with tempfile.TemporaryDirectory(prefix="egv-cli-public-") as directory:
            projection_path = Path(directory) / "projection"
            fixture.projection.export(projection_path)
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "egv",
                    "verify-public",
                    "--projection",
                    str(projection_path),
                    "--public-key",
                    str(projection_path / "receipts" / "evaluator-public-key.pem"),
                    "--protocol-digest",
                    fixture.protocol_digest,
                    "--evaluator-digest",
                    fixture.evaluator_digest,
                    "--json",
                ],
                text=True,
            )
        payload = json.loads(output)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["replay_type"], "public cryptographic decision replay")


if __name__ == "__main__":
    unittest.main()
