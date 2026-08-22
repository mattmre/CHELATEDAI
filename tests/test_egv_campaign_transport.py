from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from egv.canonical import GENESIS_HASH, canonical_bytes, digest_bytes, digest_for
from egv.campaign.authority import (
    EvaluatorAuthority,
    EvaluatorAuthorityError,
    EvaluatorReceipt,
    sign_evaluator_receipt,
)
from egv.campaign.coordinator import (
    PHASE_EVIDENCE,
    CampaignCoordinator,
    CampaignCoordinatorError,
    CampaignGate,
    PHASE_EVIDENCE_SCHEMA,
)
from egv.campaign.state import CampaignState, CampaignStateStore
from egv.campaign.transport import (
    ArtifactDescriptor,
    ArtifactStagingStore,
    ArtifactTransportError,
    TransferManifest,
)
from egv.receipts import ReceiptSigner


def _evidence_payload(campaign_id: str, phase: str, role: str, subject: str) -> bytes:
    return canonical_bytes({
        "schema_version": PHASE_EVIDENCE_SCHEMA,
        "campaign_id": campaign_id,
        "phase": phase,
        "role": role,
        "subject_digest": digest_for(subject),
        "status": "PASS",
    })


def _manifest(payloads=None, artifact_roles=None, evidence_phase="P1", **overrides) -> tuple:
    artifact_roles = artifact_roles or {
        "adapter": "PREFLIGHT_REPORT",
        "report": "RUNTIME_COMPATIBILITY",
    }
    campaign_id = overrides.get("campaign_id", "campaign-public")
    if payloads is None:
        payloads = {
            artifact_id: _evidence_payload(campaign_id, evidence_phase, role, artifact_id)
            for artifact_id, role in artifact_roles.items()
        }
    if set(artifact_roles) != set(payloads):
        raise ValueError("test artifact role map must exactly match payloads")
    artifacts = []
    for artifact_id in sorted(payloads):
        payload = payloads[artifact_id]
        artifacts.append({
            "artifact_id": artifact_id,
            "role": artifact_roles[artifact_id],
            "media_type": "application/octet-stream",
            "byte_count": len(payload),
            "sha256": digest_bytes(payload),
        })
    value = {
        "schema_version": "egv-campaign-transfer-manifest-v1",
        "campaign_id": campaign_id,
        "transfer_id": "transfer-001",
        "source_role": "EVALUATOR",
        "destination_role": "TRAINER",
        "artifacts": artifacts,
    }
    value.update(overrides)
    return TransferManifest.from_mapping(value), payloads


def _receipt(signer, manifest, **overrides) -> EvaluatorReceipt:
    fields = {
        "campaign_id": manifest.campaign_id,
        "phase": "P1",
        "transfer_digest": manifest.digest,
        "artifact_set_digest": manifest.artifact_set_digest,
        "protocol_digest": digest_for("protocol"),
        "evaluator_digest": digest_for("evaluator-image"),
        "decision": "ACCEPT",
        "metrics_digest": digest_for("metrics"),
        "sequence": 1,
        "previous_receipt_digest": GENESIS_HASH,
    }
    fields.update(overrides)
    return sign_evaluator_receipt(fields, signer)


class CampaignTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.staging = ArtifactStagingStore(self.root / "staging")
        self.manifest, self.payloads = _manifest()
        self.signer = ReceiptSigner.generate()
        self.authority = EvaluatorAuthority(
            self.signer.public_key,
            campaign_id="campaign-public",
            protocol_digest=digest_for("protocol"),
            evaluator_digest=digest_for("evaluator-image"),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_stage_and_verify_are_content_addressed_and_idempotent(self) -> None:
        first = self.staging.stage(self.manifest, self.payloads)
        second = self.staging.stage(self.manifest, self.payloads)
        self.assertEqual(first, self.manifest.digest)
        self.assertEqual(second, first)
        self.assertEqual(self.staging.verify(self.manifest), first)

    def test_manifest_rejects_traversal_duplicate_and_same_side_transfer(self) -> None:
        raw = self.manifest.to_dict()
        raw["artifacts"][0]["artifact_id"] = "../escape"
        with self.assertRaisesRegex(ArtifactTransportError, "unsafe"):
            TransferManifest.from_mapping(raw)
        raw = self.manifest.to_dict()
        raw["artifacts"][1]["sha256"] = raw["artifacts"][0]["sha256"]
        with self.assertRaisesRegex(ArtifactTransportError, "duplicated"):
            TransferManifest.from_mapping(raw)
        with self.assertRaisesRegex(ArtifactTransportError, "authority boundary"):
            _manifest(destination_role="EVALUATOR")

    def test_direct_constructor_cannot_bypass_transport_or_receipt_validation(self) -> None:
        with self.assertRaisesRegex(ArtifactTransportError, "unsafe"):
            ArtifactDescriptor("../escape", "ADAPTER", "application/octet-stream", 1, digest_for("x"))
        with self.assertRaisesRegex(ArtifactTransportError, "authority boundary"):
            TransferManifest(
                "campaign-public", "transfer-unsafe", "TRAINER", "TRAINER", self.manifest.artifacts
            )
        raw = _receipt(self.signer, self.manifest).to_dict()
        raw["sequence"] = 0
        with self.assertRaisesRegex(EvaluatorAuthorityError, "positive"):
            EvaluatorReceipt(**{key: raw[key] for key in raw})

    def test_missing_extra_or_substituted_payload_fails_before_publish(self) -> None:
        for payloads in (
            {"adapter": self.payloads["adapter"]},
            {**self.payloads, "extra": b"x"},
            {**self.payloads, "adapter": b"substitution"},
        ):
            with self.assertRaises(ArtifactTransportError):
                self.staging.stage(self.manifest, payloads)
        self.assertFalse((self.root / "staging" / self.manifest.digest).exists())

    def test_post_stage_tamper_symlink_and_unexpected_file_fail(self) -> None:
        self.staging.stage(self.manifest, self.payloads)
        root = self.root / "staging" / self.manifest.digest
        adapter = self.manifest.artifacts[0]
        blob = root / "blobs" / (adapter.sha256 + ".blob")
        original = blob.read_bytes()
        blob.write_bytes(b"tampered")
        with self.assertRaisesRegex(ArtifactTransportError, "truncated or substituted"):
            self.staging.verify(self.manifest)
        blob.write_bytes(original)
        (root / "unexpected.txt").write_text("smuggled", encoding="utf-8")
        with self.assertRaisesRegex(ArtifactTransportError, "unexpected"):
            self.staging.verify(self.manifest)

    def test_noncanonical_manifest_file_and_manifest_substitution_fail(self) -> None:
        self.staging.stage(self.manifest, self.payloads)
        root = self.root / "staging" / self.manifest.digest
        (root / "manifest.json").write_text(json.dumps(self.manifest.to_dict(), indent=2), encoding="utf-8")
        with self.assertRaisesRegex(ArtifactTransportError, "canonical"):
            self.staging.verify(self.manifest)
        with self.assertRaisesRegex(ArtifactTransportError, "substitution"):
            self.staging.verify(self.manifest, expected_digest=digest_for("wrong"))

    def test_authority_accepts_only_exact_signed_pinned_receipt(self) -> None:
        receipt = _receipt(self.signer, self.manifest)
        verified = self.authority.verify(
            receipt,
            phase="P1",
            transfer_digest=self.manifest.digest,
            artifact_set_digest=self.manifest.artifact_set_digest,
            expected_sequence=1,
            expected_previous_digest=None,
        )
        self.assertEqual(verified, receipt.digest)

    def test_authority_rejects_wrong_key_protocol_evaluator_chain_and_signature(self) -> None:
        wrong_signer = ReceiptSigner.generate()
        cases = [
            (_receipt(wrong_signer, self.manifest), self.authority),
            (_receipt(self.signer, self.manifest, protocol_digest=digest_for("wrong")), self.authority),
            (_receipt(self.signer, self.manifest, evaluator_digest=digest_for("wrong")), self.authority),
            (_receipt(self.signer, self.manifest, previous_receipt_digest=digest_for("wrong")), self.authority),
        ]
        for receipt, authority in cases:
            with self.assertRaises(EvaluatorAuthorityError):
                authority.verify(
                    receipt,
                    phase="P1",
                    transfer_digest=self.manifest.digest,
                    artifact_set_digest=self.manifest.artifact_set_digest,
                    expected_sequence=1,
                    expected_previous_digest=None,
                )
        raw = _receipt(self.signer, self.manifest).to_dict()
        raw["signature"] = ("A" if raw["signature"][0] != "A" else "B") + raw["signature"][1:]
        tampered = EvaluatorReceipt.from_mapping(raw)
        with self.assertRaisesRegex(EvaluatorAuthorityError, "signature"):
            self.authority.verify(
                tampered, phase="P1", transfer_digest=self.manifest.digest,
                artifact_set_digest=self.manifest.artifact_set_digest,
                expected_sequence=1, expected_previous_digest=None,
            )
        noncanonical = _receipt(self.signer, self.manifest).to_dict()
        tail_alias = {"A": "B", "Q": "R", "g": "h", "w": "x"}
        noncanonical["signature"] = (
            noncanonical["signature"][:-1] + tail_alias[noncanonical["signature"][-1]]
        )
        with self.assertRaisesRegex(EvaluatorAuthorityError, "canonical"):
            EvaluatorReceipt.from_mapping(noncanonical)

    def _coordinator(self):
        store = CampaignStateStore(self.root / "state.json")
        state = store.initialize(campaign_id="campaign-public")
        self.staging.stage(self.manifest, self.payloads)
        return CampaignCoordinator(store, self.staging, self.authority), store, state

    def test_coordinator_commits_verified_accept_and_exact_retry_is_idempotent(self) -> None:
        coordinator, store, state = self._coordinator()
        gate = CampaignGate("P0", "P1", self.manifest, _receipt(self.signer, self.manifest))
        committed = coordinator.apply(gate, expected_state_digest=state.digest)
        retried = coordinator.apply(gate, expected_state_digest=state.digest)
        self.assertEqual(retried.digest, committed.digest)
        self.assertEqual(store.load().sequence, 1)

    def test_coordinator_requires_contiguous_signed_receipt_chain(self) -> None:
        coordinator, store, state = self._coordinator()
        first_gate = CampaignGate("P0", "P1", self.manifest, _receipt(self.signer, self.manifest))
        first = coordinator.apply(first_gate, expected_state_digest=state.digest)
        second_manifest, second_payloads = _manifest(
            artifact_roles={
                "inventory": "PROTECTED_INVENTORY_ATTESTATION",
                "snapshot": "RESTORE_SNAPSHOT",
            },
            evidence_phase="P2",
            transfer_id="transfer-002",
        )
        self.staging.stage(second_manifest, second_payloads)
        second_receipt = _receipt(
            self.signer,
            second_manifest,
            phase="P2",
            sequence=2,
            previous_receipt_digest=first_gate.receipt.digest,
        )
        second = coordinator.apply(
            CampaignGate("P1", "P2", second_manifest, second_receipt),
            expected_state_digest=first.digest,
        )
        self.assertEqual((second.phase, second.sequence), ("P2", 2))
        replay = _receipt(self.signer, second_manifest, phase="P2", sequence=2)
        with self.assertRaises(EvaluatorAuthorityError):
            self.authority.verify(
                replay,
                phase="P2",
                transfer_digest=second_manifest.digest,
                artifact_set_digest=second_manifest.artifact_set_digest,
                expected_sequence=2,
                expected_previous_digest=first_gate.receipt.digest,
            )

    def test_reject_stale_state_replay_and_rejected_decision_do_not_advance(self) -> None:
        coordinator, store, state = self._coordinator()
        rejected = CampaignGate(
            "P0", "P1", self.manifest, _receipt(self.signer, self.manifest, decision="REJECT")
        )
        with self.assertRaisesRegex(CampaignCoordinatorError, "rejected"):
            coordinator.apply(rejected, expected_state_digest=state.digest)
        self.assertEqual(store.load().phase, "P0")
        with self.assertRaisesRegex(CampaignCoordinatorError, "stale"):
            coordinator.apply(
                CampaignGate("P0", "P1", self.manifest, _receipt(self.signer, self.manifest)),
                expected_state_digest=digest_for("stale"),
            )
        self.assertEqual(store.load().phase, "P0")

    def test_artifact_substitution_and_wrong_campaign_execute_no_state_change(self) -> None:
        coordinator, store, state = self._coordinator()
        blob = self.root / "staging" / self.manifest.digest / "blobs" / (
            self.manifest.artifacts[0].sha256 + ".blob"
        )
        blob.write_bytes(b"bad")
        with self.assertRaises(ArtifactTransportError):
            coordinator.apply(
                CampaignGate("P0", "P1", self.manifest, _receipt(self.signer, self.manifest)),
                expected_state_digest=state.digest,
            )
        self.assertEqual(store.load().digest, state.digest)
        other, payloads = _manifest(campaign_id="other-campaign", transfer_id="transfer-other")
        self.staging.stage(other, payloads)
        receipt = _receipt(self.signer, other, campaign_id="other-campaign")
        with self.assertRaisesRegex(CampaignCoordinatorError, "identity"):
            coordinator.apply(CampaignGate("P0", "P1", other, receipt), expected_state_digest=state.digest)

    def test_every_phase_requires_its_exact_evidence_roles_and_direction(self) -> None:
        for phase_index in range(1, 14):
            to_phase = "P{}".format(phase_index)
            with self.subTest(phase=to_phase):
                phase_root = self.root / ("phase-{}".format(phase_index))
                staging = ArtifactStagingStore(phase_root / "staging")
                store = CampaignStateStore(phase_root / "state.json")
                state = store.initialize(campaign_id="campaign-public")
                for index in range(1, phase_index):
                    state = store.advance("P{}".format(index), expected_digest=state.digest)
                    if index == 3:
                        state = store.require_restore(expected_digest=state.digest, service_state="STOPPED")
                if phase_index == 13:
                    state = CampaignState(
                        campaign_id=state.campaign_id,
                        phase=state.phase,
                        sequence=state.sequence + 1,
                        previous_state_digest=state.digest,
                        restore_required=False,
                        service_state="RESTORED",
                        inventory_digest=state.inventory_digest,
                        last_receipt_digest=digest_for("verified-restore-receipt-fixture"),
                    )
                    store._atomic_write(state)
                source, destination, roles = PHASE_EVIDENCE[to_phase]
                payloads = {
                    "artifact-{:03d}".format(index): _evidence_payload(
                        "campaign-public", to_phase, role, "subject-{}".format(role)
                    )
                    for index, role in enumerate(sorted(roles), start=1)
                }
                role_map = {
                    artifact_id: role
                    for artifact_id, role in zip(sorted(payloads), sorted(roles))
                }
                manifest, payloads = _manifest(
                    payloads,
                    artifact_roles=role_map,
                    transfer_id="transfer-{}".format(to_phase.lower()),
                    source_role=source,
                    destination_role=destination,
                )
                staging.stage(manifest, payloads)
                receipt = _receipt(
                    self.signer,
                    manifest,
                    phase=to_phase,
                    sequence=state.sequence + 1,
                    previous_receipt_digest=state.last_receipt_digest or GENESIS_HASH,
                )
                coordinator = CampaignCoordinator(store, staging, self.authority)
                advanced = coordinator.apply(
                    CampaignGate(state.phase, to_phase, manifest, receipt),
                    expected_state_digest=state.digest,
                )
                self.assertEqual(advanced.phase, to_phase)

    def test_well_signed_arbitrary_artifacts_cannot_advance_phase(self) -> None:
        coordinator, store, state = self._coordinator()
        payloads = {"generic": b"not-preflight-evidence"}
        arbitrary, payloads = _manifest(
            payloads,
            artifact_roles={"generic": "GENERIC_SIGNED_ARTIFACT"},
            transfer_id="transfer-arbitrary",
        )
        self.staging.stage(arbitrary, payloads)
        signed = _receipt(self.signer, arbitrary)
        with self.assertRaisesRegex(CampaignCoordinatorError, "evidence contract"):
            coordinator.apply(
                CampaignGate("P0", "P1", arbitrary, signed),
                expected_state_digest=state.digest,
            )
        self.assertEqual(store.load().digest, state.digest)

    def test_correct_role_labels_with_noncanonical_or_failed_evidence_cannot_advance(self) -> None:
        for mode in ("garbage", "failed"):
            with self.subTest(mode=mode):
                root = self.root / mode
                staging = ArtifactStagingStore(root / "staging")
                store = CampaignStateStore(root / "state.json")
                state = store.initialize(campaign_id="campaign-public")
                roles = {"preflight": "PREFLIGHT_REPORT", "runtime": "RUNTIME_COMPATIBILITY"}
                payloads = {
                    artifact_id: _evidence_payload("campaign-public", "P1", role, artifact_id)
                    for artifact_id, role in roles.items()
                }
                if mode == "garbage":
                    payloads["preflight"] = b"signed but not evidence"
                else:
                    failed = json.loads(payloads["preflight"].decode("utf-8"))
                    failed["status"] = "FAIL"
                    payloads["preflight"] = canonical_bytes(failed)
                manifest, payloads = _manifest(
                    payloads,
                    artifact_roles=roles,
                    transfer_id="transfer-{}".format(mode),
                )
                staging.stage(manifest, payloads)
                gate = CampaignGate("P0", "P1", manifest, _receipt(self.signer, manifest))
                with self.assertRaisesRegex(CampaignCoordinatorError, "evidence"):
                    CampaignCoordinator(store, staging, self.authority).apply(
                        gate,
                        expected_state_digest=state.digest,
                    )
                self.assertEqual(store.load().digest, state.digest)


if __name__ == "__main__":
    unittest.main()
