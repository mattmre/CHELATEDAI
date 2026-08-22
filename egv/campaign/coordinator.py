"""Resumable, exact-once Campaign phase coordination."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Optional

from ..canonical import canonical_bytes, validate_sha256
from .authority import EvaluatorAuthority, EvaluatorReceipt
from .errors import CampaignError
from .state import CampaignState, CampaignStateStore, PHASES
from .transport import ArtifactStagingStore, TransferManifest


class CampaignCoordinatorError(CampaignError):
    """A resumable campaign transition cannot be proven safe."""


# Exact semantic evidence contract for every destination phase. A cryptographic
# signature authenticates an evaluator; it does not turn an arbitrary blob into
# the evidence required by a scientific or lifecycle gate.
PHASE_EVIDENCE = {
    "P1": ("EVALUATOR", "TRAINER", frozenset({"PREFLIGHT_REPORT", "RUNTIME_COMPATIBILITY"})),
    "P2": ("EVALUATOR", "TRAINER", frozenset({"PROTECTED_INVENTORY_ATTESTATION", "RESTORE_SNAPSHOT"})),
    "P3": ("EVALUATOR", "TRAINER", frozenset({"RESTORE_INTENT", "STOP_VERIFICATION"})),
    "P4": (
        "TRAINER", "EVALUATOR",
        frozenset({"DATA_MANIFEST", "MODEL_MANIFEST", "SCAN_REPORT", "SOFTWARE_MANIFEST", "SOURCE_MANIFEST"}),
    ),
    "P5": ("EVALUATOR", "TRAINER", frozenset({"AUTHORITY_NEGATIVE_CONTROLS", "RECEIPT_CHAIN_HEAD"})),
    "P6": ("TRAINER", "EVALUATOR", frozenset({"ADVERSARIAL_REVIEW", "PROTOCOL_MANIFEST"})),
    "P7": ("TRAINER", "EVALUATOR", frozenset({"LINEAGE_REPORT", "REPLAY_REPORT", "TRAJECTORY_MANIFEST"})),
    "P8": (
        "TRAINER", "EVALUATOR",
        frozenset({"BASE_IMMUTABILITY_REPORT", "SEALED_ADAPTER", "TRAINING_REPORT"}),
    ),
    "P9": ("EVALUATOR", "TRAINER", frozenset({"ABLATION_RESULTS", "EVALUATOR_VERDICTS"})),
    "P10": ("EVALUATOR", "TRAINER", frozenset({"DISPOSITION_REPORT", "GATE_VECTOR", "REPLAY_REPORT"})),
    "P11": ("TRAINER", "EVALUATOR", frozenset({"PRIVATE_PACKAGE_MANIFEST", "PRIVATE_SCAN_REPORT"})),
    "P12": ("EVALUATOR", "TRAINER", frozenset({"PUBLIC_RESTORE_RECEIPT", "RESTORE_SNAPSHOT"})),
    "P13": (
        "EVALUATOR", "TRAINER",
        frozenset({"PUBLIC_BUNDLE_MANIFEST", "PUBLIC_SCAN_REPORT", "TERMINAL_SEAL"}),
    ),
}
PHASE_EVIDENCE_SCHEMA = "egv-campaign-phase-evidence-v1"
_PHASE_EVIDENCE_FIELDS = frozenset(
    {"schema_version", "campaign_id", "phase", "role", "subject_digest", "status"}
)
_MAX_PHASE_EVIDENCE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class CampaignGate:
    from_phase: str
    to_phase: str
    transfer: TransferManifest
    receipt: EvaluatorReceipt

    def __post_init__(self) -> None:
        if self.from_phase not in PHASES or self.to_phase not in PHASES:
            raise CampaignCoordinatorError("campaign gate phase is outside P0-P13")
        if PHASES.index(self.to_phase) != PHASES.index(self.from_phase) + 1:
            raise CampaignCoordinatorError("campaign gate must represent one monotonic phase")
        if type(self.transfer) is not TransferManifest or type(self.receipt) is not EvaluatorReceipt:
            raise CampaignCoordinatorError("campaign gate requires exact validated evidence types")


class CampaignCoordinator:
    """Commit a phase only after staged bytes and evaluator authority verify."""

    def __init__(
        self,
        state_store: CampaignStateStore,
        staging: ArtifactStagingStore,
        authority: EvaluatorAuthority,
    ) -> None:
        if type(state_store) is not CampaignStateStore:
            raise CampaignCoordinatorError("coordinator requires an exact campaign state store")
        if type(staging) is not ArtifactStagingStore or type(authority) is not EvaluatorAuthority:
            raise CampaignCoordinatorError("coordinator boundaries are not validated")
        self.state_store = state_store
        self.staging = staging
        self.authority = authority

    def apply(self, gate: CampaignGate, *, expected_state_digest: str) -> CampaignState:
        if type(gate) is not CampaignGate:
            raise CampaignCoordinatorError("coordinator requires an exact validated gate")
        current = self.state_store.load()
        if current is None:
            raise CampaignCoordinatorError("campaign lifecycle is not initialized")
        if current.campaign_id != self.authority.campaign_id or gate.transfer.campaign_id != current.campaign_id:
            raise CampaignCoordinatorError("campaign identity differs across coordinator boundaries")
        self._verify_phase_state(current, gate.to_phase)

        # A retry after the durable state replacement is idempotent only when
        # every binding proves this is the exact previously committed step.
        if current.phase == gate.to_phase:
            receipt_digest = gate.receipt.digest
            if (
                current.previous_state_digest != expected_state_digest
                or current.last_receipt_digest != receipt_digest
            ):
                raise CampaignCoordinatorError("retry does not match the committed campaign step")
            self._verify_gate(
                gate,
                previous_receipt_digest=self._previous_receipt_for_retry(gate),
                expected_sequence=current.sequence,
            )
            return current

        if current.digest != expected_state_digest or current.phase != gate.from_phase:
            raise CampaignCoordinatorError("campaign state is stale or gate phase does not match")
        receipt_digest = self._verify_gate(
            gate,
            previous_receipt_digest=current.last_receipt_digest,
            expected_sequence=current.sequence + 1,
        )
        if gate.receipt.decision != "ACCEPT":
            raise CampaignCoordinatorError("rejected evaluator gate cannot advance the campaign")
        return self.state_store.advance(
            gate.to_phase,
            expected_digest=current.digest,
            last_receipt_digest=receipt_digest,
        )

    @staticmethod
    def _previous_receipt_for_retry(gate: CampaignGate) -> Optional[str]:
        value = gate.receipt.previous_receipt_digest
        return None if value == "0" * 64 else value

    def _verify_gate(
        self,
        gate: CampaignGate,
        *,
        previous_receipt_digest: Optional[str],
        expected_sequence: int,
    ) -> str:
        self.staging.verify(gate.transfer, expected_digest=gate.receipt.transfer_digest)
        self._verify_phase_evidence(gate)
        return self.authority.verify(
            gate.receipt,
            phase=gate.to_phase,
            transfer_digest=gate.transfer.digest,
            artifact_set_digest=gate.transfer.artifact_set_digest,
            expected_sequence=expected_sequence,
            expected_previous_digest=previous_receipt_digest,
        )

    def _verify_phase_evidence(self, gate: CampaignGate) -> None:
        try:
            source_role, destination_role, required_roles = PHASE_EVIDENCE[gate.to_phase]
        except KeyError as exc:
            raise CampaignCoordinatorError("destination phase has no closed evidence contract") from exc
        observed_roles = [artifact.role for artifact in gate.transfer.artifacts]
        if len(observed_roles) != len(set(observed_roles)):
            raise CampaignCoordinatorError("phase evidence contains duplicate semantic roles")
        if (
            gate.transfer.source_role != source_role
            or gate.transfer.destination_role != destination_role
            or frozenset(observed_roles) != required_roles
        ):
            raise CampaignCoordinatorError("transfer does not satisfy destination phase evidence contract")
        for artifact in gate.transfer.artifacts:
            if artifact.byte_count > _MAX_PHASE_EVIDENCE_BYTES:
                raise CampaignCoordinatorError("phase evidence envelope exceeds the bounded size")
            raw = self.staging.payload_bytes(gate.transfer, artifact.artifact_id)
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeError, ValueError) as exc:
                raise CampaignCoordinatorError("phase evidence is not canonical UTF-8 JSON") from exc
            if not isinstance(value, dict) or set(value) != _PHASE_EVIDENCE_FIELDS:
                raise CampaignCoordinatorError("phase evidence envelope is not a closed schema")
            if raw != canonical_bytes(value):
                raise CampaignCoordinatorError("phase evidence envelope is not canonical JSON")
            if (
                value["schema_version"] != PHASE_EVIDENCE_SCHEMA
                or value["campaign_id"] != gate.transfer.campaign_id
                or value["phase"] != gate.to_phase
                or value["role"] != artifact.role
                or value["status"] != "PASS"
            ):
                raise CampaignCoordinatorError("phase evidence envelope binding or status failed")
            try:
                validate_sha256(value["subject_digest"], "phase evidence subject digest")
            except Exception as exc:
                raise CampaignCoordinatorError("phase evidence subject digest is invalid") from exc

    @staticmethod
    def _verify_phase_state(state: CampaignState, destination_phase: str) -> None:
        phase_index = PHASES.index(destination_phase)
        if phase_index <= 3:
            valid = not state.restore_required and state.service_state == "RUNNING"
        elif phase_index <= 12:
            valid = state.restore_required and state.service_state == "STOPPED"
        else:
            valid = not state.restore_required and state.service_state == "RESTORED"
        if not valid:
            raise CampaignCoordinatorError("service lifecycle state does not authorize destination phase")


__all__ = [
    "CampaignCoordinator", "CampaignCoordinatorError", "CampaignGate", "PHASE_EVIDENCE",
    "PHASE_EVIDENCE_SCHEMA",
]
