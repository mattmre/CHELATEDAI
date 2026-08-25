"""Evidence-governed experimental campaign orchestration."""

from .heldout import (
    CoordinateOperationStore,
    FrozenHeldoutProtocol,
    HeldoutCoordinate,
    HeldoutJournal,
    HeldoutProtocolError,
    analyze_heldout_campaign,
    build_private_campaign_record,
    build_signed_reconciliation,
    build_signed_result_envelope,
    run_pending_coordinates,
    verify_signed_result_envelope,
)
from .remote import HeldoutVerifierServiceManifest, run_heldout_verifier_once
from .runtime import build_trainer_evidence_package, ordered_public_heldout_task_records

__all__ = [
    "CoordinateOperationStore",
    "FrozenHeldoutProtocol",
    "HeldoutCoordinate",
    "HeldoutJournal",
    "HeldoutProtocolError",
    "HeldoutVerifierServiceManifest",
    "analyze_heldout_campaign",
    "build_private_campaign_record",
    "build_signed_reconciliation",
    "build_signed_result_envelope",
    "build_trainer_evidence_package",
    "ordered_public_heldout_task_records",
    "run_heldout_verifier_once",
    "run_pending_coordinates",
    "verify_signed_result_envelope",
]
