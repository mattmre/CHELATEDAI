"""Bounded success/failure evidence retrieval policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from ..canonical import digest_for
from ..ledger import EvidenceLedger
from .arms import ArmIsolation
from .errors import VariationIsolationError


RETRIEVAL_POLICIES = (
    "SUCCESS_ONLY",
    "ORDINARY_FAILURE_SUMMARY",
    "CORRECTION_AWARE",
)


@dataclass(frozen=True)
class RetrievedEvidence:
    event_id: str
    event_type: str
    subject_id: Optional[str]
    task_id: Optional[str]
    recorded_disposition: str
    diagnostic_enum: Optional[str]
    failure_family_root: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "subject_id": self.subject_id,
            "task_id": self.task_id,
            "recorded_disposition": self.recorded_disposition,
        }
        if self.diagnostic_enum is not None:
            result["diagnostic_enum"] = self.diagnostic_enum
        if self.failure_family_root is not None:
            result["failure_family_root"] = self.failure_family_root
        return result


@dataclass(frozen=True)
class RetrievalResult:
    policy: str
    arm_id: str
    task_id: str
    records: Tuple[RetrievedEvidence, ...]
    evidence_digest: str

    @property
    def evidence_ids(self) -> Tuple[str, ...]:
        return tuple(record.event_id for record in self.records)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "arm_id": self.arm_id,
            "task_id": self.task_id,
            "evidence_ids": list(self.evidence_ids),
            "evidence_digest": self.evidence_digest,
            "records": [record.to_dict() for record in self.records],
        }


def _run_arms(ledger: EvidenceLedger, campaign_id: str, arm_id: str) -> Set[str]:
    rows = ledger.connection.execute(
        "SELECT run_id FROM runs WHERE campaign_id=? AND arm=? ORDER BY run_id", (campaign_id, arm_id)
    ).fetchall()
    return {str(row[0]) for row in rows}


def _candidate_diagnostic(ledger: EvidenceLedger, candidate_id: str) -> Tuple[Optional[str], Optional[str]]:
    for receipt in ledger.receipts():
        if receipt.get("candidate_id") != candidate_id:
            continue
        if receipt.get("receipt_type") == "VERDICT":
            return receipt.get("diagnostic_enum"), receipt.get("failure_family_root")
    return None, None


class EvidenceRetrievalPolicy:
    """Read-only policy facade over the authoritative ledger."""

    def __init__(self, policy: str) -> None:
        if policy not in RETRIEVAL_POLICIES:
            raise ValueError("unknown frozen Variation retrieval policy")
        self.policy = policy

    @property
    def digest(self) -> str:
        return digest_for({"schema_version": "egv-variation-retrieval-v1", "policy": self.policy})

    def _candidate_record(self, ledger: EvidenceLedger, event: Mapping[str, Any]) -> RetrievedEvidence:
        candidate_id = event.get("subject_id")
        disposition = "OBSERVED"
        diagnostic = None
        failure_root = None
        if isinstance(candidate_id, str):
            disposition = ledger.candidate_disposition(candidate_id)
            diagnostic, failure_root = _candidate_diagnostic(ledger, candidate_id)
        return RetrievedEvidence(
            event_id=str(event["event_id"]),
            event_type=str(event["event_type"]),
            subject_id=candidate_id,
            task_id=event.get("task_id"),
            recorded_disposition=disposition,
            diagnostic_enum=diagnostic,
            failure_family_root=failure_root,
        )

    def retrieve(
        self,
        ledger: EvidenceLedger,
        *,
        campaign_id: str,
        arm_id: str,
        task_id: str,
        isolation: Optional[ArmIsolation] = None,
    ) -> RetrievalResult:
        run_ids = _run_arms(ledger, campaign_id, arm_id)
        records: List[RetrievedEvidence] = []
        for event in ledger.current_valid_events():
            if event.get("campaign_id") != campaign_id or event.get("run_id") not in run_ids:
                continue
            if event.get("task_id") not in {None, task_id}:
                continue
            event_type = event.get("event_type")
            if event_type == "CANDIDATE":
                candidate = self._candidate_record(ledger, event)
                if self.policy == "SUCCESS_ONLY" and candidate.recorded_disposition != "PROMOTED":
                    continue
                if self.policy == "ORDINARY_FAILURE_SUMMARY" and candidate.recorded_disposition not in {"REJECTED", "ABSTAINED"}:
                    continue
                records.append(candidate)
            elif self.policy == "CORRECTION_AWARE" and event_type in {"CORRECTION", "RETRACTION"}:
                records.append(
                    RetrievedEvidence(
                        event_id=str(event["event_id"]),
                        event_type=str(event_type),
                        subject_id=event.get("subject_id"),
                        task_id=event.get("task_id"),
                        recorded_disposition=str(event.get("disposition") or "OBSERVED"),
                        diagnostic_enum=None,
                        failure_family_root=None,
                    )
                )
        records.sort(key=lambda item: item.event_id)
        if isolation is not None:
            for record in records:
                event = next(item for item in ledger.current_valid_events() if item["event_id"] == record.event_id)
                event_run = event.get("run_id")
                if event_run is None:
                    raise VariationIsolationError("retrieved evidence is not bound to an arm run")
                row = ledger.connection.execute("SELECT arm FROM runs WHERE run_id=?", (event_run,)).fetchone()
                isolation.assert_retrieval_arm(arm_id, str(row[0]) if row is not None else None)
        digest = digest_for([record.to_dict() for record in records])
        return RetrievalResult(self.policy, arm_id, task_id, tuple(records), digest)


class SuccessOnlyRetrieval(EvidenceRetrievalPolicy):
    def __init__(self) -> None:
        super().__init__("SUCCESS_ONLY")


class OrdinaryFailureRetrieval(EvidenceRetrievalPolicy):
    def __init__(self) -> None:
        super().__init__("ORDINARY_FAILURE_SUMMARY")


class CorrectionAwareRetrieval(EvidenceRetrievalPolicy):
    def __init__(self) -> None:
        super().__init__("CORRECTION_AWARE")


def retrieval_policy(name: str) -> EvidenceRetrievalPolicy:
    return EvidenceRetrievalPolicy(name)


__all__ = [
    "CorrectionAwareRetrieval",
    "EvidenceRetrievalPolicy",
    "OrdinaryFailureRetrieval",
    "RETRIEVAL_POLICIES",
    "RetrievedEvidence",
    "RetrievalResult",
    "SuccessOnlyRetrieval",
    "retrieval_policy",
]
