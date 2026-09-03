"""Production correction-shock engine over the sealed Qwen/evaluator boundary.

The runtime in :mod:`egv.experiment.shock_runtime` owns the experiment cursor.
This module owns the non-idempotent attempt boundary.  A model call is never
repeated after its durable start marker; recovery is possible only from exact
private generation evidence.  Evaluator recovery replays the remote service's
content-addressed response and requires the signed receipt suffix to remain
byte-for-byte identical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..campaign.state import _exclusive_path_lock
from ..canonical import (
    canonical_bytes,
    canonical_json,
    content_id,
    digest_bytes,
    digest_for,
    validate_sha256,
)
from ..evaluation.authority import AuthorityPolicy
from ..evaluation.controller import EvaluationResult
from ..evaluation.diagnostics import Diagnostic, validate_diagnostic, validate_disposition, validate_resource_bucket
from ..evaluation.shock import SHOCK_SEEDS, DependencyGraph
from ..ledger import EvidenceLedger
from ..receipts import receipt_hash
from ..variation.arms import ArmIsolation, arm_policy
from ..variation.generator import (
    CandidateContext,
    CandidateGenerationFailure,
    CandidateGenerationFailureEvidence,
    CandidateGenerationEvidence,
    CandidateProposal,
    ModelCandidateGenerator,
)
from ..variation.loop import CANDIDATE_SOURCE_LIMIT, VARIATION_PROTOCOL_DIGEST
from ..variation.model import MODEL_REVISION
from ..variation.private import PrivateTrajectoryStore
from ..variation.remote import RemoteControllerEvaluationGateway
from ..variation.retrieval import RetrievedEvidence, RetrievalResult, retrieval_policy
from .heldout import HeldoutCoordinate, HeldoutProtocolError, SHOCK_PHASE
from .shock_runtime import PreShockState, ShockAttemptObservation, ShockVerificationEvidence


SHOCK_OPERATION_SCHEMA = "egv-production-shock-operation-v1"
SHOCK_ATTEMPT_EVENT_SCHEMA = "egv-production-shock-attempt-v1"
SHOCK_PREMISE_SCHEMA = "egv-production-shock-premise-v1"
SHOCK_CORRECTION_SCHEMA = "egv-production-shock-correction-v1"
SHOCK_POLICY_SCHEMA = "egv-production-shock-policy-v1"
SHOCK_UNRELATED_SCHEMA = "egv-production-shock-unrelated-evidence-v1"
SHOCK_GENERATION_FAILURE_SCHEMA = "egv-production-shock-generation-failure-v1"
_OPERATION_STATUSES = {"STARTED", "GENERATED", "GENERATION_FAILED", "COMPLETE"}
_TERMINAL_OPERATION_STATUSES = {"GENERATION_FAILED", "COMPLETE"}
_OPERATION_FIELDS = {
    "schema_version",
    "coordinate_id",
    "operation_id",
    "phase",
    "attempt",
    "candidate_id",
    "run_id",
    "context_digest",
    "status",
    "generation_evidence_digest",
    "candidate_source_digest",
    "evaluation_result",
    "observation",
    "rng_state_evidence",
    "rng_state_evidence_digest",
}


def _require_digest(value: Any, label: str) -> str:
    validate_sha256(value, label)
    return str(value)


def _context_digest(context: CandidateContext) -> str:
    if type(context) is not CandidateContext:
        raise HeldoutProtocolError("shock candidate context type is not exact")
    return digest_for(asdict(context))


def _result_from_mapping(value: Mapping[str, Any]) -> EvaluationResult:
    required = {
        "candidate_id",
        "task_id",
        "candidate_artifact_digest",
        "diagnostic_enum",
        "resource_bucket",
        "disposition",
        "infrastructure_loss",
        "receipt_ids",
        "output_digest",
    }
    optional = {"infrastructure_incident_id", "failure_family_root"}
    if (
        not isinstance(value, Mapping)
        or not required.issubset(value)
        or set(value) - required - optional
        or not isinstance(value.get("receipt_ids"), list)
    ):
        raise HeldoutProtocolError("durable shock evaluation result is not closed")
    payload = dict(value)
    payload["receipt_ids"] = tuple(payload["receipt_ids"])
    try:
        result = EvaluationResult(**payload)
    except TypeError as exc:
        raise HeldoutProtocolError("durable shock evaluation result is malformed") from exc
    return result


def _failure_diagnostic(evidence: CandidateGenerationFailureEvidence) -> str:
    """Map a closed generation-failure stage to the frozen diagnostic vocabulary."""

    evidence_stage = str(evidence.stage)
    if evidence_stage in {"PROMPT_INTEGRITY", "RESPONSE_CONTRACT"}:
        return Diagnostic.PROTOCOL_VIOLATION.value
    if evidence_stage in {"PROMPT_RENDER", "MODEL_GENERATION"}:
        return Diagnostic.INTERNAL_ERROR.value
    raise HeldoutProtocolError("shock generation failure stage is outside the closed vocabulary")


def _validate_failure_observation(observation: ShockAttemptObservation) -> None:
    observation.validate()
    if (
        observation.promoted
        or observation.receipt_valid
        or observation.tokens != 0
        or float(observation.evaluator_seconds) != 0.0
        or observation.evidence_used
        or observation.authority_challenge
        or observation.valid_authority_denial
        or observation.promoted_node_ids
        or observation.independent_hidden_fixture_passed
        or observation.verdict_receipt_digest is not None
        or observation.effect_receipt_digest is not None
        or observation.diagnostic_enum
        not in {Diagnostic.PROTOCOL_VIOLATION.value, Diagnostic.INTERNAL_ERROR.value}
    ):
        raise HeldoutProtocolError("terminal shock generation failure carries evaluator or promotion claims")


class DurableShockOperationStore:
    """Canonical per-attempt state for crash-safe model/evaluator recovery."""

    def __init__(self, root: Path, coordinate: HeldoutCoordinate) -> None:
        requested_root = Path(root)
        if requested_root.exists() and (
            requested_root.is_symlink() or not requested_root.is_dir()
        ):
            raise HeldoutProtocolError("shock operation store root was substituted")
        requested_root.mkdir(parents=True, exist_ok=True)
        if requested_root.is_symlink():
            raise HeldoutProtocolError("shock operation store root was substituted")
        self.root = requested_root.resolve() / "operations"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise HeldoutProtocolError("shock operation root is not a regular directory")
        self.coordinate = coordinate
        self.lock_path = self.root.parent / ".shock-engine.lock"
        if self.lock_path.is_symlink():
            raise HeldoutProtocolError("shock operation lock was substituted")
        for path in self.root.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise HeldoutProtocolError("shock operation inventory contains an unexpected object")
            self._load_path(path)

    def _path(self, operation_id: str) -> Path:
        _require_digest(operation_id, "shock operation ID")
        return self.root / (operation_id + ".json")

    def _load_path(self, path: Path) -> Dict[str, Any]:
        try:
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise HeldoutProtocolError("shock operation state is unreadable") from exc
        if (
            not isinstance(value, dict)
            or set(value) != _OPERATION_FIELDS
            or canonical_bytes(value) != raw
            or value["schema_version"] != SHOCK_OPERATION_SCHEMA
            or value["coordinate_id"] != self.coordinate.coordinate_id
            or value["status"] not in _OPERATION_STATUSES
            or value["phase"] not in {"PRE", "POST"}
            or type(value["attempt"]) is not int
            or not 1 <= value["attempt"] <= 6
            or not isinstance(value["candidate_id"], str)
            or not value["candidate_id"]
            or not isinstance(value["run_id"], str)
            or not value["run_id"]
        ):
            raise HeldoutProtocolError("shock operation state is noncanonical or inconsistent")
        _require_digest(value["operation_id"], "shock operation ID")
        _require_digest(value["context_digest"], "shock context digest")
        expected_operation = digest_for(
            {
                "block_id": self.coordinate.block_id if value["phase"] == "PRE" else None,
                "coordinate_id": (
                    self.coordinate.coordinate_id if value["phase"] == "POST" else None
                ),
                "phase": value["phase"],
                "attempt": value["attempt"],
            }
        )
        if (
            path.name != value["operation_id"] + ".json"
            or value["operation_id"] != expected_operation
        ):
            raise HeldoutProtocolError("shock operation filename differs from its identity")
        status = value["status"]
        generated = status in {"GENERATED", "COMPLETE"}
        failed = status == "GENERATION_FAILED"
        if generated:
            _require_digest(value["generation_evidence_digest"], "generation evidence digest")
            _require_digest(value["candidate_source_digest"], "candidate source digest")
        elif failed:
            _require_digest(value["generation_evidence_digest"], "generation failure evidence digest")
            if value["candidate_source_digest"] is not None:
                raise HeldoutProtocolError("failed shock generation carries candidate source")
        elif value["generation_evidence_digest"] is not None or value["candidate_source_digest"] is not None:
            raise HeldoutProtocolError("unresolved shock generation carries completed evidence")
        if status == "COMPLETE":
            result = _result_from_mapping(value["evaluation_result"])
            observation = ShockAttemptObservation.from_mapping(value["observation"])
            if result.candidate_id != value["candidate_id"] or observation.operation_id != value["operation_id"]:
                raise HeldoutProtocolError("completed shock operation crossed its candidate or operation")
            if not isinstance(value["rng_state_evidence"], Mapping):
                raise HeldoutProtocolError("completed shock operation lacks actual RNG evidence")
            _require_digest(value["rng_state_evidence_digest"], "actual RNG evidence digest")
            if digest_for(dict(value["rng_state_evidence"])) != value["rng_state_evidence_digest"]:
                raise HeldoutProtocolError("actual RNG evidence digest mismatch")
        elif failed:
            if value["evaluation_result"] is not None:
                raise HeldoutProtocolError("failed shock generation carries an evaluator result")
            observation = ShockAttemptObservation.from_mapping(value["observation"])
            _validate_failure_observation(observation)
            if observation.operation_id != value["operation_id"]:
                raise HeldoutProtocolError("failed shock observation crossed its operation")
            if not isinstance(value["rng_state_evidence"], Mapping):
                raise HeldoutProtocolError("failed shock generation lacks actual RNG evidence")
            _require_digest(value["rng_state_evidence_digest"], "actual RNG evidence digest")
            if digest_for(dict(value["rng_state_evidence"])) != value["rng_state_evidence_digest"]:
                raise HeldoutProtocolError("actual RNG evidence digest mismatch")
        elif any(value[field] is not None for field in (
            "evaluation_result", "observation", "rng_state_evidence", "rng_state_evidence_digest"
        )):
            raise HeldoutProtocolError("incomplete shock operation carries terminal evidence")
        return value

    def load(self, operation_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(operation_id)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise HeldoutProtocolError("shock operation path was substituted")
        return self._load_path(path)

    def records(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(self._load_path(path) for path in sorted(self.root.glob("*.json")))

    def begin(
        self,
        *,
        operation_id: str,
        phase: str,
        attempt: int,
        candidate_id: str,
        run_id: str,
        context_digest: str,
    ) -> Tuple[Dict[str, Any], bool]:
        value = {
            "schema_version": SHOCK_OPERATION_SCHEMA,
            "coordinate_id": self.coordinate.coordinate_id,
            "operation_id": operation_id,
            "phase": phase,
            "attempt": attempt,
            "candidate_id": candidate_id,
            "run_id": run_id,
            "context_digest": context_digest,
            "status": "STARTED",
            "generation_evidence_digest": None,
            "candidate_source_digest": None,
            "evaluation_result": None,
            "observation": None,
            "rng_state_evidence": None,
            "rng_state_evidence_digest": None,
        }
        with _exclusive_path_lock(self.lock_path):
            prior = self.load(operation_id)
            if prior is not None:
                for field in (
                    "coordinate_id", "operation_id", "phase", "attempt", "candidate_id", "run_id", "context_digest"
                ):
                    if prior[field] != value[field]:
                        raise HeldoutProtocolError("shock operation identity was reused with different inputs")
                return prior, False
            self._write(self._path(operation_id), value)
            return (self.load(operation_id) or value), True

    def generated(self, operation_id: str, *, evidence_digest: str, source_digest: str) -> Dict[str, Any]:
        return self._transition(
            operation_id,
            allowed={"STARTED", "GENERATED"},
            updates={
                "status": "GENERATED",
                "generation_evidence_digest": _require_digest(evidence_digest, "generation evidence digest"),
                "candidate_source_digest": _require_digest(source_digest, "candidate source digest"),
            },
        )

    def generation_failed(
        self,
        operation_id: str,
        *,
        evidence_digest: str,
        observation: ShockAttemptObservation,
        rng_state_evidence: Mapping[str, Any],
    ) -> Dict[str, Any]:
        terminal_observation = observation
        _validate_failure_observation(terminal_observation)
        evidence = dict(rng_state_evidence)
        return self._transition(
            operation_id,
            allowed={"STARTED", "GENERATION_FAILED"},
            updates={
                "status": "GENERATION_FAILED",
                "generation_evidence_digest": _require_digest(
                    evidence_digest, "generation failure evidence digest"
                ),
                "evaluation_result": None,
                "observation": terminal_observation.to_dict(),
                "rng_state_evidence": evidence,
                "rng_state_evidence_digest": digest_for(evidence),
            },
        )

    def complete(
        self,
        operation_id: str,
        *,
        result: EvaluationResult,
        observation: ShockAttemptObservation,
        rng_state_evidence: Mapping[str, Any],
    ) -> Dict[str, Any]:
        evidence = dict(rng_state_evidence)
        observation.validate()
        return self._transition(
            operation_id,
            allowed={"GENERATED", "COMPLETE"},
            updates={
                "status": "COMPLETE",
                "evaluation_result": result.to_dict(),
                "observation": observation.to_dict(),
                "rng_state_evidence": evidence,
                "rng_state_evidence_digest": digest_for(evidence),
            },
        )

    def _transition(
        self, operation_id: str, *, allowed: set[str], updates: Mapping[str, Any]
    ) -> Dict[str, Any]:
        with _exclusive_path_lock(self.lock_path):
            value = self.load(operation_id)
            if value is None or value["status"] not in allowed:
                raise HeldoutProtocolError("shock operation transition is not admissible")
            updated = dict(value)
            updated.update(updates)
            if value["status"] == updates.get("status"):
                if canonical_bytes(updated) != canonical_bytes(value):
                    raise HeldoutProtocolError("shock operation replay conflicts with durable state")
                return value
            self._write(self._path(operation_id), updated)
            return self.load(operation_id) or updated

    def _write(self, path: Path, value: Mapping[str, Any]) -> None:
        descriptor, name = tempfile.mkstemp(prefix=".shock-operation-", dir=str(self.root))
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_bytes(value))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(path))
        finally:
            if temporary.exists():
                temporary.unlink()


@dataclass(frozen=True)
class ProductionShockEngineFactory:
    """Build exact production engines after the pinned model is loaded once."""

    generator: ModelCandidateGenerator
    evaluator_manifest: Path
    evaluator_public_key: Path
    evaluator_command: Path
    evaluator_python_executable: Path
    evaluator_python_digest: str
    source_commit: str
    model_revision: str
    data_manifest_digest: str
    response_contract_digest: str
    generation_profile_digest: str

    def __post_init__(self) -> None:
        if type(self.generator) is not ModelCandidateGenerator:
            raise HeldoutProtocolError("production shock factory requires exact ModelCandidateGenerator")
        self.generator.validate_production_integrity()
        if self.generator.adapter_digest is None:
            raise HeldoutProtocolError("production correction shock requires the sealed trained adapter")
        if self.model_revision != MODEL_REVISION:
            raise HeldoutProtocolError("production shock model revision differs from frozen Qwen")
        if (
            self.response_contract_digest != self.generator.response_contract_digest
            or self.generation_profile_digest != self.generator.generation_profile_digest
            or self.generator.response_contract == "closed-json-v1"
        ):
            raise HeldoutProtocolError("production shock generator differs from its source-only profile")
        _require_digest(self.data_manifest_digest, "shock data manifest digest")
        _require_digest(self.evaluator_python_digest, "shock evaluator Python executable digest")
        if not isinstance(self.source_commit, str) or not self.source_commit:
            raise HeldoutProtocolError("production shock source commit is missing")

    def __call__(self, **kwargs: Any) -> "ProductionShockAttemptEngine":
        ledger = kwargs.get("ledger")
        if type(ledger) is not EvidenceLedger:
            raise HeldoutProtocolError("production shock factory requires isolated EvidenceLedger")
        gateway = RemoteControllerEvaluationGateway(
            ledger=ledger,
            manifest_path=Path(self.evaluator_manifest),
            public_key_path=Path(self.evaluator_public_key),
            command=Path(self.evaluator_command),
            python_executable=Path(self.evaluator_python_executable),
            python_digest=self.evaluator_python_digest,
        )
        return ProductionShockAttemptEngine(
            **kwargs,
            generator=self.generator,
            evaluator=gateway,
            evaluator_public_key=Path(self.evaluator_public_key).read_bytes(),
            source_commit=self.source_commit,
            model_revision=self.model_revision,
            data_manifest_digest=self.data_manifest_digest,
            response_contract_digest=self.response_contract_digest,
            generation_profile_digest=self.generation_profile_digest,
        )


class ProductionShockAttemptEngine:
    """One correction-shock clone with exact private and signed evidence."""

    def __init__(
        self,
        *,
        coordinate: HeldoutCoordinate,
        task_record: Mapping[str, Any],
        initial_source: bytes,
        ledger: EvidenceLedger,
        isolation: ArmIsolation,
        private_store: PrivateTrajectoryStore,
        base_model_digest: str,
        adapter_digest: Optional[str],
        generator: ModelCandidateGenerator,
        evaluator: RemoteControllerEvaluationGateway,
        evaluator_public_key: bytes,
        source_commit: str,
        model_revision: str,
        data_manifest_digest: str,
        response_contract_digest: str,
        generation_profile_digest: str,
    ) -> None:
        if coordinate.phase != SHOCK_PHASE:
            raise HeldoutProtocolError("production shock engine received non-shock coordinate")
        if (
            type(ledger) is not EvidenceLedger
            or type(isolation) is not ArmIsolation
            or type(private_store) is not PrivateTrajectoryStore
            or type(generator) is not ModelCandidateGenerator
            or type(evaluator) is not RemoteControllerEvaluationGateway
        ):
            raise HeldoutProtocolError("production shock engine dependency type was substituted")
        if adapter_digest is None or generator.adapter_artifact is None:
            raise HeldoutProtocolError("production correction shock requires the sealed trained adapter")
        common_root = Path(isolation.root).resolve().parent
        if Path(ledger.path).resolve().parent != common_root or Path(private_store.root).resolve().parent != common_root:
            raise HeldoutProtocolError("production shock ledger/private state is not coordinate-isolated")
        self.coordinate = coordinate
        self.task_record = dict(task_record)
        self.initial_source = bytes(initial_source)
        self.ledger = ledger
        self.isolation = isolation
        self.private_store = private_store
        self.generator = generator
        self.evaluator = evaluator
        self.evaluator_public_key = bytes(evaluator_public_key)
        self.source_commit = source_commit
        self.model_revision = model_revision
        self.data_manifest_digest = _require_digest(data_manifest_digest, "shock data manifest digest")
        self.response_contract_digest = response_contract_digest
        self.generation_profile_digest = generation_profile_digest
        self.base_model_digest = _require_digest(base_model_digest, "shock base model digest")
        self.adapter_digest = adapter_digest
        self.arm_id = "E"
        self.policy = arm_policy(self.arm_id)
        self.campaign_id = isolation.campaign_id
        self.operation_store = DurableShockOperationStore(common_root / "shock-engine", coordinate)
        self._validate_bindings()
        self._ensure_ledger()

    def _validate_bindings(self) -> None:
        required = {
            "template_id",
            "family_id",
            "split",
            "ordinal",
            "source_digest",
            "public_locus",
            "public_rule_id",
        }
        if (
            not required.issubset(self.task_record)
            or self.task_record["template_id"] != self.coordinate.task_id
            or self.task_record["split"] != "heldout"
            or type(self.task_record["ordinal"]) is not int
            or self.task_record["ordinal"] < 1
            or not self.initial_source
            or len(self.initial_source) > CANDIDATE_SOURCE_LIMIT
        ):
            raise HeldoutProtocolError("production shock task/source binding is invalid")
        _require_digest(self.task_record["source_digest"], "shock repository source digest")
        try:
            self.initial_source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HeldoutProtocolError("production shock source is not UTF-8") from exc
        self.generator.validate_production_integrity()
        self.evaluator.validate_runtime()
        if (
            self.generator.model_digest != self.base_model_digest
            or self.generator.adapter_digest != self.adapter_digest
            or self.generator.response_contract_digest != self.response_contract_digest
            or self.generator.generation_profile_digest != self.generation_profile_digest
            or self.generator.response_contract == "closed-json-v1"
            or self.model_revision != MODEL_REVISION
        ):
            raise HeldoutProtocolError("production shock model/adapter/profile binding is stale")
        authority = AuthorityPolicy.candidate_execution().digest
        self.evaluator.validate_campaign_bindings(
            campaign_id=self.campaign_id,
            model_digest=self.base_model_digest,
            protocol_digest=VARIATION_PROTOCOL_DIGEST,
            policy_digest=authority,
            data_manifest_digest=self.data_manifest_digest,
        )
        public = self.evaluator.manifest.public_record(self.coordinate.task_id)
        if public is None or any(public.get(field) != self.task_record.get(field) for field in required):
            raise HeldoutProtocolError("remote hidden evaluator public task binding differs")
        self.ledger.verify_receipt_chain(self.evaluator_public_key)

    def _pre_run_id(self) -> str:
        return content_id("run-shock-pre", {"campaign_id": self.campaign_id, "block_id": self.coordinate.block_id})

    def _post_run_id(self) -> str:
        return content_id(
            "run-shock-post",
            {"campaign_id": self.campaign_id, "coordinate_id": self.coordinate.coordinate_id},
        )

    def _ensure_run(self, run_id: str) -> None:
        self.ledger.create_run(
            run_id,
            campaign_id=self.campaign_id,
            arm=self.arm_id,
            task_id=self.coordinate.task_id,
            seed=self.coordinate.seed,
            parent_checkpoint=None,
            start_state="READY",
            host_role="spark_trainer",
            software_manifest_hash=digest_for({"source_commit": self.source_commit}),
        )
        self.isolation.workspace(self.arm_id, run_id)

    def _ensure_ledger(self) -> None:
        authority = AuthorityPolicy.candidate_execution().digest
        self.ledger.create_campaign(
            self.campaign_id,
            protocol_hash=VARIATION_PROTOCOL_DIGEST,
            source_commit=self.source_commit,
            model_revision=self.model_revision,
            data_manifest_hash=self.data_manifest_digest,
            evaluator_hash=self.evaluator.evaluator_digest,
            policy_hash=authority,
            seed_set=SHOCK_SEEDS,
        )
        self._ensure_run(self._pre_run_id())
        self._premise_event()
        self._unrelated_evidence_events()

    def _premise_event(self) -> Mapping[str, Any]:
        payload = {
            "schema_version": SHOCK_PREMISE_SCHEMA,
            "block_id": self.coordinate.block_id,
            "task_id": self.coordinate.task_id,
            "seed": self.coordinate.seed,
            "fixture_digest": self.coordinate.fixture_digest,
            "accepted_premise_digest": self.coordinate.accepted_premise_digest,
        }
        return self.ledger.append_event(
            "SHOCK_PREMISE",
            payload,
            campaign_id=self.campaign_id,
            run_id=self._pre_run_id(),
            task_id=self.coordinate.task_id,
            subject_id=content_id("shock-premise", payload),
            source_class="FROZEN_PROTOCOL",
            disposition="OBSERVED",
            idempotency_key="shock-premise:" + self.coordinate.block_id,
        )

    def _unrelated_evidence_events(
        self,
    ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        binding_payload = {
            "schema_version": SHOCK_UNRELATED_SCHEMA,
            "component": "PUBLIC_TASK_BINDING",
            "block_id": self.coordinate.block_id,
            "task_id": self.coordinate.task_id,
            "task_binding_digest": digest_for(self.task_record),
            "public_rule_id": self.task_record["public_rule_id"],
            "public_locus": self.task_record["public_locus"],
        }
        root = self.ledger.append_event(
            "SHOCK_UNRELATED_ROOT",
            binding_payload,
            campaign_id=self.campaign_id,
            run_id=self._pre_run_id(),
            task_id=self.coordinate.task_id,
            subject_id=content_id("shock-unrelated-root", binding_payload),
            source_class="FROZEN_PROTOCOL",
            disposition="VERIFIED",
            idempotency_key="shock-unrelated-root:" + self.coordinate.block_id,
        )
        source_payload = {
            "schema_version": SHOCK_UNRELATED_SCHEMA,
            "component": "INITIAL_SOURCE_COMMITMENT",
            "block_id": self.coordinate.block_id,
            "task_id": self.coordinate.task_id,
            "repository_source_digest": self.task_record["source_digest"],
            "task_source_file_digest": digest_bytes(self.initial_source),
            "task_binding_event_id": root["event_id"],
        }
        child = self.ledger.append_event(
            "SHOCK_UNRELATED_EVIDENCE",
            source_payload,
            campaign_id=self.campaign_id,
            run_id=self._pre_run_id(),
            task_id=self.coordinate.task_id,
            subject_id=content_id("shock-unrelated-evidence", source_payload),
            source_class="FROZEN_PROTOCOL",
            disposition="VERIFIED",
            idempotency_key="shock-unrelated-evidence:" + self.coordinate.block_id,
        )
        self.ledger.append_dependency(
            str(root["event_id"]),
            str(child["event_id"]),
            edge_type="UNRELATED_PUBLIC_EVIDENCE",
            campaign_id=self.campaign_id,
            run_id=self._pre_run_id(),
            task_id=self.coordinate.task_id,
            idempotency_key="shock-unrelated-dependency:" + self.coordinate.block_id,
        )
        return root, child

    def _operation_id(self, phase: str, attempt: int) -> str:
        identity = {
            "block_id": self.coordinate.block_id if phase == "PRE" else None,
            "coordinate_id": self.coordinate.coordinate_id if phase == "POST" else None,
            "phase": phase,
            "attempt": attempt,
        }
        return digest_for(identity)

    def _candidate_id(self, phase: str, attempt: int) -> str:
        identity = {
            "campaign_id": self.campaign_id,
            "block_id": self.coordinate.block_id if phase == "PRE" else None,
            "coordinate_id": self.coordinate.coordinate_id if phase == "POST" else None,
            "phase": phase,
            "attempt": attempt,
        }
        candidate_id = "egv-candidate-{}-{}-{}".format(
            self.campaign_id, self.arm_id, digest_for(identity)[:40]
        )
        self.isolation.assert_candidate_id(self.arm_id, candidate_id)
        return candidate_id

    def _terminal_records(self, phase: Optional[str] = None) -> Tuple[Dict[str, Any], ...]:
        records = tuple(
            record
            for record in self.operation_store.records()
            if record["status"] in _TERMINAL_OPERATION_STATUSES
        )
        if phase is not None:
            records = tuple(record for record in records if record["phase"] == phase)
        return tuple(sorted(records, key=lambda item: (item["phase"], item["attempt"])))

    def _successful_records(self, phase: Optional[str] = None) -> Tuple[Dict[str, Any], ...]:
        records = tuple(
            record for record in self.operation_store.records() if record["status"] == "COMPLETE"
        )
        if phase is not None:
            records = tuple(record for record in records if record["phase"] == phase)
        return tuple(sorted(records, key=lambda item: (item["phase"], item["attempt"])))

    def _assert_attempt_order(self, phase: str, attempt: int) -> None:
        if type(attempt) is not int or not 1 <= attempt <= 6:
            raise HeldoutProtocolError("shock attempt is outside 1..6")
        completed = {record["attempt"] for record in self._terminal_records(phase)}
        required_prior = set(range(1, attempt))
        if not required_prior.issubset(completed) or (
            attempt not in completed and completed != required_prior
        ):
            raise HeldoutProtocolError("shock attempt order skipped or repeated an unresolved coordinate")
        if phase == "POST" and attempt not in completed:
            prior_observations = tuple(
                ShockAttemptObservation.from_mapping(record["observation"])
                for record in self._terminal_records("POST")
            )
            recovered = any(
                observation.promoted and observation.independent_hidden_fixture_passed
                for observation in prior_observations
            )
            if recovered:
                raise HeldoutProtocolError("shock recovery window continued after verified recovery")
        if (
            phase == "PRE"
            and attempt not in completed
            and self._correction_receipt(required=False) is not None
        ):
            raise HeldoutProtocolError("pre-correction generation was attempted after correction")
        if phase == "POST":
            if len(self._terminal_records("PRE")) != 6 or self._policy_receipt(required=False) is None:
                raise HeldoutProtocolError("post-correction attempt lacks exact correction/policy evidence")

    def _event(self, event_id: str) -> Mapping[str, Any]:
        for event in self.ledger.events():
            if event["event_id"] == event_id:
                return event
        raise HeldoutProtocolError("shock evidence event is missing")

    def _candidate_retrieved_record(self, event: Mapping[str, Any]) -> RetrievedEvidence:
        candidate_id = str(event["subject_id"])
        diagnostic = None
        failure_root = None
        for receipt in self.ledger.receipts():
            if receipt.get("candidate_id") == candidate_id and receipt.get("receipt_type") == "VERDICT":
                diagnostic = receipt.get("diagnostic_enum")
                failure_root = receipt.get("failure_family_root")
        return RetrievedEvidence(
            event_id=str(event["event_id"]),
            event_type="CANDIDATE",
            subject_id=candidate_id,
            task_id=self.coordinate.task_id,
            recorded_disposition=self.ledger.candidate_disposition(candidate_id),
            diagnostic_enum=diagnostic,
            failure_family_root=failure_root,
        )

    def _replacement_event(self) -> Mapping[str, Any]:
        correction = self._correction_receipt(required=True)
        assert correction is not None
        payload = correction.get("payload")
        if not isinstance(payload, Mapping):
            raise HeldoutProtocolError("shock correction receipt payload is missing")
        replacement_id = payload.get("replacement_event_id")
        if not isinstance(replacement_id, str) or not replacement_id:
            raise HeldoutProtocolError("shock correction receipt lacks its replacement event")
        replacement = self._event(replacement_id)
        replacement_payload = replacement.get("payload")
        if (
            replacement.get("event_type") != "SHOCK_CORRECTED_PREMISE"
            or replacement.get("campaign_id") != self.campaign_id
            or replacement.get("task_id") != self.coordinate.task_id
            or not isinstance(replacement_payload, Mapping)
            or replacement_payload.get("schema_version") != SHOCK_CORRECTION_SCHEMA
            or replacement_payload.get("block_id") != self.coordinate.block_id
            or replacement_payload.get("correction_event_digest")
            != self.coordinate.correction_event_digest
        ):
            raise HeldoutProtocolError("shock corrected premise is not bound to the frozen coordinate")
        row = self.ledger.connection.execute(
            "SELECT superseded_id,replacement_id FROM corrections WHERE correction_id=?",
            (payload.get("correction_id"),),
        ).fetchone()
        if (
            row is None
            or row["superseded_id"] != self._premise_event()["event_id"]
            or row["replacement_id"] != replacement_id
        ):
            raise HeldoutProtocolError("shock correction ledger row crossed its premise or replacement")
        return replacement

    def _replacement_retrieved_record(self) -> RetrievedEvidence:
        replacement = self._replacement_event()
        return RetrievedEvidence(
            event_id=str(replacement["event_id"]),
            event_type="CORRECTION",
            subject_id=replacement.get("subject_id"),
            task_id=replacement.get("task_id"),
            recorded_disposition=self.ledger.event_disposition(str(replacement["event_id"])),
            diagnostic_enum=None,
            failure_family_root=None,
        )

    def _unrelated_retrieved_record(self) -> RetrievedEvidence:
        _root, child = self._unrelated_evidence_events()
        if self.ledger.event_disposition(str(child["event_id"])) != "VERIFIED":
            raise HeldoutProtocolError("unrelated shock evidence is no longer verified")
        return RetrievedEvidence(
            event_id=str(child["event_id"]),
            event_type="EVIDENCE",
            subject_id=child.get("subject_id"),
            task_id=child.get("task_id"),
            recorded_disposition="VERIFIED",
            diagnostic_enum=None,
            failure_family_root=None,
        )

    def _retrieval(self, phase: str) -> RetrievalResult:
        if phase == "PRE":
            return retrieval_policy("CORRECTION_AWARE").retrieve(
                self.ledger,
                campaign_id=self.campaign_id,
                arm_id=self.arm_id,
                task_id=self.coordinate.task_id,
                isolation=self.isolation,
            )
        records = [self._replacement_retrieved_record()]
        if self.coordinate.treatment == "full-restart":
            allowed_runs = {self._post_run_id()}
            events = self.ledger.current_valid_events()
        elif self.coordinate.treatment == "dependency-aware":
            records.append(self._unrelated_retrieved_record())
            allowed_runs = {self._pre_run_id(), self._post_run_id()}
            events = self.ledger.current_valid_events()
        else:
            records.append(self._unrelated_retrieved_record())
            allowed_runs = {self._pre_run_id(), self._post_run_id()}
            events = self.ledger.events()
        for event in events:
            if event.get("campaign_id") != self.campaign_id or event.get("task_id") != self.coordinate.task_id:
                continue
            if event.get("event_type") == "CANDIDATE" and event.get("run_id") in allowed_runs:
                records.append(self._candidate_retrieved_record(event))
        records.sort(key=lambda item: item.event_id)
        digest = digest_for([record.to_dict() for record in records])
        return RetrievalResult("CORRECTION_AWARE", self.arm_id, self.coordinate.task_id, tuple(records), digest)

    def _parent_candidate(self, phase: str) -> Optional[str]:
        if phase == "PRE":
            records = self._successful_records("PRE")
            return str(records[-1]["candidate_id"]) if records else None
        post = self._successful_records("POST")
        if post:
            return str(post[-1]["candidate_id"])
        if self.coordinate.treatment == "naive-reuse":
            pre = self._successful_records("PRE")
            return str(pre[-1]["candidate_id"]) if pre else None
        return None

    def _context(self, phase: str, attempt: int, run_id: str) -> CandidateContext:
        retrieval = self._retrieval(phase)
        provisional = CandidateContext(
            campaign_id=self.campaign_id,
            run_id=run_id,
            seed=self.coordinate.seed,
            arm_id=self.arm_id,
            task_id=self.coordinate.task_id,
            family_id=str(self.task_record["family_id"]),
            public_locus=str(self.task_record["public_locus"]),
            public_rule_id=str(self.task_record["public_rule_id"]),
            attempt_index=attempt if phase == "PRE" else attempt + 6,
            parent_candidate_id=self._parent_candidate(phase),
            retrieval_records=tuple(record.to_dict() for record in retrieval.records),
            retrieval_digest=retrieval.evidence_digest,
            model_digest=self.base_model_digest,
            adapter_digest=self.adapter_digest,
            prompt_digest=digest_for("pending-shock-prompt"),
            task_statement="Repair the bounded {} task at {}.".format(
                self.task_record["family_id"], self.task_record["public_locus"]
            ),
            initial_source=self.initial_source.decode("utf-8"),
            initial_source_digest=digest_bytes(self.initial_source),
            response_contract=self.generator.response_contract,
            response_contract_digest=self.response_contract_digest,
            generation_profile_digest=self.generation_profile_digest,
        )
        try:
            prompt_digest = self.generator.prompt_digest_for(provisional)
        except Exception as exc:
            raise HeldoutProtocolError("shock prompt bytes cannot be reproduced") from exc
        return replace(provisional, prompt_digest=prompt_digest)

    def pre_correction_attempt(self, attempt: int, *, idempotency_key: str) -> ShockAttemptObservation:
        return self._attempt("PRE", attempt, idempotency_key)

    def post_correction_attempt(self, attempt: int, *, idempotency_key: str) -> ShockAttemptObservation:
        return self._attempt("POST", attempt, idempotency_key)

    def _attempt(self, phase: str, attempt: int, idempotency_key: str) -> ShockAttemptObservation:
        self._validate_bindings()
        self._assert_attempt_order(phase, attempt)
        expected_operation = self._operation_id(phase, attempt)
        if idempotency_key != expected_operation:
            raise HeldoutProtocolError("shock attempt idempotency key differs from frozen coordinate")
        existing = self.operation_store.load(idempotency_key)
        if existing is not None:
            observation = self._recover_operation(existing)
            if observation is None:
                raise HeldoutProtocolError("shock attempt cannot be safely regenerated")
            return observation
        run_id = self._pre_run_id() if phase == "PRE" else self._post_run_id()
        self._ensure_run(run_id)
        context = self._context(phase, attempt, run_id)
        candidate_id = self._candidate_id(phase, attempt)
        state, created = self.operation_store.begin(
            operation_id=idempotency_key,
            phase=phase,
            attempt=attempt,
            candidate_id=candidate_id,
            run_id=run_id,
            context_digest=_context_digest(context),
        )
        if not created:
            observation = self._recover_operation(state)
            if observation is None:
                raise HeldoutProtocolError("shock attempt cannot be safely regenerated")
            return observation
        self.private_store.record_generation_start(candidate_id=candidate_id, context=context)
        try:
            evidence = self.generator.propose_with_evidence(context)
        except CandidateGenerationFailure as exc:
            evidence_digest = self.private_store.record_generation_failure(
                candidate_id=candidate_id,
                context=context,
                evidence=exc.evidence,
            )
            return self._finish_generation_failure(
                self.operation_store.load(idempotency_key) or {},
                context,
                exc.evidence,
                evidence_digest,
            )
        evidence_digest = self.private_store.record_generation_success(
            candidate_id=candidate_id,
            context=context,
            evidence=evidence,
        )
        self.operation_store.generated(
            idempotency_key,
            evidence_digest=evidence_digest,
            source_digest=digest_bytes(evidence.proposal.source),
        )
        return self._finish_operation(
            self.operation_store.load(idempotency_key) or {}, context, evidence, evidence_digest
        )

    def reconcile_attempt(self, idempotency_key: str) -> Optional[ShockAttemptObservation]:
        expected = {
            self._operation_id(phase, attempt)
            for phase in ("PRE", "POST")
            for attempt in range(1, 7)
        }
        if idempotency_key not in expected:
            raise HeldoutProtocolError("shock reconciliation requested an unknown operation")
        state = self.operation_store.load(idempotency_key)
        if state is None:
            return None
        return self._recover_operation(state)

    def _recover_operation(self, state: Mapping[str, Any]) -> Optional[ShockAttemptObservation]:
        expected_run = self._pre_run_id() if state["phase"] == "PRE" else self._post_run_id()
        if (
            state["operation_id"] != self._operation_id(str(state["phase"]), int(state["attempt"]))
            or state["candidate_id"]
            != self._candidate_id(str(state["phase"]), int(state["attempt"]))
            or state["run_id"] != expected_run
        ):
            raise HeldoutProtocolError("shock operation crossed its frozen attempt identity")
        candidate_id = str(state["candidate_id"])
        if state["status"] == "GENERATION_FAILED":
            try:
                context, failure, evidence_digest = (
                    self.private_store.load_generation_failure_read_only(candidate_id)
                )
            except Exception as exc:
                raise HeldoutProtocolError(
                    "terminal shock generation failure lacks exact private replay evidence"
                ) from exc
            return self._validate_generation_failure_operation(
                state, context, failure, evidence_digest
            )
        try:
            if state["status"] in {"GENERATED", "COMPLETE"}:
                context, evidence, evidence_digest = (
                    self.private_store.load_generation_success_read_only(candidate_id)
                )
            else:
                context, evidence, evidence_digest = (
                    self.private_store.load_generation_success(candidate_id)
                )
        except Exception as success_exc:
            if state["status"] != "STARTED":
                raise HeldoutProtocolError(
                    "shock model call has no recoverable generation result; duplicate generation is forbidden"
                ) from success_exc
            try:
                context, failure, evidence_digest = self.private_store.load_generation_failure(
                    candidate_id
                )
            except Exception as failure_exc:
                raise HeldoutProtocolError(
                    "shock model call has no recoverable generation result; duplicate generation is forbidden"
                ) from failure_exc
            if _context_digest(context) != state["context_digest"]:
                raise HeldoutProtocolError("recovered shock generation failure crossed its exact context")
            return self._finish_generation_failure(state, context, failure, evidence_digest)
        if _context_digest(context) != state["context_digest"]:
            raise HeldoutProtocolError("recovered shock generation crossed its exact context")
        if state["status"] == "STARTED":
            self.operation_store.generated(
                state["operation_id"],
                evidence_digest=evidence_digest,
                source_digest=digest_bytes(evidence.proposal.source),
            )
            state = self.operation_store.load(str(state["operation_id"])) or state
        if (
            state["generation_evidence_digest"] != evidence_digest
            or state["candidate_source_digest"] != digest_bytes(evidence.proposal.source)
        ):
            raise HeldoutProtocolError("recovered shock generation differs from durable operation state")
        if state["status"] == "COMPLETE":
            return self._validate_complete_operation(state, context, evidence, evidence_digest)
        return self._finish_operation(state, context, evidence, evidence_digest)

    def _lineage_parent(self, state: Mapping[str, Any], context: CandidateContext) -> str:
        if context.parent_candidate_id is not None:
            return context.parent_candidate_id
        if state["phase"] == "PRE":
            return str(self._premise_event()["event_id"])
        correction = self._correction_receipt(required=True)
        assert correction is not None
        return str(self._replacement_event()["event_id"])

    def _candidate_payload(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
        evidence: CandidateGenerationEvidence,
        evidence_digest: str,
    ) -> Dict[str, Any]:
        proposal = evidence.proposal
        metadata = {
            "schema_version": "egv-production-shock-candidate-v1",
            "arm_id": self.arm_id,
            "phase": state["phase"],
            "attempt_index": context.attempt_index,
            "public_rule_id": context.public_rule_id,
            "public_locus": context.public_locus,
            "retrieval_digest": context.retrieval_digest,
            "evidence_ids": list(proposal.evidence_ids),
            "candidate_artifact_digest": digest_bytes(proposal.source),
            "response_contract": context.response_contract,
            "response_contract_digest": context.response_contract_digest,
            "generation_evidence_digest": evidence_digest,
            "generation_profile_digest": context.generation_profile_digest,
            "shock_operation_id": state["operation_id"],
        }
        return {
            "candidate_id": str(state["candidate_id"]),
            "campaign_id": self.campaign_id,
            "run_id": str(state["run_id"]),
            "task_id": self.coordinate.task_id,
            "parent_candidate_id": context.parent_candidate_id,
            "mutation_family": str(self.task_record["family_id"]),
            "patch_hash": proposal.mutation_digest,
            "requested_authority": proposal.requested_authority,
            "prompt_hash": context.prompt_digest,
            "model_hash": self.base_model_digest,
            "adapter_hash": self.adapter_digest,
            "metadata": metadata,
        }

    def _candidate_dependencies(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
        proposal: CandidateProposal,
    ) -> set[tuple[str, str]]:
        expected = {(item, "EVIDENCE_USED") for item in proposal.evidence_ids}
        expected.add((self._lineage_parent(state, context), "SHOCK_LINEAGE"))
        return expected

    def _persist_candidate(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
        evidence: CandidateGenerationEvidence,
        evidence_digest: str,
    ) -> None:
        proposal = evidence.proposal
        candidate_id = str(state["candidate_id"])
        if tuple(proposal.evidence_ids) != tuple(
            sorted(str(item["event_id"]) for item in context.retrieval_records)
        ):
            raise HeldoutProtocolError("shock source-only proposal omitted or substituted presented evidence")
        payload = self._candidate_payload(state, context, evidence, evidence_digest)
        self.ledger.append_candidate(
            candidate_id,
            campaign_id=str(payload["campaign_id"]),
            run_id=str(payload["run_id"]),
            task_id=str(payload["task_id"]),
            parent_candidate_id=payload["parent_candidate_id"],
            mutation_family=str(payload["mutation_family"]),
            patch_hash=str(payload["patch_hash"]),
            requested_authority=str(payload["requested_authority"]),
            prompt_hash=str(payload["prompt_hash"]),
            model_hash=str(payload["model_hash"]),
            adapter_hash=payload["adapter_hash"],
            metadata=payload["metadata"],
        )
        self.private_store.record(candidate_id=candidate_id, context=context, source=proposal.source)
        expected = self._candidate_dependencies(state, context, proposal)
        existing = {
            (str(row["parent_id"]), str(row["edge_type"]))
            for row in self.ledger.connection.execute(
                "SELECT parent_id,edge_type FROM dependencies WHERE child_id=?", (candidate_id,)
            ).fetchall()
        }
        if not existing.issubset(expected):
            raise HeldoutProtocolError("shock candidate carries an unexpected dependency")
        for parent, edge_type in sorted(expected):
            self.ledger.append_dependency(
                parent,
                candidate_id,
                edge_type=edge_type,
                campaign_id=self.campaign_id,
                run_id=str(state["run_id"]),
                task_id=self.coordinate.task_id,
                idempotency_key="shock-dependency:{}:{}:{}".format(parent, candidate_id, edge_type),
            )
        final = {
            (str(row["parent_id"]), str(row["edge_type"]))
            for row in self.ledger.connection.execute(
                "SELECT parent_id,edge_type FROM dependencies WHERE child_id=?", (candidate_id,)
            ).fetchall()
        }
        if final != expected:
            raise HeldoutProtocolError("shock candidate dependency set is incomplete")

    def _generation_failure_payload(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
        evidence: CandidateGenerationFailureEvidence,
        evidence_digest: str,
    ) -> Dict[str, Any]:
        evidence.validate(context)
        raw_digests = [
            value
            for value in (
                evidence.rendered_prompt_digest,
                evidence.decoded_model_response_digest,
                evidence.contract_response_digest,
            )
            if value is not None
        ]
        return {
            "schema_version": SHOCK_GENERATION_FAILURE_SCHEMA,
            "block_id": self.coordinate.block_id if state["phase"] == "PRE" else None,
            "coordinate_id": self.coordinate.coordinate_id if state["phase"] == "POST" else None,
            "treatment": self.coordinate.treatment if state["phase"] == "POST" else None,
            "operation_id": state["operation_id"],
            "phase": state["phase"],
            "attempt": state["attempt"],
            "candidate_id": state["candidate_id"],
            "run_id": state["run_id"],
            "context_digest": _context_digest(context),
            "generation_evidence_digest": _require_digest(
                evidence_digest, "generation failure evidence digest"
            ),
            "failure_stage": evidence.stage,
            "diagnostic_enum": _failure_diagnostic(evidence),
            "response_contract_digest": context.response_contract_digest,
            "generation_profile_digest": context.generation_profile_digest,
            "raw_artifact_count": len(raw_digests),
            "raw_artifact_digest_root": digest_for(raw_digests),
            "error_code_digest": digest_for({"error_code": evidence.error_code}),
        }

    def _assert_generation_failure_has_no_external_effect(self, candidate_id: str) -> None:
        tables = ("candidates", "receipts", "verdicts", "effect_receipts")
        for table in tables:
            count = self.ledger.connection.execute(
                "SELECT COUNT(*) FROM {} WHERE candidate_id=?".format(table),
                (candidate_id,),
            ).fetchone()[0]
            if int(count) != 0:
                raise HeldoutProtocolError(
                    "terminal shock generation failure materialized candidate or evaluator effects"
                )
        attempt_events = self.ledger.connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='SHOCK_ATTEMPT' AND subject_id=?",
            (candidate_id,),
        ).fetchone()[0]
        if int(attempt_events) != 0:
            raise HeldoutProtocolError(
                "terminal shock generation failure materialized an evaluated attempt"
            )

    def _expected_generation_failure_dependencies(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
    ) -> set[tuple[str, str]]:
        expected = {
            (str(item["event_id"]), "EVIDENCE_USED") for item in context.retrieval_records
        }
        expected.add((self._lineage_parent(state, context), "SHOCK_LINEAGE"))
        return expected

    @staticmethod
    def _generation_failure_dependency_idempotency(
        parent_id: str,
        child_id: str,
        edge_type: str,
    ) -> str:
        return "shock-failure-dependency:{}:{}:{}".format(
            parent_id, child_id, edge_type
        )

    def _validate_generation_failure_dependency_bindings(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
    ) -> tuple[set[str], set[str]]:
        candidate_id = str(state["candidate_id"])
        expected = self._expected_generation_failure_dependencies(state, context)
        rows = tuple(
            dict(row)
            for row in self.ledger.connection.execute(
                "SELECT dependency_id,parent_id,child_id,edge_type,insertion_event_id "
                "FROM dependencies WHERE child_id=?",
                (candidate_id,),
            ).fetchall()
        )
        actual_edges = {
            (str(row["parent_id"]), str(row["edge_type"])) for row in rows
        }
        if len(rows) != len(expected) or actual_edges != expected:
            raise HeldoutProtocolError(
                "shock generation failure dependency set is incomplete"
            )

        dependency_ids = set()
        insertion_event_ids = set()
        for row in rows:
            parent_id = str(row["parent_id"])
            child_id = str(row["child_id"])
            edge_type = str(row["edge_type"])
            payload = {
                "parent_id": parent_id,
                "child_id": child_id,
                "edge_type": edge_type,
            }
            expected_dependency_id = content_id("dep", payload)
            expected_idempotency = self._generation_failure_dependency_idempotency(
                parent_id, child_id, edge_type
            )
            try:
                insertion_event = self._event(str(row["insertion_event_id"]))
            except Exception as exc:
                raise HeldoutProtocolError(
                    "shock generation failure dependency lacks its insertion event"
                ) from exc
            if (
                row["dependency_id"] != expected_dependency_id
                or child_id != candidate_id
                or insertion_event.get("event_type") != "DEPENDENCY"
                or insertion_event.get("campaign_id") != self.campaign_id
                or insertion_event.get("run_id") != state["run_id"]
                or insertion_event.get("task_id") != self.coordinate.task_id
                or insertion_event.get("subject_id") != candidate_id
                or insertion_event.get("payload") != payload
                or insertion_event.get("source_class") != "FROZEN_PROTOCOL"
                or insertion_event.get("disposition") != "OBSERVED"
                or insertion_event.get("idempotency_key") != expected_idempotency
            ):
                raise HeldoutProtocolError(
                    "shock generation failure dependency insertion event binding differs"
                )
            dependency_ids.add(expected_dependency_id)
            insertion_event_ids.add(str(insertion_event["event_id"]))
        if len(dependency_ids) != len(expected) or len(insertion_event_ids) != len(expected):
            raise HeldoutProtocolError(
                "shock generation failure dependency inventory is not one-to-one"
            )
        return dependency_ids, insertion_event_ids

    def _assert_generation_failure_event_binding(
        self,
        event: Mapping[str, Any],
        *,
        state: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        candidate_id = str(state["candidate_id"])
        if (
            event.get("event_type") != "SHOCK_GENERATION_FAILURE"
            or event.get("campaign_id") != self.campaign_id
            or event.get("run_id") != state["run_id"]
            or event.get("task_id") != self.coordinate.task_id
            or event.get("subject_id") != candidate_id
            or event.get("payload") != dict(payload)
            or event.get("source_class") != "PINNED_MODEL"
            or event.get("disposition") != "REJECTED"
            or event.get("idempotency_key")
            != "shock-generation-failure:" + str(state["operation_id"])
        ):
            raise HeldoutProtocolError(
                "shock generation failure event crossed its exact attempt binding"
            )

    def _persist_generation_failure(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
        evidence: CandidateGenerationFailureEvidence,
        evidence_digest: str,
    ) -> Mapping[str, Any]:
        candidate_id = str(state["candidate_id"])
        self._assert_generation_failure_has_no_external_effect(candidate_id)
        payload = self._generation_failure_payload(state, context, evidence, evidence_digest)
        event = self.ledger.append_event(
            "SHOCK_GENERATION_FAILURE",
            payload,
            campaign_id=self.campaign_id,
            run_id=str(state["run_id"]),
            task_id=self.coordinate.task_id,
            subject_id=candidate_id,
            source_class="PINNED_MODEL",
            disposition="REJECTED",
            idempotency_key="shock-generation-failure:" + str(state["operation_id"]),
        )
        self._assert_generation_failure_event_binding(event, state=state, payload=payload)
        expected_dependencies = self._expected_generation_failure_dependencies(state, context)
        existing = {
            (str(row["parent_id"]), str(row["edge_type"]))
            for row in self.ledger.connection.execute(
                "SELECT parent_id,edge_type FROM dependencies WHERE child_id=?",
                (candidate_id,),
            ).fetchall()
        }
        if not existing.issubset(expected_dependencies):
            raise HeldoutProtocolError("shock generation failure carries an unexpected dependency")
        for parent, edge_type in sorted(expected_dependencies):
            self.ledger.append_dependency(
                parent,
                candidate_id,
                edge_type=edge_type,
                campaign_id=self.campaign_id,
                run_id=str(state["run_id"]),
                task_id=self.coordinate.task_id,
                idempotency_key=self._generation_failure_dependency_idempotency(
                    parent, candidate_id, edge_type
                ),
            )
        self._validate_generation_failure_dependency_bindings(state, context)
        self._assert_generation_failure_has_no_external_effect(candidate_id)
        return event

    def _validate_generation_failure_inventory(self) -> None:
        operations = self.operation_store.records()
        failed = tuple(
            state for state in operations if state["status"] == "GENERATION_FAILED"
        )
        ledger_events = tuple(self.ledger.events())

        def is_generation_failure_event(event: Mapping[str, Any]) -> bool:
            payload = event.get("payload")
            return (
                event.get("event_type") == "SHOCK_GENERATION_FAILURE"
                or str(event.get("idempotency_key", "")).startswith(
                    "shock-generation-failure:"
                )
                or (
                    isinstance(payload, Mapping)
                    and payload.get("schema_version")
                    == SHOCK_GENERATION_FAILURE_SCHEMA
                )
            )

        failure_events = tuple(
            event
            for event in ledger_events
            if is_generation_failure_event(event)
        )
        failed_operation_ids = {str(state["operation_id"]) for state in failed}
        failed_subjects = {str(state["candidate_id"]) for state in failed}
        if len(failed_operation_ids) != len(failed) or len(failed_subjects) != len(failed):
            raise HeldoutProtocolError(
                "terminal shock generation failure operation inventory is not one-to-one"
            )

        expected_event_ids = set()
        expected_dependency_ids = set()
        expected_dependency_event_ids = set()
        for state in failed:
            candidate_id = str(state["candidate_id"])
            try:
                context, evidence, evidence_digest = (
                    self.private_store.load_generation_failure_read_only(candidate_id)
                )
            except Exception as exc:
                raise HeldoutProtocolError(
                    "terminal shock generation failure inventory lacks private evidence"
                ) from exc
            if (
                state["generation_evidence_digest"] != evidence_digest
                or state["candidate_source_digest"] is not None
                or state["evaluation_result"] is not None
                or _context_digest(context) != state["context_digest"]
            ):
                raise HeldoutProtocolError(
                    "terminal shock generation failure inventory differs from durable evidence"
                )
            payload = self._generation_failure_payload(
                state, context, evidence, evidence_digest
            )
            idempotency_key = "shock-generation-failure:" + str(state["operation_id"])
            matching = tuple(
                event
                for event in failure_events
                if event.get("idempotency_key") == idempotency_key
            )
            if len(matching) != 1:
                raise HeldoutProtocolError(
                    "terminal shock generation failure lacks exactly one bound event"
                )
            event = matching[0]
            self._assert_generation_failure_event_binding(
                event, state=state, payload=payload
            )
            dependency_ids, dependency_event_ids = (
                self._validate_generation_failure_dependency_bindings(state, context)
            )
            expected_dependency_ids.update(dependency_ids)
            expected_dependency_event_ids.update(dependency_event_ids)
            expected_event_ids.add(str(event["event_id"]))

        actual_event_ids = {str(event["event_id"]) for event in failure_events}
        actual_subjects = {str(event.get("subject_id")) for event in failure_events}
        actual_operation_bindings = {
            str(event.get("idempotency_key", "")).removeprefix(
                "shock-generation-failure:"
            )
            for event in failure_events
        }
        if (
            len(failure_events) != len(failed)
            or actual_event_ids != expected_event_ids
            or actual_subjects != failed_subjects
            or actual_operation_bindings != failed_operation_ids
        ):
            raise HeldoutProtocolError(
                "shock generation failure event inventory differs from failed operations"
            )

        def is_relevant_dependency_event(event: Mapping[str, Any]) -> bool:
            payload = event.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            reserved = str(event.get("idempotency_key", "")).startswith(
                "shock-failure-dependency:"
            )
            dependency_shaped = (
                event.get("event_type") == "DEPENDENCY"
                or reserved
                or any(key in payload for key in ("parent_id", "child_id", "edge_type"))
            )
            return dependency_shaped and (
                event.get("subject_id") in failed_subjects
                or payload.get("parent_id") in failed_subjects
                or payload.get("child_id") in failed_subjects
                or reserved
            )

        relevant_dependency_events = tuple(
            event
            for event in ledger_events
            if is_relevant_dependency_event(event)
        )
        relevant_dependency_event_ids = {
            str(event["event_id"]) for event in relevant_dependency_events
        }
        dependency_rows = tuple(
            dict(row)
            for row in self.ledger.connection.execute(
                "SELECT dependency_id,parent_id,child_id,edge_type,insertion_event_id "
                "FROM dependencies"
            ).fetchall()
        )
        relevant_dependency_rows = tuple(
            row
            for row in dependency_rows
            if str(row["parent_id"]) in failed_subjects
            or str(row["child_id"]) in failed_subjects
            or str(row["insertion_event_id"]) in relevant_dependency_event_ids
        )
        actual_dependency_ids = {
            str(row["dependency_id"]) for row in relevant_dependency_rows
        }
        actual_dependency_insertion_ids = {
            str(row["insertion_event_id"]) for row in relevant_dependency_rows
        }
        if (
            len(relevant_dependency_rows) != len(expected_dependency_ids)
            or len(relevant_dependency_events) != len(expected_dependency_event_ids)
            or actual_dependency_ids != expected_dependency_ids
            or actual_dependency_insertion_ids != expected_dependency_event_ids
            or relevant_dependency_event_ids != expected_dependency_event_ids
        ):
            raise HeldoutProtocolError(
                "shock generation failure dependency event inventory differs"
            )

    def _generation_failure_observation(
        self,
        state: Mapping[str, Any],
        evidence: CandidateGenerationFailureEvidence,
    ) -> ShockAttemptObservation:
        observation = ShockAttemptObservation(
            promoted=False,
            diagnostic_enum=_failure_diagnostic(evidence),
            receipt_valid=False,
            tokens=0,
            evaluator_seconds=0.0,
            evidence_used=False,
            authority_challenge=False,
            valid_authority_denial=False,
            promoted_node_ids=tuple(),
            independent_hidden_fixture_passed=False,
            verdict_receipt_digest=None,
            effect_receipt_digest=None,
            operation_id=str(state["operation_id"]),
        )
        _validate_failure_observation(observation)
        return observation

    def _finish_generation_failure(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
        evidence: CandidateGenerationFailureEvidence,
        evidence_digest: str,
    ) -> ShockAttemptObservation:
        if state["status"] != "STARTED" or _context_digest(context) != state["context_digest"]:
            raise HeldoutProtocolError("shock generation failure crossed its durable start state")
        self._persist_generation_failure(state, context, evidence, evidence_digest)
        observation = self._generation_failure_observation(state, evidence)
        self.operation_store.generation_failed(
            str(state["operation_id"]),
            evidence_digest=evidence_digest,
            observation=observation,
            rng_state_evidence=self._rng_evidence(),
        )
        self._validate_generation_failure_inventory()
        return observation

    def _validate_generation_failure_operation(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
        evidence: CandidateGenerationFailureEvidence,
        evidence_digest: str,
    ) -> ShockAttemptObservation:
        if (
            state["status"] != "GENERATION_FAILED"
            or state["generation_evidence_digest"] != evidence_digest
            or state["candidate_source_digest"] is not None
            or state["evaluation_result"] is not None
            or _context_digest(context) != state["context_digest"]
        ):
            raise HeldoutProtocolError("terminal shock generation failure differs from private replay")
        rebuilt = self._generation_failure_observation(state, evidence)
        stored = ShockAttemptObservation.from_mapping(state["observation"])
        _validate_failure_observation(stored)
        if rebuilt != stored:
            raise HeldoutProtocolError("terminal shock generation failure observation differs from replay")
        self._assert_generation_failure_has_no_external_effect(str(state["candidate_id"]))
        self._validate_generation_failure_inventory()
        return stored

    def _receipt_suffix(self, result: EvaluationResult, *, run_id: str) -> Tuple[Mapping[str, Any], ...]:
        receipts = []
        for receipt_id in result.receipt_ids:
            row = self.ledger.receipt_by_id(receipt_id)
            if row is None or not isinstance(row.get("receipt"), Mapping):
                raise HeldoutProtocolError("shock evaluation references a missing signed receipt")
            receipts.append(dict(row["receipt"]))
        if [item.get("receipt_type") for item in receipts] != ["AUTHORITY", "VERDICT", "EFFECT"]:
            raise HeldoutProtocolError("shock evaluation lacks exact authority/verdict/effect receipts")
        artifact = result.candidate_artifact_digest
        for index, receipt in enumerate(receipts):
            expected_type = ("AUTHORITY", "VERDICT", "EFFECT")[index]
            if (
                receipt.get("campaign_id") != self.campaign_id
                or receipt.get("run_id") != run_id
                or receipt.get("task_id") != self.coordinate.task_id
                or receipt.get("candidate_id") != result.candidate_id
                or receipt.get("candidate_artifact_digest") != artifact
                or receipt.get("protocol_digest") != VARIATION_PROTOCOL_DIGEST
                or receipt.get("policy_digest") != AuthorityPolicy.candidate_execution().digest
                or receipt.get("arm_policy_digest") != self.policy.digest
                or receipt.get("evaluator_digest") != self.evaluator.evaluator_digest
                or receipt.get("task_family") != self.task_record["family_id"]
                or receipt.get("normalized_public_locus") != self.task_record["public_locus"]
                or receipt.get("public_rule_id") != self.task_record["public_rule_id"]
                or receipt.get("request_id") != "request-{}-{}".format(expected_type.lower(), result.candidate_id)
            ):
                raise HeldoutProtocolError("shock receipt suffix crossed its exact public binding")
            if index and (
                receipt.get("sequence") != receipts[index - 1].get("sequence") + 1
                or receipt.get("previous_receipt_hash") != receipt_hash(receipts[index - 1])
            ):
                raise HeldoutProtocolError("shock receipt suffix is not contiguous")
        authority, verdict, effect = receipts
        infrastructure = result.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value
        expected_verdict = "ERROR" if infrastructure else (
            "PASS" if result.diagnostic_enum == Diagnostic.PASS.value else "FAIL"
        )
        expected_disposition = "ABSTAINED" if infrastructure else (
            "PROMOTED" if result.diagnostic_enum == Diagnostic.PASS.value else "REJECTED"
        )
        if (
            authority.get("decision") != "ALLOW"
            or verdict.get("decision") != expected_verdict
            or effect.get("decision") != ("ERROR" if infrastructure else "ALLOW")
            or verdict.get("diagnostic_enum") != result.diagnostic_enum
            or effect.get("diagnostic_enum") != result.diagnostic_enum
            or verdict.get("resource_bucket") != result.resource_bucket
            or verdict.get("output_digest") != result.output_digest
            or effect.get("output_digest") != result.output_digest
            or effect.get("normalized_action_hash")
            != digest_for({"action": "execute_candidate", "locus": self.task_record["public_locus"]})
            or result.disposition != expected_disposition
        ):
            raise HeldoutProtocolError("shock result disagrees with its signed receipt suffix")
        self.ledger.verify_receipt_chain(self.evaluator_public_key)
        return tuple(receipts)

    def _materialize_result(
        self,
        state: Mapping[str, Any],
        proposal: CandidateProposal,
        result: EvaluationResult,
    ) -> Tuple[Mapping[str, Any], ...]:
        if (
            result.candidate_id != state["candidate_id"]
            or result.task_id != self.coordinate.task_id
            or result.candidate_artifact_digest != digest_bytes(proposal.source)
        ):
            raise HeldoutProtocolError("shock evaluator result crossed candidate/task/source")
        try:
            validate_diagnostic(result.diagnostic_enum)
            validate_resource_bucket(result.resource_bucket)
            validate_disposition(result.disposition)
        except ValueError as exc:
            raise HeldoutProtocolError("shock result is outside the closed evaluator vocabulary") from exc
        receipts = self._receipt_suffix(result, run_id=str(state["run_id"]))
        verdict = receipts[1]
        effect = receipts[2]
        self.ledger.append_verdict(
            content_id(
                "shock-verdict",
                {"operation_id": state["operation_id"], "receipt_id": verdict["receipt_id"]},
            ),
            candidate_id=str(state["candidate_id"]),
            correctness=result.diagnostic_enum == Diagnostic.PASS.value,
            performance={"resource_bucket": result.resource_bucket},
            hidden_test_set_hash=digest_for(
                {"evaluator": self.evaluator.evaluator_digest, "task": self.coordinate.task_id}
            ),
            evaluator_revision=self.evaluator.evaluator_revision,
            receipt_id=str(verdict["receipt_id"]),
            signed_receipt_hash=receipt_hash(verdict),
        )
        self.ledger.append_effect_receipt(
            str(effect["request_id"]),
            candidate_id=str(state["candidate_id"]),
            identity=str(effect.get("identity", "remote-frozen-evaluator")),
            normalized_action_hash=str(effect["normalized_action_hash"]),
            decision=str(effect["decision"]),
            policy_hash=str(effect["policy_digest"]),
            sandbox_id=str(effect["sandbox_id"]),
            started_at=str(effect["started_at"]),
            finished_at=str(effect["finished_at"]),
            exit_status_class=str(effect["exit_status_class"]),
            output_hash=effect.get("output_digest"),
            environment_diff_hash=effect.get("environment_diff_digest"),
            signature=str(effect["signature"]),
            receipt_id=str(effect["receipt_id"]),
        )
        self.ledger.append_event(
            "SHOCK_ATTEMPT",
            {
                "schema_version": SHOCK_ATTEMPT_EVENT_SCHEMA,
                "operation_id": state["operation_id"],
                "phase": state["phase"],
                "attempt": state["attempt"],
                "candidate_id": state["candidate_id"],
                "candidate_artifact_digest": result.candidate_artifact_digest,
                "diagnostic_enum": result.diagnostic_enum,
                "resource_bucket": result.resource_bucket,
                "disposition": result.disposition,
                "receipt_ids": list(result.receipt_ids),
            },
            campaign_id=self.campaign_id,
            run_id=str(state["run_id"]),
            task_id=self.coordinate.task_id,
            subject_id=str(state["candidate_id"]),
            source_class="FROZEN_EVALUATOR",
            disposition="VERIFIED",
            idempotency_key="shock-attempt:" + str(state["operation_id"]),
        )
        materialized = self.ledger.candidate_disposition(str(state["candidate_id"]))
        stale_naive_reuse = (
            state["phase"] == "POST"
            and self.coordinate.treatment == "naive-reuse"
            and materialized == "STALE_DEPENDENT"
        )
        corrected_pre_history = (
            state["phase"] == "PRE"
            and materialized == "STALE_DEPENDENT"
            and self._correction_receipt(required=False) is not None
        )
        if (
            materialized != result.disposition
            and not stale_naive_reuse
            and not corrected_pre_history
        ):
            raise HeldoutProtocolError("materialized shock disposition differs from signed evaluator result")
        return receipts

    @staticmethod
    def _signed_seconds(effect: Mapping[str, Any]) -> float:
        try:
            started = datetime.fromisoformat(str(effect["started_at"]).replace("Z", "+00:00"))
            finished = datetime.fromisoformat(str(effect["finished_at"]).replace("Z", "+00:00"))
            seconds = (finished - started).total_seconds()
        except (KeyError, TypeError, ValueError) as exc:
            raise HeldoutProtocolError("signed evaluator receipt lacks parseable timing") from exc
        if seconds < 0:
            raise HeldoutProtocolError("signed evaluator timing is negative")
        return float(seconds)

    def _token_count(self, source: bytes) -> int:
        try:
            encoded = self.generator.tokenizer(source.decode("utf-8"), add_special_tokens=False)
            ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
            if hasattr(ids, "shape"):
                count = int(ids.shape[-1])
            elif isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
                count = len(ids[0])
            else:
                count = len(ids)
        except Exception as exc:
            raise HeldoutProtocolError("pinned tokenizer cannot measure shock candidate tokens") from exc
        if count <= 0:
            raise HeldoutProtocolError("pinned tokenizer returned an empty shock candidate")
        return count

    def _rng_evidence(self) -> Mapping[str, Any]:
        try:
            import torch

            cpu_state = torch.get_rng_state().detach().cpu().numpy().tobytes()
            cuda_states = []
            if torch.cuda.is_available():
                cuda_states = [
                    digest_bytes(state.detach().cpu().numpy().tobytes())
                    for state in torch.cuda.get_rng_state_all()
                ]
        except Exception as exc:
            raise HeldoutProtocolError("actual torch RNG boundary cannot be captured") from exc
        return {
            "schema_version": "egv-shock-rng-evidence-v1",
            "generation_mode": "deterministic-greedy-v1",
            "do_sample": False,
            "num_beams": 1,
            "torch_cpu_state_digest": digest_bytes(cpu_state),
            "torch_cuda_state_digests": cuda_states,
            "generation_profile_digest": self.generation_profile_digest,
        }

    def _observation(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
        proposal: CandidateProposal,
        result: EvaluationResult,
        receipts: Sequence[Mapping[str, Any]],
    ) -> ShockAttemptObservation:
        materialized = self.ledger.candidate_disposition(str(state["candidate_id"]))
        hidden_fixture_passed = (
            result.diagnostic_enum == Diagnostic.PASS.value and result.disposition == "PROMOTED"
        )
        corrected_pre_history = (
            state["phase"] == "PRE"
            and materialized == "STALE_DEPENDENT"
            and self._correction_receipt(required=False) is not None
        )
        promoted = hidden_fixture_passed and (
            materialized == "PROMOTED" or corrected_pre_history
        )
        authority_denied = receipts[0].get("decision") == "DENY"
        observation = ShockAttemptObservation(
            promoted=promoted,
            diagnostic_enum=result.diagnostic_enum,
            receipt_valid=True,
            tokens=self._token_count(proposal.source),
            evaluator_seconds=self._signed_seconds(receipts[2]),
            evidence_used=bool(proposal.evidence_ids),
            authority_challenge=authority_denied,
            valid_authority_denial=bool(
                authority_denied and receipts[2].get("decision") == "DENY" and not promoted
            ),
            promoted_node_ids=(str(state["candidate_id"]),) if promoted else tuple(),
            independent_hidden_fixture_passed=bool(
                hidden_fixture_passed and state["phase"] == "POST"
            ),
            verdict_receipt_digest=receipt_hash(receipts[1]),
            effect_receipt_digest=receipt_hash(receipts[2]) if promoted else None,
            operation_id=str(state["operation_id"]),
        )
        observation.validate()
        return observation

    def _finish_operation(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
        evidence: CandidateGenerationEvidence,
        evidence_digest: str,
    ) -> ShockAttemptObservation:
        self._persist_candidate(state, context, evidence, evidence_digest)
        before = tuple(self.ledger.receipts())
        prior_candidate_receipts = tuple(
            receipt for receipt in before
            if receipt.get("candidate_id") == state["candidate_id"]
        )
        if len(prior_candidate_receipts) not in {0, 3}:
            raise HeldoutProtocolError("pending shock evaluation has a partial receipt suffix")
        result = self.evaluator.evaluate(
            candidate_id=str(state["candidate_id"]),
            task_id=self.coordinate.task_id,
            source=evidence.proposal.source,
            opaque_input=None,
            requested_authority=evidence.proposal.requested_authority,
            declared_locus=evidence.proposal.declared_locus,
        )
        after_first = tuple(self.ledger.receipts())
        replay = self.evaluator.evaluate(
            candidate_id=str(state["candidate_id"]),
            task_id=self.coordinate.task_id,
            source=evidence.proposal.source,
            opaque_input=None,
            requested_authority=evidence.proposal.requested_authority,
            declared_locus=evidence.proposal.declared_locus,
        )
        if replay.to_dict() != result.to_dict() or tuple(self.ledger.receipts()) != after_first:
            raise HeldoutProtocolError("remote hidden evaluator replay duplicated or changed its effect")
        expected_added = 3 if not prior_candidate_receipts else 0
        if len(after_first) - len(before) != expected_added:
            raise HeldoutProtocolError("shock evaluator did not append one exact signed receipt suffix")
        candidate_receipts = tuple(
            receipt for receipt in after_first
            if receipt.get("candidate_id") == state["candidate_id"]
        )
        if len(candidate_receipts) != 3:
            raise HeldoutProtocolError("shock evaluator recovery lacks one exact receipt suffix")
        receipts = self._materialize_result(state, evidence.proposal, result)
        observation = self._observation(state, context, evidence.proposal, result, receipts)
        rng_evidence = self._rng_evidence()
        self.operation_store.complete(
            str(state["operation_id"]),
            result=result,
            observation=observation,
            rng_state_evidence=rng_evidence,
        )
        return observation

    def _validate_complete_candidate_projection(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
        evidence: CandidateGenerationEvidence,
        evidence_digest: str,
    ) -> None:
        candidate_id = str(state["candidate_id"])
        proposal = evidence.proposal
        expected_evidence_ids = tuple(
            sorted(str(item["event_id"]) for item in context.retrieval_records)
        )
        if tuple(proposal.evidence_ids) != expected_evidence_ids:
            raise HeldoutProtocolError(
                "terminal shock candidate omitted or substituted presented evidence"
            )
        try:
            trajectory_context, trajectory_source, _trajectory_digest = (
                self.private_store.load_trajectory_read_only(candidate_id)
            )
        except Exception as exc:
            raise HeldoutProtocolError(
                "terminal shock candidate lacks its private trajectory"
            ) from exc
        if trajectory_context != context or trajectory_source != proposal.source:
            raise HeldoutProtocolError(
                "terminal shock private trajectory differs from generation replay"
            )

        expected_payload = self._candidate_payload(
            state, context, evidence, evidence_digest
        )
        row = self.ledger.connection.execute(
            "SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)
        ).fetchone()
        scalar_fields = (
            "candidate_id",
            "campaign_id",
            "run_id",
            "task_id",
            "parent_candidate_id",
            "mutation_family",
            "patch_hash",
            "requested_authority",
            "prompt_hash",
            "model_hash",
            "adapter_hash",
        )
        if (
            row is None
            or any(row[field] != expected_payload[field] for field in scalar_fields)
            or row["candidate_json"] != canonical_json(expected_payload)
        ):
            raise HeldoutProtocolError(
                "terminal shock candidate ledger projection is missing or substituted"
            )
        candidate_event = self._event(str(row["event_id"]))
        if (
            candidate_event.get("event_type") != "CANDIDATE"
            or candidate_event.get("payload") != expected_payload
            or candidate_event.get("campaign_id") != self.campaign_id
            or candidate_event.get("run_id") != state["run_id"]
            or candidate_event.get("task_id") != self.coordinate.task_id
            or candidate_event.get("subject_id") != candidate_id
            or candidate_event.get("source_class") is not None
            or candidate_event.get("disposition") != "OBSERVED"
            or candidate_event.get("idempotency_key") is not None
        ):
            raise HeldoutProtocolError(
                "terminal shock candidate event projection is substituted"
            )

        expected_dependencies = self._candidate_dependencies(
            state, context, proposal
        )
        dependency_rows = tuple(
            dict(row)
            for row in self.ledger.connection.execute(
                "SELECT dependency_id,parent_id,child_id,edge_type,insertion_event_id "
                "FROM dependencies WHERE child_id=? ORDER BY dependency_id",
                (candidate_id,),
            ).fetchall()
        )
        actual_dependencies = {
            (str(row["parent_id"]), str(row["edge_type"]))
            for row in dependency_rows
        }
        if (
            len(dependency_rows) != len(expected_dependencies)
            or actual_dependencies != expected_dependencies
        ):
            raise HeldoutProtocolError(
                "terminal shock candidate dependency set is incomplete or substituted"
            )
        for dependency in dependency_rows:
            parent_id = str(dependency["parent_id"])
            edge_type = str(dependency["edge_type"])
            payload = {
                "parent_id": parent_id,
                "child_id": candidate_id,
                "edge_type": edge_type,
            }
            insertion = self._event(str(dependency["insertion_event_id"]))
            if (
                dependency["dependency_id"] != content_id("dep", payload)
                or dependency["child_id"] != candidate_id
                or insertion.get("event_type") != "DEPENDENCY"
                or insertion.get("payload") != payload
                or insertion.get("campaign_id") != self.campaign_id
                or insertion.get("run_id") != state["run_id"]
                or insertion.get("task_id") != self.coordinate.task_id
                or insertion.get("subject_id") != candidate_id
                or insertion.get("source_class") != "FROZEN_PROTOCOL"
                or insertion.get("disposition") != "OBSERVED"
                or insertion.get("idempotency_key")
                != "shock-dependency:{}:{}:{}".format(
                    parent_id, candidate_id, edge_type
                )
            ):
                raise HeldoutProtocolError(
                    "terminal shock candidate dependency event is substituted"
                )

    def _validate_complete_result_projection(
        self,
        state: Mapping[str, Any],
        proposal: CandidateProposal,
        result: EvaluationResult,
    ) -> Tuple[Mapping[str, Any], ...]:
        candidate_id = str(state["candidate_id"])
        if (
            result.candidate_id != candidate_id
            or result.task_id != self.coordinate.task_id
            or result.candidate_artifact_digest != digest_bytes(proposal.source)
        ):
            raise HeldoutProtocolError(
                "terminal shock evaluator result crossed candidate/task/source"
            )
        try:
            validate_diagnostic(result.diagnostic_enum)
            validate_resource_bucket(result.resource_bucket)
            validate_disposition(result.disposition)
        except ValueError as exc:
            raise HeldoutProtocolError(
                "terminal shock result is outside the closed evaluator vocabulary"
            ) from exc
        receipts = self._receipt_suffix(result, run_id=str(state["run_id"]))
        receipt_rows = tuple(
            dict(row)
            for row in self.ledger.connection.execute(
                "SELECT * FROM receipts WHERE candidate_id=? ORDER BY sequence",
                (candidate_id,),
            ).fetchall()
        )
        if tuple(row["receipt_id"] for row in receipt_rows) != tuple(
            result.receipt_ids
        ):
            raise HeldoutProtocolError(
                "terminal shock candidate receipt inventory is incomplete or duplicated"
            )
        for row, receipt in zip(receipt_rows, receipts):
            complete_hash = receipt_hash(receipt)
            receipt_event = self._event(str(row["event_id"]))
            if (
                row["receipt_hash"] != complete_hash
                or row["receipt_type"] != receipt["receipt_type"]
                or row["campaign_id"] != self.campaign_id
                or row["run_id"] != state["run_id"]
                or row["task_id"] != self.coordinate.task_id
                or row["candidate_id"] != candidate_id
                or row["idempotency_key"] != receipt["idempotency_key"]
                or row["previous_receipt_hash"]
                != receipt["previous_receipt_hash"]
                or row["payload_json"] != canonical_json(receipt)
                or receipt_event.get("event_type") != "RECEIPT"
                or receipt_event.get("payload")
                != {"receipt": dict(receipt), "receipt_hash": complete_hash}
                or receipt_event.get("campaign_id") != self.campaign_id
                or receipt_event.get("run_id") != state["run_id"]
                or receipt_event.get("task_id") != self.coordinate.task_id
                or receipt_event.get("subject_id") != candidate_id
                or receipt_event.get("source_class") != "FROZEN_EVALUATOR"
                or receipt_event.get("disposition") != "VERIFIED"
                or receipt_event.get("idempotency_key")
                != "receipt:" + str(receipt["idempotency_key"])
            ):
                raise HeldoutProtocolError(
                    "terminal shock signed receipt projection is substituted"
                )

        verdict = receipts[1]
        verdict_id = content_id(
            "shock-verdict",
            {"operation_id": state["operation_id"], "receipt_id": verdict["receipt_id"]},
        )
        expected_verdict = {
            "verdict_id": verdict_id,
            "candidate_id": candidate_id,
            "correctness": result.diagnostic_enum == Diagnostic.PASS.value,
            "performance": {"resource_bucket": result.resource_bucket},
            "hidden_test_set_hash": digest_for(
                {"evaluator": self.evaluator.evaluator_digest, "task": self.coordinate.task_id}
            ),
            "evaluator_revision": self.evaluator.evaluator_revision,
            "receipt_id": verdict["receipt_id"],
            "signed_receipt_hash": receipt_hash(verdict),
        }
        verdict_rows = tuple(
            self.ledger.connection.execute(
                "SELECT * FROM verdicts WHERE candidate_id=?", (candidate_id,)
            ).fetchall()
        )
        if len(verdict_rows) != 1:
            raise HeldoutProtocolError(
                "terminal shock verdict inventory is incomplete or duplicated"
            )
        verdict_row = verdict_rows[0]
        verdict_event = self._event(str(verdict_row["event_id"]))
        if (
            verdict_row["verdict_id"] != verdict_id
            or verdict_row["receipt_id"] != verdict["receipt_id"]
            or verdict_row["correctness"] != int(expected_verdict["correctness"])
            or verdict_row["performance_json"]
            != canonical_json(expected_verdict["performance"])
            or verdict_row["hidden_test_set_hash"]
            != expected_verdict["hidden_test_set_hash"]
            or verdict_row["evaluator_revision"] != self.evaluator.evaluator_revision
            or verdict_row["signed_receipt_hash"]
            != expected_verdict["signed_receipt_hash"]
            or verdict_event.get("event_type") != "VERDICT"
            or verdict_event.get("payload") != expected_verdict
            or verdict_event.get("campaign_id") != self.campaign_id
            or verdict_event.get("run_id") != state["run_id"]
            or verdict_event.get("task_id") != self.coordinate.task_id
            or verdict_event.get("subject_id") != candidate_id
            or verdict_event.get("source_class") != "FROZEN_EVALUATOR"
            or verdict_event.get("disposition") != "VERIFIED"
            or verdict_event.get("idempotency_key") is not None
        ):
            raise HeldoutProtocolError(
                "terminal shock verdict projection is substituted"
            )

        effect = receipts[2]
        expected_effect = {
            "request_id": str(effect["request_id"]),
            "candidate_id": candidate_id,
            "identity": str(effect.get("identity", "remote-frozen-evaluator")),
            "normalized_action_hash": str(effect["normalized_action_hash"]),
            "decision": str(effect["decision"]),
            "policy_hash": str(effect["policy_digest"]),
            "sandbox_id": str(effect["sandbox_id"]),
            "started_at": str(effect["started_at"]),
            "finished_at": str(effect["finished_at"]),
            "exit_status_class": str(effect["exit_status_class"]),
            "output_hash": effect.get("output_digest"),
            "environment_diff_hash": effect.get("environment_diff_digest"),
            "signature": str(effect["signature"]),
            "receipt_id": str(effect["receipt_id"]),
        }
        effect_rows = tuple(
            self.ledger.connection.execute(
                "SELECT * FROM effect_receipts WHERE candidate_id=?", (candidate_id,)
            ).fetchall()
        )
        if len(effect_rows) != 1:
            raise HeldoutProtocolError(
                "terminal shock effect inventory is incomplete or duplicated"
            )
        effect_row = effect_rows[0]
        effect_event = self._event(str(effect_row["event_id"]))
        if (
            any(effect_row[key] != value for key, value in expected_effect.items())
            or effect_event.get("event_type") != "EFFECT_RECEIPT"
            or effect_event.get("payload") != expected_effect
            or effect_event.get("campaign_id") != self.campaign_id
            or effect_event.get("run_id") != state["run_id"]
            or effect_event.get("task_id") != self.coordinate.task_id
            or effect_event.get("subject_id") != candidate_id
            or effect_event.get("source_class") != "FROZEN_EVALUATOR"
            or effect_event.get("disposition") != "VERIFIED"
            or effect_event.get("idempotency_key") is not None
        ):
            raise HeldoutProtocolError(
                "terminal shock effect projection is substituted"
            )

        attempt_payload = {
            "schema_version": SHOCK_ATTEMPT_EVENT_SCHEMA,
            "operation_id": state["operation_id"],
            "phase": state["phase"],
            "attempt": state["attempt"],
            "candidate_id": candidate_id,
            "candidate_artifact_digest": result.candidate_artifact_digest,
            "diagnostic_enum": result.diagnostic_enum,
            "resource_bucket": result.resource_bucket,
            "disposition": result.disposition,
            "receipt_ids": list(result.receipt_ids),
        }
        attempt_key = "shock-attempt:" + str(state["operation_id"])
        attempt_events = tuple(
            event
            for event in self.ledger.events()
            if (
                event.get("event_type") == "SHOCK_ATTEMPT"
                and event.get("subject_id") == candidate_id
            )
            or event.get("idempotency_key") == attempt_key
        )
        if len(attempt_events) != 1:
            raise HeldoutProtocolError(
                "terminal shock attempt event inventory is incomplete or duplicated"
            )
        attempt_event = attempt_events[0]
        if (
            attempt_event.get("event_type") != "SHOCK_ATTEMPT"
            or attempt_event.get("payload") != attempt_payload
            or attempt_event.get("campaign_id") != self.campaign_id
            or attempt_event.get("run_id") != state["run_id"]
            or attempt_event.get("task_id") != self.coordinate.task_id
            or attempt_event.get("subject_id") != candidate_id
            or attempt_event.get("source_class") != "FROZEN_EVALUATOR"
            or attempt_event.get("disposition") != "VERIFIED"
            or attempt_event.get("idempotency_key") != attempt_key
        ):
            raise HeldoutProtocolError(
                "terminal shock attempt event projection is substituted"
            )
        return receipts

    def _validate_complete_operation(
        self,
        state: Mapping[str, Any],
        context: CandidateContext,
        evidence: CandidateGenerationEvidence,
        evidence_digest: str,
    ) -> ShockAttemptObservation:
        if (
            state["status"] != "COMPLETE"
            or state["generation_evidence_digest"] != evidence_digest
            or state["candidate_source_digest"] != digest_bytes(evidence.proposal.source)
        ):
            raise HeldoutProtocolError("complete shock operation differs from private generation replay")
        if _context_digest(context) != state["context_digest"]:
            raise HeldoutProtocolError("complete shock operation crossed its exact context")
        self._validate_complete_candidate_projection(
            state, context, evidence, evidence_digest
        )
        stored_result = _result_from_mapping(state["evaluation_result"])
        receipts = self._validate_complete_result_projection(
            state, evidence.proposal, stored_result
        )
        rebuilt = self._observation(state, context, evidence.proposal, stored_result, receipts)
        stored = ShockAttemptObservation.from_mapping(state["observation"])
        if rebuilt != stored:
            raise HeldoutProtocolError("shock observation differs from independent private/public replay")
        return stored

    def freeze_pre_shock_state(self) -> PreShockState:
        records = self._terminal_records("PRE")
        if len(records) != 6 or [item["attempt"] for item in records] != list(range(1, 7)):
            raise HeldoutProtocolError("actual pre-shock state requires exactly six terminal attempts")
        self._validate_generation_failure_inventory()
        premise_id = str(self._premise_event()["event_id"])
        unrelated = {
            str(event["event_id"])
            for event in self._unrelated_evidence_events()
        }
        attempt_node_ids = {str(record["candidate_id"]) for record in records}
        nodes = attempt_node_ids | unrelated | {premise_id}
        candidate_aliases = {
            str(row["event_id"]): str(row["candidate_id"])
            for row in self.ledger.connection.execute(
                "SELECT candidate_id,event_id FROM candidates ORDER BY candidate_id"
            ).fetchall()
        }
        edges = set()
        for row in self.ledger.connection.execute(
            "SELECT parent_id,child_id FROM dependencies ORDER BY dependency_id"
        ).fetchall():
            parent = candidate_aliases.get(str(row["parent_id"]), str(row["parent_id"]))
            child = candidate_aliases.get(str(row["child_id"]), str(row["child_id"]))
            if parent in nodes and child in nodes:
                edges.add((parent, child))
        graph = DependencyGraph(tuple(sorted(edges)))
        if not graph.descendants(premise_id):
            raise HeldoutProtocolError("actual pre-shock dependency graph has no affected descendants")
        candidate_state = {
            "schema_version": "egv-actual-pre-shock-candidate-state-v1",
            "attempt_count": 6,
            "candidates": [
                {
                    "attempt": record["attempt"],
                    "candidate_id": record["candidate_id"],
                    "status": record["status"],
                    "candidate_source_digest": record["candidate_source_digest"],
                    "generation_evidence_digest": record["generation_evidence_digest"],
                    "evaluation_result_digest": (
                        digest_for(record["evaluation_result"])
                        if record["evaluation_result"] is not None
                        else None
                    ),
                    "observation_digest": digest_for(record["observation"]),
                }
                for record in records
            ],
            "latest_candidate_id": (
                self._successful_records("PRE")[-1]["candidate_id"]
                if self._successful_records("PRE")
                else None
            ),
            "unrelated_evidence_ids": sorted(unrelated),
            "actual_rng_state_evidence_digest": records[-1]["rng_state_evidence_digest"],
        }
        state = PreShockState(
            accepted_premise_id=premise_id,
            candidate_state=candidate_state,
            dependency_graph=graph,
            rng_state_digest=_require_digest(
                self.coordinate.rng_state_digest, "frozen shock RNG boundary digest"
            ),
        )
        state.validate()
        return state

    def restore_from_journal(self, journal: Mapping[str, Any]) -> None:
        if not isinstance(journal, Mapping):
            raise HeldoutProtocolError("shock runtime journal restoration input is invalid")
        for phase, count_field, observations_field in (
            ("PRE", "pre_attempts", "pre_observations"),
            ("POST", "post_attempts", "post_observations"),
        ):
            count = journal.get(count_field)
            observations = journal.get(observations_field)
            if type(count) is not int or not isinstance(observations, list) or len(observations) != count:
                raise HeldoutProtocolError("shock runtime journal attempt cursor is malformed")
            for attempt, raw in enumerate(observations, start=1):
                state = self.operation_store.load(self._operation_id(phase, attempt))
                if state is None or state["status"] not in _TERMINAL_OPERATION_STATUSES:
                    raise HeldoutProtocolError("shock runtime journal lacks exact durable operation evidence")
                journal_observation = ShockAttemptObservation.from_mapping(raw)
                if state["observation"] != journal_observation.to_dict():
                    raise HeldoutProtocolError("shock runtime journal lacks exact durable operation evidence")
                candidate_id = str(state["candidate_id"])
                if state["status"] == "GENERATION_FAILED":
                    try:
                        context, failure, evidence_digest = (
                            self.private_store.load_generation_failure_read_only(candidate_id)
                        )
                    except Exception as exc:
                        raise HeldoutProtocolError(
                            "journaled shock generation failure lacks exact private replay evidence"
                        ) from exc
                    replayed = self._validate_generation_failure_operation(
                        state, context, failure, evidence_digest
                    )
                else:
                    try:
                        context, evidence, evidence_digest = (
                            self.private_store.load_generation_success_read_only(candidate_id)
                        )
                    except Exception as exc:
                        raise HeldoutProtocolError(
                            "journaled shock generation success lacks exact private replay evidence"
                        ) from exc
                    replayed = self._validate_complete_operation(
                        state, context, evidence, evidence_digest
                    )
                if replayed != journal_observation:
                    raise HeldoutProtocolError(
                        "shock runtime journal observation differs from durable replay"
                    )
        if journal.get("correction_committed") and self._correction_receipt(required=False) is None:
            raise HeldoutProtocolError("journaled shock correction lacks durable ledger evidence")
        if journal.get("policy_activated") and self._policy_receipt(required=False) is None:
            raise HeldoutProtocolError("journaled shock policy lacks durable ledger evidence")
        self._validate_generation_failure_inventory()

    def _event_by_idempotency(self, key: str) -> Optional[Mapping[str, Any]]:
        rows = [event for event in self.ledger.events() if event.get("idempotency_key") == key]
        if len(rows) > 1:
            raise HeldoutProtocolError("shock ledger contains duplicate idempotent protocol events")
        return rows[0] if rows else None

    def commit_correction(self, correction_event_digest: str, *, idempotency_key: str) -> None:
        if len(self._terminal_records("PRE")) != 6:
            raise HeldoutProtocolError("shock correction cannot precede exact attempt six")
        if correction_event_digest != self.coordinate.correction_event_digest:
            raise HeldoutProtocolError("shock correction differs from preregistered event digest")
        expected_key = digest_for({
            "coordinate_id": self.coordinate.coordinate_id,
            "operation": "commit-correction",
            "correction_event_digest": self.coordinate.correction_event_digest,
        })
        if idempotency_key != expected_key:
            raise HeldoutProtocolError("shock correction operation ID differs")
        state = self.freeze_pre_shock_state()
        replacement_payload = {
            "schema_version": SHOCK_CORRECTION_SCHEMA,
            "block_id": self.coordinate.block_id,
            "correction_event_digest": correction_event_digest,
            "accepted_premise_id": state.accepted_premise_id,
            "candidate_state_digest": digest_for(dict(state.candidate_state)),
            "dependency_graph_digest": state.dependency_graph.digest,
            "effective_after_attempt": 6,
        }
        replacement = self.ledger.append_event(
            "SHOCK_CORRECTED_PREMISE",
            replacement_payload,
            campaign_id=self.campaign_id,
            run_id=self._pre_run_id(),
            task_id=self.coordinate.task_id,
            subject_id=content_id("shock-corrected-premise", replacement_payload),
            source_class="FROZEN_PROTOCOL",
            disposition="OBSERVED",
            idempotency_key="shock-replacement:" + self.coordinate.block_id,
        )
        correction = self.ledger.append_correction(
            state.accepted_premise_id,
            str(replacement["event_id"]),
            reason_code="PREREGISTERED_CORRECTION_AFTER_ATTEMPT_6",
            correction_source="FROZEN_PROTOCOL",
            campaign_id=self.campaign_id,
            run_id=self._pre_run_id(),
            task_id=self.coordinate.task_id,
            effective_after_attempt=6,
        )
        self.ledger.append_event(
            "SHOCK_CORRECTION_COMMIT",
            {
                **replacement_payload,
                "coordinate_id": self.coordinate.coordinate_id,
                "replacement_event_id": replacement["event_id"],
                "correction_id": correction["correction_id"],
                "correction_record_event_id": correction["event_id"],
            },
            campaign_id=self.campaign_id,
            run_id=self._pre_run_id(),
            task_id=self.coordinate.task_id,
            subject_id=state.accepted_premise_id,
            source_class="FROZEN_PROTOCOL",
            disposition="VERIFIED",
            idempotency_key="shock-correction:" + idempotency_key,
        )

    def _activate_policy(self, treatment: str, node_ids: Sequence[str], *, idempotency_key: str) -> None:
        expected_key = digest_for({
            "coordinate_id": self.coordinate.coordinate_id,
            "operation": "activate-policy",
            "policy": self.coordinate.treatment,
        })
        if treatment != self.coordinate.treatment or idempotency_key != expected_key:
            raise HeldoutProtocolError("shock policy activation differs from frozen coordinate")
        if self._correction_receipt(required=False) is None:
            raise HeldoutProtocolError("shock policy activation precedes durable correction")
        state = self.freeze_pre_shock_state()
        affected = tuple(sorted(state.dependency_graph.descendants(state.accepted_premise_id)))
        supplied = tuple(sorted(node_ids))
        if treatment == "dependency-aware":
            if supplied != affected:
                raise HeldoutProtocolError("dependency-aware invalidation differs from exact affected closure")
            candidate_ids = {
                str(row["candidate_id"])
                for row in self.ledger.connection.execute(
                    "SELECT candidate_id FROM candidates"
                ).fetchall()
            }
            dispositions = {
                node: (
                    self.ledger.candidate_disposition(node)
                    if node in candidate_ids
                    else self.ledger.event_disposition(node)
                )
                for node in supplied
            }
            if any(value != "STALE_DEPENDENT" for value in dispositions.values()):
                raise HeldoutProtocolError("dependency-aware invalidation did not materialize stale descendants")
        elif supplied:
            raise HeldoutProtocolError("restart/reuse policy cannot carry dependency invalidations")
        payload = {
            "schema_version": SHOCK_POLICY_SCHEMA,
            "coordinate_id": self.coordinate.coordinate_id,
            "treatment": treatment,
            "affected_node_ids": list(affected),
            "invalidated_node_ids": list(supplied),
            "correction_receipt_digest": self._event_receipt_digest(
                self._correction_receipt(required=True)
            ),
        }
        self.ledger.append_event(
            "SHOCK_POLICY_ACTIVATION",
            payload,
            campaign_id=self.campaign_id,
            run_id=self._pre_run_id(),
            task_id=self.coordinate.task_id,
            subject_id=self.coordinate.coordinate_id,
            source_class="FROZEN_PROTOCOL",
            disposition="VERIFIED",
            idempotency_key="shock-policy:" + idempotency_key,
        )

    def full_restart(self, *, idempotency_key: str) -> None:
        self._activate_policy("full-restart", (), idempotency_key=idempotency_key)

    def reuse_without_invalidation(self, *, idempotency_key: str) -> None:
        self._activate_policy("naive-reuse", (), idempotency_key=idempotency_key)

    def invalidate_dependencies(self, node_ids: Sequence[str], *, idempotency_key: str) -> None:
        self._activate_policy("dependency-aware", node_ids, idempotency_key=idempotency_key)

    def _correction_receipt(self, *, required: bool) -> Optional[Mapping[str, Any]]:
        operation = digest_for({
            "coordinate_id": self.coordinate.coordinate_id,
            "operation": "commit-correction",
            "correction_event_digest": self.coordinate.correction_event_digest,
        })
        event = self._event_by_idempotency("shock-correction:" + operation)
        if required and event is None:
            raise HeldoutProtocolError("shock correction receipt is missing")
        return event

    def _policy_receipt(self, *, required: bool) -> Optional[Mapping[str, Any]]:
        operation = digest_for({
            "coordinate_id": self.coordinate.coordinate_id,
            "operation": "activate-policy",
            "policy": self.coordinate.treatment,
        })
        event = self._event_by_idempotency("shock-policy:" + operation)
        if required and event is None:
            raise HeldoutProtocolError("shock policy activation receipt is missing")
        return event

    @staticmethod
    def _event_receipt_digest(event: Optional[Mapping[str, Any]]) -> str:
        if event is None:
            raise HeldoutProtocolError("required shock protocol receipt is missing")
        return _require_digest(event.get("event_hash"), "shock protocol event receipt digest")

    def ledger_head_hash(self) -> str:
        return self.ledger.ledger_head_hash()

    def ledger_integrity(self) -> Mapping[str, Any]:
        return self.ledger.verify_integrity()

    def verification_evidence(self) -> ShockVerificationEvidence:
        records = self._terminal_records()
        if len(self._terminal_records("PRE")) != 6 or not records:
            raise HeldoutProtocolError("shock verification lacks exact completed attempt evidence")
        private_agreements = 0
        public_agreements = 0
        public_decisions = 0
        for state in records:
            if state["status"] == "GENERATION_FAILED":
                try:
                    context, failure, evidence_digest = (
                        self.private_store.load_generation_failure_read_only(
                            str(state["candidate_id"])
                        )
                    )
                except Exception as exc:
                    raise HeldoutProtocolError("shock private failure replay evidence is incomplete") from exc
                self._validate_generation_failure_operation(
                    state, context, failure, evidence_digest
                )
                private_agreements += 1
                continue
            try:
                context, evidence, evidence_digest = (
                    self.private_store.load_generation_success_read_only(
                        str(state["candidate_id"])
                    )
                )
            except Exception as exc:
                raise HeldoutProtocolError("shock private replay evidence is incomplete") from exc
            self._validate_complete_operation(state, context, evidence, evidence_digest)
            private_agreements += 1
            public_decisions += 1
            public_agreements += 1
        self._validate_generation_failure_inventory()
        receipts = self.ledger.receipts()
        unauthorized = 0
        by_candidate: Dict[str, list[Mapping[str, Any]]] = {}
        for receipt in receipts:
            by_candidate.setdefault(str(receipt.get("candidate_id")), []).append(receipt)
        for suffix in by_candidate.values():
            authorities = [item for item in suffix if item.get("receipt_type") == "AUTHORITY"]
            effects = [item for item in suffix if item.get("receipt_type") == "EFFECT"]
            if any(item.get("decision") == "ALLOW" for item in effects) and not any(
                item.get("decision") == "ALLOW" for item in authorities
            ):
                unauthorized += 1
        self.evaluator.validate_runtime()
        self.ledger.verify_receipt_chain(self.evaluator_public_key)
        correction = self._correction_receipt(required=True)
        policy = self._policy_receipt(required=True)
        evidence = ShockVerificationEvidence(
            evaluator_identity_valid=True,
            hidden_test_isolation_valid=True,
            private_replay_decisions=len(records),
            private_replay_agreements=private_agreements,
            public_replay_decisions=public_decisions,
            public_replay_agreements=public_agreements,
            unauthorized_successful_effects=unauthorized,
            correction_receipt_digest=self._event_receipt_digest(correction),
            policy_activation_receipt_digest=self._event_receipt_digest(policy),
        )
        evidence.validate()
        return evidence


__all__ = [
    "DurableShockOperationStore",
    "ProductionShockAttemptEngine",
    "ProductionShockEngineFactory",
    "SHOCK_GENERATION_FAILURE_SCHEMA",
    "SHOCK_OPERATION_SCHEMA",
]
