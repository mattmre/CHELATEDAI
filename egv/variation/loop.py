"""Bounded candidate loop integrated with the Evidence and Evaluation slices."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple
from weakref import WeakKeyDictionary

from ..canonical import content_id, digest_bytes, digest_for, failure_family_root
from ..evaluation.artifacts import ContentAddressedArtifactStore
from ..evaluation.controller import EvaluationResult, EvaluatorController, HiddenEvaluatorRunner
from ..evaluation.dataset import MicroRepo
from ..evaluation.diagnostics import Diagnostic, validate_diagnostic, validate_disposition, validate_resource_bucket
from ..evaluation.prompts import PROMPT_IDS, PromptRegistry
from ..evaluation.sandbox import DockerCandidateSandbox, DockerSandboxConfig
from ..ledger import EvidenceLedger
from ..receipts import receipt_hash
from .adapter import SealedAdapterArtifact, validate_applied_peft_model
from .arms import ArmIsolation, ArmPolicy, arm_policy
from .checkpoint import CheckpointStore, VariationCheckpoint
from .errors import VariationBudgetError, VariationCheckpointError, VariationConfigurationError, VariationDependencyError
from .generator import CandidateContext, CandidateGenerator, CandidateProposal, ModelCandidateGenerator
from .model import ADAPTER_ATTESTATION_SCHEMA, AdapterApplicationAttestation, MODEL_REVISION
from .retrieval import EvidenceRetrievalPolicy, RetrievalResult, retrieval_policy
from .remote import RemoteControllerEvaluationGateway


_ORIGINAL_DOCKER_SANDBOX_EXECUTE = DockerCandidateSandbox.execute
_ORIGINAL_DOCKER_SANDBOX_INTEGRITY = DockerCandidateSandbox.validate_production_integrity
_ORIGINAL_DOCKER_CONFIG_VERIFY_IMAGE = DockerSandboxConfig.verify_image
_ORIGINAL_DOCKER_CONFIG_VALIDATE = DockerSandboxConfig.validate
_ORIGINAL_MODEL_GENERATOR_PROPOSE = ModelCandidateGenerator.propose
_ORIGINAL_MODEL_GENERATOR_INTEGRITY = ModelCandidateGenerator.validate_production_integrity


@dataclass(frozen=True)
class _RuntimeIdentityRecord:
    """Canonical loop bindings held outside the writable loop instance."""

    fixture_mode: bool
    evaluator: Any
    generator: Any
    isolation: Any


def _make_runtime_identity_authority():
    records = WeakKeyDictionary()
    lock = RLock()

    def register(loop: Any, record: _RuntimeIdentityRecord) -> None:
        with lock:
            if loop in records:
                raise VariationDependencyError("Variation runtime identity was already registered")
            records[loop] = record

    def get(loop: Any) -> _RuntimeIdentityRecord:
        with lock:
            try:
                return records[loop]
            except KeyError as exc:
                raise VariationDependencyError("Variation runtime identity is not registered") from exc

    def registered(loop: Any) -> bool:
        with lock:
            return loop in records

    return register, get, registered


_register_runtime_identity, _get_runtime_identity, _runtime_identity_registered = _make_runtime_identity_authority()


MAX_CANDIDATE_ATTEMPTS = 12
CANDIDATE_SOURCE_LIMIT = 256 * 1024
VARIATION_PROTOCOL_SCHEMA = "egv-variation-protocol-v1"
VARIATION_PROMPT_MANIFEST_DIGEST = PromptRegistry().manifest_digest()
VARIATION_PROTOCOL_DIGEST = digest_for(
    {
        "schema_version": VARIATION_PROTOCOL_SCHEMA,
        "max_candidate_attempts": MAX_CANDIDATE_ATTEMPTS,
        "candidate_source_limit": CANDIDATE_SOURCE_LIMIT,
        "success_stops_loop": True,
        "model_revision": MODEL_REVISION,
        "production_backend": "docker-enforced-v1",
        "production_generator": "ModelCandidateGenerator",
        "adapter_attestation_schema": ADAPTER_ATTESTATION_SCHEMA,
        "prompt_ids": list(PROMPT_IDS),
        "prompt_manifest_digest": VARIATION_PROMPT_MANIFEST_DIGEST,
        "retrieval_policies": ["SUCCESS_ONLY", "ORDINARY_FAILURE_SUMMARY", "CORRECTION_AWARE"],
    }
)


@dataclass(frozen=True)
class VariationTask:
    """Public task contract plus evaluator-private input held by the caller."""

    task_id: str
    family_id: str
    public_locus: str
    public_rule_id: str
    task_statement: str
    evaluator_input: Any

    @classmethod
    def from_microrepo(cls, repo: MicroRepo) -> "VariationTask":
        return cls(
            task_id=repo.template_id,
            family_id=repo.family_id,
            public_locus=repo.public_locus,
            public_rule_id=repo.public_rule_id,
            task_statement="Repair the bounded {} task at {}.".format(repo.family_id, repo.public_locus),
            evaluator_input=repo.evaluator_input,
        )

    @classmethod
    def from_public_record(cls, record: Mapping[str, Any]) -> "VariationTask":
        """Construct a Spark1-safe task that never materializes evaluator input."""

        required = {"template_id", "family_id", "public_locus", "public_rule_id"}
        if not isinstance(record, Mapping) or any(not isinstance(record.get(key), str) or not record[key] for key in required):
            raise VariationConfigurationError("public Variation task record is incomplete")
        return cls(
            task_id=record["template_id"],
            family_id=record["family_id"],
            public_locus=record["public_locus"],
            public_rule_id=record["public_rule_id"],
            task_statement="Repair the bounded {} task at {}.".format(record["family_id"], record["public_locus"]),
            evaluator_input=None,
        )

    def public_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family_id": self.family_id,
            "public_locus": self.public_locus,
            "public_rule_id": self.public_rule_id,
            "task_statement": self.task_statement,
        }


class EvaluationGateway(Protocol):
    enforceable: bool
    evaluator_digest: str
    evaluator_revision: str

    def evaluate(
        self,
        *,
        candidate_id: str,
        task_id: str,
        source: bytes,
        opaque_input: Any,
        requested_authority: str,
        declared_locus: str,
        candidate_source_path: Optional[str] = None,
    ) -> EvaluationResult:
        ...


class ControllerEvaluationGateway:
    """Production adapter over the merged Docker-backed evaluator controller."""

    enforceable = True

    def __init__(self, controller: EvaluatorController) -> None:
        if not isinstance(controller, EvaluatorController):
            raise VariationDependencyError("Variation production path requires the real EvaluatorController")
        if not isinstance(controller.hidden_runner, HiddenEvaluatorRunner):
            raise VariationDependencyError("Variation production path requires the real hidden evaluator runner")
        if not isinstance(controller.sandbox, DockerCandidateSandbox) or not getattr(controller.sandbox, "enforceable", False):
            raise VariationDependencyError("Variation production path requires the enforceable Docker evaluator")
        self._validate_sandbox_method_identity(controller.sandbox)
        try:
            controller.sandbox.validate_production_integrity()
        except Exception as exc:
            raise VariationDependencyError("Variation production Docker sandbox failed integrity validation") from exc
        self.controller = controller
        self.hidden_runner = controller.hidden_runner
        self.task_registry = self.hidden_runner
        self.sandbox = controller.sandbox
        self._pinned_controller = controller
        self._pinned_hidden_runner = controller.hidden_runner
        self._pinned_sandbox = controller.sandbox
        self._pinned_config = controller.sandbox.config
        self._pinned_config_digest = digest_for(dict(controller.sandbox.config.__dict__))
        self._pinned_image_id = controller.sandbox.image_id
        self._pinned_docker_binary = controller.sandbox.docker_binary
        self._pinned_artifacts = controller.sandbox.artifacts
        self.evaluator_revision = self.hidden_runner.evaluator_revision
        self.evaluator_digest = digest_for(self.evaluator_revision)
        self._gateway_contract = digest_for(
            {
                "controller_type": type(controller).__qualname__,
                "hidden_runner_type": type(controller.hidden_runner).__qualname__,
                "sandbox_type": type(controller.sandbox).__qualname__,
                "image_id": self._pinned_image_id,
                "docker_binary": self._pinned_docker_binary,
            }
        )

    @staticmethod
    def _validate_sandbox_method_identity(sandbox: DockerCandidateSandbox) -> None:
        if "execute" in sandbox.__dict__ or "validate_production_integrity" in sandbox.__dict__:
            raise VariationDependencyError("Variation production sandbox methods cannot be overridden on an instance")
        if type(sandbox).execute is not _ORIGINAL_DOCKER_SANDBOX_EXECUTE:
            raise VariationDependencyError("Variation production Docker execute method was altered")
        if type(sandbox).validate_production_integrity is not _ORIGINAL_DOCKER_SANDBOX_INTEGRITY:
            raise VariationDependencyError("Variation production Docker integrity method was altered")
        config = getattr(sandbox, "config", None)
        if type(config) is not DockerSandboxConfig:
            raise VariationDependencyError("Variation production Docker config type is invalid")
        if DockerSandboxConfig.verify_image is not _ORIGINAL_DOCKER_CONFIG_VERIFY_IMAGE:
            raise VariationDependencyError("Variation production Docker image verification method was altered")
        if DockerSandboxConfig.validate is not _ORIGINAL_DOCKER_CONFIG_VALIDATE:
            raise VariationDependencyError("Variation production Docker config validation method was altered")

    def validate_runtime(self) -> None:
        if type(self) is not ControllerEvaluationGateway:
            raise VariationDependencyError("Variation production gateway type is not frozen")
        if "evaluate" in self.__dict__:
            raise VariationDependencyError("Variation production gateway evaluate cannot be overridden")
        if ControllerEvaluationGateway.evaluate is not _ORIGINAL_GATEWAY_EVALUATE:
            raise VariationDependencyError("Variation production gateway evaluate method was altered")
        if not isinstance(self.controller, EvaluatorController):
            raise VariationDependencyError("Variation production controller identity is invalid")
        if not isinstance(self.controller.hidden_runner, HiddenEvaluatorRunner):
            raise VariationDependencyError("Variation production hidden evaluator runner is unavailable")
        if not isinstance(self.controller.sandbox, DockerCandidateSandbox) or not getattr(
            self.controller.sandbox, "enforceable", False
        ):
            raise VariationDependencyError("Variation production Docker sandbox is unavailable")
        self._validate_sandbox_method_identity(self.controller.sandbox)
        if not getattr(self, "enforceable", False):
            raise VariationDependencyError("Variation production gateway is not enforceable")
        if self.controller is not self._pinned_controller:
            raise VariationDependencyError("Variation gateway controller binding changed")
        if self.controller.hidden_runner is not self.hidden_runner:
            raise VariationDependencyError("Variation gateway hidden evaluator binding changed")
        if self.controller.hidden_runner is not self._pinned_hidden_runner:
            raise VariationDependencyError("Variation gateway hidden evaluator identity changed")
        if self.controller.sandbox is not self.sandbox or self.sandbox is not self._pinned_sandbox:
            raise VariationDependencyError("Variation gateway Docker sandbox identity changed")
        if self.sandbox.config is not self._pinned_config:
            raise VariationDependencyError("Variation gateway Docker config binding changed")
        if digest_for(dict(self.sandbox.config.__dict__)) != self._pinned_config_digest:
            raise VariationDependencyError("Variation gateway Docker config contract changed")
        if self.sandbox.image_id != self._pinned_image_id or self.sandbox.docker_binary != self._pinned_docker_binary:
            raise VariationDependencyError("Variation gateway Docker image or binary binding changed")
        if self.sandbox.artifacts is not self._pinned_artifacts:
            raise VariationDependencyError("Variation gateway artifact store binding changed")
        if digest_for(
            {
                "controller_type": type(self.controller).__qualname__,
                "hidden_runner_type": type(self.controller.hidden_runner).__qualname__,
                "sandbox_type": type(self.sandbox).__qualname__,
                "image_id": self.sandbox.image_id,
                "docker_binary": self.sandbox.docker_binary,
            }
        ) != self._gateway_contract:
            raise VariationDependencyError("Variation production contract digest changed")
        try:
            self.sandbox.validate_production_integrity()
        except Exception as exc:
            raise VariationDependencyError("Variation production Docker sandbox failed runtime integrity validation") from exc

    def evaluate(self, **kwargs: Any) -> EvaluationResult:
        self.validate_runtime()
        return self.controller.evaluate(**kwargs)


_ORIGINAL_GATEWAY_EVALUATE = ControllerEvaluationGateway.evaluate
_ORIGINAL_GATEWAY_VALIDATE_RUNTIME = ControllerEvaluationGateway.validate_runtime


@dataclass(frozen=True)
class AttemptRecord:
    attempt_index: int
    candidate_id: str
    candidate_artifact_digest: str
    retrieval_digest: str
    evidence_ids: Tuple[str, ...]
    diagnostic_enum: str
    resource_bucket: str
    disposition: str
    receipt_ids: Tuple[str, ...]
    ledger_head_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "candidate_id": self.candidate_id,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "retrieval_digest": self.retrieval_digest,
            "evidence_ids": list(self.evidence_ids),
            "diagnostic_enum": self.diagnostic_enum,
            "resource_bucket": self.resource_bucket,
            "disposition": self.disposition,
            "receipt_ids": list(self.receipt_ids),
            "ledger_head_hash": self.ledger_head_hash,
        }


@dataclass(frozen=True)
class VariationReport:
    campaign_id: str
    run_id: str
    arm_id: str
    task_id: str
    seed: int
    attempts: Tuple[AttemptRecord, ...]
    terminal_status: str
    checkpoint_path: str
    ledger_head_hash: str
    ledger_integrity: Mapping[str, Any]
    model_digest: str
    adapter_digest: Optional[str]
    retrieval_policy: str
    authority_enforced: bool

    @property
    def promoted(self) -> bool:
        return self.terminal_status == "PROMOTED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-variation-report-v1",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "terminal_status": self.terminal_status,
            "checkpoint_path": Path(self.checkpoint_path).name,
            "ledger_head_hash": self.ledger_head_hash,
            "ledger_integrity": dict(self.ledger_integrity),
            "model_digest": self.model_digest,
            "adapter_digest": self.adapter_digest,
            "retrieval_policy": self.retrieval_policy,
            "authority_enforced": self.authority_enforced,
        }

    def to_public_dict(self) -> Dict[str, Any]:
        """Return a redacted report suitable for a publishable smoke artifact."""

        return {
            "schema_version": "egv-variation-public-report-v1",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "task_binding_digest": digest_for({"task_id": self.task_id}),
            "attempts": [
                {
                    "attempt_index": attempt.attempt_index,
                    "retrieval_digest": attempt.retrieval_digest,
                    "evidence_count": len(attempt.evidence_ids),
                    "diagnostic_enum": attempt.diagnostic_enum,
                    "resource_bucket": attempt.resource_bucket,
                    "disposition": attempt.disposition,
                    "receipt_count": len(attempt.receipt_ids),
                    "ledger_head_hash": attempt.ledger_head_hash,
                }
                for attempt in self.attempts
            ],
            "terminal_status": self.terminal_status,
            "checkpoint_digest": Path(self.checkpoint_path).stem.removeprefix("checkpoint-"),
            "ledger_head_hash": self.ledger_head_hash,
            "ledger_integrity": dict(self.ledger_integrity),
            "model_digest": self.model_digest,
            "adapter_digest": self.adapter_digest,
            "retrieval_policy": self.retrieval_policy,
            "authority_enforced": self.authority_enforced,
        }


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise VariationConfigurationError("{} must be a SHA-256 digest".format(label))
    try:
        int(value, 16)
    except ValueError as exc:
        raise VariationConfigurationError("{} must be hexadecimal".format(label)) from exc


def _validate_model_generator_identity(generator: Any) -> None:
    if type(generator) is not ModelCandidateGenerator:
        raise VariationDependencyError("production/LoRA Variation requires the exact ModelCandidateGenerator")
    if "propose" in generator.__dict__:
        raise VariationDependencyError("production model generator propose cannot be overridden")
    if type(generator).propose is not _ORIGINAL_MODEL_GENERATOR_PROPOSE:
        raise VariationDependencyError("production model generator propose method was altered")
    if "validate_production_integrity" in generator.__dict__:
        raise VariationDependencyError("production model generator integrity cannot be overridden")
    if type(generator).validate_production_integrity is not _ORIGINAL_MODEL_GENERATOR_INTEGRITY:
        raise VariationDependencyError("production model generator integrity method was altered")
    try:
        generator.validate_production_integrity()
    except Exception as exc:
        if isinstance(exc, VariationDependencyError):
            raise
        raise VariationDependencyError("production model generator failed integrity validation") from exc
    adapter_artifact = getattr(generator, "adapter_artifact", None)
    if adapter_artifact is not None:
        try:
            validate_applied_peft_model(generator.loaded_model.model, adapter_artifact)
        except Exception as exc:
            if isinstance(exc, VariationDependencyError):
                raise
            raise VariationDependencyError("production model generator PEFT runtime validation failed") from exc


class BoundedCandidateLoop:
    """One immutable, resumable candidate trajectory."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "BoundedCandidateLoop":
        if cls is BoundedCandidateLoop:
            target = _FixtureBoundedCandidateLoop if kwargs.get("fixture_mode", False) else _ProductionBoundedCandidateLoop
            return object.__new__(target)
        return object.__new__(cls)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_max_attempts" and "_budget_contract" in self.__dict__:
            raise AttributeError("Variation attempt budget is immutable after construction")
        if name in {"fixture_mode", "evaluator", "generator", "isolation"} and _runtime_identity_registered(self):
            raise AttributeError("Variation runtime identity is immutable after construction")
        super().__setattr__(name, value)

    def __init__(
        self,
        *,
        ledger: EvidenceLedger,
        evaluator: EvaluationGateway,
        generator: CandidateGenerator,
        isolation: ArmIsolation,
        workspace_root: Path,
        campaign_id: str,
        source_commit: str,
        model_revision: str,
        model_digest: str,
        data_manifest_digest: str,
        policy_digest: str,
        arm_id: str,
        max_attempts: int = MAX_CANDIDATE_ATTEMPTS,
        adapter_digest: Optional[str] = None,
        adapter_artifact: Optional[SealedAdapterArtifact] = None,
        seed_set: Sequence[int] = (0, 1, 2),
        fixture_mode: bool = False,
    ) -> None:
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts <= 0 or max_attempts > MAX_CANDIDATE_ATTEMPTS:
            raise VariationBudgetError("candidate attempt budget must be between 1 and 12")
        self.ledger = ledger
        self.evaluator = evaluator
        self.generator = generator
        self.isolation = isolation
        self.workspace_root = Path(workspace_root).resolve()
        self.campaign_id = campaign_id
        self.source_commit = source_commit
        self.model_revision = model_revision
        self.model_digest = model_digest
        self.data_manifest_digest = data_manifest_digest
        self.policy_digest = policy_digest
        self.policy: ArmPolicy = arm_policy(arm_id)
        self._max_attempts = max_attempts
        self._budget_contract = digest_for({"max_attempts": max_attempts})
        self.adapter_artifact = adapter_artifact
        self.adapter_digest = adapter_digest
        self.seed_set = tuple(sorted(set(int(seed) for seed in seed_set)))
        self.fixture_mode = fixture_mode
        if not self.seed_set or any(seed < 0 for seed in self.seed_set):
            raise VariationConfigurationError("Variation seed set must contain nonnegative integers")
        _require_digest(model_digest, "model digest")
        _require_digest(data_manifest_digest, "data manifest digest")
        _require_digest(policy_digest, "authority policy digest")
        if model_revision != MODEL_REVISION:
            raise VariationConfigurationError("Variation campaign must use the frozen Qwen model revision")
        if self.policy.requires_adapter:
            if adapter_artifact is None:
                raise VariationDependencyError("LoRA arms require a sealed adapter artifact; Training is not part of Variation")
            if type(adapter_artifact) is not SealedAdapterArtifact:
                raise VariationDependencyError("LoRA arms require a SealedAdapterArtifact, not a digest-like object")
            adapter_artifact.verify()
            derived_adapter_digest = adapter_artifact.digest
            if adapter_digest is not None and adapter_digest != derived_adapter_digest:
                raise VariationConfigurationError("adapter digest differs from the sealed adapter manifest")
            self.adapter_digest = derived_adapter_digest
        elif adapter_digest is not None or adapter_artifact is not None:
            raise VariationConfigurationError("base arms cannot carry a LoRA adapter")
        if getattr(generator, "model_digest", model_digest) != model_digest:
            raise VariationConfigurationError("candidate generator model digest differs from the frozen campaign")
        if getattr(generator, "adapter_digest", self.adapter_digest) != self.adapter_digest:
            raise VariationConfigurationError("candidate generator adapter digest differs from the frozen campaign")
        generator_adapter = getattr(generator, "adapter_artifact", None)
        if (generator_adapter is None) != (adapter_artifact is None):
            raise VariationDependencyError("candidate generator and Variation arm disagree about sealed adapter state")
        if generator_adapter is not None and generator_adapter.digest != self.adapter_digest:
            raise VariationConfigurationError("candidate generator sealed adapter differs from the frozen campaign")
        if self.policy.requires_adapter:
            if type(generator) is not ModelCandidateGenerator:
                raise VariationDependencyError("LoRA arms require the exact ModelCandidateGenerator")
            attestation = getattr(generator, "adapter_attestation", None)
            if type(attestation) is not AdapterApplicationAttestation:
                raise VariationDependencyError("LoRA arm has no loader-issued adapter application attestation")
            attestation.validate()
            if (
                attestation.schema_version != ADAPTER_ATTESTATION_SCHEMA
                or attestation.adapter_digest != self.adapter_digest
                or attestation.base_model_manifest_digest != self.model_digest
            ):
                raise VariationDependencyError("LoRA arm adapter attestation is not bound to the frozen campaign")
            _validate_model_generator_identity(generator)
        if getattr(generator, "test_only", False) and not fixture_mode:
            raise VariationDependencyError("test-only candidate generators cannot enter the production Variation path")
        if not fixture_mode and type(evaluator) not in {ControllerEvaluationGateway, RemoteControllerEvaluationGateway}:
            raise VariationDependencyError("production Variation requires an exact sealed Evaluation gateway")
        if not fixture_mode and type(generator) is not ModelCandidateGenerator:
            raise VariationDependencyError("production Variation requires the exact ModelCandidateGenerator")
        if not getattr(evaluator, "enforceable", False) and not fixture_mode:
            raise VariationDependencyError("non-enforceable evaluators are test-only and cannot run Variation")
        if type(evaluator) is RemoteControllerEvaluationGateway:
            evaluator.validate_campaign_bindings(
                campaign_id=campaign_id,
                model_digest=model_digest,
                protocol_digest=VARIATION_PROTOCOL_DIGEST,
                policy_digest=policy_digest,
                data_manifest_digest=data_manifest_digest,
            )
        self._validate_construction_boundary()
        _register_runtime_identity(
            self,
            _RuntimeIdentityRecord(
                fixture_mode=bool(self.fixture_mode),
                evaluator=self.evaluator,
                generator=self.generator,
                isolation=self.isolation,
            ),
        )

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def _validate_budget(self) -> None:
        if not isinstance(self._max_attempts, int) or isinstance(self._max_attempts, bool):
            raise VariationBudgetError("candidate attempt budget is not an integer")
        if self._max_attempts < 1 or self._max_attempts > MAX_CANDIDATE_ATTEMPTS:
            raise VariationBudgetError("candidate attempt budget must be between 1 and 12")
        if digest_for({"max_attempts": self._max_attempts}) != self._budget_contract:
            raise VariationBudgetError("candidate attempt budget changed after construction")

    def _validate_identity(self) -> None:
        record = _get_runtime_identity(self)
        missing = object()
        fixture_mode = getattr(self, "fixture_mode", missing)
        evaluator = getattr(self, "evaluator", missing)
        generator = getattr(self, "generator", missing)
        isolation = getattr(self, "isolation", missing)
        if any(value is missing for value in (fixture_mode, evaluator, generator, isolation)):
            raise VariationDependencyError("Variation runtime identity fields are missing")
        if fixture_mode != record.fixture_mode:
            raise VariationDependencyError("Variation fixture mode changed after construction")
        if evaluator is not record.evaluator:
            raise VariationDependencyError("Variation evaluator identity changed after construction")
        if generator is not record.generator:
            raise VariationDependencyError("Variation generator identity changed after construction")
        if isolation is not record.isolation:
            raise VariationDependencyError("Variation isolation identity changed after construction")

    def _validate_construction_boundary(self) -> None:
        if type(self) in {_ProductionBoundedCandidateLoop, _FixtureBoundedCandidateLoop}:
            self._validate_production_boundary()
            return
        raise VariationDependencyError("Variation loop concrete execution class is not recognized")

    def _validate_production_boundary(self) -> None:
        """Validate production unconditionally; this method has no fixture path."""

        if type(self) is not _ProductionBoundedCandidateLoop:
            raise VariationDependencyError("production Variation requires its concrete production loop class")
        if self.fixture_mode is not False:
            raise VariationDependencyError("production Variation cannot carry fixture mode")
        _validate_model_generator_identity(self.generator)
        if type(self.evaluator) not in {ControllerEvaluationGateway, RemoteControllerEvaluationGateway}:
            raise VariationDependencyError("production Variation requires an exact sealed Evaluation gateway")
        if type(self.evaluator) is ControllerEvaluationGateway:
            if "validate_runtime" in self.evaluator.__dict__ or "evaluate" in self.evaluator.__dict__:
                raise VariationDependencyError("production evaluator methods cannot be overridden")
            if type(self.evaluator).validate_runtime is not _ORIGINAL_GATEWAY_VALIDATE_RUNTIME:
                raise VariationDependencyError("production evaluator runtime validation method was altered")
            if type(self.evaluator).evaluate is not _ORIGINAL_GATEWAY_EVALUATE:
                raise VariationDependencyError("production evaluator evaluate method was altered")
        self.evaluator.validate_runtime()

    def _validate_fixture_boundary(self) -> None:
        """Validate the test-only fixture path on its separate concrete class."""

        if type(self) is not _FixtureBoundedCandidateLoop:
            raise VariationDependencyError("fixture Variation requires its concrete fixture loop class")
        if self.fixture_mode is not True:
            raise VariationDependencyError("fixture loop must be explicitly marked fixture mode")

    def _run_id(self, task_id: str, seed: int) -> str:
        return "egv-run-{}".format(
            digest_for({"campaign_id": self.campaign_id, "arm_id": self.policy.arm_id, "task_id": task_id, "seed": seed})[:40]
        )

    def _candidate_id(self, *, run_id: str, task_id: str, seed: int, attempt: int, parent: Optional[str]) -> str:
        candidate_id = "egv-candidate-{}-{}-{}".format(
            self.campaign_id,
            self.policy.arm_id,
            digest_for({"run_id": run_id, "task_id": task_id, "seed": seed, "attempt": attempt, "parent": parent})[:40],
        )
        self.isolation.assert_candidate_id(self.policy.arm_id, candidate_id)
        return candidate_id

    def _ensure_campaign(self, *, seed: int) -> None:
        self.ledger.create_campaign(
            self.campaign_id,
            protocol_hash=VARIATION_PROTOCOL_DIGEST,
            source_commit=self.source_commit,
            model_revision=self.model_revision,
            data_manifest_hash=self.data_manifest_digest,
            evaluator_hash=getattr(self.evaluator, "evaluator_digest", digest_for("unknown-evaluator")),
            policy_hash=self.policy_digest,
            seed_set=self.seed_set,
        )

    def _validate_task_binding(self, task: VariationTask) -> None:
        registry = getattr(self.evaluator, "task_registry", getattr(self.evaluator, "hidden_runner", None))
        public_record = registry.public_record(task.task_id) if registry is not None else None
        if not isinstance(public_record, Mapping):
            raise VariationConfigurationError("Variation task is not present in the immutable Evaluation manifest")
        expected = {
            "family_id": task.family_id,
            "public_locus": task.public_locus,
            "public_rule_id": task.public_rule_id,
        }
        actual = {key: public_record.get(key) for key in expected}
        if actual != expected:
            raise VariationConfigurationError("Variation task metadata differs from the immutable Evaluation manifest")

    def _write_candidate_artifact(self, workspace: Path, source: bytes) -> str:
        # Artifacts live below the isolated arm/run root, never in a shared
        # campaign workspace.
        artifacts = ContentAddressedArtifactStore(workspace / "candidates" / "artifacts")
        ref = artifacts.put(source, media_type="text/x-python", role="private-candidate-source")
        return str(artifacts.root / ref.relative_path)

    def _receipt_for(self, receipt_ids: Sequence[str], receipt_type: str) -> Optional[Dict[str, Any]]:
        for receipt_id in receipt_ids:
            row = self.ledger.receipt_by_id(receipt_id)
            if row is not None and row["receipt"].get("receipt_type") == receipt_type:
                return row["receipt"]
        return None

    def _materialize_result(
        self,
        *,
        task: VariationTask,
        run_id: str,
        candidate_id: str,
        proposal: CandidateProposal,
        retrieval: RetrievalResult,
        result: EvaluationResult,
        attempt_index: int,
    ) -> AttemptRecord:
        expected_artifact_digest = digest_bytes(proposal.source)
        if result.candidate_id != candidate_id or result.task_id != task.task_id:
            raise VariationCheckpointError("evaluator result is not bound to the requested candidate/task")
        if result.candidate_artifact_digest != expected_artifact_digest:
            raise VariationCheckpointError("evaluator result artifact digest differs from the immutable candidate bytes")
        try:
            validate_diagnostic(result.diagnostic_enum)
            validate_resource_bucket(result.resource_bucket)
            validate_disposition(result.disposition)
        except ValueError as exc:
            raise VariationCheckpointError("evaluator result uses a value outside the closed Evaluation contract") from exc
        if len(set(result.receipt_ids)) != len(result.receipt_ids):
            raise VariationCheckpointError("evaluator returned duplicate receipt IDs")
        if result.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value:
            if not result.infrastructure_loss or not result.infrastructure_incident_id or not result.failure_family_root:
                raise VariationCheckpointError("INTERNAL_ERROR result is missing its infrastructure incident/root pair")
            expected_root = failure_family_root(
                task.family_id,
                Diagnostic.INTERNAL_ERROR.value,
                task.public_locus,
                task.public_rule_id,
                infrastructure_incident_id=result.infrastructure_incident_id,
            )
            if result.failure_family_root != expected_root:
                raise VariationCheckpointError("INTERNAL_ERROR result failure root is not bound to the task and incident")
        elif result.infrastructure_loss or result.infrastructure_incident_id or result.failure_family_root:
            raise VariationCheckpointError("non-INTERNAL_ERROR result carries infrastructure-loss fields")
        verdict_receipt = self._receipt_for(result.receipt_ids, "VERDICT")
        effect_receipt = self._receipt_for(result.receipt_ids, "EFFECT")
        if verdict_receipt is not None:
            verdict_id = content_id(
                "verdict",
                {"candidate_id": candidate_id, "receipt_id": verdict_receipt["receipt_id"], "attempt_index": attempt_index},
            )
            self.ledger.append_verdict(
                verdict_id,
                candidate_id=candidate_id,
                correctness=result.diagnostic_enum == Diagnostic.PASS.value,
                performance={"resource_bucket": result.resource_bucket},
                hidden_test_set_hash=digest_for({"evaluator": result.candidate_artifact_digest, "task": task.task_id}),
                evaluator_revision=getattr(self.evaluator, "evaluator_revision", "frozen-evaluator"),
                receipt_id=verdict_receipt["receipt_id"],
                signed_receipt_hash=receipt_hash(verdict_receipt),
            )
        if effect_receipt is not None:
            self.ledger.append_effect_receipt(
                effect_receipt["request_id"],
                candidate_id=candidate_id,
                identity=str(effect_receipt.get("identity", "frozen-evaluator")),
                normalized_action_hash=effect_receipt["normalized_action_hash"],
                decision=effect_receipt["decision"],
                policy_hash=effect_receipt["policy_digest"],
                sandbox_id=effect_receipt["sandbox_id"],
                started_at=effect_receipt["started_at"],
                finished_at=effect_receipt["finished_at"],
                exit_status_class=effect_receipt["exit_status_class"],
                output_hash=effect_receipt.get("output_digest"),
                environment_diff_hash=effect_receipt.get("environment_diff_digest"),
                signature=effect_receipt["signature"],
                receipt_id=effect_receipt["receipt_id"],
            )
        self.ledger.append_event(
            "VARIATION_ATTEMPT",
            {
                "schema_version": "egv-variation-attempt-v1",
                "attempt_index": attempt_index,
                "candidate_id": candidate_id,
                "arm_id": self.policy.arm_id,
                "retrieval_policy": self.policy.retrieval_policy,
                "retrieval_digest": retrieval.evidence_digest,
                "evidence_ids": list(proposal.evidence_ids),
                "candidate_artifact_digest": result.candidate_artifact_digest,
                "diagnostic_enum": result.diagnostic_enum,
                "resource_bucket": result.resource_bucket,
                "disposition": result.disposition,
                "receipt_ids": list(result.receipt_ids),
            },
            campaign_id=self.campaign_id,
            run_id=run_id,
            task_id=task.task_id,
            subject_id=candidate_id,
            source_class="GENERATOR",
            disposition=result.disposition,
            idempotency_key="variation-attempt:{}:{}".format(candidate_id, attempt_index),
        )
        actual_disposition = self.ledger.candidate_disposition(candidate_id)
        if actual_disposition != result.disposition:
            raise VariationCheckpointError(
                "ledger disposition {} differs from evaluator disposition {}".format(actual_disposition, result.disposition)
            )
        return AttemptRecord(
            attempt_index,
            candidate_id,
            result.candidate_artifact_digest,
            retrieval.evidence_digest,
            tuple(proposal.evidence_ids),
            result.diagnostic_enum,
            result.resource_bucket,
            actual_disposition,
            tuple(result.receipt_ids),
            self.ledger.ledger_head_hash(),
        )

    def _checkpoint(
        self,
        *,
        workspace: Path,
        task: VariationTask,
        run_id: str,
        seed: int,
        attempts: Sequence[AttemptRecord],
        attempt: AttemptRecord,
        status: str,
    ) -> Path:
        if not attempts or attempts[-1] != attempt:
            raise VariationCheckpointError("checkpoint attempts are not ordered or do not end at the requested attempt")
        state_digest = digest_for(
            {
                "attempts": [item.to_dict() for item in attempts],
                "run_id": run_id,
                "arm_id": self.policy.arm_id,
                "task_id": task.task_id,
                "status": status,
            }
        )
        projection_generation = digest_for(
            {
                "ledger_head_event_id": self.ledger.ledger_head_event_id(),
                "ledger_head_hash": self.ledger.ledger_head_hash(),
                "arm_id": self.policy.arm_id,
                "run_id": run_id,
            }
        )
        artifact_manifest_hash = digest_for(
            {
                "artifacts": [
                    {
                        "attempt_index": item.attempt_index,
                        "candidate_id": item.candidate_id,
                        "candidate_artifact_digest": item.candidate_artifact_digest,
                    }
                    for item in attempts
                ]
            }
        )
        checkpoint = VariationCheckpoint(
            campaign_id=self.campaign_id,
            run_id=run_id,
            arm_id=self.policy.arm_id,
            task_id=task.task_id,
            seed=seed,
            attempt_index=attempt.attempt_index,
            last_candidate_id=attempt.candidate_id,
            ledger_head_event_id=self.ledger.ledger_head_event_id(),
            ledger_head_hash=self.ledger.ledger_head_hash(),
            projection_generation=projection_generation,
            artifact_manifest_hash=artifact_manifest_hash,
            protocol_digest=VARIATION_PROTOCOL_DIGEST,
            model_digest=self.model_digest,
            adapter_digest=self.adapter_digest,
            retrieval_policy_digest=self.retrieval_policy.digest,
            state_digest=state_digest,
            status=status,
        )
        store = CheckpointStore(workspace.checkpoints)
        path = store.save(checkpoint)
        self.ledger.add_checkpoint(
            checkpoint.digest,
            campaign_id=self.campaign_id,
            last_completed_phase="VARIATION_COMPLETE" if status != "RUNNING" else "VARIATION_ATTEMPT",
            projection_generation=projection_generation,
            artifact_manifest_hash=artifact_manifest_hash,
        )
        return path

    def _validate_resume(self, checkpoint: VariationCheckpoint, *, task: VariationTask, run_id: str, seed: int) -> None:
        """Validate a checkpoint against the durable ledger and private artifacts."""

        checkpoint.validate()
        expected = {
            "campaign_id": self.campaign_id,
            "run_id": run_id,
            "arm_id": self.policy.arm_id,
            "task_id": task.task_id,
            "seed": seed,
            "protocol_digest": VARIATION_PROTOCOL_DIGEST,
            "model_digest": self.model_digest,
            "adapter_digest": self.adapter_digest,
            "retrieval_policy_digest": self.retrieval_policy.digest,
        }
        for field, value in expected.items():
            if getattr(checkpoint, field) != value:
                raise VariationCheckpointError("resume checkpoint {} differs from current immutable input".format(field))
        if checkpoint.ledger_head_event_id != self.ledger.ledger_head_event_id():
            raise VariationCheckpointError("ledger head event differs from the durable Variation checkpoint")
        if checkpoint.ledger_head_hash != self.ledger.ledger_head_hash():
            raise VariationCheckpointError("ledger head differs from the durable Variation checkpoint")
        if checkpoint.attempt_index > self.max_attempts:
            raise VariationCheckpointError("resume checkpoint exceeds the current bounded attempt budget")

    def _validate_durable_attempts(
        self,
        attempts: Sequence[AttemptRecord],
        *,
        workspace: Path,
        task: VariationTask,
        run_id: str,
        seed: int,
        checkpoint: Optional[VariationCheckpoint] = None,
    ) -> None:
        if not attempts or len(attempts) > self.max_attempts:
            raise VariationCheckpointError("durable Variation attempts are outside the frozen budget")
        artifact_store = ContentAddressedArtifactStore(workspace.root / "candidates" / "artifacts")
        previous_candidate: Optional[str] = None
        for expected_index, attempt in enumerate(attempts, start=1):
            if attempt.attempt_index != expected_index:
                raise VariationCheckpointError("durable Variation attempt indices are not contiguous")
            if attempt.candidate_id != self._candidate_id(
                run_id=run_id,
                task_id=task.task_id,
                seed=seed,
                attempt=attempt.attempt_index,
                parent=previous_candidate,
            ):
                # The checkpoint seed is required for resume; the no-checkpoint
                # path is only used with the deterministic run seed below.
                if checkpoint is not None:
                    raise VariationCheckpointError("durable candidate ID is not bound to the frozen run trajectory")
            row = self.ledger.connection.execute(
                "SELECT * FROM candidates WHERE candidate_id=?", (attempt.candidate_id,)
            ).fetchone()
            if row is None or row["campaign_id"] != self.campaign_id or row["run_id"] != run_id or row["task_id"] != task.task_id:
                raise VariationCheckpointError("durable candidate is not bound to the Variation run")
            if row["parent_candidate_id"] != previous_candidate:
                raise VariationCheckpointError("durable candidate parent chain differs from the checkpoint")
            try:
                candidate_payload = json.loads(row["candidate_json"])
            except (TypeError, ValueError) as exc:
                raise VariationCheckpointError("durable candidate payload is not canonical JSON") from exc
            metadata = candidate_payload.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get("candidate_artifact_digest") != attempt.candidate_artifact_digest:
                raise VariationCheckpointError("durable candidate is not bound to its source artifact digest")
            try:
                artifact_store.read(attempt.candidate_artifact_digest)
            except Exception as exc:
                raise VariationCheckpointError("durable candidate source artifact is missing or corrupt") from exc
            if self.ledger.candidate_disposition(attempt.candidate_id) != attempt.disposition:
                raise VariationCheckpointError("durable attempt disposition differs from the ledger projection")
            for receipt_id in attempt.receipt_ids:
                if self.ledger.receipt_by_id(receipt_id) is None:
                    raise VariationCheckpointError("durable attempt references a missing receipt")
            previous_candidate = attempt.candidate_id
        if checkpoint is not None:
            expected_state = digest_for(
                {
                    "attempts": [item.to_dict() for item in attempts],
                    "run_id": run_id,
                    "arm_id": self.policy.arm_id,
                    "task_id": task.task_id,
                    "status": checkpoint.status,
                }
            )
            if checkpoint.state_digest != expected_state:
                raise VariationCheckpointError("checkpoint state digest does not match durable attempts")
            expected_artifacts = digest_for(
                {
                    "artifacts": [
                        {
                            "attempt_index": item.attempt_index,
                            "candidate_id": item.candidate_id,
                            "candidate_artifact_digest": item.candidate_artifact_digest,
                        }
                        for item in attempts
                    ]
                }
            )
            if checkpoint.artifact_manifest_hash != expected_artifacts:
                raise VariationCheckpointError("checkpoint artifact manifest does not match durable attempts")
            expected_projection = digest_for(
                {
                    "ledger_head_event_id": self.ledger.ledger_head_event_id(),
                    "ledger_head_hash": self.ledger.ledger_head_hash(),
                    "arm_id": self.policy.arm_id,
                    "run_id": run_id,
                }
            )
            if checkpoint.projection_generation != expected_projection:
                raise VariationCheckpointError("checkpoint projection generation does not match the ledger head")
            last_disposition = attempts[-1].disposition
            last_diagnostic = attempts[-1].diagnostic_enum
            if last_disposition == "PROMOTED" and last_diagnostic != Diagnostic.PASS.value:
                raise VariationCheckpointError("promoted checkpoint is not bound to a PASS diagnostic")
            if checkpoint.last_candidate_id != attempts[-1].candidate_id or checkpoint.attempt_index != len(attempts):
                raise VariationCheckpointError("checkpoint cursor does not match durable attempts")
            expected_status = "PROMOTED" if last_disposition == "PROMOTED" else None
            if last_diagnostic == Diagnostic.INTERNAL_ERROR.value:
                expected_status = "FAILED"
            elif expected_status is None and len(attempts) == self.max_attempts:
                expected_status = "BUDGET_EXHAUSTED"
            elif expected_status is None and last_disposition == "REJECTED":
                expected_status = "RUNNING"
            if checkpoint.status != expected_status:
                raise VariationCheckpointError("checkpoint terminal status is not derived from the ledger disposition")

    def _durable_attempts(self, *, run_id: str, task_id: str) -> List[AttemptRecord]:
        """Reconstruct attempts from the append-only ledger, never from a cache."""

        attempts: List[AttemptRecord] = []
        for event in self.ledger.current_valid_events():
            if (
                event.get("event_type") != "VARIATION_ATTEMPT"
                or event.get("run_id") != run_id
                or event.get("task_id") != task_id
            ):
                continue
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                raise VariationCheckpointError("durable Variation attempt is missing its payload")
            try:
                attempts.append(
                    AttemptRecord(
                        int(payload["attempt_index"]),
                        str(payload["candidate_id"]),
                        str(payload["candidate_artifact_digest"]),
                        str(payload["retrieval_digest"]),
                        tuple(str(item) for item in payload["evidence_ids"]),
                        str(payload["diagnostic_enum"]),
                        str(payload["resource_bucket"]),
                        str(payload["disposition"]),
                        tuple(str(item) for item in payload["receipt_ids"]),
                        str(event["event_hash"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise VariationCheckpointError("durable Variation attempt has an invalid payload") from exc
        attempts.sort(key=lambda item: item.attempt_index)
        return attempts

    @property
    def retrieval_policy(self) -> EvidenceRetrievalPolicy:
        return retrieval_policy(self.policy.retrieval_policy)

    def _run_trajectory(
        self,
        task: VariationTask,
        *,
        seed: int = 0,
        resume_from: Optional[Path] = None,
    ) -> VariationReport:
        if seed < 0:
            raise VariationConfigurationError("Variation seed must be nonnegative")
        if seed not in self.seed_set:
            raise VariationConfigurationError("run seed is not in the frozen campaign seed set")
        self._validate_task_binding(task)
        self._ensure_campaign(seed=seed)
        run_id = self._run_id(task.task_id, seed)
        workspace = self.isolation.workspace(self.policy.arm_id, run_id)
        self.ledger.create_run(
            run_id,
            campaign_id=self.campaign_id,
            arm=self.policy.arm_id,
            task_id=task.task_id,
            seed=seed,
            parent_checkpoint=None,
            start_state="READY",
            host_role="spark_trainer",
            software_manifest_hash=digest_for({"model": self.model_digest, "protocol": VARIATION_PROTOCOL_DIGEST}),
        )
        checkpoint: Optional[VariationCheckpoint] = None
        if resume_from is not None:
            checkpoint = CheckpointStore(workspace.checkpoints).load(Path(resume_from))
        attempts: List[AttemptRecord] = (
            self._durable_attempts(run_id=run_id, task_id=task.task_id) if checkpoint is not None else []
        )
        if checkpoint is not None:
            self._validate_resume(checkpoint, task=task, run_id=run_id, seed=seed)
            self._validate_durable_attempts(
                attempts,
                workspace=workspace,
                task=task,
                run_id=run_id,
                seed=seed,
                checkpoint=checkpoint,
            )
        start_attempt = checkpoint.attempt_index + 1 if checkpoint is not None else 1
        if checkpoint is not None and checkpoint.status in {"PROMOTED", "BUDGET_EXHAUSTED", "FAILED"}:
            return VariationReport(
                self.campaign_id,
                run_id,
                self.policy.arm_id,
                task.task_id,
                seed,
                tuple(attempts),
                checkpoint.status,
                str(resume_from),
                self.ledger.ledger_head_hash(),
                self.ledger.verify_integrity(),
                self.model_digest,
                self.adapter_digest,
                self.policy.retrieval_policy,
                self.policy.authority_enforced and bool(getattr(self.evaluator, "enforceable", False)),
            )
        last_candidate_id = checkpoint.last_candidate_id if checkpoint is not None else None
        checkpoint_path: Optional[Path] = Path(resume_from) if resume_from is not None else None
        terminal_status = "RUNNING"
        for attempt_index in range(start_attempt, self.max_attempts + 1):
            retrieval = self.retrieval_policy.retrieve(
                self.ledger,
                campaign_id=self.campaign_id,
                arm_id=self.policy.arm_id,
                task_id=task.task_id,
                isolation=self.isolation,
            )
            prompt_digest = digest_for(
                {
                    "schema_version": "egv-variation-prompt-context-v1",
                    "prompt_ids": list(PROMPT_IDS),
                    "prompt_manifest_digest": VARIATION_PROMPT_MANIFEST_DIGEST,
                    "task": task.public_dict(),
                    "seed": seed,
                    "attempt_index": attempt_index,
                    "retrieval_digest": retrieval.evidence_digest,
                }
            )
            context = CandidateContext(
                self.campaign_id,
                run_id,
                seed,
                self.policy.arm_id,
                task.task_id,
                task.family_id,
                task.public_locus,
                task.public_rule_id,
                attempt_index,
                last_candidate_id,
                tuple(record.to_dict() for record in retrieval.records),
                retrieval.evidence_digest,
                self.model_digest,
                self.adapter_digest,
                prompt_digest,
            )
            proposal = self.generator.propose(context)
            proposal.validate(context, source_limit=CANDIDATE_SOURCE_LIMIT)
            candidate_id = self._candidate_id(
                run_id=run_id,
                task_id=task.task_id,
                seed=seed,
                attempt=attempt_index,
                parent=last_candidate_id,
            )
            artifact_path = self._write_candidate_artifact(workspace.root, proposal.source)
            self.ledger.append_candidate(
                candidate_id,
                campaign_id=self.campaign_id,
                run_id=run_id,
                task_id=task.task_id,
                parent_candidate_id=last_candidate_id,
                mutation_family=task.family_id,
                patch_hash=proposal.mutation_digest,
                requested_authority=proposal.requested_authority,
                prompt_hash=prompt_digest,
                model_hash=self.model_digest,
                adapter_hash=self.adapter_digest,
                metadata={
                    "schema_version": "egv-variation-candidate-v1",
                    "arm_id": self.policy.arm_id,
                    "attempt_index": attempt_index,
                    "public_rule_id": task.public_rule_id,
                    "public_locus": task.public_locus,
                    "retrieval_digest": retrieval.evidence_digest,
                    "evidence_ids": list(proposal.evidence_ids),
                    "candidate_artifact_digest": digest_bytes(proposal.source),
                },
            )
            for evidence_id in proposal.evidence_ids:
                self.ledger.append_dependency(
                    evidence_id,
                    candidate_id,
                    edge_type="EVIDENCE_USED",
                    campaign_id=self.campaign_id,
                    run_id=run_id,
                    task_id=task.task_id,
                    idempotency_key="evidence-used:{}:{}".format(evidence_id, candidate_id),
                )
            result = self.evaluator.evaluate(
                candidate_id=candidate_id,
                task_id=task.task_id,
                source=proposal.source,
                opaque_input=task.evaluator_input,
                requested_authority=proposal.requested_authority,
                declared_locus=proposal.declared_locus,
                candidate_source_path=artifact_path,
            )
            attempt = self._materialize_result(
                task=task,
                run_id=run_id,
                candidate_id=candidate_id,
                proposal=proposal,
                retrieval=retrieval,
                result=result,
                attempt_index=attempt_index,
            )
            attempts.append(attempt)
            last_candidate_id = candidate_id
            if result.infrastructure_loss or result.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value:
                terminal_status = "FAILED"
                checkpoint_path = self._checkpoint(
                    workspace=workspace,
                    task=task,
                    run_id=run_id,
                    seed=seed,
                    attempts=attempts,
                    attempt=attempt,
                    status=terminal_status,
                )
                break
            if attempt.disposition == "PROMOTED":
                terminal_status = "PROMOTED"
                checkpoint_path = self._checkpoint(
                    workspace=workspace,
                    task=task,
                    run_id=run_id,
                    seed=seed,
                    attempts=attempts,
                    attempt=attempt,
                    status=terminal_status,
                )
                break
            if attempt_index == self.max_attempts:
                terminal_status = "BUDGET_EXHAUSTED"
            checkpoint_path = self._checkpoint(
                workspace=workspace,
                task=task,
                run_id=run_id,
                seed=seed,
                attempts=attempts,
                attempt=attempt,
                status=terminal_status,
            )
        if checkpoint_path is None:
            raise VariationCheckpointError("Variation loop ended without a durable checkpoint")
        return VariationReport(
            self.campaign_id,
            run_id,
            self.policy.arm_id,
            task.task_id,
            seed,
            tuple(attempts),
            terminal_status,
            str(checkpoint_path),
            self.ledger.ledger_head_hash(),
            self.ledger.verify_integrity(),
            self.model_digest,
            self.adapter_digest,
            self.policy.retrieval_policy,
            self.policy.authority_enforced and bool(getattr(self.evaluator, "enforceable", False)),
        )


class _ProductionBoundedCandidateLoop(BoundedCandidateLoop):
    """Concrete production trajectory with no fixture execution path."""

    __slots__ = ("_production_layout_marker",)

    def _validate_production_boundary(self) -> None:
        super()._validate_production_boundary()

    def run(
        self,
        task: VariationTask,
        *,
        seed: int = 0,
        resume_from: Optional[Path] = None,
    ) -> VariationReport:
        self._validate_production_boundary()
        self._validate_identity()
        self._validate_budget()
        from .remote import RemoteControllerEvaluationGateway

        if type(self.evaluator) is RemoteControllerEvaluationGateway and task.evaluator_input is not None:
            raise VariationConfigurationError("remote Variation trajectories require a public-only task")
        return self._run_trajectory(task, seed=seed, resume_from=resume_from)


class _FixtureBoundedCandidateLoop(BoundedCandidateLoop):
    """Concrete fixture trajectory; this is the only fixture execution path."""

    __slots__ = ("_fixture_layout_marker",)

    def _validate_production_boundary(self) -> None:
        self._validate_fixture_boundary()

    def run(
        self,
        task: VariationTask,
        *,
        seed: int = 0,
        resume_from: Optional[Path] = None,
    ) -> VariationReport:
        self._validate_production_boundary()
        self._validate_identity()
        self._validate_budget()
        return self._run_trajectory(task, seed=seed, resume_from=resume_from)


VariationRunner = BoundedCandidateLoop


__all__ = [
    "AttemptRecord",
    "BoundedCandidateLoop",
    "CANDIDATE_SOURCE_LIMIT",
    "ControllerEvaluationGateway",
    "EvaluationGateway",
    "MAX_CANDIDATE_ATTEMPTS",
    "VARIATION_PROTOCOL_DIGEST",
    "VARIATION_PROTOCOL_SCHEMA",
    "VariationReport",
    "VariationRunner",
    "VariationTask",
]
