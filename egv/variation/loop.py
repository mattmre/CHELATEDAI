"""Bounded candidate loop integrated with the Evidence and Evaluation slices."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple
from weakref import WeakKeyDictionary

from ..canonical import GENESIS_HASH, canonical_json, content_id, digest_bytes, digest_for, failure_family_root
from ..evaluation.artifacts import ContentAddressedArtifactStore
from ..evaluation.controller import EvaluationResult, EvaluatorController, HiddenEvaluatorRunner
from ..evaluation.dataset import MicroRepo
from ..evaluation.diagnostics import Diagnostic, validate_diagnostic, validate_disposition, validate_resource_bucket
from ..evaluation.prompts import PROMPT_IDS, PromptRegistry
from ..evaluation.sandbox import DockerCandidateSandbox, DockerSandboxConfig
from ..ledger import EvidenceLedger
from ..identities import commissioning_run_id
from ..receipts import receipt_hash
from .adapter import SealedAdapterArtifact, validate_applied_peft_model
from .arms import ArmIsolation, ArmPolicy, arm_policy
from .checkpoint import CheckpointStore, VariationCheckpoint
from .errors import VariationBudgetError, VariationCheckpointError, VariationConfigurationError, VariationDependencyError
from .generator import (
    CandidateContext,
    CandidateGenerationFailure,
    CandidateGenerator,
    CandidateProposal,
    ModelCandidateGenerator,
)
from .model import ADAPTER_ATTESTATION_SCHEMA, AdapterApplicationAttestation, MODEL_REVISION
from .private import LegacyGenerationBundle, PrivateTrajectoryStore
from .retrieval import EvidenceRetrievalPolicy, RetrievalResult, RetrievedEvidence, retrieval_policy
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
    private_store: Any


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


class SourceContractBudgetExhausted(VariationBudgetError):
    """The final bounded attempt failed the immutable source-only contract."""

    def __init__(self, *, run_id: str, failure_count: int, last_failure_digest: str) -> None:
        super().__init__("source-only response contract exhausted the bounded attempt budget")
        self.run_id = run_id
        self.failure_count = failure_count
        self.last_failure_digest = last_failure_digest


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
        if name in {
            "initial_source", "initial_source_digest", "response_contract_digest",
            "generation_profile_digest",
        } and "_source_contract_seal" in self.__dict__:
            raise AttributeError("Variation source response contract is immutable after construction")
        if name in {"fixture_mode", "evaluator", "generator", "isolation", "private_store"} and _runtime_identity_registered(self):
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
        private_store: Optional[PrivateTrajectoryStore] = None,
        initial_source: Optional[bytes] = None,
        response_contract_digest: Optional[str] = None,
        generation_profile_digest: Optional[str] = None,
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
        if private_store is not None and type(private_store) is not PrivateTrajectoryStore:
            raise VariationDependencyError("private trajectory persistence requires the exact store type")
        self.private_store = private_store
        generator_response_contract = getattr(generator, "response_contract", "closed-json-v1")
        generator_response_contract_digest = getattr(generator, "response_contract_digest", None)
        if generator_response_contract == "closed-json-v1":
            if (
                initial_source is not None
                or response_contract_digest is not None
                or generation_profile_digest is not None
            ):
                raise VariationConfigurationError("closed-JSON Variation cannot carry source-only inputs")
            self.initial_source = None
            self.initial_source_digest = None
            self.response_contract_digest = generator_response_contract_digest
            self.generation_profile_digest = None
        else:
            generator_generation_profile_digest = getattr(generator, "generation_profile_digest", None)
            if (
                not isinstance(initial_source, bytes)
                or not initial_source
                or len(initial_source) > CANDIDATE_SOURCE_LIMIT
                or type(private_store) is not PrivateTrajectoryStore
                or not isinstance(response_contract_digest, str)
                or response_contract_digest != generator_response_contract_digest
                or not isinstance(generation_profile_digest, str)
                or generation_profile_digest != generator_generation_profile_digest
            ):
                raise VariationConfigurationError(
                    "source-only Variation requires exact source bytes, generation profile, contract digest, and private evidence store"
                )
            try:
                initial_source.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise VariationConfigurationError("source-only Variation source is not UTF-8") from exc
            self.initial_source = bytes(initial_source)
            self.initial_source_digest = digest_bytes(initial_source)
            self.response_contract_digest = response_contract_digest
            self.generation_profile_digest = generation_profile_digest
        self._source_contract_seal = digest_for({
            "response_contract": generator_response_contract,
            "response_contract_digest": self.response_contract_digest,
            "initial_source_digest": self.initial_source_digest,
            "generation_profile_digest": self.generation_profile_digest,
        })
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
                private_store=self.private_store,
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
        private_store = getattr(self, "private_store", missing)
        if any(value is missing for value in (fixture_mode, evaluator, generator, isolation, private_store)):
            raise VariationDependencyError("Variation runtime identity fields are missing")
        if fixture_mode != record.fixture_mode:
            raise VariationDependencyError("Variation fixture mode changed after construction")
        if evaluator is not record.evaluator:
            raise VariationDependencyError("Variation evaluator identity changed after construction")
        if generator is not record.generator:
            raise VariationDependencyError("Variation generator identity changed after construction")
        if isolation is not record.isolation:
            raise VariationDependencyError("Variation isolation identity changed after construction")
        if private_store is not record.private_store:
            raise VariationDependencyError("Variation private trajectory store identity changed after construction")
        expected_source_seal = digest_for({
            "response_contract": getattr(self.generator, "response_contract", "closed-json-v1"),
            "response_contract_digest": self.response_contract_digest,
            "initial_source_digest": (
                digest_bytes(self.initial_source) if isinstance(self.initial_source, bytes) else None
            ),
            "generation_profile_digest": self.generation_profile_digest,
        })
        if expected_source_seal != self._source_contract_seal:
            raise VariationDependencyError("Variation source response contract changed after construction")

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
        return commissioning_run_id(
            campaign_id=self.campaign_id,
            task_id=task_id,
            arm_id=self.policy.arm_id,
            seed=seed,
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

    @staticmethod
    def _candidate_artifact_path(workspace: Path, artifact_digest: str) -> Path:
        return (
            workspace
            / "candidates"
            / "artifacts"
            / "blobs"
            / "sha256"
            / artifact_digest[:2]
            / artifact_digest[2:4]
            / artifact_digest
        )

    def _candidate_row(self, candidate_id: str) -> Optional[Mapping[str, Any]]:
        row = self.ledger.connection.execute(
            """SELECT c.*,e.sequence AS event_sequence,e.event_hash AS candidate_event_hash
               FROM candidates c JOIN events e ON e.event_id=c.event_id
               WHERE c.candidate_id=?""",
            (candidate_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def _recovery_retrieval(
        self,
        *,
        context: CandidateContext,
        task: VariationTask,
        candidate_row: Optional[Mapping[str, Any]],
    ) -> RetrievalResult:
        """Validate the exact retrieval snapshot used by a pending generation."""

        current = self.retrieval_policy.retrieve(
            self.ledger,
            campaign_id=self.campaign_id,
            arm_id=self.policy.arm_id,
            task_id=task.task_id,
            isolation=self.isolation,
        )
        records = current.records
        if candidate_row is not None:
            cutoff = candidate_row.get("event_sequence")
            if not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 1:
                raise VariationCheckpointError("pending candidate lacks its immutable ledger cutoff")
            sequences = {
                str(event["event_id"]): int(event["sequence"])
                for event in self.ledger.current_valid_events()
            }
            try:
                records = tuple(record for record in records if sequences[record.event_id] < cutoff)
            except KeyError as exc:
                raise VariationCheckpointError("pending retrieval references an unavailable ledger event") from exc
        recovered = RetrievalResult(
            self.policy.retrieval_policy,
            self.policy.arm_id,
            task.task_id,
            tuple(records),
            digest_for([record.to_dict() for record in records]),
        )
        try:
            persisted = tuple(
                RetrievedEvidence(
                    event_id=str(value["event_id"]),
                    event_type=str(value["event_type"]),
                    subject_id=value.get("subject_id"),
                    task_id=value.get("task_id"),
                    recorded_disposition=str(value["recorded_disposition"]),
                    diagnostic_enum=value.get("diagnostic_enum"),
                    failure_family_root=value.get("failure_family_root"),
                )
                for value in context.retrieval_records
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VariationCheckpointError("pending retrieval record is malformed") from exc
        if (
            any(record.to_dict() != dict(value) for record, value in zip(persisted, context.retrieval_records))
            or len(persisted) != len(context.retrieval_records)
            or tuple(record.to_dict() for record in persisted) != tuple(record.to_dict() for record in recovered.records)
            or context.retrieval_digest != recovered.evidence_digest
        ):
            raise VariationCheckpointError("pending generation retrieval differs from the immutable ledger snapshot")
        return recovered

    def _validate_recovery_context(
        self,
        *,
        context: CandidateContext,
        task: VariationTask,
        run_id: str,
        seed: int,
        attempt_index: int,
        parent_candidate_id: Optional[str],
        retrieval: RetrievalResult,
    ) -> None:
        expected = {
            "campaign_id": self.campaign_id,
            "run_id": run_id,
            "seed": seed,
            "arm_id": self.policy.arm_id,
            "task_id": task.task_id,
            "family_id": task.family_id,
            "public_locus": task.public_locus,
            "public_rule_id": task.public_rule_id,
            "attempt_index": attempt_index,
            "parent_candidate_id": parent_candidate_id,
            "retrieval_records": tuple(record.to_dict() for record in retrieval.records),
            "retrieval_digest": retrieval.evidence_digest,
            "model_digest": self.model_digest,
            "adapter_digest": self.adapter_digest,
            "task_statement": task.task_statement,
            "initial_source": self.initial_source.decode("utf-8") if self.initial_source is not None else None,
            "initial_source_digest": self.initial_source_digest,
            "response_contract": getattr(self.generator, "response_contract", "closed-json-v1"),
            "response_contract_digest": self.response_contract_digest,
            "generation_profile_digest": self.generation_profile_digest,
        }
        if any(getattr(context, field) != value for field, value in expected.items()):
            raise VariationCheckpointError("pending generation context differs from the frozen trajectory")
        try:
            expected_prompt_digest = self.generator.prompt_digest_for(context)
        except Exception as exc:
            raise VariationCheckpointError("pending generation prompt cannot be reproduced") from exc
        if context.prompt_digest != expected_prompt_digest:
            raise VariationCheckpointError("pending generation prompt differs from the frozen profile")

    def _persist_candidate(
        self,
        *,
        workspace: Path,
        task: VariationTask,
        run_id: str,
        candidate_id: str,
        context: CandidateContext,
        proposal: CandidateProposal,
        retrieval: RetrievalResult,
        generation_evidence_digest: Optional[str],
        recovering: bool,
    ) -> str:
        """Persist or exactly reconcile the pre-evaluation candidate boundary."""

        artifact_digest = digest_bytes(proposal.source)
        existing = self._candidate_row(candidate_id)
        artifact_store = ContentAddressedArtifactStore(workspace / "candidates" / "artifacts")
        if existing is not None:
            try:
                durable_source = artifact_store.read(artifact_digest)
            except Exception as exc:
                raise VariationCheckpointError("pending candidate source artifact is missing or corrupt") from exc
            if durable_source != proposal.source:
                raise VariationCheckpointError("pending candidate source artifact differs from generation evidence")
            artifact_path = str(self._candidate_artifact_path(workspace, artifact_digest))
        else:
            artifact_path = self._write_candidate_artifact(workspace, proposal.source)
        metadata = {
            "schema_version": "egv-variation-candidate-v1",
            "arm_id": self.policy.arm_id,
            "attempt_index": context.attempt_index,
            "public_rule_id": task.public_rule_id,
            "public_locus": task.public_locus,
            "retrieval_digest": retrieval.evidence_digest,
            "evidence_ids": list(proposal.evidence_ids),
            "candidate_artifact_digest": artifact_digest,
            "response_contract": context.response_contract,
            "response_contract_digest": context.response_contract_digest,
            "generation_evidence_digest": generation_evidence_digest,
            "generation_profile_digest": context.generation_profile_digest,
        }
        try:
            self.ledger.append_candidate(
                candidate_id,
                campaign_id=self.campaign_id,
                run_id=run_id,
                task_id=task.task_id,
                parent_candidate_id=context.parent_candidate_id,
                mutation_family=task.family_id,
                patch_hash=proposal.mutation_digest,
                requested_authority=proposal.requested_authority,
                prompt_hash=context.prompt_digest,
                model_hash=self.model_digest,
                adapter_hash=self.adapter_digest,
                metadata=metadata,
            )
        except Exception as exc:
            if recovering:
                raise VariationCheckpointError("pending candidate ledger row conflicts with generation evidence") from exc
            raise
        if self.private_store is not None:
            self.private_store.record(candidate_id=candidate_id, context=context, source=proposal.source)
        expected_dependencies = {
            (evidence_id, candidate_id, "EVIDENCE_USED")
            for evidence_id in proposal.evidence_ids
        }
        existing_dependencies = {
            (str(row["parent_id"]), str(row["child_id"]), str(row["edge_type"]))
            for row in self.ledger.connection.execute(
                "SELECT parent_id,child_id,edge_type FROM dependencies WHERE child_id=?",
                (candidate_id,),
            ).fetchall()
        }
        if not existing_dependencies.issubset(expected_dependencies):
            raise VariationCheckpointError("pending candidate carries an unexpected dependency edge")
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
        final_dependencies = {
            (str(row["parent_id"]), str(row["child_id"]), str(row["edge_type"]))
            for row in self.ledger.connection.execute(
                "SELECT parent_id,child_id,edge_type FROM dependencies WHERE child_id=?",
                (candidate_id,),
            ).fetchall()
        }
        if final_dependencies != expected_dependencies:
            raise VariationCheckpointError("pending candidate dependency set is incomplete or conflicting")
        return artifact_path

    def _receipt_for(self, receipt_ids: Sequence[str], receipt_type: str) -> Optional[Dict[str, Any]]:
        for receipt_id in receipt_ids:
            row = self.ledger.receipt_by_id(receipt_id)
            if row is not None and row["receipt"].get("receipt_type") == receipt_type:
                return row["receipt"]
        return None

    def _validate_attempt_receipts(
        self,
        attempt: AttemptRecord,
        *,
        task: VariationTask,
        run_id: str,
    ) -> None:
        """Bind a durable attempt to its exact ordered authenticated receipt suffix."""

        receipts = []
        for receipt_id in attempt.receipt_ids:
            stored = self.ledger.receipt_by_id(receipt_id)
            if stored is None:
                raise VariationCheckpointError("durable attempt references a missing receipt")
            receipt = stored.get("receipt")
            if not isinstance(receipt, Mapping) or receipt.get("receipt_id") != receipt_id:
                raise VariationCheckpointError("durable attempt receipt projection is malformed")
            receipts.append(dict(receipt))
        receipt_types = [receipt.get("receipt_type") for receipt in receipts]
        if receipt_types not in (["AUTHORITY"], ["AUTHORITY", "VERDICT", "EFFECT"]):
            raise VariationCheckpointError("durable attempt receipt suffix has an invalid type order")
        for index, receipt in enumerate(receipts):
            expected_type = receipt_types[index]
            expected_request_id = "{}-{}-{}".format(
                "fixture" if self.fixture_mode else "request",
                str(expected_type).lower(),
                attempt.candidate_id,
            )
            if (
                receipt.get("campaign_id") != self.campaign_id
                or receipt.get("task_id") != task.task_id
                or receipt.get("candidate_id") != attempt.candidate_id
                or receipt.get("candidate_artifact_digest") != attempt.candidate_artifact_digest
                or receipt.get("protocol_digest") != VARIATION_PROTOCOL_DIGEST
                or receipt.get("policy_digest") != self.policy_digest
                or receipt.get("evaluator_digest") != getattr(self.evaluator, "evaluator_digest", None)
                or receipt.get("task_family") != task.family_id
                or receipt.get("normalized_public_locus") != task.public_locus
                or receipt.get("public_rule_id") != task.public_rule_id
                or receipt.get("request_id") != expected_request_id
            ):
                raise VariationCheckpointError(
                    "durable attempt receipt suffix crossed its immutable candidate binding"
                )
            if type(self.evaluator) is not ControllerEvaluationGateway and receipt.get("run_id") != run_id:
                raise VariationCheckpointError("durable attempt receipt suffix crossed its run binding")
            if type(self.evaluator) is RemoteControllerEvaluationGateway:
                if receipt.get("arm_policy_digest") != self.policy.digest:
                    raise VariationCheckpointError(
                        "durable remote receipt differs from the frozen arm policy"
                    )
            elif receipt.get("arm_policy_digest") is not None and (
                receipt.get("arm_policy_digest") != self.policy.digest
            ):
                raise VariationCheckpointError("durable attempt receipt carries a crossed arm policy")
            if index:
                if (
                    receipt.get("sequence") != receipts[index - 1].get("sequence") + 1
                    or receipt.get("previous_receipt_hash") != receipt_hash(receipts[index - 1])
                ):
                    raise VariationCheckpointError(
                        "durable attempt receipt suffix is not contiguous and ordered"
                    )
        if len(receipts) == 1:
            authority = receipts[0]
            allowed = {
                Diagnostic.PROTOCOL_VIOLATION.value: "REJECTED",
                Diagnostic.MUTATION_LOCUS_VIOLATION.value: "REJECTED",
                Diagnostic.AUTHORITY_DENIED.value: "ABSTAINED",
                Diagnostic.INTERNAL_ERROR.value: "ABSTAINED",
            }
            if (
                authority.get("decision") != "DENY"
                or attempt.diagnostic_enum not in allowed
                or attempt.disposition != allowed[attempt.diagnostic_enum]
                or attempt.resource_bucket != "UNDER_25"
                or (
                    attempt.diagnostic_enum == Diagnostic.AUTHORITY_DENIED.value
                    and authority.get("diagnostic_enum") is not None
                )
                or (
                    attempt.diagnostic_enum != Diagnostic.AUTHORITY_DENIED.value
                    and authority.get("diagnostic_enum") != attempt.diagnostic_enum
                )
            ):
                raise VariationCheckpointError(
                    "durable authority-only receipt disagrees with its attempt result"
                )
        else:
            authority, verdict, effect = receipts
            infrastructure = attempt.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value
            expected_verdict = (
                "ERROR" if infrastructure
                else "PASS" if attempt.diagnostic_enum == Diagnostic.PASS.value
                else "FAIL"
            )
            expected_disposition = (
                "ABSTAINED" if infrastructure
                else "PROMOTED" if attempt.diagnostic_enum == Diagnostic.PASS.value
                else "REJECTED"
            )
            if (
                authority.get("decision") != "ALLOW"
                or verdict.get("decision") != expected_verdict
                or effect.get("decision") != ("ERROR" if infrastructure else "ALLOW")
                or verdict.get("diagnostic_enum") != attempt.diagnostic_enum
                or effect.get("diagnostic_enum") != attempt.diagnostic_enum
                or verdict.get("resource_bucket") != attempt.resource_bucket
                or verdict.get("output_digest") != effect.get("output_digest")
                or effect.get("normalized_action_hash")
                != digest_for({"action": "execute_candidate", "locus": task.public_locus})
                or attempt.disposition != expected_disposition
            ):
                raise VariationCheckpointError(
                    "durable receipt suffix disagrees with its attempt result"
                )
        incident_receipts = [
            receipt for receipt in receipts
            if receipt.get("infrastructure_incident_id") is not None
            or receipt.get("failure_family_root") is not None
        ]
        if attempt.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value:
            if not incident_receipts:
                raise VariationCheckpointError("durable INTERNAL_ERROR receipt suffix lacks an incident")
            incidents = {receipt.get("infrastructure_incident_id") for receipt in incident_receipts}
            roots = {receipt.get("failure_family_root") for receipt in incident_receipts}
            if len(incidents) != 1 or len(roots) != 1 or None in incidents or None in roots:
                raise VariationCheckpointError("durable infrastructure receipt suffix is inconsistent")
            incident = next(iter(incidents))
            expected_root = failure_family_root(
                task.family_id,
                attempt.diagnostic_enum,
                task.public_locus,
                task.public_rule_id,
                infrastructure_incident_id=incident,
            )
            if roots != {expected_root}:
                raise VariationCheckpointError("durable infrastructure receipt root is invalid")
        elif incident_receipts:
            raise VariationCheckpointError("durable non-infrastructure receipt carries an incident")

    def _legacy_pre_materialization_receipt_ids(
        self,
        bundle: LegacyGenerationBundle,
        *,
        task: VariationTask,
        run_id: str,
    ) -> Tuple[str, ...]:
        """Read-only proof for the one preserved pre-materialization orphan."""

        if type(bundle) is not LegacyGenerationBundle:
            raise VariationCheckpointError("legacy generation migration bundle type is invalid")
        candidate_id = bundle.candidate_id
        candidate_ids = [
            str(row["candidate_id"])
            for row in self.ledger.connection.execute(
                "SELECT candidate_id FROM candidates ORDER BY candidate_id",
            ).fetchall()
        ]
        receipt_rows = self.ledger.connection.execute(
            "SELECT receipt_id,sequence,payload_json FROM receipts ORDER BY sequence",
        ).fetchall()
        try:
            receipts = [json.loads(row["payload_json"]) for row in receipt_rows]
        except (TypeError, ValueError) as exc:
            raise VariationCheckpointError("legacy evaluator receipt suffix is unreadable") from exc
        projection_counts = {
            "campaigns": self.ledger.connection.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0] != 1,
            "runs": self.ledger.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] != 1,
            "verdicts": self.ledger.connection.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0],
            "effects": self.ledger.connection.execute("SELECT COUNT(*) FROM effect_receipts").fetchone()[0],
            "checkpoints": self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0],
            "corrections": self.ledger.connection.execute("SELECT COUNT(*) FROM corrections").fetchone()[0],
            "retractions": self.ledger.connection.execute("SELECT COUNT(*) FROM retractions").fetchone()[0],
            "quarantines": self.ledger.connection.execute("SELECT COUNT(*) FROM quarantines").fetchone()[0],
            "projection_queue": self.ledger.connection.execute(
                "SELECT COUNT(*) FROM projection_queue"
            ).fetchone()[0],
            "blobs": self.ledger.connection.execute("SELECT COUNT(*) FROM blobs").fetchone()[0],
            "quarantined_meta": self.ledger.connection.execute(
                "SELECT value FROM meta WHERE key='quarantined'"
            ).fetchone()[0]
            != "0",
            "attempts": sum(
                1
                for event in self.ledger.current_valid_events()
                if event.get("event_type") == "VARIATION_ATTEMPT"
                and event.get("run_id") == run_id
                and event.get("task_id") == task.task_id
            ),
        }
        expected_dependencies = {
            (evidence_id, candidate_id, "EVIDENCE_USED")
            for evidence_id in bundle.evidence.proposal.evidence_ids
        }
        actual_dependencies = {
            (str(row["parent_id"]), str(row["child_id"]), str(row["edge_type"]))
            for row in self.ledger.connection.execute(
                "SELECT parent_id,child_id,edge_type FROM dependencies",
            ).fetchall()
        }
        receipt_ids = tuple(str(row["receipt_id"]) for row in receipt_rows)
        current_events = self.ledger.current_valid_events()
        if (
            candidate_ids != [candidate_id]
            or bundle.context.attempt_index != 1
            or bundle.context.parent_candidate_id is not None
            or bundle.context.run_id != run_id
            or bundle.context.task_id != task.task_id
            or bundle.context.arm_id != self.policy.arm_id
            or len(receipts) != 3
            or [int(row["sequence"]) for row in receipt_rows] != [1, 2, 3]
            or [receipt.get("receipt_type") for receipt in receipts]
            != ["AUTHORITY", "VERDICT", "EFFECT"]
            or receipts[0].get("previous_receipt_hash") != GENESIS_HASH
            or any(receipt.get("receipt_id") != receipt_id for receipt, receipt_id in zip(receipts, receipt_ids))
            or any(projection_counts.values())
            or actual_dependencies != expected_dependencies
            or [event.get("event_type") for event in current_events]
            != ["CAMPAIGN", "RUN", "CANDIDATE", "RECEIPT", "RECEIPT", "RECEIPT"]
        ):
            raise VariationCheckpointError(
                "legacy generation migration is outside the preserved pre-materialization boundary"
            )
        return receipt_ids

    def _validate_legacy_cached_result(
        self,
        bundle: LegacyGenerationBundle,
        *,
        task: VariationTask,
        run_id: str,
        retrieval: RetrievalResult,
        result: EvaluationResult,
        expected_receipt_ids: Tuple[str, ...],
        ledger_head_before_replay: str,
    ) -> str:
        """Require an exact cache replay before legacy private evidence mutates."""

        proposal = bundle.evidence.proposal
        if (
            self.ledger.ledger_head_hash() != ledger_head_before_replay
            or result.candidate_id != bundle.candidate_id
            or result.task_id != task.task_id
            or result.candidate_artifact_digest != digest_bytes(proposal.source)
            or tuple(result.receipt_ids) != expected_receipt_ids
            or len(set(result.receipt_ids)) != len(result.receipt_ids)
        ):
            raise VariationCheckpointError(
                "legacy evaluator replay changed or crossed the preserved orphan"
            )
        try:
            validate_diagnostic(result.diagnostic_enum)
            validate_resource_bucket(result.resource_bucket)
            validate_disposition(result.disposition)
        except ValueError as exc:
            raise VariationCheckpointError(
                "legacy evaluator replay uses a value outside the closed contract"
            ) from exc
        if result.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value:
            if not result.infrastructure_loss or not result.infrastructure_incident_id or not result.failure_family_root:
                raise VariationCheckpointError("legacy evaluator replay lacks its infrastructure proof")
        elif result.infrastructure_loss or result.infrastructure_incident_id or result.failure_family_root:
            raise VariationCheckpointError("legacy evaluator replay carries crossed infrastructure proof")
        provisional = AttemptRecord(
            bundle.context.attempt_index,
            bundle.candidate_id,
            result.candidate_artifact_digest,
            retrieval.evidence_digest,
            tuple(proposal.evidence_ids),
            result.diagnostic_enum,
            result.resource_bucket,
            result.disposition,
            tuple(result.receipt_ids),
            ledger_head_before_replay,
        )
        self._validate_attempt_receipts(provisional, task=task, run_id=run_id)
        projected_disposition = self.ledger.candidate_disposition(bundle.candidate_id)
        allowed_projection = {
            "PROMOTED": {"PROMOTED", "ABSTAINED"},
            "REJECTED": {"REJECTED"},
            "ABSTAINED": {"ABSTAINED"},
        }.get(result.disposition)
        if allowed_projection is None or projected_disposition not in allowed_projection:
            raise VariationCheckpointError(
                "legacy evaluator replay differs from the authenticated receipt projection"
            )
        try:
            self.ledger.verify_integrity()
        except Exception as exc:
            raise VariationCheckpointError(
                "legacy evaluator replay failed full ledger integrity validation"
            ) from exc
        return digest_for({
            "schema_version": "egv-legacy-replay-proof-v1",
            "candidate_id": bundle.candidate_id,
            "generation_record_digest": bundle.generation_record_digest,
            "trajectory_record_digest": bundle.trajectory_record_digest,
            "orphan_artifact_digests": list(bundle.orphan_artifact_digests),
            "ledger_head_hash": ledger_head_before_replay,
            "receipt_ids": list(result.receipt_ids),
            "candidate_artifact_digest": result.candidate_artifact_digest,
            "diagnostic_enum": result.diagnostic_enum,
            "resource_bucket": result.resource_bucket,
            "disposition": result.disposition,
            "output_digest": result.output_digest,
            "infrastructure_loss": result.infrastructure_loss,
            "infrastructure_incident_id": result.infrastructure_incident_id,
            "failure_family_root": result.failure_family_root,
        })

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
        # Evaluation gateways authenticate and admit signed receipts. This
        # loop boundary deliberately materializes their verdict/effect rows;
        # candidate_disposition must not be read from a bare gateway call.
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
        failures = self._source_contract_failure_records(run_id=run_id, task_id=task.task_id)
        state_value = {
            "attempts": [item.to_dict() for item in attempts],
            "run_id": run_id,
            "arm_id": self.policy.arm_id,
            "task_id": task.task_id,
            "status": status,
        }
        if failures:
            state_value["source_contract_failures"] = [dict(item) for item in failures]
        state_digest = digest_for(state_value)
        projection_generation = digest_for(
            {
                "ledger_head_event_id": self.ledger.ledger_head_event_id(),
                "ledger_head_hash": self.ledger.ledger_head_hash(),
                "arm_id": self.policy.arm_id,
                "run_id": run_id,
            }
        )
        artifact_value = {
            "artifacts": [
                    {
                        "attempt_index": item.attempt_index,
                        "candidate_id": item.candidate_id,
                        "candidate_artifact_digest": item.candidate_artifact_digest,
                    }
                    for item in attempts
                ]
        }
        if failures:
            artifact_value["source_contract_failure_digests"] = [
                item["record_digest"] for item in failures
            ]
        artifact_manifest_hash = digest_for(artifact_value)
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
        durable_head = self.ledger.connection.execute(
            "SELECT event_hash FROM events WHERE event_id=?",
            (checkpoint.ledger_head_event_id,),
        ).fetchone()
        if durable_head is None or str(durable_head["event_hash"]) != checkpoint.ledger_head_hash:
            raise VariationCheckpointError("checkpoint ledger head is absent from the durable event chain")
        if checkpoint.attempt_index > self.max_attempts:
            raise VariationCheckpointError("resume checkpoint exceeds the current bounded attempt budget")

    def _reconcile_checkpoint_projection(
        self,
        checkpoint: VariationCheckpoint,
        *,
        allow_missing: bool,
    ) -> None:
        """Validate, or safely finish, the file-to-ledger checkpoint boundary."""

        row = self.ledger.connection.execute(
            "SELECT * FROM checkpoints WHERE checkpoint_id=?",
            (checkpoint.digest,),
        ).fetchone()
        if row is None:
            if not allow_missing or (
                self.ledger.ledger_head_event_id() != checkpoint.ledger_head_event_id
                or self.ledger.ledger_head_hash() != checkpoint.ledger_head_hash
            ):
                raise VariationCheckpointError(
                    "checkpoint file lacks its exact durable ledger projection"
                )
            self.ledger.add_checkpoint(
                checkpoint.digest,
                campaign_id=self.campaign_id,
                last_completed_phase=(
                    "VARIATION_COMPLETE" if checkpoint.status != "RUNNING" else "VARIATION_ATTEMPT"
                ),
                projection_generation=checkpoint.projection_generation,
                artifact_manifest_hash=checkpoint.artifact_manifest_hash,
            )
            row = self.ledger.connection.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id=?",
                (checkpoint.digest,),
            ).fetchone()
        expected = {
            "checkpoint_id": checkpoint.digest,
            "campaign_id": self.campaign_id,
            "last_completed_phase": (
                "VARIATION_COMPLETE" if checkpoint.status != "RUNNING" else "VARIATION_ATTEMPT"
            ),
            "last_durable_event_id": checkpoint.ledger_head_event_id,
            "ledger_hash": checkpoint.ledger_head_hash,
            "projection_generation": checkpoint.projection_generation,
            "artifact_manifest_hash": checkpoint.artifact_manifest_hash,
        }
        if row is None or any(row[key] != value for key, value in expected.items()):
            raise VariationCheckpointError("checkpoint ledger projection differs from its immutable file")

    def _source_contract_failure_records(
        self,
        *,
        run_id: str,
        task_id: str,
    ) -> Tuple[Mapping[str, Any], ...]:
        if getattr(self.generator, "response_contract", "closed-json-v1") == "closed-json-v1":
            return tuple()
        if type(self.private_store) is not PrivateTrajectoryStore:
            raise VariationCheckpointError("source-only resume lacks its private evidence store")
        return self.private_store.source_contract_failures(
            run_id=run_id,
            task_id=task_id,
            arm_id=self.policy.arm_id,
        )

    def _validate_durable_attempts(
        self,
        attempts: Sequence[AttemptRecord],
        *,
        workspace: Path,
        task: VariationTask,
        run_id: str,
        seed: int,
        failures: Sequence[Mapping[str, Any]] = (),
        checkpoint: Optional[VariationCheckpoint] = None,
    ) -> None:
        if (not attempts and not failures) or len(attempts) + len(failures) > self.max_attempts:
            raise VariationCheckpointError("durable Variation attempts are outside the frozen budget")
        artifact_store = ContentAddressedArtifactStore(workspace.root / "candidates" / "artifacts")
        candidate_indices = {attempt.attempt_index for attempt in attempts}
        failure_by_index = {int(item["attempt_index"]): item for item in failures}
        if len(candidate_indices) != len(attempts) or candidate_indices & set(failure_by_index):
            raise VariationCheckpointError("durable candidate and source-failure attempt indices conflict")
        all_indices = sorted(candidate_indices | set(failure_by_index))
        if all_indices != list(range(1, max(all_indices) + 1)):
            raise VariationCheckpointError("durable Variation attempt indices are not contiguous")
        previous_candidate: Optional[str] = None
        attempts_by_index = {attempt.attempt_index: attempt for attempt in attempts}
        for expected_index in all_indices:
            failure = failure_by_index.get(expected_index)
            if failure is not None:
                expected_failure_id = self._candidate_id(
                    run_id=run_id,
                    task_id=task.task_id,
                    seed=seed,
                    attempt=expected_index,
                    parent=previous_candidate,
                )
                if (
                    failure.get("candidate_id") != expected_failure_id
                    or failure.get("parent_candidate_id") != previous_candidate
                ):
                    raise VariationCheckpointError("source-contract failure is not bound to trajectory lineage")
                if type(self.private_store) is not PrivateTrajectoryStore:
                    raise VariationCheckpointError("source-contract failure lacks its private evidence store")
                failure_context, failure_evidence, failure_digest = self.private_store.load_generation_failure(
                    expected_failure_id
                )
                next_attempt = next(
                    (attempts_by_index[index] for index in all_indices if index > expected_index and index in attempts_by_index),
                    None,
                )
                cutoff_row = self._candidate_row(next_attempt.candidate_id) if next_attempt is not None else None
                failure_retrieval = self._recovery_retrieval(
                    context=failure_context,
                    task=task,
                    candidate_row=cutoff_row,
                )
                self._validate_recovery_context(
                    context=failure_context,
                    task=task,
                    run_id=run_id,
                    seed=seed,
                    attempt_index=expected_index,
                    parent_candidate_id=previous_candidate,
                    retrieval=failure_retrieval,
                )
                if (
                    failure_digest != failure.get("record_digest")
                    or failure_context.prompt_digest != failure.get("prompt_digest")
                    or failure_evidence.stage != "RESPONSE_CONTRACT"
                ):
                    raise VariationCheckpointError("source-contract failure differs from its durable raw evidence")
                continue
            attempt = attempts_by_index[expected_index]
            if attempt.candidate_id != self._candidate_id(
                run_id=run_id,
                task_id=task.task_id,
                seed=seed,
                attempt=attempt.attempt_index,
                parent=previous_candidate,
            ):
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
            if (
                row["candidate_json"] != canonical_json(candidate_payload)
                or set(candidate_payload) != {
                    "candidate_id", "campaign_id", "run_id", "task_id", "parent_candidate_id",
                    "mutation_family", "patch_hash", "requested_authority", "prompt_hash",
                    "model_hash", "adapter_hash", "metadata",
                }
            ):
                raise VariationCheckpointError("durable candidate payload is not canonical and closed")
            metadata = candidate_payload.get("metadata")
            expected_metadata_fields = {
                "schema_version", "arm_id", "attempt_index", "public_rule_id", "public_locus",
                "retrieval_digest", "evidence_ids", "candidate_artifact_digest", "response_contract",
                "response_contract_digest", "generation_evidence_digest", "generation_profile_digest",
            }
            if (
                not isinstance(metadata, Mapping)
                or set(metadata) != expected_metadata_fields
                or metadata.get("schema_version") != "egv-variation-candidate-v1"
                or metadata.get("arm_id") != self.policy.arm_id
                or metadata.get("attempt_index") != attempt.attempt_index
                or metadata.get("public_rule_id") != task.public_rule_id
                or metadata.get("public_locus") != task.public_locus
                or metadata.get("retrieval_digest") != attempt.retrieval_digest
                or metadata.get("evidence_ids") != list(attempt.evidence_ids)
                or metadata.get("candidate_artifact_digest") != attempt.candidate_artifact_digest
                or metadata.get("response_contract") != getattr(self.generator, "response_contract", "closed-json-v1")
                or metadata.get("response_contract_digest")
                != (self.response_contract_digest or getattr(self.generator, "response_contract_digest", ""))
                or metadata.get("generation_profile_digest") != self.generation_profile_digest
            ):
                raise VariationCheckpointError("durable candidate is not bound to its source artifact digest")
            expected_payload_scalars = {
                "candidate_id": attempt.candidate_id,
                "campaign_id": self.campaign_id,
                "run_id": run_id,
                "task_id": task.task_id,
                "parent_candidate_id": previous_candidate,
                "mutation_family": task.family_id,
                "requested_authority": row["requested_authority"],
                "prompt_hash": row["prompt_hash"],
                "model_hash": self.model_digest,
                "adapter_hash": self.adapter_digest,
            }
            if any(candidate_payload.get(field) != value for field, value in expected_payload_scalars.items()):
                raise VariationCheckpointError("durable candidate payload differs from its run binding")
            if any(
                candidate_payload.get(field) != row[field]
                for field in (
                    "candidate_id", "campaign_id", "run_id", "task_id", "parent_candidate_id",
                    "mutation_family", "patch_hash", "requested_authority", "prompt_hash",
                    "model_hash", "adapter_hash",
                )
            ):
                raise VariationCheckpointError("durable candidate table columns differ from canonical payload")
            try:
                candidate_source = artifact_store.read(attempt.candidate_artifact_digest)
            except Exception as exc:
                raise VariationCheckpointError("durable candidate source artifact is missing or corrupt") from exc
            if getattr(self.generator, "response_contract", "closed-json-v1") != "closed-json-v1":
                if type(self.private_store) is not PrivateTrajectoryStore:
                    raise VariationCheckpointError("durable source-only attempt lacks its private evidence store")
                private_attempt = self.private_store.load_attempts([attempt.candidate_id])[0]
                generation_context, generation, generation_digest = self.private_store.load_generation_success(
                    attempt.candidate_id
                )
                proposal = generation.proposal
                if (
                    generation_context != private_attempt.context
                    or private_attempt.candidate_source != candidate_source
                    or private_attempt.context.attempt_index != attempt.attempt_index
                    or private_attempt.context.parent_candidate_id != previous_candidate
                    or private_attempt.context.retrieval_digest != attempt.retrieval_digest
                    or private_attempt.generation_evidence_digest != generation_digest
                    or generation_digest != metadata.get("generation_evidence_digest")
                    or proposal.source != candidate_source
                    or proposal.mutation_digest != row["patch_hash"]
                    or proposal.requested_authority != row["requested_authority"]
                    or proposal.evidence_ids != attempt.evidence_ids
                    or proposal.declared_locus != task.public_locus
                    or private_attempt.context.prompt_digest != row["prompt_hash"]
                    or private_attempt.context.model_digest != row["model_hash"]
                    or private_attempt.context.adapter_digest != row["adapter_hash"]
                ):
                    raise VariationCheckpointError("durable source-only candidate differs from private generation evidence")
                recovered_retrieval = self._recovery_retrieval(
                    context=private_attempt.context,
                    task=task,
                    candidate_row=self._candidate_row(attempt.candidate_id),
                )
                self._validate_recovery_context(
                    context=private_attempt.context,
                    task=task,
                    run_id=run_id,
                    seed=seed,
                    attempt_index=attempt.attempt_index,
                    parent_candidate_id=previous_candidate,
                    retrieval=recovered_retrieval,
                )
            expected_dependencies = {
                (evidence_id, attempt.candidate_id, "EVIDENCE_USED")
                for evidence_id in attempt.evidence_ids
            }
            actual_dependencies = {
                (str(item["parent_id"]), str(item["child_id"]), str(item["edge_type"]))
                for item in self.ledger.connection.execute(
                    "SELECT parent_id,child_id,edge_type FROM dependencies WHERE child_id=?",
                    (attempt.candidate_id,),
                ).fetchall()
            }
            if actual_dependencies != expected_dependencies:
                raise VariationCheckpointError("durable candidate dependency set differs from its evidence binding")
            if self.ledger.candidate_disposition(attempt.candidate_id) != attempt.disposition:
                raise VariationCheckpointError("durable attempt disposition differs from the ledger projection")
            self._validate_attempt_receipts(attempt, task=task, run_id=run_id)
            previous_candidate = attempt.candidate_id
        if checkpoint is not None:
            expected_state = digest_for(
                {
                    "attempts": [item.to_dict() for item in attempts],
                    "run_id": run_id,
                    "arm_id": self.policy.arm_id,
                    "task_id": task.task_id,
                    "status": checkpoint.status,
                    **(
                        {"source_contract_failures": [dict(item) for item in failures]}
                        if failures else {}
                    ),
                }
            )
            if checkpoint.state_digest != expected_state:
                raise VariationCheckpointError("checkpoint state digest does not match durable attempts")
            expected_artifact_value = {
                "artifacts": [
                        {
                            "attempt_index": item.attempt_index,
                            "candidate_id": item.candidate_id,
                            "candidate_artifact_digest": item.candidate_artifact_digest,
                        }
                    for item in attempts
                    ]
            }
            if failures:
                expected_artifact_value["source_contract_failure_digests"] = [
                    item["record_digest"] for item in failures
                ]
            expected_artifacts = digest_for(expected_artifact_value)
            if checkpoint.artifact_manifest_hash != expected_artifacts:
                raise VariationCheckpointError("checkpoint artifact manifest does not match durable attempts")
            expected_projection = digest_for(
                {
                    "ledger_head_event_id": checkpoint.ledger_head_event_id,
                    "ledger_head_hash": checkpoint.ledger_head_hash,
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
            if checkpoint.last_candidate_id != attempts[-1].candidate_id or checkpoint.attempt_index != attempts[-1].attempt_index:
                raise VariationCheckpointError("checkpoint cursor does not match durable attempts")
            expected_status = "PROMOTED" if last_disposition == "PROMOTED" else None
            if last_diagnostic == Diagnostic.INTERNAL_ERROR.value:
                expected_status = "FAILED"
            elif expected_status is None and max(all_indices) == self.max_attempts:
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
            required = {
                "schema_version", "attempt_index", "candidate_id", "arm_id", "retrieval_policy",
                "retrieval_digest", "evidence_ids", "candidate_artifact_digest", "diagnostic_enum",
                "resource_bucket", "disposition", "receipt_ids",
            }
            if set(payload) != required or payload.get("schema_version") != "egv-variation-attempt-v1":
                raise VariationCheckpointError("durable Variation attempt payload is not closed")
            attempt_index = payload.get("attempt_index")
            candidate_id = payload.get("candidate_id")
            retrieval_digest = payload.get("retrieval_digest")
            artifact_digest = payload.get("candidate_artifact_digest")
            evidence_ids = payload.get("evidence_ids")
            receipt_ids = payload.get("receipt_ids")
            if (
                event.get("campaign_id") != self.campaign_id
                or event.get("subject_id") != candidate_id
                or event.get("source_class") != "GENERATOR"
                or event.get("disposition") != payload.get("disposition")
                or event.get("evaluator_identity") is not None
                or event.get("valid_time") is not None
                or event.get("idempotency_key")
                != "variation-attempt:{}:{}".format(candidate_id, attempt_index)
                or
                not isinstance(attempt_index, int)
                or isinstance(attempt_index, bool)
                or attempt_index < 1
                or not isinstance(candidate_id, str)
                or not candidate_id
                or payload.get("arm_id") != self.policy.arm_id
                or payload.get("retrieval_policy") != self.policy.retrieval_policy
                or not isinstance(retrieval_digest, str)
                or not isinstance(artifact_digest, str)
                or any(len(value) != 64 or value != value.lower() for value in (retrieval_digest, artifact_digest))
                or not isinstance(evidence_ids, list)
                or any(not isinstance(item, str) or not item for item in evidence_ids)
                or evidence_ids != sorted(set(evidence_ids))
                or not isinstance(receipt_ids, list)
                or any(not isinstance(item, str) or not item for item in receipt_ids)
                or len(receipt_ids) != len(set(receipt_ids))
            ):
                raise VariationCheckpointError("durable Variation attempt payload has invalid native fields")
            try:
                int(retrieval_digest, 16)
                int(artifact_digest, 16)
                validate_diagnostic(payload.get("diagnostic_enum"))
                validate_resource_bucket(payload.get("resource_bucket"))
                validate_disposition(payload.get("disposition"))
            except (TypeError, ValueError) as exc:
                raise VariationCheckpointError("durable Variation attempt payload violates closed vocabularies") from exc
            try:
                attempts.append(
                    AttemptRecord(
                        attempt_index,
                        candidate_id,
                        artifact_digest,
                        retrieval_digest,
                        tuple(evidence_ids),
                        payload["diagnostic_enum"],
                        payload["resource_bucket"],
                        payload["disposition"],
                        tuple(receipt_ids),
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
        checkpoint_store = CheckpointStore(workspace.checkpoints)
        checkpoint_history = checkpoint_store.inventory(run_id=run_id)
        latest_checkpoint = checkpoint_history[-1] if checkpoint_history else None
        checkpoint: Optional[VariationCheckpoint] = None
        if resume_from is not None:
            requested_path = Path(resume_from).resolve()
            checkpoint = checkpoint_store.load(requested_path)
            if latest_checkpoint is None or requested_path != latest_checkpoint[0].resolve():
                raise VariationCheckpointError(
                    "explicit resume checkpoint is not the unique latest isolated state"
                )
            resume_from = latest_checkpoint[0]
        elif latest_checkpoint is not None:
            resume_from, checkpoint = latest_checkpoint
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
        attempts: List[AttemptRecord] = self._durable_attempts(run_id=run_id, task_id=task.task_id)
        legacy_bundle: Optional[LegacyGenerationBundle] = None
        legacy_receipt_ids: Tuple[str, ...] = tuple()
        if getattr(self.generator, "response_contract", "closed-json-v1") != "closed-json-v1":
            if type(self.private_store) is not PrivateTrajectoryStore:
                raise VariationCheckpointError("source-only recovery lacks its exact private evidence store")
            legacy_bundles = self.private_store.legacy_generation_bundles()
            if legacy_bundles:
                # This is a one-orphan compatibility boundary, not a general
                # legacy import path.  The authenticated evaluator cache must
                # replay exactly before start/intent evidence is synthesized.
                if (
                    checkpoint_history
                    or len(legacy_bundles) != 1
                    or attempts
                    or (
                        not self.fixture_mode
                        and type(self.evaluator) is not RemoteControllerEvaluationGateway
                    )
                ):
                    raise VariationCheckpointError(
                        "legacy generation migration is outside the preserved orphan boundary"
                    )
                legacy_bundle = legacy_bundles[0]
                legacy_receipt_ids = self._legacy_pre_materialization_receipt_ids(
                    legacy_bundle,
                    task=task,
                    run_id=run_id,
                )
                try:
                    self.ledger.verify_integrity()
                except Exception as exc:
                    raise VariationCheckpointError(
                        "legacy generation migration ledger failed full integrity validation"
                    ) from exc
        failures = (
            tuple()
            if legacy_bundle is not None
            else self._source_contract_failure_records(run_id=run_id, task_id=task.task_id)
        )
        try:
            self.ledger.verify_integrity()
        except Exception as exc:
            raise VariationCheckpointError("Variation ledger failed integrity validation before recovery") from exc
        checkpoint_attempt_indices = [item[1].attempt_index for item in checkpoint_history]
        durable_attempt_indices = [item.attempt_index for item in attempts]
        if (
            checkpoint_attempt_indices != durable_attempt_indices[:len(checkpoint_attempt_indices)]
            or len(durable_attempt_indices) - len(checkpoint_attempt_indices) > 1
        ):
            raise VariationCheckpointError(
                "checkpoint files are not the complete prefix of durable Variation attempts"
            )
        projected_checkpoint_ids = {
            str(row["checkpoint_id"])
            for row in self.ledger.connection.execute(
                """SELECT DISTINCT c.checkpoint_id
                   FROM checkpoints AS c
                   JOIN events AS e ON e.event_id=c.last_durable_event_id
                   WHERE e.run_id=?""",
                (run_id,),
            ).fetchall()
        }
        file_checkpoint_ids = {item[1].digest for item in checkpoint_history}
        missing_files = projected_checkpoint_ids - file_checkpoint_ids
        missing_projections = file_checkpoint_ids - projected_checkpoint_ids
        permitted_missing_projection = (
            {checkpoint_history[-1][1].digest} if checkpoint_history else set()
        )
        if missing_files or not missing_projections.issubset(permitted_missing_projection):
            raise VariationCheckpointError(
                "checkpoint file history differs from its durable ledger projections"
            )
        for checkpoint_index, (_checkpoint_path, historical_checkpoint) in enumerate(checkpoint_history):
            self._validate_resume(historical_checkpoint, task=task, run_id=run_id, seed=seed)
            checkpoint_attempts = [
                item for item in attempts if item.attempt_index <= historical_checkpoint.attempt_index
            ]
            checkpoint_failures = [
                item for item in failures
                if int(item["attempt_index"]) <= historical_checkpoint.attempt_index
            ]
            self._validate_durable_attempts(
                checkpoint_attempts,
                workspace=workspace,
                task=task,
                run_id=run_id,
                seed=seed,
                failures=checkpoint_failures,
                checkpoint=historical_checkpoint,
            )
            self._reconcile_checkpoint_projection(
                historical_checkpoint,
                allow_missing=checkpoint_index == len(checkpoint_history) - 1,
            )
        if attempts or failures:
            self._validate_durable_attempts(
                attempts,
                workspace=workspace,
                task=task,
                run_id=run_id,
                seed=seed,
                failures=failures,
            )
        cursor = max(
            [attempt.attempt_index for attempt in attempts]
            + [int(item["attempt_index"]) for item in failures]
            + [0]
        )
        response_contract = getattr(self.generator, "response_contract", "closed-json-v1")
        completed_candidate_ids = {attempt.candidate_id for attempt in attempts}
        successful_generations = ()
        if response_contract != "closed-json-v1":
            if type(self.private_store) is not PrivateTrajectoryStore:
                raise VariationCheckpointError("source-only recovery lacks its exact private evidence store")
            if legacy_bundle is not None:
                successful_generations = (
                    (
                        legacy_bundle.candidate_id,
                        legacy_bundle.context,
                        legacy_bundle.evidence,
                        legacy_bundle.generation_record_digest,
                    ),
                )
            else:
                successful_generations = self.private_store.successful_generations(
                    run_id=run_id,
                    task_id=task.task_id,
                    arm_id=self.policy.arm_id,
                )
            success_ids = {item[0] for item in successful_generations}
            if not completed_candidate_ids.issubset(success_ids):
                raise VariationCheckpointError("durable source-only attempt lacks successful generation evidence")
        successful_by_candidate = {item[0]: item for item in successful_generations}
        pending_generations = [
            item for item in successful_generations if item[0] not in completed_candidate_ids
        ]
        if len(pending_generations) > 1:
            raise VariationCheckpointError("multiple uncheckpointed successful generations require quarantine")
        pending_generation = pending_generations[0] if pending_generations else None
        last_candidate_id = attempts[-1].candidate_id if attempts else None
        if pending_generation is not None:
            pending_candidate_id, pending_context, _pending_evidence, _pending_digest = pending_generation
            if cursor + 1 > self.max_attempts:
                raise VariationCheckpointError("pending generation exceeds the frozen attempt budget")
            expected_pending_id = self._candidate_id(
                run_id=run_id,
                task_id=task.task_id,
                seed=seed,
                attempt=cursor + 1,
                parent=last_candidate_id,
            )
            if (
                pending_candidate_id != expected_pending_id
                or pending_context.attempt_index != cursor + 1
                or pending_context.parent_candidate_id != last_candidate_id
            ):
                raise VariationCheckpointError("pending generation is not the unique next trajectory attempt")
        candidate_rows = self.ledger.connection.execute(
            "SELECT candidate_id FROM candidates WHERE run_id=? ORDER BY candidate_id",
            (run_id,),
        ).fetchall()
        allowed_candidate_ids = set(completed_candidate_ids)
        if pending_generation is not None:
            allowed_candidate_ids.add(str(pending_generation[0]))
        row_candidate_ids = {str(row["candidate_id"]) for row in candidate_rows}
        if not row_candidate_ids.issubset(allowed_candidate_ids):
            raise VariationCheckpointError("Variation run contains an orphan or quasi-matching candidate")
        if attempts or failures or checkpoint is not None or pending_generation is not None or row_candidate_ids:
            try:
                self.ledger.verify_integrity()
            except Exception as exc:
                raise VariationCheckpointError("Variation recovery ledger failed full integrity validation") from exc

        checkpoint_path: Optional[Path] = Path(resume_from) if resume_from is not None else None
        if checkpoint is not None and checkpoint.status in {"PROMOTED", "BUDGET_EXHAUSTED", "FAILED"}:
            if pending_generation is not None or cursor != checkpoint.attempt_index:
                raise VariationCheckpointError("terminal checkpoint has later uncheckpointed trajectory state")
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

        latest_attempt_index = attempts[-1].attempt_index if attempts else 0
        checkpoint_attempt_index = checkpoint.attempt_index if checkpoint is not None else 0
        if (
            attempts
            and pending_generation is None
            and cursor == latest_attempt_index
            and latest_attempt_index > checkpoint_attempt_index
        ):
            latest_attempt = attempts[-1]
            if self.ledger.ledger_head_hash() != latest_attempt.ledger_head_hash:
                raise VariationCheckpointError("uncheckpointed Variation attempt is not the durable ledger head")
            recovery_generation = successful_by_candidate.get(latest_attempt.candidate_id)
            if recovery_generation is None:
                raise VariationCheckpointError(
                    "uncheckpointed Variation attempt has no exact generation evidence for evaluator replay"
                )
            candidate_id, context, generation, generation_evidence_digest = recovery_generation
            candidate_row = self._candidate_row(candidate_id)
            if candidate_row is None:
                raise VariationCheckpointError("uncheckpointed Variation attempt lacks its candidate row")
            retrieval = self._recovery_retrieval(
                context=context,
                task=task,
                candidate_row=candidate_row,
            )
            self._validate_recovery_context(
                context=context,
                task=task,
                run_id=run_id,
                seed=seed,
                attempt_index=latest_attempt.attempt_index,
                parent_candidate_id=attempts[-2].candidate_id if len(attempts) > 1 else None,
                retrieval=retrieval,
            )
            proposal = generation.proposal
            proposal.validate(context, source_limit=CANDIDATE_SOURCE_LIMIT)
            artifact_path = self._persist_candidate(
                workspace=workspace.root,
                task=task,
                run_id=run_id,
                candidate_id=candidate_id,
                context=context,
                proposal=proposal,
                retrieval=retrieval,
                generation_evidence_digest=generation_evidence_digest,
                recovering=True,
            )
            verified_result = self.evaluator.evaluate(
                candidate_id=candidate_id,
                task_id=task.task_id,
                source=proposal.source,
                opaque_input=task.evaluator_input,
                requested_authority=proposal.requested_authority,
                declared_locus=proposal.declared_locus,
                candidate_source_path=artifact_path,
            )
            replayed_attempt = self._materialize_result(
                task=task,
                run_id=run_id,
                candidate_id=candidate_id,
                proposal=proposal,
                retrieval=retrieval,
                result=verified_result,
                attempt_index=latest_attempt.attempt_index,
            )
            if replayed_attempt != latest_attempt:
                raise VariationCheckpointError(
                    "verified evaluator replay differs from the durable uncheckpointed attempt"
                )
            recovered_status = "RUNNING"
            if latest_attempt.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value:
                recovered_status = "FAILED"
            elif latest_attempt.disposition == "PROMOTED":
                recovered_status = "PROMOTED"
            elif cursor == self.max_attempts:
                recovered_status = "BUDGET_EXHAUSTED"
            checkpoint_path = self._checkpoint(
                workspace=workspace,
                task=task,
                run_id=run_id,
                seed=seed,
                attempts=attempts,
                attempt=latest_attempt,
                status=recovered_status,
            )
            if recovered_status != "RUNNING":
                return VariationReport(
                    self.campaign_id,
                    run_id,
                    self.policy.arm_id,
                    task.task_id,
                    seed,
                    tuple(attempts),
                    recovered_status,
                    str(checkpoint_path),
                    self.ledger.ledger_head_hash(),
                    self.ledger.verify_integrity(),
                    self.model_digest,
                    self.adapter_digest,
                    self.policy.retrieval_policy,
                    self.policy.authority_enforced and bool(getattr(self.evaluator, "enforceable", False)),
                )
        if cursor >= self.max_attempts and failures and (
            not attempts or attempts[-1].disposition != "PROMOTED"
        ):
            raise SourceContractBudgetExhausted(
                run_id=run_id,
                failure_count=len(failures),
                last_failure_digest=str(failures[-1]["record_digest"]),
            )
        start_attempt = cursor + 1
        terminal_status = "RUNNING"
        for attempt_index in range(start_attempt, self.max_attempts + 1):
            recovering = pending_generation is not None and attempt_index == pending_generation[1].attempt_index
            if recovering:
                candidate_id, context, generation, generation_evidence_digest = pending_generation
                candidate_row = self._candidate_row(candidate_id)
                retrieval = self._recovery_retrieval(
                    context=context,
                    task=task,
                    candidate_row=candidate_row,
                )
                self._validate_recovery_context(
                    context=context,
                    task=task,
                    run_id=run_id,
                    seed=seed,
                    attempt_index=attempt_index,
                    parent_candidate_id=last_candidate_id,
                    retrieval=retrieval,
                )
                proposal = generation.proposal
            else:
                retrieval = self.retrieval_policy.retrieve(
                    self.ledger,
                    campaign_id=self.campaign_id,
                    arm_id=self.policy.arm_id,
                    task_id=task.task_id,
                    isolation=self.isolation,
                )
                if response_contract == "closed-json-v1":
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
                else:
                    prompt_digest = digest_for("unrendered-source-only-attempt")
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
                    task.task_statement,
                    self.initial_source.decode("utf-8") if self.initial_source is not None else None,
                    self.initial_source_digest,
                    response_contract,
                    self.response_contract_digest or getattr(self.generator, "response_contract_digest", ""),
                    self.generation_profile_digest,
                )
                if response_contract != "closed-json-v1":
                    prompt_digest = self.generator.prompt_digest_for(context)
                    context = replace(context, prompt_digest=prompt_digest)
                candidate_id = self._candidate_id(
                    run_id=run_id,
                    task_id=task.task_id,
                    seed=seed,
                    attempt=attempt_index,
                    parent=last_candidate_id,
                )
                generation_evidence_digest = None
                if response_contract == "closed-json-v1":
                    proposal = self.generator.propose(context)
                else:
                    if self.private_store is None:
                        raise VariationDependencyError(
                            "source-only generation has no private evidence store"
                        )
                    self.private_store.record_generation_start(
                        candidate_id=candidate_id,
                        context=context,
                    )
                    try:
                        generation = self.generator.propose_with_evidence(context)
                    except CandidateGenerationFailure as exc:
                        if self.private_store is None:
                            raise VariationDependencyError(
                                "source-only generation failure has no private evidence store"
                            ) from exc
                        self.private_store.record_generation_failure(
                            candidate_id=candidate_id,
                            context=context,
                            evidence=exc.evidence,
                        )
                        exc.candidate_id = candidate_id
                        if exc.evidence.stage != "RESPONSE_CONTRACT":
                            raise
                        failures = self._source_contract_failure_records(
                            run_id=run_id,
                            task_id=task.task_id,
                        )
                        self._validate_durable_attempts(
                            attempts,
                            workspace=workspace,
                            task=task,
                            run_id=run_id,
                            seed=seed,
                            failures=failures,
                        )
                        if attempt_index == self.max_attempts:
                            raise SourceContractBudgetExhausted(
                                run_id=run_id,
                                failure_count=len(failures),
                                last_failure_digest=str(failures[-1]["record_digest"]),
                            ) from exc
                        continue
                    generation_evidence_digest = self.private_store.record_generation_success(
                        candidate_id=candidate_id,
                        context=context,
                        evidence=generation,
                    )
                    proposal = generation.proposal
            proposal.validate(context, source_limit=CANDIDATE_SOURCE_LIMIT)
            legacy_recovery = (
                legacy_bundle is not None
                and recovering
                and candidate_id == legacy_bundle.candidate_id
            )
            legacy_head_before_replay = (
                self.ledger.ledger_head_hash() if legacy_recovery else ""
            )
            artifact_path = self._persist_candidate(
                workspace=workspace.root,
                task=task,
                run_id=run_id,
                candidate_id=candidate_id,
                context=context,
                proposal=proposal,
                retrieval=retrieval,
                generation_evidence_digest=generation_evidence_digest,
                recovering=recovering,
            )
            if legacy_recovery and self.ledger.ledger_head_hash() != legacy_head_before_replay:
                raise VariationCheckpointError(
                    "legacy candidate reconciliation changed the preserved ledger"
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
            if legacy_recovery:
                if type(self.private_store) is not PrivateTrajectoryStore:
                    raise VariationCheckpointError(
                        "legacy evaluator replay lacks its exact private evidence store"
                    )
                legacy_replay_proof_digest = self._validate_legacy_cached_result(
                    legacy_bundle,
                    task=task,
                    run_id=run_id,
                    retrieval=retrieval,
                    result=result,
                    expected_receipt_ids=legacy_receipt_ids,
                    ledger_head_before_replay=legacy_head_before_replay,
                )
                self.private_store.commit_legacy_generation_bundle(
                    legacy_bundle,
                    generation_record_digest=legacy_bundle.generation_record_digest,
                    trajectory_record_digest=legacy_bundle.trajectory_record_digest,
                    replay_proof_digest=legacy_replay_proof_digest,
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
    "SourceContractBudgetExhausted",
    "VARIATION_PROTOCOL_DIGEST",
    "VARIATION_PROTOCOL_SCHEMA",
    "VariationReport",
    "VariationRunner",
    "VariationTask",
]
