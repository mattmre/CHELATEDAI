"""Bounded candidate loop integrated with the Evidence and Evaluation slices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from ..canonical import content_id, digest_bytes, digest_for, failure_family_root
from ..evaluation.artifacts import ContentAddressedArtifactStore
from ..evaluation.controller import EvaluationResult, EvaluatorController
from ..evaluation.dataset import MicroRepo
from ..evaluation.diagnostics import Diagnostic, validate_diagnostic, validate_disposition, validate_resource_bucket
from ..evaluation.prompts import PROMPT_IDS, PromptRegistry
from ..ledger import EvidenceLedger
from ..receipts import receipt_hash
from .arms import ArmIsolation, ArmPolicy, arm_policy
from .checkpoint import CheckpointStore, VariationCheckpoint
from .errors import VariationBudgetError, VariationCheckpointError, VariationConfigurationError, VariationDependencyError
from .generator import CandidateContext, CandidateGenerator, CandidateProposal
from .model import MODEL_REVISION
from .retrieval import EvidenceRetrievalPolicy, RetrievalResult, retrieval_policy


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
        if not getattr(controller.sandbox, "enforceable", False):
            raise VariationDependencyError("Variation production path requires the enforceable Docker evaluator")
        self.controller = controller
        self.evaluator_revision = controller.hidden_runner.evaluator_revision
        self.evaluator_digest = digest_for(self.evaluator_revision)

    def evaluate(self, **kwargs: Any) -> EvaluationResult:
        return self.controller.evaluate(**kwargs)


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
            "seed": self.seed,
            "attempts": [
                {
                    "attempt_index": attempt.attempt_index,
                    "candidate_artifact_digest": attempt.candidate_artifact_digest,
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


class BoundedCandidateLoop:
    """One immutable, resumable candidate trajectory."""

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
        seed_set: Sequence[int] = (0, 1, 2),
        fixture_mode: bool = False,
    ) -> None:
        if max_attempts <= 0 or max_attempts > MAX_CANDIDATE_ATTEMPTS:
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
        self.max_attempts = max_attempts
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
        if self.policy.requires_adapter and not adapter_digest:
            raise VariationDependencyError("LoRA arms require a sealed adapter; Training is not part of Variation")
        if adapter_digest is not None:
            _require_digest(adapter_digest, "adapter digest")
        if getattr(generator, "model_digest", model_digest) != model_digest:
            raise VariationConfigurationError("candidate generator model digest differs from the frozen campaign")
        if getattr(generator, "adapter_digest", adapter_digest) != adapter_digest:
            raise VariationConfigurationError("candidate generator adapter digest differs from the frozen campaign")
        if getattr(generator, "test_only", False) and not fixture_mode:
            raise VariationDependencyError("test-only candidate generators cannot enter the production Variation path")
        if not fixture_mode and not isinstance(evaluator, ControllerEvaluationGateway):
            raise VariationDependencyError("production Variation requires ControllerEvaluationGateway")
        if not getattr(evaluator, "enforceable", False) and not fixture_mode:
            raise VariationDependencyError("non-enforceable evaluators are test-only and cannot run Variation")

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
        hidden_runner = getattr(self.evaluator, "hidden_runner", None)
        public_record = hidden_runner.public_record(task.task_id) if hidden_runner is not None else None
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
        attempt: AttemptRecord,
        status: str,
    ) -> Path:
        state_digest = digest_for(
            {
                "attempt": attempt.to_dict(),
                "run_id": run_id,
                "arm_id": self.policy.arm_id,
                "task_id": task.task_id,
            }
        )
        projection_generation = digest_for(
            {"ledger_head": attempt.ledger_head_hash, "arm_id": self.policy.arm_id, "run_id": run_id}
        )
        artifact_manifest_hash = digest_for(
            {"candidate_artifact_digest": attempt.candidate_artifact_digest, "attempt_index": attempt.attempt_index}
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
            ledger_head_hash=attempt.ledger_head_hash,
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
        if checkpoint.ledger_head_hash != self.ledger.ledger_head_hash():
            raise VariationCheckpointError("ledger head differs from the durable Variation checkpoint")
        if checkpoint.attempt_index > self.max_attempts:
            raise VariationCheckpointError("resume checkpoint exceeds the current bounded attempt budget")
        if checkpoint.status == "RUNNING" and checkpoint.attempt_index >= self.max_attempts:
            raise VariationCheckpointError("running checkpoint has no remaining bounded candidate attempt")

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

    def run(
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
        if checkpoint is not None:
            self._validate_resume(checkpoint, task=task, run_id=run_id, seed=seed)
        attempts: List[AttemptRecord] = (
            self._durable_attempts(run_id=run_id, task_id=task.task_id) if checkpoint is not None else []
        )
        start_attempt = checkpoint.attempt_index + 1 if checkpoint is not None else 1
        if checkpoint is not None and checkpoint.status in {"PROMOTED", "REJECTED", "BUDGET_EXHAUSTED", "FAILED"}:
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
                    workspace=workspace, task=task, run_id=run_id, seed=seed, attempt=attempt, status=terminal_status
                )
                break
            if attempt.disposition == "PROMOTED":
                terminal_status = "PROMOTED"
                checkpoint_path = self._checkpoint(
                    workspace=workspace, task=task, run_id=run_id, seed=seed, attempt=attempt, status=terminal_status
                )
                break
            if attempt_index == self.max_attempts:
                terminal_status = "BUDGET_EXHAUSTED"
            checkpoint_path = self._checkpoint(
                workspace=workspace, task=task, run_id=run_id, seed=seed, attempt=attempt, status=terminal_status
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
