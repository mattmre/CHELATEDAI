"""Deterministic, public-safe preparation for the dual-Spark commissioning run."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

from ..canonical import digest_for, validate_sha256
from ..evaluation.dataset import EvaluationCorpus, FAMILY_SPECS
from ..variation.arms import arm_policy
from ..variation.loop import VARIATION_PROTOCOL_DIGEST, VariationTask
from .trajectories import COMMISSIONING_ARMS, COMMISSIONING_SEEDS, GenerationRequest


COMMISSIONING_SCHEMA = "egv-commissioning-inputs-v1"
TRAINER_INPUTS_SCHEMA = "egv-commissioning-trainer-inputs-v1"
TRAIN_MANIFEST_SCHEMA = "egv-commissioning-train-tasks-v1"
PRIVATE_DEV_MANIFEST_SCHEMA = "egv-commissioning-private-dev-v1"
REQUEST_MANIFEST_SCHEMA = "egv-commissioning-generation-requests-v1"
TRAIN_TASK_COUNT = 20
PRIVATE_DEV_TASK_COUNT = 8
GENERATION_REQUEST_COUNT = 80
_PUBLIC_TASK_FIELDS = frozenset(
    {"template_id", "family_id", "split", "ordinal", "source_digest", "public_rule_id", "public_locus"}
)
_PRIVATE_TASK_FIELDS = _PUBLIC_TASK_FIELDS | frozenset({"hidden_spec_digest", "corrected_source_digest"})


class CommissioningPreparationError(RuntimeError):
    """Frozen commissioning inputs are incomplete, contaminated, or ambiguous."""


def _digest(value: Any, field_name: str) -> str:
    try:
        return validate_sha256(value, field_name)
    except Exception as exc:
        raise CommissioningPreparationError(str(exc)) from exc


@dataclass(frozen=True)
class CommissioningPlan:
    campaign_id: str
    corpus_manifest_digest: str
    model_manifest_digest: str
    variation_protocol_digest: str
    train_records: Tuple[Mapping[str, Any], ...]
    generation_requests: Tuple[GenerationRequest, ...]
    _private_dev_records: Tuple[Mapping[str, Any], ...] = field(repr=False)
    _private_dev_tasks: Tuple[VariationTask, ...] = field(repr=False)
    schema_version: str = COMMISSIONING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != COMMISSIONING_SCHEMA:
            raise CommissioningPreparationError("unsupported commissioning input schema")
        if not isinstance(self.campaign_id, str) or not self.campaign_id or "/" in self.campaign_id or "\\" in self.campaign_id:
            raise CommissioningPreparationError("campaign ID is not a bounded public identifier")
        for name in ("corpus_manifest_digest", "model_manifest_digest", "variation_protocol_digest"):
            _digest(getattr(self, name), name)
        if self.variation_protocol_digest != VARIATION_PROTOCOL_DIGEST:
            raise CommissioningPreparationError("commissioning protocol differs from frozen Variation protocol")
        if not isinstance(self.train_records, tuple) or not isinstance(self._private_dev_records, tuple):
            raise CommissioningPreparationError("commissioning task manifests must be immutable tuples")
        if len(self.train_records) != TRAIN_TASK_COUNT or len(self._private_dev_records) != PRIVATE_DEV_TASK_COUNT:
            raise CommissioningPreparationError("commissioning task counts differ from the frozen 20/8 allocation")
        if any(not isinstance(record, Mapping) or set(record) != _PUBLIC_TASK_FIELDS for record in self.train_records):
            raise CommissioningPreparationError("training task record is not the closed public evaluation schema")
        if any(
            not isinstance(record, Mapping) or set(record) != _PRIVATE_TASK_FIELDS
            for record in self._private_dev_records
        ):
            raise CommissioningPreparationError("private dev record is not the closed evaluator schema")
        object.__setattr__(
            self,
            "train_records",
            tuple(MappingProxyType(dict(record)) for record in self.train_records),
        )
        object.__setattr__(
            self,
            "_private_dev_records",
            tuple(MappingProxyType(dict(record)) for record in self._private_dev_records),
        )
        if len(self._private_dev_tasks) != PRIVATE_DEV_TASK_COUNT:
            raise CommissioningPreparationError("evaluator-private dev lookup is incomplete")
        train_ids = [record.get("template_id") for record in self.train_records]
        dev_ids = [record.get("template_id") for record in self._private_dev_records]
        if len(set(train_ids)) != TRAIN_TASK_COUNT or len(set(dev_ids)) != PRIVATE_DEV_TASK_COUNT:
            raise CommissioningPreparationError("commissioning task IDs are duplicated")
        if set(train_ids) & set(dev_ids):
            raise CommissioningPreparationError("train and evaluator-private dev tasks overlap")
        if any(record.get("split") != "train" for record in self.train_records):
            raise CommissioningPreparationError("public training manifest contains a non-train task")
        if any(record.get("split") != "dev" for record in self._private_dev_records):
            raise CommissioningPreparationError("private dev manifest contains a non-dev task")
        for task, record in zip(self._private_dev_tasks, self._private_dev_records):
            if type(task) is not VariationTask:
                raise CommissioningPreparationError("private dev lookup contains an unvalidated task")
            public_task = task.public_dict()
            if (
                public_task["task_id"] != record["template_id"]
                or public_task["family_id"] != record["family_id"]
                or public_task["public_locus"] != record["public_locus"]
                or public_task["public_rule_id"] != record["public_rule_id"]
            ):
                raise CommissioningPreparationError("private dev lookup differs from its frozen manifest record")
        if tuple(request.request_id for request in self.generation_requests) != tuple(
            sorted(request.request_id for request in self.generation_requests)
        ):
            raise CommissioningPreparationError("generation requests are not in canonical ID order")
        if len(self.generation_requests) != GENERATION_REQUEST_COUNT:
            raise CommissioningPreparationError("commissioning requires exactly 80 B/D generation requests")
        request_coordinates = {
            (request.task_id, request.arm_id, request.seed) for request in self.generation_requests
        }
        expected_coordinates = {
            (task_id, arm_id, seed)
            for task_id in train_ids
            for arm_id in COMMISSIONING_ARMS
            for seed in COMMISSIONING_SEEDS
        }
        if request_coordinates != expected_coordinates:
            raise CommissioningPreparationError("generation request matrix is not exact 20 x B/D x two seeds")
        if any(request.status != "PENDING" for request in self.generation_requests):
            raise CommissioningPreparationError("prepared generation requests must remain pending")
        train_by_id = {record["template_id"]: record for record in self.train_records}
        for request in self.generation_requests:
            if type(request) is not GenerationRequest:
                raise CommissioningPreparationError("request manifest contains an unvalidated envelope")
            task_record = train_by_id.get(request.task_id)
            if (
                task_record is None
                or request.campaign_id != self.campaign_id
                or request.task_family != task_record["family_id"]
                or request.task_record_digest != digest_for(dict(task_record))
                or request.corpus_manifest_digest != self.corpus_manifest_digest
                or request.model_manifest_digest != self.model_manifest_digest
                or request.variation_protocol_digest != self.variation_protocol_digest
            ):
                raise CommissioningPreparationError("generation request differs from frozen commissioning inputs")

    @property
    def train_manifest(self) -> Dict[str, Any]:
        family_counts = {
            family.family_id: sum(record["family_id"] == family.family_id for record in self.train_records)
            for family in FAMILY_SPECS
        }
        return {
            "schema_version": TRAIN_MANIFEST_SCHEMA,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "task_count": TRAIN_TASK_COUNT,
            "family_counts": family_counts,
            "tasks": [dict(record) for record in self.train_records],
        }

    @property
    def train_manifest_digest(self) -> str:
        return digest_for(self.train_manifest)

    @property
    def private_dev_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": PRIVATE_DEV_MANIFEST_SCHEMA,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "task_count": PRIVATE_DEV_TASK_COUNT,
            "tasks": [dict(record) for record in self._private_dev_records],
            "evaluator_lookup_required": True,
        }

    @property
    def private_dev_manifest_digest(self) -> str:
        return digest_for(self.private_dev_manifest)

    @property
    def private_dev_tasks(self) -> Tuple[VariationTask, ...]:
        """Evaluator-only task objects; never included in the public projection."""

        return self._private_dev_tasks

    @property
    def request_manifest(self) -> Dict[str, Any]:
        requests = [request.to_dict() for request in self.generation_requests]
        return {
            "schema_version": REQUEST_MANIFEST_SCHEMA,
            "request_count": GENERATION_REQUEST_COUNT,
            "arms": list(COMMISSIONING_ARMS),
            "seeds": list(COMMISSIONING_SEEDS),
            "requests": requests,
        }

    @property
    def request_manifest_digest(self) -> str:
        return digest_for(self.request_manifest)

    def public_manifest(self) -> Dict[str, Any]:
        """Return a projection containing no dev IDs, evaluator inputs, or hidden digests."""

        value = {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "model_manifest_digest": self.model_manifest_digest,
            "variation_protocol_digest": self.variation_protocol_digest,
            "train_task_count": TRAIN_TASK_COUNT,
            "train_manifest_digest": self.train_manifest_digest,
            "private_dev_task_count": PRIVATE_DEV_TASK_COUNT,
            "private_dev_manifest_digest": self.private_dev_manifest_digest,
            "generation_request_count": GENERATION_REQUEST_COUNT,
            "generation_request_manifest_digest": self.request_manifest_digest,
            "arms": list(COMMISSIONING_ARMS),
            "arm_policy_digests": {
                arm_id: arm_policy(arm_id).digest for arm_id in COMMISSIONING_ARMS
            },
            "seeds": list(COMMISSIONING_SEEDS),
            "accepted_response_count": 0,
            "status": "PENDING",
            "live_model_executed": False,
        }
        value["manifest_digest"] = digest_for(value)
        return value

    def private_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-commissioning-private-inputs-v1",
            "public_manifest_digest": digest_for(self.public_manifest()),
            "train_manifest": self.train_manifest,
            "private_dev_manifest": self.private_dev_manifest,
            "request_manifest": self.request_manifest,
        }

    def trainer_inputs(self) -> Dict[str, Any]:
        """Return the complete trainer bundle with no development identities or inputs."""

        value = {
            "schema_version": TRAINER_INPUTS_SCHEMA,
            "campaign_id": self.campaign_id,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "model_manifest_digest": self.model_manifest_digest,
            "variation_protocol_digest": self.variation_protocol_digest,
            "train_manifest": self.train_manifest,
            "request_manifest": self.request_manifest,
        }
        value["trainer_inputs_digest"] = digest_for(value)
        return value


def prepare_commissioning(
    corpus: EvaluationCorpus,
    *,
    campaign_id: str,
    model_manifest_digest: str,
    variation_protocol_digest: str = VARIATION_PROTOCOL_DIGEST,
) -> CommissioningPlan:
    """Freeze exact corpus-derived inputs without running a model or evaluator."""

    if type(corpus) is not EvaluationCorpus:
        raise CommissioningPreparationError("commissioning requires the exact evaluator-owned corpus type")
    corpus.validate()
    corpus_digest = corpus.manifest_digest()
    _digest(model_manifest_digest, "model manifest digest")
    if variation_protocol_digest != VARIATION_PROTOCOL_DIGEST:
        raise CommissioningPreparationError("commissioning cannot substitute the Variation protocol")
    train_repositories = corpus.split("train")
    dev_repositories = corpus.split("dev")
    if len(train_repositories) != TRAIN_TASK_COUNT or len(dev_repositories) != PRIVATE_DEV_TASK_COUNT:
        raise CommissioningPreparationError("evaluator corpus does not provide exact 20/8 commissioning splits")
    train_records = tuple(repo.public_manifest_record() for repo in train_repositories)
    private_dev_records = tuple(repo.private_manifest_record() for repo in dev_repositories)
    private_dev_tasks = tuple(VariationTask.from_microrepo(repo) for repo in dev_repositories)
    requests = []
    for record in train_records:
        for arm_id in COMMISSIONING_ARMS:
            for seed in COMMISSIONING_SEEDS:
                requests.append(
                    GenerationRequest.build(
                        campaign_id=campaign_id,
                        task_record=record,
                        corpus_manifest_digest=corpus_digest,
                        arm_id=arm_id,
                        seed=seed,
                        model_manifest_digest=model_manifest_digest,
                        variation_protocol_digest=variation_protocol_digest,
                    )
                )
    requests.sort(key=lambda request: request.request_id)
    return CommissioningPlan(
        campaign_id=campaign_id,
        corpus_manifest_digest=corpus_digest,
        model_manifest_digest=model_manifest_digest,
        variation_protocol_digest=variation_protocol_digest,
        train_records=train_records,
        generation_requests=tuple(requests),
        _private_dev_records=private_dev_records,
        _private_dev_tasks=private_dev_tasks,
    )


__all__ = [
    "COMMISSIONING_SCHEMA", "GENERATION_REQUEST_COUNT", "PRIVATE_DEV_MANIFEST_SCHEMA",
    "PRIVATE_DEV_TASK_COUNT", "REQUEST_MANIFEST_SCHEMA", "TRAIN_MANIFEST_SCHEMA", "TRAIN_TASK_COUNT",
    "CommissioningPlan", "CommissioningPreparationError", "TRAINER_INPUTS_SCHEMA", "prepare_commissioning",
]
