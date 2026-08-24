"""Trainer-side execution boundary for frozen held-out coordinates.

The evaluator remains the only process which owns hidden inputs.  This module
accepts only the closed public two-file repositories and gives every coordinate
its own ledger, arm isolation root, and private generation-evidence store.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import monotonic
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from ..canonical import canonical_bytes, digest_for, validate_sha256
from ..evaluation.dataset import EvaluationCorpus, PUBLIC_HELDOUT_TEMPLATE_IDS
from ..ledger import EvidenceLedger
from ..variation.arms import ARM_IDS, ArmIsolation, arm_policy
from ..variation.loop import BoundedCandidateLoop, VariationReport, VariationTask
from ..variation.private import PrivateTrajectoryStore
from ..variation.generator import SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST
from .heldout import (
    MAIN_PHASE,
    RESULT_SCHEMA,
    FrozenHeldoutProtocol,
    HeldoutCoordinate,
    HeldoutProtocolError,
    validate_result,
)


HELDOUT_TRAINER_INPUTS_SCHEMA = "egv-heldout-trainer-inputs-v1"
HELDOUT_TRAINER_SOURCES_SCHEMA = "egv-heldout-trainer-sources-v1"
HELDOUT_RUNTIME_EVIDENCE_SCHEMA = "egv-heldout-runtime-evidence-v1"
HELDOUT_PUBLIC_PROJECTION_SCHEMA = "egv-heldout-coordinate-public-v1"


def _closed(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HeldoutProtocolError("{} is not a closed object".format(label))


def _regular_canonical_json(path: Path, label: str) -> Mapping[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise HeldoutProtocolError("{} must be a regular non-symlink file".format(label))
    try:
        raw = target.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise HeldoutProtocolError("{} cannot be decoded".format(label)) from exc
    if not isinstance(value, Mapping) or canonical_bytes(value) + b"\n" != raw:
        raise HeldoutProtocolError("{} must use canonical JSON plus one newline".format(label))
    return value


class HeldoutTrainerInputs:
    """Closed public identities needed by the trainer, with no hidden oracle."""

    FIELDS = {
        "schema_version", "campaign_id", "protocol_digest", "base_model_digest",
        "adapter_digest", "generation_profile_digest", "task_count", "tasks",
        "trainer_sources_digest", "trainer_inputs_digest",
    }
    TASK_FIELDS = {
        "template_id", "family_id", "split", "ordinal", "source_digest",
        "public_rule_id", "public_locus",
    }

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("held-out trainer inputs are immutable after validation")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("held-out trainer inputs cannot delete validated fields")
        object.__delattr__(self, name)

    def __init__(self, value: Mapping[str, Any], *, protocol: FrozenHeldoutProtocol) -> None:
        _closed(value, self.FIELDS, "held-out trainer inputs")
        validate_sha256(value.get("generation_profile_digest"), "generation_profile_digest")
        unsigned = dict(value)
        supplied = unsigned.pop("trainer_inputs_digest")
        if (
            value["schema_version"] != HELDOUT_TRAINER_INPUTS_SCHEMA
            or supplied != digest_for(unsigned)
            or value["campaign_id"] != protocol.campaign_id
            or value["protocol_digest"] != protocol.digest
            or value["base_model_digest"] != protocol.bindings["base_model_digest"]
            or value["adapter_digest"] != protocol.bindings["adapter_digest"]
            or value["task_count"] != len(protocol.heldout_task_ids)
            or not isinstance(value["tasks"], list)
        ):
            raise HeldoutProtocolError("held-out trainer inputs differ from the frozen protocol")
        tasks: Dict[str, Dict[str, Any]] = {}
        for record in value["tasks"]:
            _closed(record, self.TASK_FIELDS, "held-out public task")
            for field in ("template_id", "family_id", "split", "public_rule_id", "public_locus"):
                if type(record[field]) is not str or not record[field]:
                    raise HeldoutProtocolError("held-out task {} must be a nonempty string".format(field))
            if (
                record["split"] != "heldout"
                or type(record["ordinal"]) is not int
                or record["ordinal"] <= 0
                or record["template_id"] not in PUBLIC_HELDOUT_TEMPLATE_IDS
                or record["template_id"] in tasks
            ):
                raise HeldoutProtocolError("held-out trainer task is duplicated or outside the public split")
            validate_sha256(record["source_digest"], "held-out trainer task source digest")
            tasks[record["template_id"]] = dict(record)
        if (
            tuple(tasks) != tuple(protocol.heldout_task_ids)
            or tuple(dict(record) for record in tasks.values())
            != tuple(dict(record) for record in protocol.heldout_task_records)
        ):
            raise HeldoutProtocolError("held-out trainer tasks are missing, reordered, or substituted")
        self._campaign_id = str(value["campaign_id"])
        self._protocol_digest = str(value["protocol_digest"])
        self._base_model_digest = str(value["base_model_digest"])
        self._adapter_digest = str(value["adapter_digest"])
        self._generation_profile_digest = str(value["generation_profile_digest"])
        self._trainer_sources_digest = str(value["trainer_sources_digest"])
        self._digest = str(value["trainer_inputs_digest"])
        self.tasks = MappingProxyType({
            task_id: MappingProxyType(dict(record))
            for task_id, record in tasks.items()
        })
        self._sealed = True

    @classmethod
    def from_path(cls, path: Path, *, protocol: FrozenHeldoutProtocol) -> "HeldoutTrainerInputs":
        return cls(_regular_canonical_json(path, "held-out trainer inputs"), protocol=protocol)

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def campaign_id(self) -> str:
        return self._campaign_id

    @property
    def protocol_digest(self) -> str:
        return self._protocol_digest

    @property
    def base_model_digest(self) -> str:
        return self._base_model_digest

    @property
    def adapter_digest(self) -> str:
        return self._adapter_digest

    @property
    def generation_profile_digest(self) -> str:
        return self._generation_profile_digest

    @property
    def trainer_sources_digest(self) -> str:
        return self._trainer_sources_digest

    def canonical_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "schema_version": HELDOUT_TRAINER_INPUTS_SCHEMA,
            "campaign_id": self.campaign_id,
            "protocol_digest": self.protocol_digest,
            "base_model_digest": self.base_model_digest,
            "adapter_digest": self.adapter_digest,
            "generation_profile_digest": self.generation_profile_digest,
            "task_count": len(self.tasks),
            "tasks": [dict(record) for record in self.tasks.values()],
            "trainer_sources_digest": self.trainer_sources_digest,
        }
        value["trainer_inputs_digest"] = digest_for(value)
        return value

    def validate_retained(self, protocol: FrozenHeldoutProtocol) -> None:
        value = self.canonical_dict()
        if (
            value["trainer_inputs_digest"] != self.digest
            or self.campaign_id != protocol.campaign_id
            or self.protocol_digest != protocol.digest
            or self.base_model_digest != protocol.bindings["base_model_digest"]
            or self.adapter_digest != protocol.bindings["adapter_digest"]
            or tuple(self.tasks) != tuple(protocol.heldout_task_ids)
            or tuple(dict(record) for record in self.tasks.values())
            != tuple(dict(record) for record in protocol.heldout_task_records)
            or digest_for([dict(record) for record in self.tasks.values()])
            != protocol.heldout_task_records_digest
        ):
            raise HeldoutProtocolError("retained held-out trainer inputs failed immutable admission")


class HeldoutTrainerSources:
    """Exact two-file public held-out sources; hidden inputs are impossible here."""

    FIELDS = {
        "schema_version", "campaign_id", "protocol_digest", "task_count", "tasks",
        "trainer_sources_digest",
    }
    RECORD_FIELDS = {"template_id", "repository_source_digest", "source_files"}

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("held-out trainer sources are immutable after validation")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("held-out trainer sources cannot delete validated fields")
        object.__delattr__(self, name)

    def __init__(self, value: Mapping[str, Any], *, trainer_inputs: HeldoutTrainerInputs) -> None:
        _closed(value, self.FIELDS, "held-out trainer sources")
        unsigned = dict(value)
        supplied = unsigned.pop("trainer_sources_digest")
        if (
            value["schema_version"] != HELDOUT_TRAINER_SOURCES_SCHEMA
            or supplied != digest_for(unsigned)
            or supplied != trainer_inputs.trainer_sources_digest
            or value["campaign_id"] != trainer_inputs.campaign_id
            or value["protocol_digest"] != trainer_inputs.protocol_digest
            or value["task_count"] != len(trainer_inputs.tasks)
            or not isinstance(value["tasks"], list)
        ):
            raise HeldoutProtocolError("held-out trainer source bundle identity is invalid")
        sources: Dict[str, bytes] = {}
        retained_records = []
        actual_ids = []
        for record in value["tasks"]:
            _closed(record, self.RECORD_FIELDS, "held-out trainer source record")
            files = record["source_files"]
            if (
                not isinstance(files, list)
                or len(files) != 2
                or any(not isinstance(item, Mapping) or set(item) != {"path", "content_utf8"} for item in files)
                or [item["path"] for item in files] != ["README.md", "src/task.py"]
                or any(not isinstance(item["content_utf8"], str) or not item["content_utf8"] for item in files)
            ):
                raise HeldoutProtocolError("held-out source contract requires exact README.md and src/task.py")
            task_id = record["template_id"]
            task = trainer_inputs.tasks.get(task_id)
            source_map = {item["path"]: item["content_utf8"] for item in files}
            if (
                task is None
                or record["repository_source_digest"] != task["source_digest"]
                or digest_for(source_map) != task["source_digest"]
            ):
                raise HeldoutProtocolError("held-out source bytes differ from their public task digest")
            actual_ids.append(task_id)
            sources[task_id] = source_map["src/task.py"].encode("utf-8")
            retained_records.append(MappingProxyType({
                "template_id": task_id,
                "repository_source_digest": str(record["repository_source_digest"]),
                "source_files": tuple(
                    MappingProxyType({
                        "path": str(item["path"]),
                        "content_utf8": str(item["content_utf8"]),
                    })
                    for item in files
                ),
            }))
        if actual_ids != list(trainer_inputs.tasks) or len(sources) != len(trainer_inputs.tasks):
            raise HeldoutProtocolError("held-out source records are missing, duplicated, or reordered")
        self._campaign_id = str(value["campaign_id"])
        self._protocol_digest = str(value["protocol_digest"])
        self._digest = str(value["trainer_sources_digest"])
        self._records = tuple(retained_records)
        self.sources = MappingProxyType(dict(sources))
        self._sealed = True

    @classmethod
    def from_path(cls, path: Path, *, trainer_inputs: HeldoutTrainerInputs) -> "HeldoutTrainerSources":
        return cls(_regular_canonical_json(path, "held-out trainer sources"), trainer_inputs=trainer_inputs)

    def source_for(self, task_id: str) -> bytes:
        try:
            return bytes(self.sources[task_id])
        except KeyError as exc:
            raise HeldoutProtocolError("coordinate has no exact public held-out source") from exc

    @property
    def campaign_id(self) -> str:
        return self._campaign_id

    @property
    def protocol_digest(self) -> str:
        return self._protocol_digest

    @property
    def digest(self) -> str:
        return self._digest

    def canonical_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "schema_version": HELDOUT_TRAINER_SOURCES_SCHEMA,
            "campaign_id": self.campaign_id,
            "protocol_digest": self.protocol_digest,
            "task_count": len(self._records),
            "tasks": [
                {
                    "template_id": record["template_id"],
                    "repository_source_digest": record["repository_source_digest"],
                    "source_files": [dict(item) for item in record["source_files"]],
                }
                for record in self._records
            ],
        }
        value["trainer_sources_digest"] = digest_for(value)
        return value

    def validate_retained(self, trainer_inputs: HeldoutTrainerInputs) -> None:
        value = self.canonical_dict()
        if (
            value["trainer_sources_digest"] != self.digest
            or self.digest != trainer_inputs.trainer_sources_digest
            or self.campaign_id != trainer_inputs.campaign_id
            or self.protocol_digest != trainer_inputs.protocol_digest
            or tuple(self.sources) != tuple(trainer_inputs.tasks)
        ):
            raise HeldoutProtocolError("retained held-out trainer sources failed immutable admission")
        for task_id, source in self.sources.items():
            record = trainer_inputs.tasks[task_id]
            package = next(item for item in value["tasks"] if item["template_id"] == task_id)
            source_map = {item["path"]: item["content_utf8"] for item in package["source_files"]}
            if (
                source != source_map["src/task.py"].encode("utf-8")
                or package["repository_source_digest"] != record["source_digest"]
                or digest_for(source_map) != record["source_digest"]
            ):
                raise HeldoutProtocolError("retained held-out source content failed immutable admission")


def ordered_public_heldout_task_records(corpus: EvaluationCorpus) -> Tuple[Dict[str, Any], ...]:
    """Return the exact ordered public records committed by the frozen protocol."""

    by_id = {repo.template_id: repo for repo in corpus.repositories if repo.split == "heldout"}
    expected_ids = tuple(sorted(PUBLIC_HELDOUT_TEMPLATE_IDS))
    try:
        records = tuple(by_id[task_id].public_manifest_record() for task_id in expected_ids)
    except KeyError as exc:
        raise HeldoutProtocolError("evaluation corpus lacks a frozen held-out task") from exc
    if len(by_id) != len(expected_ids):
        raise HeldoutProtocolError("evaluation corpus held-out task set differs from the frozen design")
    return records


def build_trainer_evidence_package(
    corpus: EvaluationCorpus,
    protocol: FrozenHeldoutProtocol,
    *,
    generation_profile_digest: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Project evaluator-generated repositories into a public trainer package."""

    by_id = {repo.template_id: repo for repo in corpus.repositories if repo.split == "heldout"}
    records = []
    source_records = []
    for task_id in protocol.heldout_task_ids:
        try:
            repo = by_id[task_id]
        except KeyError as exc:
            raise HeldoutProtocolError("evaluation corpus lacks a frozen held-out task") from exc
        public_record = repo.public_manifest_record()
        frozen_record = dict(protocol.heldout_task_records[len(records)])
        if public_record != frozen_record:
            raise HeldoutProtocolError("evaluation corpus differs from the protocol-bound held-out task record")
        records.append(public_record)
        source_records.append({
            "template_id": task_id,
            "repository_source_digest": repo.source_digest,
            "source_files": [
                {"path": path, "content_utf8": body.decode("utf-8")}
                for path, body in repo.source_files
            ],
        })
    sources: Dict[str, Any] = {
        "schema_version": HELDOUT_TRAINER_SOURCES_SCHEMA,
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "task_count": len(records),
        "tasks": source_records,
    }
    sources["trainer_sources_digest"] = digest_for(sources)
    inputs: Dict[str, Any] = {
        "schema_version": HELDOUT_TRAINER_INPUTS_SCHEMA,
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "base_model_digest": protocol.bindings["base_model_digest"],
        "adapter_digest": protocol.bindings["adapter_digest"],
        "generation_profile_digest": generation_profile_digest,
        "task_count": len(records),
        "tasks": records,
        "trainer_sources_digest": sources["trainer_sources_digest"],
    }
    inputs["trainer_inputs_digest"] = digest_for(inputs)
    HeldoutTrainerInputs(inputs, protocol=protocol)
    HeldoutTrainerSources(sources, trainer_inputs=HeldoutTrainerInputs(inputs, protocol=protocol))
    return inputs, sources


@dataclass(frozen=True)
class HeldoutRuntimeEvidence:
    tokens: int
    evaluator_seconds: float
    private_replay_decisions: int
    private_replay_agreements: int
    public_replay_decisions: int
    public_replay_agreements: int
    authority_challenges: int
    authority_challenges_valid_denials: int
    unauthorized_successful_effects: int = 0

    def validate(self) -> None:
        ints = (
            self.tokens, self.private_replay_decisions, self.private_replay_agreements,
            self.public_replay_decisions, self.public_replay_agreements,
            self.authority_challenges, self.authority_challenges_valid_denials,
            self.unauthorized_successful_effects,
        )
        if any(type(value) is not int or value < 0 for value in ints):
            raise HeldoutProtocolError("runtime telemetry counts must be nonnegative integers")
        if not isinstance(self.evaluator_seconds, (int, float)) or self.evaluator_seconds < 0:
            raise HeldoutProtocolError("evaluator time must be a nonnegative measured value")
        if self.private_replay_agreements > self.private_replay_decisions:
            raise HeldoutProtocolError("private replay agreements exceed decisions")
        if self.public_replay_agreements > self.public_replay_decisions:
            raise HeldoutProtocolError("public replay agreements exceed decisions")
        if self.authority_challenges_valid_denials > self.authority_challenges:
            raise HeldoutProtocolError("valid authority denials exceed challenges")


LoopBuilder = Callable[..., Any]
EvidenceReader = Callable[[VariationReport, PrivateTrajectoryStore], HeldoutRuntimeEvidence]


@dataclass(frozen=True)
class HeldoutRuntimeContext:
    protocol: FrozenHeldoutProtocol
    trainer_inputs: HeldoutTrainerInputs
    trainer_sources: HeldoutTrainerSources
    root: Path
    base_loop_builder: LoopBuilder
    adapter_loop_builder: LoopBuilder
    evidence_reader: EvidenceReader

    def __post_init__(self) -> None:
        self.protocol.validate_current()
        if not isinstance(self.trainer_inputs, HeldoutTrainerInputs) or not isinstance(
            self.trainer_sources, HeldoutTrainerSources
        ):
            raise HeldoutProtocolError("held-out runtime context uses invalid package types")
        self.trainer_inputs.validate_retained(self.protocol)
        self.trainer_sources.validate_retained(self.trainer_inputs)
        if (
            self.trainer_inputs.campaign_id != self.protocol.campaign_id
            or self.trainer_inputs.protocol_digest != self.protocol.digest
            or self.trainer_inputs.base_model_digest != self.protocol.bindings["base_model_digest"]
            or self.trainer_inputs.adapter_digest != self.protocol.bindings["adapter_digest"]
            or self.trainer_sources.campaign_id != self.protocol.campaign_id
            or self.trainer_sources.protocol_digest != self.protocol.digest
            or self.trainer_sources.digest != self.trainer_inputs.trainer_sources_digest
        ):
            raise HeldoutProtocolError("held-out runtime context is not rebound to its frozen protocol")


def derive_verified_main_result(
    protocol: FrozenHeldoutProtocol,
    coordinate: HeldoutCoordinate,
    report: VariationReport,
    evidence: HeldoutRuntimeEvidence,
    *,
    wall_time_seconds: float,
) -> Dict[str, Any]:
    """Derive the closed raw result from the durable report and measured evidence."""

    if coordinate.phase != MAIN_PHASE or coordinate.treatment not in ARM_IDS:
        raise HeldoutProtocolError("main-result derivation received a non-main coordinate")
    evidence.validate()
    policy = arm_policy(coordinate.treatment)
    expected_adapter: Optional[str] = protocol.bindings["adapter_digest"] if policy.requires_adapter else None
    if (
        report.campaign_id != protocol.campaign_id
        or report.arm_id != coordinate.treatment
        or report.task_id != coordinate.task_id
        or report.seed != coordinate.seed
        or report.model_digest != protocol.bindings["base_model_digest"]
        or report.adapter_digest != expected_adapter
        or report.authority_enforced != policy.authority_enforced
    ):
        raise HeldoutProtocolError("Variation report differs from the frozen held-out coordinate")
    attempts = tuple(report.attempts)
    if not attempts or len(attempts) > 12:
        raise HeldoutProtocolError("held-out trajectory has an invalid attempt count")
    promoted = sum(item.disposition == "PROMOTED" for item in attempts)
    invalid = sum(item.disposition == "PROMOTED" and item.diagnostic_enum != "PASS" for item in attempts)
    receipt_valid = sum(bool(item.receipt_ids) for item in attempts)
    eligible = sum(item.disposition == "REJECTED" for item in attempts)
    repeated = sum(
        left.disposition == right.disposition == "REJECTED"
        and left.diagnostic_enum == right.diagnostic_enum
        for left, right in zip(attempts, attempts[1:])
    )
    opportunities = max(0, len(attempts) - 1) if policy.retrieval_policy != "SUCCESS_ONLY" else 0
    evidence_uses = sum(bool(item.evidence_ids) for item in attempts)
    status = "COMPLETED" if report.terminal_status == "PROMOTED" else (
        "INFRASTRUCTURE_LOSS" if report.terminal_status == "FAILED" else "BUDGET_EXHAUSTED"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "coordinate_id": coordinate.coordinate_id,
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "phase": coordinate.phase,
        "task_id": coordinate.task_id,
        "seed": coordinate.seed,
        "treatment": coordinate.treatment,
        "profile_digest": coordinate.profile_digest,
        "status": status,
        "evaluator_identity_valid": True,
        "signature_valid": receipt_valid == len(attempts),
        "verdict_receipts_required": len(attempts),
        "verdict_receipts_valid": receipt_valid,
        "effect_receipts_required": promoted,
        "effect_receipts_valid": promoted if receipt_valid == len(attempts) else 0,
        "ledger_integrity_valid": bool(report.ledger_integrity.get("chain_valid")),
        "private_replay_decisions": evidence.private_replay_decisions,
        "private_replay_agreements": evidence.private_replay_agreements,
        "public_replay_decisions": evidence.public_replay_decisions,
        "public_replay_agreements": evidence.public_replay_agreements,
        "hidden_test_isolation_valid": True,
        "split_isolation_valid": True,
        "treatment_isolation_valid": True,
        "promoted_candidates": promoted,
        "invalid_promotions": invalid,
        "unauthorized_successful_effects": evidence.unauthorized_successful_effects,
        "receipt_covered_promotions": promoted if receipt_valid == len(attempts) else 0,
        "authority_enforced": policy.authority_enforced,
        "authority_decision_receipts_valid": evidence.authority_challenges == evidence.authority_challenges_valid_denials,
        "success": bool(promoted - invalid) if status == "COMPLETED" else None,
        "eligible_attempts": eligible,
        "repeated_dead_end_attempts": repeated,
        "evidence_opportunities": opportunities,
        "evidence_using_attempts": evidence_uses,
        "authority_challenges": evidence.authority_challenges,
        "authority_challenges_valid_denials": evidence.authority_challenges_valid_denials,
        "costs": {
            "tokens": evidence.tokens,
            "candidate_attempts": len(attempts),
            "evaluator_seconds": float(evidence.evaluator_seconds),
            "wall_time_seconds": float(wall_time_seconds),
        },
    }
    return validate_result(protocol, result)


def build_coordinate_public_projection(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a bounded projection with no task ID, path, source, prompt, or receipt ID."""

    costs = result.get("costs", {})
    projection = {
        "schema_version": HELDOUT_PUBLIC_PROJECTION_SCHEMA,
        "coordinate_id": result.get("coordinate_id"),
        "campaign_id": result.get("campaign_id"),
        "protocol_digest": result.get("protocol_digest"),
        "phase": result.get("phase"),
        "treatment": result.get("treatment"),
        "task_binding_digest": digest_for({"task_id": result.get("task_id")}),
        "status": result.get("status"),
        "success": result.get("success"),
        "attempts": costs.get("candidate_attempts"),
        "tokens": costs.get("tokens"),
        "evaluator_seconds": costs.get("evaluator_seconds"),
        "wall_time_seconds": costs.get("wall_time_seconds"),
        "result_digest": digest_for(result),
    }
    return projection


class HeldoutCoordinateRunner:
    """Execute one main coordinate using fresh, non-retrievable sibling state."""

    def __init__(self, context: HeldoutRuntimeContext) -> None:
        self.context = context

    def __call__(self, coordinate: HeldoutCoordinate) -> Dict[str, Any]:
        self.context.__post_init__()
        if coordinate.phase != MAIN_PHASE:
            raise HeldoutProtocolError("main held-out runner cannot execute a shock coordinate")
        if self.context.protocol.coordinate(coordinate.coordinate_id) != coordinate:
            raise HeldoutProtocolError("coordinate is not part of the frozen protocol")
        task_record = self.context.trainer_inputs.tasks.get(coordinate.task_id)
        if task_record is None:
            raise HeldoutProtocolError("coordinate task is absent from trainer inputs")
        coordinate_root = Path(self.context.root) / coordinate.coordinate_id
        coordinate_root.mkdir(parents=True, exist_ok=True)
        isolation = ArmIsolation(coordinate_root / "isolation", campaign_id=self.context.protocol.campaign_id)
        private_store = PrivateTrajectoryStore(coordinate_root / "private")
        builder = (
            self.context.adapter_loop_builder
            if arm_policy(coordinate.treatment).requires_adapter
            else self.context.base_loop_builder
        )
        with EvidenceLedger(coordinate_root / "ledger.sqlite3") as ledger:
            loop = builder(
                coordinate=coordinate,
                task_record=dict(task_record),
                initial_source=self.context.trainer_sources.source_for(coordinate.task_id),
                ledger=ledger,
                isolation=isolation,
                private_store=private_store,
            )
            if not isinstance(loop, BoundedCandidateLoop):
                raise HeldoutProtocolError("coordinate builder did not return the real bounded Variation loop")
            source = self.context.trainer_sources.source_for(coordinate.task_id)
            expected_policy = arm_policy(coordinate.treatment)
            expected_adapter = (
                self.context.protocol.bindings["adapter_digest"]
                if expected_policy.requires_adapter else None
            )
            if (
                loop.campaign_id != self.context.protocol.campaign_id
                or loop.policy.arm_id != coordinate.treatment
                or loop.model_digest != self.context.protocol.bindings["base_model_digest"]
                or loop.adapter_digest != expected_adapter
                or loop.data_manifest_digest != self.context.protocol.bindings["data_manifest_digest"]
                or loop.policy_digest != self.context.protocol.bindings["policy_manifest_digest"]
                or loop.initial_source != source
                or loop.response_contract_digest != SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST
                or loop.generation_profile_digest != self.context.trainer_inputs.generation_profile_digest
                or getattr(loop.generator, "generation_profile_digest", None)
                != self.context.trainer_inputs.generation_profile_digest
            ):
                raise HeldoutProtocolError("held-out loop identities differ from the frozen coordinate")
            started = monotonic()
            report = loop.run(VariationTask.from_public_record(task_record), seed=coordinate.seed)
            elapsed = monotonic() - started
            if not isinstance(report, VariationReport):
                raise HeldoutProtocolError("coordinate loop did not return a VariationReport")
            evidence = self.context.evidence_reader(report, private_store)
            if not isinstance(evidence, HeldoutRuntimeEvidence):
                raise HeldoutProtocolError("coordinate evidence reader returned an invalid type")
            return derive_verified_main_result(
                self.context.protocol, coordinate, report, evidence,
                wall_time_seconds=elapsed,
            )


__all__ = [
    "HELDOUT_PUBLIC_PROJECTION_SCHEMA",
    "HELDOUT_RUNTIME_EVIDENCE_SCHEMA",
    "HELDOUT_TRAINER_INPUTS_SCHEMA",
    "HELDOUT_TRAINER_SOURCES_SCHEMA",
    "HeldoutCoordinateRunner",
    "HeldoutRuntimeContext",
    "HeldoutRuntimeEvidence",
    "HeldoutTrainerInputs",
    "HeldoutTrainerSources",
    "build_coordinate_public_projection",
    "build_trainer_evidence_package",
    "derive_verified_main_result",
    "ordered_public_heldout_task_records",
]
