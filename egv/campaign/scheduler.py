"""Deterministic, authenticated control-plane scheduling for EGV campaigns.

This module never launches work.  It persists a dependency-aware state machine,
verifies signed activity evidence, emits bounded controller actions, and returns
logical dispatches.  The local state file is explicitly a trusted but
non-authoritative control-plane cache; it is not scientific or cross-host proof.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..canonical import GENESIS_HASH, canonical_bytes, digest_for, validate_sha256
from ..errors import OptionalDependencyError, ReceiptVerificationError
from ..receipts import key_id_for_public_key, public_key_bytes, receipt_hash, verify_receipt
from .errors import CampaignError
from .state import _durable_replace, _exclusive_path_lock


SCHEDULER_STATE_SCHEMA = "egv-campaign-scheduler-state-v4"
SCHEDULER_PUBLIC_SCHEMA = "egv-campaign-scheduler-public-v3"
SCHEDULER_JOURNAL_SCHEMA = "egv-campaign-scheduler-transition-journal-v1"
SCHEDULER_EFFECT_FENCE_SCHEMA = "egv-campaign-scheduler-effect-fence-v1"
STATE_AUTHORITY = "TRUSTED_LOCAL_CONTROL_PLANE_NON_AUTHORITATIVE"
CLAIM_BOUNDARY = "ENGINEERING_ONLY_NO_MODEL_QUALITY_OR_LIVE_DISTRIBUTED_CLAIM"
PROGRESS_CHECKPOINT_MS = 5 * 60 * 1000
STALL_TIMEOUT_MS = 10 * 60 * 1000
AVOIDABLE_IDLE_DEFECT_MS = 5 * 60 * 1000
PILOT_CHECKPOINT_REQUEST = 6
MAX_STALL_RETRIES = 1
MAX_PROGRESS_EVIDENCE_DIGESTS = 1024
PROGRESS_KINDS = {
    "GENERATION": "GENERATED_ARTIFACT",
    "EVALUATION": "EVALUATION_ARTIFACT",
    "DIAGNOSTIC": "DIAGNOSTIC_ARTIFACT",
    "TRAINING": "TRAINING_CHECKPOINT",
    "VALIDATION": "VALIDATION_ARTIFACT",
    "PROMOTION": "PROMOTION_ARTIFACT",
    "REVIEW": "REVIEW_ARTIFACT",
}

RESOURCE_POOLS = frozenset({"ACCELERATOR", "CPU"})
TASK_OWNER_ROLES = frozenset({"CONTROLLER", "EVALUATOR", "REVIEWER", "TRAINER"})
TASK_KINDS = frozenset({"GENERATION", "EVALUATION", "DIAGNOSTIC", "TRAINING", "VALIDATION", "PROMOTION", "REVIEW"})
WORKER_CAPABILITIES = TASK_KINDS | frozenset({"CANCEL", "OBSERVE"})
TASK_STAGES = frozenset({"SCREENING", "POST_SCREEN", "DIAGNOSTIC"})
POST_SCREEN_KINDS = frozenset({"TRAINING", "VALIDATION", "PROMOTION"})
TASK_STATUSES = frozenset({"PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"})
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
ARM_STATUSES = frozenset({"UNGATED", "SCREENING", "CONTINUING", "PAUSED_ZERO_SIGNAL", "INCOMPLETE_FAIL_CLOSED"})
TERMINAL_REASONS = frozenset(
    {
        "COMPLETED",
        "ACTIVITY_FAILED",
        "STALL_RETRY_EXHAUSTED",
        "STALL_NO_REASSIGNMENT",
        "DEPENDENCY_NOT_COMPLETED",
        "ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT",
        "CHECKPOINT_INCOMPLETE_FAIL_CLOSED",
    }
)
ATTEMPT_DISPOSITIONS = frozenset({"RUNNING", "COMPLETED", "ACTIVITY_FAILED", "STALL_RETRY", "STALL_FAILED"})
CONTROL_ACTIONS = frozenset({"EMIT_PROGRESS_CHECKPOINT", "CANCEL_STALLED_ATTEMPT"})

_SAFE_INTERNAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_PUBLIC_ALIAS = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_PUBLIC_SLOT_ALIAS = re.compile(r"^slot-(?:[a-z]|[1-9][0-9]{0,2})$")
_HEX_PUBLIC_KEY = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_MUTEX = threading.RLock()

_PUBLIC_FORBIDDEN_TOKENS = frozenset(
    {
        "api",
        "auth",
        "container",
        "credential",
        "db",
        "evaluator",
        "gateway",
        "host",
        "hostname",
        "key",
        "node",
        "password",
        "pod",
        "private",
        "qdrant",
        "router",
        "secret",
        "service",
        "spark",
        "token",
        "trainer",
        "user",
        "worker",
    }
)

_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "state_authority",
        "campaign_id",
        "plan_digest",
        "effect_fence_instance_id",
        "sequence",
        "created_ms",
        "updated_ms",
        "config",
        "workers",
        "arms",
        "nodes",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "progress_checkpoint_ms",
        "stall_timeout_ms",
        "avoidable_idle_defect_ms",
        "pilot_checkpoint_request",
        "max_stall_retries",
    }
)
_WORKER_SPEC_FIELDS = frozenset(
    {"worker_id", "resource_pool", "owner_role", "capabilities", "public_key_hex", "signing_key_id"}
)
_WORKER_STATE_FIELDS = _WORKER_SPEC_FIELDS | frozenset(
    {
        "busy_ms",
        "idle_ms",
        "avoidable_idle_ms",
        "avoidable_idle_streak_ms",
        "max_avoidable_idle_streak_ms",
    }
)
_TASK_SPEC_FIELDS = frozenset(
    {
        "node_id",
        "kind",
        "resource_pool",
        "owner_role",
        "input_digest",
        "stage",
        "dependencies",
        "arm_id",
        "request_index",
        "promotion_key",
        "promotion_authority_node_id",
        "manifest_ordinal",
        "counterbalance_rank",
        "node_priority_tuple",
    }
)
_NODE_STATE_FIELDS = _TASK_SPEC_FIELDS | frozenset(
    {
        "status",
        "assigned_worker_id",
        "attempt_count",
        "failed_worker_ids",
        "started_ms",
        "last_progress_ms",
        "finished_ms",
        "resolved_input_digest",
        "progress_digest",
        "progress_evidence_digests",
        "output_digest",
        "semantic_passed",
        "terminal_reason",
        "attempt_receipts",
        "positive_promotion_authority",
    }
)
_ATTEMPT_FIELDS = frozenset(
    {
        "attempt",
        "attempt_id",
        "worker_id",
        "resolved_input_digest",
        "started_ms",
        "last_progress_ms",
        "finished_ms",
        "disposition",
        "receipts",
        "checkpoint_digest",
        "budget_limit",
        "budget_used",
        "predecessor_attempt_id",
        "predecessor_receipt_hash",
        "isolation_digest",
        "lease_id",
        "release_digest",
        "capability_digest",
        "model_digest",
        "protocol_digest",
        "evaluator_digest",
        "sandbox_digest",
        "manifest_digest",
        "monotonic_counter",
        "execution_idempotency_key",
        "progress_observations",
        "termination_observation",
    }
)
_ARM_STATE_FIELDS = frozenset(
    {
        "arm_id",
        "status",
        "checkpoint_complete",
        "checkpoint_terminal_requests",
        "checkpoint_completed_requests",
        "checkpoint_semantic_passes",
    }
)
_SIGNED_ACTIVITY_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_type",
        "campaign_id",
        "run_id",
        "task_id",
        "receipt_id",
        "request_id",
        "candidate_id",
        "decision",
        "sequence",
        "previous_receipt_hash",
        "idempotency_key",
        "signing_key_id",
        "signature",
        "input_digest",
        "output_digest",
        "resource_bucket",
        "effect_kind",
        "protocol_digest",
        "policy_digest",
        "arm_policy_digest",
        "evaluator_digest",
        "candidate_artifact_digest",
        "environment_diff_digest",
        "normalized_action_hash",
        "public_dependency_set_digest",
        "sandbox_id",
    }
)

_PROMOTION_AUTHORITY_FIELDS = frozenset(
    {"authority_node_id", "candidate_digest", "decision", "attempt_id", "receipt_hash", "chain_head"}
)
_PUBLIC_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_digest",
        "plan_digest",
        "state_sequence",
        "window_ms",
        "workers",
        "aggregate",
        "tasks",
        "arms",
        "state_authority",
        "claim_boundary",
    }
)
_PUBLIC_WORKER_FIELDS = frozenset(
    {
        "alias",
        "resource_pool",
        "busy_ms",
        "idle_ms",
        "avoidable_idle_ms",
        "legitimately_blocked_ms",
        "max_avoidable_idle_streak_ms",
        "avoidable_idle_defect",
    }
)
_PUBLIC_AGGREGATE_FIELDS = frozenset(
    {
        "capacity_ms",
        "busy_ms",
        "idle_ms",
        "avoidable_idle_ms",
        "legitimately_blocked_ms",
        "max_avoidable_idle_streak_ms",
        "avoidable_idle_defect",
    }
)

_EXTERNAL_OBSERVATION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_type",
        "campaign_id",
        "run_id",
        "task_id",
        "receipt_id",
        "request_id",
        "candidate_id",
        "decision",
        "sequence",
        "previous_receipt_hash",
        "idempotency_key",
        "signing_key_id",
        "signature",
        "input_digest",
        "output_digest",
        "resource_bucket",
        "effect_kind",
        "protocol_digest",
        "policy_digest",
        "arm_policy_digest",
        "evaluator_digest",
        "candidate_artifact_digest",
        "environment_diff_digest",
        "normalized_action_hash",
        "public_dependency_set_digest",
        "sandbox_id",
        "exit_status_class",
    }
)

_TERMINATION_EXIT_CLASSES = frozenset(
    {"SUCCESS", "NONZERO", "TIMEOUT", "SIGNAL", "RESOURCE_LIMIT", "OUTPUT_LIMIT", "INFRASTRUCTURE_LOSS"}
)


class CampaignSchedulerError(CampaignError):
    """A scheduler transition, state record, or public projection is unsafe."""


def _require_internal_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_INTERNAL_ID.fullmatch(value):
        raise CampaignSchedulerError("{} must be a bounded logical ID".format(field))
    if ".." in value or "/" in value or "\\" in value:
        raise CampaignSchedulerError("{} contains a path-like value".format(field))
    return value


def _require_digest(value: Any, field: str) -> str:
    try:
        return validate_sha256(value, field)
    except Exception as exc:
        raise CampaignSchedulerError(str(exc)) from exc


def _require_nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CampaignSchedulerError("{} must be a nonnegative integer".format(field))
    return value


def _require_closed(value: Any, fields: frozenset, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CampaignSchedulerError("{} is not a closed schema".format(label))
    return value


def _require_public_alias(value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_PUBLIC_ALIAS.fullmatch(value):
        raise CampaignSchedulerError("public worker alias must be a bounded lowercase pseudonym")
    tokens = frozenset(token for token in value.split("-") if token)
    if tokens & _PUBLIC_FORBIDDEN_TOKENS:
        raise CampaignSchedulerError("public worker alias contains secret or topology vocabulary")
    if not _PUBLIC_SLOT_ALIAS.fullmatch(value):
        raise CampaignSchedulerError("public worker alias must use the closed slot pseudonym format")
    return value


@dataclass(frozen=True)
class SchedulerWorker:
    worker_id: str
    resource_pool: str
    owner_role: str
    capabilities: Tuple[str, ...]
    public_key: bytes

    def __post_init__(self) -> None:
        _require_internal_id(self.worker_id, "worker ID")
        if self.resource_pool not in RESOURCE_POOLS:
            raise CampaignSchedulerError("worker resource pool is outside the closed vocabulary")
        if self.owner_role not in TASK_OWNER_ROLES:
            raise CampaignSchedulerError("worker owner role is outside the closed vocabulary")
        if isinstance(self.capabilities, (str, bytes)):
            raise CampaignSchedulerError("worker capabilities must be an ordered sequence")
        capabilities = tuple(sorted(tuple(self.capabilities)))
        if not capabilities or len(capabilities) != len(set(capabilities)):
            raise CampaignSchedulerError("worker capabilities must be nonempty and unique")
        if not set(capabilities) <= WORKER_CAPABILITIES:
            raise CampaignSchedulerError("worker capability is outside the closed vocabulary")
        object.__setattr__(self, "capabilities", capabilities)
        try:
            raw = public_key_bytes(self.public_key)
        except Exception as exc:
            raise CampaignSchedulerError("worker public key is not valid Ed25519 material") from exc
        object.__setattr__(self, "public_key", raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "resource_pool": self.resource_pool,
            "owner_role": self.owner_role,
            "capabilities": list(self.capabilities),
            "public_key_hex": self.public_key.hex(),
            "signing_key_id": key_id_for_public_key(self.public_key),
        }


@dataclass(frozen=True)
class SchedulerTask:
    node_id: str
    kind: str
    resource_pool: str
    owner_role: str
    input_digest: str
    stage: str
    manifest_ordinal: int
    counterbalance_rank: int
    dependencies: Tuple[str, ...] = tuple()
    arm_id: Optional[str] = None
    request_index: Optional[int] = None
    promotion_key: Optional[str] = None
    promotion_authority_node_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_internal_id(self.node_id, "scheduler node ID")
        if self.kind not in TASK_KINDS or self.resource_pool not in RESOURCE_POOLS:
            raise CampaignSchedulerError("scheduler task kind or resource pool is invalid")
        if self.owner_role not in TASK_OWNER_ROLES:
            raise CampaignSchedulerError("scheduler task owner is outside the closed vocabulary")
        _require_digest(self.input_digest, "scheduler task input digest")
        if self.stage not in TASK_STAGES:
            raise CampaignSchedulerError("scheduler task stage is outside the closed vocabulary")
        if isinstance(self.dependencies, (str, bytes)):
            raise CampaignSchedulerError("scheduler dependencies must be an ordered sequence of IDs")
        try:
            dependencies = tuple(sorted(tuple(self.dependencies)))
        except TypeError as exc:
            raise CampaignSchedulerError("scheduler dependencies must be an ordered sequence") from exc
        if len(dependencies) != len(set(dependencies)):
            raise CampaignSchedulerError("scheduler task carries duplicate dependencies")
        for dependency in dependencies:
            _require_internal_id(dependency, "dependency ID")
        if self.node_id in dependencies:
            raise CampaignSchedulerError("scheduler task cannot depend on itself")
        object.__setattr__(self, "dependencies", dependencies)
        if self.arm_id is not None:
            _require_internal_id(self.arm_id, "arm ID")
        if self.arm_id is not None and self.kind in {"GENERATION", "EVALUATION"} and self.request_index is None:
            raise CampaignSchedulerError("arm-bound generation and evaluation require a frozen request index")
        if self.request_index is not None:
            if self.arm_id is None or not isinstance(self.request_index, int) or isinstance(self.request_index, bool):
                raise CampaignSchedulerError("request-indexed task requires an arm and positive index")
            if self.request_index < 1:
                raise CampaignSchedulerError("request index must be positive")
            expected_stage = "SCREENING" if self.request_index <= PILOT_CHECKPOINT_REQUEST else "POST_SCREEN"
            if self.stage != expected_stage:
                raise CampaignSchedulerError("request-indexed task stage bypasses the frozen checkpoint")
        if self.kind == "EVALUATION" and (self.arm_id is None or self.request_index is None):
            raise CampaignSchedulerError("evaluation task requires arm and request bindings")
        if self.kind == "DIAGNOSTIC" and self.stage != "DIAGNOSTIC":
            raise CampaignSchedulerError("diagnostic task requires the diagnostic stage")
        if self.stage == "DIAGNOSTIC" and self.kind != "DIAGNOSTIC":
            raise CampaignSchedulerError("only diagnostic tasks may use the diagnostic stage")
        if self.kind in POST_SCREEN_KINDS and self.stage != "POST_SCREEN":
            raise CampaignSchedulerError("training, validation, and promotion are post-screen tasks")
        if self.stage == "POST_SCREEN" and self.arm_id is None:
            raise CampaignSchedulerError("post-screen task requires an arm binding")
        if self.kind == "PROMOTION":
            if self.promotion_key is None or self.promotion_authority_node_id is None:
                raise CampaignSchedulerError("promotion task requires candidate and positive authority bindings")
            _require_digest(self.promotion_key, "promotion candidate evidence digest")
            _require_internal_id(self.promotion_authority_node_id, "promotion authority node ID")
            if self.promotion_authority_node_id not in dependencies:
                raise CampaignSchedulerError("promotion authority must be an intended direct dependency")
        elif self.promotion_key is not None or self.promotion_authority_node_id is not None:
            raise CampaignSchedulerError("only a promotion task may carry promotion authority")
        _require_nonnegative_int(self.manifest_ordinal, "scheduler task manifest ordinal")
        _require_nonnegative_int(self.counterbalance_rank, "scheduler task counterbalance rank")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "resource_pool": self.resource_pool,
            "owner_role": self.owner_role,
            "input_digest": self.input_digest,
            "stage": self.stage,
            "dependencies": list(self.dependencies),
            "arm_id": self.arm_id,
            "request_index": self.request_index,
            "promotion_key": self.promotion_key,
            "promotion_authority_node_id": self.promotion_authority_node_id,
            "manifest_ordinal": self.manifest_ordinal,
            "counterbalance_rank": self.counterbalance_rank,
            "node_priority_tuple": list(self.node_priority_tuple),
        }

    @property
    def node_priority_tuple(self) -> Tuple[int, int, int, int, int, str]:
        stage_rank = {"SCREENING": 0, "DIAGNOSTIC": 1, "POST_SCREEN": 2}[self.stage]
        kind_rank = {kind: index for index, kind in enumerate(sorted(TASK_KINDS))}[self.kind]
        request_rank = self.request_index if self.request_index is not None else 2**31 - 1
        return (
            stage_rank,
            request_rank,
            self.manifest_ordinal,
            self.counterbalance_rank,
            kind_rank,
            self.node_id,
        )


@dataclass(frozen=True)
class SchedulerPlan:
    campaign_id: str
    workers: Tuple[SchedulerWorker, ...]
    tasks: Tuple[SchedulerTask, ...]

    def __post_init__(self) -> None:
        _require_internal_id(self.campaign_id, "scheduler campaign ID")
        if isinstance(self.workers, (str, bytes)) or isinstance(self.tasks, (str, bytes)):
            raise CampaignSchedulerError("scheduler workers and tasks must be ordered sequences")
        workers = tuple(self.workers)
        tasks = tuple(self.tasks)
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "tasks", tasks)
        if not workers or not tasks:
            raise CampaignSchedulerError("scheduler plan requires workers and tasks")
        if any(type(worker) is not SchedulerWorker for worker in workers) or any(
            type(task) is not SchedulerTask for task in tasks
        ):
            raise CampaignSchedulerError("scheduler plan requires exact validated records")
        worker_ids = [worker.worker_id for worker in workers]
        worker_key_ids = [worker.to_dict()["signing_key_id"] for worker in workers]
        node_ids = [task.node_id for task in tasks]
        if len(worker_ids) != len(set(worker_ids)) or len(node_ids) != len(set(node_ids)):
            raise CampaignSchedulerError("scheduler plan identities must be unique")
        if len(worker_key_ids) != len(set(worker_key_ids)):
            raise CampaignSchedulerError("scheduler worker signing authorities must be cryptographically distinct")
        priority_seals = [(task.manifest_ordinal, task.counterbalance_rank) for task in tasks]
        if len(priority_seals) != len(set(priority_seals)):
            raise CampaignSchedulerError("scheduler task manifest/counterbalance priority seals must be unique")
        known_nodes = set(node_ids)
        by_node = {task.node_id: task for task in tasks}
        for task in tasks:
            if not set(task.dependencies) <= known_nodes:
                raise CampaignSchedulerError("scheduler task references an unknown dependency")
            compatible = [
                worker
                for worker in workers
                if worker.resource_pool == task.resource_pool
                and worker.owner_role == task.owner_role
                and task.kind in worker.capabilities
            ]
            if not compatible:
                raise CampaignSchedulerError("scheduler task has no role-and-capability-compatible worker")
        remaining = {task.node_id: set(task.dependencies) for task in tasks}
        while remaining:
            ready = sorted(node_id for node_id, dependencies in remaining.items() if not dependencies)
            if not ready:
                raise CampaignSchedulerError("scheduler task graph contains a cycle")
            for node_id in ready:
                remaining.pop(node_id)
            for dependencies in remaining.values():
                dependencies.difference_update(ready)
        promotion_keys = [task.promotion_key for task in tasks if task.promotion_key is not None]
        if len(promotion_keys) != len(set(promotion_keys)):
            raise CampaignSchedulerError("scheduler plan contains duplicate promotion authority")
        for task in tasks:
            if task.kind != "PROMOTION":
                continue
            authority = by_node[task.promotion_authority_node_id]
            if authority.kind not in {"EVALUATION", "VALIDATION"}:
                raise CampaignSchedulerError("promotion authority must be a validation or evaluator node")
            if authority.arm_id != task.arm_id:
                raise CampaignSchedulerError("promotion authority must belong to the promoted arm")
        for arm_id in self.arm_ids:
            checkpoint_ids = {
                task.node_id
                for task in tasks
                if task.arm_id == arm_id
                and task.kind == "EVALUATION"
                and task.request_index is not None
                and task.request_index <= PILOT_CHECKPOINT_REQUEST
            }
            checkpoint_requests = sorted(
                task.request_index
                for task in tasks
                if task.node_id in checkpoint_ids and task.request_index is not None
            )
            gated = any(
                task.arm_id == arm_id and (task.request_index is not None or task.stage == "POST_SCREEN")
                for task in tasks
            )
            if gated and checkpoint_requests != list(range(1, PILOT_CHECKPOINT_REQUEST + 1)):
                raise CampaignSchedulerError("arm must bind one completed evaluator to each frozen checkpoint request")
            for task in tasks:
                if task.arm_id != arm_id or task.stage != "POST_SCREEN":
                    continue
                ancestors = set(task.dependencies)
                frontier = list(task.dependencies)
                while frontier:
                    dependency = frontier.pop()
                    for ancestor in by_node[dependency].dependencies:
                        if ancestor not in ancestors:
                            ancestors.add(ancestor)
                            frontier.append(ancestor)
                if not checkpoint_ids <= ancestors:
                    raise CampaignSchedulerError("post-screen task bypasses the completed checkpoint gate")

    @property
    def arm_ids(self) -> Tuple[str, ...]:
        return tuple(sorted({task.arm_id for task in self.tasks if task.arm_id is not None}))

    def arm_is_gated(self, arm_id: str) -> bool:
        return any(
            task.arm_id == arm_id and (task.request_index is not None or task.stage == "POST_SCREEN")
            for task in self.tasks
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "workers": [worker.to_dict() for worker in sorted(self.workers, key=lambda item: item.worker_id)],
            "tasks": [task.to_dict() for task in sorted(self.tasks, key=lambda item: item.node_id)],
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())


@dataclass(frozen=True)
class Dispatch:
    node_id: str
    worker_id: str
    attempt: int
    attempt_id: str
    resolved_input_digest: str
    execution_idempotency_key: str


@dataclass(frozen=True)
class StallRecovery:
    node_id: str
    previous_worker_id: str
    disposition: str


@dataclass(frozen=True)
class SchedulerControlAction:
    node_id: str
    worker_id: str
    attempt: int
    action: str
    due_ms: int


def _worker_from_row(row: Mapping[str, Any]) -> SchedulerWorker:
    if not isinstance(row.get("public_key_hex"), str) or not _HEX_PUBLIC_KEY.fullmatch(row["public_key_hex"]):
        raise CampaignSchedulerError("worker public key encoding is invalid")
    worker = SchedulerWorker(
        worker_id=row["worker_id"],
        resource_pool=row["resource_pool"],
        owner_role=row["owner_role"],
        capabilities=tuple(row["capabilities"]),
        public_key=bytes.fromhex(row["public_key_hex"]),
    )
    if row["signing_key_id"] != worker.to_dict()["signing_key_id"]:
        raise CampaignSchedulerError("worker signing key ID is not bound to its public key")
    return worker


def _task_from_node(row: Mapping[str, Any]) -> SchedulerTask:
    task = SchedulerTask(
        node_id=row["node_id"],
        kind=row["kind"],
        resource_pool=row["resource_pool"],
        owner_role=row["owner_role"],
        input_digest=row["input_digest"],
        stage=row["stage"],
        dependencies=tuple(row["dependencies"]),
        arm_id=row["arm_id"],
        request_index=row["request_index"],
        promotion_key=row["promotion_key"],
        promotion_authority_node_id=row["promotion_authority_node_id"],
        manifest_ordinal=row["manifest_ordinal"],
        counterbalance_rank=row["counterbalance_rank"],
    )
    if row["node_priority_tuple"] != list(task.node_priority_tuple):
        raise CampaignSchedulerError("scheduler node priority differs from its sealed task identity")
    return task


def _plan_from_state(state: Mapping[str, Any]) -> SchedulerPlan:
    workers = tuple(_worker_from_row(row) for row in state["workers"])
    tasks = tuple(_task_from_node(row) for row in state["nodes"])
    return SchedulerPlan(state["campaign_id"], workers, tasks)


def _initial_state(plan: SchedulerPlan, now_ms: int, *, effect_fence_instance_id: str) -> Dict[str, Any]:
    _require_nonnegative_int(now_ms, "scheduler start time")
    _require_digest(effect_fence_instance_id, "external effect fence instance ID")
    workers = []
    for worker in sorted(plan.workers, key=lambda item: item.worker_id):
        workers.append(
            {
                **worker.to_dict(),
                "busy_ms": 0,
                "idle_ms": 0,
                "avoidable_idle_ms": 0,
                "avoidable_idle_streak_ms": 0,
                "max_avoidable_idle_streak_ms": 0,
            }
        )
    arms = [
        {
            "arm_id": arm_id,
            "status": "SCREENING" if plan.arm_is_gated(arm_id) else "UNGATED",
            "checkpoint_complete": False,
            "checkpoint_terminal_requests": [],
            "checkpoint_completed_requests": [],
            "checkpoint_semantic_passes": 0,
        }
        for arm_id in plan.arm_ids
    ]
    nodes = []
    for task in sorted(plan.tasks, key=lambda item: item.node_id):
        nodes.append(
            {
                **task.to_dict(),
                "status": "PENDING",
                "assigned_worker_id": None,
                "attempt_count": 0,
                "failed_worker_ids": [],
                "started_ms": None,
                "last_progress_ms": None,
                "finished_ms": None,
                "resolved_input_digest": None,
                "progress_digest": None,
                "progress_evidence_digests": [],
                "output_digest": None,
                "semantic_passed": None,
                "terminal_reason": None,
                "attempt_receipts": [],
                "positive_promotion_authority": None,
            }
        )
    return {
        "schema_version": SCHEDULER_STATE_SCHEMA,
        "state_authority": STATE_AUTHORITY,
        "campaign_id": plan.campaign_id,
        "plan_digest": plan.digest,
        "effect_fence_instance_id": effect_fence_instance_id,
        "sequence": 0,
        "created_ms": now_ms,
        "updated_ms": now_ms,
        "config": {
            "progress_checkpoint_ms": PROGRESS_CHECKPOINT_MS,
            "stall_timeout_ms": STALL_TIMEOUT_MS,
            "avoidable_idle_defect_ms": AVOIDABLE_IDLE_DEFECT_MS,
            "pilot_checkpoint_request": PILOT_CHECKPOINT_REQUEST,
            "max_stall_retries": MAX_STALL_RETRIES,
        },
        "workers": workers,
        "arms": arms,
        "nodes": nodes,
    }


def _nodes_by_id(state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {node["node_id"]: node for node in state["nodes"]}


def _workers_by_id(state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {worker["worker_id"]: worker for worker in state["workers"]}


def _arms_by_id(state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {arm["arm_id"]: arm for arm in state["arms"]}


def _attempt_control_bindings(
    state: Mapping[str, Any],
    node: Mapping[str, Any],
    worker: Mapping[str, Any],
    attempt_number: int,
    *,
    prior_attempt: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    prior = prior_attempt
    if prior is None and attempt_number > 1 and node["attempt_receipts"]:
        prior = node["attempt_receipts"][-1]
    predecessor_receipt_hash = (
        receipt_hash(prior["receipts"][-1]) if prior is not None and prior["receipts"] else GENESIS_HASH
    )
    checkpoint = _arms_by_id(state).get(node["arm_id"])
    checkpoint_digest = digest_for(checkpoint if checkpoint is not None else {"arm": None, "status": "UNGATED"})
    worker_identity = {key: worker[key] for key in sorted(_WORKER_SPEC_FIELDS)}
    frozen_task = {key: node[key] for key in sorted(_TASK_SPEC_FIELDS)}
    base = {
        "campaign_id": state["campaign_id"],
        "plan_digest": state["plan_digest"],
        "node_id": node["node_id"],
        "attempt": attempt_number,
        "worker": worker_identity,
        "task": frozen_task,
    }
    lease_id = digest_for({**base, "authority": "BOUNDED_EXECUTION_LEASE"})
    execution_idempotency_key = digest_for(
        {
            "campaign_id": state["campaign_id"],
            "plan_digest": state["plan_digest"],
            "effect_fence_instance_id": state["effect_fence_instance_id"],
            "node_id": node["node_id"],
            "attempt": attempt_number,
            "worker_id": worker["worker_id"],
            "authority": "EXTERNAL_EXECUTION_IDEMPOTENCY",
        }
    )
    return {
        "checkpoint_digest": checkpoint_digest,
        "budget_limit": MAX_STALL_RETRIES + 1,
        "budget_used": attempt_number,
        "predecessor_attempt_id": prior["attempt_id"] if prior is not None else GENESIS_HASH,
        "predecessor_receipt_hash": predecessor_receipt_hash,
        "isolation_digest": digest_for({**base, "isolation": "LOGICAL_PROCESS_LOCAL_ONLY"}),
        "lease_id": lease_id,
        "release_digest": digest_for({"lease_id": lease_id, "release": "REQUIRED_BEFORE_REASSIGNMENT"}),
        "capability_digest": digest_for(worker_identity),
        "model_digest": digest_for({"input_digest": node["input_digest"], "kind": node["kind"]}),
        "protocol_digest": digest_for({"plan_digest": state["plan_digest"], "task": frozen_task}),
        "evaluator_digest": digest_for(
            {"authority": worker["signing_key_id"] if node["kind"] in {"EVALUATION", "VALIDATION"} else None}
        ),
        "sandbox_digest": digest_for({"resource_pool": node["resource_pool"], "isolation": "UNPROVEN"}),
        "manifest_digest": digest_for({"plan_digest": state["plan_digest"], "node": frozen_task}),
        "monotonic_counter": attempt_number,
        "execution_idempotency_key": execution_idempotency_key,
    }


def _qualified(worker: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    return (
        worker["resource_pool"] == node["resource_pool"]
        and worker["owner_role"] == node["owner_role"]
        and node["kind"] in worker["capabilities"]
    )


def _compatible(worker: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    return _qualified(worker, node) and worker["worker_id"] not in node["failed_worker_ids"]


def _resolved_input_digest(state: Mapping[str, Any], node: Mapping[str, Any]) -> str:
    if not node["dependencies"]:
        return node["input_digest"]
    nodes = _nodes_by_id(state)
    dependencies = []
    for dependency_id in node["dependencies"]:
        dependency = nodes[dependency_id]
        dependencies.append(
            {
                "node_id": dependency_id,
                "status": dependency["status"],
                "output_digest": dependency["output_digest"],
                "terminal_reason": dependency["terminal_reason"],
            }
        )
    return digest_for(
        {
            "declared_input_digest": node["input_digest"],
            "dependency_evidence": dependencies,
        }
    )


def _receipt_actor(
    state: Mapping[str, Any],
    node: Mapping[str, Any],
    attempt: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    workers = _workers_by_id(state)
    if receipt.get("effect_kind") == "SCHEDULER_CANCELLATION":
        matches = [
            worker
            for worker in state["workers"]
            if worker["signing_key_id"] == receipt.get("signing_key_id")
            and worker["owner_role"] == "CONTROLLER"
            and "CANCEL" in worker["capabilities"]
        ]
        if len(matches) != 1:
            raise CampaignSchedulerError("cancellation receipt lacks one controller capability authority")
        return matches[0]
    actor = workers.get(attempt["worker_id"])
    if actor is None or not _qualified(actor, node):
        raise CampaignSchedulerError("attempt worker no longer matches task owner and capability")
    return actor


def _external_observer(
    state: Mapping[str, Any], attempt: Mapping[str, Any], receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    matches = [
        worker
        for worker in state["workers"]
        if worker["signing_key_id"] == receipt.get("signing_key_id") and "OBSERVE" in worker["capabilities"]
    ]
    if len(matches) != 1 or matches[0]["worker_id"] == attempt["worker_id"]:
        raise CampaignSchedulerError("external observation lacks one independent observer authority")
    return matches[0]


def _verify_progress_observation(
    state: Mapping[str, Any],
    node: Mapping[str, Any],
    attempt: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> Dict[str, Any]:
    value = dict(_require_closed(receipt, _EXTERNAL_OBSERVATION_RECEIPT_FIELDS, "external progress observation"))
    actor = _external_observer(state, attempt, value)
    observations = attempt["progress_observations"]
    expected_sequence = len(observations) + 1
    previous_receipt = receipt_hash(observations[-1]) if observations else GENESIS_HASH
    previous_artifact = observations[-1]["output_digest"] if observations else attempt["resolved_input_digest"]
    progress_kind = PROGRESS_KINDS[node["kind"]]
    expected_policy = digest_for(
        {
            "attempt_id": attempt["attempt_id"],
            "progress_kind": progress_kind,
            "external_counter": expected_sequence,
            "previous_artifact_digest": previous_artifact,
        }
    )
    expected_action = digest_for(
        {
            "attempt_id": attempt["attempt_id"],
            "progress_kind": progress_kind,
            "external_counter": expected_sequence,
            "previous_artifact_digest": previous_artifact,
            "artifact_digest": value["output_digest"],
            "manifest_digest": attempt["manifest_digest"],
        }
    )
    bindings = (
        value["receipt_type"] == "EFFECT",
        value["decision"] == "ALLOW",
        value["effect_kind"] == "EXTERNAL_PROGRESS_OBSERVATION",
        value["campaign_id"] == state["campaign_id"],
        value["run_id"] == state["campaign_id"],
        value["task_id"] == node["node_id"],
        value["request_id"] == attempt["attempt_id"],
        value["candidate_id"] == attempt["attempt_id"],
        value["input_digest"] == previous_artifact,
        value["output_digest"] != previous_artifact,
        value["resource_bucket"] == node["resource_pool"],
        value["protocol_digest"] == attempt["protocol_digest"],
        value["policy_digest"] == expected_policy,
        value["arm_policy_digest"] == attempt["checkpoint_digest"],
        value["evaluator_digest"] == digest_for({"observer_signing_key_id": actor["signing_key_id"]}),
        value["candidate_artifact_digest"] == value["output_digest"],
        value["environment_diff_digest"] == attempt["sandbox_digest"],
        value["normalized_action_hash"] == expected_action,
        value["public_dependency_set_digest"] == attempt["manifest_digest"],
        value["sandbox_id"] == attempt["lease_id"],
        value["exit_status_class"] == "NOT_APPLICABLE",
        value["idempotency_key"]
        == digest_for(
            {
                "attempt_id": attempt["attempt_id"],
                "external_counter": expected_sequence,
                "effect_kind": "EXTERNAL_PROGRESS_OBSERVATION",
            }
        ),
    )
    if not all(bindings):
        raise CampaignSchedulerError("external progress observation is not bound to advancing artifact evidence")
    _require_digest(value["output_digest"], "external progress artifact digest")
    try:
        verify_receipt(
            value,
            bytes.fromhex(actor["public_key_hex"]),
            expected_key_id=actor["signing_key_id"],
            expected_sequence=expected_sequence,
            expected_previous_hash=previous_receipt,
        )
    except (OptionalDependencyError, ReceiptVerificationError, ValueError) as exc:
        raise CampaignSchedulerError("external progress observation signature or counter chain is invalid") from exc
    return value


def _verify_termination_observation(
    state: Mapping[str, Any],
    node: Mapping[str, Any],
    attempt: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> Dict[str, Any]:
    value = dict(_require_closed(receipt, _EXTERNAL_OBSERVATION_RECEIPT_FIELDS, "external termination observation"))
    actor = _external_observer(state, attempt, value)
    expected_action = digest_for(
        {
            "attempt_id": attempt["attempt_id"],
            "observed_execution_state": "TERMINATED",
            "termination_digest": value["output_digest"],
            "lease_id": attempt["lease_id"],
            "required_release_digest": attempt["release_digest"],
            "lease_revocation_digest": value["arm_policy_digest"],
            "fencing_token_digest": value["policy_digest"],
            "runtime_checkpoint_digest": value["candidate_artifact_digest"],
            "resource_teardown_digest": value["environment_diff_digest"],
            "exit_status_class": value["exit_status_class"],
        }
    )
    externally_supplied = {
        value["output_digest"],
        value["arm_policy_digest"],
        value["policy_digest"],
        value["candidate_artifact_digest"],
        value["environment_diff_digest"],
    }
    scheduler_placeholders = {
        attempt["checkpoint_digest"],
        attempt["release_digest"],
        attempt["sandbox_digest"],
        attempt["manifest_digest"],
        attempt["isolation_digest"],
        attempt["capability_digest"],
        attempt["model_digest"],
        attempt["protocol_digest"],
        attempt["lease_id"],
    }
    bindings = (
        value["receipt_type"] == "EFFECT",
        value["decision"] == "ALLOW",
        value["effect_kind"] == "EXTERNAL_TERMINATION_OBSERVATION",
        value["campaign_id"] == state["campaign_id"],
        value["run_id"] == state["campaign_id"],
        value["task_id"] == node["node_id"],
        value["request_id"] == attempt["attempt_id"],
        value["candidate_id"] == attempt["attempt_id"],
        value["input_digest"] == attempt["lease_id"],
        value["resource_bucket"] == node["resource_pool"],
        value["protocol_digest"] == attempt["protocol_digest"],
        value["evaluator_digest"] == digest_for({"observer_signing_key_id": actor["signing_key_id"]}),
        value["normalized_action_hash"] == expected_action,
        value["public_dependency_set_digest"] == attempt["manifest_digest"],
        value["sandbox_id"] == attempt["lease_id"],
        value["exit_status_class"] in _TERMINATION_EXIT_CLASSES,
        len(externally_supplied) == 5,
        not externally_supplied & scheduler_placeholders,
        value["idempotency_key"]
        == digest_for(
            {
                "attempt_id": attempt["attempt_id"],
                "effect_kind": "EXTERNAL_TERMINATION_OBSERVATION",
                "lease_id": attempt["lease_id"],
            }
        ),
    )
    if not all(bindings):
        raise CampaignSchedulerError(
            "external termination observation lacks termination, revocation, fencing, checkpoint, or teardown evidence"
        )
    for field in (
        "output_digest",
        "arm_policy_digest",
        "policy_digest",
        "candidate_artifact_digest",
        "environment_diff_digest",
    ):
        _require_digest(value[field], "external termination {}".format(field))
    try:
        verify_receipt(
            value,
            bytes.fromhex(actor["public_key_hex"]),
            expected_key_id=actor["signing_key_id"],
            expected_sequence=1,
            expected_previous_hash=GENESIS_HASH,
        )
    except (OptionalDependencyError, ReceiptVerificationError, ValueError) as exc:
        raise CampaignSchedulerError("external termination observation signature is invalid") from exc
    return value


def _verify_activity_receipt(
    state: Mapping[str, Any],
    node: Mapping[str, Any],
    attempt: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_effect: Optional[str] = None,
    external_observation: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    value = dict(_require_closed(receipt, _SIGNED_ACTIVITY_RECEIPT_FIELDS, "scheduler activity receipt"))
    effect_kind = value["effect_kind"]
    allowed_effects = {
        "SCHEDULER_PROGRESS",
        "SCHEDULER_COMPLETION",
        "SCHEDULER_FAILURE",
        "SCHEDULER_CANCELLATION",
        "SCHEDULER_VERDICT",
    }
    if effect_kind not in allowed_effects or (expected_effect is not None and effect_kind != expected_effect):
        raise CampaignSchedulerError("scheduler activity receipt has the wrong effect authority")
    expected_sequence = len(attempt["receipts"]) + 1
    previous = receipt_hash(attempt["receipts"][-1]) if attempt["receipts"] else GENESIS_HASH
    actor = _receipt_actor(state, node, attempt, value)
    if (
        effect_kind == "SCHEDULER_CANCELLATION"
        and external_observation is not None
        and actor["signing_key_id"] == external_observation.get("signing_key_id")
    ):
        raise CampaignSchedulerError("cancellation authority and external termination observer must be independent")
    expected_type = "VERDICT" if effect_kind == "SCHEDULER_VERDICT" else "EFFECT"
    expected_decisions = {
        "SCHEDULER_PROGRESS": {"ALLOW"},
        "SCHEDULER_COMPLETION": {"ALLOW"},
        "SCHEDULER_FAILURE": {"ERROR"},
        "SCHEDULER_CANCELLATION": {"ALLOW"},
        "SCHEDULER_VERDICT": {"PASS", "FAIL"},
    }
    expected_progress_kind = PROGRESS_KINDS[node["kind"]] if effect_kind == "SCHEDULER_PROGRESS" else "NONE"
    if effect_kind in {"SCHEDULER_PROGRESS", "SCHEDULER_CANCELLATION"}:
        if external_observation is None:
            raise CampaignSchedulerError("scheduler progress and cancellation require external observation evidence")
        external_observation_digest = receipt_hash(external_observation)
    elif external_observation is not None:
        raise CampaignSchedulerError("external observation was supplied for an unrelated scheduler effect")
    else:
        external_observation_digest = None
    expected_policy_digest = digest_for(
        {
            field: attempt[field]
            for field in (
                "budget_limit",
                "budget_used",
                "predecessor_attempt_id",
                "predecessor_receipt_hash",
                "lease_id",
                "release_digest",
                "model_digest",
                "monotonic_counter",
                "execution_idempotency_key",
            )
        }
    )
    expected_environment_digest = digest_for(
        {field: attempt[field] for field in ("isolation_digest", "capability_digest", "sandbox_digest")}
    )
    expected_terminal_binding = digest_for(
        {
            "attempt_id": attempt["attempt_id"],
            "effect_kind": effect_kind,
            "decision": value["decision"],
            "output_digest": value["output_digest"],
            "release_digest": attempt["release_digest"],
        }
    )
    if effect_kind == "SCHEDULER_PROGRESS":
        expected_output_digest = external_observation["output_digest"]
    elif effect_kind == "SCHEDULER_CANCELLATION":
        expected_output_digest = external_observation_digest
    else:
        expected_output_digest = value["output_digest"]
    expected_action_hash = digest_for(
        {
            "progress_kind": expected_progress_kind,
            "terminal_binding": expected_terminal_binding,
            "external_observation_digest": external_observation_digest,
        }
    )
    bindings = (
        value["receipt_type"] == expected_type,
        value["decision"] in expected_decisions[effect_kind],
        value["campaign_id"] == state["campaign_id"],
        value["run_id"] == state["campaign_id"],
        value["task_id"] == node["node_id"],
        value["request_id"] == attempt["attempt_id"],
        value["candidate_id"] == attempt["attempt_id"],
        value["input_digest"] == attempt["resolved_input_digest"],
        value["resource_bucket"] == node["resource_pool"],
        value["protocol_digest"] == attempt["protocol_digest"],
        value["policy_digest"] == expected_policy_digest,
        value["arm_policy_digest"] == attempt["checkpoint_digest"],
        value["evaluator_digest"] == attempt["evaluator_digest"],
        value["candidate_artifact_digest"] == value["output_digest"],
        value["output_digest"] == expected_output_digest,
        value["environment_diff_digest"] == expected_environment_digest,
        value["normalized_action_hash"] == expected_action_hash,
        value["public_dependency_set_digest"] == attempt["manifest_digest"],
        value["sandbox_id"] == attempt["lease_id"],
        value["idempotency_key"]
        == digest_for(
            {
                "attempt_id": attempt["attempt_id"],
                "sequence": expected_sequence,
                "effect_kind": effect_kind,
                "external_observation_digest": external_observation_digest,
            }
        ),
    )
    if not all(bindings):
        raise CampaignSchedulerError("scheduler activity receipt is not bound to this exact attempt")
    _require_digest(value["output_digest"], "scheduler receipt output digest")
    try:
        verified_digest = verify_receipt(
            value,
            bytes.fromhex(actor["public_key_hex"]),
            expected_key_id=actor["signing_key_id"],
            expected_sequence=expected_sequence,
            expected_previous_hash=previous,
        )
    except (OptionalDependencyError, ReceiptVerificationError, ValueError) as exc:
        raise CampaignSchedulerError("scheduler activity receipt signature or chain is invalid") from exc
    return value, verified_digest


def _validate_attempt_receipts(state: Mapping[str, Any], node: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    receipts = attempt["receipts"]
    if not isinstance(receipts, list):
        raise CampaignSchedulerError("attempt receipts must be a list")
    observations = attempt["progress_observations"]
    if not isinstance(observations, list):
        raise CampaignSchedulerError("attempt progress observations must be a list")
    termination = attempt["termination_observation"]
    if termination is not None and not isinstance(termination, Mapping):
        raise CampaignSchedulerError("attempt termination observation must be an object")
    replay = {**attempt, "receipts": [], "progress_observations": [], "termination_observation": None}
    terminal_effect = None
    progress_index = 0
    for index, receipt in enumerate(receipts):
        effect = receipt.get("effect_kind") if isinstance(receipt, Mapping) else None
        external_observation = None
        if effect == "SCHEDULER_PROGRESS":
            if progress_index >= len(observations):
                raise CampaignSchedulerError("progress receipt lacks its external observation")
            external_observation = _verify_progress_observation(state, node, replay, observations[progress_index])
        elif effect == "SCHEDULER_CANCELLATION":
            if termination is None:
                raise CampaignSchedulerError("cancellation receipt lacks external termination observation")
            external_observation = _verify_termination_observation(state, node, replay, termination)
        verified, _ = _verify_activity_receipt(
            state,
            node,
            replay,
            receipt,
            external_observation=external_observation,
        )
        effect = verified["effect_kind"]
        if terminal_effect is not None or (effect != "SCHEDULER_PROGRESS" and index != len(receipts) - 1):
            raise CampaignSchedulerError("attempt receipt chain continued after terminal evidence")
        if effect != "SCHEDULER_PROGRESS":
            terminal_effect = effect
        replay["receipts"].append(verified)
        if effect == "SCHEDULER_PROGRESS":
            replay["progress_observations"].append(external_observation)
            progress_index += 1
        elif effect == "SCHEDULER_CANCELLATION":
            replay["termination_observation"] = external_observation
    if progress_index != len(observations):
        raise CampaignSchedulerError("external progress observations outnumber signed progress receipts")
    expected_terminal = {
        "RUNNING": None,
        "COMPLETED": ("SCHEDULER_VERDICT" if node["kind"] in {"EVALUATION", "VALIDATION"} else "SCHEDULER_COMPLETION"),
        "ACTIVITY_FAILED": "SCHEDULER_FAILURE",
        "STALL_RETRY": "SCHEDULER_CANCELLATION",
        "STALL_FAILED": "SCHEDULER_CANCELLATION",
    }[attempt["disposition"]]
    if terminal_effect != expected_terminal:
        raise CampaignSchedulerError("attempt disposition differs from signed terminal evidence")
    if (termination is not None) != (expected_terminal == "SCHEDULER_CANCELLATION"):
        raise CampaignSchedulerError("termination observation differs from attempt disposition")


def _checkpoint_projection(state: Mapping[str, Any], arm_id: str) -> Dict[str, Any]:
    checkpoint = [
        node
        for node in state["nodes"]
        if node["arm_id"] == arm_id
        and node["kind"] == "EVALUATION"
        and node["request_index"] is not None
        and node["request_index"] <= PILOT_CHECKPOINT_REQUEST
    ]
    terminal = sorted(node["request_index"] for node in checkpoint if node["status"] in TERMINAL_STATUSES)
    completed = sorted(node["request_index"] for node in checkpoint if node["status"] == "COMPLETED")
    passes = sum(1 for node in checkpoint if node["status"] == "COMPLETED" and node["semantic_passed"] is True)
    incomplete = any(node["status"] in {"FAILED", "CANCELLED"} for node in checkpoint)
    all_completed = completed == list(range(1, PILOT_CHECKPOINT_REQUEST + 1))
    if incomplete:
        status = "INCOMPLETE_FAIL_CLOSED"
    elif all_completed:
        status = "CONTINUING" if passes else "PAUSED_ZERO_SIGNAL"
    else:
        status = "SCREENING"
    return {
        "status": status,
        "checkpoint_complete": all_completed,
        "checkpoint_terminal_requests": terminal,
        "checkpoint_completed_requests": completed,
        "checkpoint_semantic_passes": passes,
    }


def _validate_state(state: Any) -> Dict[str, Any]:
    state = dict(_require_closed(state, _STATE_FIELDS, "scheduler state"))
    if state["schema_version"] != SCHEDULER_STATE_SCHEMA or state["state_authority"] != STATE_AUTHORITY:
        raise CampaignSchedulerError("scheduler state identity or authority is invalid")
    _require_internal_id(state["campaign_id"], "scheduler campaign ID")
    _require_digest(state["plan_digest"], "scheduler plan digest")
    _require_digest(state["effect_fence_instance_id"], "external effect fence instance ID")
    _require_nonnegative_int(state["sequence"], "scheduler state sequence")
    created_ms = _require_nonnegative_int(state["created_ms"], "scheduler creation time")
    updated_ms = _require_nonnegative_int(state["updated_ms"], "scheduler update time")
    if updated_ms < created_ms:
        raise CampaignSchedulerError("scheduler update time precedes creation")
    expected_config = {
        "progress_checkpoint_ms": PROGRESS_CHECKPOINT_MS,
        "stall_timeout_ms": STALL_TIMEOUT_MS,
        "avoidable_idle_defect_ms": AVOIDABLE_IDLE_DEFECT_MS,
        "pilot_checkpoint_request": PILOT_CHECKPOINT_REQUEST,
        "max_stall_retries": MAX_STALL_RETRIES,
    }
    if dict(_require_closed(state["config"], _CONFIG_FIELDS, "scheduler config")) != expected_config:
        raise CampaignSchedulerError("scheduler config differs from frozen control values")
    if not isinstance(state["workers"], list) or not state["workers"]:
        raise CampaignSchedulerError("scheduler state requires workers")
    worker_ids = []
    for row in state["workers"]:
        worker = dict(_require_closed(row, _WORKER_STATE_FIELDS, "scheduler worker state"))
        _worker_from_row(worker)
        worker_ids.append(worker["worker_id"])
        for field in (
            "busy_ms",
            "idle_ms",
            "avoidable_idle_ms",
            "avoidable_idle_streak_ms",
            "max_avoidable_idle_streak_ms",
        ):
            _require_nonnegative_int(worker[field], "worker {}".format(field))
        if worker["busy_ms"] + worker["idle_ms"] != updated_ms - created_ms:
            raise CampaignSchedulerError("worker time does not reconcile to scheduler window")
        if worker["avoidable_idle_ms"] > worker["idle_ms"]:
            raise CampaignSchedulerError("avoidable idle exceeds worker idle")
        if worker["max_avoidable_idle_streak_ms"] > worker["avoidable_idle_ms"]:
            raise CampaignSchedulerError("avoidable-idle maximum exceeds accumulated time")
        if worker["avoidable_idle_streak_ms"] > worker["max_avoidable_idle_streak_ms"]:
            raise CampaignSchedulerError("avoidable-idle streak exceeds its maximum")
    if worker_ids != sorted(worker_ids) or len(worker_ids) != len(set(worker_ids)):
        raise CampaignSchedulerError("scheduler workers are unordered or ambiguous")
    workers = _workers_by_id(state)
    if not isinstance(state["nodes"], list) or not state["nodes"]:
        raise CampaignSchedulerError("scheduler state requires nodes")
    node_ids = []
    running_workers = []
    for raw_node in state["nodes"]:
        node = dict(_require_closed(raw_node, _NODE_STATE_FIELDS, "scheduler node state"))
        task = _task_from_node(node)
        node_ids.append(task.node_id)
        if node["status"] not in TASK_STATUSES:
            raise CampaignSchedulerError("scheduler node status is invalid")
        attempts = node["attempt_receipts"]
        if not isinstance(attempts, list) or len(attempts) != node["attempt_count"]:
            raise CampaignSchedulerError("attempt journal differs from attempt count")
        if node["attempt_count"] > MAX_STALL_RETRIES + 1:
            raise CampaignSchedulerError("node attempts exceed retry budget")
        expected_progress = []
        stalled_workers = []
        for index, raw_attempt in enumerate(attempts, start=1):
            attempt = dict(_require_closed(raw_attempt, _ATTEMPT_FIELDS, "scheduler attempt receipt"))
            if attempt["attempt"] != index or attempt["worker_id"] not in workers:
                raise CampaignSchedulerError("attempt identity or worker is invalid")
            _require_digest(attempt["attempt_id"], "scheduler attempt ID")
            _require_digest(attempt["resolved_input_digest"], "attempt resolved input digest")
            expected_attempt_id = digest_for(
                {
                    "campaign_id": state["campaign_id"],
                    "node_id": node["node_id"],
                    "attempt": index,
                    "worker_id": attempt["worker_id"],
                    "resolved_input_digest": attempt["resolved_input_digest"],
                }
            )
            if attempt["attempt_id"] != expected_attempt_id:
                raise CampaignSchedulerError("attempt ID is not bound to its dispatch")
            expected_control = _attempt_control_bindings(
                state,
                node,
                workers[attempt["worker_id"]],
                index,
                prior_attempt=attempts[index - 2] if index > 1 else None,
            )
            _require_digest(attempt["checkpoint_digest"], "attempt checkpoint digest")
            expected_control["checkpoint_digest"] = attempt["checkpoint_digest"]
            if any(attempt[field] != value for field, value in expected_control.items()):
                raise CampaignSchedulerError("attempt control, lineage, or budget binding is invalid")
            for field in ("started_ms", "last_progress_ms"):
                _require_nonnegative_int(attempt[field], "attempt {}".format(field))
            if attempt["finished_ms"] is not None:
                _require_nonnegative_int(attempt["finished_ms"], "attempt finish time")
            if attempt["disposition"] not in ATTEMPT_DISPOSITIONS:
                raise CampaignSchedulerError("attempt disposition is invalid")
            if not attempt["started_ms"] <= attempt["last_progress_ms"] <= updated_ms:
                raise CampaignSchedulerError("attempt time lineage is invalid")
            if attempt["finished_ms"] is not None and not (
                attempt["last_progress_ms"] <= attempt["finished_ms"] <= updated_ms
            ):
                raise CampaignSchedulerError("attempt finish time lineage is invalid")
            if (attempt["disposition"] == "RUNNING") != (attempt["finished_ms"] is None):
                raise CampaignSchedulerError("attempt finish time differs from its disposition")
            _validate_attempt_receipts(state, node, attempt)
            for receipt in attempt["receipts"]:
                if receipt["effect_kind"] == "SCHEDULER_PROGRESS":
                    if receipt["output_digest"] in expected_progress:
                        raise CampaignSchedulerError("progress receipt repeated old evidence")
                    expected_progress.append(receipt["output_digest"])
            if attempt["disposition"] in {"STALL_RETRY", "STALL_FAILED"}:
                stalled_workers.append(attempt["worker_id"])
        if node["failed_worker_ids"] != sorted(stalled_workers):
            raise CampaignSchedulerError("failed-worker exclusions differ from retry receipts")
        if node["progress_evidence_digests"] != expected_progress:
            raise CampaignSchedulerError("node progress history differs from signed receipts")
        expected_progress_digest = expected_progress[-1] if expected_progress else None
        if node["progress_digest"] != expected_progress_digest:
            raise CampaignSchedulerError("node progress head differs from signed receipts")
        assigned = node["assigned_worker_id"]
        if node["status"] == "PENDING":
            if any(
                node[field] is not None
                for field in (
                    "assigned_worker_id",
                    "started_ms",
                    "last_progress_ms",
                    "finished_ms",
                    "resolved_input_digest",
                    "output_digest",
                    "semantic_passed",
                    "terminal_reason",
                )
            ):
                raise CampaignSchedulerError("pending node contains execution evidence")
            if attempts and attempts[-1]["disposition"] != "STALL_RETRY":
                raise CampaignSchedulerError("pending retry lacks a stall-retry receipt")
        elif node["status"] == "RUNNING":
            if not attempts or attempts[-1]["disposition"] != "RUNNING" or assigned != attempts[-1]["worker_id"]:
                raise CampaignSchedulerError("running node lacks one current attempt")
            running_workers.append(assigned)
        elif node["status"] == "COMPLETED":
            if not attempts or attempts[-1]["disposition"] != "COMPLETED" or node["terminal_reason"] != "COMPLETED":
                raise CampaignSchedulerError("completed node lacks signed completion")
            final_receipt = attempts[-1]["receipts"][-1]
            if node["output_digest"] != final_receipt["output_digest"]:
                raise CampaignSchedulerError("completed output differs from signed receipt")
            expected_semantic = (
                final_receipt["decision"] == "PASS" if task.kind in {"EVALUATION", "VALIDATION"} else None
            )
            if node["semantic_passed"] != expected_semantic:
                raise CampaignSchedulerError("semantic result is not derived from signed verdict")
            expected_authority = None
            if task.kind in {"EVALUATION", "VALIDATION"} and expected_semantic is True:
                expected_authority = {
                    "authority_node_id": node["node_id"],
                    "candidate_digest": node["output_digest"],
                    "decision": "PASS",
                    "attempt_id": attempts[-1]["attempt_id"],
                    "receipt_hash": receipt_hash(final_receipt),
                    "chain_head": receipt_hash(final_receipt),
                }
            if node["positive_promotion_authority"] != expected_authority:
                raise CampaignSchedulerError("promotion authority is not the closed signed positive verdict record")
        elif node["status"] == "FAILED":
            if not attempts or attempts[-1]["disposition"] not in {"ACTIVITY_FAILED", "STALL_FAILED"}:
                raise CampaignSchedulerError("failed node lacks signed failure or cancellation")
            if node["output_digest"] is not None or node["semantic_passed"] is not None:
                raise CampaignSchedulerError("failed node carries fabricated output")
            if attempts[-1]["disposition"] == "ACTIVITY_FAILED":
                if node["terminal_reason"] != "ACTIVITY_FAILED":
                    raise CampaignSchedulerError("activity failure reason differs from its signed receipt")
            elif node["terminal_reason"] not in {"STALL_RETRY_EXHAUSTED", "STALL_NO_REASSIGNMENT"}:
                raise CampaignSchedulerError("stall failure reason differs from its cancellation receipt")
        else:
            if assigned is not None or node["output_digest"] is not None or node["semantic_passed"] is not None:
                raise CampaignSchedulerError("cancelled node contains execution evidence")
            if attempts and attempts[-1]["disposition"] != "STALL_RETRY":
                raise CampaignSchedulerError("cancelled retry lacks a closed prior attempt")
            if node["finished_ms"] is None:
                raise CampaignSchedulerError("cancelled node lacks terminal time")
            if node["terminal_reason"] not in {
                "DEPENDENCY_NOT_COMPLETED",
                "ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT",
                "CHECKPOINT_INCOMPLETE_FAIL_CLOSED",
            }:
                raise CampaignSchedulerError("cancelled node terminal reason is invalid")
        if node["status"] != "COMPLETED" and node["positive_promotion_authority"] is not None:
            raise CampaignSchedulerError("non-completed node carries promotion authority")
        if node["status"] in {"RUNNING", "COMPLETED", "FAILED"}:
            if (
                assigned is None
                or assigned not in workers
                or node["resolved_input_digest"] is None
                or node["started_ms"] is None
                or node["last_progress_ms"] is None
                or node["finished_ms"] is None
                and node["status"] != "RUNNING"
            ):
                raise CampaignSchedulerError("executed node evidence is incomplete")
            _require_digest(node["resolved_input_digest"], "resolved task input digest")
            if node["resolved_input_digest"] != attempts[-1]["resolved_input_digest"]:
                raise CampaignSchedulerError("node input differs from its current attempt binding")
            if (
                node["assigned_worker_id"] != attempts[-1]["worker_id"]
                or node["started_ms"] != attempts[-1]["started_ms"]
                or node["last_progress_ms"] != attempts[-1]["last_progress_ms"]
                or node["finished_ms"] != attempts[-1]["finished_ms"]
            ):
                raise CampaignSchedulerError("node execution projection differs from its attempt journal")
            if not _qualified(workers[assigned], node):
                raise CampaignSchedulerError("assigned worker violates owner or capability")
    if node_ids != sorted(node_ids) or len(node_ids) != len(set(node_ids)):
        raise CampaignSchedulerError("scheduler nodes are unordered or ambiguous")
    if len(running_workers) != len(set(running_workers)):
        raise CampaignSchedulerError("one worker is assigned to multiple running nodes")
    for worker_id in running_workers:
        if workers[worker_id]["avoidable_idle_streak_ms"]:
            raise CampaignSchedulerError("running worker retained avoidable idle streak")
    plan = _plan_from_state(state)
    if plan.digest != state["plan_digest"]:
        raise CampaignSchedulerError("scheduler plan differs from persisted digest")
    nodes = _nodes_by_id(state)
    for node in state["nodes"]:
        if node["status"] in {"RUNNING", "COMPLETED"}:
            acceptable = TERMINAL_STATUSES if node["kind"] == "DIAGNOSTIC" else {"COMPLETED"}
            if any(nodes[dependency]["status"] not in acceptable for dependency in node["dependencies"]):
                raise CampaignSchedulerError("executed node has incomplete dependencies")
            if node["resolved_input_digest"] != _resolved_input_digest(state, node):
                raise CampaignSchedulerError("executed node input is not bound to dependency outputs")
    if not isinstance(state["arms"], list):
        raise CampaignSchedulerError("scheduler arms must be a list")
    arm_ids = []
    for raw_arm in state["arms"]:
        arm = dict(_require_closed(raw_arm, _ARM_STATE_FIELDS, "scheduler arm state"))
        arm_id = _require_internal_id(arm["arm_id"], "arm ID")
        arm_ids.append(arm_id)
        expected = (
            _checkpoint_projection(state, arm_id)
            if plan.arm_is_gated(arm_id)
            else {
                "status": "UNGATED",
                "checkpoint_complete": False,
                "checkpoint_terminal_requests": [],
                "checkpoint_completed_requests": [],
                "checkpoint_semantic_passes": 0,
            }
        )
        if {key: arm[key] for key in expected} != expected:
            raise CampaignSchedulerError("arm state differs from frozen checkpoint evidence")
        if arm["status"] in {"PAUSED_ZERO_SIGNAL", "INCOMPLETE_FAIL_CLOSED"}:
            expected_reason = (
                "ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT"
                if arm["status"] == "PAUSED_ZERO_SIGNAL"
                else "CHECKPOINT_INCOMPLETE_FAIL_CLOSED"
            )
            for node in state["nodes"]:
                if node["arm_id"] == arm_id and node["stage"] == "POST_SCREEN":
                    if node["status"] != "CANCELLED" or node["terminal_reason"] != expected_reason:
                        raise CampaignSchedulerError("closed checkpoint retained post-screen work")
    if arm_ids != sorted(arm_ids) or tuple(arm_ids) != plan.arm_ids:
        raise CampaignSchedulerError("scheduler arms are unordered or differ from plan")
    return state


class _SchedulerStateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.journal_path = self.path.with_name(self.path.name + ".transitions")

    def create(self, plan: SchedulerPlan, *, effect_fence_instance_id: str, now_ms: int) -> Dict[str, Any]:
        with _PROCESS_MUTEX:
            with _exclusive_path_lock(self.lock_path):
                if self.path.exists() or self.journal_path.exists():
                    raise CampaignSchedulerError("scheduler state is already initialized")
                state = _initial_state(
                    plan,
                    now_ms,
                    effect_fence_instance_id=effect_fence_instance_id,
                )
                _validate_state(state)
                self._atomic_write(state)
                self._append_journal(state)
                return deepcopy(state)

    def load(self) -> Dict[str, Any]:
        with _PROCESS_MUTEX:
            with _exclusive_path_lock(self.lock_path):
                return deepcopy(self._load_unlocked())

    def transact(self, transition: Callable[[Dict[str, Any]], Any]) -> Any:
        with _PROCESS_MUTEX:
            with _exclusive_path_lock(self.lock_path):
                state = self._load_unlocked()
                before = canonical_bytes(state)
                result = transition(state)
                if canonical_bytes(state) != before:
                    state["sequence"] += 1
                    _validate_state(state)
                    self._atomic_write(state)
                    self._append_journal(state)
                return result

    def _load_unlocked(self) -> Dict[str, Any]:
        if not self.path.exists() or self.path.is_symlink() or not self.path.is_file():
            raise CampaignSchedulerError("scheduler state must be a regular existing file")
        try:
            raw = self.path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise CampaignSchedulerError("scheduler state cannot be decoded") from exc
        state = _validate_state(value)
        if raw != canonical_bytes(state):
            raise CampaignSchedulerError("scheduler state is not canonical JSON")
        journal = self._load_journal()
        latest = journal[-1]
        if latest["sequence"] != state["sequence"] or latest["state_digest"] != digest_for(state):
            raise CampaignSchedulerError("scheduler state does not match the local transition journal head")
        return state

    def _load_journal(self) -> list:
        if not self.journal_path.exists() or self.journal_path.is_symlink() or not self.journal_path.is_file():
            raise CampaignSchedulerError("scheduler transition journal must be a regular existing file")
        try:
            raw_lines = self.journal_path.read_bytes().splitlines(keepends=True)
        except OSError as exc:
            raise CampaignSchedulerError("scheduler transition journal cannot be read") from exc
        if not raw_lines:
            raise CampaignSchedulerError("scheduler transition journal is empty")
        entries = []
        previous = GENESIS_HASH
        for expected_sequence, raw in enumerate(raw_lines):
            if not raw.endswith(b"\n"):
                raise CampaignSchedulerError("scheduler transition journal has a torn tail")
            try:
                entry = json.loads(raw[:-1].decode("utf-8"))
            except (UnicodeError, ValueError) as exc:
                raise CampaignSchedulerError("scheduler transition journal cannot be decoded") from exc
            expected_fields = {"schema_version", "sequence", "state_digest", "previous_entry_digest"}
            if not isinstance(entry, Mapping) or set(entry) != expected_fields:
                raise CampaignSchedulerError("scheduler transition journal entry is not closed")
            if (
                entry["schema_version"] != SCHEDULER_JOURNAL_SCHEMA
                or entry["sequence"] != expected_sequence
                or entry["previous_entry_digest"] != previous
            ):
                raise CampaignSchedulerError("scheduler transition journal chain is invalid")
            _require_digest(entry["state_digest"], "scheduler journal state digest")
            canonical_line = canonical_bytes(entry) + b"\n"
            if raw != canonical_line:
                raise CampaignSchedulerError("scheduler transition journal is not canonical")
            previous = digest_for(entry)
            entries.append(entry)
        return entries

    def _append_journal(self, state: Mapping[str, Any]) -> None:
        if self.journal_path.exists():
            entries = self._load_journal()
            previous = digest_for(entries[-1])
            expected_sequence = entries[-1]["sequence"] + 1
        else:
            previous = GENESIS_HASH
            expected_sequence = 0
        if state["sequence"] != expected_sequence:
            raise CampaignSchedulerError("scheduler transition journal sequence would fork")
        entry = {
            "schema_version": SCHEDULER_JOURNAL_SCHEMA,
            "sequence": expected_sequence,
            "state_digest": digest_for(state),
            "previous_entry_digest": previous,
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if expected_sequence == 0:
            flags |= os.O_EXCL
        descriptor = os.open(self.journal_path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "ab", closefd=True) as handle:
                handle.write(canonical_bytes(entry) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    def _atomic_write(self, state: Mapping[str, Any]) -> None:
        descriptor, name = tempfile.mkstemp(prefix=".campaign-scheduler-", dir=str(self.path.parent))
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_bytes(state))
                handle.flush()
                os.fsync(handle.fileno())
            _durable_replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()


class _ExternalEffectFence:
    """Independent local idempotency journal outside the scheduler rollback pair.

    This is deliberately a separate authority path supplied by the caller.  It
    prevents the scheduler from redispatching an execution after only the state
    and transition journal are coherently restored.  It is not a transparency
    log and does not survive an operator rolling back this authority too.
    """

    _FIELDS = frozenset(
        {
            "schema_version",
            "sequence",
            "campaign_id",
            "plan_digest",
            "effect_fence_instance_id",
            "execution_idempotency_key",
            "binding_digest",
            "previous_entry_digest",
        }
    )

    def __init__(
        self,
        path: Path,
        *,
        campaign_id: str,
        plan_digest: str,
        expected_instance_id: Optional[str] = None,
    ) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.campaign_id = campaign_id
        self.plan_digest = plan_digest
        entries = self._load()
        header = entries[0]
        if header["campaign_id"] != campaign_id or header["plan_digest"] != plan_digest:
            raise CampaignSchedulerError("external effect fence is bound to a different campaign plan")
        self.instance_id = header["effect_fence_instance_id"]
        if expected_instance_id is not None and self.instance_id != expected_instance_id:
            raise CampaignSchedulerError("external effect fence instance differs from the pinned scheduler authority")

    @classmethod
    def create(cls, path: Path, *, campaign_id: str, plan_digest: str) -> "_ExternalEffectFence":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(path.name + ".lock")
        with _PROCESS_MUTEX:
            with _exclusive_path_lock(lock_path):
                if path.exists():
                    raise CampaignSchedulerError("external effect fence is already initialized")
                instance_id = digest_for(
                    {
                        "campaign_id": campaign_id,
                        "plan_digest": plan_digest,
                        "instance_nonce": os.urandom(32).hex(),
                    }
                )
                header = {
                    "schema_version": SCHEDULER_EFFECT_FENCE_SCHEMA,
                    "sequence": 0,
                    "campaign_id": campaign_id,
                    "plan_digest": plan_digest,
                    "effect_fence_instance_id": instance_id,
                    "execution_idempotency_key": GENESIS_HASH,
                    "binding_digest": digest_for(
                        {
                            "campaign_id": campaign_id,
                            "plan_digest": plan_digest,
                            "effect_fence_instance_id": instance_id,
                            "authority": "EXTERNAL_EXECUTION_IDEMPOTENCY",
                        }
                    ),
                    "previous_entry_digest": GENESIS_HASH,
                }
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    handle.write(canonical_bytes(header) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return cls(path, campaign_id=campaign_id, plan_digest=plan_digest)

    def _load(self) -> list:
        if not self.path.exists() or self.path.is_symlink() or not self.path.is_file():
            raise CampaignSchedulerError("external effect fence must be a regular existing file")
        try:
            raw_lines = self.path.read_bytes().splitlines(keepends=True)
        except OSError as exc:
            raise CampaignSchedulerError("external effect fence cannot be read") from exc
        if not raw_lines:
            raise CampaignSchedulerError("external effect fence is empty")
        entries = []
        previous = GENESIS_HASH
        seen = set()
        instance_id = None
        for expected_sequence, raw in enumerate(raw_lines):
            if not raw.endswith(b"\n"):
                raise CampaignSchedulerError("external effect fence has a torn tail")
            try:
                entry = json.loads(raw[:-1].decode("utf-8"))
            except (UnicodeError, ValueError) as exc:
                raise CampaignSchedulerError("external effect fence cannot be decoded") from exc
            if not isinstance(entry, Mapping) or set(entry) != self._FIELDS:
                raise CampaignSchedulerError("external effect fence entry is not closed")
            if expected_sequence == 0:
                instance_id = _require_digest(
                    entry["effect_fence_instance_id"],
                    "external effect fence instance ID",
                )
            if (
                entry["schema_version"] != SCHEDULER_EFFECT_FENCE_SCHEMA
                or entry["sequence"] != expected_sequence
                or entry["previous_entry_digest"] != previous
                or entry["campaign_id"] != self.campaign_id
                or entry["plan_digest"] != self.plan_digest
                or entry["effect_fence_instance_id"] != instance_id
            ):
                raise CampaignSchedulerError("external effect fence chain or identity is invalid")
            _require_digest(entry["execution_idempotency_key"], "external execution idempotency key")
            _require_digest(entry["binding_digest"], "external execution binding digest")
            if expected_sequence and entry["execution_idempotency_key"] in seen:
                raise CampaignSchedulerError("external effect fence contains a duplicate execution key")
            if raw != canonical_bytes(entry) + b"\n":
                raise CampaignSchedulerError("external effect fence is not canonical")
            seen.add(entry["execution_idempotency_key"])
            previous = digest_for(entry)
            entries.append(entry)
        return entries

    def claim_many(self, claims: Mapping[str, str]) -> None:
        if not isinstance(claims, Mapping):
            raise CampaignSchedulerError("external effect fence claims must be a mapping")
        normalized = []
        for execution_key, binding_digest in sorted(claims.items()):
            _require_digest(execution_key, "external execution idempotency key")
            _require_digest(binding_digest, "external execution binding digest")
            normalized.append((execution_key, binding_digest))
        with _PROCESS_MUTEX:
            with _exclusive_path_lock(self.lock_path):
                entries = self._load()
                existing = {entry["execution_idempotency_key"]: entry for entry in entries[1:]}
                for execution_key, binding_digest in normalized:
                    if execution_key in existing:
                        if existing[execution_key]["binding_digest"] != binding_digest:
                            raise CampaignSchedulerError(
                                "external execution idempotency key conflicts with prior binding"
                            )
                        raise CampaignSchedulerError("external execution idempotency fence rejects replayed dispatch")
                if not normalized:
                    return
                previous = digest_for(entries[-1])
                sequence = len(entries)
                descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
                with os.fdopen(descriptor, "ab", closefd=True) as handle:
                    for execution_key, binding_digest in normalized:
                        entry = {
                            "schema_version": SCHEDULER_EFFECT_FENCE_SCHEMA,
                            "sequence": sequence,
                            "campaign_id": self.campaign_id,
                            "plan_digest": self.plan_digest,
                            "effect_fence_instance_id": self.instance_id,
                            "execution_idempotency_key": execution_key,
                            "binding_digest": binding_digest,
                            "previous_entry_digest": previous,
                        }
                        handle.write(canonical_bytes(entry) + b"\n")
                        previous = digest_for(entry)
                        sequence += 1
                    handle.flush()
                    os.fsync(handle.fileno())


def _task_ready(state: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    if node["status"] != "PENDING":
        return False
    nodes = _nodes_by_id(state)
    if node["kind"] == "DIAGNOSTIC":
        if not all(nodes[dependency]["status"] in TERMINAL_STATUSES for dependency in node["dependencies"]):
            return False
    elif not all(nodes[dependency]["status"] == "COMPLETED" for dependency in node["dependencies"]):
        return False
    if node["arm_id"] is not None:
        arm = _arms_by_id(state)[node["arm_id"]]
        if node["stage"] == "POST_SCREEN" and arm["status"] != "CONTINUING":
            return False
        if arm["status"] == "INCOMPLETE_FAIL_CLOSED" and node["stage"] != "DIAGNOSTIC":
            return False
        if arm["status"] == "PAUSED_ZERO_SIGNAL" and node["stage"] != "DIAGNOSTIC":
            return False
    if node["kind"] == "PROMOTION":
        authority = nodes[node["promotion_authority_node_id"]]
        expected = authority["positive_promotion_authority"]
        if (
            expected is None
            or set(expected) != _PROMOTION_AUTHORITY_FIELDS
            or expected["authority_node_id"] != node["promotion_authority_node_id"]
            or expected["candidate_digest"] != node["promotion_key"]
            or expected["decision"] != "PASS"
        ):
            return False
    return True


def _candidate_workers(
    state: Mapping[str, Any], node: Mapping[str, Any], available_worker_ids: Sequence[str]
) -> Tuple[str, ...]:
    workers = _workers_by_id(state)
    return tuple(
        sorted(
            worker_id
            for worker_id in available_worker_ids
            if worker_id in workers and _compatible(workers[worker_id], node)
        )
    )


def _maximum_matching(
    state: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]], available_worker_ids: Sequence[str]
) -> Dict[str, str]:
    """Return the lexicographically minimal maximum matching under sealed task priority."""

    candidates = {node["node_id"]: _candidate_workers(state, node, available_worker_ids) for node in nodes}
    ordered_ids = [node["node_id"] for node in sorted(nodes, key=lambda item: tuple(item["node_priority_tuple"]))]

    def cardinality(node_ids: Sequence[str], worker_ids: frozenset) -> int:
        worker_to_node: Dict[str, str] = {}

        def assign(node_id: str, seen: set) -> bool:
            for worker_id in candidates[node_id]:
                if worker_id not in worker_ids or worker_id in seen:
                    continue
                seen.add(worker_id)
                incumbent = worker_to_node.get(worker_id)
                if incumbent is None or assign(incumbent, seen):
                    worker_to_node[worker_id] = node_id
                    return True
            return False

        for node_id in node_ids:
            assign(node_id, set())
        return len(worker_to_node)

    remaining_workers = frozenset(available_worker_ids)
    needed = cardinality(ordered_ids, remaining_workers)
    result: Dict[str, str] = {}
    for position, node_id in enumerate(ordered_ids):
        tail = ordered_ids[position + 1 :]
        choices = [worker_id for worker_id in candidates[node_id] if worker_id in remaining_workers]
        choices.append(None)
        for worker_id in choices:
            next_workers = remaining_workers if worker_id is None else remaining_workers - {worker_id}
            consumed = 0 if worker_id is None else 1
            if consumed + cardinality(tail, next_workers) < needed:
                continue
            if worker_id is not None:
                result[node_id] = worker_id
                remaining_workers = next_workers
                needed -= 1
            break
    return result


def _avoidable_idle_workers(state: Mapping[str, Any]) -> frozenset:
    running = {node["assigned_worker_id"] for node in state["nodes"] if node["status"] == "RUNNING"}
    available = [worker["worker_id"] for worker in state["workers"] if worker["worker_id"] not in running]
    ready = [node for node in state["nodes"] if _task_ready(state, node)]
    return frozenset(_maximum_matching(state, ready, available).values())


def _accrue_utilization(state: Dict[str, Any], now_ms: int) -> None:
    _require_nonnegative_int(now_ms, "scheduler transition time")
    if now_ms < state["updated_ms"]:
        raise CampaignSchedulerError("scheduler transition time regressed")
    delta = now_ms - state["updated_ms"]
    if not delta:
        return
    running = {node["assigned_worker_id"] for node in state["nodes"] if node["status"] == "RUNNING"}
    avoidable = _avoidable_idle_workers(state)
    for worker in state["workers"]:
        if worker["worker_id"] in running:
            worker["busy_ms"] += delta
            worker["avoidable_idle_streak_ms"] = 0
        else:
            worker["idle_ms"] += delta
            if worker["worker_id"] in avoidable:
                worker["avoidable_idle_ms"] += delta
                worker["avoidable_idle_streak_ms"] += delta
                worker["max_avoidable_idle_streak_ms"] = max(
                    worker["max_avoidable_idle_streak_ms"], worker["avoidable_idle_streak_ms"]
                )
            else:
                worker["avoidable_idle_streak_ms"] = 0
    state["updated_ms"] = now_ms


def _cancel_node(node: Dict[str, Any], *, now_ms: int, reason: str) -> bool:
    if node["status"] != "PENDING":
        return False
    node["status"] = "CANCELLED"
    node["finished_ms"] = now_ms
    node["terminal_reason"] = reason
    return True


def _refresh_checkpoints(state: Dict[str, Any], *, now_ms: int) -> bool:
    changed = False
    plan = _plan_from_state(state)
    for arm in state["arms"]:
        if not plan.arm_is_gated(arm["arm_id"]):
            continue
        projection = _checkpoint_projection(state, arm["arm_id"])
        for key, value in projection.items():
            if arm[key] != value:
                arm[key] = value
                changed = True
        if projection["status"] not in {"PAUSED_ZERO_SIGNAL", "INCOMPLETE_FAIL_CLOSED"}:
            continue
        reason = (
            "ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT"
            if projection["status"] == "PAUSED_ZERO_SIGNAL"
            else "CHECKPOINT_INCOMPLETE_FAIL_CLOSED"
        )
        for node in state["nodes"]:
            if node["arm_id"] != arm["arm_id"] or node["stage"] == "DIAGNOSTIC":
                continue
            if projection["status"] == "PAUSED_ZERO_SIGNAL" and node["stage"] != "POST_SCREEN":
                continue
            if _cancel_node(node, now_ms=now_ms, reason=reason):
                changed = True
            elif node["status"] == "CANCELLED" and node["stage"] == "POST_SCREEN" and node["terminal_reason"] != reason:
                node["terminal_reason"] = reason
                changed = True
    return changed


def _propagate_unreachable(state: Dict[str, Any], *, now_ms: int) -> bool:
    changed = False
    nodes = _nodes_by_id(state)
    for node in state["nodes"]:
        if (
            node["kind"] != "DIAGNOSTIC"
            and node["status"] == "PENDING"
            and any(nodes[dependency]["status"] in {"FAILED", "CANCELLED"} for dependency in node["dependencies"])
        ):
            changed = _cancel_node(node, now_ms=now_ms, reason="DEPENDENCY_NOT_COMPLETED") or changed
    return changed


def _settle_state(state: Dict[str, Any], *, now_ms: int) -> None:
    for _ in range(len(state["nodes"]) + len(state["arms"]) + 2):
        changed = _refresh_checkpoints(state, now_ms=now_ms)
        changed = _propagate_unreachable(state, now_ms=now_ms) or changed
        if not changed:
            return
    raise CampaignSchedulerError("scheduler dependency settlement did not converge")


class CampaignScheduler:
    """Durable scheduler over logical workers and authenticated activity receipts."""

    def __init__(self, state_path: Path, *, effect_fence_path: Path) -> None:
        self._store = _SchedulerStateStore(Path(state_path))
        state = self._store.load()
        forbidden = {
            self._store.path.resolve(),
            self._store.journal_path.resolve(),
            self._store.lock_path.resolve(),
        }
        if Path(effect_fence_path).resolve() in forbidden:
            raise CampaignSchedulerError("external effect fence must not reuse a scheduler state path")
        self._effect_fence = _ExternalEffectFence(
            Path(effect_fence_path),
            campaign_id=state["campaign_id"],
            plan_digest=state["plan_digest"],
            expected_instance_id=state["effect_fence_instance_id"],
        )

    @classmethod
    def initialize(
        cls,
        state_path: Path,
        plan: SchedulerPlan,
        *,
        effect_fence_path: Path,
        now_ms: int = 0,
    ) -> "CampaignScheduler":
        if type(plan) is not SchedulerPlan:
            raise CampaignSchedulerError("scheduler initialization requires an exact validated plan")
        state_store = _SchedulerStateStore(Path(state_path))
        if state_store.path.exists() or state_store.journal_path.exists():
            raise CampaignSchedulerError("scheduler state is already initialized")
        forbidden = {state_store.path.resolve(), state_store.journal_path.resolve(), state_store.lock_path.resolve()}
        if Path(effect_fence_path).resolve() in forbidden:
            raise CampaignSchedulerError("external effect fence must not reuse a scheduler state path")
        effect_fence = _ExternalEffectFence.create(
            Path(effect_fence_path),
            campaign_id=plan.campaign_id,
            plan_digest=plan.digest,
        )
        _SchedulerStateStore(Path(state_path)).create(
            plan,
            effect_fence_instance_id=effect_fence.instance_id,
            now_ms=now_ms,
        )
        return cls(state_path, effect_fence_path=effect_fence_path)

    def snapshot(self) -> Dict[str, Any]:
        return self._store.load()

    def controller_actions(self, *, now_ms: int) -> Tuple[SchedulerControlAction, ...]:
        """Return deterministic controller work; this does not contact or cancel a process."""

        state = self._store.load()
        _require_nonnegative_int(now_ms, "scheduler controller time")
        if now_ms < state["updated_ms"]:
            raise CampaignSchedulerError("scheduler controller time regressed")
        actions = []
        for node in state["nodes"]:
            if node["status"] != "RUNNING":
                continue
            elapsed = now_ms - node["last_progress_ms"]
            if elapsed >= STALL_TIMEOUT_MS:
                action = "CANCEL_STALLED_ATTEMPT"
                due_ms = node["last_progress_ms"] + STALL_TIMEOUT_MS
            elif elapsed >= PROGRESS_CHECKPOINT_MS:
                action = "EMIT_PROGRESS_CHECKPOINT"
                due_ms = node["last_progress_ms"] + PROGRESS_CHECKPOINT_MS
            else:
                continue
            actions.append(
                SchedulerControlAction(
                    node["node_id"],
                    node["assigned_worker_id"],
                    node["attempt_count"],
                    action,
                    due_ms,
                )
            )
        return tuple(sorted(actions, key=lambda item: (item.due_ms, item.node_id)))

    def dispatch_ready(self, *, now_ms: int) -> Tuple[Dispatch, ...]:
        def transition(state: Dict[str, Any]) -> Tuple[Dispatch, ...]:
            _accrue_utilization(state, now_ms)
            _settle_state(state, now_ms=now_ms)
            occupied = {node["assigned_worker_id"] for node in state["nodes"] if node["status"] == "RUNNING"}
            available = [worker["worker_id"] for worker in state["workers"] if worker["worker_id"] not in occupied]
            ready = [node for node in state["nodes"] if _task_ready(state, node)]
            matching = _maximum_matching(state, ready, available)
            workers = _workers_by_id(state)
            prepared = []
            fence_claims = {}
            for node_id in sorted(matching):
                node = _nodes_by_id(state)[node_id]
                worker_id = matching[node_id]
                resolved = _resolved_input_digest(state, node)
                attempt_number = node["attempt_count"] + 1
                attempt_id = digest_for(
                    {
                        "campaign_id": state["campaign_id"],
                        "node_id": node_id,
                        "attempt": attempt_number,
                        "worker_id": worker_id,
                        "resolved_input_digest": resolved,
                    }
                )
                control_bindings = _attempt_control_bindings(state, node, workers[worker_id], attempt_number)
                execution_key = control_bindings["execution_idempotency_key"]
                fence_claims[execution_key] = digest_for(
                    {
                        "attempt_id": attempt_id,
                        "worker_id": worker_id,
                        "resolved_input_digest": resolved,
                        "lease_id": control_bindings["lease_id"],
                        "manifest_digest": control_bindings["manifest_digest"],
                    }
                )
                prepared.append((node_id, worker_id, resolved, attempt_number, attempt_id, control_bindings))
            self._effect_fence.claim_many(fence_claims)
            dispatches = []
            for node_id, worker_id, resolved, attempt_number, attempt_id, control_bindings in prepared:
                node = _nodes_by_id(state)[node_id]
                node["status"] = "RUNNING"
                node["assigned_worker_id"] = worker_id
                node["attempt_count"] = attempt_number
                node["started_ms"] = now_ms
                node["last_progress_ms"] = now_ms
                node["finished_ms"] = None
                node["resolved_input_digest"] = resolved
                node["output_digest"] = None
                node["semantic_passed"] = None
                node["terminal_reason"] = None
                node["attempt_receipts"].append(
                    {
                        "attempt": attempt_number,
                        "attempt_id": attempt_id,
                        "worker_id": worker_id,
                        "resolved_input_digest": resolved,
                        "started_ms": now_ms,
                        "last_progress_ms": now_ms,
                        "finished_ms": None,
                        "disposition": "RUNNING",
                        "receipts": [],
                        "progress_observations": [],
                        "termination_observation": None,
                        **control_bindings,
                    }
                )
                workers[worker_id]["avoidable_idle_streak_ms"] = 0
                dispatches.append(
                    Dispatch(
                        node_id,
                        worker_id,
                        attempt_number,
                        attempt_id,
                        resolved,
                        control_bindings["execution_idempotency_key"],
                    )
                )
            return tuple(dispatches)

        return self._store.transact(transition)

    def record_progress(
        self,
        node_id: str,
        *,
        worker_id: str,
        receipt: Mapping[str, Any],
        observation_receipt: Mapping[str, Any],
        now_ms: int,
    ) -> Dict[str, Any]:
        _require_internal_id(node_id, "scheduler node ID")
        _require_internal_id(worker_id, "worker ID")

        def transition(state: Dict[str, Any]) -> Dict[str, Any]:
            _accrue_utilization(state, now_ms)
            node = _nodes_by_id(state).get(node_id)
            if node is None or node["status"] != "RUNNING" or node["assigned_worker_id"] != worker_id:
                raise CampaignSchedulerError("progress receipt does not match one running assignment")
            attempt = node["attempt_receipts"][-1]
            if attempt["receipts"] and canonical_bytes(attempt["receipts"][-1]) == canonical_bytes(receipt):
                if attempt["receipts"][-1]["effect_kind"] != "SCHEDULER_PROGRESS":
                    raise CampaignSchedulerError("terminal receipt cannot be replayed as progress")
                if not attempt["progress_observations"] or canonical_bytes(
                    attempt["progress_observations"][-1]
                ) != canonical_bytes(observation_receipt):
                    raise CampaignSchedulerError("progress retry conflicts with its external observation")
                return deepcopy(node)
            observation = _verify_progress_observation(state, node, attempt, observation_receipt)
            verified, _ = _verify_activity_receipt(
                state,
                node,
                attempt,
                receipt,
                expected_effect="SCHEDULER_PROGRESS",
                external_observation=observation,
            )
            evidence_digest = verified["output_digest"]
            if evidence_digest in node["progress_evidence_digests"]:
                raise CampaignSchedulerError("progress receipt did not advance stored evidence")
            if len(node["progress_evidence_digests"]) >= MAX_PROGRESS_EVIDENCE_DIGESTS:
                raise CampaignSchedulerError("progress evidence history reached its frozen bound")
            attempt["receipts"].append(verified)
            attempt["progress_observations"].append(observation)
            attempt["last_progress_ms"] = now_ms
            node["progress_evidence_digests"].append(evidence_digest)
            node["progress_digest"] = evidence_digest
            node["last_progress_ms"] = now_ms
            return deepcopy(node)

        return self._store.transact(transition)

    def complete(
        self,
        node_id: str,
        *,
        worker_id: str,
        receipt: Mapping[str, Any],
        now_ms: int,
    ) -> Dict[str, Any]:
        _require_internal_id(node_id, "scheduler node ID")
        _require_internal_id(worker_id, "worker ID")

        def transition(state: Dict[str, Any]) -> Dict[str, Any]:
            _accrue_utilization(state, now_ms)
            node = _nodes_by_id(state).get(node_id)
            if node is None:
                raise CampaignSchedulerError("completion references an unknown node")
            if node["status"] == "COMPLETED":
                final = node["attempt_receipts"][-1]["receipts"][-1]
                if node["assigned_worker_id"] == worker_id and canonical_bytes(final) == canonical_bytes(receipt):
                    return deepcopy(node)
                raise CampaignSchedulerError("completion retry conflicts with durable signed evidence")
            if node["status"] != "RUNNING" or node["assigned_worker_id"] != worker_id:
                raise CampaignSchedulerError("completion does not match one running assignment")
            attempt = node["attempt_receipts"][-1]
            effect = "SCHEDULER_VERDICT" if node["kind"] in {"EVALUATION", "VALIDATION"} else "SCHEDULER_COMPLETION"
            verified, _ = _verify_activity_receipt(state, node, attempt, receipt, expected_effect=effect)
            attempt["receipts"].append(verified)
            attempt["last_progress_ms"] = now_ms
            attempt["finished_ms"] = now_ms
            attempt["disposition"] = "COMPLETED"
            node["status"] = "COMPLETED"
            node["last_progress_ms"] = now_ms
            node["finished_ms"] = now_ms
            node["output_digest"] = verified["output_digest"]
            node["semantic_passed"] = verified["decision"] == "PASS" if effect == "SCHEDULER_VERDICT" else None
            node["positive_promotion_authority"] = (
                {
                    "authority_node_id": node["node_id"],
                    "candidate_digest": verified["output_digest"],
                    "decision": "PASS",
                    "attempt_id": attempt["attempt_id"],
                    "receipt_hash": receipt_hash(verified),
                    "chain_head": receipt_hash(verified),
                }
                if effect == "SCHEDULER_VERDICT" and verified["decision"] == "PASS"
                else None
            )
            node["terminal_reason"] = "COMPLETED"
            _settle_state(state, now_ms=now_ms)
            return deepcopy(node)

        return self._store.transact(transition)

    def fail(
        self,
        node_id: str,
        *,
        worker_id: str,
        receipt: Mapping[str, Any],
        now_ms: int,
    ) -> Dict[str, Any]:
        _require_internal_id(node_id, "scheduler node ID")
        _require_internal_id(worker_id, "worker ID")

        def transition(state: Dict[str, Any]) -> Dict[str, Any]:
            _accrue_utilization(state, now_ms)
            node = _nodes_by_id(state).get(node_id)
            if node is None:
                raise CampaignSchedulerError("failure references an unknown node")
            if node["status"] == "FAILED" and node["terminal_reason"] == "ACTIVITY_FAILED":
                final = node["attempt_receipts"][-1]["receipts"][-1]
                if node["assigned_worker_id"] == worker_id and canonical_bytes(final) == canonical_bytes(receipt):
                    return deepcopy(node)
                raise CampaignSchedulerError("failure retry conflicts with durable signed evidence")
            if node["status"] != "RUNNING" or node["assigned_worker_id"] != worker_id:
                raise CampaignSchedulerError("failure does not match one running assignment")
            attempt = node["attempt_receipts"][-1]
            verified, _ = _verify_activity_receipt(state, node, attempt, receipt, expected_effect="SCHEDULER_FAILURE")
            attempt["receipts"].append(verified)
            attempt["last_progress_ms"] = now_ms
            attempt["finished_ms"] = now_ms
            attempt["disposition"] = "ACTIVITY_FAILED"
            node["status"] = "FAILED"
            node["last_progress_ms"] = now_ms
            node["finished_ms"] = now_ms
            node["terminal_reason"] = "ACTIVITY_FAILED"
            _settle_state(state, now_ms=now_ms)
            return deepcopy(node)

        return self._store.transact(transition)

    def recover_stalls(
        self,
        *,
        now_ms: int,
        cancellation_receipts: Mapping[str, Mapping[str, Any]],
        termination_observations: Mapping[str, Mapping[str, Any]],
    ) -> Tuple[StallRecovery, ...]:
        if not isinstance(cancellation_receipts, Mapping):
            raise CampaignSchedulerError("stall recovery requires a cancellation receipt mapping")
        if not isinstance(termination_observations, Mapping):
            raise CampaignSchedulerError("stall recovery requires a termination observation mapping")

        def transition(state: Dict[str, Any]) -> Tuple[StallRecovery, ...]:
            _accrue_utilization(state, now_ms)
            stalled = [
                node
                for node in state["nodes"]
                if node["status"] == "RUNNING" and now_ms - node["last_progress_ms"] >= STALL_TIMEOUT_MS
            ]
            if set(cancellation_receipts) != {node["node_id"] for node in stalled}:
                raise CampaignSchedulerError("stall recovery requires exact bounded cancellation evidence")
            if set(termination_observations) != {node["node_id"] for node in stalled}:
                raise CampaignSchedulerError("stall recovery requires exact external termination evidence")
            recoveries = []
            worker_ids = [worker["worker_id"] for worker in state["workers"]]
            for node in sorted(stalled, key=lambda item: item["node_id"]):
                attempt = node["attempt_receipts"][-1]
                observation = _verify_termination_observation(
                    state,
                    node,
                    attempt,
                    termination_observations[node["node_id"]],
                )
                verified, _ = _verify_activity_receipt(
                    state,
                    node,
                    attempt,
                    cancellation_receipts[node["node_id"]],
                    expected_effect="SCHEDULER_CANCELLATION",
                    external_observation=observation,
                )
                attempt["receipts"].append(verified)
                attempt["termination_observation"] = observation
                attempt["last_progress_ms"] = now_ms
                attempt["finished_ms"] = now_ms
                previous_worker = node["assigned_worker_id"]
                node["failed_worker_ids"] = sorted(node["failed_worker_ids"] + [previous_worker])
                alternates = _candidate_workers(state, node, worker_ids)
                if node["attempt_count"] <= MAX_STALL_RETRIES and alternates:
                    attempt["disposition"] = "STALL_RETRY"
                    node["status"] = "PENDING"
                    node["assigned_worker_id"] = None
                    node["started_ms"] = None
                    node["last_progress_ms"] = None
                    node["finished_ms"] = None
                    node["resolved_input_digest"] = None
                    recoveries.append(StallRecovery(node["node_id"], previous_worker, "RETRY_PENDING"))
                else:
                    attempt["disposition"] = "STALL_FAILED"
                    node["status"] = "FAILED"
                    node["last_progress_ms"] = now_ms
                    node["finished_ms"] = now_ms
                    node["terminal_reason"] = (
                        "STALL_RETRY_EXHAUSTED"
                        if node["attempt_count"] > MAX_STALL_RETRIES
                        else "STALL_NO_REASSIGNMENT"
                    )
                    recoveries.append(StallRecovery(node["node_id"], previous_worker, "FAILED_CLOSED"))
            _settle_state(state, now_ms=now_ms)
            return tuple(recoveries)

        return self._store.transact(transition)

    def public_report(self, *, now_ms: int, worker_aliases: Mapping[str, str]) -> Dict[str, Any]:
        if not isinstance(worker_aliases, Mapping):
            raise CampaignSchedulerError("public worker aliases must be a mapping")
        state = self._store.load()
        internal_ids = {worker["worker_id"] for worker in state["workers"]}
        if set(worker_aliases) != internal_ids:
            raise CampaignSchedulerError("public aliases must cover the exact private worker inventory")
        aliases = {worker_id: _require_public_alias(worker_aliases[worker_id]) for worker_id in internal_ids}
        if len(set(aliases.values())) != len(aliases):
            raise CampaignSchedulerError("public worker aliases must be unique")

        def accrue(current: Dict[str, Any]) -> None:
            _accrue_utilization(current, now_ms)

        self._store.transact(accrue)
        state = self._store.load()
        workers = []
        for worker in state["workers"]:
            maximum = worker["max_avoidable_idle_streak_ms"]
            workers.append(
                {
                    "alias": aliases[worker["worker_id"]],
                    "resource_pool": worker["resource_pool"],
                    "busy_ms": worker["busy_ms"],
                    "idle_ms": worker["idle_ms"],
                    "avoidable_idle_ms": worker["avoidable_idle_ms"],
                    "legitimately_blocked_ms": worker["idle_ms"] - worker["avoidable_idle_ms"],
                    "max_avoidable_idle_streak_ms": maximum,
                    "avoidable_idle_defect": maximum > AVOIDABLE_IDLE_DEFECT_MS,
                }
            )
        workers.sort(key=lambda item: item["alias"])
        aggregate_maximum = max(worker["max_avoidable_idle_streak_ms"] for worker in workers)
        report = {
            "schema_version": SCHEDULER_PUBLIC_SCHEMA,
            "campaign_digest": digest_for({"campaign_id": state["campaign_id"]}),
            "plan_digest": state["plan_digest"],
            "state_sequence": state["sequence"],
            "window_ms": state["updated_ms"] - state["created_ms"],
            "workers": workers,
            "aggregate": {
                "capacity_ms": (state["updated_ms"] - state["created_ms"]) * len(workers),
                "busy_ms": sum(worker["busy_ms"] for worker in workers),
                "idle_ms": sum(worker["idle_ms"] for worker in workers),
                "avoidable_idle_ms": sum(worker["avoidable_idle_ms"] for worker in workers),
                "legitimately_blocked_ms": sum(worker["legitimately_blocked_ms"] for worker in workers),
                "max_avoidable_idle_streak_ms": aggregate_maximum,
                "avoidable_idle_defect": aggregate_maximum > AVOIDABLE_IDLE_DEFECT_MS,
            },
            "tasks": {
                status: sum(1 for node in state["nodes"] if node["status"] == status)
                for status in sorted(TASK_STATUSES)
            },
            "arms": {
                status: sum(1 for arm in state["arms"] if arm["status"] == status) for status in sorted(ARM_STATUSES)
            },
            "state_authority": STATE_AUTHORITY,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        return validate_public_scheduler_report(report)


def validate_public_scheduler_report(value: Mapping[str, Any]) -> Dict[str, Any]:
    report = dict(_require_closed(value, _PUBLIC_FIELDS, "public scheduler report"))
    if (
        report["schema_version"] != SCHEDULER_PUBLIC_SCHEMA
        or report["state_authority"] != STATE_AUTHORITY
        or report["claim_boundary"] != CLAIM_BOUNDARY
    ):
        raise CampaignSchedulerError("public scheduler identity or claim boundary is invalid")
    _require_digest(report["campaign_digest"], "public campaign digest")
    _require_digest(report["plan_digest"], "public plan digest")
    _require_nonnegative_int(report["state_sequence"], "public scheduler sequence")
    window_ms = _require_nonnegative_int(report["window_ms"], "public scheduler window")
    if not isinstance(report["workers"], list) or not report["workers"]:
        raise CampaignSchedulerError("public scheduler report requires worker aggregates")
    aliases = []
    for raw_worker in report["workers"]:
        worker = _require_closed(raw_worker, _PUBLIC_WORKER_FIELDS, "public scheduler worker")
        aliases.append(_require_public_alias(worker["alias"]))
        if worker["resource_pool"] not in RESOURCE_POOLS:
            raise CampaignSchedulerError("public scheduler worker resource is invalid")
        for field in (
            "busy_ms",
            "idle_ms",
            "avoidable_idle_ms",
            "legitimately_blocked_ms",
            "max_avoidable_idle_streak_ms",
        ):
            _require_nonnegative_int(worker[field], "public worker {}".format(field))
        if not isinstance(worker["avoidable_idle_defect"], bool):
            raise CampaignSchedulerError("public worker defect marker must be boolean")
        if worker["avoidable_idle_ms"] > worker["idle_ms"]:
            raise CampaignSchedulerError("public avoidable idle exceeds total idle")
        if worker["legitimately_blocked_ms"] != worker["idle_ms"] - worker["avoidable_idle_ms"]:
            raise CampaignSchedulerError("public legitimate blocking differs from idle evidence")
        if worker["avoidable_idle_defect"] != (worker["max_avoidable_idle_streak_ms"] > AVOIDABLE_IDLE_DEFECT_MS):
            raise CampaignSchedulerError("public worker defect marker differs from frozen threshold")
    if aliases != sorted(aliases) or len(aliases) != len(set(aliases)):
        raise CampaignSchedulerError("public worker aliases are unordered or ambiguous")
    aggregate = _require_closed(report["aggregate"], _PUBLIC_AGGREGATE_FIELDS, "public scheduler aggregate")
    for field in (
        "capacity_ms",
        "busy_ms",
        "idle_ms",
        "avoidable_idle_ms",
        "legitimately_blocked_ms",
        "max_avoidable_idle_streak_ms",
    ):
        _require_nonnegative_int(aggregate[field], "public aggregate {}".format(field))
    if not isinstance(aggregate["avoidable_idle_defect"], bool):
        raise CampaignSchedulerError("public aggregate defect marker must be boolean")
    if aggregate["capacity_ms"] != window_ms * len(report["workers"]):
        raise CampaignSchedulerError("public scheduler capacity differs from window and workers")
    if aggregate["busy_ms"] + aggregate["idle_ms"] != aggregate["capacity_ms"]:
        raise CampaignSchedulerError("public scheduler capacity is not fully accounted")
    if aggregate["legitimately_blocked_ms"] != aggregate["idle_ms"] - aggregate["avoidable_idle_ms"]:
        raise CampaignSchedulerError("public aggregate legitimate blocking differs from idle evidence")
    for field in ("busy_ms", "idle_ms", "avoidable_idle_ms", "legitimately_blocked_ms"):
        if aggregate[field] != sum(worker[field] for worker in report["workers"]):
            raise CampaignSchedulerError("public scheduler aggregate differs from worker accounting")
    observed_maximum = max(worker["max_avoidable_idle_streak_ms"] for worker in report["workers"])
    if aggregate["max_avoidable_idle_streak_ms"] != observed_maximum or aggregate["avoidable_idle_defect"] != (
        observed_maximum > AVOIDABLE_IDLE_DEFECT_MS
    ):
        raise CampaignSchedulerError("public aggregate defect evidence differs from worker records")
    if not isinstance(report["tasks"], Mapping) or set(report["tasks"]) != set(TASK_STATUSES):
        raise CampaignSchedulerError("public task counts are not a closed vocabulary")
    if not isinstance(report["arms"], Mapping) or set(report["arms"]) != set(ARM_STATUSES):
        raise CampaignSchedulerError("public arm counts are not a closed vocabulary")
    for label, counts in (("task", report["tasks"]), ("arm", report["arms"])):
        for count in counts.values():
            _require_nonnegative_int(count, "public {} count".format(label))
    return deepcopy(report)


__all__ = [
    "ARM_STATUSES",
    "AVOIDABLE_IDLE_DEFECT_MS",
    "CLAIM_BOUNDARY",
    "CONTROL_ACTIONS",
    "CampaignScheduler",
    "CampaignSchedulerError",
    "Dispatch",
    "MAX_STALL_RETRIES",
    "PILOT_CHECKPOINT_REQUEST",
    "PROGRESS_CHECKPOINT_MS",
    "RESOURCE_POOLS",
    "SCHEDULER_PUBLIC_SCHEMA",
    "SCHEDULER_STATE_SCHEMA",
    "STALL_TIMEOUT_MS",
    "STATE_AUTHORITY",
    "SchedulerControlAction",
    "SchedulerPlan",
    "SchedulerTask",
    "SchedulerWorker",
    "StallRecovery",
    "TASK_KINDS",
    "TASK_OWNER_ROLES",
    "TASK_STAGES",
    "TASK_STATUSES",
    "WORKER_CAPABILITIES",
    "validate_public_scheduler_report",
]
