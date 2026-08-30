"""Deterministic, crash-safe scheduling for the EGV capability screen.

The scheduler is deliberately a control-plane state machine.  It decides which
logical DAG nodes may run, persists every transition, and accounts for logical
worker time.  Callers remain responsible for executing model, evaluator, and
training activities and for returning content digests.  Consequently this
module supports issue #307's engineering checks only; it is not evidence of
model quality or live cross-host execution.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from ..canonical import canonical_bytes, digest_for, validate_sha256
from .errors import CampaignError
from .state import _durable_replace, _exclusive_path_lock


SCHEDULER_STATE_SCHEMA = "egv-campaign-scheduler-state-v1"
SCHEDULER_PUBLIC_SCHEMA = "egv-campaign-scheduler-public-v1"
STALL_TIMEOUT_MS = 10 * 60 * 1000
AVOIDABLE_IDLE_DEFECT_MS = 5 * 60 * 1000
PILOT_CHECKPOINT_REQUEST = 6
MAX_STALL_RETRIES = 1
MAX_PROGRESS_EVIDENCE_DIGESTS = 1024
CLAIM_BOUNDARY = "ENGINEERING_ONLY_NO_MODEL_QUALITY_OR_LIVE_DISTRIBUTED_CLAIM"

RESOURCE_POOLS = frozenset({"ACCELERATOR", "CPU"})
TASK_OWNER_ROLES = frozenset({"CONTROLLER", "EVALUATOR", "REVIEWER", "TRAINER"})
TASK_KINDS = frozenset({"GENERATION", "EVALUATION", "DIAGNOSTIC", "TRAINING", "VALIDATION", "PROMOTION", "REVIEW"})
TASK_STATUSES = frozenset({"PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"})
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
ARM_STATUSES = frozenset({"UNGATED", "SCREENING", "CONTINUING", "PAUSED_ZERO_SIGNAL"})
TERMINAL_REASONS = frozenset(
    {
        "COMPLETED",
        "ACTIVITY_FAILED",
        "STALL_RETRY_EXHAUSTED",
        "STALL_NO_REASSIGNMENT",
        "DEPENDENCY_NOT_COMPLETED",
        "ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT",
    }
)

_SAFE_INTERNAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_PUBLIC_ALIAS = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_PUBLIC_SLOT_ALIAS = re.compile(r"^slot-(?:[a-z]|[1-9][0-9]{0,2})$")
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
        "campaign_id",
        "plan_digest",
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
    {"stall_timeout_ms", "avoidable_idle_defect_ms", "pilot_checkpoint_request", "max_stall_retries"}
)
_WORKER_STATE_FIELDS = frozenset(
    {
        "worker_id",
        "resource_pool",
        "busy_ms",
        "idle_ms",
        "avoidable_idle_ms",
        "avoidable_idle_streak_ms",
        "max_avoidable_idle_streak_ms",
    }
)
_ARM_STATE_FIELDS = frozenset(
    {
        "arm_id",
        "status",
        "checkpoint_complete",
        "checkpoint_terminal_requests",
        "checkpoint_semantic_passes",
    }
)
_TASK_SPEC_FIELDS = frozenset(
    {
        "node_id",
        "kind",
        "resource_pool",
        "owner_role",
        "input_digest",
        "dependencies",
        "arm_id",
        "request_index",
        "promotion_key",
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
        "progress_digest",
        "progress_evidence_digests",
        "output_digest",
        "semantic_passed",
        "terminal_reason",
    }
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


class CampaignSchedulerError(CampaignError):
    """A scheduler transition, persisted state, or public projection is unsafe."""


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

    def __post_init__(self) -> None:
        _require_internal_id(self.worker_id, "worker ID")
        if self.resource_pool not in RESOURCE_POOLS:
            raise CampaignSchedulerError("worker resource pool is outside the closed vocabulary")

    def to_dict(self) -> Dict[str, Any]:
        return {"worker_id": self.worker_id, "resource_pool": self.resource_pool}


@dataclass(frozen=True)
class SchedulerTask:
    node_id: str
    kind: str
    resource_pool: str
    owner_role: str
    input_digest: str
    dependencies: Tuple[str, ...] = tuple()
    arm_id: Optional[str] = None
    request_index: Optional[int] = None
    promotion_key: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.dependencies, (str, bytes)):
            raise CampaignSchedulerError("scheduler dependencies must be an ordered sequence of IDs")
        try:
            dependencies = tuple(self.dependencies)
        except TypeError as exc:
            raise CampaignSchedulerError("scheduler dependencies must be an ordered sequence") from exc
        object.__setattr__(self, "dependencies", dependencies)
        _require_internal_id(self.node_id, "scheduler node ID")
        if self.kind not in TASK_KINDS:
            raise CampaignSchedulerError("scheduler task kind is outside the closed vocabulary")
        if self.resource_pool not in RESOURCE_POOLS:
            raise CampaignSchedulerError("scheduler task resource pool is outside the closed vocabulary")
        if self.owner_role not in TASK_OWNER_ROLES:
            raise CampaignSchedulerError("scheduler task owner is outside the closed vocabulary")
        _require_digest(self.input_digest, "scheduler task input digest")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise CampaignSchedulerError("scheduler task carries duplicate dependencies")
        for dependency in self.dependencies:
            _require_internal_id(dependency, "dependency ID")
        if self.node_id in self.dependencies:
            raise CampaignSchedulerError("scheduler task cannot depend on itself")
        if self.arm_id is not None:
            _require_internal_id(self.arm_id, "arm ID")
        if self.request_index is not None:
            if self.arm_id is None:
                raise CampaignSchedulerError("request-indexed task requires an arm ID")
            if (
                not isinstance(self.request_index, int)
                or isinstance(self.request_index, bool)
                or self.request_index < 1
            ):
                raise CampaignSchedulerError("request index must be a positive integer")
        if self.kind == "EVALUATION" and (self.arm_id is None or self.request_index is None):
            raise CampaignSchedulerError("evaluation task requires arm and request bindings")
        if self.kind == "PROMOTION":
            if self.promotion_key is None:
                raise CampaignSchedulerError("promotion task requires a content digest key")
            _require_digest(self.promotion_key, "promotion key")
        elif self.promotion_key is not None:
            raise CampaignSchedulerError("only a promotion task may carry a promotion key")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "resource_pool": self.resource_pool,
            "owner_role": self.owner_role,
            "input_digest": self.input_digest,
            "dependencies": sorted(self.dependencies),
            "arm_id": self.arm_id,
            "request_index": self.request_index,
            "promotion_key": self.promotion_key,
        }


@dataclass(frozen=True)
class SchedulerPlan:
    campaign_id: str
    workers: Tuple[SchedulerWorker, ...]
    tasks: Tuple[SchedulerTask, ...]

    def __post_init__(self) -> None:
        try:
            workers = tuple(self.workers)
            tasks = tuple(self.tasks)
        except TypeError as exc:
            raise CampaignSchedulerError("scheduler workers and tasks must be ordered sequences") from exc
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "tasks", tasks)
        _require_internal_id(self.campaign_id, "scheduler campaign ID")
        if not self.workers or not self.tasks:
            raise CampaignSchedulerError("scheduler plan requires workers and tasks")
        if any(type(worker) is not SchedulerWorker for worker in self.workers) or any(
            type(task) is not SchedulerTask for task in self.tasks
        ):
            raise CampaignSchedulerError("scheduler plan requires exact validated worker and task records")
        worker_ids = [worker.worker_id for worker in self.workers]
        node_ids = [task.node_id for task in self.tasks]
        if len(worker_ids) != len(set(worker_ids)) or len(node_ids) != len(set(node_ids)):
            raise CampaignSchedulerError("scheduler plan identities must be unique")
        known_nodes = set(node_ids)
        known_pools = {worker.resource_pool for worker in self.workers}
        for task in self.tasks:
            if not set(task.dependencies) <= known_nodes:
                raise CampaignSchedulerError("scheduler task references an unknown dependency")
            if task.resource_pool not in known_pools:
                raise CampaignSchedulerError("scheduler task has no worker in its resource pool")
        remaining = {task.node_id: set(task.dependencies) for task in self.tasks}
        while remaining:
            ready = sorted(node_id for node_id, dependencies in remaining.items() if not dependencies)
            if not ready:
                raise CampaignSchedulerError("scheduler task graph contains a cycle")
            for node_id in ready:
                remaining.pop(node_id)
            for dependencies in remaining.values():
                dependencies.difference_update(ready)
        promotion_keys = [task.promotion_key for task in self.tasks if task.promotion_key is not None]
        if len(promotion_keys) != len(set(promotion_keys)):
            raise CampaignSchedulerError("scheduler plan contains duplicate promotion authority")
        for arm_id in self.arm_ids:
            arm_tasks = [task for task in self.tasks if task.arm_id == arm_id]
            if any(task.request_index is not None for task in arm_tasks):
                checkpoint_evaluations = [
                    task.request_index
                    for task in arm_tasks
                    if task.kind == "EVALUATION"
                    and task.request_index is not None
                    and task.request_index <= PILOT_CHECKPOINT_REQUEST
                ]
                expected = list(range(1, PILOT_CHECKPOINT_REQUEST + 1))
                if sorted(checkpoint_evaluations) != expected:
                    raise CampaignSchedulerError(
                        "request-indexed arm must bind one evaluation to every frozen pilot checkpoint request"
                    )

    @property
    def arm_ids(self) -> Tuple[str, ...]:
        return tuple(sorted({task.arm_id for task in self.tasks if task.arm_id is not None}))

    def arm_is_gated(self, arm_id: str) -> bool:
        return any(task.arm_id == arm_id and task.request_index is not None for task in self.tasks)

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


@dataclass(frozen=True)
class StallRecovery:
    node_id: str
    previous_worker_id: str
    disposition: str


def _task_from_node(node: Mapping[str, Any]) -> SchedulerTask:
    return SchedulerTask(
        node_id=node["node_id"],
        kind=node["kind"],
        resource_pool=node["resource_pool"],
        owner_role=node["owner_role"],
        input_digest=node["input_digest"],
        dependencies=tuple(node["dependencies"]),
        arm_id=node["arm_id"],
        request_index=node["request_index"],
        promotion_key=node["promotion_key"],
    )


def _plan_from_state(state: Mapping[str, Any]) -> SchedulerPlan:
    workers = tuple(
        SchedulerWorker(worker_id=row["worker_id"], resource_pool=row["resource_pool"]) for row in state["workers"]
    )
    tasks = tuple(_task_from_node(row) for row in state["nodes"])
    return SchedulerPlan(campaign_id=state["campaign_id"], workers=workers, tasks=tasks)


def _initial_state(plan: SchedulerPlan, now_ms: int) -> Dict[str, Any]:
    _require_nonnegative_int(now_ms, "scheduler start time")
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
    arms = []
    for arm_id in plan.arm_ids:
        arms.append(
            {
                "arm_id": arm_id,
                "status": "SCREENING" if plan.arm_is_gated(arm_id) else "UNGATED",
                "checkpoint_complete": False,
                "checkpoint_terminal_requests": [],
                "checkpoint_semantic_passes": 0,
            }
        )
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
                "progress_digest": None,
                "progress_evidence_digests": [],
                "output_digest": None,
                "semantic_passed": None,
                "terminal_reason": None,
            }
        )
    return {
        "schema_version": SCHEDULER_STATE_SCHEMA,
        "campaign_id": plan.campaign_id,
        "plan_digest": plan.digest,
        "sequence": 0,
        "created_ms": now_ms,
        "updated_ms": now_ms,
        "config": {
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


def _arms_by_id(state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {arm["arm_id"]: arm for arm in state["arms"]}


def _workers_by_id(state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {worker["worker_id"]: worker for worker in state["workers"]}


def _validate_state(state: Any) -> Dict[str, Any]:
    state = dict(_require_closed(state, _STATE_FIELDS, "scheduler state"))
    if state["schema_version"] != SCHEDULER_STATE_SCHEMA:
        raise CampaignSchedulerError("unsupported scheduler state schema")
    _require_internal_id(state["campaign_id"], "scheduler campaign ID")
    _require_digest(state["plan_digest"], "scheduler plan digest")
    sequence = _require_nonnegative_int(state["sequence"], "scheduler state sequence")
    created_ms = _require_nonnegative_int(state["created_ms"], "scheduler creation time")
    updated_ms = _require_nonnegative_int(state["updated_ms"], "scheduler update time")
    if updated_ms < created_ms:
        raise CampaignSchedulerError("scheduler update time precedes creation")
    config = _require_closed(state["config"], _CONFIG_FIELDS, "scheduler config")
    expected_config = {
        "stall_timeout_ms": STALL_TIMEOUT_MS,
        "avoidable_idle_defect_ms": AVOIDABLE_IDLE_DEFECT_MS,
        "pilot_checkpoint_request": PILOT_CHECKPOINT_REQUEST,
        "max_stall_retries": MAX_STALL_RETRIES,
    }
    if dict(config) != expected_config:
        raise CampaignSchedulerError("scheduler config differs from the frozen control values")
    if not isinstance(state["workers"], list) or not state["workers"]:
        raise CampaignSchedulerError("scheduler state requires worker records")
    if not isinstance(state["nodes"], list) or not state["nodes"]:
        raise CampaignSchedulerError("scheduler state requires node records")
    if not isinstance(state["arms"], list):
        raise CampaignSchedulerError("scheduler arms must be a list")

    worker_ids = []
    for worker in state["workers"]:
        row = _require_closed(worker, _WORKER_STATE_FIELDS, "scheduler worker state")
        worker_ids.append(_require_internal_id(row["worker_id"], "worker ID"))
        if row["resource_pool"] not in RESOURCE_POOLS:
            raise CampaignSchedulerError("worker resource pool is invalid")
        for field in (
            "busy_ms",
            "idle_ms",
            "avoidable_idle_ms",
            "avoidable_idle_streak_ms",
            "max_avoidable_idle_streak_ms",
        ):
            _require_nonnegative_int(row[field], "worker {}".format(field))
        if row["avoidable_idle_ms"] > row["idle_ms"]:
            raise CampaignSchedulerError("avoidable idle exceeds total idle time")
        if row["busy_ms"] + row["idle_ms"] != updated_ms - created_ms:
            raise CampaignSchedulerError("worker time does not reconcile to the scheduler window")
        if row["max_avoidable_idle_streak_ms"] > row["avoidable_idle_ms"]:
            raise CampaignSchedulerError("worker avoidable-idle maximum exceeds its accumulated time")
        if row["avoidable_idle_streak_ms"] > row["max_avoidable_idle_streak_ms"]:
            raise CampaignSchedulerError("worker avoidable-idle streak exceeds its recorded maximum")
    if worker_ids != sorted(worker_ids) or len(worker_ids) != len(set(worker_ids)):
        raise CampaignSchedulerError("scheduler worker state is unordered or ambiguous")
    workers = _workers_by_id(state)

    node_ids = []
    running_assignments = []
    for node in state["nodes"]:
        row = _require_closed(node, _NODE_STATE_FIELDS, "scheduler node state")
        task = _task_from_node(row)
        node_ids.append(task.node_id)
        if row["dependencies"] != sorted(row["dependencies"]):
            raise CampaignSchedulerError("scheduler dependencies are not canonically ordered")
        if row["status"] not in TASK_STATUSES:
            raise CampaignSchedulerError("scheduler node status is invalid")
        attempt_count = _require_nonnegative_int(row["attempt_count"], "node attempt count")
        if attempt_count > MAX_STALL_RETRIES + 1:
            raise CampaignSchedulerError("node attempts exceed the frozen retry budget")
        if not isinstance(row["failed_worker_ids"], list):
            raise CampaignSchedulerError("failed worker IDs must be a list")
        failed_workers = row["failed_worker_ids"]
        if failed_workers != sorted(failed_workers) or len(failed_workers) != len(set(failed_workers)):
            raise CampaignSchedulerError("failed worker IDs are unordered or duplicated")
        for worker_id in failed_workers:
            if worker_id not in workers or workers[worker_id]["resource_pool"] != task.resource_pool:
                raise CampaignSchedulerError("failed worker identity is outside the task resource pool")
        assigned_worker_id = row["assigned_worker_id"]
        if assigned_worker_id is not None:
            if assigned_worker_id not in workers or workers[assigned_worker_id]["resource_pool"] != task.resource_pool:
                raise CampaignSchedulerError("node assignment is outside its resource pool")
        for field in ("started_ms", "last_progress_ms", "finished_ms"):
            value = row[field]
            if value is not None:
                _require_nonnegative_int(value, "node {}".format(field))
                if value > updated_ms:
                    raise CampaignSchedulerError("node timestamp exceeds scheduler update time")
        for field in ("progress_digest", "output_digest"):
            if row[field] is not None:
                _require_digest(row[field], "node {}".format(field))
        if not isinstance(row["progress_evidence_digests"], list):
            raise CampaignSchedulerError("progress evidence digest history must be a list")
        progress_history = row["progress_evidence_digests"]
        if len(progress_history) != len(set(progress_history)) or len(progress_history) > MAX_PROGRESS_EVIDENCE_DIGESTS:
            raise CampaignSchedulerError("progress evidence digest history is duplicated or exceeds its bound")
        for evidence_digest in progress_history:
            _require_digest(evidence_digest, "progress evidence digest")
        if row["progress_digest"] is not None and (
            not progress_history or row["progress_digest"] != progress_history[-1]
        ):
            raise CampaignSchedulerError("current progress digest differs from its evidence history")
        if row["semantic_passed"] is not None and not isinstance(row["semantic_passed"], bool):
            raise CampaignSchedulerError("semantic result must be boolean or null")
        if row["terminal_reason"] is not None and row["terminal_reason"] not in TERMINAL_REASONS:
            raise CampaignSchedulerError("node terminal reason is invalid")
        status = row["status"]
        if status == "PENDING":
            if any(
                row[field] is not None
                for field in (
                    "assigned_worker_id",
                    "started_ms",
                    "last_progress_ms",
                    "finished_ms",
                    "progress_digest",
                    "output_digest",
                    "semantic_passed",
                    "terminal_reason",
                )
            ):
                raise CampaignSchedulerError("pending node contains execution or terminal evidence")
        elif status == "RUNNING":
            if (
                assigned_worker_id is None
                or attempt_count < 1
                or row["started_ms"] is None
                or row["last_progress_ms"] is None
                or row["finished_ms"] is not None
                or row["output_digest"] is not None
                or row["semantic_passed"] is not None
                or row["terminal_reason"] is not None
            ):
                raise CampaignSchedulerError("running node execution evidence is incomplete")
            if assigned_worker_id in failed_workers or row["last_progress_ms"] < row["started_ms"]:
                raise CampaignSchedulerError("running node assignment or time lineage is invalid")
            running_assignments.append(assigned_worker_id)
        elif status == "COMPLETED":
            if (
                assigned_worker_id is None
                or attempt_count < 1
                or row["started_ms"] is None
                or row["last_progress_ms"] is None
                or row["finished_ms"] is None
                or row["output_digest"] is None
                or row["terminal_reason"] != "COMPLETED"
            ):
                raise CampaignSchedulerError("completed node lacks exact execution evidence")
            if assigned_worker_id in failed_workers or not (
                row["started_ms"] <= row["last_progress_ms"] <= row["finished_ms"]
            ):
                raise CampaignSchedulerError("completed node assignment or time lineage is invalid")
            if (task.kind == "EVALUATION") != isinstance(row["semantic_passed"], bool):
                raise CampaignSchedulerError("only completed evaluation nodes carry semantic results")
        elif status == "FAILED":
            if (
                assigned_worker_id is None
                or attempt_count < 1
                or row["started_ms"] is None
                or row["last_progress_ms"] is None
                or row["finished_ms"] is None
                or row["output_digest"] is not None
                or row["semantic_passed"] is not None
                or row["terminal_reason"]
                not in {
                    "ACTIVITY_FAILED",
                    "STALL_RETRY_EXHAUSTED",
                    "STALL_NO_REASSIGNMENT",
                }
            ):
                raise CampaignSchedulerError("failed node evidence is incomplete")
            if not row["started_ms"] <= row["last_progress_ms"] <= row["finished_ms"]:
                raise CampaignSchedulerError("failed node time lineage is invalid")
        else:
            if (
                assigned_worker_id is not None
                or row["started_ms"] is not None
                or row["last_progress_ms"] is not None
                or row["progress_digest"] is not None
                or row["finished_ms"] is None
                or row["output_digest"] is not None
                or row["semantic_passed"] is not None
                or row["terminal_reason"]
                not in {"DEPENDENCY_NOT_COMPLETED", "ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT"}
            ):
                raise CampaignSchedulerError("cancelled node evidence is incomplete")
    if node_ids != sorted(node_ids) or len(node_ids) != len(set(node_ids)):
        raise CampaignSchedulerError("scheduler nodes are unordered or ambiguous")
    if len(running_assignments) != len(set(running_assignments)):
        raise CampaignSchedulerError("one worker is assigned to multiple running nodes")
    if any(workers[worker_id]["avoidable_idle_streak_ms"] for worker_id in running_assignments):
        raise CampaignSchedulerError("running worker retained an avoidable-idle streak")

    plan = _plan_from_state(state)
    if plan.digest != state["plan_digest"]:
        raise CampaignSchedulerError("scheduler plan digest differs from persisted task inventory")
    nodes = _nodes_by_id(state)
    for node in state["nodes"]:
        if node["status"] in {"RUNNING", "COMPLETED"}:
            acceptable_statuses = TERMINAL_STATUSES if node["kind"] == "DIAGNOSTIC" else {"COMPLETED"}
            if any(nodes[dependency]["status"] not in acceptable_statuses for dependency in node["dependencies"]):
                raise CampaignSchedulerError("running or completed node has an incomplete dependency")

    arm_ids = []
    for arm in state["arms"]:
        row = _require_closed(arm, _ARM_STATE_FIELDS, "scheduler arm state")
        arm_id = _require_internal_id(row["arm_id"], "arm ID")
        arm_ids.append(arm_id)
        if row["status"] not in ARM_STATUSES or not isinstance(row["checkpoint_complete"], bool):
            raise CampaignSchedulerError("scheduler arm state is invalid")
        if not isinstance(row["checkpoint_terminal_requests"], list):
            raise CampaignSchedulerError("checkpoint terminal requests must be a list")
        requests = row["checkpoint_terminal_requests"]
        if requests != sorted(requests) or len(requests) != len(set(requests)):
            raise CampaignSchedulerError("checkpoint terminal requests are unordered or duplicated")
        for request_index in requests:
            if (
                not isinstance(request_index, int)
                or isinstance(request_index, bool)
                or request_index < 1
                or request_index > PILOT_CHECKPOINT_REQUEST
            ):
                raise CampaignSchedulerError("checkpoint terminal request is outside the frozen range")
        semantic_passes = _require_nonnegative_int(row["checkpoint_semantic_passes"], "checkpoint semantic pass count")
        checkpoint_nodes = [
            node
            for node in state["nodes"]
            if node["arm_id"] == arm_id
            and node["kind"] == "EVALUATION"
            and node["request_index"] is not None
            and node["request_index"] <= PILOT_CHECKPOINT_REQUEST
        ]
        observed_requests = sorted(
            node["request_index"] for node in checkpoint_nodes if node["status"] in TERMINAL_STATUSES
        )
        observed_passes = sum(
            1 for node in checkpoint_nodes if node["status"] == "COMPLETED" and node["semantic_passed"] is True
        )
        if requests != observed_requests or semantic_passes != observed_passes:
            raise CampaignSchedulerError("scheduler arm projection differs from checkpoint node evidence")
        gated = plan.arm_is_gated(arm_id)
        complete = gated and observed_requests == list(range(1, PILOT_CHECKPOINT_REQUEST + 1))
        expected_status = "UNGATED"
        if gated and not complete:
            expected_status = "SCREENING"
        elif complete:
            expected_status = "CONTINUING" if observed_passes else "PAUSED_ZERO_SIGNAL"
        if row["checkpoint_complete"] != complete or row["status"] != expected_status:
            raise CampaignSchedulerError("scheduler arm status differs from frozen checkpoint evidence")
        if expected_status == "PAUSED_ZERO_SIGNAL":
            for node in state["nodes"]:
                if (
                    node["arm_id"] == arm_id
                    and node["request_index"] is not None
                    and node["request_index"] > PILOT_CHECKPOINT_REQUEST
                    and (
                        node["status"] != "CANCELLED"
                        or node["terminal_reason"] != "ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT"
                    )
                ):
                    raise CampaignSchedulerError("zero-signal arm retained post-checkpoint work")
    if arm_ids != sorted(arm_ids) or len(arm_ids) != len(set(arm_ids)):
        raise CampaignSchedulerError("scheduler arm state is unordered or ambiguous")
    if tuple(arm_ids) != plan.arm_ids:
        raise CampaignSchedulerError("scheduler arm projection differs from the frozen plan")
    _ = sequence
    return state


class _SchedulerStateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def create(self, plan: SchedulerPlan, *, now_ms: int) -> Dict[str, Any]:
        with _exclusive_path_lock(self.lock_path):
            if self.path.exists():
                raise CampaignSchedulerError("scheduler state is already initialized")
            state = _initial_state(plan, now_ms)
            _validate_state(state)
            self._atomic_write(state)
            return deepcopy(state)

    def load(self) -> Dict[str, Any]:
        with _exclusive_path_lock(self.lock_path):
            return deepcopy(self._load_unlocked())

    def transact(self, transition: Callable[[Dict[str, Any]], Any]) -> Any:
        with _exclusive_path_lock(self.lock_path):
            state = self._load_unlocked()
            before = canonical_bytes(state)
            result = transition(state)
            if canonical_bytes(state) != before:
                state["sequence"] += 1
                _validate_state(state)
                self._atomic_write(state)
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
        return state

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


def _task_ready(state: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    if node["status"] != "PENDING":
        return False
    nodes = _nodes_by_id(state)
    if node["kind"] == "DIAGNOSTIC":
        dependencies_satisfied = all(
            nodes[dependency]["status"] in TERMINAL_STATUSES for dependency in node["dependencies"]
        )
    else:
        dependencies_satisfied = all(nodes[dependency]["status"] == "COMPLETED" for dependency in node["dependencies"])
    if not dependencies_satisfied:
        return False
    if node["arm_id"] is None:
        return True
    arm = _arms_by_id(state)[node["arm_id"]]
    if arm["status"] == "PAUSED_ZERO_SIGNAL":
        return node["kind"] == "DIAGNOSTIC" and (
            node["request_index"] is None or node["request_index"] <= PILOT_CHECKPOINT_REQUEST
        )
    if (
        node["request_index"] is not None
        and node["request_index"] > PILOT_CHECKPOINT_REQUEST
        and arm["status"] != "CONTINUING"
    ):
        return False
    return True


def _avoidable_idle_workers(state: Mapping[str, Any]) -> frozenset:
    running_workers = {node["assigned_worker_id"] for node in state["nodes"] if node["status"] == "RUNNING"}
    idle_workers = [worker for worker in state["workers"] if worker["worker_id"] not in running_workers]
    available = {worker["worker_id"]: worker for worker in idle_workers}
    matched = set()
    ready_nodes = sorted(
        (node for node in state["nodes"] if _task_ready(state, node)), key=lambda item: item["node_id"]
    )
    for node in ready_nodes:
        candidates = sorted(
            worker_id
            for worker_id, worker in available.items()
            if worker["resource_pool"] == node["resource_pool"]
            and worker_id not in node["failed_worker_ids"]
            and worker_id not in matched
        )
        if candidates:
            matched.add(candidates[0])
    return frozenset(matched)


def _accrue_utilization(state: Dict[str, Any], now_ms: int) -> None:
    _require_nonnegative_int(now_ms, "scheduler transition time")
    if now_ms < state["updated_ms"]:
        raise CampaignSchedulerError("scheduler transition time regressed")
    delta = now_ms - state["updated_ms"]
    if delta == 0:
        return
    running_workers = {node["assigned_worker_id"] for node in state["nodes"] if node["status"] == "RUNNING"}
    avoidable_workers = _avoidable_idle_workers(state)
    for worker in state["workers"]:
        worker_id = worker["worker_id"]
        if worker_id in running_workers:
            worker["busy_ms"] += delta
            worker["avoidable_idle_streak_ms"] = 0
        else:
            worker["idle_ms"] += delta
            if worker_id in avoidable_workers:
                worker["avoidable_idle_ms"] += delta
                worker["avoidable_idle_streak_ms"] += delta
                worker["max_avoidable_idle_streak_ms"] = max(
                    worker["max_avoidable_idle_streak_ms"], worker["avoidable_idle_streak_ms"]
                )
            else:
                worker["avoidable_idle_streak_ms"] = 0
    state["updated_ms"] = now_ms


def _cancel_node(node: Dict[str, Any], *, now_ms: int, reason: str) -> None:
    if node["status"] != "PENDING":
        return
    node["status"] = "CANCELLED"
    node["finished_ms"] = now_ms
    node["terminal_reason"] = reason


def _propagate_unreachable(state: Dict[str, Any], *, now_ms: int) -> bool:
    changed = False
    nodes = _nodes_by_id(state)
    for node in state["nodes"]:
        if (
            node["kind"] != "DIAGNOSTIC"
            and node["status"] == "PENDING"
            and any(nodes[dependency]["status"] in {"FAILED", "CANCELLED"} for dependency in node["dependencies"])
        ):
            _cancel_node(node, now_ms=now_ms, reason="DEPENDENCY_NOT_COMPLETED")
            changed = True
    return changed


def _refresh_checkpoints(state: Dict[str, Any], *, now_ms: int) -> bool:
    changed = False
    for arm in state["arms"]:
        if arm["status"] == "UNGATED":
            continue
        checkpoint_nodes = [
            node
            for node in state["nodes"]
            if node["arm_id"] == arm["arm_id"]
            and node["kind"] == "EVALUATION"
            and node["request_index"] is not None
            and node["request_index"] <= PILOT_CHECKPOINT_REQUEST
        ]
        terminal_requests = sorted(
            node["request_index"] for node in checkpoint_nodes if node["status"] in TERMINAL_STATUSES
        )
        semantic_passes = sum(
            1 for node in checkpoint_nodes if node["status"] == "COMPLETED" and node["semantic_passed"] is True
        )
        if arm["checkpoint_terminal_requests"] != terminal_requests:
            arm["checkpoint_terminal_requests"] = terminal_requests
            changed = True
        if arm["checkpoint_semantic_passes"] != semantic_passes:
            arm["checkpoint_semantic_passes"] = semantic_passes
            changed = True
        complete = terminal_requests == list(range(1, PILOT_CHECKPOINT_REQUEST + 1))
        if arm["checkpoint_complete"] != complete:
            arm["checkpoint_complete"] = complete
            changed = True
        expected_status = "SCREENING"
        if complete:
            expected_status = "CONTINUING" if semantic_passes else "PAUSED_ZERO_SIGNAL"
        if arm["status"] != expected_status:
            arm["status"] = expected_status
            changed = True
        if expected_status == "PAUSED_ZERO_SIGNAL":
            for node in state["nodes"]:
                if (
                    node["arm_id"] == arm["arm_id"]
                    and node["request_index"] is not None
                    and node["request_index"] > PILOT_CHECKPOINT_REQUEST
                ):
                    if node["status"] == "PENDING":
                        _cancel_node(
                            node,
                            now_ms=now_ms,
                            reason="ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT",
                        )
                        changed = True
                    elif (
                        node["status"] == "CANCELLED"
                        and node["terminal_reason"] != "ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT"
                    ):
                        node["terminal_reason"] = "ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT"
                        changed = True
    return changed


def _settle_state(state: Dict[str, Any], *, now_ms: int) -> None:
    maximum_rounds = len(state["nodes"]) + len(state["arms"]) + 1
    for _ in range(maximum_rounds):
        changed = _propagate_unreachable(state, now_ms=now_ms)
        changed = _refresh_checkpoints(state, now_ms=now_ms) or changed
        if not changed:
            return
    raise CampaignSchedulerError("scheduler dependency settlement did not converge")


class CampaignScheduler:
    """Durable work-conserving scheduler over logical workers and digest receipts."""

    def __init__(self, state_path: Path) -> None:
        self._store = _SchedulerStateStore(Path(state_path))
        self._store.load()

    @classmethod
    def initialize(cls, state_path: Path, plan: SchedulerPlan, *, now_ms: int = 0) -> "CampaignScheduler":
        if type(plan) is not SchedulerPlan:
            raise CampaignSchedulerError("scheduler initialization requires an exact validated plan")
        store = _SchedulerStateStore(Path(state_path))
        store.create(plan, now_ms=now_ms)
        return cls(state_path)

    def snapshot(self) -> Dict[str, Any]:
        """Return the private, closed scheduler state for trusted orchestration."""

        return self._store.load()

    def dispatch_ready(self, *, now_ms: int) -> Tuple[Dispatch, ...]:
        """Assign every currently ready node for which an eligible worker is idle."""

        def transition(state: Dict[str, Any]) -> Tuple[Dispatch, ...]:
            _accrue_utilization(state, now_ms)
            _settle_state(state, now_ms=now_ms)
            occupied = {node["assigned_worker_id"] for node in state["nodes"] if node["status"] == "RUNNING"}
            available = {
                worker["worker_id"]: worker for worker in state["workers"] if worker["worker_id"] not in occupied
            }
            dispatches = []
            for node in sorted(state["nodes"], key=lambda item: item["node_id"]):
                if not _task_ready(state, node):
                    continue
                candidates = sorted(
                    worker_id
                    for worker_id, worker in available.items()
                    if worker["resource_pool"] == node["resource_pool"] and worker_id not in node["failed_worker_ids"]
                )
                if not candidates:
                    continue
                worker_id = candidates[0]
                worker = available.pop(worker_id)
                node["status"] = "RUNNING"
                node["assigned_worker_id"] = worker_id
                node["attempt_count"] += 1
                node["started_ms"] = now_ms
                node["last_progress_ms"] = now_ms
                worker["avoidable_idle_streak_ms"] = 0
                dispatches.append(Dispatch(node["node_id"], worker_id, node["attempt_count"]))
            return tuple(dispatches)

        return self._store.transact(transition)

    def record_progress(
        self,
        node_id: str,
        *,
        worker_id: str,
        evidence_digest: str,
        now_ms: int,
    ) -> Dict[str, Any]:
        """Record new evidence; duplicate heartbeats do not reset the stall clock."""

        _require_internal_id(node_id, "scheduler node ID")
        _require_internal_id(worker_id, "worker ID")
        evidence_digest = _require_digest(evidence_digest, "progress evidence digest")

        def transition(state: Dict[str, Any]) -> Dict[str, Any]:
            _accrue_utilization(state, now_ms)
            node = _nodes_by_id(state).get(node_id)
            if node is None or node["status"] != "RUNNING" or node["assigned_worker_id"] != worker_id:
                raise CampaignSchedulerError("progress evidence does not match one running assignment")
            if node["progress_digest"] != evidence_digest:
                if evidence_digest not in node["progress_evidence_digests"]:
                    if len(node["progress_evidence_digests"]) >= MAX_PROGRESS_EVIDENCE_DIGESTS:
                        raise CampaignSchedulerError("progress evidence history reached its frozen bound")
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
        output_digest: str,
        now_ms: int,
        semantic_passed: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Commit one exact completion; exact retries are idempotent after restart."""

        _require_internal_id(node_id, "scheduler node ID")
        _require_internal_id(worker_id, "worker ID")
        output_digest = _require_digest(output_digest, "node output digest")
        if semantic_passed is not None and not isinstance(semantic_passed, bool):
            raise CampaignSchedulerError("semantic result must be boolean or null")

        def transition(state: Dict[str, Any]) -> Dict[str, Any]:
            _accrue_utilization(state, now_ms)
            node = _nodes_by_id(state).get(node_id)
            if node is None:
                raise CampaignSchedulerError("completion references an unknown node")
            expected_semantic = node["kind"] == "EVALUATION"
            if expected_semantic != isinstance(semantic_passed, bool):
                raise CampaignSchedulerError("evaluation completion alone requires a semantic result")
            if node["status"] == "COMPLETED":
                if (
                    node["assigned_worker_id"] != worker_id
                    or node["output_digest"] != output_digest
                    or node["semantic_passed"] != semantic_passed
                ):
                    raise CampaignSchedulerError("completion retry conflicts with durable evidence")
                return deepcopy(node)
            if node["status"] != "RUNNING" or node["assigned_worker_id"] != worker_id:
                raise CampaignSchedulerError("completion does not match one running assignment")
            node["status"] = "COMPLETED"
            node["last_progress_ms"] = now_ms
            node["finished_ms"] = now_ms
            node["output_digest"] = output_digest
            node["semantic_passed"] = semantic_passed
            node["terminal_reason"] = "COMPLETED"
            _settle_state(state, now_ms=now_ms)
            return deepcopy(node)

        return self._store.transact(transition)

    def fail(self, node_id: str, *, worker_id: str, now_ms: int) -> Dict[str, Any]:
        """Fail one exact running assignment and close now-unreachable descendants."""

        _require_internal_id(node_id, "scheduler node ID")
        _require_internal_id(worker_id, "worker ID")

        def transition(state: Dict[str, Any]) -> Dict[str, Any]:
            _accrue_utilization(state, now_ms)
            node = _nodes_by_id(state).get(node_id)
            if node is None:
                raise CampaignSchedulerError("failure references an unknown node")
            if node["status"] == "FAILED":
                if node["assigned_worker_id"] == worker_id and node["terminal_reason"] == "ACTIVITY_FAILED":
                    return deepcopy(node)
                raise CampaignSchedulerError("failure retry conflicts with durable evidence")
            if node["status"] != "RUNNING" or node["assigned_worker_id"] != worker_id:
                raise CampaignSchedulerError("failure does not match one running assignment")
            node["status"] = "FAILED"
            node["last_progress_ms"] = now_ms
            node["finished_ms"] = now_ms
            node["terminal_reason"] = "ACTIVITY_FAILED"
            _settle_state(state, now_ms=now_ms)
            return deepcopy(node)

        return self._store.transact(transition)

    def recover_stalls(self, *, now_ms: int) -> Tuple[StallRecovery, ...]:
        """Requeue once onto a different worker, then fail closed at the bound."""

        def transition(state: Dict[str, Any]) -> Tuple[StallRecovery, ...]:
            _accrue_utilization(state, now_ms)
            recoveries = []
            workers = _workers_by_id(state)
            for node in sorted(state["nodes"], key=lambda item: item["node_id"]):
                if node["status"] != "RUNNING":
                    continue
                if now_ms - node["last_progress_ms"] < STALL_TIMEOUT_MS:
                    continue
                previous_worker_id = node["assigned_worker_id"]
                node["failed_worker_ids"] = sorted(set(node["failed_worker_ids"] + [previous_worker_id]))
                alternates = sorted(
                    worker_id
                    for worker_id, worker in workers.items()
                    if worker["resource_pool"] == node["resource_pool"] and worker_id not in node["failed_worker_ids"]
                )
                if node["attempt_count"] <= MAX_STALL_RETRIES and alternates:
                    node["status"] = "PENDING"
                    node["assigned_worker_id"] = None
                    node["started_ms"] = None
                    node["last_progress_ms"] = None
                    node["progress_digest"] = None
                    recoveries.append(StallRecovery(node["node_id"], previous_worker_id, "RETRY_PENDING"))
                    continue
                node["status"] = "FAILED"
                node["last_progress_ms"] = now_ms
                node["finished_ms"] = now_ms
                node["terminal_reason"] = (
                    "STALL_RETRY_EXHAUSTED" if node["attempt_count"] > MAX_STALL_RETRIES else "STALL_NO_REASSIGNMENT"
                )
                recoveries.append(StallRecovery(node["node_id"], previous_worker_id, "FAILED_CLOSED"))
            _settle_state(state, now_ms=now_ms)
            return tuple(recoveries)

        return self._store.transact(transition)

    def public_report(self, *, now_ms: int, worker_aliases: Mapping[str, str]) -> Dict[str, Any]:
        """Return a closed aggregate report with no private identifiers or paths."""

        if not isinstance(worker_aliases, Mapping):
            raise CampaignSchedulerError("public worker aliases must be a mapping")
        initial = self._store.load()
        internal_ids = {worker["worker_id"] for worker in initial["workers"]}
        if set(worker_aliases) != internal_ids:
            raise CampaignSchedulerError("public aliases must cover the exact private worker inventory")
        aliases = {worker_id: _require_public_alias(worker_aliases[worker_id]) for worker_id in internal_ids}
        if len(set(aliases.values())) != len(aliases):
            raise CampaignSchedulerError("public worker aliases must be unique")

        def accrue(state: Dict[str, Any]) -> None:
            _accrue_utilization(state, now_ms)

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
        capacity_ms = (state["updated_ms"] - state["created_ms"]) * len(workers)
        aggregate_maximum = max(worker["max_avoidable_idle_streak_ms"] for worker in workers)
        task_counts = {
            status: sum(1 for node in state["nodes"] if node["status"] == status) for status in sorted(TASK_STATUSES)
        }
        arm_counts = {
            status: sum(1 for arm in state["arms"] if arm["status"] == status) for status in sorted(ARM_STATUSES)
        }
        report = {
            "schema_version": SCHEDULER_PUBLIC_SCHEMA,
            "campaign_digest": digest_for({"campaign_id": state["campaign_id"]}),
            "plan_digest": state["plan_digest"],
            "state_sequence": state["sequence"],
            "window_ms": state["updated_ms"] - state["created_ms"],
            "workers": workers,
            "aggregate": {
                "capacity_ms": capacity_ms,
                "busy_ms": sum(worker["busy_ms"] for worker in workers),
                "idle_ms": sum(worker["idle_ms"] for worker in workers),
                "avoidable_idle_ms": sum(worker["avoidable_idle_ms"] for worker in workers),
                "legitimately_blocked_ms": sum(worker["legitimately_blocked_ms"] for worker in workers),
                "max_avoidable_idle_streak_ms": aggregate_maximum,
                "avoidable_idle_defect": aggregate_maximum > AVOIDABLE_IDLE_DEFECT_MS,
            },
            "tasks": task_counts,
            "arms": arm_counts,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        return validate_public_scheduler_report(report)


def validate_public_scheduler_report(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the closed public scheduler projection and reject leak surfaces."""

    report = dict(_require_closed(value, _PUBLIC_FIELDS, "public scheduler report"))
    if report["schema_version"] != SCHEDULER_PUBLIC_SCHEMA or report["claim_boundary"] != CLAIM_BOUNDARY:
        raise CampaignSchedulerError("public scheduler report identity or claim boundary is invalid")
    _require_digest(report["campaign_digest"], "public campaign digest")
    _require_digest(report["plan_digest"], "public plan digest")
    _require_nonnegative_int(report["state_sequence"], "public scheduler sequence")
    window_ms = _require_nonnegative_int(report["window_ms"], "public scheduler window")
    if not isinstance(report["workers"], list) or not report["workers"]:
        raise CampaignSchedulerError("public scheduler report requires worker aggregates")
    aliases = []
    for worker in report["workers"]:
        row = _require_closed(worker, _PUBLIC_WORKER_FIELDS, "public scheduler worker")
        aliases.append(_require_public_alias(row["alias"]))
        if row["resource_pool"] not in RESOURCE_POOLS:
            raise CampaignSchedulerError("public scheduler worker resource is invalid")
        for field in (
            "busy_ms",
            "idle_ms",
            "avoidable_idle_ms",
            "legitimately_blocked_ms",
            "max_avoidable_idle_streak_ms",
        ):
            _require_nonnegative_int(row[field], "public worker {}".format(field))
        if not isinstance(row["avoidable_idle_defect"], bool):
            raise CampaignSchedulerError("public worker defect marker must be boolean")
        if row["avoidable_idle_ms"] > row["idle_ms"]:
            raise CampaignSchedulerError("public avoidable idle exceeds total idle")
        if row["legitimately_blocked_ms"] != row["idle_ms"] - row["avoidable_idle_ms"]:
            raise CampaignSchedulerError("public legitimate blocking differs from worker idle evidence")
        if row["avoidable_idle_defect"] != (row["max_avoidable_idle_streak_ms"] > AVOIDABLE_IDLE_DEFECT_MS):
            raise CampaignSchedulerError("public worker defect marker differs from the frozen threshold")
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
        raise CampaignSchedulerError("public scheduler capacity differs from window and worker count")
    if aggregate["busy_ms"] + aggregate["idle_ms"] != aggregate["capacity_ms"]:
        raise CampaignSchedulerError("public scheduler capacity is not fully accounted")
    if aggregate["avoidable_idle_ms"] > aggregate["idle_ms"]:
        raise CampaignSchedulerError("public aggregate avoidable idle exceeds total idle")
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
    expected_task_fields = set(TASK_STATUSES)
    expected_arm_fields = set(ARM_STATUSES)
    if not isinstance(report["tasks"], Mapping) or set(report["tasks"]) != expected_task_fields:
        raise CampaignSchedulerError("public task counts are not a closed vocabulary")
    if not isinstance(report["arms"], Mapping) or set(report["arms"]) != expected_arm_fields:
        raise CampaignSchedulerError("public arm counts are not a closed vocabulary")
    for label, counts in (("task", report["tasks"]), ("arm", report["arms"])):
        for count in counts.values():
            _require_nonnegative_int(count, "public {} count".format(label))
    return deepcopy(report)


__all__ = [
    "ARM_STATUSES",
    "AVOIDABLE_IDLE_DEFECT_MS",
    "CLAIM_BOUNDARY",
    "CampaignScheduler",
    "CampaignSchedulerError",
    "Dispatch",
    "MAX_STALL_RETRIES",
    "MAX_PROGRESS_EVIDENCE_DIGESTS",
    "PILOT_CHECKPOINT_REQUEST",
    "RESOURCE_POOLS",
    "SCHEDULER_PUBLIC_SCHEMA",
    "SCHEDULER_STATE_SCHEMA",
    "STALL_TIMEOUT_MS",
    "SchedulerPlan",
    "SchedulerTask",
    "SchedulerWorker",
    "StallRecovery",
    "TASK_KINDS",
    "TASK_OWNER_ROLES",
    "TASK_STATUSES",
    "validate_public_scheduler_report",
]
