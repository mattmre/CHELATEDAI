"""Shared fail-closed contracts for the bounded RB-10 research lanes.

This module is deliberately dependency-light.  It supplies only the resource,
integrity, determinism, and publication boundary shared by the JO1, A1, G2,
and G2A experiment modules.  It does not implement or promote any research
mechanism.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple


MIB = 1024 * 1024

RECORD_TYPE = "prime_ring_rb10_method_dev_result"
PROTOCOL_ID = "CHELATEDAI-PRW-RB10-CONTRACT-v1"
SCHEMA_VERSION = "1.1.0"
EVIDENCE_MODE = "METHOD_DEV"
STATUS_BOUNDARY = "WORKFLOW_STATUS_ONLY_NOT_INDEPENDENT_RUNTIME_EVIDENCE"

HARD_MAX_ESTIMATED_BYTES = 256 * MIB
HARD_MAX_WORK_UNITS = 25_000_000
HARD_MAX_SECONDS = 30.0
HARD_MAX_ASSIGNMENTS = 200_000
HARD_MAX_NODES = 6
HARD_MAX_FACTORS = 8
HARD_MAX_FACTOR_ARITY = 3
HARD_MAX_OUTPUT_BYTES = 64 * MIB
HARD_MAX_INPUT_BITS = 64
HARD_MAX_JSON_DEPTH = 64
HARD_MAX_JSON_NODES = 200_000
HARD_MAX_JSON_STRING_CHARS = 1_048_576
HARD_MAX_PATH_CHARS = 4096

DEFAULT_MAX_ESTIMATED_BYTES = 32 * MIB
DEFAULT_MAX_WORK_UNITS = 5_000_000
DEFAULT_MAX_SECONDS = 10.0
DEFAULT_MAX_ASSIGNMENTS = 50_000
DEFAULT_MAX_NODES = HARD_MAX_NODES
DEFAULT_MAX_FACTORS = HARD_MAX_FACTORS
DEFAULT_MAX_FACTOR_ARITY = HARD_MAX_FACTOR_ARITY
DEFAULT_MAX_OUTPUT_BYTES = 8 * MIB

ALLOWED_STATUSES = (
    "UNEXECUTED",
    "PREFLIGHT_ADMITTED",
    "COMPLETE",
    "REFUSED",
    "TIMEOUT",
    "ERROR",
)

# These exact field names may be removed from deterministic comparisons.  The
# allowlist prevents a caller from declaring a scientific outcome field
# "nondeterministic" and thereby removing it from the content digest.
ALLOWED_NONDETERMINISTIC_TIMING_FIELDS = (
    "elapsed_ms",
    "elapsed_ns",
    "elapsed_seconds",
    "finished_at",
    "p95_latency_ms",
    "route_latency_ms",
    "sampled_at_monotonic_ns",
    "started_at",
    "timestamp",
    "wall_clock_seconds",
)
DEFAULT_NONDETERMINISTIC_TIMING_FIELDS = (
    "elapsed_ms",
    "elapsed_ns",
    "elapsed_seconds",
    "finished_at",
    "p95_latency_ms",
    "route_latency_ms",
    "sampled_at_monotonic_ns",
    "started_at",
    "timestamp",
    "wall_clock_seconds",
)


class RB10ValidationError(ValueError):
    """Raised when untrusted contract input is malformed."""


class RB10ResourceError(RuntimeError):
    """Raised before or during work when an immutable resource gate is crossed."""


class RB10IntegrityError(RuntimeError):
    """Raised when a retained artifact no longer matches its bound contract."""


def _plain_int(
    value: object,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise RB10ValidationError(f"{name} must be a plain integer")
    if value.bit_length() > HARD_MAX_INPUT_BITS:
        raise RB10ValidationError(f"{name} exceeds the {HARD_MAX_INPUT_BITS}-bit input ceiling")
    if value < minimum or value > maximum:
        raise RB10ValidationError(f"{name} must be in the closed interval [{minimum}, {maximum}]")
    return value


def _positive_seconds(value: object, name: str) -> float:
    if type(value) is int:
        if value.bit_length() > HARD_MAX_INPUT_BITS:
            raise RB10ValidationError(f"{name} exceeds the {HARD_MAX_INPUT_BITS}-bit input ceiling")
        result = float(value)
    elif type(value) is float:
        result = float.__float__(value)
    else:
        raise RB10ValidationError(f"{name} must be a plain real number")
    if not math.isfinite(result) or result <= 0.0:
        raise RB10ValidationError(f"{name} must be positive and finite")
    return result


def _plain_string(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
    maximum_chars: int = 256,
) -> str:
    if type(value) is not str:
        raise RB10ValidationError(f"{name} must be a plain string")
    if not allow_empty and not value:
        raise RB10ValidationError(f"{name} cannot be empty")
    if len(value) > maximum_chars:
        raise RB10ValidationError(f"{name} exceeds the {maximum_chars}-character ceiling")
    return value


def _plain_string_tuple(
    values: object,
    name: str,
    *,
    allowed: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    if type(values) is not tuple:
        raise RB10ValidationError(f"{name} must be a plain tuple")
    result = tuple(_plain_string(item, f"{name}[{index}]") for index, item in enumerate(values))
    if len(set(result)) != len(result):
        raise RB10ValidationError(f"{name} cannot contain duplicates")
    if allowed is not None:
        allowed_set = set(allowed)
        unknown = tuple(item for item in result if item not in allowed_set)
        if unknown:
            raise RB10ValidationError(
                f"{name} contains fields outside the nondeterministic allowlist: " + ", ".join(unknown)
            )
    return result


@dataclass(frozen=True)
class RB10Budget:
    """Caller-lowerable limits bounded by the immutable RB-10 envelope."""

    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_assignments: int = DEFAULT_MAX_ASSIGNMENTS
    max_nodes: int = DEFAULT_MAX_NODES
    max_factors: int = DEFAULT_MAX_FACTORS
    max_factor_arity: int = DEFAULT_MAX_FACTOR_ARITY
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES

    def __post_init__(self) -> None:
        integer_limits = (
            (
                "max_estimated_bytes",
                self.max_estimated_bytes,
                HARD_MAX_ESTIMATED_BYTES,
            ),
            ("max_work_units", self.max_work_units, HARD_MAX_WORK_UNITS),
            ("max_assignments", self.max_assignments, HARD_MAX_ASSIGNMENTS),
            ("max_nodes", self.max_nodes, HARD_MAX_NODES),
            ("max_factors", self.max_factors, HARD_MAX_FACTORS),
            (
                "max_factor_arity",
                self.max_factor_arity,
                HARD_MAX_FACTOR_ARITY,
            ),
            ("max_output_bytes", self.max_output_bytes, HARD_MAX_OUTPUT_BYTES),
        )
        for name, value, hard_maximum in integer_limits:
            checked = _plain_int(value, name, 1, hard_maximum)
            object.__setattr__(self, name, checked)
        seconds = _positive_seconds(self.max_seconds, "max_seconds")
        if seconds > HARD_MAX_SECONDS:
            raise RB10ValidationError(f"max_seconds cannot exceed immutable hard cap {HARD_MAX_SECONDS:g}")
        object.__setattr__(self, "max_seconds", seconds)


# Stable public name used by the individual RB-10 experiment modules.
ExperimentBudget = RB10Budget


@dataclass(frozen=True)
class ResourceEstimate:
    """Componentized allocation-free estimate retained with every result."""

    assignment_count: int
    node_count: int
    factor_count: int
    maximum_factor_arity: int
    component_bytes: Tuple[Tuple[str, int], ...]
    component_work: Tuple[Tuple[str, int], ...]
    estimated_peak_bytes: int
    estimated_work_units: int
    max_estimated_bytes: int
    max_work_units: int
    max_seconds: float
    max_assignments: int
    max_nodes: int
    max_factors: int
    max_factor_arity: int
    max_output_bytes: int
    streaming_assignments_not_materialized: bool
    measured_process_peak: bool
    estimate_is_process_rss: bool


def _normalize_components(
    values: object,
    name: str,
    hard_maximum: int,
) -> Tuple[Tuple[str, int], ...]:
    if type(values) is not tuple:
        raise RB10ValidationError(f"{name} must be a plain tuple")
    normalized = []
    names = set()
    for index, item in enumerate(values):
        if type(item) is not tuple or len(item) != 2:
            raise RB10ValidationError(f"{name}[{index}] must be a plain (name, value) tuple")
        component_name = _plain_string(
            item[0],
            f"{name}[{index}].name",
        )
        if component_name in names:
            raise RB10ValidationError(f"{name} contains duplicate component {component_name!r}")
        names.add(component_name)
        component_value = _plain_int(
            item[1],
            f"{name}[{index}].value",
            0,
            hard_maximum,
        )
        normalized.append((component_name, component_value))
    if not normalized:
        raise RB10ValidationError(f"{name} cannot be empty")
    return tuple(normalized)


def _normalize_factor_arities(
    factor_arities: object,
    *,
    node_count: int,
    budget: RB10Budget,
) -> Tuple[int, ...]:
    if type(factor_arities) is not tuple:
        raise RB10ValidationError("factor_arities must be a plain tuple")
    if len(factor_arities) > budget.max_factors:
        raise RB10ResourceError(f"factor count exceeds budget: {len(factor_arities)} > {budget.max_factors}")
    normalized = tuple(
        _plain_int(
            value,
            f"factor_arities[{index}]",
            1,
            HARD_MAX_FACTOR_ARITY,
        )
        for index, value in enumerate(factor_arities)
    )
    for index, arity in enumerate(normalized):
        if arity > budget.max_factor_arity:
            raise RB10ResourceError(
                f"factor arity exceeds budget at index {index}: {arity} > {budget.max_factor_arity}"
            )
        if arity > node_count:
            raise RB10ValidationError(f"factor_arities[{index}] cannot exceed node_count")
    return normalized


def preflight_resources(
    *,
    assignment_count: object,
    node_count: object,
    factor_arities: object,
    component_bytes: object,
    component_work: object,
    budget: RB10Budget = RB10Budget(),
    streaming_assignments_not_materialized: bool = True,
) -> ResourceEstimate:
    """Validate and admit one cell before normalization or experiment work.

    The scalar shape checks intentionally run before factor/component traversal.
    A hostile or oversized assignment count therefore cannot force inspection
    of caller containers after the cell is already known to be inadmissible.
    """

    if type(budget) is not RB10Budget:
        raise RB10ValidationError("budget must be exactly RB10Budget")
    if type(streaming_assignments_not_materialized) is not bool:
        raise RB10ValidationError("streaming_assignments_not_materialized must be a plain bool")
    assignments = _plain_int(
        assignment_count,
        "assignment_count",
        1,
        HARD_MAX_ASSIGNMENTS,
    )
    nodes = _plain_int(node_count, "node_count", 1, HARD_MAX_NODES)
    if assignments > budget.max_assignments:
        raise RB10ResourceError(f"assignment count exceeds budget: {assignments} > {budget.max_assignments}")
    if nodes > budget.max_nodes:
        raise RB10ResourceError(f"node count exceeds budget: {nodes} > {budget.max_nodes}")

    arities = _normalize_factor_arities(
        factor_arities,
        node_count=nodes,
        budget=budget,
    )
    byte_components = _normalize_components(
        component_bytes,
        "component_bytes",
        HARD_MAX_ESTIMATED_BYTES,
    )
    work_components = _normalize_components(
        component_work,
        "component_work",
        HARD_MAX_WORK_UNITS,
    )
    estimated_bytes = sum(value for _name, value in byte_components)
    estimated_work = sum(value for _name, value in work_components)
    if estimated_bytes > budget.max_estimated_bytes:
        raise RB10ResourceError(f"estimated peak exceeds byte budget: {estimated_bytes} > {budget.max_estimated_bytes}")
    if estimated_work > budget.max_work_units:
        raise RB10ResourceError(f"estimated work exceeds budget: {estimated_work} > {budget.max_work_units}")
    return ResourceEstimate(
        assignment_count=assignments,
        node_count=nodes,
        factor_count=len(arities),
        maximum_factor_arity=max(arities, default=0),
        component_bytes=byte_components,
        component_work=work_components,
        estimated_peak_bytes=estimated_bytes,
        estimated_work_units=estimated_work,
        max_estimated_bytes=budget.max_estimated_bytes,
        max_work_units=budget.max_work_units,
        max_seconds=budget.max_seconds,
        max_assignments=budget.max_assignments,
        max_nodes=budget.max_nodes,
        max_factors=budget.max_factors,
        max_factor_arity=budget.max_factor_arity,
        max_output_bytes=budget.max_output_bytes,
        streaming_assignments_not_materialized=(streaming_assignments_not_materialized),
        measured_process_peak=False,
        estimate_is_process_rss=False,
    )


@dataclass(frozen=True)
class MonotonicDeadline:
    """A cooperative monotonic deadline that is never evidence of preemption."""

    started_at: float
    expires_at: float
    max_seconds: float

    def __post_init__(self) -> None:
        for name, value in (
            ("started_at", self.started_at),
            ("expires_at", self.expires_at),
            ("max_seconds", self.max_seconds),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise RB10ValidationError(f"{name} must be a plain finite float")
        if self.started_at < 0.0:
            raise RB10ValidationError("started_at cannot be negative")
        if self.max_seconds <= 0.0 or self.max_seconds > HARD_MAX_SECONDS:
            raise RB10ValidationError("max_seconds is outside the immutable deadline envelope")
        if self.expires_at < self.started_at:
            raise RB10ValidationError("expires_at cannot precede started_at")
        expected = self.started_at + self.max_seconds
        if self.expires_at != expected:
            raise RB10ValidationError("expires_at must equal started_at + max_seconds")

    @classmethod
    def start(cls, budget: RB10Budget = RB10Budget()) -> "MonotonicDeadline":
        if type(budget) is not RB10Budget:
            raise RB10ValidationError("budget must be exactly RB10Budget")
        started = time.monotonic()
        return cls(
            started_at=started,
            expires_at=started + budget.max_seconds,
            max_seconds=budget.max_seconds,
        )

    def check(self, stage: str = "rb10_work") -> None:
        _plain_string(stage, "stage")
        if time.monotonic() > self.expires_at:
            raise RB10ResourceError(f"RB-10 deadline exceeded during {stage}")

    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())


# Stable public name used by experiment runners.
Deadline = MonotonicDeadline


def _validated_excluded_fields(values: object) -> Tuple[str, ...]:
    return _plain_string_tuple(
        values,
        "excluded_fields",
        allowed=ALLOWED_NONDETERMINISTIC_TIMING_FIELDS,
    )


def _jsonable_dataclass(value: object) -> Dict[str, object]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _normalize_json(
    value: object,
    *,
    excluded_fields: frozenset,
    seen: set,
    state: Dict[str, int],
    depth: int,
) -> object:
    if depth > HARD_MAX_JSON_DEPTH:
        raise RB10ValidationError(f"JSON value exceeds depth ceiling {HARD_MAX_JSON_DEPTH}")
    state["nodes"] += 1
    if state["nodes"] > HARD_MAX_JSON_NODES:
        raise RB10ValidationError(f"JSON value exceeds node ceiling {HARD_MAX_JSON_NODES}")

    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if value.bit_length() > 256:
            raise RB10ValidationError("JSON integer exceeds the 256-bit serialization ceiling")
        return value
    if type(value) is float:
        result = float.__float__(value)
        if not math.isfinite(result):
            raise RB10ValidationError("JSON floats must be finite")
        return result
    if type(value) is str:
        if len(value) > HARD_MAX_JSON_STRING_CHARS:
            raise RB10ValidationError("JSON string exceeds the hard character ceiling")
        return value

    if is_dataclass(value) and not isinstance(value, type):
        value = _jsonable_dataclass(value)

    if type(value) not in (dict, list, tuple):
        raise RB10ValidationError(f"unsupported JSON value type: {type(value).__name__}")
    identity = id(value)
    if identity in seen:
        raise RB10ValidationError("cyclic JSON containers are not permitted")
    seen.add(identity)
    try:
        if type(value) is dict:
            result_dict = {}
            for key, item in value.items():
                checked_key = _plain_string(
                    key,
                    "JSON object key",
                    allow_empty=True,
                    maximum_chars=256,
                )
                if checked_key in excluded_fields:
                    continue
                result_dict[checked_key] = _normalize_json(
                    item,
                    excluded_fields=excluded_fields,
                    seen=seen,
                    state=state,
                    depth=depth + 1,
                )
            return result_dict
        return [
            _normalize_json(
                item,
                excluded_fields=excluded_fields,
                seen=seen,
                state=state,
                depth=depth + 1,
            )
            for item in value
        ]
    finally:
        seen.remove(identity)


def canonical_projection(
    value: object,
    *,
    excluded_fields: Tuple[str, ...] = (DEFAULT_NONDETERMINISTIC_TIMING_FIELDS),
) -> object:
    """Return a bounded plain-JSON projection with approved timing fields removed."""

    checked_excluded = _validated_excluded_fields(excluded_fields)
    return _normalize_json(
        value,
        excluded_fields=frozenset(checked_excluded),
        seen=set(),
        state={"nodes": 0},
        depth=0,
    )


def _canonical_encoder() -> json.JSONEncoder:
    return json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _bounded_canonical_json(
    value: object,
    *,
    max_encoded_bytes: object,
    excluded_fields: Tuple[str, ...] = (),
    reserve_bytes: object = 0,
    label: str = "canonical JSON",
) -> str:
    """Encode incrementally and refuse before retaining an over-budget string."""

    encoded_cap = _plain_int(
        max_encoded_bytes,
        "max_encoded_bytes",
        1,
        HARD_MAX_OUTPUT_BYTES,
    )
    reserved = _plain_int(
        reserve_bytes,
        "reserve_bytes",
        0,
        encoded_cap,
    )
    checked_label = _plain_string(label, "label")
    projected = canonical_projection(
        value,
        excluded_fields=excluded_fields,
    )
    chunks = []
    encoded_bytes = reserved
    for chunk in _canonical_encoder().iterencode(projected):
        encoded_bytes += len(chunk.encode("utf-8"))
        if encoded_bytes > encoded_cap:
            raise RB10ResourceError(f"{checked_label} exceeds max_output_bytes")
        chunks.append(chunk)
    return "".join(chunks)


def _bounded_deterministic_digest(
    value: object,
    *,
    max_encoded_bytes: object,
    excluded_fields: Tuple[str, ...] = (DEFAULT_NONDETERMINISTIC_TIMING_FIELDS),
    label: str = "deterministic digest payload",
) -> str:
    """Hash canonical chunks without constructing an unbounded JSON string."""

    encoded_cap = _plain_int(
        max_encoded_bytes,
        "max_encoded_bytes",
        1,
        HARD_MAX_OUTPUT_BYTES,
    )
    checked_label = _plain_string(label, "label")
    projected = canonical_projection(
        value,
        excluded_fields=excluded_fields,
    )
    digest = hashlib.sha256()
    encoded_bytes = 0
    for chunk in _canonical_encoder().iterencode(projected):
        encoded = chunk.encode("utf-8")
        encoded_bytes += len(encoded)
        if encoded_bytes > encoded_cap:
            raise RB10ResourceError(f"{checked_label} exceeds max_output_bytes")
        digest.update(encoded)
    return digest.hexdigest()


def canonical_json(
    value: object,
    *,
    excluded_fields: Tuple[str, ...] = (DEFAULT_NONDETERMINISTIC_TIMING_FIELDS),
) -> str:
    """Serialize a deterministic bounded JSON projection."""

    projected = canonical_projection(
        value,
        excluded_fields=excluded_fields,
    )
    return _canonical_encoder().encode(projected)


def deterministic_digest(
    value: object,
    *,
    excluded_fields: Tuple[str, ...] = (DEFAULT_NONDETERMINISTIC_TIMING_FIELDS),
) -> str:
    """SHA-256 over canonical JSON after approved timing-field removal."""

    encoded = canonical_json(
        value,
        excluded_fields=excluded_fields,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_bytes(
    value: object,
    *,
    excluded_fields: Tuple[str, ...] = (DEFAULT_NONDETERMINISTIC_TIMING_FIELDS),
) -> bytes:
    """Return the UTF-8 bytes used by :func:`deterministic_digest`."""

    return canonical_json(
        value,
        excluded_fields=excluded_fields,
    ).encode("utf-8")


@dataclass(frozen=True)
class RunContract:
    """Immutable cross-lane description of one preregistered RB-10 run."""

    stage_id: str
    hypothesis_id: str
    decision_rule_id: str
    channel_id: str
    tie_policy_id: str
    control_ids: Tuple[str, ...]
    seeds: Tuple[int, ...]
    parameters_json: str
    budget: ExperimentBudget

    @classmethod
    def create(
        cls,
        *,
        stage_id: object,
        hypothesis_id: object,
        decision_rule_id: object,
        channel_id: object,
        tie_policy_id: object,
        control_ids: object,
        seeds: object,
        parameters: object,
        budget: ExperimentBudget = ExperimentBudget(),
    ) -> "RunContract":
        checked_controls = _plain_string_tuple(
            control_ids,
            "control_ids",
        )
        if type(seeds) is not tuple:
            raise RB10ValidationError("seeds must be a plain tuple")
        checked_seeds = tuple(
            _plain_int(
                seed,
                f"seeds[{index}]",
                0,
                (1 << 63) - 1,
            )
            for index, seed in enumerate(seeds)
        )
        if len(set(checked_seeds)) != len(checked_seeds):
            raise RB10ValidationError("seeds cannot contain duplicates")
        if type(budget) is not ExperimentBudget:
            raise RB10ValidationError("budget must be exactly ExperimentBudget")
        parameters_json = canonical_json(parameters, excluded_fields=())
        return cls(
            stage_id=_plain_string(stage_id, "stage_id"),
            hypothesis_id=_plain_string(
                hypothesis_id,
                "hypothesis_id",
            ),
            decision_rule_id=_plain_string(
                decision_rule_id,
                "decision_rule_id",
            ),
            channel_id=_plain_string(channel_id, "channel_id"),
            tie_policy_id=_plain_string(
                tie_policy_id,
                "tie_policy_id",
            ),
            control_ids=checked_controls,
            seeds=checked_seeds,
            parameters_json=parameters_json,
            budget=budget,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "hypothesis_id": self.hypothesis_id,
            "decision_rule_id": self.decision_rule_id,
            "channel_id": self.channel_id,
            "tie_policy_id": self.tie_policy_id,
            "control_ids": list(self.control_ids),
            "seeds": list(self.seeds),
            "parameters": json.loads(self.parameters_json),
            "budget": asdict(self.budget),
        }

    @classmethod
    def from_dict(cls, value: object) -> "RunContract":
        if type(value) is not dict:
            raise RB10IntegrityError("run contract must be a JSON object")
        required = {
            "stage_id",
            "hypothesis_id",
            "decision_rule_id",
            "channel_id",
            "tie_policy_id",
            "control_ids",
            "seeds",
            "parameters",
            "budget",
        }
        if set(value) != required:
            raise RB10IntegrityError("run-contract fields do not match the schema")
        budget_value = value["budget"]
        if type(budget_value) is not dict:
            raise RB10IntegrityError("run-contract budget must be a JSON object")
        budget_fields = {field.name for field in fields(ExperimentBudget)}
        if set(budget_value) != budget_fields:
            raise RB10IntegrityError("run-contract budget fields do not match the schema")
        if type(value["control_ids"]) is not list:
            raise RB10IntegrityError("run-contract control_ids must be a JSON list")
        if type(value["seeds"]) is not list:
            raise RB10IntegrityError("run-contract seeds must be a JSON list")
        try:
            budget = ExperimentBudget(**budget_value)
            return cls.create(
                stage_id=value["stage_id"],
                hypothesis_id=value["hypothesis_id"],
                decision_rule_id=value["decision_rule_id"],
                channel_id=value["channel_id"],
                tie_policy_id=value["tie_policy_id"],
                control_ids=tuple(value["control_ids"]),
                seeds=tuple(value["seeds"]),
                parameters=value["parameters"],
                budget=budget,
            )
        except (RB10ValidationError, TypeError) as exc:
            raise RB10IntegrityError("run contract violates the RB-10 schema") from exc

    @property
    def digest(self) -> str:
        return deterministic_digest(
            self.as_dict(),
            excluded_fields=(),
        )


def _resource_estimate_as_dict(
    estimate: ResourceEstimate,
) -> Dict[str, object]:
    if type(estimate) is not ResourceEstimate:
        raise RB10ValidationError("resource_estimate must be exactly ResourceEstimate")
    return asdict(estimate)


def _resource_estimate_from_dict(value: object) -> ResourceEstimate:
    if type(value) is not dict:
        raise RB10IntegrityError("resource_estimate must be a JSON object")
    required = {field.name for field in fields(ResourceEstimate)}
    if set(value) != required:
        raise RB10IntegrityError("resource_estimate fields do not match the schema")
    try:
        byte_components = tuple(tuple(item) for item in value["component_bytes"])
        work_components = tuple(tuple(item) for item in value["component_work"])
        estimate = ResourceEstimate(
            assignment_count=value["assignment_count"],
            node_count=value["node_count"],
            factor_count=value["factor_count"],
            maximum_factor_arity=value["maximum_factor_arity"],
            component_bytes=byte_components,
            component_work=work_components,
            estimated_peak_bytes=value["estimated_peak_bytes"],
            estimated_work_units=value["estimated_work_units"],
            max_estimated_bytes=value["max_estimated_bytes"],
            max_work_units=value["max_work_units"],
            max_seconds=value["max_seconds"],
            max_assignments=value["max_assignments"],
            max_nodes=value["max_nodes"],
            max_factors=value["max_factors"],
            max_factor_arity=value["max_factor_arity"],
            max_output_bytes=value["max_output_bytes"],
            streaming_assignments_not_materialized=value["streaming_assignments_not_materialized"],
            measured_process_peak=value["measured_process_peak"],
            estimate_is_process_rss=value["estimate_is_process_rss"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RB10IntegrityError("resource_estimate cannot be reconstructed") from exc
    try:
        _validate_resource_estimate(estimate)
    except (RB10ValidationError, RB10ResourceError) as exc:
        raise RB10IntegrityError("resource_estimate violates the immutable RB-10 envelope") from exc
    return estimate


def _validate_resource_estimate(estimate: ResourceEstimate) -> None:
    if type(estimate) is not ResourceEstimate:
        raise RB10IntegrityError("resource_estimate must be exactly ResourceEstimate")
    budget = RB10Budget(
        max_estimated_bytes=estimate.max_estimated_bytes,
        max_work_units=estimate.max_work_units,
        max_seconds=estimate.max_seconds,
        max_assignments=estimate.max_assignments,
        max_nodes=estimate.max_nodes,
        max_factors=estimate.max_factors,
        max_factor_arity=estimate.max_factor_arity,
        max_output_bytes=estimate.max_output_bytes,
    )
    rebuilt = preflight_resources(
        assignment_count=estimate.assignment_count,
        node_count=estimate.node_count,
        factor_arities=tuple(estimate.maximum_factor_arity for _ in range(estimate.factor_count)),
        component_bytes=estimate.component_bytes,
        component_work=estimate.component_work,
        budget=budget,
        streaming_assignments_not_materialized=(estimate.streaming_assignments_not_materialized),
    )
    comparable_fields = (
        "assignment_count",
        "node_count",
        "factor_count",
        "maximum_factor_arity",
        "component_bytes",
        "component_work",
        "estimated_peak_bytes",
        "estimated_work_units",
        "max_estimated_bytes",
        "max_work_units",
        "max_seconds",
        "max_assignments",
        "max_nodes",
        "max_factors",
        "max_factor_arity",
        "max_output_bytes",
        "streaming_assignments_not_materialized",
        "measured_process_peak",
        "estimate_is_process_rss",
    )
    for name in comparable_fields:
        if getattr(estimate, name) != getattr(rebuilt, name):
            raise RB10IntegrityError(f"resource_estimate field {name} is inconsistent")


def _artifact_digest_payload(payload: Mapping[str, object]) -> Dict[str, object]:
    result = dict(payload)
    result.pop("artifact_digest", None)
    return result


@dataclass(frozen=True)
class ArtifactEnvelope:
    """Small immutable envelope for one bounded RB-10 cell or stage."""

    stage_id: str
    status: str
    run_contract_json: str
    run_contract_digest: str
    resource_estimate: ResourceEstimate
    result_json: str
    limitations: Tuple[str, ...]
    refusal_reasons: Tuple[str, ...]
    nondeterministic_timing_fields: Tuple[str, ...]
    artifact_digest: str

    @classmethod
    def create(
        cls,
        *,
        stage_id: object,
        status: object,
        run_contract: object,
        resource_estimate: ResourceEstimate,
        result: object,
        limitations: object = (),
        refusal_reasons: object = (),
        nondeterministic_timing_fields: object = (DEFAULT_NONDETERMINISTIC_TIMING_FIELDS),
    ) -> "ArtifactEnvelope":
        checked_stage = _plain_string(stage_id, "stage_id")
        checked_status = _plain_string(status, "status")
        if checked_status not in ALLOWED_STATUSES:
            raise RB10ValidationError("status must be one of " + ", ".join(ALLOWED_STATUSES))
        checked_limitations = _plain_string_tuple(
            limitations,
            "limitations",
        )
        checked_refusals = _plain_string_tuple(
            refusal_reasons,
            "refusal_reasons",
        )
        if checked_status == "REFUSED" and not checked_refusals:
            raise RB10ValidationError("REFUSED status requires at least one refusal reason")
        if checked_status != "REFUSED" and checked_refusals:
            raise RB10ValidationError("refusal reasons are permitted only for REFUSED status")
        checked_nondeterministic = _plain_string_tuple(
            nondeterministic_timing_fields,
            "nondeterministic_timing_fields",
            allowed=ALLOWED_NONDETERMINISTIC_TIMING_FIELDS,
        )
        _validate_resource_estimate(resource_estimate)
        if type(run_contract) is not RunContract:
            raise RB10ValidationError("run_contract must be exactly RunContract")
        try:
            checked_run_contract = RunContract.from_dict(run_contract.as_dict())
        except (RB10IntegrityError, TypeError, ValueError) as exc:
            raise RB10ValidationError("run_contract does not satisfy its schema") from exc
        if checked_run_contract != run_contract:
            raise RB10ValidationError("run_contract does not match its canonical form")
        if run_contract.stage_id != checked_stage:
            raise RB10ValidationError("run contract stage_id does not match artifact stage_id")
        budget_and_guard = (
            (
                run_contract.budget.max_estimated_bytes,
                resource_estimate.max_estimated_bytes,
            ),
            (
                run_contract.budget.max_work_units,
                resource_estimate.max_work_units,
            ),
            (
                run_contract.budget.max_seconds,
                resource_estimate.max_seconds,
            ),
            (
                run_contract.budget.max_assignments,
                resource_estimate.max_assignments,
            ),
            (
                run_contract.budget.max_nodes,
                resource_estimate.max_nodes,
            ),
            (
                run_contract.budget.max_factors,
                resource_estimate.max_factors,
            ),
            (
                run_contract.budget.max_factor_arity,
                resource_estimate.max_factor_arity,
            ),
            (
                run_contract.budget.max_output_bytes,
                resource_estimate.max_output_bytes,
            ),
        )
        if any(contract_value != guard_value for contract_value, guard_value in budget_and_guard):
            raise RB10ValidationError("run-contract budget does not match the resource guard")
        output_cap = run_contract.budget.max_output_bytes
        run_contract_dict = run_contract.as_dict()
        run_contract_json = _bounded_canonical_json(
            run_contract_dict,
            max_encoded_bytes=output_cap,
            excluded_fields=(),
            label="canonical run contract",
        )
        result_json = _bounded_canonical_json(
            result,
            max_encoded_bytes=output_cap,
            excluded_fields=(),
            label="canonical result",
        )
        run_contract_digest = hashlib.sha256(run_contract_json.encode("utf-8")).hexdigest()
        provisional = cls(
            stage_id=checked_stage,
            status=checked_status,
            run_contract_json=run_contract_json,
            run_contract_digest=run_contract_digest,
            resource_estimate=resource_estimate,
            result_json=result_json,
            limitations=checked_limitations,
            refusal_reasons=checked_refusals,
            nondeterministic_timing_fields=checked_nondeterministic,
            artifact_digest="",
        )
        payload = provisional.as_dict()
        artifact_digest = _bounded_deterministic_digest(
            _artifact_digest_payload(payload),
            max_encoded_bytes=output_cap,
            excluded_fields=checked_nondeterministic,
            label="canonical artifact digest payload",
        )
        completed = cls(
            stage_id=provisional.stage_id,
            status=provisional.status,
            run_contract_json=provisional.run_contract_json,
            run_contract_digest=provisional.run_contract_digest,
            resource_estimate=provisional.resource_estimate,
            result_json=provisional.result_json,
            limitations=provisional.limitations,
            refusal_reasons=provisional.refusal_reasons,
            nondeterministic_timing_fields=(provisional.nondeterministic_timing_fields),
            artifact_digest=artifact_digest,
        )
        _bounded_canonical_json(
            completed.as_dict(),
            max_encoded_bytes=output_cap,
            excluded_fields=(),
            reserve_bytes=1,
            label="final artifact envelope",
        )
        return completed

    def as_dict(self) -> Dict[str, object]:
        return {
            "record_type": RECORD_TYPE,
            "protocol_id": PROTOCOL_ID,
            "schema_version": SCHEMA_VERSION,
            "evidence_mode": EVIDENCE_MODE,
            "stage_id": self.stage_id,
            "status": self.status,
            "promotion_eligible": False,
            "novelty_claim": False,
            "status_boundary": STATUS_BOUNDARY,
            "execution_evidence_claim": False,
            "run_contract": json.loads(self.run_contract_json),
            "run_contract_digest": self.run_contract_digest,
            "resource_estimate": _resource_estimate_as_dict(self.resource_estimate),
            "result": json.loads(self.result_json),
            "limitations": list(self.limitations),
            "refusal_reasons": list(self.refusal_reasons),
            "nondeterministic_timing_fields": list(self.nondeterministic_timing_fields),
            "artifact_digest": self.artifact_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ArtifactEnvelope":
        if type(value) is not dict:
            raise RB10IntegrityError("artifact must be a plain JSON object")
        required = {
            "record_type",
            "protocol_id",
            "schema_version",
            "evidence_mode",
            "stage_id",
            "status",
            "promotion_eligible",
            "novelty_claim",
            "status_boundary",
            "execution_evidence_claim",
            "run_contract",
            "run_contract_digest",
            "resource_estimate",
            "result",
            "limitations",
            "refusal_reasons",
            "nondeterministic_timing_fields",
            "artifact_digest",
        }
        if set(value) != required:
            raise RB10IntegrityError("artifact fields do not match the RB-10 envelope schema")
        if (
            value["record_type"] != RECORD_TYPE
            or value["protocol_id"] != PROTOCOL_ID
            or value["schema_version"] != SCHEMA_VERSION
            or value["evidence_mode"] != EVIDENCE_MODE
        ):
            raise RB10IntegrityError("artifact protocol metadata does not match this implementation")
        if value["promotion_eligible"] is not False or value["novelty_claim"] is not False:
            raise RB10IntegrityError("RB-10 METHOD_DEV artifacts cannot claim promotion or novelty")
        if value["status_boundary"] != STATUS_BOUNDARY or value["execution_evidence_claim"] is not False:
            raise RB10IntegrityError("RB-10 workflow status cannot claim independent runtime evidence")
        if type(value["limitations"]) is not list:
            raise RB10IntegrityError("artifact limitations must be a JSON list")
        if type(value["refusal_reasons"]) is not list:
            raise RB10IntegrityError("artifact refusal_reasons must be a JSON list")
        if type(value["nondeterministic_timing_fields"]) is not list:
            raise RB10IntegrityError("artifact nondeterministic fields must be a JSON list")
        estimate = _resource_estimate_from_dict(value["resource_estimate"])
        run_contract = RunContract.from_dict(value["run_contract"])
        try:
            rebuilt = cls.create(
                stage_id=value["stage_id"],
                status=value["status"],
                run_contract=run_contract,
                resource_estimate=estimate,
                result=value["result"],
                limitations=tuple(value["limitations"]),
                refusal_reasons=tuple(value["refusal_reasons"]),
                nondeterministic_timing_fields=tuple(value["nondeterministic_timing_fields"]),
            )
        except (RB10ValidationError, RB10ResourceError) as exc:
            raise RB10IntegrityError("artifact content violates the RB-10 contract") from exc
        if value["run_contract_digest"] != rebuilt.run_contract_digest:
            raise RB10IntegrityError("artifact run-contract digest mismatch")
        if value["artifact_digest"] != rebuilt.artifact_digest:
            raise RB10IntegrityError("artifact deterministic digest mismatch")
        return rebuilt


def validate_artifact_envelope(value: object) -> bool:
    """Rebuild and validate an in-memory or JSON-loaded artifact envelope."""

    if type(value) is ArtifactEnvelope:
        ArtifactEnvelope.from_dict(value.as_dict())
        return True
    ArtifactEnvelope.from_dict(value)
    return True


def preflight_experiment(
    *,
    assignment_count: object,
    node_count: object,
    factor_arities: object,
    component_bytes: object,
    component_work: object,
    budget: ExperimentBudget = ExperimentBudget(),
    streaming_assignments_not_materialized: bool = True,
) -> ResourceEstimate:
    """Stable public wrapper for the allocation-free RB-10 preflight."""

    return preflight_resources(
        assignment_count=assignment_count,
        node_count=node_count,
        factor_arities=factor_arities,
        component_bytes=component_bytes,
        component_work=component_work,
        budget=budget,
        streaming_assignments_not_materialized=(streaming_assignments_not_materialized),
    )


def atomic_write_json(
    output: object,
    payload: object,
    *,
    max_encoded_bytes: object = DEFAULT_MAX_OUTPUT_BYTES,
    deadline: Optional[MonotonicDeadline] = None,
) -> None:
    """Stream bounded JSON through a same-directory fsync and atomic replace."""

    if type(output) is str:
        output_path = Path(output)
    elif isinstance(output, Path):
        output_path = output
    else:
        raise RB10ValidationError("output must be a string or pathlib.Path")
    output_text = str(output_path)
    if not output_text:
        raise RB10ValidationError("output path cannot be empty")
    if len(output_text) > HARD_MAX_PATH_CHARS:
        raise RB10ValidationError(f"output path exceeds the {HARD_MAX_PATH_CHARS}-character ceiling")
    caller_cap = _plain_int(
        max_encoded_bytes,
        "max_encoded_bytes",
        1,
        HARD_MAX_OUTPUT_BYTES,
    )
    if deadline is not None and type(deadline) is not MonotonicDeadline:
        raise RB10ValidationError("deadline must be exactly MonotonicDeadline")
    if deadline is not None:
        deadline.check("before_artifact_normalization")
    if type(payload) is ArtifactEnvelope:
        validate_artifact_envelope(payload)
        encoded_cap = min(
            caller_cap,
            payload.resource_estimate.max_output_bytes,
        )
        normalized = canonical_projection(
            payload.as_dict(),
            excluded_fields=(),
        )
    else:
        encoded_cap = caller_cap
        normalized = canonical_projection(payload, excluded_fields=())
    if deadline is not None:
        deadline.check("after_artifact_normalization")

    if output_path.exists() and output_path.is_dir():
        raise RB10ValidationError("output path must name a file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(output_path.parent),
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            encoder = _canonical_encoder()
            encoded_bytes = 0
            for chunk in encoder.iterencode(normalized):
                if deadline is not None:
                    deadline.check("artifact_encoding")
                encoded = chunk.encode("utf-8")
                encoded_bytes += len(encoded)
                if encoded_bytes + 1 > encoded_cap:
                    raise RB10ResourceError("encoded JSON artifact exceeds max_encoded_bytes")
                temporary.write(encoded)
            if encoded_bytes + 1 > encoded_cap:
                raise RB10ResourceError("encoded JSON artifact exceeds max_encoded_bytes")
            temporary.write(b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        if deadline is not None:
            deadline.check("before_artifact_atomic_replace")
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "ALLOWED_NONDETERMINISTIC_TIMING_FIELDS",
    "ALLOWED_STATUSES",
    "ArtifactEnvelope",
    "Deadline",
    "DEFAULT_MAX_ASSIGNMENTS",
    "DEFAULT_MAX_ESTIMATED_BYTES",
    "DEFAULT_MAX_FACTOR_ARITY",
    "DEFAULT_MAX_FACTORS",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_WORK_UNITS",
    "DEFAULT_NONDETERMINISTIC_TIMING_FIELDS",
    "EVIDENCE_MODE",
    "ExperimentBudget",
    "HARD_MAX_ASSIGNMENTS",
    "HARD_MAX_ESTIMATED_BYTES",
    "HARD_MAX_FACTOR_ARITY",
    "HARD_MAX_FACTORS",
    "HARD_MAX_NODES",
    "HARD_MAX_OUTPUT_BYTES",
    "HARD_MAX_SECONDS",
    "HARD_MAX_WORK_UNITS",
    "MonotonicDeadline",
    "PROTOCOL_ID",
    "RB10Budget",
    "RB10IntegrityError",
    "RB10ResourceError",
    "RB10ValidationError",
    "RECORD_TYPE",
    "ResourceEstimate",
    "RunContract",
    "SCHEMA_VERSION",
    "STATUS_BOUNDARY",
    "atomic_write_json",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_projection",
    "deterministic_digest",
    "preflight_resources",
    "preflight_experiment",
    "validate_artifact_envelope",
]
