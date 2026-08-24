"""Deterministic, CPU-only RB-13 Wave 0A method-development experiments.

The module is isolated from production mutation paths. PRW-EK0 observes the
current memory implementation through temporary stores; the other cards use
bounded synthetic fixtures. Results are characterization artifacts, never
scientific, production, or novelty claims.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROTOCOL_ID = "CHELATEDAI-RB13-WAVE0A-v8"
FROZEN_PROTOCOL_SHA256 = "2fa9b950be0c68cea82e0922dbeb452015773e42efc352d8681f5dc88b9026aa"
ARTIFACT_SCHEMA_VERSION = 1
STAGE_IDS = ("PRW-EK0", "PRW-BIL1", "PRW-SPU0", "PRW-VAR1")
SELECT_SEEDS = (1301, 1303, 1307)
REPORT_SEEDS = (2309, 2311, 2333)
DEFAULT_MAX_BYTES = 768 * 1024 * 1024
DEFAULT_MAX_SECONDS = 120.0
MIN_ESTIMATED_BYTES = 16 * 1024 * 1024
MIN_OUTPUT_BYTES = 16 * 1024
STAGE_MAX_BYTES = {
    "PRW-EK0": 768 * 1024 * 1024,
    "PRW-BIL1": 256 * 1024 * 1024,
    "PRW-SPU0": 256 * 1024 * 1024,
    "PRW-VAR1": 768 * 1024 * 1024,
}


class Wave0AValidationError(ValueError):
    """Raised when a frozen fixture or invocation is malformed."""


class Wave0AResourceError(RuntimeError):
    """Raised before work when the modeled resource contract is exceeded."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_payload(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, object], max_bytes: int) -> Path:
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    if len(encoded) > max_bytes:
        raise Wave0AResourceError("artifact exceeds the frozen output-byte ceiling")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class Wave0ABudget:
    max_estimated_bytes: int = DEFAULT_MAX_BYTES
    max_seconds_per_stage: float = DEFAULT_MAX_SECONDS
    max_output_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if type(self.max_estimated_bytes) is not int or not (
            MIN_ESTIMATED_BYTES <= self.max_estimated_bytes <= 2 * 1024**3
        ):
            raise Wave0AValidationError(
                "max_estimated_bytes must be a plain integer in [16 MiB, 2 GiB]"
            )
        if type(self.max_seconds_per_stage) is not float:
            raise Wave0AValidationError("max_seconds_per_stage must be a plain float")
        if not 0.01 <= self.max_seconds_per_stage <= 480.0:
            raise Wave0AValidationError("max_seconds_per_stage must be in [0.01, 480]")
        if type(self.max_output_bytes) is not int or not (
            MIN_OUTPUT_BYTES <= self.max_output_bytes <= 64 * 1024**2
        ):
            raise Wave0AValidationError(
                "max_output_bytes must be a plain integer in [16 KiB, 64 MiB]"
            )


def current_rss_bytes() -> int:
    """Return this single-process working set without adding a dependency."""

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        get_process_memory_info.restype = wintypes.BOOL
        if get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        raise Wave0AResourceError("unable to measure the Windows process working set")
    statm = Path("/proc/self/statm")
    if statm.is_file():
        fields = statm.read_text(encoding="ascii").split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    raise Wave0AResourceError("this platform has no supported live RSS measurement")


def _preflight(stage_id: str, estimated_bytes: int, work_units: int, budget: Wave0ABudget) -> Dict[str, int]:
    if stage_id not in STAGE_IDS:
        raise Wave0AValidationError(f"unknown stage: {stage_id}")
    stage_ceiling = min(budget.max_estimated_bytes, STAGE_MAX_BYTES[stage_id])
    rss_at_preflight = current_rss_bytes()
    modeled_peak = rss_at_preflight + estimated_bytes
    if modeled_peak > stage_ceiling:
        raise Wave0AResourceError(
            f"{stage_id} modeled peak {modeled_peak} exceeds {stage_ceiling} bytes"
        )
    if estimated_bytes <= 0 or work_units <= 0:
        raise Wave0AValidationError("resource estimates must be positive")
    return {
        "estimated_incremental_bytes": int(estimated_bytes),
        "rss_at_preflight_bytes": rss_at_preflight,
        "modeled_peak_bytes": modeled_peak,
        "stage_ceiling_bytes": stage_ceiling,
        "work_units": int(work_units),
    }


def _deadline(budget: Wave0ABudget) -> float:
    return time.monotonic() + float(budget.max_seconds_per_stage)


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() > deadline:
        raise Wave0AResourceError(f"deadline exceeded during {stage}")


def _check_live_rss(resource: Mapping[str, int], stage: str) -> None:
    rss = current_rss_bytes()
    if rss > int(resource["stage_ceiling_bytes"]):
        raise Wave0AResourceError(
            f"live RSS {rss} exceeds {resource['stage_ceiling_bytes']} bytes during {stage}"
        )


def _minimal_artifact(prompt_hash: str) -> Dict[str, object]:
    return {
        "capture": {"prompt_hash": prompt_hash, "metadata": {"fixture": "rb13-wave0a"}},
        "runtime": {"model_name": "synthetic-contract-fixture", "device": "cpu"},
    }


def run_ek0(*, budget: Wave0ABudget = Wave0ABudget()) -> Dict[str, object]:
    """Characterize current memory behavior without changing the implementation."""

    # This import transitively loads Torch.  It is intentionally part of the
    # measured process baseline rather than hidden behind the incremental model.
    from model_scope_memory import (
        MemorySegmentConfig,
        ModelScopeMemoryStore,
        PersistentMemory,
    )

    resource = _preflight("PRW-EK0", 4 * 1024 * 1024, 64, budget)
    deadline = _deadline(budget)
    _check_live_rss(resource, "ek0_production_import")
    configs = [
        MemorySegmentConfig(name=name, max_entries=2, allow_promotion=name == "episode")
        for name in ("working", "episode", "expectation", "persistent")
    ]
    store = ModelScopeMemoryStore(configs)
    store.record_observation(_minimal_artifact("ek0-a"), query_text="alpha")
    second = store.record_observation(_minimal_artifact("ek0-b"), query_text="beta")
    store.record_observation(_minimal_artifact("ek0-c"), query_text="gamma")
    _check_deadline(deadline, "ek0_overflow")
    episode_hashes = [entry["payload"]["query_hash"] for entry in store.get_segment_entries("episode")]
    before_count = len(store.get_segment_entries("episode"))
    annotated = store.annotate_entry(
        "episode", second["episode_entry_id"], update={"characterization_label": "mutated"}
    )
    after_count = len(store.get_segment_entries("episode"))
    direct = store.record_observation(
        _minimal_artifact("ek0-direct"), query_text="direct", promote=True
    )
    missing_status = store.promote_episode(
        direct["episode_entry_id"], reason="characterization_missing_status"
    )
    blocked = store.promote_episode(
        direct["episode_entry_id"],
        reason="characterization_negative_status",
        promotion_status={"promotion_ready": False, "reasons": ["frozen_negative"]},
    )
    _check_deadline(deadline, "ek0_promotion")
    with tempfile.TemporaryDirectory(prefix="rb13-ek0-") as directory:
        base = Path(directory)
        persistent = PersistentMemory(base)
        persistent.save("a", {"version": 1})
        persistent.save("b", {"version": 1})
        persistent.save("a", {"version": 2})
        restarted = PersistentMemory(base)
        reloaded = restarted.load("a")
        keys_before_delete = sorted(restarted.list_keys())
        deleted = restarted.delete("b")
        keys_after_delete = sorted(PersistentMemory(base).list_keys())
        saved_files = len(list(base.glob("*.json")))
    observations = {
        "fifo_overflow": {
            "observed": episode_hashes == ["ek0-b", "ek0-c"],
            "classification": "MIGRATION_GAP",
        },
        "annotation_mutates_payload": {
            "observed": before_count == after_count
            and annotated["payload"].get("characterization_label") == "mutated",
            "classification": "MIGRATION_GAP",
        },
        "direct_promote_bypasses_gate": {
            "observed": direct["persistent_entry_id"] is not None,
            "classification": "BUG",
        },
        "missing_status_promotes": {
            "observed": bool(missing_status["promoted"]),
            "classification": "BUG",
        },
        "negative_status_fails_closed": {
            "observed": not bool(blocked["promoted"]),
            "classification": "CURRENT_CONTRACT",
        },
        "persistent_save_overwrites_key": {
            "observed": reloaded is not None
            and reloaded.value == {"version": 2}
            and keys_before_delete == ["a", "b"],
            "classification": "MIGRATION_GAP",
        },
        "persistent_delete_removes_key": {
            "observed": deleted and keys_after_delete == ["a"] and saved_files == 1,
            "classification": "MIGRATION_GAP",
        },
        "restart_loads_latest_value": {
            "observed": reloaded is not None and reloaded.value == {"version": 2},
            "classification": "CURRENT_CONTRACT",
        },
    }
    null_survives = not any(
        bool(item["observed"]) and item["classification"] in ("BUG", "MIGRATION_GAP")
        for item in observations.values()
    )
    _check_live_rss(resource, "ek0_complete")
    return {
        "stage_id": "PRW-EK0",
        "evidence_state": "CHARACTERIZED",
        "scientific_claim_status": "UNCONFIRMED",
        "novelty_status": "UNCONFIRMED",
        "null_survives": null_survives,
        "disposition": "NULL_SURVIVES" if null_survives else "REQUIRE_COMPATIBILITY_AND_MIGRATION_DESIGN",
        "observations": observations,
        "resource_estimate": resource,
        "limitations": ["current code contract only", "temporary stores only", "no production mutation"],
    }


def _state(support: bool, refute: bool) -> str:
    return {
        (False, False): "UNKNOWN",
        (True, False): "SUPPORTED",
        (False, True): "REFUTED",
        (True, True): "CONFLICTED",
    }[(support, refute)]


def _bil1_events() -> Tuple[Dict[str, object], ...]:
    return (
        {"id": "e1", "lineage": "l1", "polarity": "support", "tx": 1, "valid_from": 1, "valid_to": 9},
        {"id": "e2", "lineage": "l1", "polarity": "support", "tx": 2, "valid_from": 1, "valid_to": 9},
        {"id": "e3", "lineage": "l2", "polarity": "refute", "tx": 3, "valid_from": 3, "valid_to": 7},
        {"id": "e4", "lineage": "l3", "polarity": "support", "tx": 4, "valid_from": 6, "valid_to": 12},
        {"id": "e5", "lineage": "l4", "polarity": "refute", "tx": 5, "valid_from": 8, "valid_to": 12},
        {"id": "e3", "lineage": "l2", "polarity": "refute", "tx": 6, "valid_from": 3, "valid_to": 7, "retracted": True},
    )


def _active_events(events: Iterable[Mapping[str, object]], valid_at: int, tx_at: int) -> List[Mapping[str, object]]:
    latest: Dict[str, Mapping[str, object]] = {}
    for event in events:
        if int(event["tx"]) <= tx_at:
            current = latest.get(str(event["id"]))
            if current is None or int(event["tx"]) >= int(current["tx"]):
                latest[str(event["id"])] = event
    return [
        event
        for event in latest.values()
        if not bool(event.get("retracted", False))
        and int(event["valid_from"]) <= valid_at < int(event["valid_to"])
    ]


def _candidate_merge(events: Sequence[Mapping[str, object]]) -> Tuple[str, int]:
    facts = {(str(event["lineage"]), str(event["polarity"])) for event in events}
    support = any(polarity == "support" for _, polarity in facts)
    refute = any(polarity == "refute" for _, polarity in facts)
    return _state(support, refute), len(events) + len(facts)


def _bil1_controls(events: Sequence[Mapping[str, object]]) -> Dict[str, str]:
    lineage_facts = {(str(event["lineage"]), str(event["polarity"])) for event in events}
    scalar = sum(1 if polarity == "support" else -1 for _, polarity in lineage_facts)
    scalar_state = "SUPPORTED" if scalar > 0 else "REFUTED" if scalar < 0 else "UNKNOWN"
    if events:
        last = max(events, key=lambda event: (int(event["tx"]), str(event["id"])))
        lww = "SUPPORTED" if last["polarity"] == "support" else "REFUTED"
    else:
        lww = "UNKNOWN"
    overwrite_support = False
    overwrite_refute = False
    for event in events:
        overwrite_support = event["polarity"] == "support"
        overwrite_refute = event["polarity"] == "refute"
    union_support = any(event["polarity"] == "support" for event in events)
    union_refute = any(event["polarity"] == "refute" for event in events)
    return {
        "scalar": scalar_state,
        "last_write_wins": lww,
        "paired_overwrite": _state(overwrite_support, overwrite_refute),
        "paired_boolean_union": _state(union_support, union_refute),
    }


def _fixed_point(
    nodes: Sequence[str], edges: Sequence[Tuple[str, str]], initial: Mapping[str, set]
) -> Tuple[Dict[str, set], int, int]:
    state = {node: set(initial.get(node, set())) for node in nodes}
    operations = 0
    for iteration in range(len(nodes) * max(len(edges), 1) + 1):
        changed = False
        for source, target in edges:
            operations += len(state[source]) + 1
            merged = state[target] | state[source]
            if merged != state[target]:
                state[target] = merged
                changed = True
        if not changed:
            return state, iteration + 1, operations
    raise Wave0AValidationError("finite monotone propagation did not converge")


def run_bil1(*, seed: int, budget: Wave0ABudget = Wave0ABudget()) -> Dict[str, object]:
    events = _bil1_events()
    resource = _preflight("PRW-BIL1", 8 * 1024 * 1024, 100_000, budget)
    deadline = _deadline(budget)
    rng = np.random.default_rng(seed)
    views = ((2, 2), (4, 4), (6, 6), (8, 8), (10, 10))
    rows: List[Dict[str, object]] = []
    permutation_invariant = True
    duplicate_idempotent = True
    unknown_conflict_distinct = _state(False, False) != _state(True, True)
    for valid_at, tx_at in views:
        active = _active_events(events, valid_at, tx_at)
        oracle, operations = _candidate_merge(active)
        states = set()
        for _ in range(16):
            permutation = list(active)
            rng.shuffle(permutation)
            states.add(_candidate_merge(permutation)[0])
        permutation_invariant &= states == {oracle}
        deduplicated = list({(event["lineage"], event["polarity"]): event for event in active}.values())
        duplicate_idempotent &= _candidate_merge(active)[0] == _candidate_merge(deduplicated)[0]
        rows.append(
            {
                "valid_at": valid_at,
                "transaction_at": tx_at,
                "candidate": oracle,
                "controls": _bil1_controls(active),
                "active_event_count": len(active),
                "operations": operations,
            }
        )
        _check_deadline(deadline, "bil1_temporal_views")
    nodes = ("a", "b", "c", "d")
    edges = (("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"))
    initial = {"a": {("l1", "support")}, "c": {("l2", "refute")}}
    first, first_iterations, first_operations = _fixed_point(nodes, edges, initial)
    second, second_iterations, second_operations = _fixed_point(
        tuple(reversed(nodes)), tuple(reversed(edges)), initial
    )
    fixed_point_equal = first == second
    candidate_states = [row["candidate"] for row in rows]
    union_states = [row["controls"]["paired_boolean_union"] for row in rows]
    invariants = {
        "permutation_invariant": permutation_invariant,
        "lineage_duplicate_idempotent": duplicate_idempotent,
        "unknown_conflict_distinct": unknown_conflict_distinct,
        "fixed_point_converges": first_iterations <= 17 and second_iterations <= 17,
        "fixed_point_order_invariant": fixed_point_equal,
        "temporal_oracle_matches": candidate_states == ["SUPPORTED", "CONFLICTED", "SUPPORTED", "CONFLICTED", "CONFLICTED"],
    }
    passed = all(invariants.values())
    reduced = candidate_states == union_states and passed
    _check_live_rss(resource, "bil1_complete")
    return {
        "stage_id": "PRW-BIL1",
        "seed": seed,
        "evidence_state": "COMPLETE",
        "scientific_claim_status": "UNCONFIRMED",
        "novelty_status": "UNCONFIRMED",
        "rows": rows,
        "invariants": invariants,
        "false_promotion_count": sum(row["candidate"] == "SUPPORTED" and row["controls"]["paired_boolean_union"] != "SUPPORTED" for row in rows),
        "fixed_point": {
            "iterations": [first_iterations, second_iterations],
            "operations": first_operations + second_operations,
        },
        "disposition": "REDUCE_TO_PAIRED_BOOLEAN_UNION" if reduced else "PASS_SEMANTICS" if passed else "KILL_CANDIDATE_SEMANTICS",
        "resource_estimate": resource,
        "limitations": ["synthetic finite events", "no production evidence graph"],
    }


def _pairwise_distances(vectors: np.ndarray) -> np.ndarray:
    differences = vectors[:, None, :] - vectors[None, :, :]
    return np.linalg.norm(differences, axis=2)


def _stable_neighbor_ranks(distances: np.ndarray) -> List[List[int]]:
    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)
    return [np.argsort(row, kind="stable").astype(int).tolist() for row in distances]


def _relative_error(absolute_error: float, reference: np.ndarray) -> float:
    denominator = float(np.max(np.abs(reference)))
    if denominator == 0.0:
        return 0.0 if absolute_error == 0.0 else math.inf
    return absolute_error / denominator


def _spu0_accounting(dimension: int, factor_count: int, vector_count: int) -> Dict[str, int]:
    dense_multiplications = vector_count * dimension**2
    dense_additions = vector_count * dimension * (dimension - 1)
    factor_multiplications = factor_count * dense_multiplications
    factor_additions = factor_count * dense_additions
    distance_work = 3 * vector_count**2 * max(3 * dimension - 1, 1)
    ranking_work = (
        3 * vector_count**2 * max(int(math.ceil(math.log2(vector_count))), 1)
    )
    sqrt_count = 3 * vector_count**2
    svd_work_upper_bound = 20 * dimension**3
    return {
        "dense_bytes": dimension**2 * 8,
        "factorized_bytes": factor_count * dimension**2 * 8,
        "dense_build_multiplications": factor_count * dimension**3,
        "dense_build_additions": factor_count * dimension**2 * (dimension - 1),
        "dense_multiplications": dense_multiplications,
        "dense_additions": dense_additions,
        "factorized_multiplications": factor_multiplications,
        "factorized_additions": factor_additions,
        "executed_apply_multiplications": 2 * factor_multiplications + dense_multiplications,
        "executed_apply_additions": 2 * factor_additions + dense_additions,
        "executed_distance_passes": 3,
        "distance_scalar_work_units": distance_work,
        "sqrt_count": sqrt_count,
        "ranking_comparison_upper_bound": ranking_work,
        "matrix_rank_svd_work_upper_bound": svd_work_upper_bound,
        "rank_list_bytes_upper_bound": 3 * vector_count**2 * 32,
        "matrix_rank_workspace_bytes_upper_bound": 32 * dimension**2,
    }


def _spu0_estimated_bytes(dimension: int, factor_count: int, vector_count: int) -> int:
    accounting = _spu0_accounting(dimension, factor_count, vector_count)
    numeric_arrays = (
        factor_count * dimension**2
        + 4 * vector_count * dimension
        + dimension**2
        + vector_count**2 * dimension
        + 3 * vector_count**2
    ) * 8
    return (
        numeric_arrays
        + accounting["rank_list_bytes_upper_bound"]
        + accounting["matrix_rank_workspace_bytes_upper_bound"]
    )


def _spu0_work_units(dimension: int, factor_count: int, vector_count: int) -> int:
    accounting = _spu0_accounting(dimension, factor_count, vector_count)
    return sum(
        accounting[key]
        for key in (
            "dense_build_multiplications", "dense_build_additions",
            "executed_apply_multiplications", "executed_apply_additions",
            "distance_scalar_work_units", "sqrt_count",
            "ranking_comparison_upper_bound", "matrix_rank_svd_work_upper_bound",
        )
    )


def recompute_spu0_witness(
    *, seed: int, dimension: int, factor_count: int, vector_count: int
) -> Dict[str, object]:
    """Regenerate SPU0 independently of artifact fields using shared helpers."""

    rng = np.random.default_rng(seed)
    factors = [
        np.eye(dimension) + rng.normal(0.0, 0.04, (dimension, dimension))
        for _ in range(factor_count)
    ]
    vectors = rng.normal(size=(vector_count, dimension))
    lazy = vectors.copy()
    flat = np.eye(dimension)
    for factor in factors:
        lazy = lazy @ factor.T
        flat = factor @ flat
    dense = vectors @ flat.T
    factorized = vectors.copy()
    for factor in factors:
        factorized = factorized @ factor.T
    output_error = float(np.max(np.abs(lazy - dense)))
    factorized_output_error = float(np.max(np.abs(factorized - dense)))
    distance_lazy = _pairwise_distances(lazy)
    distance_dense = _pairwise_distances(dense)
    distance_factorized = _pairwise_distances(factorized)
    distance_error = float(np.max(np.abs(distance_lazy - distance_dense)))
    dense_ranks = _stable_neighbor_ranks(distance_dense)
    rank_equal = (
        _stable_neighbor_ranks(distance_lazy) == dense_ranks
        and _stable_neighbor_ranks(distance_factorized) == dense_ranks
    )
    tolerance = 1e-10
    errors = {
        "output_max_abs_error": output_error,
        "output_max_relative_error": _relative_error(output_error, dense),
        "factorized_output_max_abs_error": factorized_output_error,
        "factorized_output_max_relative_error": _relative_error(
            factorized_output_error, dense
        ),
        "distance_max_abs_error": distance_error,
        "distance_max_relative_error": _relative_error(distance_error, distance_dense),
    }
    equivalent = all(value <= tolerance for value in errors.values()) and rank_equal
    accounting = _spu0_accounting(dimension, factor_count, vector_count)
    work = _spu0_work_units(dimension, factor_count, vector_count)
    accounting["total_declared_work_upper_bound"] = work
    return {
        "dimensions": {"d": dimension, "K": factor_count, "vectors": vector_count},
        **errors,
        "rank_equal": rank_equal,
        "dense_rank": int(np.linalg.matrix_rank(flat)),
        "arithmetic_tolerance": tolerance,
        "resource_frontier": accounting,
        "disposition": (
            "ORDINARY_FACTORIZATION_EQUIVALENCE"
            if equivalent
            else "CLASSIFY_MISMATCH_BEFORE_ROUTING"
        ),
        "estimated_incremental_bytes": _spu0_estimated_bytes(
            dimension, factor_count, vector_count
        ),
        "work_units": work,
    }


def run_spu0(
    *, seed: int, dimension: int = 32, factor_count: int = 5, vector_count: int = 48,
    budget: Wave0ABudget = Wave0ABudget(),
) -> Dict[str, object]:
    if not 2 <= dimension <= 128 or not 1 <= factor_count <= 8 or not 2 <= vector_count <= 256:
        raise Wave0AValidationError("SPU0 dimensions exceed the frozen bounds")
    resource = _preflight(
        "PRW-SPU0", _spu0_estimated_bytes(dimension, factor_count, vector_count),
        _spu0_work_units(dimension, factor_count, vector_count), budget,
    )
    witness = recompute_spu0_witness(
        seed=seed, dimension=dimension, factor_count=factor_count,
        vector_count=vector_count,
    )
    witness.pop("estimated_incremental_bytes")
    witness.pop("work_units")
    _check_live_rss(resource, "spu0_complete")
    return {
        "stage_id": "PRW-SPU0",
        "seed": seed,
        "evidence_state": "COMPLETE",
        "scientific_claim_status": "UNCONFIRMED",
        "novelty_status": "UNCONFIRMED",
        **witness,
        "resource_estimate": resource,
        "limitations": ["fixed linear float64 maps only", "resource difference is not expressivity"],
    }


def _objective(mask: np.ndarray, unary: np.ndarray, edges: Sequence[Tuple[int, int]], lam: float) -> float:
    return float(np.dot(unary, mask) + lam * sum(abs(mask[i] - mask[j]) for i, j in edges))


def _quadratic_relaxation_objective(
    values: np.ndarray, unary: np.ndarray, edges: Sequence[Tuple[int, int]], lam: float
) -> float:
    return float(np.dot(unary, values) + lam * sum((values[i] - values[j]) ** 2 for i, j in edges))


def _exact_mask(unary: np.ndarray, edges: Sequence[Tuple[int, int]], lam: float) -> Tuple[np.ndarray, float]:
    best_mask: Optional[np.ndarray] = None
    best_value = math.inf
    for bits in itertools.product((0.0, 1.0), repeat=len(unary)):
        mask = np.asarray(bits, dtype=np.float64)
        value = _objective(mask, unary, edges, lam)
        if value < best_value - 1e-12:
            best_mask, best_value = mask, value
    if best_mask is None:
        raise Wave0AValidationError("exact enumeration produced no mask")
    return best_mask, best_value


def _round_mask(values: np.ndarray) -> np.ndarray:
    return (values >= 0.5).astype(np.float64)


def _smooth_relaxation(unary: np.ndarray, edges: Sequence[Tuple[int, int]], lam: float) -> Tuple[np.ndarray, np.ndarray]:
    values = np.full(len(unary), 0.5, dtype=np.float64)
    degree = max((sum(i in edge for edge in edges) for i in range(len(unary))), default=0)
    learning_rate = 1.0 / (1.0 + 4.0 * lam * max(degree, 1))
    for _ in range(256):
        gradient = unary.copy()
        for i, j in edges:
            difference = values[i] - values[j]
            gradient[i] += 2.0 * lam * difference
            gradient[j] -= 2.0 * lam * difference
        values = np.clip(values - learning_rate * gradient, 0.0, 1.0)
    return values, _round_mask(values)


def _proximal_tv(unary: np.ndarray, edges: Sequence[Tuple[int, int]], lam: float) -> np.ndarray:
    values = np.full(len(unary), 0.5, dtype=np.float64)
    for step in range(1, 513):
        gradient = unary.copy()
        for i, j in edges:
            sign = float(np.sign(values[i] - values[j]))
            gradient[i] += lam * sign
            gradient[j] -= lam * sign
        values = np.clip(values - (0.2 / math.sqrt(step)) * gradient, 0.0, 1.0)
    return _round_mask(values)


def _graph_cut_mask(unary: np.ndarray, edges: Sequence[Tuple[int, int]], lam: float) -> np.ndarray:
    n = len(unary)
    source, sink = n, n + 1
    residual: Dict[int, Dict[int, float]] = {node: {} for node in range(n + 2)}

    def add_arc(a: int, b: int, capacity: float) -> None:
        residual[a][b] = residual[a].get(b, 0.0) + max(0.0, capacity)
        residual[b].setdefault(a, 0.0)

    for i, cost_one in enumerate(unary):
        shift = min(0.0, float(cost_one))
        add_arc(source, i, float(cost_one) - shift)
        add_arc(i, sink, -shift)
    for i, j in edges:
        add_arc(i, j, lam)
        add_arc(j, i, lam)
    while True:
        parent = {source: -1}
        queue = [source]
        for node in queue:
            for neighbor, capacity in residual[node].items():
                if capacity > 1e-12 and neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
                    if neighbor == sink:
                        break
            if sink in parent:
                break
        if sink not in parent:
            break
        flow = math.inf
        node = sink
        while node != source:
            flow = min(flow, residual[parent[node]][node])
            node = parent[node]
        node = sink
        while node != source:
            previous = parent[node]
            residual[previous][node] -= flow
            residual[node][previous] = residual[node].get(previous, 0.0) + flow
            node = previous
    reachable = {source}
    queue = [source]
    for node in queue:
        for neighbor, capacity in residual[node].items():
            if capacity > 1e-12 and neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    return np.asarray([0.0 if i in reachable else 1.0 for i in range(n)], dtype=np.float64)


def _greedy_mask(unary: np.ndarray, edges: Sequence[Tuple[int, int]], lam: float) -> np.ndarray:
    mask = np.zeros(len(unary), dtype=np.float64)
    current = _objective(mask, unary, edges, lam)
    while True:
        choices = []
        for index in range(len(unary)):
            if mask[index] == 0.0:
                candidate = mask.copy()
                candidate[index] = 1.0
                choices.append((_objective(candidate, unary, edges, lam), index, candidate))
        if not choices:
            return mask
        value, _, candidate = min(choices, key=lambda item: (item[0], item[1]))
        if value >= current - 1e-12:
            return mask
        mask, current = candidate, value


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right)
    if not np.any(union):
        return 1.0
    return float(np.logical_and(left, right).sum() / union.sum())


def _recall(mask: np.ndarray, relevant: Sequence[int]) -> float:
    if not relevant:
        return 1.0
    return float(sum(mask[index] == 1.0 for index in relevant) / len(relevant))


def _var1_cells(seed: int, n: int) -> List[Tuple[str, np.ndarray, Tuple[Tuple[int, int], ...], float, Tuple[int, ...]]]:
    rng = np.random.default_rng(seed)
    path = tuple((i, i + 1) for i in range(n - 1))
    ring = path + ((n - 1, 0),)
    star = tuple((0, i) for i in range(1, n))
    unaries = [
        np.linspace(-1.2, 0.8, n),
        np.asarray([(-1.0 if i % 2 == 0 else 0.6) for i in range(n)]),
        rng.normal(-0.05, 0.7, n),
        rng.normal(0.1, 0.9, n),
    ]
    edges = [path, ring, star, path]
    lambdas = [0.25, 0.8, 0.45, 1.1]
    cells = []
    for index, (unary, graph, lam) in enumerate(zip(unaries, edges, lambdas)):
        relevant = tuple(int(i) for i in np.flatnonzero(unary < 0.0))
        cells.append((f"cell-{index + 1}", unary, graph, lam, relevant))
    return cells


def _compute_var1_row(
    cell_id: str, unary: np.ndarray, edges: Sequence[Tuple[int, int]], lam: float,
    relevant: Sequence[int],
) -> Dict[str, object]:
    n = len(unary)
    exact, exact_value = _exact_mask(unary, edges, lam)
    relaxed_values, smooth = _smooth_relaxation(unary, edges, lam)
    methods = {
        "smooth_rounding": smooth,
        "proximal_graph_tv": _proximal_tv(unary, edges, lam),
        "graph_cut": _graph_cut_mask(unary, edges, lam),
        "submodular_greedy": _greedy_mask(unary, edges, lam),
    }
    metrics: Dict[str, object] = {}
    for name, mask in methods.items():
        value = _objective(mask, unary, edges, lam)
        metrics[name] = {
            "objective": value,
            "objective_gap": value - exact_value,
            "selection_jaccard": _jaccard(mask, exact),
            "feasible": bool(len(mask) == n and np.all(np.isin(mask, (0.0, 1.0)))),
            "downstream_recall": _recall(mask, relevant),
            "mask": mask.astype(int).tolist(),
        }
    if abs(float(metrics["graph_cut"]["objective_gap"])) > 1e-9:
        raise Wave0AValidationError(
            "graph-cut control disagrees with exact attractive-Potts oracle"
        )
    relaxed_objective = _quadratic_relaxation_objective(
        relaxed_values, unary, edges, lam
    )
    return {
        "cell_id": cell_id,
        "n": n,
        "lambda": lam,
        "edge_count": len(edges),
        "unary_digest": sha256_payload(unary.tolist()),
        "relevant_indices": list(relevant),
        "exact": {
            "objective": exact_value,
            "mask": exact.astype(int).tolist(),
            "downstream_recall": _recall(exact, relevant),
        },
        "relaxed_objective": relaxed_objective,
        "integrality_gap": exact_value - relaxed_objective,
        "methods": metrics,
    }


def recompute_var1_witness(*, seed: int, n: int) -> List[Dict[str, object]]:
    """Regenerate VAR1 independently of artifact fields using shared helpers."""

    return [
        _compute_var1_row(cell_id, unary, edges, lam, relevant)
        for cell_id, unary, edges, lam, relevant in _var1_cells(seed, n)
    ]


def run_var1(*, seed: int, n: int = 12, budget: Wave0ABudget = Wave0ABudget()) -> Dict[str, object]:
    if not 2 <= n <= 20:
        raise Wave0AValidationError("VAR1 exact enumeration requires n in [2, 20]")
    cell_count = 4
    work = cell_count * (2**n) * (n + n)
    resource = _preflight("PRW-VAR1", 16 * 1024 * 1024 + 2**n * 8, work, budget)
    deadline = _deadline(budget)
    rows: List[Dict[str, object]] = []
    for cell_id, unary, edges, lam, relevant in _var1_cells(seed, n):
        started = time.perf_counter()
        row = _compute_var1_row(cell_id, unary, edges, lam, relevant)
        row["runtime_seconds"] = time.perf_counter() - started
        rows.append(row)
        _check_deadline(deadline, "var1_cell")
    smooth_equivalent = all(
        abs(float(row["methods"]["smooth_rounding"]["objective_gap"])) <= 1e-9
        and float(row["methods"]["smooth_rounding"]["downstream_recall"])
        == float(row["exact"]["downstream_recall"])
        for row in rows
    )
    _check_live_rss(resource, "var1_complete")
    return {
        "stage_id": "PRW-VAR1",
        "seed": seed,
        "evidence_state": "COMPLETE",
        "scientific_claim_status": "UNCONFIRMED",
        "novelty_status": "UNCONFIRMED",
        "rows": rows,
        "null_survives": smooth_equivalent,
        "disposition": "RETAIN_SMOOTH_HARD_MASK_NULL" if smooth_equivalent else "ROUTE_SMOOTH_METHOD_TO_DECLARED_STATE_SUBPROBLEM_ONLY",
        "resource_estimate": resource,
        "limitations": ["small attractive binary Potts fixtures", "no large-n optimizer claim"],
    }


def make_stage_artifact(
    stage_id: str, *, seed: int, phase: str, protocol_digest: str, budget: Wave0ABudget
) -> Dict[str, object]:
    if protocol_digest != FROZEN_PROTOCOL_SHA256:
        raise Wave0AValidationError("artifact protocol digest is not the canonical preregistration")
    if phase not in ("SELECT", "REPORT") and not phase.startswith("UNIT-"):
        raise Wave0AValidationError("phase must be SELECT, REPORT, or UNIT-prefixed")
    execution_platform_system = platform.system()
    if execution_platform_system not in ("Linux", "Windows"):
        raise Wave0AValidationError("Wave 0A execution platform must be Linux or Windows")
    if stage_id == "PRW-EK0":
        result = run_ek0(budget=budget)
    elif stage_id == "PRW-BIL1":
        result = run_bil1(seed=seed, budget=budget)
    elif stage_id == "PRW-SPU0":
        result = run_spu0(seed=seed, budget=budget)
    elif stage_id == "PRW-VAR1":
        result = run_var1(seed=seed, budget=budget)
    else:
        raise Wave0AValidationError(f"unknown stage: {stage_id}")
    payload: Dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": protocol_digest,
        "execution_platform_system": execution_platform_system,
        "phase": phase,
        "seed": seed,
        "stage_id": stage_id,
        "status": "COMPLETE",
        "result": result,
        "budget": asdict(budget),
    }
    payload["artifact_digest"] = sha256_payload(payload)
    return payload


__all__ = [
    "ARTIFACT_SCHEMA_VERSION", "DEFAULT_MAX_BYTES", "FROZEN_PROTOCOL_SHA256",
    "MIN_ESTIMATED_BYTES", "MIN_OUTPUT_BYTES", "PROTOCOL_ID",
    "REPORT_SEEDS", "SELECT_SEEDS", "STAGE_IDS",
    "Wave0ABudget", "Wave0AResourceError", "Wave0AValidationError", "atomic_write_json",
    "canonical_json", "make_stage_artifact", "recompute_spu0_witness",
    "recompute_var1_witness", "run_bil1", "run_ek0", "run_spu0", "run_var1",
    "sha256_file", "sha256_payload",
]
