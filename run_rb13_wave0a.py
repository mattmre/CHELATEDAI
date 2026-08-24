"""Run and verify frozen RB-13 Wave 0A SELECT/REPORT campaigns."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import signal
import stat
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rb13_wave0a_experiments import (
    ARTIFACT_SCHEMA_VERSION,
    FROZEN_PROTOCOL_SHA256,
    MIN_ESTIMATED_BYTES,
    MIN_OUTPUT_BYTES,
    PROTOCOL_ID,
    REPORT_SEEDS,
    SELECT_SEEDS,
    STAGE_IDS,
    STAGE_MAX_BYTES,
    Wave0ABudget,
    Wave0AResourceError,
    Wave0AValidationError,
    atomic_write_json,
    canonical_json,
    current_rss_bytes,
    make_stage_artifact,
    recompute_spu0_witness,
    recompute_var1_witness,
    sha256_file,
    sha256_payload,
)

DEFAULT_PROTOCOL = Path("docs/research/rb13-wave0a-method-dev-protocol-2026-08-15.md")
MANIFEST_SCHEMA_VERSION = 1
INVALID_DETAIL_TRUNCATION_MARKER = "...[UTF8_BUDGET_TRUNCATED]"
CLEANUP_GRACE_SECONDS = 5.0
RSS_SAMPLE_INTERVAL_SECONDS = 0.05
INVALID_RUN_EVIDENCE_MAX_BYTES = MIN_OUTPUT_BYTES
ARCHIVE_VAR1_ABS_TOLERANCE = 2.0e-15
ARCHIVE_SPU0_ABS_TOLERANCE = 2.5e-31
ARCHIVE_MANIFEST_FILE_SHA256 = {
    "SELECT": "f17defbae0c6ac7a49a5eba295604ead4a70e66ee5dae749c7f2289153339ab2",
    "REPORT": "e03f090d632307f58f5316f2dbec9a2b214ecc8bcbcff5cc8910bbd17a371a2c",
}


def _platform_monitor_flags(system: str) -> Tuple[bool, bool]:
    if system == "Linux":
        return True, False
    if system == "Windows":
        return False, True
    raise Wave0AValidationError("execution platform must be Linux or Windows")


def _ek0_dependency_specs_available() -> bool:
    """Check import discoverability without importing Torch or measuring its RSS."""

    return all(
        importlib.util.find_spec(module_name) is not None
        for module_name in ("torch", "model_scope_memory")
    )


def _stage_worker_reservation(
    stage_id: str, parent_rss_bytes: int, campaign_ceiling_bytes: int
) -> Tuple[int, int]:
    """Return worker reservation and sampled aggregate backstop before launch."""

    if stage_id == "PRW-EK0" and campaign_ceiling_bytes < STAGE_MAX_BYTES[stage_id]:
        raise Wave0AResourceError("EK0 requires its frozen 768 MiB aggregate ceiling")
    sampled_backstop_ceiling = min(STAGE_MAX_BYTES[stage_id], campaign_ceiling_bytes)
    worker_reservation = sampled_backstop_ceiling - parent_rss_bytes
    if (
        worker_reservation < MIN_ESTIMATED_BYTES
        or sampled_backstop_ceiling > campaign_ceiling_bytes
    ):
        raise Wave0AResourceError(
            f"parent cannot reserve the modeled worker budget for {stage_id}"
        )
    return worker_reservation, sampled_backstop_ceiling


def _require_finite(value: object, path: str = "artifact") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise Wave0AValidationError(f"nonfinite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, f"{path}[{index}]")


def _verify_archived_regeneration(
    observed: object, expected: object, *, absolute_tolerance: float, path: str
) -> float:
    """Compare regenerated numerics portably while keeping structure/types exact."""

    if type(observed) is not type(expected):
        raise Wave0AValidationError(f"archived regeneration type drift at {path}")
    if isinstance(expected, dict):
        if set(observed) != set(expected):
            raise Wave0AValidationError(f"archived regeneration schema drift at {path}")
        return max(
            (
                _verify_archived_regeneration(
                    observed[key], expected[key], absolute_tolerance=absolute_tolerance,
                    path=f"{path}.{key}",
                )
                for key in expected
            ),
            default=0.0,
        )
    if isinstance(expected, list):
        if len(observed) != len(expected):
            raise Wave0AValidationError(f"archived regeneration length drift at {path}")
        return max(
            (
                _verify_archived_regeneration(
                    left, right, absolute_tolerance=absolute_tolerance,
                    path=f"{path}[{index}]",
                )
                for index, (left, right) in enumerate(zip(observed, expected))
            ),
            default=0.0,
        )
    if isinstance(expected, float):
        if not math.isfinite(observed) or not math.isfinite(expected):
            raise Wave0AValidationError(
                f"archived regeneration nonfinite float at {path}"
            )
        difference = abs(observed - expected)
        if not math.isfinite(difference):
            raise Wave0AValidationError(
                f"archived regeneration nonfinite difference at {path}"
            )
        if difference > absolute_tolerance:
            raise Wave0AValidationError(
                f"archived regeneration numeric drift {difference} exceeds "
                f"{absolute_tolerance} at {path}"
            )
        return difference
    if observed != expected:
        raise Wave0AValidationError(f"archived regeneration exact drift at {path}")
    return 0.0


def _validate_archive_entry_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise Wave0AValidationError("archived entry path is not canonical POSIX relative text")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise Wave0AValidationError("archived entry path is unsafe")
    if path.as_posix() != value:
        raise Wave0AValidationError("archived entry path is not canonical")
    return value


def _path_is_reparse_point(path: Path) -> bool:
    """Detect Windows reparse points on Python versions without is_junction."""

    try:
        attributes = int(getattr(os.lstat(path), "st_file_attributes", 0))
    except OSError as error:
        raise Wave0AValidationError(f"cannot lstat archived path: {path}") from error
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _require_archive_containment(root: Path, candidate: Path) -> None:
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise Wave0AValidationError(f"cannot resolve archived path: {candidate}") from error
    if resolved != root and root not in resolved.parents:
        raise Wave0AValidationError("archived path resolves outside campaign root")


def _archive_regular_file_set(directory: Path) -> set[str]:
    if directory.is_symlink() or _path_is_reparse_point(directory):
        raise Wave0AValidationError("archived campaign root cannot be a link")
    resolved_root = directory.resolve(strict=True)
    observed: set[str] = set()
    for root, directory_names, file_names in os.walk(directory, followlinks=False):
        root_path = Path(root)
        _require_archive_containment(resolved_root, root_path)
        for name in directory_names:
            candidate = root_path / name
            if candidate.is_symlink() or _path_is_reparse_point(candidate):
                raise Wave0AValidationError("archived campaign contains a linked directory")
            _require_archive_containment(resolved_root, candidate)
        for name in file_names:
            candidate = root_path / name
            if (
                candidate.is_symlink()
                or _path_is_reparse_point(candidate)
                or not candidate.is_file()
            ):
                raise Wave0AValidationError("archived campaign contains a linked/nonregular file")
            _require_archive_containment(resolved_root, candidate)
            observed.add(candidate.relative_to(directory).as_posix())
    return observed


def _verify_resource_estimate(
    resource: Mapping[str, object], budget: Mapping[str, object], stage_id: str
) -> None:
    keys = {
        "estimated_incremental_bytes", "rss_at_preflight_bytes", "modeled_peak_bytes",
        "stage_ceiling_bytes", "work_units",
    }
    if set(resource) != keys or any(type(resource[key]) is not int for key in keys):
        raise Wave0AValidationError("resource estimate schema/types are invalid")
    if any(int(resource[key]) < 0 for key in keys):
        raise Wave0AValidationError("resource estimate cannot be negative")
    if resource["modeled_peak_bytes"] != (
        resource["rss_at_preflight_bytes"] + resource["estimated_incremental_bytes"]
    ):
        raise Wave0AValidationError("resource modeled peak is inconsistent")
    expected_ceiling = min(int(budget["max_estimated_bytes"]), STAGE_MAX_BYTES[stage_id])
    if resource["stage_ceiling_bytes"] != expected_ceiling:
        raise Wave0AValidationError("resource stage ceiling is inconsistent")
    if resource["modeled_peak_bytes"] > expected_ceiling or resource["work_units"] <= 0:
        raise Wave0AValidationError("resource estimate exceeds its frozen boundary")


def _verify_result_semantics(
    result: Mapping[str, object], budget: Mapping[str, object], stage_id: str,
    *, archived_cross_platform: bool = False,
) -> Dict[str, float]:
    _verify_resource_estimate(result["resource_estimate"], budget, stage_id)
    if stage_id == "PRW-EK0":
        observations = result["observations"]
        expected_classifications = {
            "fifo_overflow": "MIGRATION_GAP",
            "annotation_mutates_payload": "MIGRATION_GAP",
            "direct_promote_bypasses_gate": "BUG",
            "missing_status_promotes": "BUG",
            "negative_status_fails_closed": "CURRENT_CONTRACT",
            "persistent_save_overwrites_key": "MIGRATION_GAP",
            "persistent_delete_removes_key": "MIGRATION_GAP",
            "restart_loads_latest_value": "CURRENT_CONTRACT",
        }
        expected_observations = {
            name: {"observed": True, "classification": classification}
            for name, classification in expected_classifications.items()
        }
        if observations != expected_observations:
            raise Wave0AValidationError("EK0 observations differ from the frozen store witness")
        if result["null_survives"] is not False or result["disposition"] != (
            "REQUIRE_COMPATIBILITY_AND_MIGRATION_DESIGN"
        ):
            raise Wave0AValidationError("EK0 frozen null/disposition mismatch")
        return {}
    if stage_id == "PRW-BIL1":
        invariants = result["invariants"]
        invariant_keys = {
            "permutation_invariant", "lineage_duplicate_idempotent", "unknown_conflict_distinct",
            "fixed_point_converges", "fixed_point_order_invariant", "temporal_oracle_matches",
        }
        if set(invariants) != invariant_keys or any(value is not True for value in invariants.values()):
            raise Wave0AValidationError("BIL1 frozen invariants must all hold")
        rows = result["rows"]
        states = ("SUPPORTED", "CONFLICTED", "SUPPORTED", "CONFLICTED", "CONFLICTED")
        scalar = ("SUPPORTED", "UNKNOWN", "SUPPORTED", "SUPPORTED", "UNKNOWN")
        active = (2, 3, 3, 4, 2)
        operations = (3, 5, 5, 7, 4)
        expected_rows = []
        for index, state in enumerate(states):
            paired = state
            expected_rows.append({
                "valid_at": 2 * (index + 1), "transaction_at": 2 * (index + 1),
                "candidate": state,
                "controls": {
                    "scalar": scalar[index],
                    "last_write_wins": "REFUTED" if state == "CONFLICTED" else state,
                    "paired_overwrite": "REFUTED" if state == "CONFLICTED" else state,
                    "paired_boolean_union": paired,
                },
                "active_event_count": active[index], "operations": operations[index],
            })
        if rows != expected_rows:
            raise Wave0AValidationError("BIL1 rows differ from the frozen lineage witness")
        if result["fixed_point"] != {"iterations": [3, 4], "operations": 76}:
            raise Wave0AValidationError("BIL1 fixed-point witness mismatch")
        if result["false_promotion_count"] != 0:
            raise Wave0AValidationError("BIL1 false-promotion count mismatch")
        if result["disposition"] != "REDUCE_TO_PAIRED_BOOLEAN_UNION":
            raise Wave0AValidationError("BIL1 frozen disposition mismatch")
        return {}
    if stage_id == "PRW-SPU0":
        dimensions = result["dimensions"]
        if set(dimensions) != {"d", "K", "vectors"} or any(type(value) is not int for value in dimensions.values()):
            raise Wave0AValidationError("SPU0 dimensions are invalid")
        d, factors, vectors = dimensions["d"], dimensions["K"], dimensions["vectors"]
        if (d, factors, vectors) != (32, 5, 48):
            raise Wave0AValidationError("SPU0 official artifact dimensions are not frozen")
        tolerance = result["arithmetic_tolerance"]
        errors = [
            result["output_max_abs_error"], result["output_max_relative_error"],
            result["factorized_output_max_abs_error"], result["factorized_output_max_relative_error"],
            result["distance_max_abs_error"], result["distance_max_relative_error"],
        ]
        if tolerance != 1e-10 or any(type(value) not in (int, float) or value < 0 for value in errors):
            raise Wave0AValidationError("SPU0 arithmetic fields are invalid")
        witness = recompute_spu0_witness(
            seed=int(result["seed"]), dimension=d, factor_count=factors,
            vector_count=vectors,
        )
        expected_bytes = witness.pop("estimated_incremental_bytes")
        expected_work = witness.pop("work_units")
        observed = {
            key: result[key]
            for key in witness
        }
        if archived_cross_platform:
            maximum_difference = _verify_archived_regeneration(
                observed, witness, absolute_tolerance=ARCHIVE_SPU0_ABS_TOLERANCE,
                path="SPU0",
            )
        else:
            maximum_difference = 0.0
            if observed != witness:
                raise Wave0AValidationError(
                    "SPU0 result differs from artifact-independent shared-code regeneration"
                )
        if result["resource_estimate"]["estimated_incremental_bytes"] != expected_bytes:
            raise Wave0AValidationError("SPU0 preflight allocation upper bound mismatch")
        if result["resource_estimate"]["work_units"] != expected_work:
            raise Wave0AValidationError("SPU0 declared work upper bound mismatch")
        return {"spu0_max_abs_regeneration_difference": maximum_difference}
    rows = result["rows"]
    smooth_equivalent = True
    for row in rows:
        row_keys = {
            "cell_id", "n", "lambda", "edge_count", "unary_digest", "relevant_indices",
            "exact", "relaxed_objective", "integrality_gap", "methods", "runtime_seconds",
        }
        if not isinstance(row, dict) or set(row) != row_keys:
            raise Wave0AValidationError("VAR1 row schema keys are not exact")
        n = row.get("n")
        exact = row.get("exact")
        methods = row.get("methods")
        if type(n) is not int or not 2 <= n <= 20 or not isinstance(exact, dict):
            raise Wave0AValidationError("VAR1 row dimensions/exact schema are invalid")
        if set(exact) != {"objective", "mask", "downstream_recall"}:
            raise Wave0AValidationError("VAR1 exact schema is invalid")
        exact_mask = exact["mask"]
        if not isinstance(exact_mask, list) or len(exact_mask) != n or any(bit not in (0, 1) for bit in exact_mask):
            raise Wave0AValidationError("VAR1 exact mask is invalid")
        relevant = row["relevant_indices"]
        if not isinstance(relevant, list) or any(type(index) is not int or not 0 <= index < n for index in relevant):
            raise Wave0AValidationError("VAR1 relevant-index schema is invalid")
        expected_exact_recall = 1.0 if not relevant else sum(exact_mask[index] == 1 for index in relevant) / len(relevant)
        if exact["downstream_recall"] != expected_exact_recall:
            raise Wave0AValidationError("VAR1 exact recall is inconsistent")
        if type(row["runtime_seconds"]) not in (int, float) or row["runtime_seconds"] < 0:
            raise Wave0AValidationError("VAR1 runtime is invalid")
        for name, metrics in methods.items():
            metric_keys = {"objective", "objective_gap", "selection_jaccard", "feasible", "downstream_recall", "mask"}
            if not isinstance(metrics, dict) or set(metrics) != metric_keys:
                raise Wave0AValidationError("VAR1 metric schema is invalid")
            mask = metrics["mask"]
            feasible = isinstance(mask, list) and len(mask) == n and all(bit in (0, 1) for bit in mask)
            if metrics["feasible"] is not feasible or not feasible:
                raise Wave0AValidationError("VAR1 feasibility is not derived from the mask")
            if abs((metrics["objective"] - exact["objective"]) - metrics["objective_gap"]) > 1e-9:
                raise Wave0AValidationError("VAR1 objective gap is inconsistent")
            selected, exact_selected = set(i for i, bit in enumerate(mask) if bit), set(
                i for i, bit in enumerate(exact_mask) if bit
            )
            union = selected | exact_selected
            expected_jaccard = 1.0 if not union else len(selected & exact_selected) / len(union)
            expected_recall = 1.0 if not relevant else sum(mask[index] == 1 for index in relevant) / len(relevant)
            if metrics["selection_jaccard"] != expected_jaccard or metrics["downstream_recall"] != expected_recall:
                raise Wave0AValidationError("VAR1 Jaccard/recall is not derived from masks")
            if not 0.0 <= metrics["selection_jaccard"] <= 1.0 or not 0.0 <= metrics["downstream_recall"] <= 1.0:
                raise Wave0AValidationError("VAR1 bounded metric is invalid")
            if name == "graph_cut" and abs(metrics["objective_gap"]) > 1e-9:
                raise Wave0AValidationError("VAR1 graph cut disagrees with exact oracle")
        smooth = methods["smooth_rounding"]
        smooth_equivalent &= (
            abs(smooth["objective_gap"]) <= 1e-9
            and smooth["downstream_recall"] == exact["downstream_recall"]
        )
    if any(row["n"] != 12 for row in rows):
        raise Wave0AValidationError("VAR1 official artifact dimension is not frozen")
    regenerated = recompute_var1_witness(seed=int(result["seed"]), n=12)
    observed_witness = [
        {key: value for key, value in row.items() if key != "runtime_seconds"}
        for row in rows
    ]
    if archived_cross_platform:
        maximum_difference = _verify_archived_regeneration(
            observed_witness, regenerated,
            absolute_tolerance=ARCHIVE_VAR1_ABS_TOLERANCE, path="VAR1",
        )
    else:
        maximum_difference = 0.0
        if observed_witness != regenerated:
            raise Wave0AValidationError(
                "VAR1 cells differ from artifact-independent shared-code regeneration"
            )
    disposition = (
        "RETAIN_SMOOTH_HARD_MASK_NULL" if smooth_equivalent
        else "ROUTE_SMOOTH_METHOD_TO_DECLARED_STATE_SUBPROBLEM_ONLY"
    )
    if result["null_survives"] is not smooth_equivalent or result["disposition"] != disposition:
        raise Wave0AValidationError("VAR1 null/disposition is not derived from every cell")
    return {"var1_max_abs_regeneration_difference": maximum_difference}


def _verify_stage_artifact(
    payload: Mapping[str, object], *, stage_id: str, seed: int, phase: str,
    archived_cross_platform: bool,
) -> Dict[str, float]:
    required = {
        "schema_version", "protocol_id", "protocol_digest", "phase", "seed",
        "stage_id", "status", "execution_platform_system", "result", "budget",
        "artifact_digest",
    }
    if set(payload) != required:
        raise Wave0AValidationError("stage artifact schema keys are not exact")
    exact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": FROZEN_PROTOCOL_SHA256,
        "phase": phase,
        "seed": seed,
        "stage_id": stage_id,
        "status": "COMPLETE",
    }
    for key, expected in exact.items():
        if payload.get(key) != expected or type(payload.get(key)) is not type(expected):
            raise Wave0AValidationError(f"stage artifact {key} mismatch")
    if payload.get("execution_platform_system") not in ("Linux", "Windows"):
        raise Wave0AValidationError("stage execution platform is not allowlisted")
    result, budget = payload.get("result"), payload.get("budget")
    if not isinstance(result, dict) or not isinstance(budget, dict):
        raise Wave0AValidationError("stage artifact result and budget must be objects")
    if result.get("stage_id") != stage_id:
        raise Wave0AValidationError("result stage_id mismatch")
    if result.get("scientific_claim_status") != "UNCONFIRMED":
        raise Wave0AValidationError("scientific claim boundary was promoted")
    if result.get("novelty_status") != "UNCONFIRMED":
        raise Wave0AValidationError("novelty claim boundary was promoted")
    expected_evidence_state = {
        "PRW-EK0": "CHARACTERIZED", "PRW-BIL1": "COMPLETE",
        "PRW-SPU0": "COMPLETE", "PRW-VAR1": "COMPLETE",
    }[stage_id]
    expected_limitations = {
        "PRW-EK0": [
            "current code contract only", "temporary stores only", "no production mutation",
        ],
        "PRW-BIL1": ["synthetic finite events", "no production evidence graph"],
        "PRW-SPU0": [
            "fixed linear float64 maps only", "resource difference is not expressivity",
        ],
        "PRW-VAR1": [
            "small attractive binary Potts fixtures", "no large-n optimizer claim",
        ],
    }[stage_id]
    if result.get("evidence_state") != expected_evidence_state:
        raise Wave0AValidationError("stage evidence-state boundary mismatch")
    if result.get("limitations") != expected_limitations:
        raise Wave0AValidationError("stage limitations boundary mismatch")
    result_schemas = {
        "PRW-EK0": {
            "stage_id", "evidence_state", "scientific_claim_status", "novelty_status",
            "null_survives", "disposition", "observations", "resource_estimate", "limitations",
        },
        "PRW-BIL1": {
            "stage_id", "seed", "evidence_state", "scientific_claim_status", "novelty_status",
            "rows", "invariants", "false_promotion_count", "fixed_point", "disposition",
            "resource_estimate", "limitations",
        },
        "PRW-SPU0": {
            "stage_id", "seed", "evidence_state", "scientific_claim_status", "novelty_status",
            "dimensions", "output_max_abs_error", "output_max_relative_error",
            "factorized_output_max_abs_error", "factorized_output_max_relative_error",
            "distance_max_abs_error", "distance_max_relative_error", "rank_equal", "dense_rank",
            "arithmetic_tolerance", "resource_frontier", "disposition", "resource_estimate",
            "limitations",
        },
        "PRW-VAR1": {
            "stage_id", "seed", "evidence_state", "scientific_claim_status", "novelty_status",
            "rows", "null_survives", "disposition", "resource_estimate", "limitations",
        },
    }
    if set(result) != result_schemas[stage_id]:
        raise Wave0AValidationError("stage result schema keys are not exact")
    if stage_id != "PRW-EK0" and (type(result.get("seed")) is not int or result["seed"] != seed):
        raise Wave0AValidationError("stage result seed mismatch")
    if not isinstance(result.get("resource_estimate"), dict) or not isinstance(result.get("limitations"), list):
        raise Wave0AValidationError("stage result resource/limitations types are invalid")
    if stage_id == "PRW-EK0" and not isinstance(result.get("observations"), dict):
        raise Wave0AValidationError("EK0 observations must be an object")
    if stage_id == "PRW-BIL1":
        if not isinstance(result.get("rows"), list) or len(result["rows"]) != 5:
            raise Wave0AValidationError("BIL1 temporal row schema mismatch")
        if not isinstance(result.get("invariants"), dict):
            raise Wave0AValidationError("BIL1 invariants must be an object")
    if stage_id == "PRW-SPU0" and not isinstance(result.get("resource_frontier"), dict):
        raise Wave0AValidationError("SPU0 resource frontier must be an object")
    if stage_id == "PRW-VAR1":
        rows = result.get("rows")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise Wave0AValidationError("VAR1 corresponding cell schema mismatch")
        if [row.get("cell_id") for row in rows] != [
            "cell-1", "cell-2", "cell-3", "cell-4"
        ]:
            raise Wave0AValidationError("VAR1 corresponding cell schema mismatch")
        expected_methods = {"smooth_rounding", "proximal_graph_tv", "graph_cut", "submodular_greedy"}
        if any(set(row.get("methods", {})) != expected_methods for row in rows):
            raise Wave0AValidationError("VAR1 method schema mismatch")
    budget_keys = {"max_estimated_bytes", "max_seconds_per_stage", "max_output_bytes"}
    if set(budget) != budget_keys:
        raise Wave0AValidationError("artifact budget schema mismatch")
    if type(budget["max_estimated_bytes"]) is not int or type(budget["max_output_bytes"]) is not int:
        raise Wave0AValidationError("artifact byte budgets must be integers")
    if type(budget["max_seconds_per_stage"]) is not float:
        raise Wave0AValidationError("artifact stage deadline budget must be a float")
    try:
        Wave0ABudget(**budget)
    except (TypeError, ValueError) as error:
        raise Wave0AValidationError("artifact budget violates frozen bounds") from error
    _require_finite(payload)
    regeneration = _verify_result_semantics(
        result, budget, stage_id,
        archived_cross_platform=archived_cross_platform,
    )
    without_digest = dict(payload)
    artifact_digest = without_digest.pop("artifact_digest")
    if type(artifact_digest) is not str or artifact_digest != sha256_payload(without_digest):
        raise Wave0AValidationError("artifact digest mismatch")
    return regeneration


def verify_stage_artifact(
    payload: Mapping[str, object], *, stage_id: str, seed: int, phase: str
) -> None:
    """Verify a live/current-platform artifact with exact regeneration."""

    _verify_stage_artifact(
        payload, stage_id=stage_id, seed=seed, phase=phase,
        archived_cross_platform=False,
    )


def verify_archived_stage_artifact(
    payload: Mapping[str, object], *, stage_id: str, seed: int, phase: str
) -> Dict[str, float]:
    """Verify immutable archive bytes with frozen cross-platform tolerances."""

    return _verify_stage_artifact(
        payload, stage_id=stage_id, seed=seed, phase=phase,
        archived_cross_platform=True,
    )


def _read_verified_artifact(path: Path, *, stage_id: str, seed: int, phase: str) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Wave0AValidationError("stage artifact root must be an object")
    verify_stage_artifact(payload, stage_id=stage_id, seed=seed, phase=phase)
    return payload


def _read_verified_archived_artifact(
    path: Path, *, stage_id: str, seed: int, phase: str
) -> Tuple[Dict[str, object], Dict[str, float]]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise Wave0AValidationError("stage artifact root must be an object")
    if raw != (canonical_json(payload) + "\n").encode("utf-8"):
        raise Wave0AValidationError("archived stage artifact bytes are not canonical")
    regeneration = verify_archived_stage_artifact(
        payload, stage_id=stage_id, seed=seed, phase=phase
    )
    return payload, regeneration


def _campaign_summary(artifacts: Sequence[Mapping[str, object]], phase: str) -> Dict[str, object]:
    by_cell: Dict[str, List[Tuple[float, bool]]] = {}
    var_seeds = set()
    for artifact in artifacts:
        if artifact["stage_id"] != "PRW-VAR1":
            continue
        var_seeds.add(int(artifact["seed"]))
        for row in artifact["result"]["rows"]:
            by_cell.setdefault(str(row["cell_id"]), []).append(
                (
                    float(row["methods"]["smooth_rounding"]["objective_gap"]),
                    row["methods"]["smooth_rounding"]["downstream_recall"]
                    == row["exact"]["downstream_recall"],
                )
            )
    if var_seeds and set(by_cell) != {"cell-1", "cell-2", "cell-3", "cell-4"}:
        raise Wave0AValidationError("VAR1 campaign summary is incomplete")
    if any(len(values) != len(var_seeds) for values in by_cell.values()):
        raise Wave0AValidationError("VAR1 corresponding seed/cell grid is incomplete")
    cells = {
        cell_id: {
            "seed_count": len(values),
            "mean_objective_gap": statistics.fmean(value[0] for value in values),
            "population_seed_variance": statistics.pvariance(value[0] for value in values),
            "all_objective_and_recall_equivalent": all(
                abs(value[0]) <= 1e-9 and value[1] for value in values
            ),
        }
        for cell_id, values in sorted(by_cell.items())
    }
    null_survives = bool(cells) and all(
        cell["all_objective_and_recall_equivalent"] for cell in cells.values()
    )
    return {
        "phase": phase,
        "fixture_scope": "synthetic_non_independent_labels",
        "var1_corresponding_cell_seed_summary": cells,
        "cross_cell_values_not_used_as_seed_variance": True,
        "var1_null_survives_all_seeds_and_cells": null_survives,
        "var1_phase_disposition": (
            "RETAIN_SMOOTH_HARD_MASK_NULL" if null_survives
            else "ROUTE_SMOOTH_METHOD_TO_DECLARED_STATE_SUBPROBLEM_ONLY" if cells
            else "NOT_APPLICABLE"
        ),
    }


def _rss_for_pid(pid: int) -> int:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                ("peak_ws", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                ("qpp", ctypes.c_size_t), ("qp", ctypes.c_size_t),
                ("qnpp", ctypes.c_size_t), ("qnp", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
            ]

        open_process = ctypes.windll.kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        handle = open_process(0x0410, False, pid)
        if not handle:
            return 0
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            get_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_info.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
            get_info.restype = wintypes.BOOL
            if get_info(handle, ctypes.byref(counters), counters.cb):
                return int(counters.working_set)
            return 0
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    statm = Path(f"/proc/{pid}/statm")
    if not statm.is_file():
        return 0
    return int(statm.read_text(encoding="ascii").split()[1]) * int(os.sysconf("SC_PAGE_SIZE"))


def _descendants(root_pid: int) -> Tuple[int, ...]:
    if sys.platform == "win32":
        return (root_pid,)
    pending, found = [root_pid], []
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.append(pid)
        children = Path(f"/proc/{pid}/task/{pid}/children")
        if children.is_file():
            pending.extend(int(item) for item in children.read_text(encoding="ascii").split())
    return tuple(found)


def _linux_group_pids(pgid: int) -> Tuple[int, ...]:
    if sys.platform == "win32":
        return ()
    members = []
    for proc_stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            remainder = proc_stat.read_text(encoding="ascii").rsplit(")", 1)[1].split()
            if int(remainder[2]) == pgid:
                members.append(int(proc_stat.parent.name))
        except (IndexError, OSError, ValueError):
            continue
    return tuple(members)


def _terminate_worker(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        if sys.platform == "win32":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if sys.platform != "win32" and _linux_group_pids(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.communicate(timeout=CLEANUP_GRACE_SECONDS)
    if sys.platform != "win32" and _linux_group_pids(process.pid):
        raise Wave0AResourceError("worker process group survived cleanup grace")


def _execute_worker(
    command: Sequence[str], environment: Mapping[str, str], *, deadline_seconds: float,
    effective_ceiling_bytes: int,
) -> Tuple[int, str, str, int]:
    options: Dict[str, object] = {
        "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True,
        "env": dict(environment),
    }
    if sys.platform == "win32":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    process = subprocess.Popen(command, **options)
    deadline = time.monotonic() + deadline_seconds
    peak_tree_rss = current_rss_bytes()
    try:
        while process.poll() is None:
            tree_rss = current_rss_bytes() + sum(
                _rss_for_pid(pid) for pid in _descendants(process.pid)
            )
            peak_tree_rss = max(peak_tree_rss, tree_rss)
            if tree_rss > effective_ceiling_bytes:
                _terminate_worker(process)
                raise Wave0AResourceError(
                    f"process-tree RSS {tree_rss} exceeds {effective_ceiling_bytes} bytes"
                )
            if time.monotonic() > deadline:
                _terminate_worker(process)
                raise Wave0AResourceError("child deadline exceeded")
            time.sleep(RSS_SAMPLE_INTERVAL_SECONDS)
        stdout, stderr = process.communicate(timeout=5)
        return int(process.returncode), stdout, stderr, peak_tree_rss
    finally:
        _terminate_worker(process)


def _invalid_run_payload(
    *, phase: str, stage_id: str, seed: int, category: str,
    exception_type: str, detail: str, execution_platform_system: str,
) -> Dict[str, object]:
    full_detail_sha256 = hashlib.sha256(
        detail.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    fixed: Dict[str, object] = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "protocol_digest": FROZEN_PROTOCOL_SHA256, "status": "INVALID_RUN",
        "execution_platform_system": execution_platform_system,
        "phase": phase, "stage_id": stage_id, "seed": seed,
        "failure_category": category, "exception_type": exception_type,
        "detail_full_sha256": full_detail_sha256,
    }

    def candidate(detail_value: str, truncated: bool) -> Dict[str, object]:
        payload = dict(fixed)
        payload.update({
            "detail": detail_value,
            "detail_truncated": truncated,
            # Reserve the exact encoded width of the final digest while sizing.
            "failure_digest": "0" * 64,
        })
        return payload

    payload = candidate(detail, False)
    if len((canonical_json(payload) + "\n").encode("utf-8")) > INVALID_RUN_EVIDENCE_MAX_BYTES:
        low, high = 0, len(detail)
        while low < high:
            midpoint = (low + high + 1) // 2
            bounded = INVALID_DETAIL_TRUNCATION_MARKER + detail[-midpoint:]
            encoded = (canonical_json(candidate(bounded, True)) + "\n").encode("utf-8")
            if len(encoded) <= INVALID_RUN_EVIDENCE_MAX_BYTES:
                low = midpoint
            else:
                high = midpoint - 1
        payload = candidate(
            INVALID_DETAIL_TRUNCATION_MARKER + detail[-low:] if low else (
                INVALID_DETAIL_TRUNCATION_MARKER
            ),
            True,
        )
    payload.pop("failure_digest")
    payload["failure_digest"] = sha256_payload(payload)
    if len((canonical_json(payload) + "\n").encode("utf-8")) > INVALID_RUN_EVIDENCE_MAX_BYTES:
        raise Wave0AValidationError("bounded INVALID_RUN payload exceeded its byte allowance")
    return payload


def run(
    *, output_directory: Path, phase: str, protocol_path: Path,
    expected_protocol_sha256: Optional[str] = None,
    budget: Wave0ABudget = Wave0ABudget(),
) -> Dict[str, object]:
    if phase not in ("SELECT", "REPORT"):
        raise Wave0AValidationError("official phase must be SELECT or REPORT")
    if output_directory.exists():
        raise Wave0AValidationError("output directory already exists; official runs never overwrite")
    protocol_digest = sha256_file(protocol_path)
    if protocol_digest != FROZEN_PROTOCOL_SHA256:
        raise Wave0AValidationError("protocol bytes do not match the embedded frozen digest")
    if expected_protocol_sha256 is not None and expected_protocol_sha256.lower() != FROZEN_PROTOCOL_SHA256:
        raise Wave0AValidationError("caller digest disagrees with the embedded frozen digest")
    seeds = SELECT_SEEDS if phase == "SELECT" else REPORT_SEEDS
    parent = output_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    partial = parent / f".{output_directory.name}.partial-{uuid.uuid4().hex}"
    failure = {"stage_id": "CAMPAIGN", "seed": -1, "category": "CAMPAIGN_FAILURE"}
    execution_platform_system = "UNKNOWN"
    try:
        partial.mkdir()
        execution_platform_system = platform.system()
        linux_monitor, windows_monitor = _platform_monitor_flags(
            execution_platform_system
        )
        parent_rss = current_rss_bytes()
        config: Dict[str, object] = {
            "protocol_id": PROTOCOL_ID, "protocol_digest": protocol_digest, "phase": phase,
            "seeds": list(seeds), "stage_ids": list(STAGE_IDS),
            "budget": dict(vars(budget)),
            "worker_budget_policy": (
                "min_stage_campaign_aggregate_ceiling_minus_parent_worker_budget"
            ),
            "parent_rss_at_campaign_start_bytes": parent_rss,
            "ek0_parent_dependency_precheck": "find_spec_only_no_import_or_rss_guarantee",
            "ek0_import_admission_limitation": (
                "import_allocation_precedes_worker_measurement_and_may_oom_before_first_sample"
            ),
            "execution_intent": "cpu_only_intent_no_explicit_network_or_model_invocation",
            "execution_platform_system": execution_platform_system,
            "platform_attestation_scope": "unkeyed_self_attested_runtime_value",
            "rss_sample_interval_seconds": RSS_SAMPLE_INTERVAL_SECONDS,
            "rss_sampling_limitation": (
                "short_lived_touched_memory_overages_may_evade_sampling"
            ),
            "linux_sampled_process_tree_monitor": linux_monitor,
            "windows_sampled_direct_child_monitor_only": windows_monitor,
        }
        entries: List[Dict[str, object]] = []
        artifacts: List[Dict[str, object]] = []
        for seed in seeds:
            for stage_id in STAGE_IDS:
                failure = {"stage_id": stage_id, "seed": seed, "category": "WORKER_FAILURE"}
                refreshed_parent_rss = current_rss_bytes()
                dependency_precheck = "NOT_APPLICABLE"
                if stage_id == "PRW-EK0":
                    if not _ek0_dependency_specs_available():
                        failure["category"] = "DEPENDENCY_UNAVAILABLE"
                        raise Wave0AResourceError(
                            "EK0 dependency specification is unavailable before launch"
                        )
                    dependency_precheck = "AVAILABLE_FIND_SPEC_ONLY"
                try:
                    worker_reservation, sampled_backstop_ceiling = (
                        _stage_worker_reservation(
                            stage_id, refreshed_parent_rss, budget.max_estimated_bytes
                        )
                    )
                except Wave0AResourceError:
                    failure["category"] = "RESOURCE_OR_DEADLINE"
                    raise
                child_budget = Wave0ABudget(
                    max_estimated_bytes=worker_reservation,
                    max_seconds_per_stage=budget.max_seconds_per_stage,
                    max_output_bytes=budget.max_output_bytes,
                )
                relative = Path(f"seed-{seed}") / f"{stage_id.lower()}.json"
                path = partial / relative
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker-stage", stage_id,
                    "--worker-seed", str(seed), "--worker-phase", phase,
                    "--worker-protocol-digest", protocol_digest, "--worker-output", str(path),
                    "--worker-max-bytes", str(child_budget.max_estimated_bytes),
                    "--worker-max-seconds", str(child_budget.max_seconds_per_stage),
                    "--worker-max-output-bytes", str(child_budget.max_output_bytes),
                ]
                environment = dict(os.environ)
                environment.update({
                    "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
                })
                returncode, stdout, stderr, peak_rss = _execute_worker(
                    command, environment,
                    deadline_seconds=float(child_budget.max_seconds_per_stage),
                    effective_ceiling_bytes=sampled_backstop_ceiling,
                )
                if returncode != 0:
                    failure["category"] = "WORKER_NONZERO"
                    raise RuntimeError((stderr or stdout).strip())
                artifact = _read_verified_artifact(path, stage_id=stage_id, seed=seed, phase=phase)
                artifacts.append(artifact)
                entries.append({
                    "stage_id": stage_id, "seed": seed, "path": relative.as_posix(),
                    "artifact_digest": artifact["artifact_digest"], "file_sha256": sha256_file(path),
                    "status": "COMPLETE", "scientific_claim_status": "UNCONFIRMED",
                    "novelty_status": "UNCONFIRMED",
                    "sampled_peak_parent_plus_worker_rss_bytes": peak_rss,
                    "sampled_rss_backstop_ceiling_bytes": sampled_backstop_ceiling,
                    "parent_rss_before_stage_bytes": refreshed_parent_rss,
                    "worker_modeled_reservation_bytes": worker_reservation,
                    "parent_dependency_precheck": dependency_precheck,
                    "execution_platform_system": execution_platform_system,
                })
        summary = _campaign_summary(artifacts, phase)
        manifest: Dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA_VERSION, "config": config,
            "config_digest": sha256_payload(config), "entries": entries,
            "entry_count": len(entries), "summary": summary, "status": "COMPLETE",
        }
        manifest["manifest_digest"] = sha256_payload(manifest)
        atomic_write_json(partial / "manifest.json", manifest, budget.max_output_bytes)
        os.replace(partial, output_directory)
        return manifest
    except BaseException as error:
        if not partial.exists():
            raise
        category = "RESOURCE_OR_DEADLINE" if isinstance(error, Wave0AResourceError) else str(failure["category"])
        try:
            try:
                invalid = _invalid_run_payload(
                    phase=phase, stage_id=str(failure["stage_id"]),
                    seed=int(failure["seed"]), category=category,
                    exception_type=type(error).__name__, detail=str(error),
                    execution_platform_system=execution_platform_system,
                )
                atomic_write_json(
                    partial / "invalid_run.json", invalid,
                    INVALID_RUN_EVIDENCE_MAX_BYTES,
                )
            except BaseException:
                # Evidence failure must never replace the triggering exception.
                pass
            quarantine = parent / f"{output_directory.name}.QUARANTINED-{uuid.uuid4().hex}"
            if partial.exists():
                os.replace(partial, quarantine)
        except BaseException:
            # Quarantine failure also must not replace the triggering exception.
            pass
        raise


def run_worker(
    *, stage_id: str, seed: int, phase: str, protocol_digest: str,
    output: Path, budget: Wave0ABudget,
) -> None:
    if protocol_digest != FROZEN_PROTOCOL_SHA256:
        raise Wave0AValidationError("worker protocol digest is not canonical")
    artifact = make_stage_artifact(
        stage_id, seed=seed, phase=phase, protocol_digest=protocol_digest, budget=budget
    )
    verify_stage_artifact(artifact, stage_id=stage_id, seed=seed, phase=phase)
    atomic_write_json(output, artifact, budget.max_output_bytes)


def _verify_campaign(
    directory: Path, *, expected_phase: Optional[str], archived_cross_platform: bool
) -> Dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise Wave0AValidationError("manifest root must be an object")
    if archived_cross_platform and manifest_raw != (
        canonical_json(manifest) + "\n"
    ).encode("utf-8"):
        raise Wave0AValidationError("archived manifest bytes are not canonical")
    manifest_keys = {
        "schema_version", "config", "config_digest", "entries", "entry_count",
        "summary", "status", "manifest_digest",
    }
    if set(manifest) != manifest_keys:
        raise Wave0AValidationError("manifest schema keys are not exact")
    digest, without_digest = manifest.get("manifest_digest"), dict(manifest)
    without_digest.pop("manifest_digest", None)
    if type(digest) is not str or digest != sha256_payload(without_digest):
        raise Wave0AValidationError("manifest digest mismatch")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or manifest.get("status") != "COMPLETE":
        raise Wave0AValidationError("manifest schema/status mismatch")
    config = manifest.get("config")
    if not isinstance(config, dict) or manifest.get("config_digest") != sha256_payload(config):
        raise Wave0AValidationError("config digest mismatch")
    config_keys = {
        "protocol_id", "protocol_digest", "phase", "seeds", "stage_ids", "budget",
        "worker_budget_policy", "parent_rss_at_campaign_start_bytes", "execution_intent",
        "ek0_parent_dependency_precheck", "ek0_import_admission_limitation",
        "execution_platform_system", "platform_attestation_scope",
        "rss_sample_interval_seconds", "rss_sampling_limitation",
        "linux_sampled_process_tree_monitor",
        "windows_sampled_direct_child_monitor_only",
    }
    if set(config) != config_keys:
        raise Wave0AValidationError("config schema keys are not exact")
    phase = config.get("phase")
    if phase not in ("SELECT", "REPORT") or (expected_phase and phase != expected_phase):
        raise Wave0AValidationError("campaign phase mismatch")
    if archived_cross_platform and hashlib.sha256(manifest_raw).hexdigest() != (
        ARCHIVE_MANIFEST_FILE_SHA256[phase]
    ):
        raise Wave0AValidationError("archived manifest custody root SHA-256 mismatch")
    if config.get("protocol_id") != PROTOCOL_ID or config.get("protocol_digest") != FROZEN_PROTOCOL_SHA256:
        raise Wave0AValidationError("campaign protocol mismatch")
    if config.get("worker_budget_policy") != (
        "min_stage_campaign_aggregate_ceiling_minus_parent_worker_budget"
    ):
        raise Wave0AValidationError("campaign worker budget policy mismatch")
    if (
        config.get("ek0_parent_dependency_precheck")
        != "find_spec_only_no_import_or_rss_guarantee"
        or config.get("ek0_import_admission_limitation")
        != "import_allocation_precedes_worker_measurement_and_may_oom_before_first_sample"
    ):
        raise Wave0AValidationError("campaign EK0 admission boundary mismatch")
    if config.get("execution_intent") != (
        "cpu_only_intent_no_explicit_network_or_model_invocation"
    ):
        raise Wave0AValidationError("campaign execution-intent boundary mismatch")
    recorded_platform = config.get("execution_platform_system")
    if config.get("platform_attestation_scope") != "unkeyed_self_attested_runtime_value":
        raise Wave0AValidationError("campaign platform-attestation scope mismatch")
    expected_linux_monitor, expected_windows_monitor = _platform_monitor_flags(
        str(recorded_platform)
    )
    if (
        config.get("rss_sample_interval_seconds") != RSS_SAMPLE_INTERVAL_SECONDS
        or config.get("rss_sampling_limitation")
        != "short_lived_touched_memory_overages_may_evade_sampling"
        or type(config.get("linux_sampled_process_tree_monitor")) is not bool
        or type(config.get("windows_sampled_direct_child_monitor_only")) is not bool
        or config["linux_sampled_process_tree_monitor"] is not expected_linux_monitor
        or config["windows_sampled_direct_child_monitor_only"]
        is not expected_windows_monitor
    ):
        raise Wave0AValidationError("campaign process-monitor scope is invalid")
    if type(config.get("parent_rss_at_campaign_start_bytes")) is not int or config[
        "parent_rss_at_campaign_start_bytes"
    ] < 0:
        raise Wave0AValidationError("campaign parent RSS is invalid")
    campaign_budget = config.get("budget")
    if not isinstance(campaign_budget, dict):
        raise Wave0AValidationError("campaign budget is invalid")
    Wave0ABudget(**campaign_budget)
    expected_seeds = SELECT_SEEDS if phase == "SELECT" else REPORT_SEEDS
    if config.get("seeds") != list(expected_seeds) or config.get("stage_ids") != list(STAGE_IDS):
        raise Wave0AValidationError("campaign seed/stage grid mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or manifest.get("entry_count") != len(entries):
        raise Wave0AValidationError("manifest entry count mismatch")
    if archived_cross_platform:
        expected_files = {"manifest.json"}
        for entry in entries:
            if not isinstance(entry, dict):
                raise Wave0AValidationError("manifest entry must be an object")
            expected_files.add(_validate_archive_entry_path(entry.get("path")))
        if len(expected_files) != len(entries) + 1:
            raise Wave0AValidationError("archived manifest contains duplicate file paths")
        if _archive_regular_file_set(directory) != expected_files:
            raise Wave0AValidationError("archived campaign regular-file set mismatch")
    expected_pairs = {(seed, stage) for seed in expected_seeds for stage in STAGE_IDS}
    artifacts, observed_pairs = [], set()
    regeneration_differences: List[float] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise Wave0AValidationError("manifest entry must be an object")
        entry_keys = {
            "stage_id", "seed", "path", "artifact_digest", "file_sha256", "status",
            "scientific_claim_status", "novelty_status",
            "sampled_peak_parent_plus_worker_rss_bytes",
            "sampled_rss_backstop_ceiling_bytes", "parent_rss_before_stage_bytes",
            "worker_modeled_reservation_bytes", "parent_dependency_precheck",
            "execution_platform_system",
        }
        if set(entry) != entry_keys:
            raise Wave0AValidationError("manifest entry schema keys are not exact")
        seed, stage = entry.get("seed"), entry.get("stage_id")
        if type(seed) is not int or type(stage) is not str:
            raise Wave0AValidationError("manifest entry identity types are invalid")
        peak = entry.get("sampled_peak_parent_plus_worker_rss_bytes")
        sampled_backstop = entry.get("sampled_rss_backstop_ceiling_bytes")
        parent_before = entry.get("parent_rss_before_stage_bytes")
        worker_reservation = entry.get("worker_modeled_reservation_bytes")
        if any(
            type(value) is not int
            for value in (peak, sampled_backstop, parent_before, worker_reservation)
        ):
            raise Wave0AValidationError("manifest resource telemetry types are invalid")
        if entry.get("execution_platform_system") != recorded_platform:
            raise Wave0AValidationError("manifest entry platform does not match config")
        if stage == "PRW-EK0" and campaign_budget["max_estimated_bytes"] < (
            STAGE_MAX_BYTES[stage]
        ):
            raise Wave0AValidationError("campaign does not retain the frozen EK0 ceiling")
        expected_sampled_backstop = min(
            STAGE_MAX_BYTES.get(stage, -1), campaign_budget["max_estimated_bytes"]
        )
        expected_worker_reservation = expected_sampled_backstop - parent_before
        expected_dependency_precheck = (
            "AVAILABLE_FIND_SPEC_ONLY" if stage == "PRW-EK0" else "NOT_APPLICABLE"
        )
        if (
            peak < 0 or sampled_backstop != expected_sampled_backstop
            or parent_before < 0 or worker_reservation != expected_worker_reservation
            or worker_reservation <= 0 or sampled_backstop > campaign_budget["max_estimated_bytes"]
            or peak > sampled_backstop
            or entry.get("parent_dependency_precheck") != expected_dependency_precheck
        ):
            raise Wave0AValidationError("manifest resource telemetry exceeds or breaks its ceiling")
        path = directory / str(entry.get("path"))
        if directory.resolve() not in path.resolve().parents:
            raise Wave0AValidationError("manifest entry path escapes campaign directory")
        if sha256_file(path) != entry.get("file_sha256"):
            raise Wave0AValidationError("stage file SHA-256 mismatch")
        if archived_cross_platform:
            artifact, regeneration = _read_verified_archived_artifact(
                path, stage_id=stage, seed=seed, phase=phase
            )
            regeneration_differences.extend(regeneration.values())
        else:
            artifact = _read_verified_artifact(
                path, stage_id=stage, seed=seed, phase=phase
            )
        if artifact["artifact_digest"] != entry.get("artifact_digest"):
            raise Wave0AValidationError("manifest/artifact digest mismatch")
        if artifact["execution_platform_system"] != recorded_platform:
            raise Wave0AValidationError("stage artifact platform does not match campaign")
        if artifact["budget"] != {
            "max_estimated_bytes": worker_reservation,
            "max_seconds_per_stage": campaign_budget["max_seconds_per_stage"],
            "max_output_bytes": campaign_budget["max_output_bytes"],
        }:
            raise Wave0AValidationError("manifest child budget does not match artifact budget")
        if (entry.get("status"), entry.get("scientific_claim_status"), entry.get("novelty_status")) != (
            "COMPLETE", "UNCONFIRMED", "UNCONFIRMED"
        ):
            raise Wave0AValidationError("manifest claim boundary mismatch")
        observed_pairs.add((seed, stage))
        artifacts.append(artifact)
    if observed_pairs != expected_pairs or len(observed_pairs) != len(entries):
        raise Wave0AValidationError("campaign grid is incomplete or duplicated")
    summary = _campaign_summary(artifacts, phase)
    if manifest.get("summary") != summary:
        raise Wave0AValidationError("campaign summary mismatch")
    if archived_cross_platform:
        maximum_difference = max(regeneration_differences, default=0.0)
        return {
            "status": "ARCHIVED_CAMPAIGN_VERIFIED",
            "phase": phase,
            "entry_count": len(entries),
            "summary": summary,
            "archive_source_platform_system": config["execution_platform_system"],
            "manifest_file_sha256_custody_root": ARCHIVE_MANIFEST_FILE_SHA256[phase],
            "custody_anchor_scope": (
                "code_distribution_anchor_not_signature_or_host_attestation"
            ),
            "current_platform_exact_regeneration": maximum_difference == 0.0,
            "maximum_absolute_regeneration_difference": maximum_difference,
            "frozen_archive_tolerances": {
                "PRW-SPU0": ARCHIVE_SPU0_ABS_TOLERANCE,
                "PRW-VAR1": ARCHIVE_VAR1_ABS_TOLERANCE,
            },
        }
    return {
        "status": "VERIFIED", "phase": phase, "entry_count": len(entries),
        "summary": summary,
    }


def verify_campaign(
    directory: Path, *, expected_phase: Optional[str] = None
) -> Dict[str, object]:
    """Exact live/current-platform campaign verification."""

    return _verify_campaign(
        directory, expected_phase=expected_phase, archived_cross_platform=False
    )


def verify_archived_campaign(
    directory: Path, *, expected_phase: Optional[str] = None
) -> Dict[str, object]:
    """Portable verification for immutable copied campaign archives."""

    return _verify_campaign(
        directory, expected_phase=expected_phase, archived_cross_platform=True
    )


def verify_cross_phase(select_directory: Path, report_directory: Path) -> Dict[str, object]:
    select = verify_campaign(select_directory, expected_phase="SELECT")
    report = verify_campaign(report_directory, expected_phase="REPORT")
    select_cells = select["summary"]["var1_corresponding_cell_seed_summary"]
    report_cells = report["summary"]["var1_corresponding_cell_seed_summary"]
    if set(select_cells) != set(report_cells):
        raise Wave0AValidationError("SELECT/REPORT VAR1 cell identities differ")
    survives = (
        select["summary"]["var1_null_survives_all_seeds_and_cells"]
        and report["summary"]["var1_null_survives_all_seeds_and_cells"]
    )
    return {
        "status": "VERIFIED", "fixture_scope": "synthetic_non_independent_labels",
        "select": select, "report": report,
        "corresponding_cell_mean_gap_delta": {
            cell: report_cells[cell]["mean_objective_gap"] - select_cells[cell]["mean_objective_gap"]
            for cell in sorted(select_cells)
        },
        "var1_cross_phase_null_survives": survives,
        "var1_cross_phase_disposition": (
            "RETAIN_SMOOTH_HARD_MASK_NULL" if survives
            else "ROUTE_SMOOTH_METHOD_TO_DECLARED_STATE_SUBPROBLEM_ONLY"
        ),
    }


def verify_archived_cross_phase(
    select_directory: Path, report_directory: Path
) -> Dict[str, object]:
    """Portable cross-phase verification that never reports live VERIFIED."""

    select = verify_archived_campaign(select_directory, expected_phase="SELECT")
    report = verify_archived_campaign(report_directory, expected_phase="REPORT")
    select_cells = select["summary"]["var1_corresponding_cell_seed_summary"]
    report_cells = report["summary"]["var1_corresponding_cell_seed_summary"]
    if set(select_cells) != set(report_cells):
        raise Wave0AValidationError("archived SELECT/REPORT VAR1 cell identities differ")
    survives = (
        select["summary"]["var1_null_survives_all_seeds_and_cells"]
        and report["summary"]["var1_null_survives_all_seeds_and_cells"]
    )
    return {
        "status": "ARCHIVED_CROSS_PHASE_VERIFIED",
        "fixture_scope": "synthetic_non_independent_labels",
        "select": select,
        "report": report,
        "current_platform_exact_regeneration": (
            select["current_platform_exact_regeneration"]
            and report["current_platform_exact_regeneration"]
        ),
        "maximum_absolute_regeneration_difference": max(
            select["maximum_absolute_regeneration_difference"],
            report["maximum_absolute_regeneration_difference"],
        ),
        "corresponding_cell_mean_gap_delta": {
            cell: report_cells[cell]["mean_objective_gap"]
            - select_cells[cell]["mean_objective_gap"]
            for cell in sorted(select_cells)
        },
        "var1_cross_phase_null_survives": survives,
        "var1_cross_phase_disposition": (
            "RETAIN_SMOOTH_HARD_MASK_NULL"
            if survives
            else "ROUTE_SMOOTH_METHOD_TO_DECLARED_STATE_SUBPROBLEM_ONLY"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("SELECT", "REPORT"))
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--verify-campaign", type=Path)
    parser.add_argument("--verify-cross-phase", nargs=2, type=Path)
    parser.add_argument("--verify-archived-campaign", type=Path)
    parser.add_argument("--verify-archived-cross-phase", nargs=2, type=Path)
    parser.add_argument("--worker-stage", choices=STAGE_IDS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-phase", help=argparse.SUPPRESS)
    parser.add_argument("--worker-protocol-digest", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-max-bytes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-max-seconds", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-max-output-bytes", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_archived_campaign:
        print(json.dumps(
            verify_archived_campaign(args.verify_archived_campaign),
            sort_keys=True, indent=2,
        ))
        return 0
    if args.verify_archived_cross_phase:
        print(json.dumps(
            verify_archived_cross_phase(*args.verify_archived_cross_phase),
            sort_keys=True, indent=2,
        ))
        return 0
    if args.verify_campaign:
        print(json.dumps(verify_campaign(args.verify_campaign), sort_keys=True, indent=2))
        return 0
    if args.verify_cross_phase:
        print(json.dumps(verify_cross_phase(*args.verify_cross_phase), sort_keys=True, indent=2))
        return 0
    if args.worker_stage:
        required = (
            args.worker_seed, args.worker_phase, args.worker_protocol_digest, args.worker_output,
            args.worker_max_bytes, args.worker_max_seconds, args.worker_max_output_bytes,
        )
        if any(value is None for value in required):
            parser.error("worker invocation is incomplete")
        run_worker(
            stage_id=args.worker_stage, seed=args.worker_seed, phase=args.worker_phase,
            protocol_digest=args.worker_protocol_digest, output=args.worker_output,
            budget=Wave0ABudget(
                max_estimated_bytes=args.worker_max_bytes,
                max_seconds_per_stage=args.worker_max_seconds,
                max_output_bytes=args.worker_max_output_bytes,
            ),
        )
        return 0
    if args.phase is None or args.output_directory is None:
        parser.error("--phase and --output-directory are required")
    manifest = run(
        output_directory=args.output_directory, phase=args.phase, protocol_path=args.protocol,
        expected_protocol_sha256=args.protocol_sha256,
    )
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
