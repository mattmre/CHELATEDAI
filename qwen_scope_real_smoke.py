"""Fail-closed real-model smoke for one pinned Qwen-Scope residual SAE.

This module proves only that a pinned model and official SAE can be loaded and
connected at the declared residual-stream layer.  It does not evaluate model
quality, semantic usefulness, training gains, chelation, or novelty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


SCHEMA_VERSION = "qwen-scope-real-smoke.v1"
OPT_IN_ENV = "CHELATED_QWEN_SCOPE_REAL_SMOKE"
MODEL_REPO = "Qwen/Qwen3.5-2B-Base"
MODEL_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100"
SAE_REVISION = "027267657257a8d490296286e8fab41e1c1a1a3d"
SAE_FILENAME = "layer11.sae.pt"
SAE_SHA256 = "d1828ace348b13cca9104f61fb47672e439e963d9d5fc5496f4c6b068a06499f"
LAYER_INDEX = 11
HIDDEN_SIZE = 2048
SAE_WIDTH = 32768
TOP_K = 100
FIXED_PROMPT = "A careful measurement should be repeatable because"
MIN_TRANSFORMERS_VERSION = (4, 57, 0)
FROZEN_MAX_WALL_SECONDS = 1800.0
FROZEN_MAX_RSS_GIB = 32.0
FROZEN_MIN_FREE_DISK_GIB = 12.0
FROZEN_MIN_FREE_GPU_GIB = 12.0
FROZEN_FEATURE_ATOL = 1e-4
ARTIFACT_FILENAME = "qwen_scope_real_smoke.json"
MANIFEST_FILENAME = "manifest.json"
COMMIT_RECEIPT_FILENAME = "COMMIT.json"
MAX_ARTIFACT_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_COMMIT_RECEIPT_BYTES = 64 * 1024


class SmokeFailure(RuntimeError):
    """Raised when any frozen smoke contract is not met."""


@dataclass(frozen=True)
class SmokeConfig:
    output_dir: Path
    offline: bool = False
    max_wall_seconds: float = FROZEN_MAX_WALL_SECONDS
    max_rss_gib: float = FROZEN_MAX_RSS_GIB
    min_free_disk_gib: float = FROZEN_MIN_FREE_DISK_GIB
    min_free_gpu_gib: float = FROZEN_MIN_FREE_GPU_GIB
    feature_atol: float = FROZEN_FEATURE_ATOL
    model_repo: str = MODEL_REPO
    model_revision: str = MODEL_REVISION
    sae_repo: str = SAE_REPO
    sae_revision: str = SAE_REVISION
    sae_filename: str = SAE_FILENAME
    sae_sha256: str = SAE_SHA256
    layer_index: int = LAYER_INDEX
    hidden_size: int = HIDDEN_SIZE
    sae_width: int = SAE_WIDTH
    top_k: int = TOP_K


@dataclass(frozen=True)
class PassObservation:
    residual_shape: tuple[int, ...]
    residual_digest: str
    selected_indices: tuple[int, ...]
    selected_values: tuple[float, ...]


class SmokeBackend(Protocol):
    def preflight(self, config: SmokeConfig) -> dict[str, Any]: ...

    def load(self, config: SmokeConfig) -> dict[str, Any]: ...

    def run_pass(self, prompt: str, config: SmokeConfig) -> PassObservation: ...

    def measurements(self) -> dict[str, Any]: ...


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _strict_json(raw: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SmokeFailure(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeFailure(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _prepare_commit_receipt(final_dir: Path, payload: Any) -> Path:
    """Serialize, write, and fsync a hidden receipt candidate without publishing it."""

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{COMMIT_RECEIPT_FILENAME}.", suffix=".prepared", dir=final_dir
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return temporary_path


def _publish_commit_receipt(prepared_path: Path, final_path: Path) -> None:
    """Atomically publish a prepared receipt as the final lifecycle operation."""

    _promote_directory_noreplace(prepared_path, final_path)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _promote_directory_noreplace(stage_dir: Path, final_dir: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""

    if os.name == "posix":
        import ctypes
        import errno

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise SmokeFailure("renameat2(RENAME_NOREPLACE) is required for fail-closed promotion")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(stage_dir),
            -100,
            os.fsencode(final_dir),
            1,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise SmokeFailure("final output target appeared before promotion; refusing replacement")
            raise OSError(error, os.strerror(error), str(final_dir))
        return
    os.rename(stage_dir, final_dir)


def _require_finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SmokeFailure(f"{label} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SmokeFailure(f"{label} must be finite")
    return numeric


def _validate_config(config: SmokeConfig) -> None:
    if not isinstance(config.output_dir, Path):
        raise SmokeFailure("output_dir must be a pathlib.Path")
    if not isinstance(config.offline, bool):
        raise SmokeFailure("offline must be boolean")
    wall = _require_finite_number(config.max_wall_seconds, label="max_wall_seconds")
    rss = _require_finite_number(config.max_rss_gib, label="max_rss_gib")
    disk = _require_finite_number(config.min_free_disk_gib, label="min_free_disk_gib")
    gpu = _require_finite_number(config.min_free_gpu_gib, label="min_free_gpu_gib")
    atol = _require_finite_number(config.feature_atol, label="feature_atol")
    if not (0 < wall <= FROZEN_MAX_WALL_SECONDS):
        raise SmokeFailure(f"max_wall_seconds must be in (0, {FROZEN_MAX_WALL_SECONDS:g}]")
    if not (0 < rss <= FROZEN_MAX_RSS_GIB):
        raise SmokeFailure(f"max_rss_gib must be in (0, {FROZEN_MAX_RSS_GIB:g}]")
    if disk < FROZEN_MIN_FREE_DISK_GIB:
        raise SmokeFailure(f"min_free_disk_gib must be >= {FROZEN_MIN_FREE_DISK_GIB:g}")
    if gpu < FROZEN_MIN_FREE_GPU_GIB:
        raise SmokeFailure(f"min_free_gpu_gib must be >= {FROZEN_MIN_FREE_GPU_GIB:g}")
    if atol != FROZEN_FEATURE_ATOL:
        raise SmokeFailure(f"feature_atol is frozen at {FROZEN_FEATURE_ATOL:g}")
    frozen_identity = {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "sae_repo": SAE_REPO,
        "sae_revision": SAE_REVISION,
        "sae_filename": SAE_FILENAME,
        "sae_sha256": SAE_SHA256,
        "layer_index": LAYER_INDEX,
        "hidden_size": HIDDEN_SIZE,
        "sae_width": SAE_WIDTH,
        "top_k": TOP_K,
    }
    for field_name, expected in frozen_identity.items():
        if getattr(config, field_name) != expected:
            raise SmokeFailure(f"{field_name} is frozen at {expected!r}")


def _validate_exact_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SmokeFailure(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise SmokeFailure(f"{label} keys differ: expected {sorted(expected)}, got {sorted(actual)}")
    return value


def _check_deadline(started: float, config: SmokeConfig, phase: str) -> None:
    elapsed = time.monotonic() - started
    if elapsed > config.max_wall_seconds:
        raise SmokeFailure(
            f"cooperative wall-time ceiling exceeded after {phase}: "
            f"{elapsed:.3f}s > {config.max_wall_seconds:.3f}s"
        )


def _max_abs_delta(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        return float("inf")
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def _numeric_version_prefix(value: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for component in value.split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    return tuple((parts + [0, 0, 0])[:3])


def _verify_official_sae_state(state: Any, config: SmokeConfig) -> None:
    """Validate all four tensors documented by the official model card."""

    if not isinstance(state, dict):
        raise SmokeFailure("official SAE checkpoint root must be a dict")
    expected_shapes = {
        "W_enc": (config.sae_width, config.hidden_size),
        "W_dec": (config.hidden_size, config.sae_width),
        "b_enc": (config.sae_width,),
        "b_dec": (config.hidden_size,),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in state:
            raise SmokeFailure(f"official SAE checkpoint is missing {key}")
        actual_shape = tuple(int(value) for value in state[key].shape)
        if actual_shape != expected_shape:
            raise SmokeFailure(
                f"official SAE {key} shape {actual_shape} does not match {expected_shape}"
            )


def _verify_result_dir(
    result_dir: str | Path,
    *,
    allow_staged: bool,
    require_current_target: bool,
) -> dict[str, Any]:
    """Verify result bytes; staged paths are private pre-promotion inputs only."""

    root = Path(result_dir)
    is_known_stage = root.name.startswith(".") and ".stage-" in root.name
    if is_known_stage and not allow_staged:
        raise SmokeFailure("hidden sibling staging directories are not promoted PASS evidence")
    if not root.is_dir() or root.is_symlink():
        raise SmokeFailure("result path must be a non-symlink directory")
    entries = sorted(path.name for path in root.iterdir())
    expected_entries = sorted([MANIFEST_FILENAME, ARTIFACT_FILENAME])
    if not allow_staged:
        expected_entries = sorted([*expected_entries, COMMIT_RECEIPT_FILENAME])
    if entries != expected_entries:
        raise SmokeFailure(f"result directory entries differ from frozen lifecycle set: {entries}")
    artifact_path = root / ARTIFACT_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    checked_paths = [
        (artifact_path, MAX_ARTIFACT_BYTES, "artifact"),
        (manifest_path, MAX_MANIFEST_BYTES, "manifest"),
    ]
    receipt_path = root / COMMIT_RECEIPT_FILENAME
    if not allow_staged:
        checked_paths.append((receipt_path, MAX_COMMIT_RECEIPT_BYTES, "commit receipt"))
    for path, maximum, label in checked_paths:
        if path.is_symlink() or not path.is_file():
            raise SmokeFailure(f"{label} must be a regular non-symlink file")
        size = path.stat().st_size
        if not (0 < size <= maximum):
            raise SmokeFailure(f"{label} byte size {size} is outside (0, {maximum}]")

    artifact_raw = artifact_path.read_bytes()
    manifest_raw = manifest_path.read_bytes()
    receipt_raw = receipt_path.read_bytes() if not allow_staged else None
    artifact = _strict_json(artifact_raw, label="artifact")
    manifest = _strict_json(manifest_raw, label="manifest")
    receipt = _strict_json(receipt_raw, label="commit receipt") if receipt_raw is not None else None
    if artifact_raw != _canonical_json(artifact):
        raise SmokeFailure("artifact bytes are not canonical JSON")
    if manifest_raw != _canonical_json(manifest):
        raise SmokeFailure("manifest bytes are not canonical JSON")
    if receipt_raw is not None and receipt_raw != _canonical_json(receipt):
        raise SmokeFailure("commit receipt bytes are not canonical JSON")

    artifact = _validate_exact_keys(
        artifact,
        {
            "schema_version",
            "status",
            "lifecycle_id",
            "claim_boundary",
            "inputs",
            "tolerances",
            "preflight",
            "load_contract",
            "observation",
            "resources",
        },
        label="artifact",
    )
    manifest = _validate_exact_keys(
        manifest,
        {"schema_version", "status", "lifecycle_id", "artifacts", "source_contract", "commit_timing"},
        label="manifest",
    )
    if artifact["schema_version"] != SCHEMA_VERSION or manifest["schema_version"] != SCHEMA_VERSION:
        raise SmokeFailure("artifact and manifest schema versions must equal the frozen schema")
    if artifact["status"] != "PASS" or manifest["status"] != "PASS":
        raise SmokeFailure("artifact and manifest must both duplicate status PASS")
    lifecycle_id = artifact["lifecycle_id"]
    if not isinstance(lifecycle_id, str) or len(lifecycle_id) != 32 or any(
        character not in "0123456789abcdef" for character in lifecycle_id
    ):
        raise SmokeFailure("artifact lifecycle_id must be 128-bit lowercase hexadecimal")
    if manifest["lifecycle_id"] != lifecycle_id:
        raise SmokeFailure("artifact and manifest lifecycle IDs disagree")

    claim = _validate_exact_keys(
        artifact["claim_boundary"], {"proves", "does_not_prove"}, label="claim_boundary"
    )
    expected_proves = [
        "pinned real model load",
        "exact residual layer hook",
        "official SAE checkpoint contract",
        "top-k sparse encoding",
        "fixed-input repeatability within declared tolerance",
        "artifact and resource measurement path",
    ]
    expected_exclusions = [
        "chelation utility",
        "training gain",
        "semantic improvement",
        "production readiness",
        "novelty",
    ]
    if claim["proves"] != expected_proves or claim["does_not_prove"] != expected_exclusions:
        raise SmokeFailure("claim boundary differs from the frozen claim boundary")

    inputs = _validate_exact_keys(
        artifact["inputs"],
        {
            "model_repo",
            "model_revision",
            "sae_repo",
            "sae_revision",
            "sae_filename",
            "sae_sha256",
            "layer_index",
            "hidden_size",
            "sae_width",
            "top_k",
            "prompt_sha256",
            "prompt_retained",
            "offline",
        },
        label="inputs",
    )
    expected_inputs = {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "sae_repo": SAE_REPO,
        "sae_revision": SAE_REVISION,
        "sae_filename": SAE_FILENAME,
        "sae_sha256": SAE_SHA256,
        "layer_index": LAYER_INDEX,
        "hidden_size": HIDDEN_SIZE,
        "sae_width": SAE_WIDTH,
        "top_k": TOP_K,
        "prompt_sha256": _sha256_bytes(FIXED_PROMPT.encode("utf-8")),
        "prompt_retained": False,
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise SmokeFailure(f"artifact source/input identity {key} differs from frozen value")
    if not isinstance(inputs["offline"], bool):
        raise SmokeFailure("inputs.offline must be boolean")

    tolerances = _validate_exact_keys(
        artifact["tolerances"], {"residual_repeatability", "feature_atol"}, label="tolerances"
    )
    if tolerances["residual_repeatability"] != "exact_float32_tensor_sha256":
        raise SmokeFailure("residual repeatability contract differs")
    if _require_finite_number(tolerances["feature_atol"], label="feature_atol") != FROZEN_FEATURE_ATOL:
        raise SmokeFailure("retained feature_atol differs from frozen value")

    preflight = _validate_exact_keys(
        artifact["preflight"],
        {"cuda_device", "free_gpu_gib", "total_gpu_gib", "free_disk_gib"},
        label="preflight",
    )
    if not isinstance(preflight["cuda_device"], str) or not preflight["cuda_device"]:
        raise SmokeFailure("preflight.cuda_device must be non-empty")
    free_gpu = _require_finite_number(preflight["free_gpu_gib"], label="preflight.free_gpu_gib")
    total_gpu = _require_finite_number(preflight["total_gpu_gib"], label="preflight.total_gpu_gib")
    free_disk = _require_finite_number(preflight["free_disk_gib"], label="preflight.free_disk_gib")
    if free_gpu < FROZEN_MIN_FREE_GPU_GIB or total_gpu < free_gpu:
        raise SmokeFailure("retained GPU preflight violates frozen bounds")
    if free_disk < FROZEN_MIN_FREE_DISK_GIB:
        raise SmokeFailure("retained disk preflight violates frozen bounds")

    load = _validate_exact_keys(
        artifact["load_contract"],
        {
            "model_class",
            "model_layer_count",
            "sae_d_model",
            "sae_d_sae",
            "sae_top_k",
            "sae_file_sha256_verified",
            "sae_four_tensor_contract_verified",
            "hook_layer_index",
            "transformers_version",
        },
        label="load_contract",
    )
    load_expected = {
        "sae_d_model": HIDDEN_SIZE,
        "sae_d_sae": SAE_WIDTH,
        "sae_top_k": TOP_K,
        "sae_file_sha256_verified": True,
        "sae_four_tensor_contract_verified": True,
        "hook_layer_index": LAYER_INDEX,
    }
    for key, expected in load_expected.items():
        if load.get(key) != expected:
            raise SmokeFailure(f"retained load contract {key} differs")
    if not isinstance(load["model_class"], str) or not load["model_class"]:
        raise SmokeFailure("load_contract.model_class must be non-empty")
    if not isinstance(load["model_layer_count"], int) or isinstance(load["model_layer_count"], bool):
        raise SmokeFailure("load_contract.model_layer_count must be an integer")
    if load["model_layer_count"] <= LAYER_INDEX:
        raise SmokeFailure("retained model layer count does not contain hook layer")
    if not isinstance(load["transformers_version"], str) or (
        _numeric_version_prefix(load["transformers_version"]) < MIN_TRANSFORMERS_VERSION
    ):
        raise SmokeFailure("retained transformers version is unsupported")

    observation = _validate_exact_keys(
        artifact["observation"],
        {
            "residual_shape",
            "residual_sha256",
            "selected_feature_count",
            "selected_feature_ids",
            "repeat_feature_max_abs_delta",
        },
        label="observation",
    )
    shape = observation["residual_shape"]
    if not isinstance(shape, list) or len(shape) != 3 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape
    ):
        raise SmokeFailure("retained residual shape is invalid")
    if shape[-1] != HIDDEN_SIZE:
        raise SmokeFailure("retained residual hidden size differs")
    residual_digest = observation["residual_sha256"]
    if not isinstance(residual_digest, str) or len(residual_digest) != 64 or any(
        character not in "0123456789abcdef" for character in residual_digest
    ):
        raise SmokeFailure("retained residual digest is invalid")
    feature_ids = observation["selected_feature_ids"]
    if observation["selected_feature_count"] != TOP_K or not isinstance(feature_ids, list):
        raise SmokeFailure("retained selected feature count differs")
    if len(feature_ids) != TOP_K or len(set(feature_ids)) != TOP_K or any(
        not isinstance(value, int) or isinstance(value, bool) or not (0 <= value < SAE_WIDTH)
        for value in feature_ids
    ):
        raise SmokeFailure("retained selected feature IDs are invalid")
    delta = _require_finite_number(
        observation["repeat_feature_max_abs_delta"], label="repeat_feature_max_abs_delta"
    )
    if delta < 0 or delta > FROZEN_FEATURE_ATOL:
        raise SmokeFailure("retained repeat feature delta violates frozen tolerance")

    resources = _validate_exact_keys(
        artifact["resources"],
        {
            "peak_rss_gib",
            "peak_cuda_allocated_gib",
            "peak_cuda_reserved_gib",
            "backend_elapsed_seconds",
            "pre_persistence_wall_seconds",
            "max_wall_seconds",
            "max_rss_gib",
            "min_free_disk_gib",
            "min_free_gpu_gib",
        },
        label="resources",
    )
    numeric_resources = {
        key: _require_finite_number(value, label=f"resources.{key}")
        for key, value in resources.items()
    }
    if any(value < 0 for key, value in numeric_resources.items() if not key.startswith("min_free")):
        raise SmokeFailure("retained resource measurement is negative")
    if not (0 < numeric_resources["max_wall_seconds"] <= FROZEN_MAX_WALL_SECONDS):
        raise SmokeFailure("retained max wall gate violates frozen bound")
    if not (0 < numeric_resources["max_rss_gib"] <= FROZEN_MAX_RSS_GIB):
        raise SmokeFailure("retained max RSS gate violates frozen bound")
    if numeric_resources["min_free_disk_gib"] < FROZEN_MIN_FREE_DISK_GIB:
        raise SmokeFailure("retained disk floor weakens frozen bound")
    if numeric_resources["min_free_gpu_gib"] < FROZEN_MIN_FREE_GPU_GIB:
        raise SmokeFailure("retained GPU floor weakens frozen bound")
    if free_disk < numeric_resources["min_free_disk_gib"]:
        raise SmokeFailure("retained free disk does not meet retained disk floor")
    if free_gpu < numeric_resources["min_free_gpu_gib"]:
        raise SmokeFailure("retained free GPU memory does not meet retained GPU floor")
    if numeric_resources["pre_persistence_wall_seconds"] > numeric_resources["max_wall_seconds"]:
        raise SmokeFailure("retained pre-persistence wall measurement exceeds retained gate")
    if numeric_resources["peak_rss_gib"] > numeric_resources["max_rss_gib"]:
        raise SmokeFailure("retained RSS measurement exceeds retained gate")

    source = _validate_exact_keys(
        manifest["source_contract"],
        {"model_revision", "sae_revision", "sae_file_sha256"},
        label="source_contract",
    )
    expected_source = {
        "model_revision": MODEL_REVISION,
        "sae_revision": SAE_REVISION,
        "sae_file_sha256": SAE_SHA256,
    }
    if source != expected_source:
        raise SmokeFailure("manifest source identity differs from frozen source identity")
    if source["model_revision"] != inputs["model_revision"] or source["sae_revision"] != inputs["sae_revision"]:
        raise SmokeFailure("duplicated source revisions disagree")
    if source["sae_file_sha256"] != inputs["sae_sha256"]:
        raise SmokeFailure("duplicated SAE digest disagrees")

    timing = _validate_exact_keys(
        manifest["commit_timing"],
        {"checkpoint", "precommit_checkpoint_wall_seconds", "live_deadline_rechecked_before_promotion"},
        label="commit_timing",
    )
    if timing["checkpoint"] != "after_initial_stage_verify_fsync_before_final_manifest_rebuild":
        raise SmokeFailure("manifest commit timing checkpoint differs")
    precommit_elapsed = _require_finite_number(
        timing["precommit_checkpoint_wall_seconds"], label="precommit_checkpoint_wall_seconds"
    )
    if precommit_elapsed < numeric_resources["pre_persistence_wall_seconds"]:
        raise SmokeFailure("retained precommit checkpoint predates pre-persistence checkpoint")
    if precommit_elapsed > numeric_resources["max_wall_seconds"]:
        raise SmokeFailure("retained precommit checkpoint exceeds retained wall gate")
    if timing["live_deadline_rechecked_before_promotion"] is not True:
        raise SmokeFailure("manifest must declare the live pre-promotion deadline recheck")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise SmokeFailure("manifest must contain exactly one artifact entry")
    entry = _validate_exact_keys(artifacts[0], {"path", "byte_size", "sha256"}, label="artifact entry")
    if entry["path"] != ARTIFACT_FILENAME:
        raise SmokeFailure("manifest artifact path differs from frozen safe relative path")
    safe_path = root / entry["path"]
    if safe_path.parent.resolve() != root.resolve() or safe_path.resolve() != artifact_path.resolve():
        raise SmokeFailure("manifest artifact path escapes result directory")
    if not isinstance(entry["byte_size"], int) or isinstance(entry["byte_size"], bool):
        raise SmokeFailure("manifest artifact byte_size must be an integer")
    if entry["byte_size"] != len(artifact_raw):
        raise SmokeFailure("manifest artifact byte_size does not match canonical bytes")
    actual_digest = _sha256_bytes(artifact_raw)
    if entry["sha256"] != actual_digest:
        raise SmokeFailure("manifest artifact digest does not match canonical bytes")
    if not allow_staged:
        receipt = _validate_exact_keys(
            receipt,
            {"schema_version", "status", "lifecycle_id", "lifecycle", "target", "bindings", "deadline"},
            label="commit receipt",
        )
        if receipt["schema_version"] != SCHEMA_VERSION or receipt["status"] != "COMMITTED":
            raise SmokeFailure("commit receipt schema/status differs from frozen committed state")
        if receipt["lifecycle_id"] != lifecycle_id:
            raise SmokeFailure("commit receipt lifecycle ID disagrees")
        if receipt["lifecycle"] != "created_after_no_replace_rename_and_parent_fsync":
            raise SmokeFailure("commit receipt lifecycle differs from frozen lifecycle")
        target = _validate_exact_keys(
            receipt["target"], {"directory_name", "resolved_path_sha256"}, label="commit target"
        )
        source_directory_name = target["directory_name"]
        source_target_digest = target["resolved_path_sha256"]
        if (
            not isinstance(source_directory_name, str)
            or not source_directory_name
            or source_directory_name in {".", ".."}
            or Path(source_directory_name).name != source_directory_name
        ):
            raise SmokeFailure("commit receipt recorded source directory name is unsafe")
        if (
            not isinstance(source_target_digest, str)
            or len(source_target_digest) != 64
            or any(character not in "0123456789abcdef" for character in source_target_digest)
        ):
            raise SmokeFailure("commit receipt recorded source target digest is invalid")
        if require_current_target:
            expected_target_digest = _sha256_bytes(str(root.resolve()).encode("utf-8"))
            if source_directory_name != root.name or source_target_digest != expected_target_digest:
                raise SmokeFailure("commit receipt target identity does not match result directory")
        bindings = _validate_exact_keys(
            receipt["bindings"],
            {"artifact_sha256", "artifact_byte_size", "manifest_sha256", "manifest_byte_size"},
            label="commit bindings",
        )
        expected_bindings = {
            "artifact_sha256": actual_digest,
            "artifact_byte_size": len(artifact_raw),
            "manifest_sha256": _sha256_bytes(manifest_raw),
            "manifest_byte_size": len(manifest_raw),
        }
        if bindings != expected_bindings:
            raise SmokeFailure("commit receipt does not bind the exact artifact and manifest bytes")
        deadline = _validate_exact_keys(
            receipt["deadline"],
            {
                "pre_receipt_checkpoint_elapsed_seconds",
                "max_wall_seconds",
                "within_cooperative_deadline_at_pre_receipt_checkpoint",
            },
            label="commit deadline",
        )
        pre_receipt_elapsed = _require_finite_number(
            deadline["pre_receipt_checkpoint_elapsed_seconds"],
            label="pre_receipt_checkpoint_elapsed_seconds",
        )
        receipt_max_wall = _require_finite_number(
            deadline["max_wall_seconds"], label="commit max_wall_seconds"
        )
        if receipt_max_wall != numeric_resources["max_wall_seconds"]:
            raise SmokeFailure("commit receipt wall gate disagrees with artifact")
        if pre_receipt_elapsed < precommit_elapsed or pre_receipt_elapsed > receipt_max_wall:
            raise SmokeFailure("pre-receipt checkpoint violates retained deadline sequence")
        if deadline["within_cooperative_deadline_at_pre_receipt_checkpoint"] is not True:
            raise SmokeFailure("pre-receipt cooperative deadline predicate is not true")
    result = {
        "status": "PASS",
        "artifact_sha256": actual_digest,
        "artifact_byte_size": len(artifact_raw),
        "manifest_sha256": _sha256_bytes(manifest_raw),
        "manifest_byte_size": len(manifest_raw),
        "model_revision": inputs["model_revision"],
        "sae_revision": inputs["sae_revision"],
        "lifecycle_id": lifecycle_id,
    }
    if receipt_raw is not None:
        result["receipt_sha256"] = _sha256_bytes(receipt_raw)
        result["receipt_byte_size"] = len(receipt_raw)
        result["recorded_source_target"] = {
            "directory_name": receipt["target"]["directory_name"],
            "resolved_path_sha256": receipt["target"]["resolved_path_sha256"],
        }
    return result


def verify_result_dir(result_dir: str | Path) -> dict[str, Any]:
    """Distrust and verify one promoted result; never accept staging paths."""

    verified = _verify_result_dir(result_dir, allow_staged=False, require_current_target=True)
    # Preserve the established public live-verifier response while archive
    # custody mode reports digests for all three transported members.
    verified.pop("manifest_sha256")
    verified.pop("manifest_byte_size")
    return verified


def verify_archived_copy_dir(result_dir: str | Path) -> dict[str, Any]:
    """Verify copied evidence bytes without asserting current-target lifecycle PASS."""

    verified = _verify_result_dir(result_dir, allow_staged=False, require_current_target=False)
    return {
        **verified,
        "status": "ARCHIVED_COPY_VERIFIED",
        "current_directory_lifecycle_pass": False,
    }


def run_smoke(config: SmokeConfig, backend: SmokeBackend) -> dict[str, Any]:
    """Run the pinned two-pass contract and atomically retain its evidence."""

    started = time.monotonic()
    _validate_config(config)
    for label, digest, length in (
        ("model_revision", config.model_revision, 40),
        ("sae_revision", config.sae_revision, 40),
        ("sae_sha256", config.sae_sha256, 64),
    ):
        if len(digest) != length or any(character not in "0123456789abcdef" for character in digest):
            raise SmokeFailure(f"{label} must be a lowercase {length}-character hexadecimal digest")

    final_dir = config.output_dir
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(final_dir):
        raise SmokeFailure("final output target must be absent; refusing stale or replacement PASS")

    preflight = backend.preflight(config)
    _check_deadline(started, config, "preflight")
    load_contract = backend.load(config)
    required_load_contract = {
        "sae_d_model": config.hidden_size,
        "sae_d_sae": config.sae_width,
        "sae_top_k": config.top_k,
        "sae_file_sha256_verified": True,
        "sae_four_tensor_contract_verified": True,
        "hook_layer_index": config.layer_index,
    }
    for key, expected in required_load_contract.items():
        if load_contract.get(key) != expected:
            raise SmokeFailure(
                f"load contract {key}={load_contract.get(key)!r} does not match {expected!r}"
            )
    _check_deadline(started, config, "load")

    observations = [backend.run_pass(FIXED_PROMPT, config) for _ in range(2)]
    _check_deadline(started, config, "two deterministic passes")
    first, second = observations

    expected_last_dim = config.hidden_size
    for index, observation in enumerate(observations, start=1):
        if len(observation.residual_shape) != 3:
            raise SmokeFailure(f"pass {index}: residual must be rank 3")
        if observation.residual_shape[-1] != expected_last_dim:
            raise SmokeFailure(
                f"pass {index}: residual hidden size {observation.residual_shape[-1]} "
                f"!= {expected_last_dim}"
            )
        if len(observation.selected_indices) != config.top_k:
            raise SmokeFailure(f"pass {index}: selected feature count is not top_k")
        if len(set(observation.selected_indices)) != config.top_k:
            raise SmokeFailure(f"pass {index}: selected feature indices are not unique")
        if len(observation.selected_values) != config.top_k:
            raise SmokeFailure(f"pass {index}: feature value count is not top_k")
        if not all(math.isfinite(value) for value in observation.selected_values):
            raise SmokeFailure(f"pass {index}: selected feature values contain a non-finite value")
        if min(observation.selected_indices) < 0 or max(observation.selected_indices) >= config.sae_width:
            raise SmokeFailure(f"pass {index}: feature index outside SAE width")

    if first.residual_digest != second.residual_digest:
        raise SmokeFailure("raw residual digest changed across fixed-prompt passes")
    if first.selected_indices != second.selected_indices:
        raise SmokeFailure("selected top-k feature indices changed across fixed-prompt passes")
    feature_delta = _max_abs_delta(first.selected_values, second.selected_values)
    if feature_delta > config.feature_atol:
        raise SmokeFailure(
            f"top-k feature repeatability delta {feature_delta:.9g} exceeds {config.feature_atol:.9g}"
        )

    measurement = backend.measurements()
    peak_rss_gib = float(measurement["peak_rss_gib"])
    if peak_rss_gib > config.max_rss_gib:
        raise SmokeFailure(f"peak RSS {peak_rss_gib:.3f} GiB exceeds {config.max_rss_gib:.3f} GiB")

    _check_deadline(started, config, "measurements")
    pre_persistence_elapsed = time.monotonic() - started
    lifecycle_id = secrets.token_hex(16)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "lifecycle_id": lifecycle_id,
        "claim_boundary": {
            "proves": [
                "pinned real model load",
                "exact residual layer hook",
                "official SAE checkpoint contract",
                "top-k sparse encoding",
                "fixed-input repeatability within declared tolerance",
                "artifact and resource measurement path",
            ],
            "does_not_prove": [
                "chelation utility",
                "training gain",
                "semantic improvement",
                "production readiness",
                "novelty",
            ],
        },
        "inputs": {
            "model_repo": config.model_repo,
            "model_revision": config.model_revision,
            "sae_repo": config.sae_repo,
            "sae_revision": config.sae_revision,
            "sae_filename": config.sae_filename,
            "sae_sha256": config.sae_sha256,
            "layer_index": config.layer_index,
            "hidden_size": config.hidden_size,
            "sae_width": config.sae_width,
            "top_k": config.top_k,
            "prompt_sha256": _sha256_bytes(FIXED_PROMPT.encode("utf-8")),
            "prompt_retained": False,
            "offline": config.offline,
        },
        "tolerances": {
            "residual_repeatability": "exact_float32_tensor_sha256",
            "feature_atol": config.feature_atol,
        },
        "preflight": preflight,
        "load_contract": load_contract,
        "observation": {
            "residual_shape": list(first.residual_shape),
            "residual_sha256": first.residual_digest,
            "selected_feature_count": len(first.selected_indices),
            "selected_feature_ids": list(first.selected_indices),
            "repeat_feature_max_abs_delta": feature_delta,
        },
        "resources": {
            **measurement,
            "pre_persistence_wall_seconds": pre_persistence_elapsed,
            "max_wall_seconds": config.max_wall_seconds,
            "max_rss_gib": config.max_rss_gib,
            "min_free_disk_gib": config.min_free_disk_gib,
            "min_free_gpu_gib": config.min_free_gpu_gib,
        },
    }
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.stage-", dir=final_dir.parent))
    stage_owned = True
    try:
        artifact_path = stage_dir / ARTIFACT_FILENAME
        _atomic_json(artifact_path, artifact)
        artifact_digest = _sha256_file(artifact_path)
        provisional_manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "lifecycle_id": lifecycle_id,
            "artifacts": [
                {
                    "path": artifact_path.name,
                    "byte_size": artifact_path.stat().st_size,
                    "sha256": artifact_digest,
                }
            ],
            "source_contract": {
                "model_revision": config.model_revision,
                "sae_revision": config.sae_revision,
                "sae_file_sha256": config.sae_sha256,
            },
            "commit_timing": {
                "checkpoint": "after_initial_stage_verify_fsync_before_final_manifest_rebuild",
                "precommit_checkpoint_wall_seconds": pre_persistence_elapsed,
                "live_deadline_rechecked_before_promotion": True,
            },
        }
        _atomic_json(stage_dir / MANIFEST_FILENAME, provisional_manifest)
        _verify_result_dir(stage_dir, allow_staged=True, require_current_target=False)
        _fsync_directory(stage_dir)
        _check_deadline(started, config, "initial staged write, verification, and fsync")
        precommit_checkpoint_elapsed = time.monotonic() - started
        final_manifest = {
            **provisional_manifest,
            "commit_timing": {
                **provisional_manifest["commit_timing"],
                "precommit_checkpoint_wall_seconds": precommit_checkpoint_elapsed,
            },
        }
        _atomic_json(stage_dir / MANIFEST_FILENAME, final_manifest)
        _verify_result_dir(stage_dir, allow_staged=True, require_current_target=False)
        _fsync_directory(stage_dir)
        _check_deadline(started, config, "final staged manifest rebuild, verification, and fsync")
        _fsync_directory(final_dir.parent)
        _check_deadline(started, config, "parent fsync immediately before promotion")
        _promote_directory_noreplace(stage_dir, final_dir)
        stage_owned = False
    finally:
        if stage_owned and stage_dir.exists():
            shutil.rmtree(stage_dir)
    _fsync_directory(final_dir.parent)
    final_artifact_path = final_dir / ARTIFACT_FILENAME
    final_manifest_path = final_dir / MANIFEST_FILENAME
    target_identity = {
        "directory_name": final_dir.name,
        "resolved_path_sha256": _sha256_bytes(str(final_dir.resolve()).encode("utf-8")),
    }
    committed_bindings = {
        "artifact_sha256": _sha256_file(final_artifact_path),
        "artifact_byte_size": final_artifact_path.stat().st_size,
        "manifest_sha256": _sha256_file(final_manifest_path),
        "manifest_byte_size": final_manifest_path.stat().st_size,
    }
    pre_receipt_checkpoint_elapsed = time.monotonic() - started
    commit_receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMMITTED",
        "lifecycle_id": lifecycle_id,
        "lifecycle": "created_after_no_replace_rename_and_parent_fsync",
        "target": target_identity,
        "bindings": committed_bindings,
        "deadline": {
            "pre_receipt_checkpoint_elapsed_seconds": pre_receipt_checkpoint_elapsed,
            "max_wall_seconds": config.max_wall_seconds,
            "within_cooperative_deadline_at_pre_receipt_checkpoint": True,
        },
    }
    prepared_receipt = _prepare_commit_receipt(final_dir, commit_receipt)
    try:
        _check_deadline(started, config, "prepared commit receipt write and fsync")
        _publish_commit_receipt(prepared_receipt, final_dir / COMMIT_RECEIPT_FILENAME)
    except BaseException:
        try:
            prepared_receipt.unlink()
        except FileNotFoundError:
            pass
        raise
    return artifact


class TorchQwenScopeBackend:
    """Production backend; construction itself does not download or load anything."""

    def __init__(self) -> None:
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._sae: Any = None
        self._started = time.monotonic()
        self._sae_path: Path | None = None

    def preflight(self, config: SmokeConfig) -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            raise SmokeFailure("CUDA is required for the official Spark smoke")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gpu_gib = free_bytes / (1024**3)
        if free_gpu_gib < config.min_free_gpu_gib:
            raise SmokeFailure(
                f"free GPU memory {free_gpu_gib:.3f} GiB below required {config.min_free_gpu_gib:.3f} GiB"
            )
        free_disk_gib = shutil.disk_usage(config.output_dir.parent).free / (1024**3)
        if free_disk_gib < config.min_free_disk_gib:
            raise SmokeFailure(
                f"free disk {free_disk_gib:.3f} GiB below required {config.min_free_disk_gib:.3f} GiB"
            )
        self._torch = torch
        return {
            "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "free_gpu_gib": free_gpu_gib,
            "total_gpu_gib": total_bytes / (1024**3),
            "free_disk_gib": free_disk_gib,
        }

    def load(self, config: SmokeConfig) -> dict[str, Any]:
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

        from qwen_scope_adapter import QwenScopeLayerSAE

        if _numeric_version_prefix(__version__) < MIN_TRANSFORMERS_VERSION:
            required = ".".join(str(value) for value in MIN_TRANSFORMERS_VERSION)
            raise SmokeFailure(f"transformers {__version__} is below required {required}")

        local_only = bool(config.offline)
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_repo, revision=config.model_revision, local_files_only=local_only
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_repo,
            revision=config.model_revision,
            local_files_only=local_only,
            torch_dtype=self._torch.bfloat16,
        ).to("cuda")
        self._model.eval()
        layers = self._model.model.layers
        if config.layer_index >= len(layers):
            raise SmokeFailure(f"model exposes only {len(layers)} layers")
        self._sae_path = Path(
            hf_hub_download(
                repo_id=config.sae_repo,
                filename=config.sae_filename,
                revision=config.sae_revision,
                local_files_only=local_only,
            )
        )
        actual_digest = _sha256_file(self._sae_path)
        if actual_digest != config.sae_sha256:
            raise SmokeFailure(
                f"SAE sha256 mismatch: expected {config.sae_sha256}, got {actual_digest}"
            )
        state = self._torch.load(self._sae_path, map_location="cpu", weights_only=True)
        _verify_official_sae_state(state, config)
        self._sae = QwenScopeLayerSAE.from_state_dict(
            state, layer_index=config.layer_index, top_k=config.top_k
        )
        del state
        if self._sae.d_model != config.hidden_size or self._sae.d_sae != config.sae_width:
            raise SmokeFailure(
                f"SAE shape mismatch: d_model={self._sae.d_model}, d_sae={self._sae.d_sae}"
            )
        return {
            "model_class": type(self._model).__name__,
            "model_layer_count": len(layers),
            "sae_d_model": self._sae.d_model,
            "sae_d_sae": self._sae.d_sae,
            "sae_top_k": self._sae.top_k,
            "sae_file_sha256_verified": True,
            "sae_four_tensor_contract_verified": True,
            "hook_layer_index": config.layer_index,
            "transformers_version": __version__,
        }

    def run_pass(self, prompt: str, config: SmokeConfig) -> PassObservation:
        captured: dict[str, Any] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured["residual"] = hidden.detach().to(device="cpu", dtype=self._torch.float32)

        handle = self._model.model.layers[config.layer_index].register_forward_hook(hook)
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with self._torch.inference_mode():
                self._model(**inputs)
            self._torch.cuda.synchronize()
        finally:
            handle.remove()
        if "residual" not in captured:
            raise SmokeFailure("declared residual hook did not fire")
        residual = captured["residual"]
        if tuple(residual.shape)[-1] != config.hidden_size:
            raise SmokeFailure(f"captured residual shape is {tuple(residual.shape)}")
        last = residual[0, -1]
        pre_acts = last @ self._sae.w_enc.t() + self._sae.b_enc
        values, indices = pre_acts.topk(config.top_k, dim=-1)
        residual_digest = _sha256_bytes(residual.contiguous().numpy().tobytes())
        return PassObservation(
            residual_shape=tuple(int(v) for v in residual.shape),
            residual_digest=residual_digest,
            selected_indices=tuple(int(v) for v in indices.tolist()),
            selected_values=tuple(float(v) for v in values.tolist()),
        )

    def measurements(self) -> dict[str, Any]:
        import resource

        peak_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_gib = float(peak_raw) / (1024**2)
        return {
            "peak_rss_gib": peak_rss_gib,
            "peak_cuda_allocated_gib": self._torch.cuda.max_memory_allocated() / (1024**3),
            "peak_cuda_reserved_gib": self._torch.cuda.max_memory_reserved() / (1024**3),
            "backend_elapsed_seconds": time.monotonic() - self._started,
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-real-model", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Require all pinned files in local HF cache")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output-dir", type=Path)
    target.add_argument("--verify-result-dir", type=Path)
    target.add_argument("--verify-archived-copy-dir", type=Path)
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--max-rss-gib", type=float, default=32.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=12.0)
    parser.add_argument("--min-free-gpu-gib", type=float, default=12.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_result_dir is not None:
        try:
            verified = verify_result_dir(args.verify_result_dir)
        except (SmokeFailure, OSError, ValueError) as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(verified, sort_keys=True))
        return 0
    if args.verify_archived_copy_dir is not None:
        try:
            verified = verify_archived_copy_dir(args.verify_archived_copy_dir)
        except (SmokeFailure, OSError, ValueError) as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(verified, sort_keys=True))
        return 0
    if not args.allow_real_model or os.environ.get(OPT_IN_ENV) != "1":
        print(
            f"REFUSED: real-model execution requires --allow-real-model and {OPT_IN_ENV}=1",
            file=sys.stderr,
        )
        return 2
    config = SmokeConfig(
        output_dir=args.output_dir,
        offline=args.offline,
        max_wall_seconds=args.max_wall_seconds,
        max_rss_gib=args.max_rss_gib,
        min_free_disk_gib=args.min_free_disk_gib,
        min_free_gpu_gib=args.min_free_gpu_gib,
    )
    try:
        artifact = run_smoke(config, TorchQwenScopeBackend())
    except (SmokeFailure, OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    try:
        print(json.dumps({"status": artifact["status"], "output_dir": str(config.output_dir)}))
    except OSError:
        # COMMIT.json is the authoritative terminal result; stdout is not a
        # lifecycle operation and cannot roll an already committed PASS back.
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
