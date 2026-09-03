"""Evidence-bound, resumable safetensors checkpoints for EGV Training."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from ..canonical import canonical_bytes, canonical_value, digest_for, validate_sha256
from .artifacts import ContentAddressedArtifactStore
from .errors import TrainingArtifactError, TrainingCheckpointError


TRAINING_CHECKPOINT_SCHEMA = "egv-training-checkpoint-v2"
CHECKPOINT_ARTIFACT_KIND = "EGV_TRAINING_CHECKPOINT"
STATE_NAME = "state.json"
ADAPTER_TENSORS_NAME = "adapter_model.safetensors"
OPTIMIZER_TENSORS_NAME = "optimizer.safetensors"
RNG_TENSORS_NAME = "rng.safetensors"
_STATE_FIELDS = frozenset({
    "schema_version", "campaign_id", "run_id", "global_step",
    "frozen_dataset_digest", "target_manifest_digest", "tokenizer_manifest_digest",
    "software_manifest_digest", "base_model_manifest_hash", "base_snapshot_digest",
    "ledger_cutoff_hash", "protocol_hash", "parent_artifact_digest", "data_cursor",
    "scheduler_state", "rng_state", "optimizer_metadata",
})
_CURSOR_FIELDS = frozenset({"schema_version", "epoch", "row_index", "consumed_examples", "sampler_seed", "dataset_digest"})
_SCHEDULER_FIELDS = frozenset({"schema_version", "scheduler_class", "last_epoch", "step_count", "base_lrs", "last_lrs"})
_RNG_FIELDS = frozenset({"schema_version", "python_state", "numpy_state", "torch_cpu_tensor_name", "torch_cuda_tensor_names", "cuda_device_count"})
_PYTHON_RNG_FIELDS = frozenset({"version", "state", "gauss"})
_NUMPY_RNG_FIELDS = frozenset({"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"})
_OPTIMIZER_FIELDS = frozenset({"schema_version", "optimizer_class", "parameter_names", "parameter_groups", "tensor_parameter_names"})
_OPTIMIZER_GROUP_FIELDS = frozenset({"parameter_names", "lr", "betas", "eps", "weight_decay"})


def _digest(value: Any, field: str) -> str:
    try:
        return validate_sha256(value, field)
    except Exception as exc:
        raise TrainingCheckpointError(str(exc)) from exc


def _closed(value: Any, fields: frozenset, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TrainingCheckpointError(f"{label} is not a closed schema")
    try:
        return dict(canonical_value(value))
    except Exception as exc:
        raise TrainingCheckpointError(f"{label} is not canonical JSON data") from exc


def _nonnegative(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TrainingCheckpointError(f"{field} must be a nonnegative integer")
    return value


def _float_list(value: Any, field: str) -> list:
    if not isinstance(value, list) or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        raise TrainingCheckpointError(f"{field} must be a numeric list")
    return value


def _validate_cursor(value: Any, dataset_digest: str) -> Dict[str, Any]:
    result = _closed(value, _CURSOR_FIELDS, "data_cursor")
    if result["schema_version"] != "egv-training-cursor-v1":
        raise TrainingCheckpointError("unsupported data cursor schema")
    for field in ("epoch", "row_index", "consumed_examples", "sampler_seed"):
        _nonnegative(result[field], f"data_cursor.{field}")
    if _digest(result["dataset_digest"], "data_cursor.dataset_digest") != dataset_digest:
        raise TrainingCheckpointError("data cursor is bound to another frozen dataset")
    return result


def _validate_scheduler(value: Any) -> Dict[str, Any]:
    result = _closed(value, _SCHEDULER_FIELDS, "scheduler_state")
    if result["schema_version"] != "egv-training-scheduler-v1" or not isinstance(result["scheduler_class"], str) or not result["scheduler_class"]:
        raise TrainingCheckpointError("scheduler state identity is invalid")
    _nonnegative(result["last_epoch"], "scheduler_state.last_epoch")
    _nonnegative(result["step_count"], "scheduler_state.step_count")
    base = _float_list(result["base_lrs"], "scheduler_state.base_lrs")
    last = _float_list(result["last_lrs"], "scheduler_state.last_lrs")
    if not base or len(base) != len(last):
        raise TrainingCheckpointError("scheduler LR topology is inconsistent")
    return result


def _validate_rng(value: Any) -> Dict[str, Any]:
    result = _closed(value, _RNG_FIELDS, "rng_state")
    if result["schema_version"] != "egv-training-rng-v1":
        raise TrainingCheckpointError("unsupported RNG state schema")
    python_state = _closed(result["python_state"], _PYTHON_RNG_FIELDS, "python RNG state")
    numpy_state = _closed(result["numpy_state"], _NUMPY_RNG_FIELDS, "NumPy RNG state")
    if python_state["version"] != 3 or not isinstance(python_state["state"], list) or any(not isinstance(item, int) for item in python_state["state"]):
        raise TrainingCheckpointError("Python RNG state is malformed")
    if python_state["gauss"] is not None and not isinstance(python_state["gauss"], (int, float)):
        raise TrainingCheckpointError("Python RNG Gaussian cache is malformed")
    if not isinstance(numpy_state["bit_generator"], str) or not numpy_state["bit_generator"]:
        raise TrainingCheckpointError("NumPy RNG bit generator is missing")
    if not isinstance(numpy_state["keys"], list) or any(not isinstance(item, int) for item in numpy_state["keys"]):
        raise TrainingCheckpointError("NumPy RNG keys are malformed")
    for field in ("position", "has_gauss"):
        _nonnegative(numpy_state[field], f"numpy_state.{field}")
    if not isinstance(numpy_state["cached_gaussian"], (int, float)):
        raise TrainingCheckpointError("NumPy RNG Gaussian cache is malformed")
    count = _nonnegative(result["cuda_device_count"], "rng_state.cuda_device_count")
    if result["torch_cpu_tensor_name"] != "torch_cpu_rng":
        raise TrainingCheckpointError("CPU RNG tensor has the wrong frozen name")
    if result["torch_cuda_tensor_names"] != [f"torch_cuda_rng.{index}" for index in range(count)]:
        raise TrainingCheckpointError("RNG state does not enumerate every CUDA device")
    return result


def _validate_optimizer(value: Any) -> Dict[str, Any]:
    result = _closed(value, _OPTIMIZER_FIELDS, "optimizer_metadata")
    if result["schema_version"] != "egv-training-optimizer-v1" or not isinstance(result["optimizer_class"], str) or not result["optimizer_class"]:
        raise TrainingCheckpointError("optimizer identity is invalid")
    names = result["parameter_names"]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)) or names != sorted(names):
        raise TrainingCheckpointError("optimizer parameter-name topology is invalid")
    if not isinstance(result["parameter_groups"], list) or not result["parameter_groups"]:
        raise TrainingCheckpointError("optimizer parameter groups are missing")
    grouped = []
    for raw in result["parameter_groups"]:
        group = _closed(raw, _OPTIMIZER_GROUP_FIELDS, "optimizer parameter group")
        if not isinstance(group["parameter_names"], list) or not group["parameter_names"]:
            raise TrainingCheckpointError("optimizer group has no parameter names")
        grouped.extend(group["parameter_names"])
        if len(_float_list(group["betas"], "optimizer group betas")) != 2:
            raise TrainingCheckpointError("optimizer betas must contain two values")
        for field in ("lr", "eps", "weight_decay"):
            if not isinstance(group[field], (int, float)) or isinstance(group[field], bool) or group[field] < 0:
                raise TrainingCheckpointError(f"optimizer group {field} is invalid")
    if sorted(grouped) != names or len(grouped) != len(set(grouped)):
        raise TrainingCheckpointError("optimizer groups do not exactly cover named parameters")
    topology = result["tensor_parameter_names"]
    if not isinstance(topology, Mapping) or not topology or any(not isinstance(key, str) or not key or parameter not in names for key, parameter in topology.items()):
        raise TrainingCheckpointError("optimizer tensor topology is invalid")
    return result


@dataclass(frozen=True)
class TrainingCheckpoint:
    campaign_id: str
    run_id: str
    global_step: int
    frozen_dataset_digest: str
    target_manifest_digest: str
    tokenizer_manifest_digest: str
    software_manifest_digest: str
    base_model_manifest_hash: str
    base_snapshot_digest: str
    ledger_cutoff_hash: str
    protocol_hash: str
    data_cursor: Mapping[str, Any]
    scheduler_state: Mapping[str, Any]
    rng_state: Mapping[str, Any]
    optimizer_metadata: Mapping[str, Any]
    parent_artifact_digest: Optional[str] = None
    schema_version: str = TRAINING_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_CHECKPOINT_SCHEMA:
            raise TrainingCheckpointError("unsupported EGV Training checkpoint schema")
        if not isinstance(self.campaign_id, str) or not self.campaign_id or not isinstance(self.run_id, str) or not self.run_id:
            raise TrainingCheckpointError("checkpoint identity fields must be non-empty")
        _nonnegative(self.global_step, "global_step")
        for field in (
            "frozen_dataset_digest", "target_manifest_digest", "tokenizer_manifest_digest",
            "software_manifest_digest", "base_model_manifest_hash", "base_snapshot_digest",
            "ledger_cutoff_hash", "protocol_hash",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if self.parent_artifact_digest is not None:
            object.__setattr__(self, "parent_artifact_digest", _digest(self.parent_artifact_digest, "parent_artifact_digest"))
        object.__setattr__(self, "data_cursor", _validate_cursor(self.data_cursor, self.frozen_dataset_digest))
        object.__setattr__(self, "scheduler_state", _validate_scheduler(self.scheduler_state))
        object.__setattr__(self, "rng_state", _validate_rng(self.rng_state))
        object.__setattr__(self, "optimizer_metadata", _validate_optimizer(self.optimizer_metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in _STATE_FIELDS}

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingCheckpoint":
        if not isinstance(value, Mapping) or set(value) != _STATE_FIELDS:
            raise TrainingCheckpointError("Training checkpoint state is not a closed schema")
        return cls(**{field: value[field] for field in _STATE_FIELDS})

    @property
    def provenance(self) -> Tuple[str, ...]:
        return tuple(getattr(self, field) for field in (
            "campaign_id", "run_id", "frozen_dataset_digest", "target_manifest_digest",
            "tokenizer_manifest_digest", "software_manifest_digest", "base_model_manifest_hash",
            "base_snapshot_digest", "ledger_cutoff_hash", "protocol_hash",
        ))


@dataclass(frozen=True)
class LoadedTrainingCheckpoint:
    checkpoint: TrainingCheckpoint
    artifact_digest: str
    artifact_path: Path
    adapter_weights: Mapping[str, Any]
    optimizer_tensors: Mapping[str, Any]
    rng_tensors: Mapping[str, Any]
    lineage: Tuple[str, ...] = ()


def _safetensors_runtime() -> Any:
    try:
        return importlib.import_module("safetensors.torch")
    except ImportError as exc:
        raise TrainingCheckpointError("safetensors.torch is required for Training checkpoints") from exc


def _normalize_tensors(tensors: Any, field: str) -> Dict[str, Any]:
    if not isinstance(tensors, Mapping) or not tensors:
        raise TrainingCheckpointError(f"{field} must contain at least one tensor")
    torch = importlib.import_module("torch")
    normalized = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TrainingCheckpointError(f"{field} contains an invalid tensor entry")
        normalized[name] = tensor.detach().cpu().contiguous().clone()
    return dict(sorted(normalized.items()))


def _encode_tensors(tensors: Mapping[str, Any], role: str) -> bytes:
    try:
        return _safetensors_runtime().save(dict(tensors))
    except Exception as exc:
        raise TrainingCheckpointError(f"{role} tensors cannot be encoded as safetensors") from exc


def _decode_tensors(payload: bytes, role: str) -> Mapping[str, Any]:
    try:
        return _normalize_tensors(_safetensors_runtime().load(payload), role)
    except Exception as exc:
        if isinstance(exc, TrainingCheckpointError):
            raise
        raise TrainingCheckpointError(f"{role} safetensors payload is invalid") from exc


def _validate_rng_tensors(tensors: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    torch = importlib.import_module("torch")
    expected = {state["torch_cpu_tensor_name"], *state["torch_cuda_tensor_names"]}
    if set(tensors) != expected:
        raise TrainingCheckpointError("safetensors RNG inventory does not match CPU/all-CUDA declaration")
    if any(tensor.dtype != torch.uint8 or tensor.ndim != 1 for tensor in tensors.values()):
        raise TrainingCheckpointError("PyTorch RNG states must be one-dimensional uint8 tensors")


class TrainingCheckpointStore:
    def __init__(self, root: Path) -> None:
        self.artifacts = ContentAddressedArtifactStore(root)

    def save(self, checkpoint: TrainingCheckpoint, *, adapter_weights: Mapping[str, Any], optimizer_tensors: Mapping[str, Any], rng_tensors: Mapping[str, Any]) -> tuple:
        if type(checkpoint) is not TrainingCheckpoint:
            raise TrainingCheckpointError("checkpoint must use the exact frozen TrainingCheckpoint type")
        adapter = _normalize_tensors(adapter_weights, "adapter_weights")
        optimizer = _normalize_tensors(optimizer_tensors, "optimizer_tensors")
        rng = _normalize_tensors(rng_tensors, "rng_tensors")
        if set(optimizer) != set(checkpoint.optimizer_metadata["tensor_parameter_names"]):
            raise TrainingCheckpointError("optimizer safetensors do not match parameter-name topology")
        _validate_rng_tensors(rng, checkpoint.rng_state)
        if checkpoint.parent_artifact_digest is not None:
            parent = self.resume(expected_artifact_digest=checkpoint.parent_artifact_digest)
            if parent.checkpoint.provenance != checkpoint.provenance or parent.checkpoint.global_step >= checkpoint.global_step:
                raise TrainingCheckpointError("checkpoint parent lineage or step is invalid")
        files = {
            STATE_NAME: canonical_bytes(checkpoint.to_dict()),
            ADAPTER_TENSORS_NAME: _encode_tensors(adapter, "adapter"),
            OPTIMIZER_TENSORS_NAME: _encode_tensors(optimizer, "optimizer"),
            RNG_TENSORS_NAME: _encode_tensors(rng, "RNG"),
        }
        try:
            path, manifest = self.artifacts.publish(artifact_kind=CHECKPOINT_ARTIFACT_KIND, metadata_digest=checkpoint.digest, files=files)
        except TrainingArtifactError as exc:
            raise TrainingCheckpointError(str(exc)) from exc
        return path, manifest.digest

    def load(self, artifact: Union[Path, str], *, expected_artifact_digest: Optional[str] = None) -> LoadedTrainingCheckpoint:
        path = self.artifacts._artifact_path(artifact) if isinstance(artifact, str) else Path(artifact)
        try:
            manifest = self.artifacts.verify(path, expected_digest=expected_artifact_digest, expected_kind=CHECKPOINT_ARTIFACT_KIND)
        except TrainingArtifactError as exc:
            raise TrainingCheckpointError(str(exc)) from exc
        if set(manifest.files) != {STATE_NAME, ADAPTER_TENSORS_NAME, OPTIMIZER_TENSORS_NAME, RNG_TENSORS_NAME}:
            raise TrainingCheckpointError("checkpoint artifact file set differs from frozen contract")
        try:
            raw = (path / STATE_NAME).read_bytes()
            checkpoint = TrainingCheckpoint.from_mapping(json.loads(raw.decode("utf-8")))
        except (OSError, UnicodeError, ValueError) as exc:
            raise TrainingCheckpointError("checkpoint state cannot be decoded") from exc
        if raw != canonical_bytes(checkpoint.to_dict()) or checkpoint.digest != manifest.metadata_digest:
            raise TrainingCheckpointError("checkpoint state is not canonically bound to manifest")
        adapter = _decode_tensors((path / ADAPTER_TENSORS_NAME).read_bytes(), "adapter")
        optimizer = _decode_tensors((path / OPTIMIZER_TENSORS_NAME).read_bytes(), "optimizer")
        rng = _decode_tensors((path / RNG_TENSORS_NAME).read_bytes(), "RNG")
        if set(optimizer) != set(checkpoint.optimizer_metadata["tensor_parameter_names"]):
            raise TrainingCheckpointError("optimizer tensor topology changed")
        _validate_rng_tensors(rng, checkpoint.rng_state)
        return LoadedTrainingCheckpoint(checkpoint, manifest.digest, path, adapter, optimizer, rng)

    def resume(self, *, expected_artifact_digest: str) -> LoadedTrainingCheckpoint:
        head = self.load(expected_artifact_digest, expected_artifact_digest=expected_artifact_digest)
        lineage = [head.artifact_digest]
        child = head
        seen = set(lineage)
        while child.checkpoint.parent_artifact_digest is not None:
            parent_digest = child.checkpoint.parent_artifact_digest
            if parent_digest in seen:
                raise TrainingCheckpointError("checkpoint lineage contains a cycle")
            parent = self.load(parent_digest, expected_artifact_digest=parent_digest)
            if parent.checkpoint.provenance != head.checkpoint.provenance:
                raise TrainingCheckpointError("checkpoint lineage provenance changed")
            if parent.checkpoint.global_step >= child.checkpoint.global_step:
                raise TrainingCheckpointError("checkpoint lineage steps are not strictly increasing")
            seen.add(parent_digest)
            lineage.append(parent_digest)
            child = parent
        return LoadedTrainingCheckpoint(
            head.checkpoint, head.artifact_digest, head.artifact_path,
            head.adapter_weights, head.optimizer_tensors, head.rng_tensors, tuple(lineage),
        )

    def latest(self, *, expected_artifact_digest: Optional[str] = None) -> LoadedTrainingCheckpoint:
        if expected_artifact_digest is None:
            raise TrainingCheckpointError("production latest/resume requires an authoritative expected digest")
        return self.resume(expected_artifact_digest=expected_artifact_digest)

    def scan_latest_diagnostic(self, *, campaign_id: Optional[str] = None, run_id: Optional[str] = None) -> Optional[LoadedTrainingCheckpoint]:
        candidates = []
        root = self.artifacts.root / "sha256"
        if not root.exists():
            return None
        for shard in root.iterdir():
            if shard.is_symlink() or not shard.is_dir():
                raise TrainingCheckpointError("checkpoint store contains an unsafe shard")
            for path in shard.iterdir():
                if path.name.startswith(".publishing-"):
                    continue
                loaded = self.load(path)
                if (campaign_id is None or loaded.checkpoint.campaign_id == campaign_id) and (run_id is None or loaded.checkpoint.run_id == run_id):
                    candidates.append(loaded)
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.checkpoint.global_step, item.artifact_digest))


__all__ = [
    "ADAPTER_TENSORS_NAME", "CHECKPOINT_ARTIFACT_KIND", "LoadedTrainingCheckpoint",
    "OPTIMIZER_TENSORS_NAME", "RNG_TENSORS_NAME", "STATE_NAME", "TRAINING_CHECKPOINT_SCHEMA",
    "TrainingCheckpoint", "TrainingCheckpointStore",
]
