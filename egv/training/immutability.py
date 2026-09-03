"""Base-weight immutability and LoRA-only mutation proofs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from ..canonical import digest_for
from .targets import LoraTargetManifest


IMMUTABILITY_SCHEMA = "egv-base-model-immutability-v1"
_ADAPTER_MARKERS = (".lora_A.", ".lora_B.")
_PEFT_TARGET_PREFIXES = ("", "base_model.model.")


class TrainingImmutabilityError(RuntimeError):
    """A base tensor, model file, or trainability boundary changed."""


def _tensor_digest(tensor: Any) -> str:
    try:
        value = tensor.detach().cpu().contiguous()
        raw = value.view(dtype=__import__("torch").uint8).numpy().tobytes()
        record = {"shape": tuple(int(item) for item in value.shape), "dtype": str(value.dtype), "bytes": hashlib.sha256(raw).hexdigest()}
    except Exception as exc:
        raise TrainingImmutabilityError("model tensor cannot be hashed deterministically") from exc
    return digest_for(record)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_model_files(root: Path, expected: Mapping[str, str]) -> Dict[str, str]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise TrainingImmutabilityError("base-model root must be an existing non-symlink directory")
    resolved = root.resolve()
    normalized: Dict[str, str] = {}
    for relative, wanted in sorted(expected.items()):
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(wanted, str)
            or len(wanted) != 64
        ):
            raise TrainingImmutabilityError("base-model file manifest contains an unsafe path")
        try:
            int(wanted, 16)
        except ValueError as exc:
            raise TrainingImmutabilityError("base-model file manifest contains a non-hex digest") from exc
        path = (resolved / relative).resolve()
        try:
            path.relative_to(resolved)
        except ValueError as exc:
            raise TrainingImmutabilityError("base-model file escapes its immutable root") from exc
        if path.is_symlink() or not path.is_file():
            raise TrainingImmutabilityError("base-model file is missing or symlinked: {}".format(relative))
        actual = _file_digest(path)
        if actual != wanted:
            raise TrainingImmutabilityError("base-model file digest changed: {}".format(relative))
        normalized[relative] = actual
    actual_names = set()
    for path in resolved.rglob("*"):
        if path.is_symlink():
            raise TrainingImmutabilityError("base-model tree gained a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise TrainingImmutabilityError("base-model tree contains a non-regular entry")
        relative = path.relative_to(resolved).as_posix()
        if relative != "model-manifest.json":
            actual_names.add(relative)
    if actual_names != set(normalized):
        raise TrainingImmutabilityError("base-model file inventory changed after preflight")
    return normalized


@dataclass(frozen=True)
class _TensorRecord:
    name: str
    kind: str
    digest: str
    value: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class BaseModelSnapshot:
    schema_version: str
    model_manifest_digest: str
    target_manifest_digest: str
    file_hashes: Mapping[str, str]
    tensor_digests: Mapping[str, str]
    digest: str
    model_root: Path = field(repr=False, compare=False)
    records: Tuple[_TensorRecord, ...] = field(repr=False, compare=False)


def capture_base_model_snapshot(
    model: Any,
    *,
    model_root: Path,
    model_manifest_digest: str,
    target_manifest: LoraTargetManifest,
    file_hashes: Mapping[str, str],
) -> BaseModelSnapshot:
    """Capture immutable file and live-tensor state immediately before PEFT wrapping."""

    if not isinstance(model_manifest_digest, str) or len(model_manifest_digest) != 64:
        raise TrainingImmutabilityError("model manifest digest is not SHA-256")
    try:
        int(model_manifest_digest, 16)
    except ValueError as exc:
        raise TrainingImmutabilityError("model manifest digest is not hexadecimal") from exc
    if type(target_manifest) is not LoraTargetManifest:
        raise TrainingImmutabilityError("base snapshot requires the sealed LoRA target manifest")
    try:
        target_manifest.verify()
    except Exception as exc:
        raise TrainingImmutabilityError("LoRA target manifest is invalid") from exc
    if target_manifest.model_manifest_digest != model_manifest_digest:
        raise TrainingImmutabilityError("LoRA target manifest is bound to a different base model")
    verified_files = _verify_model_files(model_root, file_hashes)
    records = []
    seen_ids = set()
    for kind, getter in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
        method = getattr(model, getter, None)
        if not callable(method):
            raise TrainingImmutabilityError("base model does not expose {}".format(getter))
        for name, value in method():
            if not isinstance(name, str) or not name or id(value) in seen_ids:
                raise TrainingImmutabilityError("base model tensor inventory is ambiguous")
            seen_ids.add(id(value))
            records.append(_TensorRecord(name, kind, _tensor_digest(value), value))
    if not records:
        raise TrainingImmutabilityError("base model exposes no measurable tensors")
    tensor_digests = {"{}:{}".format(record.kind, record.name): record.digest for record in records}
    public = {
        "schema_version": IMMUTABILITY_SCHEMA,
        "model_manifest_digest": model_manifest_digest,
        "target_manifest_digest": target_manifest.digest,
        "file_hashes": verified_files,
        "tensor_digests": tensor_digests,
    }
    return BaseModelSnapshot(
        IMMUTABILITY_SCHEMA,
        model_manifest_digest,
        target_manifest.digest,
        MappingProxyType(dict(verified_files)),
        MappingProxyType(dict(tensor_digests)),
        digest_for(public),
        Path(model_root).resolve(),
        tuple(records),
    )


def _adapter_target(name: str, target_manifest: LoraTargetManifest) -> Optional[str]:
    for marker in _ADAPTER_MARKERS:
        if marker in name:
            prefix = name.split(marker, 1)[0]
            for target in target_manifest.module_names:
                if prefix in tuple(wrapper + target for wrapper in _PEFT_TARGET_PREFIXES):
                    return target
            return None
    return None


def assert_lora_only_trainable(model: Any, target_manifest: LoraTargetManifest) -> Tuple[str, ...]:
    """Require every and only trainable parameter to be LoRA A/B on a frozen target."""

    try:
        target_manifest.verify()
    except Exception as exc:
        raise TrainingImmutabilityError("LoRA target manifest is invalid") from exc
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TrainingImmutabilityError("trained model does not expose named_parameters")
    trainable = []
    adapter_targets = set()
    for name, parameter in named_parameters():
        target = _adapter_target(name, target_manifest)
        if target is not None:
            adapter_targets.add(target)
            if not bool(getattr(parameter, "requires_grad", False)):
                raise TrainingImmutabilityError("LoRA adapter parameter is unexpectedly frozen: {}".format(name))
            trainable.append(name)
        elif bool(getattr(parameter, "requires_grad", False)):
            raise TrainingImmutabilityError("non-allowlisted parameter is trainable: {}".format(name))
    if not trainable:
        raise TrainingImmutabilityError("no LoRA adapter parameters are trainable")
    if adapter_targets != set(target_manifest.module_names):
        raise TrainingImmutabilityError("trainable LoRA adapter coverage differs from the frozen 114 targets")
    return tuple(sorted(trainable))


def assert_optimizer_lora_only(
    optimizer: Any, model: Any, target_manifest: LoraTargetManifest
) -> Tuple[str, ...]:
    """Ensure an optimizer cannot retain a hidden base-parameter mutation path."""

    trainable = assert_lora_only_trainable(model, target_manifest)
    names_by_id = {id(value): name for name, value in model.named_parameters()}
    expected_ids = {identifier for identifier, name in names_by_id.items() if name in trainable}
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise TrainingImmutabilityError("optimizer exposes no parameter groups")
    actual_ids = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), (list, tuple)):
            raise TrainingImmutabilityError("optimizer parameter group is malformed")
        actual_ids.extend(id(parameter) for parameter in group["params"])
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise TrainingImmutabilityError("optimizer parameters are not exactly the trainable LoRA parameters")
    return tuple(sorted(names_by_id[identifier] for identifier in actual_ids))


def assert_base_model_immutable(
    snapshot: BaseModelSnapshot,
    trained_model: Any,
    target_manifest: LoraTargetManifest,
) -> str:
    """Re-hash original tensor objects and checkpoint files after training."""

    if type(snapshot) is not BaseModelSnapshot or snapshot.schema_version != IMMUTABILITY_SCHEMA:
        raise TrainingImmutabilityError("base-model snapshot is not an EGV immutability proof")
    try:
        target_manifest.verify()
    except Exception as exc:
        raise TrainingImmutabilityError("LoRA target manifest is invalid") from exc
    if (
        target_manifest.digest != snapshot.target_manifest_digest
        or target_manifest.model_manifest_digest != snapshot.model_manifest_digest
    ):
        raise TrainingImmutabilityError("immutability proof is bound to a different target/model manifest")
    record_digests = {"{}:{}".format(record.kind, record.name): record.digest for record in snapshot.records}
    if record_digests != dict(snapshot.tensor_digests):
        raise TrainingImmutabilityError("base-model snapshot tensor inventory was altered")
    snapshot_public = {
        "schema_version": snapshot.schema_version,
        "model_manifest_digest": snapshot.model_manifest_digest,
        "target_manifest_digest": snapshot.target_manifest_digest,
        "file_hashes": dict(snapshot.file_hashes),
        "tensor_digests": dict(snapshot.tensor_digests),
    }
    if digest_for(snapshot_public) != snapshot.digest:
        raise TrainingImmutabilityError("base-model snapshot metadata was altered")
    _verify_model_files(snapshot.model_root, snapshot.file_hashes)
    current_parameters = tuple(trained_model.named_parameters())
    current_buffers = tuple(trained_model.named_buffers())
    current_ids = {id(value) for _, value in current_parameters}
    current_ids.update(id(value) for _, value in current_buffers)
    for record in snapshot.records:
        if id(record.value) not in current_ids:
            raise TrainingImmutabilityError("original base tensor is no longer attached to the trained model: {}".format(record.name))
        if _tensor_digest(record.value) != record.digest:
            raise TrainingImmutabilityError("base-model tensor changed during LoRA training: {}".format(record.name))
    original_ids = {id(record.value) for record in snapshot.records}
    for name, value in current_parameters:
        if id(value) not in original_ids and _adapter_target(name, target_manifest) is None:
            raise TrainingImmutabilityError("trained model gained a non-adapter parameter: {}".format(name))
    for name, value in current_buffers:
        if id(value) not in original_ids:
            raise TrainingImmutabilityError("trained model gained a non-base buffer: {}".format(name))
    return snapshot.digest


def validate_lora_only_training_state(
    snapshot: BaseModelSnapshot,
    trained_model: Any,
    *,
    target_manifest: LoraTargetManifest,
    optimizer: Optional[Any] = None,
) -> Mapping[str, Any]:
    trainable = assert_lora_only_trainable(trained_model, target_manifest)
    if optimizer is not None:
        assert_optimizer_lora_only(optimizer, trained_model, target_manifest)
    proof_digest = assert_base_model_immutable(snapshot, trained_model, target_manifest)
    return {
        "schema_version": IMMUTABILITY_SCHEMA,
        "snapshot_digest": proof_digest,
        "model_manifest_digest": snapshot.model_manifest_digest,
        "target_manifest_digest": target_manifest.digest,
        "base_tensor_count": len(snapshot.records),
        "base_file_count": len(snapshot.file_hashes),
        "trainable_adapter_parameters": trainable,
    }


__all__ = [
    "BaseModelSnapshot",
    "IMMUTABILITY_SCHEMA",
    "TrainingImmutabilityError",
    "assert_base_model_immutable",
    "assert_lora_only_trainable",
    "assert_optimizer_lora_only",
    "capture_base_model_snapshot",
    "validate_lora_only_training_state",
]
