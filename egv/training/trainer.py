"""Bounded LoRA training runtime and deterministic CPU fixture smoke."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..canonical import canonical_json, content_id, digest_bytes, digest_for, validate_sha256
from ..receipts import ReceiptSigner, receipt_hash
from .development import (
    DevelopmentLossEvaluation,
    DevelopmentLossGateway,
    ExternalDevelopmentLossGateway,
    TrainingCheckpoint,
    model_state_digest_for_training,
)
from .protocol import (
    SealedTrainingInputs,
    TokenizedTrainingExample,
    TrainingConfigurationError,
    TrainingDependencyError,
    TrainingError,
    TrainingIntegrityError,
    TrainingProtocol,
    TrainingRow,
    build_training_batch,
    seal_training_inputs,
)


TRAINING_RUNTIME_SCHEMA = "egv-training-runtime-v1"
SEALED_RUNTIME_DATASET_SCHEMA = "egv-sealed-training-runtime-input-v1"
SEALED_DEVELOPMENT_DATASET_SCHEMA = "egv-sealed-development-runtime-input-v1"
LORA_CHANGE_ATTESTATION_SCHEMA = "egv-lora-change-attestation-v1"
LORA_SEALED_BINDING_SCHEMA = "egv-lora-sealed-binding-v1"
SEALED_RUNTIME_DATASET_MAX_BYTES = 128 * 1024 * 1024
LORA_EFFECTIVE_DELTA_ROW_CHUNK = 16
LORA_EFFECTIVE_DELTA_COLUMN_CHUNK = 256


@dataclass(frozen=True)
class _PrivateTrainingTree:
    root: Path
    root_identity: Tuple[int, int]
    ancestor_identities: Tuple[Tuple[Path, Tuple[int, int]], ...]


def _snapshot_private_training_ancestors(path: Path) -> Tuple[Tuple[Path, Tuple[int, int]], ...]:
    snapshot = []
    for ancestor in _training_ancestors(path):
        try:
            value = ancestor.lstat()
        except OSError as exc:
            raise TrainingConfigurationError("private training output ancestor is unavailable") from exc
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode) or _is_reparse(value):
            raise TrainingIntegrityError(
                "private training output must not traverse a symlink or reparse ancestor"
            )
        snapshot.append((ancestor, _training_file_identity(value, "private training output ancestor")))
    return tuple(snapshot)


def _verify_private_training_ancestors(
    snapshot: Sequence[Tuple[Path, Tuple[int, int]]],
) -> None:
    for ancestor, expected_identity in snapshot:
        try:
            value = ancestor.lstat()
        except OSError as exc:
            raise TrainingIntegrityError("private training output ancestor changed") from exc
        if (
            stat.S_ISLNK(value.st_mode)
            or not stat.S_ISDIR(value.st_mode)
            or _is_reparse(value)
            or _training_file_identity(value, "private training output ancestor") != expected_identity
        ):
            raise TrainingIntegrityError("private training output ancestor changed")


def _scan_private_training_tree(tree: _PrivateTrainingTree) -> Tuple[Tuple[Any, ...], ...]:
    """Validate and snapshot one private output tree without following links."""

    _verify_private_training_ancestors(tree.ancestor_identities)
    try:
        root_stat = tree.root.lstat()
    except OSError as exc:
        raise TrainingIntegrityError("private training staging root changed") from exc
    if (
        stat.S_ISLNK(root_stat.st_mode)
        or not stat.S_ISDIR(root_stat.st_mode)
        or _is_reparse(root_stat)
        or _training_file_identity(root_stat, "private training staging root") != tree.root_identity
    ):
        raise TrainingIntegrityError("private training staging root identity changed")
    if os.name != "nt" and stat.S_IMODE(root_stat.st_mode) & 0o077:
        raise TrainingIntegrityError("private training staging root permissions are not owner-only")

    root_device = tree.root_identity[0]
    entries = [(".", "directory", *tree.root_identity, 0)]
    try:
        for directory, directory_names, file_names in os.walk(
            tree.root, topdown=True, followlinks=False
        ):
            directory_names.sort()
            file_names.sort()
            current = Path(directory)
            current_stat = current.lstat()
            if (
                stat.S_ISLNK(current_stat.st_mode)
                or not stat.S_ISDIR(current_stat.st_mode)
                or _is_reparse(current_stat)
                or _training_file_identity(current_stat, "private training directory")[0] != root_device
            ):
                raise TrainingIntegrityError("private training tree contains an unsafe directory")
            for name in directory_names:
                child = current / name
                value = child.lstat()
                identity = _training_file_identity(value, "private training directory")
                if (
                    stat.S_ISLNK(value.st_mode)
                    or not stat.S_ISDIR(value.st_mode)
                    or _is_reparse(value)
                    or identity[0] != root_device
                ):
                    raise TrainingIntegrityError("private training tree contains an unsafe directory")
                entries.append((child.relative_to(tree.root).as_posix(), "directory", *identity, 0))
            for name in file_names:
                child = current / name
                value = child.lstat()
                identity = _training_file_identity(value, "private training file")
                if (
                    stat.S_ISLNK(value.st_mode)
                    or not stat.S_ISREG(value.st_mode)
                    or _is_reparse(value)
                    or getattr(value, "st_nlink", 1) != 1
                    or identity[0] != root_device
                ):
                    raise TrainingIntegrityError("private training tree contains an unsafe file")
                entries.append(
                    (child.relative_to(tree.root).as_posix(), "file", *identity, int(value.st_size))
                )
    except TrainingIntegrityError:
        raise
    except OSError as exc:
        raise TrainingIntegrityError("private training tree changed during validation") from exc

    try:
        final_root = tree.root.lstat()
    except OSError as exc:
        raise TrainingIntegrityError("private training staging root changed") from exc
    if (
        stat.S_ISLNK(final_root.st_mode)
        or not stat.S_ISDIR(final_root.st_mode)
        or _is_reparse(final_root)
        or _training_file_identity(final_root, "private training staging root")
        != tree.root_identity
        or (os.name != "nt" and stat.S_IMODE(final_root.st_mode) & 0o077)
    ):
        raise TrainingIntegrityError("private training staging root identity changed")
    _verify_private_training_ancestors(tree.ancestor_identities)
    return tuple(sorted(set(entries)))


def _verify_private_training_tree(
    tree: _PrivateTrainingTree,
    *,
    expected_entries: Optional[Sequence[Tuple[Any, ...]]] = None,
) -> Tuple[Tuple[Any, ...], ...]:
    entries = _scan_private_training_tree(tree)
    if expected_entries is not None and entries != tuple(expected_entries):
        raise TrainingIntegrityError("private training tree identity changed before publication")
    return entries


def _verify_external_gateway_scratch_binding(
    tree: _PrivateTrainingTree,
    gateway: ExternalDevelopmentLossGateway,
) -> Tuple[Path, Tuple[int, int]]:
    """Require evaluator scratch to be the exact owned child of ``tree``."""

    if type(tree) is not _PrivateTrainingTree:
        raise TrainingConfigurationError("production Training requires a private staging tree")
    if type(gateway) is not ExternalDevelopmentLossGateway:
        raise TrainingDependencyError("production requires the external evaluator client")
    _verify_private_training_tree(tree)
    expected_root = Path(os.path.abspath(str(tree.root / "evaluator-scratch")))
    if gateway._scratch_root != expected_root:
        raise TrainingIntegrityError(
            "external evaluator scratch root must be the exact private staging descendant"
        )
    try:
        value = expected_root.lstat()
    except OSError as exc:
        raise TrainingIntegrityError(
            "external evaluator scratch root is unavailable in the private staging tree"
        ) from exc
    expected_identity = _training_file_identity(value, "external evaluator scratch root")
    if (
        stat.S_ISLNK(value.st_mode)
        or not stat.S_ISDIR(value.st_mode)
        or _is_reparse(value)
        or gateway._scratch_root_identity != expected_identity
    ):
        raise TrainingIntegrityError(
            "external evaluator scratch root identity differs from the private staging tree"
        )
    gateway._verify_scratch_boundary()
    _verify_private_training_tree(tree)
    return expected_root, expected_identity


def _verify_private_adapter_destination(tree: _PrivateTrainingTree, output_root: Path) -> Path:
    """Return the one adapter leaf allowed inside an owned production tree."""

    if type(tree) is not _PrivateTrainingTree:
        raise TrainingConfigurationError("production Training requires a private staging tree")
    _verify_private_training_tree(tree)
    expected = Path(os.path.abspath(str(tree.root / "adapter")))
    destination = Path(os.path.abspath(str(output_root)))
    if destination != expected:
        raise TrainingIntegrityError(
            "adapter output must be the exact private staging adapter directory"
        )
    return destination


def _create_private_training_tree(output_root: Path) -> _PrivateTrainingTree:
    """Create a random, exclusive sibling staging directory for one run."""

    output_root = Path(os.path.abspath(str(output_root)))
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    _snapshot_private_training_ancestors(parent / ".egv-training-parent-check")
    try:
        staging = Path(tempfile.mkdtemp(prefix=".egv-training-private-", dir=str(parent)))
        if os.name != "nt":
            os.chmod(staging, 0o700)
        value = staging.lstat()
        tree = _PrivateTrainingTree(
            root=staging,
            root_identity=_training_file_identity(value, "private training staging root"),
            ancestor_identities=_snapshot_private_training_ancestors(staging),
        )
        _verify_private_training_tree(tree)
        return tree
    except (TrainingError, OSError) as exc:
        if isinstance(exc, TrainingError):
            raise
        raise TrainingDependencyError("private training staging root could not be created") from exc


def _move_training_root_noreplace(source: Path, destination: Path) -> bool:
    """Atomically rename one directory without replacing a destination."""

    source = Path(os.path.abspath(str(source)))
    destination = Path(os.path.abspath(str(destination)))
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileExW
        move_file.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        move_file.restype = ctypes.c_int
        if move_file(str(source), str(destination), 0):
            return True
        error = ctypes.get_last_error()
        if error in {80, 183} or (error == 5 and os.path.lexists(destination)):
            # ERROR_FILE_EXISTS / ERROR_ALREADY_EXISTS. Windows can also use
            # ERROR_ACCESS_DENIED when the existing destination is a directory.
            return False
        raise OSError(error, "no-replace training publication failed", str(destination))

    if sys.platform.startswith("linux"):
        import ctypes
        import errno

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise TrainingDependencyError("Linux renameat2 no-replace publication is unavailable")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
            return True
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            return False
        if error in {errno.ENOSYS, errno.EOPNOTSUPP}:
            raise TrainingDependencyError("Linux renameat2 no-replace publication is unavailable")
        raise OSError(error, "no-replace training publication failed", str(destination))

    raise TrainingDependencyError("production publication has no supported no-replace primitive")


def _publish_private_training_tree(
    tree: _PrivateTrainingTree,
    destination: Path,
    *,
    expected_entries: Sequence[Tuple[Any, ...]],
) -> _PrivateTrainingTree:
    """Publish an identity-frozen private tree or retain it without cleanup."""

    destination = Path(os.path.abspath(str(destination)))
    _verify_private_training_tree(tree, expected_entries=expected_entries)
    try:
        published = _move_training_root_noreplace(tree.root, destination)
    except TrainingError:
        raise
    except OSError as exc:
        raise TrainingDependencyError("private training output could not be published atomically") from exc
    if not published:
        raise TrainingIntegrityError("production output root was claimed before publication")
    published_tree = _PrivateTrainingTree(
        root=destination,
        root_identity=tree.root_identity,
        ancestor_identities=tree.ancestor_identities,
    )
    _verify_private_training_tree(published_tree, expected_entries=expected_entries)
    return published_tree


def _training_file_identity(value: os.stat_result, label: str) -> Tuple[int, int]:
    device = getattr(value, "st_dev", None)
    inode = getattr(value, "st_ino", None)
    if type(device) is not int or type(inode) is not int or inode == 0:
        raise TrainingIntegrityError("{} has no stable file identity".format(label))
    return device, inode


def _is_reparse(value: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(value, "st_file_attributes", 0) & reparse_flag)


def _training_ancestors(path: Path) -> Tuple[Path, ...]:
    ancestors = []
    cursor = path.parent
    while True:
        ancestors.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    return tuple(reversed(ancestors))


def _snapshot_training_ancestors(path: Path) -> Tuple[Tuple[Path, Tuple[int, int]], ...]:
    snapshot = []
    for ancestor in _training_ancestors(path):
        try:
            value = ancestor.lstat()
        except OSError as exc:
            raise TrainingConfigurationError("sealed training dataset ancestor is unavailable") from exc
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode) or _is_reparse(value):
            raise TrainingIntegrityError(
                "sealed training dataset must not traverse a symlink or reparse ancestor"
            )
        snapshot.append((ancestor, _training_file_identity(value, "sealed training dataset ancestor")))
    return tuple(snapshot)


def _verify_training_ancestors(snapshot: Sequence[Tuple[Path, Tuple[int, int]]]) -> None:
    for ancestor, expected_identity in snapshot:
        try:
            value = ancestor.lstat()
        except OSError as exc:
            raise TrainingIntegrityError("sealed training dataset ancestor changed during admission") from exc
        if (
            stat.S_ISLNK(value.st_mode)
            or not stat.S_ISDIR(value.st_mode)
            or _is_reparse(value)
            or _training_file_identity(value, "sealed training dataset ancestor") != expected_identity
        ):
            raise TrainingIntegrityError("sealed training dataset ancestor changed during admission")


def _validate_training_leaf_stat(value: os.stat_result, *, expected_identity: Optional[Tuple[int, int]] = None) -> None:
    if (
        stat.S_ISLNK(value.st_mode)
        or not stat.S_ISREG(value.st_mode)
        or _is_reparse(value)
        or getattr(value, "st_nlink", 1) != 1
    ):
        raise TrainingIntegrityError(
            "sealed training dataset must be a regular non-reparse single-link file"
        )
    if value.st_size < 1 or value.st_size > SEALED_RUNTIME_DATASET_MAX_BYTES:
        raise TrainingIntegrityError("sealed training dataset exceeds its explicit byte ceiling")
    if (
        expected_identity is not None
        and _training_file_identity(value, "sealed training dataset") != expected_identity
    ):
        raise TrainingIntegrityError("sealed training dataset changed during admission")


def _read_sealed_training_artifact(path: Path) -> bytes:
    """Single-open, bounded, no-alias read for the private training artifact."""

    target = Path(os.path.abspath(str(path)))
    ancestors = _snapshot_training_ancestors(target)
    try:
        before = target.lstat()
    except OSError as exc:
        raise TrainingConfigurationError("sealed training dataset is unavailable") from exc
    _validate_training_leaf_stat(before)
    expected_identity = _training_file_identity(before, "sealed training dataset")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(target), flags)
    except OSError as exc:
        raise TrainingConfigurationError("sealed training dataset cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        _validate_training_leaf_stat(opened, expected_identity=expected_identity)
        _verify_training_ancestors(ancestors)
        chunks = []
        remaining = SEALED_RUNTIME_DATASET_MAX_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > SEALED_RUNTIME_DATASET_MAX_BYTES:
            raise TrainingIntegrityError("sealed training dataset exceeds its explicit byte ceiling")
        after = os.fstat(descriptor)
        _validate_training_leaf_stat(after, expected_identity=expected_identity)
        _verify_training_ancestors(ancestors)
        try:
            visible = target.lstat()
        except OSError as exc:
            raise TrainingIntegrityError("sealed training dataset changed during admission") from exc
        _validate_training_leaf_stat(visible, expected_identity=expected_identity)
    finally:
        os.close(descriptor)
    return raw


def _attestation_tensor_digest(value: Any) -> str:
    """Hash one LoRA tensor without depending on pickle or device storage."""

    try:
        import torch

        if not torch.is_tensor(value):
            raise TypeError("value is not a tensor")
        tensor = value.detach().cpu().contiguous()
        raw = tensor.view(dtype=torch.uint8).numpy().tobytes()
    except Exception as exc:
        raise TrainingIntegrityError("LoRA change attestation contains an unreadable tensor") from exc
    return digest_for(
        {
            "shape": [int(item) for item in tensor.shape],
            "dtype": str(tensor.dtype),
            "bytes_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _normalized_lora_tensor(value: Any, name: str) -> Any:
    try:
        import torch

        if not torch.is_tensor(value):
            raise TypeError("value is not a tensor")
        tensor = value.detach().cpu().contiguous().clone()
        if tensor.dtype is not torch.float32:
            raise TrainingIntegrityError("LoRA attestation tensors must retain frozen float32 dtype: {}".format(name))
        if tensor.ndim != 2 or any(int(item) < 1 for item in tensor.shape):
            raise TrainingIntegrityError("LoRA attestation tensor shape is invalid: {}".format(name))
        if not bool(torch.isfinite(tensor).all().item()):
            raise TrainingIntegrityError("LoRA attestation tensors must be finite: {}".format(name))
        return tensor
    except TrainingIntegrityError:
        raise
    except Exception as exc:
        raise TrainingIntegrityError("LoRA change attestation contains an unreadable tensor: {}".format(name)) from exc


def _effective_lora_pair_changed(
    initial_a: Any,
    initial_b: Any,
    selected_a: Any,
    selected_b: Any,
    *,
    scaling: float,
    target_name: str,
) -> bool:
    """Check one effective B@A delta in bounded two-dimensional chunks."""

    import torch

    if (
        int(torch.count_nonzero(initial_b).item()) == 0
        and int(torch.count_nonzero(selected_b).item()) == 0
    ) or (
        int(torch.count_nonzero(initial_a).item()) == 0
        and int(torch.count_nonzero(selected_a).item()) == 0
    ):
        return False
    output_width = int(initial_b.shape[0])
    input_width = int(initial_a.shape[1])
    for row_start in range(0, output_width, LORA_EFFECTIVE_DELTA_ROW_CHUNK):
        row_end = min(output_width, row_start + LORA_EFFECTIVE_DELTA_ROW_CHUNK)
        for column_start in range(0, input_width, LORA_EFFECTIVE_DELTA_COLUMN_CHUNK):
            column_end = min(input_width, column_start + LORA_EFFECTIVE_DELTA_COLUMN_CHUNK)
            selected = torch.matmul(
                selected_b[row_start:row_end, :], selected_a[:, column_start:column_end]
            )
            initial = torch.matmul(
                initial_b[row_start:row_end, :], initial_a[:, column_start:column_end]
            )
            delta = (selected - initial) * scaling
            if not bool(torch.isfinite(delta).all().item()):
                raise TrainingIntegrityError(
                    "effective LoRA delta is non-finite for target {}".format(target_name)
                )
            if int(torch.count_nonzero(delta).item()) > 0:
                return True
    return False


def build_lora_change_attestation(
    initial_state: Mapping[str, Any],
    selected_state: Mapping[str, Any],
    *,
    target_manifest: Any,
    protocol: TrainingProtocol,
) -> Mapping[str, Any]:
    """Prove at least one exact target pair has a nonzero effective LoRA delta."""

    from .targets import LoraTargetManifest

    if type(target_manifest) is not LoraTargetManifest:
        raise TrainingIntegrityError("LoRA change attestation requires the exact target manifest")
    try:
        target_manifest.verify()
    except Exception as exc:
        raise TrainingIntegrityError("LoRA change attestation target manifest is invalid") from exc
    if type(protocol) is not TrainingProtocol:
        raise TrainingIntegrityError("LoRA change attestation requires the exact frozen protocol")
    protocol.validate()
    if not isinstance(initial_state, Mapping) or not isinstance(selected_state, Mapping):
        raise TrainingIntegrityError("LoRA change attestation requires tensor mappings")
    initial_names = tuple(initial_state)
    selected_names = tuple(selected_state)
    if any(not isinstance(name, str) or not name for name in initial_names + selected_names):
        raise TrainingIntegrityError("LoRA change attestation tensor names are invalid")
    names = tuple(sorted(initial_names))
    if not names or set(names) != set(selected_names):
        raise TrainingIntegrityError("selected LoRA tensor inventory differs from initialization")
    actual_names = set(names)
    expected_names = set()
    target_pairs = []
    for target_name in target_manifest.module_names:
        matches = []
        for wrapper in ("", "base_model.model."):
            prefix = wrapper + target_name
            a_name = prefix + ".lora_A.default.weight"
            b_name = prefix + ".lora_B.default.weight"
            a_present = a_name in actual_names
            b_present = b_name in actual_names
            if a_present != b_present:
                raise TrainingIntegrityError(
                    "LoRA target lacks an exact A/B pair: {}".format(target_name)
                )
            if a_present:
                matches.append((a_name, b_name))
        if len(matches) != 1:
            raise TrainingIntegrityError(
                "LoRA target lacks one exact target-bound A/B pair: {}".format(target_name)
            )
        a_name, b_name = matches[0]
        expected_names.update((a_name, b_name))
        target_pairs.append((target_name, a_name, b_name))
    if actual_names != expected_names:
        raise TrainingIntegrityError("LoRA state contains a missing or extra adapter tensor")

    # Snapshot the complete caller-owned inventories before hashing or doing
    # arithmetic.  CPU-contiguous tensors can otherwise retain caller storage.
    initial_tensors = {
        name: _normalized_lora_tensor(initial_state[name], name) for name in names
    }
    selected_tensors = {
        name: _normalized_lora_tensor(selected_state[name], name) for name in names
    }
    initial_records = []
    selected_records = []
    initial_digests = {}
    selected_digests = {}
    for name in names:
        initial_tensor = initial_tensors[name]
        selected_tensor = selected_tensors[name]
        if tuple(initial_tensor.shape) != tuple(selected_tensor.shape):
            raise TrainingIntegrityError("selected LoRA tensor shape changed from initialization: {}".format(name))
        if initial_tensor.dtype != selected_tensor.dtype:
            raise TrainingIntegrityError("selected LoRA tensor dtype changed from initialization: {}".format(name))
        initial_digest = _attestation_tensor_digest(initial_tensor)
        selected_digest = _attestation_tensor_digest(selected_tensor)
        initial_digests[name] = initial_digest
        selected_digests[name] = selected_digest
        initial_records.append({"name": name, "tensor_digest": initial_digest})
        selected_records.append({"name": name, "tensor_digest": selected_digest})

    scaling = float(protocol.lora_alpha) / float(protocol.lora_rank)
    effective_changed_targets = []
    for target_name, a_name, b_name in target_pairs:
        initial_a = initial_tensors[a_name]
        initial_b = initial_tensors[b_name]
        selected_a = selected_tensors[a_name]
        selected_b = selected_tensors[b_name]
        if (
            int(initial_a.shape[0]) != protocol.lora_rank
            or int(initial_b.shape[1]) != protocol.lora_rank
            or int(initial_a.shape[0]) != int(initial_b.shape[1])
        ):
            raise TrainingIntegrityError("LoRA A/B rank differs from the frozen protocol: {}".format(target_name))
        if (
            initial_digests[a_name] == selected_digests[a_name]
            and initial_digests[b_name] == selected_digests[b_name]
        ):
            continue
        if _effective_lora_pair_changed(
            initial_a,
            initial_b,
            selected_a,
            selected_b,
            scaling=scaling,
            target_name=target_name,
        ):
            effective_changed_targets.append(target_name)
    if not effective_changed_targets:
        raise TrainingIntegrityError("no exact target-bound LoRA pair has a nonzero effective delta")
    body = {
        "schema_version": LORA_CHANGE_ATTESTATION_SCHEMA,
        "target_manifest_digest": target_manifest.digest,
        "protocol_digest": protocol.digest,
        "lora_scaling": {"alpha": protocol.lora_alpha, "rank": protocol.lora_rank},
        "effective_delta_chunk_shape": [
            LORA_EFFECTIVE_DELTA_ROW_CHUNK,
            LORA_EFFECTIVE_DELTA_COLUMN_CHUNK,
        ],
        "parameter_count": len(names),
        "target_pair_count": len(target_pairs),
        "effective_delta_changed_target_count": len(effective_changed_targets),
        "initial_state_digest": digest_for(initial_records),
        "selected_state_digest": digest_for(selected_records),
        "effective_delta_changed_target_set_digest": digest_for(effective_changed_targets),
    }
    return {**body, "attestation_digest": digest_for(body)}


def _validate_lora_change_attestation(value: Mapping[str, Any]) -> Dict[str, Any]:
    from .targets import LORA_TARGET_COUNT

    required = {
        "schema_version",
        "target_manifest_digest",
        "protocol_digest",
        "lora_scaling",
        "effective_delta_chunk_shape",
        "parameter_count",
        "target_pair_count",
        "effective_delta_changed_target_count",
        "initial_state_digest",
        "selected_state_digest",
        "effective_delta_changed_target_set_digest",
        "attestation_digest",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise TrainingIntegrityError("LoRA change attestation has an unexpected field set")
    normalized = dict(value)
    body = dict(normalized)
    attestation_digest = body.pop("attestation_digest")
    try:
        for field in (
            "target_manifest_digest",
            "protocol_digest",
            "initial_state_digest",
            "selected_state_digest",
            "effective_delta_changed_target_set_digest",
            "attestation_digest",
        ):
            validate_sha256(normalized[field], field)
    except Exception as exc:
        raise TrainingIntegrityError("LoRA change attestation contains an invalid digest") from exc
    frozen_protocol = TrainingProtocol()
    if (
        normalized["schema_version"] != LORA_CHANGE_ATTESTATION_SCHEMA
        or normalized["lora_scaling"]
        != {"alpha": frozen_protocol.lora_alpha, "rank": frozen_protocol.lora_rank}
        or normalized["protocol_digest"] != frozen_protocol.digest
        or normalized["effective_delta_chunk_shape"]
        != [LORA_EFFECTIVE_DELTA_ROW_CHUNK, LORA_EFFECTIVE_DELTA_COLUMN_CHUNK]
        or normalized["parameter_count"] != 2 * LORA_TARGET_COUNT
        or normalized["target_pair_count"] != LORA_TARGET_COUNT
        or not isinstance(normalized["effective_delta_changed_target_count"], int)
        or isinstance(normalized["effective_delta_changed_target_count"], bool)
        or not 1 <= normalized["effective_delta_changed_target_count"] <= LORA_TARGET_COUNT
        or digest_for(body) != attestation_digest
    ):
        raise TrainingIntegrityError("LoRA change attestation is not internally bound")
    return normalized


def build_lora_sealed_binding(
    *,
    selected_change_attestation: Mapping[str, Any],
    reloaded_change_attestation: Mapping[str, Any],
    selected_checkpoint_artifact_digest: str,
    base_immutability_proof_digest: str,
    sealed_adapter_digest: str,
    evaluator_approved_adapter_digest: str,
    selected_development_receipt_digest: str,
    reloaded_applied_model_state_digest: str,
) -> Mapping[str, Any]:
    """Bind selected effective change to the sealed and reloaded adapter."""

    selected = _validate_lora_change_attestation(selected_change_attestation)
    reloaded = _validate_lora_change_attestation(reloaded_change_attestation)
    if selected != reloaded:
        raise TrainingIntegrityError("reloaded LoRA effective-delta proof differs from the evaluator-selected state")
    digests = {
        "selected_checkpoint_artifact_digest": selected_checkpoint_artifact_digest,
        "base_immutability_proof_digest": base_immutability_proof_digest,
        "sealed_adapter_digest": sealed_adapter_digest,
        "evaluator_approved_adapter_digest": evaluator_approved_adapter_digest,
        "selected_development_receipt_digest": selected_development_receipt_digest,
        "reloaded_applied_model_state_digest": reloaded_applied_model_state_digest,
    }
    try:
        digests = {name: validate_sha256(value, name) for name, value in digests.items()}
    except Exception as exc:
        raise TrainingIntegrityError("sealed LoRA binding contains an invalid digest") from exc
    if digests["sealed_adapter_digest"] != digests["evaluator_approved_adapter_digest"]:
        raise TrainingIntegrityError("sealed adapter differs from the evaluator-approved adapter")
    body = {
        "schema_version": LORA_SEALED_BINDING_SCHEMA,
        "lora_change_attestation_digest": selected["attestation_digest"],
        "target_manifest_digest": selected["target_manifest_digest"],
        "protocol_digest": selected["protocol_digest"],
        **digests,
    }
    return {**body, "binding_digest": digest_for(body)}


def _restore_selected_lora_state(
    model: Any,
    adapter_state: Mapping[str, Any],
    expected_lora_names: Sequence[str],
) -> None:
    """Restore an exact LoRA inventory while tolerating unrelated base omissions."""

    expected = set(expected_lora_names)
    if set(adapter_state) != expected:
        raise TrainingIntegrityError("selected checkpoint LoRA tensor inventory differs from initialization")
    try:
        incompatible = model.load_state_dict(adapter_state, strict=False)
    except Exception as exc:
        raise TrainingIntegrityError("selected checkpoint LoRA tensors could not be restored") from exc
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    if unexpected:
        raise TrainingIntegrityError("selected adapter state contains unexpected parameters")
    missing_lora = sorted(expected & set(getattr(incompatible, "missing_keys", ())))
    if missing_lora:
        raise TrainingIntegrityError("selected adapter state left expected LoRA parameters missing")


def _require_finite_loss(value: Any, source: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise TrainingIntegrityError("{} returned a non-finite loss".format(source))
    loss = float(value)
    if loss < 0:
        raise TrainingIntegrityError("{} returned a negative loss".format(source))
    return loss


@dataclass(frozen=True)
class TrainingRunReport:
    protocol_digest: str
    model_digest: str
    data_manifest_digest: str
    checkpoint_digests: Tuple[str, ...]
    development_evaluations: Tuple[DevelopmentLossEvaluation, ...]
    selected_checkpoint_digest: str
    selected_loss: float
    production: bool
    promotion_disposition: str
    real_qwen_execution_claimed: bool = False
    schema_version: str = TRAINING_RUNTIME_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "protocol_digest": self.protocol_digest,
            "model_digest": self.model_digest,
            "data_manifest_digest": self.data_manifest_digest,
            "checkpoint_digests": list(self.checkpoint_digests),
            "development_evaluations": [
                {
                    "checkpoint_digest": evaluation.checkpoint_digest,
                    "checkpoint_id": evaluation.checkpoint_id,
                    "model_digest": evaluation.model_digest,
                    "data_manifest_digest": evaluation.data_manifest_digest,
                    "development_manifest_digest": evaluation.development_manifest_digest,
                    "protocol_digest": evaluation.protocol_digest,
                    "gateway_digest": evaluation.gateway_digest,
                    "loss": evaluation.loss,
                    "sample_count": evaluation.sample_count,
                    "receipt": dict(evaluation.receipt),
                }
                for evaluation in self.development_evaluations
            ],
            "selected_checkpoint_digest": self.selected_checkpoint_digest,
            "selected_loss": self.selected_loss,
            "production": self.production,
            "promotion_disposition": self.promotion_disposition,
            "real_qwen_execution_claimed": self.real_qwen_execution_claimed,
        }

    def validate(self) -> None:
        if self.schema_version != TRAINING_RUNTIME_SCHEMA:
            raise TrainingIntegrityError("unsupported Training runtime report schema")
        for name in ("protocol_digest", "model_digest", "data_manifest_digest", "selected_checkpoint_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise TrainingIntegrityError("Training report {} is not a SHA-256 digest".format(name))
        if not self.checkpoint_digests or not self.development_evaluations:
            raise TrainingIntegrityError("Training report has no checkpoint/evaluation evidence")
        if self.selected_checkpoint_digest not in self.checkpoint_digests:
            raise TrainingIntegrityError("selected checkpoint is not in the evaluated checkpoint set")
        _require_finite_loss(self.selected_loss, "selected development loss")
        if self.promotion_disposition != "NOT_PROMOTED_TRAINING_ARTIFACT":
            raise TrainingIntegrityError("Training cannot emit a Variation promotion disposition")
        if self.real_qwen_execution_claimed != self.production:
            raise TrainingIntegrityError("Qwen execution claim must exactly match the production runtime tier")


class LoRATrainer:
    """Run the frozen LoRA protocol against sealed inputs and a dev gateway."""

    def __init__(
        self,
        inputs: SealedTrainingInputs,
        protocol: Optional[TrainingProtocol],
        gateway: Optional[Any],
        tokenizer: Any,
        *,
        production: bool = True,
        model_root: Optional[Path] = None,
        frozen_dataset: Optional[Any] = None,
        target_manifest: Optional[Any] = None,
        checkpoint_root: Optional[Path] = None,
        private_training_tree: Optional[_PrivateTrainingTree] = None,
    ) -> None:
        if type(inputs) is not SealedTrainingInputs:
            raise TrainingIntegrityError("LoRATrainer requires sealed Training inputs")
        if type(protocol) is not TrainingProtocol:
            raise TrainingConfigurationError("LoRATrainer requires the exact frozen TrainingProtocol")
        expected_gateway_type = ExternalDevelopmentLossGateway if production else DevelopmentLossGateway
        if type(gateway) is not expected_gateway_type:
            raise TrainingDependencyError(
                "production requires the external evaluator client" if production
                else "fixture runtime requires the local fixture gateway"
            )
        if not callable(tokenizer):
            raise TrainingDependencyError("LoRATrainer requires a tokenizer callable")
        protocol.validate()
        inputs.validate(protocol)
        evaluator_scratch_root = None
        evaluator_scratch_identity = None
        if production:
            from .contracts import FrozenTrainingDataset
            from .targets import LoraTargetManifest

            if type(frozen_dataset) is not FrozenTrainingDataset or type(target_manifest) is not LoraTargetManifest:
                raise TrainingIntegrityError(
                    "production Training requires the canonical frozen dataset and sealed target manifest"
                )
            if checkpoint_root is None:
                raise TrainingConfigurationError("production Training requires a checkpoint artifact root")
            if type(private_training_tree) is not _PrivateTrainingTree:
                raise TrainingConfigurationError("production Training requires a private staging tree")
            target_manifest.verify()
            if target_manifest.model_manifest_digest != inputs.model_digest:
                raise TrainingIntegrityError("target manifest is bound to a different pinned model")
            _verify_private_training_tree(private_training_tree)
            checkpoint_path = Path(os.path.abspath(str(checkpoint_root)))
            try:
                checkpoint_path.relative_to(private_training_tree.root)
            except ValueError as exc:
                raise TrainingIntegrityError("checkpoint root escapes the private staging tree") from exc
            evaluator_scratch_root, evaluator_scratch_identity = (
                _verify_external_gateway_scratch_binding(private_training_tree, gateway)
            )
        gateway.validate_production_boundary(
            expected_model_digest=inputs.model_digest,
            expected_protocol_digest=protocol.digest,
        ) if production else self._validate_fixture_gateway(gateway, inputs, protocol)
        if production:
            _verify_external_gateway_scratch_binding(private_training_tree, gateway)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "gateway", gateway)
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "production", bool(production))
        object.__setattr__(self, "model_root", Path(model_root).resolve() if model_root is not None else None)
        object.__setattr__(self, "frozen_dataset", frozen_dataset)
        object.__setattr__(self, "target_manifest", target_manifest)
        object.__setattr__(
            self,
            "checkpoint_root",
            Path(os.path.abspath(str(checkpoint_root))) if checkpoint_root is not None else None,
        )
        object.__setattr__(self, "_private_training_tree", private_training_tree)
        object.__setattr__(
            self,
            "_contract_digest",
            digest_for(
                {
                    "inputs_seal": inputs.seal_digest,
                    "protocol_digest": protocol.digest,
                    "gateway_digest": gateway.gateway_digest,
                    "frozen_dataset_digest": frozen_dataset.digest if frozen_dataset is not None else None,
                    "target_manifest_digest": target_manifest.digest if target_manifest is not None else None,
                    "private_training_root": str(private_training_tree.root) if private_training_tree else None,
                    "private_training_identity": list(private_training_tree.root_identity) if private_training_tree else None,
                    "evaluator_scratch_root": (
                        str(evaluator_scratch_root) if evaluator_scratch_root is not None else None
                    ),
                    "evaluator_scratch_identity": (
                        list(evaluator_scratch_identity)
                        if evaluator_scratch_identity is not None
                        else None
                    ),
                    "tokenizer_id": id(tokenizer),
                    "production": bool(production),
                }
            ),
        )
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False) and name in {
            "inputs",
            "protocol",
            "gateway",
            "tokenizer",
            "production",
            "model_root",
            "frozen_dataset",
            "target_manifest",
            "checkpoint_root",
            "_private_training_tree",
            "_contract_digest",
        }:
            raise AttributeError("LoRATrainer runtime contract is immutable")
        object.__setattr__(self, name, value)

    @staticmethod
    def _validate_fixture_gateway(
        gateway: DevelopmentLossGateway, inputs: SealedTrainingInputs, protocol: TrainingProtocol
    ) -> None:
        if gateway.production:
            raise TrainingDependencyError("fixture Training runtime requires a non-production gateway")
        if gateway.model_digest != inputs.model_digest or gateway.protocol_digest != protocol.digest:
            raise TrainingIntegrityError("fixture gateway does not bind the sealed model/protocol")

    def _validate_run_boundary(self) -> None:
        if type(self) is not LoRATrainer:
            raise TrainingDependencyError("production requires the exact LoRATrainer type")
        self.protocol.validate()
        self.inputs.validate(self.protocol)
        evaluator_scratch_root = None
        evaluator_scratch_identity = None
        if self.production:
            evaluator_scratch_root, evaluator_scratch_identity = (
                _verify_external_gateway_scratch_binding(
                    self._private_training_tree, self.gateway
                )
            )
        if digest_for(
            {
                "inputs_seal": self.inputs.seal_digest,
                "protocol_digest": self.protocol.digest,
                "gateway_digest": self.gateway.gateway_digest,
                "frozen_dataset_digest": self.frozen_dataset.digest if self.frozen_dataset is not None else None,
                "target_manifest_digest": self.target_manifest.digest if self.target_manifest is not None else None,
                "private_training_root": (
                    str(self._private_training_tree.root) if self._private_training_tree else None
                ),
                "private_training_identity": (
                    list(self._private_training_tree.root_identity)
                    if self._private_training_tree else None
                ),
                "evaluator_scratch_root": (
                    str(evaluator_scratch_root) if evaluator_scratch_root is not None else None
                ),
                "evaluator_scratch_identity": (
                    list(evaluator_scratch_identity)
                    if evaluator_scratch_identity is not None
                    else None
                ),
                "tokenizer_id": id(self.tokenizer),
                "production": self.production,
            }
        ) != self._contract_digest:
            raise TrainingIntegrityError("Training runtime contract changed after construction")
        if self.production:
            if sys.version_info < (3, 10):
                raise TrainingDependencyError("production LoRA training requires Python >=3.10")
            if self.inputs.fixture_only:
                raise TrainingDependencyError("fixture-only Training inputs cannot enter production")
            self.gateway.validate_production_boundary(
                expected_model_digest=self.inputs.model_digest,
                expected_protocol_digest=self.protocol.digest,
            )
            _verify_external_gateway_scratch_binding(
                self._private_training_tree, self.gateway
            )
            self._validate_production_model()
        else:
            if not self.inputs.fixture_only:
                raise TrainingDependencyError("fixture Training runtime requires fixture-only sealed inputs")
            self._validate_fixture_gateway(self.gateway, self.inputs, self.protocol)

    def _validate_production_model(self) -> None:
        try:
            from ..variation.model import LoadedPinnedModel, PinnedModelManifest
        except ImportError as exc:
            raise TrainingDependencyError("production Training requires the merged Variation model contract") from exc
        if type(self.inputs.model) is not LoadedPinnedModel:
            raise TrainingDependencyError("production Training requires a LoadedPinnedModel, not a mock or digest")
        loaded = self.inputs.model
        if type(loaded.manifest) is not PinnedModelManifest:
            raise TrainingDependencyError("production Training requires the exact pinned model manifest")
        loaded.manifest.validate_contract()
        if loaded.manifest_digest != self.inputs.model_digest:
            raise TrainingIntegrityError("sealed Training model digest does not match the pinned manifest")
        try:
            peft = importlib.import_module("peft")
        except ImportError as exc:
            raise TrainingDependencyError("production LoRA training requires PEFT") from exc
        for name in ("LoraConfig", "get_peft_model"):
            if not callable(getattr(peft, name, None)):
                raise TrainingDependencyError("production PEFT runtime lacks {}".format(name))

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            return

    def _apply_production_lora(self) -> Any:
        self._validate_production_model()
        peft = importlib.import_module("peft")
        base_model = self.inputs.model.model
        try:
            import torch
        except ImportError as exc:
            raise TrainingDependencyError("production Training requires torch") from exc
        floating_base = [parameter for parameter in base_model.parameters() if parameter.is_floating_point()]
        if not floating_base or any(
            parameter.device.type != self.protocol.device or parameter.dtype is not torch.bfloat16
            for parameter in floating_base
        ):
            raise TrainingIntegrityError("pinned base model is not entirely CUDA bfloat16")
        from .immutability import capture_base_model_snapshot
        from .targets import validate_lora_target_modules

        try:
            validate_lora_target_modules(base_model)
        except Exception as exc:
            raise TrainingIntegrityError("pinned model differs from the sealed 114-target manifest") from exc
        target_manifest = self.target_manifest
        if self.model_root is None:
            raise TrainingConfigurationError("production Training requires the sealed base-model root")
        try:
            snapshot = capture_base_model_snapshot(
                base_model,
                model_root=self.model_root,
                model_manifest_digest=self.inputs.model.manifest_digest,
                target_manifest=target_manifest,
                file_hashes=self.inputs.model.file_hashes,
            )
        except Exception as exc:
            raise TrainingIntegrityError("base-model immutability snapshot failed") from exc
        config = peft.LoraConfig(
            r=self.protocol.lora_rank,
            lora_alpha=self.protocol.lora_alpha,
            lora_dropout=self.protocol.lora_dropout,
            target_modules=list(target_manifest.module_names),
            bias="none",
            task_type="CAUSAL_LM",
        )
        try:
            model = peft.get_peft_model(base_model, config)
        except Exception as exc:
            raise TrainingDependencyError("the frozen PEFT LoRA configuration could not be applied") from exc
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=torch.float32)
        for name, parameter in model.named_parameters():
            if not parameter.is_floating_point():
                continue
            expected_dtype = torch.float32 if parameter.requires_grad else torch.bfloat16
            if parameter.device.type != self.protocol.device or parameter.dtype is not expected_dtype:
                raise TrainingIntegrityError(
                    "production parameter device/dtype differs from CUDA BF16 plus FP32 LoRA exception: {}".format(name)
                )

        from .immutability import assert_lora_only_trainable

        try:
            trainable_names = assert_lora_only_trainable(model, target_manifest)
        except Exception as exc:
            raise TrainingIntegrityError("PEFT trainability differs from the sealed LoRA targets") from exc
        initial_lora_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if "lora_" in name
        }
        if set(initial_lora_state) != set(trainable_names):
            raise TrainingIntegrityError("PEFT state contains a missing or extra LoRA tensor")
        object.__setattr__(self, "_target_manifest", target_manifest)
        object.__setattr__(self, "_base_snapshot", snapshot)
        object.__setattr__(self, "_initial_lora_state", initial_lora_state)
        return model

    @staticmethod
    def _fixture_train_step(model: Any, example: TokenizedTrainingExample) -> float:
        train_step = getattr(model, "train_step", None)
        if not callable(train_step):
            raise TrainingDependencyError("fixture model must expose train_step()")
        return _require_finite_loss(train_step(example), "fixture train_step")

    def _production_train_step(
        self, model: Any, example: TokenizedTrainingExample, optimizer: Any, *, accumulation_divisor: int
    ) -> float:
        try:
            import torch
        except ImportError as exc:
            raise TrainingDependencyError("production LoRA training requires torch") from exc
        model.train()
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError) as exc:
            raise TrainingDependencyError("production model exposes no device-bound parameters") from exc
        input_ids = torch.tensor([list(example.input_ids)], dtype=torch.long, device=device)
        attention_mask = torch.tensor([list(example.attention_mask)], dtype=torch.long, device=device)
        labels = torch.tensor([list(example.labels)], dtype=torch.long, device=device)
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            value = _require_finite_loss(float(loss.detach().cpu().item()), "production model")
            if accumulation_divisor < 1:
                raise TrainingConfigurationError("gradient accumulation divisor must be positive")
            (loss / accumulation_divisor).backward()
            return value
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingDependencyError("production LoRA forward/backward failed") from exc

    def _make_checkpoint(self, model: Any, *, epoch: int, global_step: int) -> TrainingCheckpoint:
        state_digest = model_state_digest_for_training(model)
        adapter_digest = digest_for(
            {
                "model_digest": self.inputs.model_digest,
                "protocol_digest": self.protocol.digest,
                "state_digest": state_digest,
                "epoch": epoch,
            }
        )
        payload_digest = digest_for(
            {
                "adapter_digest": adapter_digest,
                "state_digest": state_digest,
                "global_step": global_step,
            }
        )
        return TrainingCheckpoint.create(
            epoch=epoch,
            global_step=global_step,
            model_digest=self.inputs.model_digest,
            protocol_digest=self.protocol.digest,
            data_manifest_digest=self.inputs.data_manifest.digest,
            adapter_digest=adapter_digest,
            state_digest=state_digest,
            payload_digest=payload_digest,
        )

    def run(self) -> TrainingRunReport:
        self._validate_run_boundary()
        self._seed_everything(self.protocol.train_seed)
        batches = [
            build_training_batch((row,), self.tokenizer, self.protocol)[0]
            for row in self.inputs.train_rows
        ]
        if len(batches) > self.protocol.max_train_rows:
            raise TrainingConfigurationError("sealed Training rows exceed the frozen row ceiling")
        model = self._apply_production_lora() if self.production else self.inputs.model
        optimizer = None
        scheduler = None
        if self.production:
            try:
                import torch

                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                optimizer = torch.optim.AdamW(
                    trainable,
                    lr=self.protocol.learning_rate,
                    weight_decay=self.protocol.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda step: max(0.0, 1.0 - (float(step) / float(self.protocol.max_train_steps))),
                )
            except TrainingError:
                raise
            except Exception as exc:
                raise TrainingDependencyError("frozen optimizer/scheduler could not be constructed") from exc
        checkpoints: List[TrainingCheckpoint] = []
        evaluations: List[DevelopmentLossEvaluation] = []
        best: Optional[DevelopmentLossEvaluation] = None
        best_checkpoint: Optional[TrainingCheckpoint] = None
        best_evaluation_snapshot: Optional[str] = None
        best_evaluation_snapshot_digest: Optional[str] = None
        best_checkpoint_snapshot: Optional[str] = None
        best_checkpoint_snapshot_digest: Optional[str] = None
        no_improvement = 0
        global_step = 0
        accumulation = 0
        for epoch in range(1, self.protocol.max_epochs + 1):
            for batch_index, example in enumerate(batches):
                if self.production:
                    chunk_start = batch_index - (batch_index % self.protocol.gradient_accumulation_steps)
                    accumulation_divisor = min(
                        self.protocol.gradient_accumulation_steps, len(batches) - chunk_start
                    )
                    loss = self._production_train_step(
                        model, example, optimizer, accumulation_divisor=accumulation_divisor
                    )
                else:
                    loss = self._fixture_train_step(model, example)
                _require_finite_loss(loss, "training")
                accumulation += 1
                if accumulation >= self.protocol.gradient_accumulation_steps or example is batches[-1]:
                    if self.production:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                    global_step += 1
                    accumulation = 0
                    if global_step >= self.protocol.max_train_steps:
                        break
            checkpoint = self._make_checkpoint(model, epoch=epoch, global_step=max(global_step, 1))
            if self.production:
                object.__setattr__(self, "_trained_model", model)
                object.__setattr__(self, "_optimizer", optimizer)
                object.__setattr__(self, "_scheduler", scheduler)
                object.__setattr__(self, "_global_step", global_step)
                object.__setattr__(self, "_epoch_count", epoch)
                parent = getattr(self, "_checkpoint_head_digest", None)
                head = self._save_foundation_checkpoint(self.checkpoint_root, parent_artifact_digest=parent)
                object.__setattr__(self, "_checkpoint_head_digest", head)
                evaluation = self.gateway.evaluate(
                    checkpoint, model=model, checkpoint_artifact_digest=head
                )
            else:
                evaluation = self.gateway.evaluate(checkpoint, model=model)
            if self.production:
                self.gateway.verify_evaluation(
                    evaluation,
                    checkpoint=checkpoint,
                    expected_checkpoint_artifact_digest=head,
                    expected_candidate_adapter_digest=evaluation.receipt.get(
                        "candidate_artifact_digest"
                    ),
                    expected_model_digest=self.inputs.model_digest,
                    expected_data_manifest_digest=self.inputs.data_manifest.digest,
                    expected_protocol_digest=self.protocol.digest,
                )
            else:
                self.gateway.verify_evaluation(
                    evaluation,
                    checkpoint=checkpoint,
                    expected_model_digest=self.inputs.model_digest,
                    expected_data_manifest_digest=self.inputs.data_manifest.digest,
                    expected_protocol_digest=self.protocol.digest,
                )
            checkpoints.append(checkpoint)
            evaluations.append(evaluation)
            if best is None or (evaluation.loss, evaluation.checkpoint_digest) < (best.loss, best.checkpoint_digest):
                best = evaluation
                best_checkpoint = checkpoint
                best_evaluation_snapshot = canonical_json(evaluation.to_dict())
                best_evaluation_snapshot_digest = digest_for(
                    json.loads(best_evaluation_snapshot)
                )
                best_checkpoint_snapshot = canonical_json(checkpoint.to_dict())
                best_checkpoint_snapshot_digest = digest_for(
                    json.loads(best_checkpoint_snapshot)
                )
                if self.production:
                    best_adapter_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items() if "lora_" in name
                    }
                    object.__setattr__(self, "_best_adapter_state", best_adapter_state)
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement > self.protocol.early_stopping_patience or global_step >= self.protocol.max_train_steps:
                break
        if (
            best is None
            or best_checkpoint is None
            or best_evaluation_snapshot is None
            or best_evaluation_snapshot_digest is None
            or best_checkpoint_snapshot is None
            or best_checkpoint_snapshot_digest is None
        ):
            raise TrainingIntegrityError("Training produced no valid development evaluation")
        if canonical_json(best.to_dict()) != best_evaluation_snapshot:
            raise TrainingIntegrityError("selected development evaluation changed after selection")
        selected_evaluation_value = json.loads(best_evaluation_snapshot)
        selected_evaluation = DevelopmentLossEvaluation(**selected_evaluation_value)
        object.__setattr__(self, "_selected_evaluation", best)
        object.__setattr__(self, "_selected_evaluation_snapshot", best_evaluation_snapshot)
        object.__setattr__(self, "_selected_evaluation_snapshot_digest", best_evaluation_snapshot_digest)
        object.__setattr__(self, "_selected_checkpoint_snapshot", best_checkpoint_snapshot)
        object.__setattr__(self, "_selected_checkpoint_snapshot_digest", best_checkpoint_snapshot_digest)
        if self.production:
            import torch
            from .checkpoint import TrainingCheckpointStore

            selected_artifact_digest = selected_evaluation.checkpoint_digest
            authoritative = TrainingCheckpointStore(self.checkpoint_root).resume(
                expected_artifact_digest=selected_artifact_digest
            )
            expected_adapter = self._best_adapter_state
            actual_adapter = authoritative.adapter_weights
            if set(actual_adapter) != set(expected_adapter) or any(
                not torch.equal(actual_adapter[name].detach().cpu(), expected_adapter[name])
                for name in expected_adapter
            ):
                raise TrainingIntegrityError("selected checkpoint adapter tensors differ from the evaluator-approved state")
            _restore_selected_lora_state(
                model,
                actual_adapter,
                tuple(self._initial_lora_state),
            )
            change_attestation = build_lora_change_attestation(
                self._initial_lora_state,
                actual_adapter,
                target_manifest=self._target_manifest,
                protocol=self.protocol,
            )
            object.__setattr__(self, "_selected_checkpoint_artifact_digest", selected_artifact_digest)
            object.__setattr__(self, "_selected_authoritative_adapter", dict(actual_adapter))
            object.__setattr__(self, "_lora_change_attestation", dict(change_attestation))
        report = TrainingRunReport(
            protocol_digest=self.protocol.digest,
            model_digest=self.inputs.model_digest,
            data_manifest_digest=self.inputs.data_manifest.digest,
            checkpoint_digests=tuple(
                evaluation.checkpoint_digest for evaluation in evaluations
            ) if self.production else tuple(checkpoint.digest for checkpoint in checkpoints),
            development_evaluations=tuple(evaluations),
            selected_checkpoint_digest=selected_evaluation.checkpoint_digest,
            selected_loss=selected_evaluation.loss,
            production=self.production,
            promotion_disposition="NOT_PROMOTED_TRAINING_ARTIFACT",
            real_qwen_execution_claimed=self.production,
        )
        report.validate()
        if self.production:
            from .immutability import validate_lora_only_training_state

            try:
                proof = validate_lora_only_training_state(
                    self._base_snapshot,
                    model,
                    target_manifest=self._target_manifest,
                    optimizer=optimizer,
                )
            except Exception as exc:
                raise TrainingIntegrityError("post-training base/optimizer immutability proof failed") from exc
            object.__setattr__(self, "_trained_model", model)
            object.__setattr__(self, "_immutability_proof", proof)
            object.__setattr__(self, "_optimizer", optimizer)
            object.__setattr__(self, "_scheduler", scheduler)
            object.__setattr__(self, "_global_step", global_step)
            object.__setattr__(self, "_epoch_count", len(evaluations))
        return report

    @staticmethod
    def _json_list(value: Any) -> Any:
        if isinstance(value, tuple):
            return [LoRATrainer._json_list(item) for item in value]
        if isinstance(value, list):
            return [LoRATrainer._json_list(item) for item in value]
        return value

    def _verify_private_output_tree(self) -> None:
        if self.production:
            if type(self._private_training_tree) is not _PrivateTrainingTree:
                raise TrainingIntegrityError("production private staging identity is unavailable")
            _verify_private_training_tree(self._private_training_tree)
            if type(getattr(self, "gateway", None)) is ExternalDevelopmentLossGateway:
                _verify_external_gateway_scratch_binding(
                    self._private_training_tree, self.gateway
                )

    def _reverify_selected_evaluation(self) -> DevelopmentLossEvaluation:
        """Rebuild and re-authorize the immutable selected loss evidence."""

        current = getattr(self, "_selected_evaluation", None)
        evaluation_snapshot = getattr(self, "_selected_evaluation_snapshot", None)
        evaluation_digest = getattr(self, "_selected_evaluation_snapshot_digest", None)
        checkpoint_snapshot = getattr(self, "_selected_checkpoint_snapshot", None)
        checkpoint_digest = getattr(self, "_selected_checkpoint_snapshot_digest", None)
        if (
            type(current) is not DevelopmentLossEvaluation
            or not isinstance(evaluation_snapshot, str)
            or not isinstance(checkpoint_snapshot, str)
        ):
            raise TrainingIntegrityError("selected development evaluation snapshot is unavailable")
        try:
            current_snapshot = canonical_json(current.to_dict())
            evaluation_value = json.loads(evaluation_snapshot)
            checkpoint_value = json.loads(checkpoint_snapshot)
        except Exception as exc:
            raise TrainingIntegrityError("selected development evaluation snapshot is invalid") from exc
        if (
            current_snapshot != evaluation_snapshot
            or canonical_json(evaluation_value) != evaluation_snapshot
            or digest_for(evaluation_value) != evaluation_digest
        ):
            raise TrainingIntegrityError("selected development evaluation changed after selection")
        if (
            canonical_json(checkpoint_value) != checkpoint_snapshot
            or digest_for(checkpoint_value) != checkpoint_digest
        ):
            raise TrainingIntegrityError("selected Training checkpoint snapshot changed after selection")
        try:
            frozen_evaluation = DevelopmentLossEvaluation(**evaluation_value)
            frozen_checkpoint = TrainingCheckpoint.from_mapping(checkpoint_value)
            frozen_evaluation.validate()
        except Exception as exc:
            raise TrainingIntegrityError("selected development evidence cannot be reconstructed") from exc
        selected_artifact_digest = getattr(self, "_selected_checkpoint_artifact_digest", None)
        if (
            frozen_checkpoint.digest != checkpoint_digest
            or frozen_evaluation.checkpoint_digest != selected_artifact_digest
            or frozen_evaluation.checkpoint_id != frozen_checkpoint.checkpoint_id
            or frozen_evaluation.model_digest != frozen_checkpoint.model_digest
            or frozen_evaluation.data_manifest_digest != frozen_checkpoint.data_manifest_digest
            or frozen_evaluation.protocol_digest != frozen_checkpoint.protocol_digest
        ):
            raise TrainingIntegrityError("selected development evaluation and checkpoint differ")
        verification_arguments = {
            "checkpoint": frozen_checkpoint,
            "expected_model_digest": self.inputs.model_digest,
            "expected_data_manifest_digest": self.inputs.data_manifest.digest,
            "expected_protocol_digest": self.protocol.digest,
        }
        if type(self.gateway) is ExternalDevelopmentLossGateway:
            verification_arguments.update({
                "expected_checkpoint_artifact_digest": selected_artifact_digest,
                "expected_candidate_adapter_digest": frozen_evaluation.receipt.get(
                    "candidate_artifact_digest"
                ),
            })
        self.gateway.verify_evaluation(frozen_evaluation, **verification_arguments)
        if (
            canonical_json(frozen_evaluation.to_dict()) != evaluation_snapshot
            or canonical_json(current.to_dict()) != evaluation_snapshot
            or digest_for(json.loads(evaluation_snapshot)) != evaluation_digest
        ):
            raise TrainingIntegrityError("selected development evaluation changed during revalidation")
        return frozen_evaluation

    def _save_foundation_checkpoint(
        self, output_root: Path, *, parent_artifact_digest: Optional[str] = None
    ) -> str:
        from importlib import metadata as importlib_metadata
        import numpy as np
        import torch

        from .checkpoint import TrainingCheckpoint as FoundationCheckpoint
        from .checkpoint import TrainingCheckpointStore

        self._verify_private_output_tree()
        model = self._trained_model
        named = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
        parameter_names = sorted(named.values())
        optimizer_tensors = {}
        tensor_parameter_names = {}
        for parameter, state in self._optimizer.state.items():
            parameter_name = named.get(id(parameter))
            if parameter_name is None:
                raise TrainingIntegrityError("optimizer state references a non-LoRA parameter")
            for field, value in sorted(state.items()):
                if torch.is_tensor(value):
                    if value.is_floating_point() and value.dtype is not torch.float32:
                        raise TrainingIntegrityError("optimizer floating state is not the frozen FP32 exception")
                    tensor_name = "{}.{}".format(parameter_name, field)
                    optimizer_tensors[tensor_name] = value.detach().cpu()
                    tensor_parameter_names[tensor_name] = parameter_name
        if not optimizer_tensors:
            raise TrainingIntegrityError("production optimizer emitted no resumable tensor state")
        groups = []
        for group in self._optimizer.param_groups:
            group_names = sorted(named[id(parameter)] for parameter in group["params"])
            groups.append({
                "parameter_names": group_names,
                "lr": float(group["lr"]),
                "betas": [float(item) for item in group["betas"]],
                "eps": float(group["eps"]),
                "weight_decay": float(group["weight_decay"]),
            })
        py_state = random.getstate()
        np_state = np.random.get_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        rng_tensors = {"torch_cpu_rng": torch.get_rng_state().detach().cpu()}
        for index, value in enumerate(cuda_states):
            rng_tensors["torch_cuda_rng.{}".format(index)] = value.detach().cpu()
        scheduler_state = self._scheduler.state_dict()
        tokenizer_digest = digest_for({
            "class": type(self.tokenizer).__name__,
            "name_or_path": str(getattr(self.tokenizer, "name_or_path", "local-pinned")),
            "vocab_size": int(getattr(self.tokenizer, "vocab_size", 0)),
        })
        versions = {}
        for package in ("torch", "transformers", "peft", "safetensors", "numpy"):
            try:
                versions[package] = importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError as exc:
                raise TrainingDependencyError("production software manifest is incomplete") from exc
        checkpoint = FoundationCheckpoint(
            campaign_id=self.frozen_dataset.cutoff.campaign_id,
            run_id=content_id("training-run", {
                "dataset": self.frozen_dataset.digest,
                "protocol": self.protocol.digest,
                "target": self.target_manifest.digest,
            }),
            global_step=self._global_step,
            frozen_dataset_digest=self.frozen_dataset.digest,
            target_manifest_digest=self.target_manifest.digest,
            tokenizer_manifest_digest=tokenizer_digest,
            software_manifest_digest=digest_for(versions),
            base_model_manifest_hash=self.inputs.model_digest,
            base_snapshot_digest=self._base_snapshot.digest,
            ledger_cutoff_hash=self.frozen_dataset.cutoff.digest,
            protocol_hash=self.protocol.digest,
            data_cursor={
                "schema_version": "egv-training-cursor-v1", "epoch": self._epoch_count,
                "row_index": 0, "consumed_examples": self._epoch_count * len(self.inputs.train_rows),
                "sampler_seed": self.protocol.train_seed, "dataset_digest": self.frozen_dataset.digest,
            },
            scheduler_state={
                "schema_version": "egv-training-scheduler-v1", "scheduler_class": type(self._scheduler).__name__,
                "last_epoch": max(0, int(scheduler_state.get("last_epoch", 0))),
                "step_count": self._global_step,
                "base_lrs": [float(item) for item in scheduler_state.get("base_lrs", [self.protocol.learning_rate])],
                "last_lrs": [float(item) for item in self._scheduler.get_last_lr()],
            },
            rng_state={
                "schema_version": "egv-training-rng-v1",
                "python_state": {"version": py_state[0], "state": list(py_state[1]), "gauss": py_state[2]},
                "numpy_state": {
                    "bit_generator": str(np_state[0]), "keys": [int(item) for item in np_state[1]],
                    "position": int(np_state[2]), "has_gauss": int(np_state[3]),
                    "cached_gaussian": float(np_state[4]),
                },
                "torch_cpu_tensor_name": "torch_cpu_rng",
                "torch_cuda_tensor_names": ["torch_cuda_rng.{}".format(index) for index in range(len(cuda_states))],
                "cuda_device_count": len(cuda_states),
            },
            optimizer_metadata={
                "schema_version": "egv-training-optimizer-v1", "optimizer_class": type(self._optimizer).__name__,
                "parameter_names": parameter_names, "parameter_groups": groups,
                "tensor_parameter_names": tensor_parameter_names,
            },
            parent_artifact_digest=parent_artifact_digest,
        )
        adapter_weights = {
            name: value.detach().cpu() for name, value in model.state_dict().items() if "lora_" in name
        }
        _path, artifact_digest = TrainingCheckpointStore(Path(output_root)).save(
            checkpoint, adapter_weights=adapter_weights, optimizer_tensors=optimizer_tensors, rng_tensors=rng_tensors
        )
        self._verify_private_output_tree()
        return artifact_digest

    def seal_adapter(self, output_root: Path) -> Mapping[str, Any]:
        """Persist and re-open the real PEFT adapter as a sealed Variation artifact."""

        if not self.production or not hasattr(self, "_trained_model"):
            raise TrainingIntegrityError("adapter sealing requires a completed production run")
        destination = _verify_private_adapter_destination(
            self._private_training_tree, output_root
        )
        attested = getattr(self, "_lora_change_attestation", None)
        if not isinstance(attested, Mapping):
            raise TrainingIntegrityError("adapter sealing requires a LoRA change attestation")
        current_state = {
            name: value
            for name, value in self._trained_model.state_dict().items()
            if "lora_" in name
        }
        current_attestation = build_lora_change_attestation(
            self._initial_lora_state,
            current_state,
            target_manifest=self._target_manifest,
            protocol=self.protocol,
        )
        if dict(current_attestation) != dict(attested):
            raise TrainingIntegrityError("LoRA state changed after its selected-state attestation")
        if os.path.lexists(destination):
            raise TrainingIntegrityError("adapter output directory must be newly absent")
        self._verify_private_output_tree()
        selected_evaluation = self._reverify_selected_evaluation()
        try:
            destination.mkdir()
        except FileExistsError as exc:
            raise TrainingIntegrityError("adapter output directory must be newly absent") from exc
        self._verify_private_output_tree()
        try:
            self._trained_model.save_pretrained(str(destination), safe_serialization=True)
        except Exception as exc:
            raise TrainingDependencyError("PEFT could not write the adapter artifact") from exc
        self._verify_private_output_tree()
        from ..variation.adapter import (
            ADAPTER_MANIFEST_NAME,
            SealedAdapterArtifact,
            build_local_adapter_manifest,
        )

        manifest = build_local_adapter_manifest(destination)
        (destination / ADAPTER_MANIFEST_NAME).write_text(
            canonical_json(manifest.to_dict()) + "\n", encoding="utf-8"
        )
        self._verify_private_output_tree()
        artifact = SealedAdapterArtifact(destination)
        artifact.verify()
        approved_adapter_digest = selected_evaluation.receipt.get("candidate_artifact_digest")
        if artifact.digest != approved_adapter_digest:
            raise TrainingIntegrityError(
                "published adapter digest differs from the evaluator-approved selected adapter"
            )
        from ..variation.model import PinnedModelLoader
        try:
            import torch
            reloaded = PinnedModelLoader(self.model_root).load(
                device=self.protocol.device,
                torch_dtype=torch.bfloat16,
                adapter_artifact=artifact,
            )
        except Exception as exc:
            raise TrainingIntegrityError("sealed adapter failed real pinned-model reload") from exc
        self._verify_private_output_tree()
        attestation = reloaded.adapter_attestation
        if (
            attestation is None
            or attestation.adapter_digest != artifact.digest
            or attestation.base_model_manifest_digest != self.inputs.model_digest
            or reloaded.adapter_digest != artifact.digest
        ):
            raise TrainingIntegrityError("reloaded adapter attestation differs from the selected sealed artifact")
        reloaded_lora_state = {
            name: value
            for name, value in reloaded.model.state_dict().items()
            if "lora_" in name
        }
        reloaded_change_attestation = build_lora_change_attestation(
            self._initial_lora_state,
            reloaded_lora_state,
            target_manifest=self._target_manifest,
            protocol=self.protocol,
        )
        checkpoint_digest = self._selected_checkpoint_artifact_digest
        immutability_proof_digest = digest_for(self._json_list(dict(self._immutability_proof)))
        sealed_binding = build_lora_sealed_binding(
            selected_change_attestation=current_attestation,
            reloaded_change_attestation=reloaded_change_attestation,
            selected_checkpoint_artifact_digest=checkpoint_digest,
            base_immutability_proof_digest=immutability_proof_digest,
            sealed_adapter_digest=artifact.digest,
            evaluator_approved_adapter_digest=approved_adapter_digest,
            selected_development_receipt_digest=receipt_hash(selected_evaluation.receipt),
            reloaded_applied_model_state_digest=attestation.applied_model_state_digest,
        )
        self._verify_private_output_tree()
        return {
            "adapter_manifest_digest": artifact.digest,
            "target_manifest_digest": self._target_manifest.digest,
            "base_immutability_proof_digest": immutability_proof_digest,
            "adapter_root": str(destination.resolve()),
            "checkpoint_artifact_digest": checkpoint_digest,
            "lora_change_attestation": dict(current_attestation),
            "lora_sealed_binding": dict(sealed_binding),
        }


class _FixtureTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool, truncation: bool) -> Mapping[str, Sequence[int]]:
        if add_special_tokens or truncation:
            raise AssertionError("fixture tokenizer received a mutable tokenization mode")
        return {"input_ids": [index + 1 for index, _ in enumerate(text.split())] or [1]}


class _FixtureModel:
    def __init__(self) -> None:
        self.steps = 0
        self._state = b"fixture-state-0"

    def state_dict(self) -> Mapping[str, bytes]:
        return {"fixture.weight": self._state}

    def train_step(self, _example: TokenizedTrainingExample) -> float:
        self.steps += 1
        self._state = "fixture-state-{}".format(self.steps).encode("ascii")
        return 1.0 / float(self.steps)


def run_training_smoke() -> Dict[str, Any]:
    """Run a CPU-only fixture smoke without claiming Qwen or promotion."""

    protocol = TrainingProtocol()
    train_rows = (
        TrainingRow.create(
            task_id="egv-pure_function-train-1-v1",
            task_family="PURE_FUNCTION",
            split="train",
            prompt="repair the pure function",
            target="return the corrected result",
        ),
        TrainingRow.create(
            task_id="egv-pure_function-train-2-v1",
            task_family="PURE_FUNCTION",
            split="train",
            prompt="repair the second pure function",
            target="return the corrected result",
        ),
    )
    development_rows = (
        TrainingRow.create(
            task_id="egv-pure_function-dev-1-v1",
            task_family="PURE_FUNCTION",
            split="dev",
            prompt="development prompt",
            target="development target",
        ),
    )
    from .protocol import TrainingDataManifest

    manifest = TrainingDataManifest.from_rows(train_rows, development_rows)
    model = _FixtureModel()
    model_digest = digest_for("fixture-base-model")

    def evaluate(_model: Any, _rows: Sequence[TrainingRow], _protocol: TrainingProtocol) -> float:
        return 1.0 / float(getattr(_model, "steps", 0) + 1)

    signer = ReceiptSigner(b"egv-training-fixture-signer-32bytes"[:32])
    gateway = DevelopmentLossGateway(
        development_rows,
        data_manifest=manifest,
        model_digest=model_digest,
        protocol=protocol,
        signer=signer,
        evaluator=evaluate,
        production=False,
    )
    inputs = seal_training_inputs(
        model,
        model_digest=model_digest,
        train_rows=train_rows,
        data_manifest=manifest,
        protocol=protocol,
        fixture_only=True,
    )
    report = LoRATrainer(inputs, protocol, gateway, _FixtureTokenizer(), production=False).run()
    result = report.to_dict()
    result.update(
        {
            "smoke": "PASS",
            "runtime_tier": "floor-cpu-training-fixture",
            "real_qwen_execution_claimed": False,
            "promotion_claimed": False,
            "private_development_content_published": False,
            "limitations": [
                "CPU fixture only; the pinned Qwen checkpoint was not loaded",
                "PEFT production execution is fail-closed and unclaimed here",
                "no Variation promotion is emitted by the Training slice",
            ],
        }
    )
    return result


def _load_frozen_runtime_dataset(
    path: Path,
    *,
    expected_artifact_sha256: str,
    expected_dataset_digest: str,
) -> Any:
    """Load the private export of the canonical ``FrozenTrainingDataset``."""

    from .contracts import FrozenTrainingDataset, LedgerCutoff, TrainingExample

    try:
        validate_sha256(expected_artifact_sha256, "sealed training artifact SHA-256")
        validate_sha256(expected_dataset_digest, "sealed training dataset digest")
    except Exception as exc:
        raise TrainingConfigurationError(str(exc)) from exc
    raw = _read_sealed_training_artifact(path)
    if digest_bytes(raw) != expected_artifact_sha256:
        raise TrainingIntegrityError("sealed training artifact SHA-256 differs from the operator-captured value")
    try:
        value = __import__("json").loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise TrainingConfigurationError("sealed training dataset cannot be decoded") from exc
    try:
        canonical_raw = (canonical_json(value) + "\n").encode("utf-8")
    except Exception as exc:
        raise TrainingConfigurationError("sealed training dataset is not canonical JSON") from exc
    if raw != canonical_raw:
        raise TrainingIntegrityError("sealed training artifact bytes differ from canonical freezer output")
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "manifest", "private_rows"}:
        raise TrainingConfigurationError("sealed training dataset is not a closed artifact")
    if value["schema_version"] != SEALED_RUNTIME_DATASET_SCHEMA:
        raise TrainingConfigurationError("sealed training dataset schema is unsupported")
    manifest = value["manifest"]
    rows = value["private_rows"]
    if not isinstance(manifest, Mapping) or not isinstance(rows, list):
        raise TrainingConfigurationError("sealed training dataset payload is malformed")
    cutoff_value = manifest.get("cutoff")
    if not isinstance(cutoff_value, Mapping):
        raise TrainingConfigurationError("sealed training dataset cutoff is missing")
    try:
        cutoff = LedgerCutoff(**dict(cutoff_value))
        examples = []
        for row in rows:
            if not isinstance(row, Mapping) or "sft_row" not in row:
                raise TrainingConfigurationError("sealed training private row is malformed")
            payload = dict(row)
            sft_row = payload.pop("sft_row")
            schema_version = payload.pop("schema_version", None)
            if schema_version != "egv-private-training-example-v1":
                raise TrainingConfigurationError("sealed training private row schema is unsupported")
            sft_json = canonical_json(sft_row)
            payload["sft_row_json"] = sft_json
            examples.append(TrainingExample(**payload))
        dataset = FrozenTrainingDataset(
            cutoff=cutoff,
            examples=tuple(examples),
            excluded_counts=manifest.get("excluded_counts", {}),
        )
    except (TypeError, ValueError) as exc:
        raise TrainingIntegrityError("sealed canonical training dataset failed validation") from exc
    if dataset.manifest() != dict(manifest):
        raise TrainingIntegrityError("sealed training private rows differ from the frozen manifest")
    if dataset.digest != expected_dataset_digest:
        raise TrainingIntegrityError("sealed training semantic dataset digest differs from the operator-captured value")
    tasks = sorted({row.task_id for row in dataset.examples})
    if len(tasks) != 20:
        raise TrainingIntegrityError("production selection requires exactly 20 frozen training tasks")
    return dataset


def run_production_training(
    *,
    model_root: Path,
    training_dataset: Path,
    evaluator_manifest: Path,
    evaluator_public_key: Path,
    evaluator_command: Path,
    evaluator_transfer_command: Path,
    output_root: Path,
    expected_training_artifact_sha256: str,
    expected_training_dataset_digest: str,
    device: str = "cuda",
) -> Mapping[str, Any]:
    """Execute the bounded real-Qwen LoRA lane from sealed local artifacts."""

    if device != "cuda":
        raise TrainingConfigurationError("production LoRA training is frozen to the CUDA device")
    if sys.version_info < (3, 10):
        raise TrainingDependencyError("production LoRA training requires Python >=3.10")
    output_root = Path(os.path.abspath(str(output_root)))
    if os.path.lexists(output_root):
        raise TrainingIntegrityError("production output root must be newly absent")
    try:
        expected_training_artifact_sha256 = validate_sha256(
            expected_training_artifact_sha256, "sealed training artifact SHA-256"
        )
        expected_training_dataset_digest = validate_sha256(
            expected_training_dataset_digest, "sealed training dataset digest"
        )
    except Exception as exc:
        raise TrainingConfigurationError(str(exc)) from exc

    dataset = _load_frozen_runtime_dataset(
        training_dataset,
        expected_artifact_sha256=expected_training_artifact_sha256,
        expected_dataset_digest=expected_training_dataset_digest,
    )
    private_tree = _create_private_training_tree(output_root)
    from ..variation.model import PinnedModelLoader
    from .development import ExternalDevelopmentLossGateway
    from .protocol import TrainingDataManifest

    protocol = TrainingProtocol()
    evaluator_scratch = private_tree.root / "evaluator-scratch"
    _verify_private_training_tree(private_tree)
    try:
        evaluator_scratch.mkdir()
    except FileExistsError as exc:
        raise TrainingIntegrityError("external evaluator scratch root was preclaimed") from exc
    _verify_private_training_tree(private_tree)

    def verify_evaluator_scratch() -> None:
        _verify_private_training_tree(private_tree)

    gateway = ExternalDevelopmentLossGateway(
        evaluator_manifest, public_key_path=evaluator_public_key, command=evaluator_command,
        transfer_command=evaluator_transfer_command,
        scratch_root=evaluator_scratch,
        scratch_validator=verify_evaluator_scratch,
    )
    try:
        import torch
    except ImportError as exc:
        raise TrainingDependencyError("production LoRA training requires torch") from exc
    if not torch.cuda.is_available():
        raise TrainingDependencyError("production LoRA training requires an available CUDA device")
    loaded = PinnedModelLoader(model_root).load(device=device, torch_dtype=torch.bfloat16)
    if gateway.model_digest != loaded.manifest_digest:
        raise TrainingIntegrityError("external evaluator and pinned model manifest differ")
    # The canonical dataset remains the authority. The deterministic policy
    # chooses the earliest accepted B/D trajectory for each of the 20 tasks.
    selected = []
    for task_id in sorted({row.task_id for row in dataset.examples}):
        candidates = [row for row in dataset.examples if row.task_id == task_id]
        selected.append(min(candidates, key=lambda row: (row.attempt_index, row.arm_id, row.seed, row.candidate_id)))
    rows = tuple(
        TrainingRow.create(
            task_id=row.task_id,
            task_family=__import__("json").loads(row.sft_row_json)["task_family"],
            split="train",
            prompt=row.prompt,
            target=row.target,
            source_event_ids=(row.sft_row_digest,),
            row_id=row.row_id,
        )
        for row in selected
    )
    ordered = tuple(sorted(rows, key=lambda row: row.row_id))
    train_digest = __import__("egv.canonical", fromlist=["collection_digest"]).collection_digest(
        row.public_binding() for row in ordered
    )
    runtime_manifest = TrainingDataManifest(
        train_row_ids=tuple(row.row_id for row in ordered),
        development_row_ids=gateway.development_row_ids,
        train_digest=train_digest,
        development_digest=gateway.development_manifest_digest,
    )
    inputs = seal_training_inputs(
        loaded,
        model_digest=loaded.manifest_digest,
        train_rows=ordered,
        data_manifest=runtime_manifest,
        protocol=protocol,
        fixture_only=False,
    )
    from .targets import build_lora_target_manifest

    target_manifest = build_lora_target_manifest(
        loaded.model, model_manifest_digest=loaded.manifest_digest
    )
    trainer = LoRATrainer(
        inputs, protocol, gateway, loaded.tokenizer, production=True, model_root=model_root,
        frozen_dataset=dataset, target_manifest=target_manifest,
        checkpoint_root=private_tree.root / "checkpoints",
        private_training_tree=private_tree,
    )
    report = trainer.run()
    sealed = dict(trainer.seal_adapter(private_tree.root / "adapter"))
    completed_entries = _verify_private_training_tree(private_tree)
    _publish_private_training_tree(
        private_tree,
        output_root,
        expected_entries=completed_entries,
    )
    sealed["adapter_root"] = str(output_root / "adapter")
    return {
        **report.to_dict(),
        **sealed,
        "training_dataset_digest": dataset.digest,
        "training_artifact_sha256": expected_training_artifact_sha256,
        "selected_training_row_count": len(selected),
        "selection_policy": "EARLIEST_ACCEPTED_BD_PER_TASK_V1",
    }


__all__ = [
    "LORA_CHANGE_ATTESTATION_SCHEMA",
    "LORA_SEALED_BINDING_SCHEMA",
    "LoRATrainer",
    "SEALED_RUNTIME_DATASET_MAX_BYTES",
    "TRAINING_RUNTIME_SCHEMA",
    "TrainingRunReport",
    "build_lora_change_attestation",
    "build_lora_sealed_binding",
    "run_training_smoke",
    "run_production_training",
]
