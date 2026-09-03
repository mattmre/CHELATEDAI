"""Sealed, content-addressed LoRA adapter artifacts for Variation.

Training is intentionally outside this branch.  Variation accepts an adapter
only after Training has emitted a complete local tree plus an immutable
manifest.  A bare hexadecimal digest is never an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from ..canonical import digest_for
from .errors import VariationConfigurationError, VariationDependencyError


ADAPTER_MANIFEST_SCHEMA = "egv-sealed-lora-adapter-v1"
ADAPTER_MANIFEST_NAME = "adapter-manifest.json"
_REQUIRED_FILES = frozenset({"adapter_config.json", "adapter_model.safetensors"})
_MANIFEST_FIELDS = frozenset({"schema_version", "base_model_revision", "adapter_type", "files"})


def _normalize_config_value(value: Any) -> Any:
    """Convert PEFT config values to the same JSON-shaped form as the file."""

    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _normalize_config_value(enum_value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_config_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        normalized = [_normalize_config_value(item) for item in value]
        return sorted(normalized, key=lambda item: repr(item)) if isinstance(value, (set, frozenset)) else normalized
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise VariationDependencyError("PEFT adapter config contains an unsupported value")


def _peft_runtime() -> Tuple[Any, Tuple[type, ...]]:
    try:
        peft = importlib.import_module("peft")
    except ImportError as exc:
        raise VariationDependencyError("sealed LoRA adapter requires the local PEFT runtime") from exc
    classes = tuple(
        candidate
        for candidate in (
            getattr(peft, "PeftModel", None),
            getattr(peft, "PeftModelForCausalLM", None),
        )
        if isinstance(candidate, type)
    )
    if not classes:
        raise VariationDependencyError("local PEFT runtime exposes no supported PeftModel type")
    return peft, classes


def _sealed_adapter_config(root: Path) -> Mapping[str, Any]:
    path = _safe_relative(root, "adapter_config.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VariationConfigurationError("sealed adapter config cannot be read") from exc
    if not isinstance(value, Mapping):
        raise VariationConfigurationError("sealed adapter config must be a JSON object")
    normalized = _normalize_config_value(value)
    if not isinstance(normalized, Mapping):
        raise VariationConfigurationError("sealed adapter config is not object-shaped")
    return normalized


def validate_applied_peft_model(model: Any, artifact: "SealedAdapterArtifact") -> None:
    """Require a real PEFT wrapper and an active config bound to the sealed tree."""

    if type(artifact) is not SealedAdapterArtifact:
        raise VariationDependencyError("PEFT validation requires the exact sealed adapter artifact")
    artifact.verify()
    _peft, supported_types = _peft_runtime()
    if not isinstance(model, supported_types):
        raise VariationDependencyError("sealed adapter model is not a local PeftModel/PeftModelForCausalLM instance")
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise VariationDependencyError("applied PEFT model does not expose measurable state")
    try:
        state = state_dict()
    except Exception as exc:
        raise VariationDependencyError("applied PEFT model state cannot be inspected") from exc
    if not isinstance(state, Mapping) or not state:
        raise VariationDependencyError("applied PEFT model state is empty or not mapping-shaped")

    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        active = getattr(model, "active_adapter", None)
        if callable(active):
            active = active()
    if isinstance(active, str):
        active_names = (active,)
    elif isinstance(active, (tuple, list, set, frozenset)):
        active_names = tuple(str(item) for item in active)
    else:
        active_names = ()
    if len(active_names) != 1 or not active_names[0]:
        raise VariationDependencyError("sealed adapter model does not expose exactly one active adapter")

    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, Mapping) or active_names[0] not in configs:
        raise VariationDependencyError("active PEFT adapter has no matching runtime config")
    runtime_config = configs[active_names[0]]
    to_dict = getattr(runtime_config, "to_dict", None)
    if callable(to_dict):
        runtime_config = to_dict()
    if not isinstance(runtime_config, Mapping):
        raise VariationDependencyError("active PEFT config is not mapping-shaped")
    runtime_config = _normalize_config_value(runtime_config)
    expected_config = dict(_sealed_adapter_config(Path(artifact.root).resolve()))
    expected_config.setdefault("peft_type", "LORA")
    actual_type = str(runtime_config.get("peft_type", "")).upper()
    if actual_type != "LORA":
        raise VariationDependencyError("active PEFT config is not LORA")
    for key, expected in expected_config.items():
        if key == "target_modules" and isinstance(expected, list):
            actual_targets = runtime_config.get(key)
            if (
                not isinstance(actual_targets, list)
                or len(actual_targets) != len(set(actual_targets))
                or sorted(actual_targets) != sorted(expected)
            ):
                raise VariationDependencyError("active PEFT target modules differ from the sealed adapter config")
            continue
        if key not in runtime_config or runtime_config[key] != expected:
            raise VariationDependencyError("active PEFT config is not bound to the sealed adapter config")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or Path(value).is_absolute():
        raise VariationConfigurationError("adapter manifest contains an unsafe path")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise VariationConfigurationError("adapter manifest path escapes its root") from exc
    if candidate.name == ADAPTER_MANIFEST_NAME or ".." in Path(value).parts:
        raise VariationConfigurationError("adapter manifest cannot hash itself or parent paths")
    return candidate


def _regular_tree_files(root: Path, *, excluded: Optional[Path] = None) -> Tuple[Path, ...]:
    """Enumerate an exhaustive adapter tree without following links."""

    root = Path(root)
    if root.is_symlink():
        raise VariationConfigurationError("sealed adapter root may not be a symlink")
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise VariationConfigurationError("sealed adapter root does not exist or is not a directory")
    excluded_resolved = excluded.resolve() if excluded is not None else None
    files = []
    for path in sorted(resolved_root.rglob("*")):
        if path.is_symlink():
            raise VariationConfigurationError("sealed adapter tree may not contain symlinks: {}".format(path.name[:48]))
        if path.is_dir():
            continue
        if not path.is_file():
            raise VariationConfigurationError("sealed adapter tree may contain only regular files")
        if excluded_resolved is not None and path.resolve() == excluded_resolved:
            continue
        files.append(path)
    return tuple(files)


@dataclass(frozen=True)
class SealedAdapterManifest:
    schema_version: str
    base_model_revision: str
    adapter_type: str
    files: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SealedAdapterManifest":
        if set(value) != _MANIFEST_FIELDS:
            raise VariationConfigurationError("sealed adapter manifest has an unexpected field set")
        files = value.get("files")
        if not isinstance(files, Mapping) or not files:
            raise VariationConfigurationError("sealed adapter manifest must hash its complete file tree")
        normalized: Dict[str, str] = {}
        for relative, digest in files.items():
            if (
                not isinstance(relative, str)
                or not relative
                or "\\" in relative
                or Path(relative).is_absolute()
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise VariationConfigurationError("sealed adapter entries must be relative path/SHA-256 pairs")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise VariationConfigurationError("sealed adapter contains a non-hex digest") from exc
            normalized[relative] = digest
        manifest = cls(
            schema_version=str(value["schema_version"]),
            base_model_revision=str(value["base_model_revision"]),
            adapter_type=str(value["adapter_type"]),
            files=normalized,
        )
        manifest.validate_contract()
        return manifest

    @classmethod
    def from_file(cls, path: Path) -> "SealedAdapterManifest":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise VariationConfigurationError("sealed adapter manifest cannot be read") from exc
        if not isinstance(value, Mapping):
            raise VariationConfigurationError("sealed adapter manifest must be a JSON object")
        return cls.from_mapping(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_model_revision": self.base_model_revision,
            "adapter_type": self.adapter_type,
            "files": dict(sorted(self.files.items())),
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())

    def validate_contract(self) -> None:
        from .model import MODEL_REVISION

        if self.schema_version != ADAPTER_MANIFEST_SCHEMA:
            raise VariationConfigurationError("unsupported sealed adapter manifest schema")
        if self.base_model_revision != MODEL_REVISION:
            raise VariationConfigurationError("sealed adapter is not bound to the frozen base model")
        if self.adapter_type != "LORA":
            raise VariationConfigurationError("Variation accepts only the frozen LORA adapter type")
        if not _REQUIRED_FILES.issubset(set(self.files)):
            raise VariationConfigurationError("sealed adapter must include adapter_config.json and safetensors weights")


@dataclass(frozen=True)
class SealedAdapterArtifact:
    """A verified adapter tree produced by the later Training slice."""

    root: Path
    manifest_path: Optional[Path] = None

    def _paths(self) -> Tuple[Path, Path, SealedAdapterManifest]:
        raw_root = Path(self.root)
        raw_manifest = self.manifest_path or raw_root / ADAPTER_MANIFEST_NAME
        if raw_root.is_symlink() or raw_manifest.is_symlink():
            raise VariationConfigurationError("sealed adapter root and manifest may not be symlinks")
        root = raw_root.resolve()
        if not root.is_dir():
            raise VariationConfigurationError("sealed adapter root does not exist")
        manifest_path = raw_manifest.resolve()
        try:
            manifest_path.relative_to(root)
        except ValueError as exc:
            raise VariationConfigurationError("sealed adapter manifest is outside its root") from exc
        if manifest_path.name != ADAPTER_MANIFEST_NAME:
            raise VariationConfigurationError("sealed adapter manifest has the wrong filename")
        return root, manifest_path, SealedAdapterManifest.from_file(manifest_path)

    @property
    def manifest(self) -> SealedAdapterManifest:
        return self._paths()[2]

    @property
    def digest(self) -> str:
        return self.manifest.digest

    def verify(self) -> Mapping[str, str]:
        root, manifest_path, manifest = self._paths()
        expected_paths = set(manifest.files)
        actual_paths = {path.relative_to(root).as_posix() for path in _regular_tree_files(root, excluded=manifest_path)}
        if actual_paths != expected_paths:
            extra = sorted(actual_paths - expected_paths)
            missing = sorted(expected_paths - actual_paths)
            raise VariationConfigurationError(
                "sealed adapter tree is not exhaustive (extra={}, missing={})".format(extra, missing)
            )
        hashes: Dict[str, str] = {}
        for relative, expected in sorted(manifest.files.items()):
            path = _safe_relative(root, relative)
            if not path.is_file() or path.is_symlink():
                raise VariationConfigurationError("sealed adapter file is missing or symlinked: {}".format(relative))
            actual = _sha256(path)
            if actual != expected:
                raise VariationConfigurationError("sealed adapter digest mismatch: {}".format(relative))
            hashes[relative] = actual
        return hashes

    def apply_to(self, model: Any) -> Any:
        """Apply the verified adapter through an explicit local PEFT boundary."""

        self.verify()
        peft, supported_types = _peft_runtime()
        adapter_model = getattr(peft, "PeftModel", None) or getattr(peft, "PeftModelForCausalLM", None)
        if not isinstance(adapter_model, type) or not hasattr(adapter_model, "from_pretrained"):
            raise VariationDependencyError("local PEFT runtime cannot apply the sealed adapter")
        try:
            applied = adapter_model.from_pretrained(
                model,
                str(self.root),
                local_files_only=True,
                is_trainable=False,
            )
        except Exception as exc:
            raise VariationDependencyError("sealed LoRA adapter failed local application") from exc
        if not isinstance(applied, supported_types):
            raise VariationDependencyError("sealed adapter application did not return a local PEFT model")
        validate_applied_peft_model(applied, self)
        return applied


def build_local_adapter_manifest(root: Path) -> SealedAdapterManifest:
    """Build a manifest for a Training-produced local adapter tree."""

    from .model import MODEL_REVISION

    root = Path(root)
    files: Dict[str, str] = {}
    manifest_path = root / ADAPTER_MANIFEST_NAME
    for path in _regular_tree_files(root, excluded=manifest_path):
        files[path.relative_to(root.resolve()).as_posix()] = _sha256(path)
    return SealedAdapterManifest.from_mapping(
        {
            "schema_version": ADAPTER_MANIFEST_SCHEMA,
            "base_model_revision": MODEL_REVISION,
            "adapter_type": "LORA",
            "files": files,
        }
    )


__all__ = [
    "ADAPTER_MANIFEST_NAME",
    "ADAPTER_MANIFEST_SCHEMA",
    "SealedAdapterArtifact",
    "SealedAdapterManifest",
    "build_local_adapter_manifest",
    "validate_applied_peft_model",
]
