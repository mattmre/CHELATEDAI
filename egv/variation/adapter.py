"""Sealed, content-addressed LoRA adapter artifacts for Variation.

Training is intentionally outside this branch.  Variation accepts an adapter
only after Training has emitted a complete local tree plus an immutable
manifest.  A bare hexadecimal digest is never an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from ..canonical import digest_for
from .errors import VariationConfigurationError, VariationDependencyError


ADAPTER_MANIFEST_SCHEMA = "egv-sealed-lora-adapter-v1"
ADAPTER_MANIFEST_NAME = "adapter-manifest.json"
_REQUIRED_FILES = frozenset({"adapter_config.json", "adapter_model.safetensors"})
_MANIFEST_FIELDS = frozenset({"schema_version", "base_model_revision", "adapter_type", "files"})


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
        root = self.root.resolve()
        if not root.is_dir():
            raise VariationConfigurationError("sealed adapter root does not exist")
        manifest_path = (self.manifest_path or root / ADAPTER_MANIFEST_NAME).resolve()
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
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.resolve() != manifest_path
        }
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
        try:
            peft = __import__("peft", fromlist=["PeftModel"])
            adapter_model = getattr(peft, "PeftModel")
        except (ImportError, AttributeError) as exc:
            raise VariationDependencyError("sealed LoRA adapter requires the local PEFT runtime") from exc
        try:
            return adapter_model.from_pretrained(
                model,
                str(self.root),
                local_files_only=True,
                is_trainable=False,
            )
        except Exception as exc:
            raise VariationDependencyError("sealed LoRA adapter failed local application") from exc


def build_local_adapter_manifest(root: Path) -> SealedAdapterManifest:
    """Build a manifest for a Training-produced local adapter tree."""

    from .model import MODEL_REVISION

    files: Dict[str, str] = {}
    manifest_path = (root / ADAPTER_MANIFEST_NAME).resolve()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.resolve() != manifest_path:
            files[path.relative_to(root).as_posix()] = _sha256(path)
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
]
