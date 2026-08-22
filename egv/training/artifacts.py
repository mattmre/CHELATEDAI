"""Atomic content-addressed artifact primitives for EGV Training.

Published artifacts are immutable directories.  A directory name is the
SHA-256 of its closed manifest, and that manifest exhaustively hashes every
other regular file in the directory.  Verification uses ``lstat`` and rejects
links so an apparently valid artifact cannot be redirected or replaced by a
second name after publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Dict, Mapping, Optional

from ..canonical import canonical_bytes, digest_for, validate_sha256
from .errors import TrainingArtifactError


ARTIFACT_MANIFEST_SCHEMA = "egv-training-artifact-manifest-v1"
MANIFEST_NAME = "manifest.json"
_MANIFEST_FIELDS = frozenset({"schema_version", "artifact_kind", "metadata_digest", "files"})


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability (Windows does not expose this mode)."""

    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise TrainingArtifactError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise TrainingArtifactError(f"{label} must be a regular non-link file")
    if info.st_nlink != 1:
        raise TrainingArtifactError(f"{label} must not be hard-linked")
    return info


def sha256_file(path: Path) -> str:
    _regular_file(Path(path), label="artifact file")
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise TrainingArtifactError("artifact file cannot be hashed") from exc
    return digest.hexdigest()


def _safe_name(name: Any) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise TrainingArtifactError("artifact file name is unsafe")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1 or name == MANIFEST_NAME:
        raise TrainingArtifactError("artifact file name is unsafe")
    return name


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_kind: str
    metadata_digest: str
    files: Mapping[str, str]
    schema_version: str = ARTIFACT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_MANIFEST_SCHEMA:
            raise TrainingArtifactError("unsupported training artifact manifest schema")
        if not isinstance(self.artifact_kind, str) or not self.artifact_kind:
            raise TrainingArtifactError("artifact kind must be non-empty")
        validate_sha256(self.metadata_digest, "artifact metadata digest")
        if not isinstance(self.files, Mapping) or not self.files:
            raise TrainingArtifactError("artifact manifest must hash at least one file")
        normalized: Dict[str, str] = {}
        for raw_name, raw_digest in self.files.items():
            name = _safe_name(raw_name)
            normalized[name] = validate_sha256(raw_digest, f"artifact digest for {name}")
        object.__setattr__(self, "files", dict(sorted(normalized.items())))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "metadata_digest": self.metadata_digest,
            "files": dict(self.files),
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactManifest":
        if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
            raise TrainingArtifactError("training artifact manifest is not closed")
        files = value.get("files")
        if not isinstance(files, Mapping):
            raise TrainingArtifactError("training artifact file inventory is malformed")
        return cls(
            schema_version=value.get("schema_version"),
            artifact_kind=value.get("artifact_kind"),
            metadata_digest=value.get("metadata_digest"),
            files=files,
        )


class ContentAddressedArtifactStore:
    """Publish and verify immutable artifact directories below one root."""

    def __init__(self, root: Path) -> None:
        raw = Path(root)
        if raw.exists() and raw.is_symlink():
            raise TrainingArtifactError("artifact store root may not be a symlink")
        raw.mkdir(parents=True, exist_ok=True)
        self.root = raw.resolve()

    def _artifact_path(self, digest: str) -> Path:
        digest = validate_sha256(digest, "artifact digest")
        return self.root / "sha256" / digest[:2] / digest

    def publish(
        self,
        *,
        artifact_kind: str,
        metadata_digest: str,
        files: Mapping[str, bytes],
    ) -> tuple[Path, ArtifactManifest]:
        if not isinstance(files, Mapping) or not files:
            raise TrainingArtifactError("artifact publication requires files")
        normalized: Dict[str, bytes] = {}
        for raw_name, payload in files.items():
            name = _safe_name(raw_name)
            if not isinstance(payload, bytes):
                raise TrainingArtifactError("artifact payloads must be bytes")
            normalized[name] = payload
        manifest = ArtifactManifest(
            artifact_kind=artifact_kind,
            metadata_digest=metadata_digest,
            files={name: hashlib.sha256(payload).hexdigest() for name, payload in normalized.items()},
        )
        destination = self._artifact_path(manifest.digest)
        if destination.exists():
            self.verify(destination, expected_digest=manifest.digest, expected_kind=artifact_kind)
            return destination, manifest
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".publishing-", dir=str(parent)))
        try:
            for name, payload in sorted(normalized.items()):
                path = staging / name
                with path.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            manifest_path = staging / MANIFEST_NAME
            with manifest_path.open("xb") as handle:
                handle.write(canonical_bytes(manifest.to_dict()))
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(staging)
            try:
                staging.replace(destination)
            except FileExistsError:
                self.verify(destination, expected_digest=manifest.digest, expected_kind=artifact_kind)
            _fsync_directory(parent)
        finally:
            if staging.exists():
                # Only an interrupted unpublished staging tree is cleaned up.
                for child in staging.iterdir():
                    child.unlink()
                staging.rmdir()
        self.verify(destination, expected_digest=manifest.digest, expected_kind=artifact_kind)
        return destination, manifest

    def verify(
        self,
        path: Path,
        *,
        expected_digest: Optional[str] = None,
        expected_kind: Optional[str] = None,
    ) -> ArtifactManifest:
        raw = Path(path)
        if raw.is_symlink():
            raise TrainingArtifactError("artifact directory may not be a symlink")
        resolved = raw.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise TrainingArtifactError("artifact is outside its store") from exc
        if not resolved.is_dir():
            raise TrainingArtifactError("artifact directory is unavailable")
        manifest_path = resolved / MANIFEST_NAME
        _regular_file(manifest_path, label="artifact manifest")
        try:
            raw_manifest = manifest_path.read_bytes()
            value = json.loads(raw_manifest.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise TrainingArtifactError("artifact manifest cannot be decoded") from exc
        manifest = ArtifactManifest.from_mapping(value)
        if raw_manifest != canonical_bytes(manifest.to_dict()):
            raise TrainingArtifactError("artifact manifest is not canonical JSON")
        if resolved.name != manifest.digest:
            raise TrainingArtifactError("artifact directory is not content-addressed")
        if expected_digest is not None and manifest.digest != validate_sha256(expected_digest, "expected artifact digest"):
            raise TrainingArtifactError("artifact digest does not match the trusted checkpoint reference")
        if expected_kind is not None and manifest.artifact_kind != expected_kind:
            raise TrainingArtifactError("artifact kind differs from the trusted contract")
        actual_names = set()
        for child in resolved.iterdir():
            if child.name == MANIFEST_NAME:
                continue
            _regular_file(child, label=f"artifact member {child.name}")
            actual_names.add(child.name)
        if actual_names != set(manifest.files):
            raise TrainingArtifactError("artifact file inventory is not exhaustive")
        for name, expected in manifest.files.items():
            if sha256_file(resolved / name) != expected:
                raise TrainingArtifactError(f"artifact member digest changed: {name}")
        return manifest


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA",
    "ArtifactManifest",
    "ContentAddressedArtifactStore",
    "MANIFEST_NAME",
    "sha256_file",
]
