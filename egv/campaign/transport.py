"""Content-addressed, fail-closed transport for Campaign artifacts.

This module deliberately has no network implementation.  An operator may move
the manifest and blobs using an approved channel; the receiving side admits
them only after every byte and closed-schema binding has been verified.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

from ..canonical import canonical_bytes, digest_bytes, digest_for, validate_sha256
from .errors import CampaignError
from .state import _durable_replace, _exclusive_path_lock


class ArtifactTransportError(CampaignError):
    """A staged campaign artifact is incomplete, substituted, or unsafe."""


TRANSFER_SCHEMA = "egv-campaign-transfer-manifest-v1"
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024 * 1024
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DESCRIPTOR_FIELDS = frozenset({"artifact_id", "role", "media_type", "byte_count", "sha256"})
_MANIFEST_FIELDS = frozenset(
    {"schema_version", "campaign_id", "transfer_id", "source_role", "destination_role", "artifacts"}
)


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ArtifactTransportError("{} is unsafe or ambiguous".format(field))
    return value


def _sha(value: Any, field: str) -> str:
    try:
        return validate_sha256(value, field)
    except Exception as exc:
        raise ArtifactTransportError(str(exc)) from exc


@dataclass(frozen=True)
class ArtifactDescriptor:
    artifact_id: str
    role: str
    media_type: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _safe_id(self.artifact_id, "artifact ID")
        _safe_id(self.role, "artifact role")
        if (
            not isinstance(self.media_type, str)
            or not self.media_type
            or len(self.media_type) > 127
            or any(char.isspace() for char in self.media_type)
        ):
            raise ArtifactTransportError("artifact media type is invalid")
        if (
            not isinstance(self.byte_count, int)
            or isinstance(self.byte_count, bool)
            or self.byte_count < 0
            or self.byte_count > MAX_ARTIFACT_BYTES
        ):
            raise ArtifactTransportError("artifact byte count is invalid")
        _sha(self.sha256, "artifact digest")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactDescriptor":
        if not isinstance(value, Mapping) or set(value) != _DESCRIPTOR_FIELDS:
            raise ArtifactTransportError("artifact descriptor is not a closed schema")
        count = value["byte_count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0 or count > MAX_ARTIFACT_BYTES:
            raise ArtifactTransportError("artifact byte count is invalid")
        media_type = value["media_type"]
        if not isinstance(media_type, str) or not media_type or len(media_type) > 127 or any(c.isspace() for c in media_type):
            raise ArtifactTransportError("artifact media type is invalid")
        return cls(
            _safe_id(value["artifact_id"], "artifact ID"),
            _safe_id(value["role"], "artifact role"),
            media_type,
            count,
            _sha(value["sha256"], "artifact digest"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "role": self.role,
            "media_type": self.media_type,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TransferManifest:
    campaign_id: str
    transfer_id: str
    source_role: str
    destination_role: str
    artifacts: Tuple[ArtifactDescriptor, ...]
    schema_version: str = TRANSFER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TRANSFER_SCHEMA:
            raise ArtifactTransportError("unsupported transfer manifest schema")
        _safe_id(self.campaign_id, "campaign ID")
        _safe_id(self.transfer_id, "transfer ID")
        if self.source_role not in {"TRAINER", "EVALUATOR"} or self.destination_role not in {
            "TRAINER", "EVALUATOR"
        }:
            raise ArtifactTransportError("transfer endpoint role is outside the closed vocabulary")
        if self.source_role == self.destination_role:
            raise ArtifactTransportError("artifact transfer must cross an authority boundary")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise ArtifactTransportError("transfer manifest has no artifacts")
        if any(type(item) is not ArtifactDescriptor for item in self.artifacts):
            raise ArtifactTransportError("transfer manifest contains an unvalidated artifact")
        identifiers = [item.artifact_id for item in self.artifacts]
        digests = [item.sha256 for item in self.artifacts]
        if len(identifiers) != len(set(identifiers)) or len(digests) != len(set(digests)):
            raise ArtifactTransportError("transfer artifacts are duplicated or ambiguous")
        if identifiers != sorted(identifiers):
            raise ArtifactTransportError("transfer artifacts must use canonical ID order")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TransferManifest":
        if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
            raise ArtifactTransportError("transfer manifest is not a closed schema")
        if value["schema_version"] != TRANSFER_SCHEMA:
            raise ArtifactTransportError("unsupported transfer manifest schema")
        if value["source_role"] not in {"TRAINER", "EVALUATOR"} or value["destination_role"] not in {
            "TRAINER", "EVALUATOR"
        }:
            raise ArtifactTransportError("transfer endpoint role is outside the closed vocabulary")
        if value["source_role"] == value["destination_role"]:
            raise ArtifactTransportError("artifact transfer must cross an authority boundary")
        if not isinstance(value["artifacts"], list) or not value["artifacts"]:
            raise ArtifactTransportError("transfer manifest has no artifacts")
        artifacts = tuple(ArtifactDescriptor.from_mapping(item) for item in value["artifacts"])
        identifiers = [item.artifact_id for item in artifacts]
        digests = [item.sha256 for item in artifacts]
        if len(identifiers) != len(set(identifiers)) or len(digests) != len(set(digests)):
            raise ArtifactTransportError("transfer artifacts are duplicated or ambiguous")
        if identifiers != sorted(identifiers):
            raise ArtifactTransportError("transfer artifacts must use canonical ID order")
        return cls(
            _safe_id(value["campaign_id"], "campaign ID"),
            _safe_id(value["transfer_id"], "transfer ID"),
            value["source_role"],
            value["destination_role"],
            artifacts,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "transfer_id": self.transfer_id,
            "source_role": self.source_role,
            "destination_role": self.destination_role,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())

    @property
    def artifact_set_digest(self) -> str:
        return digest_for([item.to_dict() for item in self.artifacts])


class ArtifactStagingStore:
    """Receive bytes under digest names and seal an exact canonical manifest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ArtifactTransportError("artifact staging root must be a regular directory")
        self.lock_path = self.root / ".staging.lock"

    def _transfer_root(self, manifest: TransferManifest) -> Path:
        return self.root / manifest.digest

    def _blob_path(self, root: Path, descriptor: ArtifactDescriptor) -> Path:
        return root / "blobs" / (descriptor.sha256 + ".blob")

    def stage(self, manifest: TransferManifest, payloads: Mapping[str, bytes]) -> str:
        if type(manifest) is not TransferManifest:
            raise ArtifactTransportError("staging requires an exact validated manifest")
        if not isinstance(payloads, Mapping) or set(payloads) != {item.artifact_id for item in manifest.artifacts}:
            raise ArtifactTransportError("payload set does not exactly match the manifest")
        with _exclusive_path_lock(self.lock_path):
            destination = self._transfer_root(manifest)
            if destination.exists():
                self._verify_unlocked(manifest)
                for item in manifest.artifacts:
                    payload = payloads[item.artifact_id]
                    self._verify_payload(item, payload)
                return manifest.digest
            temporary = Path(tempfile.mkdtemp(prefix=".transfer-", dir=str(self.root)))
            try:
                blobs = temporary / "blobs"
                blobs.mkdir()
                for item in manifest.artifacts:
                    payload = payloads[item.artifact_id]
                    self._verify_payload(item, payload)
                    self._write_file(self._blob_path(temporary, item), payload)
                self._write_file(temporary / "manifest.json", canonical_bytes(manifest.to_dict()))
                _durable_replace(temporary, destination)
            finally:
                if temporary.exists():
                    for child in sorted(temporary.rglob("*"), reverse=True):
                        if child.is_file() or child.is_symlink():
                            child.unlink()
                        elif child.is_dir():
                            child.rmdir()
                    temporary.rmdir()
            self._verify_unlocked(manifest)
            return manifest.digest

    @staticmethod
    def _write_file(path: Path, payload: bytes) -> None:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _verify_payload(descriptor: ArtifactDescriptor, payload: Any) -> None:
        if not isinstance(payload, bytes):
            raise ArtifactTransportError("artifact payload must be bytes")
        if len(payload) != descriptor.byte_count or digest_bytes(payload) != descriptor.sha256:
            raise ArtifactTransportError("artifact payload size or digest differs from manifest")

    def verify(self, manifest: TransferManifest, *, expected_digest: Optional[str] = None) -> str:
        with _exclusive_path_lock(self.lock_path):
            return self._verify_unlocked(manifest, expected_digest=expected_digest)

    def payload_bytes(self, manifest: TransferManifest, artifact_id: str) -> bytes:
        """Return one admitted payload only after re-verifying the complete transfer."""

        with _exclusive_path_lock(self.lock_path):
            self._verify_unlocked(manifest)
            matches = [item for item in manifest.artifacts if item.artifact_id == artifact_id]
            if len(matches) != 1:
                raise ArtifactTransportError("artifact identity is absent or ambiguous")
            payload = self._blob_path(self._transfer_root(manifest), matches[0]).read_bytes()
            self._verify_payload(matches[0], payload)
            return payload

    def _verify_unlocked(self, manifest: TransferManifest, *, expected_digest: Optional[str] = None) -> str:
        if type(manifest) is not TransferManifest:
            raise ArtifactTransportError("verification requires an exact validated manifest")
        if expected_digest is not None and manifest.digest != _sha(expected_digest, "expected transfer digest"):
            raise ArtifactTransportError("transfer manifest substitution detected")
        root = self._transfer_root(manifest)
        if root.is_symlink() or not root.is_dir():
            raise ArtifactTransportError("staged transfer is absent or unsafe")
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ArtifactTransportError("staged manifest is absent or unsafe")
        raw = manifest_path.read_bytes()
        if raw != canonical_bytes(manifest.to_dict()):
            raise ArtifactTransportError("staged manifest differs from expected canonical manifest")
        expected_files = {manifest_path}
        for item in manifest.artifacts:
            path = self._blob_path(root, item)
            expected_files.add(path)
            if path.is_symlink() or not path.is_file():
                raise ArtifactTransportError("staged artifact is absent or unsafe")
            stat_result = path.stat()
            if stat_result.st_size != item.byte_count or self._file_digest(path) != item.sha256:
                raise ArtifactTransportError("staged artifact was truncated or substituted")
        actual_files = {path for path in root.rglob("*") if path.is_file() or path.is_symlink()}
        if actual_files != expected_files:
            raise ArtifactTransportError("staged transfer contains unexpected files")
        return manifest.digest

    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()


__all__ = [
    "ArtifactDescriptor", "ArtifactStagingStore", "ArtifactTransportError", "TRANSFER_SCHEMA",
    "TransferManifest",
]
