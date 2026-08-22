"""Protected, FD-only DeepSeek restore inventory contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import stat
from typing import Any, Dict, Mapping, Tuple

from ..canonical import canonical_bytes, digest_for, validate_sha256
from .errors import ProtectedInventoryError


INVENTORY_SCHEMA = "egv-protected-restore-inventory-v1"
MAX_INVENTORY_BYTES = 1024 * 1024
COMPOSE_FILE = "docker-compose.dspark.yml"
ENV_FILE = ".env.dspark"
_INVENTORY_FIELDS = frozenset({
    "schema_version", "inventory_id", "compose_project", "compose_file", "env_file",
    "services", "dependency_order", "expected_service_count", "resource_baseline_digest",
})
_SERVICE_FIELDS = frozenset({
    "logical_id", "compose_service", "container_identity", "image_digest", "model_digest",
    "configuration_digest", "executable_digest", "health", "smoke",
})
_HEALTH_FIELDS = frozenset({"method", "path", "expected_status", "response_digest"})
_SMOKE_FIELDS = frozenset({"input_digest", "output_digest"})
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SECRET_KEY = re.compile(r"(?:password|passwd|secret|token|credential|private.?key|api.?key|authorization|cookie)", re.I)
_SECRET_VALUE = re.compile(r"(?:BEGIN [A-Z ]*PRIVATE KEY|(?:password|passwd|token|secret|api[_-]?key)\s*[:=]|://[^/@\s]+:[^/@\s]+@)", re.I)


def _digest(value: Any, field: str) -> str:
    try:
        return validate_sha256(value, field)
    except Exception as exc:
        raise ProtectedInventoryError(str(exc)) from exc


def _reject_secrets(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or _SECRET_KEY.search(key):
                raise ProtectedInventoryError("protected inventory contains a secret-bearing field")
            if key.lower() in {"command", "argv", "args", "environment", "env", "sudo", "ssh"}:
                raise ProtectedInventoryError("protected inventory may not supply command-line or environment values")
            _reject_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}[{index}]")
    elif isinstance(value, str) and _SECRET_VALUE.search(value):
        raise ProtectedInventoryError("protected inventory contains a secret-like value")


@dataclass(frozen=True)
class HealthContract:
    method: str
    path: str
    expected_status: int
    response_digest: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HealthContract":
        if not isinstance(value, Mapping) or set(value) != _HEALTH_FIELDS:
            raise ProtectedInventoryError("health contract is not closed")
        if value["method"] != "GET" or not isinstance(value["path"], str) or not value["path"].startswith("/") or "?" in value["path"]:
            raise ProtectedInventoryError("health contract method/path is unsafe")
        status = value["expected_status"]
        if not isinstance(status, int) or isinstance(status, bool) or status < 100 or status > 599:
            raise ProtectedInventoryError("health status is invalid")
        return cls("GET", value["path"], status, _digest(value["response_digest"], "health response digest"))

    def to_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "path": self.path, "expected_status": self.expected_status, "response_digest": self.response_digest}


@dataclass(frozen=True)
class SmokeContract:
    input_digest: str
    output_digest: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SmokeContract":
        if not isinstance(value, Mapping) or set(value) != _SMOKE_FIELDS:
            raise ProtectedInventoryError("smoke contract is not closed")
        return cls(_digest(value["input_digest"], "smoke input digest"), _digest(value["output_digest"], "smoke output digest"))

    def to_dict(self) -> Dict[str, str]:
        return {"input_digest": self.input_digest, "output_digest": self.output_digest}


@dataclass(frozen=True)
class ServiceInventory:
    logical_id: str
    compose_service: str
    container_identity: str
    image_digest: str
    model_digest: str
    configuration_digest: str
    executable_digest: str
    health: HealthContract
    smoke: SmokeContract

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ServiceInventory":
        if not isinstance(value, Mapping) or set(value) != _SERVICE_FIELDS:
            raise ProtectedInventoryError("service inventory is not closed")
        for field in ("logical_id", "compose_service", "container_identity"):
            if not isinstance(value[field], str) or not _SAFE_NAME.fullmatch(value[field]):
                raise ProtectedInventoryError(f"service {field} is unsafe or ambiguous")
        return cls(
            value["logical_id"], value["compose_service"], value["container_identity"],
            _digest(value["image_digest"], "image digest"), _digest(value["model_digest"], "model digest"),
            _digest(value["configuration_digest"], "configuration digest"),
            _digest(value["executable_digest"], "executable digest"),
            HealthContract.from_mapping(value["health"]), SmokeContract.from_mapping(value["smoke"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_id": self.logical_id, "compose_service": self.compose_service,
            "container_identity": self.container_identity, "image_digest": self.image_digest,
            "model_digest": self.model_digest, "configuration_digest": self.configuration_digest,
            "executable_digest": self.executable_digest, "health": self.health.to_dict(),
            "smoke": self.smoke.to_dict(),
        }


@dataclass(frozen=True)
class ProtectedRestoreInventory:
    inventory_id: str
    compose_project: str
    services: Tuple[ServiceInventory, ...]
    dependency_order: Tuple[str, ...]
    expected_service_count: int
    resource_baseline_digest: str
    compose_file: str = COMPOSE_FILE
    env_file: str = ENV_FILE
    schema_version: str = INVENTORY_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProtectedRestoreInventory":
        if not isinstance(value, Mapping) or set(value) != _INVENTORY_FIELDS:
            raise ProtectedInventoryError("protected restore inventory is not a closed schema")
        _reject_secrets(value)
        if value["schema_version"] != INVENTORY_SCHEMA or value["compose_file"] != COMPOSE_FILE or value["env_file"] != ENV_FILE:
            raise ProtectedInventoryError("inventory does not use the exact compose/env-file contract")
        if not isinstance(value["inventory_id"], str) or not _SAFE_NAME.fullmatch(value["inventory_id"]):
            raise ProtectedInventoryError("inventory ID is unsafe")
        if not isinstance(value["compose_project"], str) or not _SAFE_NAME.fullmatch(value["compose_project"]):
            raise ProtectedInventoryError("compose project is unsafe")
        if not isinstance(value["services"], list) or not value["services"]:
            raise ProtectedInventoryError("inventory has no services")
        services = tuple(ServiceInventory.from_mapping(item) for item in value["services"])
        logical_ids = [service.logical_id for service in services]
        compose_names = [service.compose_service for service in services]
        identities = [service.container_identity for service in services]
        if any(len(items) != len(set(items)) for items in (logical_ids, compose_names, identities)):
            raise ProtectedInventoryError("inventory service identities are ambiguous")
        order = value["dependency_order"]
        if not isinstance(order, list) or len(order) != len(set(order)) or set(order) != set(logical_ids):
            raise ProtectedInventoryError("dependency order must enumerate every service exactly once")
        count = value["expected_service_count"]
        if not isinstance(count, int) or isinstance(count, bool) or count != len(services):
            raise ProtectedInventoryError("expected service count differs from inventory")
        return cls(
            value["inventory_id"], value["compose_project"], services, tuple(order), count,
            _digest(value["resource_baseline_digest"], "resource baseline digest"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version, "inventory_id": self.inventory_id,
            "compose_project": self.compose_project, "compose_file": self.compose_file,
            "env_file": self.env_file, "services": [service.to_dict() for service in self.services],
            "dependency_order": list(self.dependency_order), "expected_service_count": self.expected_service_count,
            "resource_baseline_digest": self.resource_baseline_digest,
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())

    def service(self, logical_id: str) -> ServiceInventory:
        matches = [service for service in self.services if service.logical_id == logical_id]
        if len(matches) != 1:
            raise ProtectedInventoryError("service identity is absent or ambiguous")
        return matches[0]


def load_protected_inventory_fd(fd: int) -> ProtectedRestoreInventory:
    """Read only from an already-open, regular descriptor; never accept a path."""

    if not isinstance(fd, int) or isinstance(fd, bool) or fd < 0:
        raise ProtectedInventoryError("protected inventory requires an already-open file descriptor")
    try:
        info = os.fstat(fd)
    except OSError as exc:
        raise ProtectedInventoryError("protected inventory descriptor is invalid") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ProtectedInventoryError("protected inventory descriptor must reference a regular file")
    try:
        with os.fdopen(os.dup(fd), "rb") as handle:
            raw = handle.read(MAX_INVENTORY_BYTES + 1)
    except OSError as exc:
        raise ProtectedInventoryError("protected inventory descriptor cannot be read") from exc
    if not raw or len(raw) > MAX_INVENTORY_BYTES:
        raise ProtectedInventoryError("protected inventory is empty or oversized")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ProtectedInventoryError("protected inventory is not valid UTF-8 JSON") from exc
    inventory = ProtectedRestoreInventory.from_mapping(value)
    if raw != canonical_bytes(inventory.to_dict()):
        raise ProtectedInventoryError("protected inventory is not canonical JSON")
    return inventory


__all__ = [
    "COMPOSE_FILE", "ENV_FILE", "HealthContract", "INVENTORY_SCHEMA", "ProtectedRestoreInventory",
    "ServiceInventory", "SmokeContract", "load_protected_inventory_fd",
]
