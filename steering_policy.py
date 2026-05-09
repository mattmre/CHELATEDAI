"""Fail-closed policy contracts for Model-Scope steering."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Mapping


@dataclass
class SteeringRule:
    """One feature-triggered steering recommendation."""

    feature_id: str
    min_value: float = 0.0
    layer_index: int | None = None
    action_type: str = "suppress"
    strength: float = 1.0
    description: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]):
        if "feature_id" not in payload:
            raise ValueError("steering rule requires feature_id")
        layer_index = payload.get("layer_index")
        return cls(
            feature_id=str(payload["feature_id"]),
            min_value=float(payload.get("min_value", 0.0)),
            layer_index=None if layer_index is None else int(layer_index),
            action_type=str(payload.get("action_type", "suppress")),
            strength=float(payload.get("strength", 1.0)),
            description=str(payload.get("description", "")),
        )


@dataclass
class ModelScopeSteeringPolicy:
    """Policy definition for shadow-mode feature steering."""

    name: str = "model_scope_shadow_policy"
    deployment_mode: str = "shadow_mode"
    feature_space: str | None = None
    rules: List[SteeringRule] = field(default_factory=list)

    def activate(self, mode: str = "soft_scale") -> None:
        """Switch out of shadow_mode. mode must be 'soft_scale', 'suppression', or 'active'."""
        valid = {"soft_scale", "suppression", "active"}
        if mode not in valid:
            raise ValueError(f"mode must be one of {valid}, got {mode!r}")
        self.deployment_mode = mode

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]):
        rules = [SteeringRule.from_mapping(item) for item in payload.get("rules", [])]
        return cls(
            name=str(payload.get("name", "model_scope_shadow_policy")),
            deployment_mode=str(payload.get("deployment_mode", "shadow_mode")),
            feature_space=None if payload.get("feature_space") is None else str(payload.get("feature_space")),
            rules=rules,
        )


class SteeringMode(str, Enum):
    """How a steering intervention modifies (or observes) feature activations."""

    SHADOW = "SHADOW"
    SOFT_SCALE = "SOFT_SCALE"
    SUPPRESSION = "SUPPRESSION"


class PolicyStatus(str, Enum):
    """Lifecycle status of a registered steering policy."""

    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    BLOCKED = "BLOCKED"


@dataclass
class SteeringPolicyConfig:
    """Configuration for a single steering policy."""

    name: str
    mode: SteeringMode
    target_features: List[str]
    schema_version: str = "1.0"
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scale_factor: float = 1.0
    status: PolicyStatus = field(default=PolicyStatus.ACTIVE)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""
    max_interventions: int = 0

    def __post_init__(self) -> None:
        self.scale_factor = max(0.0, min(2.0, float(self.scale_factor)))
        if isinstance(self.mode, str):
            self.mode = SteeringMode(self.mode)
        if isinstance(self.status, str):
            self.status = PolicyStatus(self.status)


class PolicyRegistry:
    """Registry for steering policies with lifecycle management."""

    def __init__(self) -> None:
        self._policies: dict = {}
        self._block_reasons: dict = {}

    def register(self, config: SteeringPolicyConfig) -> str:
        """Store policy and return its policy_id."""
        self._policies[config.policy_id] = config
        return config.policy_id

    def get(self, policy_id: str) -> SteeringPolicyConfig:
        """Retrieve policy by ID; raises KeyError if not found."""
        if policy_id not in self._policies:
            raise KeyError(policy_id)
        return self._policies[policy_id]

    def list_active(self) -> List[SteeringPolicyConfig]:
        """Return only policies with ACTIVE status."""
        return [p for p in self._policies.values() if p.status == PolicyStatus.ACTIVE]

    def disable(self, policy_id: str) -> bool:
        """Set status to DISABLED; returns True if found."""
        if policy_id not in self._policies:
            return False
        self._policies[policy_id].status = PolicyStatus.DISABLED
        return True

    def block(self, policy_id: str, reason: str) -> bool:
        """Set status to BLOCKED and store reason; returns True if found."""
        if policy_id not in self._policies:
            return False
        self._policies[policy_id].status = PolicyStatus.BLOCKED
        self._block_reasons[policy_id] = reason
        return True

    def get_block_reason(self, policy_id: str) -> Any:
        """Return the block reason string, or None if not blocked / not found."""
        return self._block_reasons.get(policy_id)

    def count(self) -> int:
        """Total number of registered policies regardless of status."""
        return len(self._policies)

    def to_dict(self) -> dict:
        """Serialise registry state to a JSON-compatible dict."""
        return {
            "policies": {
                pid: {
                    "schema_version": p.schema_version,
                    "policy_id": p.policy_id,
                    "name": p.name,
                    "mode": p.mode.value,
                    "target_features": list(p.target_features),
                    "scale_factor": p.scale_factor,
                    "status": p.status.value,
                    "created_at": p.created_at,
                    "description": p.description,
                    "max_interventions": p.max_interventions,
                }
                for pid, p in self._policies.items()
            },
            "block_reasons": dict(self._block_reasons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PolicyRegistry:
        """Deserialise a PolicyRegistry from a dict produced by to_dict()."""
        registry = cls()
        for pid, pd in data.get("policies", {}).items():
            config = SteeringPolicyConfig(
                schema_version=pd["schema_version"],
                policy_id=pd["policy_id"],
                name=pd["name"],
                mode=SteeringMode(pd["mode"]),
                target_features=list(pd["target_features"]),
                scale_factor=pd["scale_factor"],
                status=PolicyStatus(pd["status"]),
                created_at=pd["created_at"],
                description=pd["description"],
                max_interventions=pd["max_interventions"],
            )
            registry._policies[pid] = config
        registry._block_reasons = dict(data.get("block_reasons", {}))
        return registry
