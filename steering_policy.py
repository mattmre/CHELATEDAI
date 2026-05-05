"""Fail-closed policy contracts for Model-Scope steering."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]):
        rules = [SteeringRule.from_mapping(item) for item in payload.get("rules", [])]
        return cls(
            name=str(payload.get("name", "model_scope_shadow_policy")),
            deployment_mode=str(payload.get("deployment_mode", "shadow_mode")),
            feature_space=None if payload.get("feature_space") is None else str(payload.get("feature_space")),
            rules=rules,
        )
