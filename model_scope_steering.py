"""Shadow-mode steering evaluation for Model-Scope artifacts."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from model_scope_features import SparseFeatureEvent
from model_scope_runtime import ActivationEvent
from steering_policy import ModelScopeSteeringPolicy, PolicyRegistry, PolicyStatus, SteeringMode


class ModelScopeShadowSteerer:
    """Evaluate Model-Scope feature summaries and emit advisory steering actions."""

    def __init__(self, policy: ModelScopeSteeringPolicy | Mapping[str, Any]):
        if isinstance(policy, ModelScopeSteeringPolicy):
            self.policy = policy
        else:
            self.policy = ModelScopeSteeringPolicy.from_mapping(policy)

    def evaluate_capture(self, artifact: Mapping[str, Any]) -> Dict[str, Any]:
        observations = artifact.get("capture", {}).get("observations", [])
        matched_rules: List[Dict[str, Any]] = []

        for observation in observations:
            feature_summary = observation.get("feature_summary")
            if not isinstance(feature_summary, Mapping):
                continue
            if (
                self.policy.feature_space is not None
                and str(feature_summary.get("feature_space")) != self.policy.feature_space
            ):
                continue
            features = {
                str(item.get("feature_id")): float(item.get("value", 0.0))
                for item in feature_summary.get("active_features", [])
            }
            for rule in self.policy.rules:
                if rule.layer_index is not None and int(observation.get("layer_index", -1)) != rule.layer_index:
                    continue
                value = features.get(rule.feature_id)
                if value is None or value < rule.min_value:
                    continue
                matched_rules.append(
                    {
                        "layer_index": int(observation.get("layer_index", -1)),
                        "feature_id": rule.feature_id,
                        "observed_value": float(value),
                        "min_value": float(rule.min_value),
                        "action_type": rule.action_type,
                        "strength": float(rule.strength),
                        "description": rule.description,
                    }
                )

        return {
            "policy_name": self.policy.name,
            "deployment_mode": self.policy.deployment_mode,
            "feature_space": self.policy.feature_space,
            "matched_rule_count": len(matched_rules),
            "recommended_actions": matched_rules,
            "runtime_applied": False,
            "provenance": {
                "policy_name": self.policy.name,
                "deployment_mode": self.policy.deployment_mode,
                "artifact_path": artifact.get("output_path"),
            },
            "rollback": {
                "off_switch": True,
                "base_weights_mutated": False,
                "disable_by_setting_deployment_mode": "disabled",
            },
            "safety_regression_required": bool(matched_rules),
        }


# ---------------------------------------------------------------------------
# Phase 3 — Steering Actuator Layer
# ---------------------------------------------------------------------------


@dataclass
class InterventionRecord:
    """Full provenance record for one policy application (or decline)."""

    schema_version: str = "1.0"
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_id: str = ""
    policy_name: str = ""
    mode: SteeringMode = SteeringMode.SHADOW
    applied: bool = False
    features_targeted: List[str] = field(default_factory=list)
    features_modified: List[str] = field(default_factory=list)
    original_values: Dict[str, float] = field(default_factory=dict)
    modified_values: Dict[str, float] = field(default_factory=dict)
    scale_factor_used: float = 1.0
    decline_reason: Optional[str] = None
    intervention_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str = ""
    layer_id: str = ""
    model_id: str = ""


def _copy_feature_event(event: SparseFeatureEvent) -> SparseFeatureEvent:
    return SparseFeatureEvent(
        schema_version=event.schema_version,
        source_activation=event.source_activation,
        feature_source=event.feature_source,
        features=dict(event.features),
        feature_count=event.feature_count,
        nonzero_count=event.nonzero_count,
        extracted_at=event.extracted_at,
    )


def _declined_record(
    config: Any,
    targeted: List[str],
    original_values: Dict[str, float],
    activation: ActivationEvent,
    reason: str,
) -> InterventionRecord:
    return InterventionRecord(
        policy_id=config.policy_id,
        policy_name=config.name,
        mode=config.mode,
        applied=False,
        features_targeted=targeted,
        features_modified=[],
        original_values=original_values,
        modified_values=dict(original_values),
        scale_factor_used=config.scale_factor,
        decline_reason=reason,
        run_id=activation.run_id,
        layer_id=activation.layer_id,
        model_id=activation.model_id,
    )


class SteeringActuator:
    """Applies steering policies to SparseFeatureEvent objects with full provenance."""

    def __init__(self, registry: PolicyRegistry, *, max_total_interventions: int = 0) -> None:
        self._registry = registry
        self._max_total = max_total_interventions
        self._records: List[InterventionRecord] = []
        self._policy_counts: Dict[str, int] = {}

    def apply(
        self,
        feature_event: SparseFeatureEvent,
        policy_id: str,
    ) -> Any:
        """Apply (or shadow-record) a policy; always returns a new SparseFeatureEvent."""
        config = self._registry.get(policy_id)  # raises KeyError if unknown
        activation = feature_event.source_activation
        targeted = [k for k in config.target_features if k in feature_event.features]
        original_values = {k: feature_event.features[k] for k in targeted}

        # Global cap check
        if self._max_total > 0 and self.total_applied() >= self._max_total:
            record = _declined_record(config, targeted, original_values, activation, "max_total_interventions_exceeded")
            self._records.append(record)
            return _copy_feature_event(feature_event), record

        # Policy status checks
        if config.status == PolicyStatus.DISABLED:
            record = _declined_record(config, targeted, original_values, activation, "policy_disabled")
            self._records.append(record)
            return _copy_feature_event(feature_event), record

        if config.status == PolicyStatus.BLOCKED:
            record = _declined_record(config, targeted, original_values, activation, "policy_blocked")
            self._records.append(record)
            return _copy_feature_event(feature_event), record

        # Per-policy cap check
        policy_applied = self._policy_counts.get(policy_id, 0)
        if config.max_interventions > 0 and policy_applied >= config.max_interventions:
            record = _declined_record(config, targeted, original_values, activation, "policy_max_interventions_exceeded")
            self._records.append(record)
            return _copy_feature_event(feature_event), record

        # SHADOW mode — observe only
        if config.mode == SteeringMode.SHADOW:
            record = InterventionRecord(
                policy_id=config.policy_id,
                policy_name=config.name,
                mode=config.mode,
                applied=False,
                features_targeted=targeted,
                features_modified=[],
                original_values=original_values,
                modified_values=dict(original_values),
                scale_factor_used=config.scale_factor,
                decline_reason=None,
                run_id=activation.run_id,
                layer_id=activation.layer_id,
                model_id=activation.model_id,
            )
            self._records.append(record)
            return _copy_feature_event(feature_event), record

        # SOFT_SCALE or SUPPRESSION
        new_features = dict(feature_event.features)
        if config.mode == SteeringMode.SOFT_SCALE:
            for k in targeted:
                new_features[k] = new_features[k] * config.scale_factor
        elif config.mode == SteeringMode.SUPPRESSION:
            for k in targeted:
                new_features[k] = 0.0

        modified_values = {k: new_features[k] for k in targeted}
        features_modified = [k for k in targeted if original_values.get(k, 0.0) != modified_values.get(k, 0.0)]

        new_event = SparseFeatureEvent(
            schema_version=feature_event.schema_version,
            source_activation=feature_event.source_activation,
            feature_source=feature_event.feature_source,
            features=new_features,
            feature_count=feature_event.feature_count,
            nonzero_count=sum(1 for v in new_features.values() if v != 0.0),
            extracted_at=feature_event.extracted_at,
        )
        record = InterventionRecord(
            policy_id=config.policy_id,
            policy_name=config.name,
            mode=config.mode,
            applied=True,
            features_targeted=targeted,
            features_modified=features_modified,
            original_values=original_values,
            modified_values=modified_values,
            scale_factor_used=config.scale_factor,
            decline_reason=None,
            run_id=activation.run_id,
            layer_id=activation.layer_id,
            model_id=activation.model_id,
        )
        self._records.append(record)
        self._policy_counts[policy_id] = policy_applied + 1
        return new_event, record

    def apply_all_active(
        self,
        feature_event: SparseFeatureEvent,
    ) -> Any:
        """Apply all ACTIVE policies in sequence; each receives the prior output."""
        active = self._registry.list_active()
        records: List[InterventionRecord] = []
        current = feature_event
        for config in active:
            current, record = self.apply(current, config.policy_id)
            records.append(record)
        return current, records

    def get_records(self) -> List[InterventionRecord]:
        """Return all accumulated intervention records."""
        return list(self._records)

    def clear_records(self) -> None:
        """Reset the accumulated record list."""
        self._records = []

    def total_applied(self) -> int:
        """Count of records where applied=True."""
        return sum(1 for r in self._records if r.applied)

    def total_shadow(self) -> int:
        """Count of records where applied=False and decline_reason is None."""
        return sum(1 for r in self._records if not r.applied and r.decline_reason is None)


def rollback_feature_event(record: InterventionRecord, feature_event: SparseFeatureEvent) -> SparseFeatureEvent:
    """Restore original_values from an InterventionRecord into a copy of feature_event.

    Raises ValueError if record.applied is False (nothing to roll back).
    """
    if not record.applied:
        raise ValueError("Cannot roll back an unapplied intervention record.")
    new_features = dict(feature_event.features)
    new_features.update(record.original_values)
    return SparseFeatureEvent(
        schema_version=feature_event.schema_version,
        source_activation=feature_event.source_activation,
        feature_source=feature_event.feature_source,
        features=new_features,
        feature_count=feature_event.feature_count,
        nonzero_count=sum(1 for v in new_features.values() if v != 0.0),
        extracted_at=feature_event.extracted_at,
    )


def intervention_record_to_dict(record: InterventionRecord) -> Dict[str, Any]:
    """Serialise an InterventionRecord to a JSON-compatible dict."""
    return {
        "schema_version": record.schema_version,
        "record_id": record.record_id,
        "policy_id": record.policy_id,
        "policy_name": record.policy_name,
        "mode": record.mode.value,
        "applied": record.applied,
        "features_targeted": list(record.features_targeted),
        "features_modified": list(record.features_modified),
        "original_values": dict(record.original_values),
        "modified_values": dict(record.modified_values),
        "scale_factor_used": record.scale_factor_used,
        "decline_reason": record.decline_reason,
        "intervention_at": record.intervention_at,
        "run_id": record.run_id,
        "layer_id": record.layer_id,
        "model_id": record.model_id,
    }


def intervention_record_from_dict(data: Dict[str, Any]) -> InterventionRecord:
    """Deserialise an InterventionRecord from a dict produced by intervention_record_to_dict()."""
    return InterventionRecord(
        schema_version=data["schema_version"],
        record_id=data["record_id"],
        policy_id=data["policy_id"],
        policy_name=data["policy_name"],
        mode=SteeringMode(data["mode"]),
        applied=data["applied"],
        features_targeted=list(data["features_targeted"]),
        features_modified=list(data["features_modified"]),
        original_values=dict(data["original_values"]),
        modified_values=dict(data["modified_values"]),
        scale_factor_used=float(data["scale_factor_used"]),
        decline_reason=data["decline_reason"],
        intervention_at=data["intervention_at"],
        run_id=data["run_id"],
        layer_id=data["layer_id"],
        model_id=data["model_id"],
    )
