"""Shadow-mode steering evaluation for Model-Scope artifacts."""

from __future__ import annotations

import json as _json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from model_scope_features import SparseFeatureEvent
from model_scope_runtime import ActivationEvent
from steering_policy import ModelScopeSteeringPolicy, PolicyRegistry, PolicyStatus, SteeringMode, SteeringPolicyConfig


_STEERER_MODE_MAP = {
    "shadow_mode": SteeringMode.SHADOW,
    "soft_scale_mode": SteeringMode.SOFT_SCALE,
    "suppression_mode": SteeringMode.SUPPRESSION,
}


class ModelScopeShadowSteerer:
    """Evaluate Model-Scope feature summaries and emit advisory steering actions."""

    def __init__(self, policy: ModelScopeSteeringPolicy | Mapping[str, Any] | None = None):
        if policy is None:
            policy = ModelScopeSteeringPolicy()
        if isinstance(policy, ModelScopeSteeringPolicy):
            self._policy = policy
        else:
            self._policy = ModelScopeSteeringPolicy.from_mapping(policy)
        # Public alias for backward compatibility
        self.policy = self._policy

        # Internal actuator bridge (System B)
        self._policy_config = SteeringPolicyConfig(
            name="shadow_steerer_internal",
            mode=SteeringMode.SHADOW,
            target_features=[],
            policy_id="shadow_steerer_internal",
        )
        self._registry = PolicyRegistry()
        self._registry.register(self._policy_config)
        self._actuator = SteeringActuator(self._registry)

    def activate(self, mode_string: str) -> None:
        """Activate deployment mode. Syncs both the policy string (System A) and actuator enum (System B)."""
        self._policy.deployment_mode = mode_string
        self._policy_config.mode = _STEERER_MODE_MAP.get(mode_string, SteeringMode.SHADOW)

    def evaluate_capture(self, artifact: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        if artifact is None:
            artifact = {}
        observations = artifact.get("capture", {}).get("observations", [])
        matched_rules: List[Dict[str, Any]] = []

        for observation in observations:
            feature_summary = observation.get("feature_summary")
            if not isinstance(feature_summary, Mapping):
                continue
            if (
                self._policy.feature_space is not None
                and str(feature_summary.get("feature_space")) != self._policy.feature_space
            ):
                continue
            features = {
                str(item.get("feature_id")): float(item.get("value", 0.0))
                for item in feature_summary.get("active_features", [])
            }
            for rule in self._policy.rules:
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

        # Build feature dict and target list from matched rules so the actuator
        # operates on real matched feature values instead of an empty event.
        matched_feature_ids = [r["feature_id"] for r in matched_rules]
        matched_features = {r["feature_id"]: r["observed_value"] for r in matched_rules}
        # Update policy config in-place so the actuator sees the current target list.
        self._policy_config.target_features = matched_feature_ids

        activation = ActivationEvent(
            schema_version="1.0",
            model_id=getattr(self._policy, "model_id", "unknown"),
            layer_id="evaluate_capture",
            token_count=len(matched_features),
            shape=(len(matched_features),),
            mean_activation=float(sum(matched_features.values()) / len(matched_features)) if matched_features else 0.0,
            norm_activation=0.0,
            captured_at=datetime.now(timezone.utc).isoformat(),
            run_id="evaluate_capture",
        )
        feature_event = SparseFeatureEvent(
            schema_version="1.0",
            source_activation=activation,
            feature_source="evaluate_capture",
            features=matched_features,
            feature_count=len(matched_features),
            nonzero_count=sum(1 for v in matched_features.values() if v != 0.0),
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )
        _, record = self._actuator.apply(feature_event, self._policy_config.policy_id)
        runtime_applied = record.applied
        # In SHADOW mode record.applied=False; in SOFT_SCALE/SUPPRESSION record.applied=True.
        # applied_actions reports which features the actuator targeted (not just modified),
        # so callers can see which matched rules drove an actual intervention.
        applied_actions = list(record.features_targeted) if record.applied else []

        return {
            "policy_name": self._policy.name,
            "deployment_mode": self._policy.deployment_mode,
            "feature_space": self._policy.feature_space,
            "matched_rule_count": len(matched_rules),
            "recommended_actions": matched_rules,
            "runtime_applied": runtime_applied,
            "mode": self._policy_config.mode.value,
            "rules_matched": len(matched_rules),
            "applied_actions": applied_actions,
            "provenance": {
                "policy_name": self._policy.name,
                "deployment_mode": self._policy.deployment_mode,
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

    def persist_records(self, path: str | Path) -> int:
        """Write all accumulated InterventionRecords to a JSON Lines file.

        Each line is a self-contained JSON object produced by
        ``intervention_record_to_dict()``.  Existing file content is
        overwritten.  Returns the number of records written.

        Important: Records are in-memory only until ``persist_records()`` is
        explicitly called by the caller.  The bridge layer
        (``ModelScopeEngineBridge``) does **not** automatically persist records
        on shutdown — the caller is responsible for persisting before the
        process exits.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = [_json.dumps(intervention_record_to_dict(r)) for r in self._records]
        output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(lines)

    def load_records(self, path: str | Path) -> int:
        """Append InterventionRecords from a JSON Lines file written by
        ``persist_records()``.

        Records are appended to the current ``_records`` list (they do not
        replace it), so the method is additive and safe to call on a fresh
        ``SteeringActuator`` instance.  Returns the number of records loaded.
        """
        loaded = 0
        for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                self._records.append(intervention_record_from_dict(_json.loads(line)))
                loaded += 1
            except (_json.JSONDecodeError, ValueError):
                import warnings
                warnings.warn(f"Skipping corrupt JSON line in {path}: {line[:80]!r}", stacklevel=2)
        return loaded


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


@dataclass
class RollbackPlan:
    """An executable, ordered rollback over a sequence of APPLIED interventions
    (rung 10 SHIM — A1a).

    Built from a ``SteeringActuator``'s applied ``InterventionRecord``s in
    application order. Executing the plan replays ``rollback_feature_event`` in
    REVERSE order, so for a feature touched by several interventions the EARLIEST
    intervention's ``original_value`` wins — restoring the exact pre-steering
    feature values. Steering operates only on ``SparseFeatureEvent`` activations
    and never mutates base model weights, so a full execution is a lossless
    rollback of the targeted features: the control-plane "you can roll back / base
    weights are not mutated" claim made EXECUTABLE and verifiable rather than a
    hardcoded assertion.
    """

    records: List[InterventionRecord] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.records)

    @property
    def policy_ids(self) -> List[str]:
        return [r.policy_id for r in self.records]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": "steering_rollback_plan",
            "step_count": self.step_count,
            "policy_ids": self.policy_ids,
            "records": [intervention_record_to_dict(r) for r in self.records],
        }


def build_rollback_plan(records: Sequence[InterventionRecord]) -> RollbackPlan:
    """Collect the APPLIED interventions (in order) into an executable RollbackPlan.

    Shadow / declined records (``applied=False``) carry no mutation and are
    skipped, so the plan contains only reversible steps.
    """
    return RollbackPlan(records=[r for r in records if r.applied])


def execute_rollback(plan: RollbackPlan, feature_event: SparseFeatureEvent) -> SparseFeatureEvent:
    """Replay ``plan``'s rollbacks (reverse application order) to restore the
    pre-steering feature values into a copy of ``feature_event``.

    Reverse order means that for a feature targeted by multiple interventions the
    FIRST (earliest) intervention's original value is applied last and therefore
    wins — the true pre-steering value. An empty plan returns an unmodified copy.
    """
    restored = _copy_feature_event(feature_event)
    for record in reversed(plan.records):
        restored = rollback_feature_event(record, restored)
    return restored


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
