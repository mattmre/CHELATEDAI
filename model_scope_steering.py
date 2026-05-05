"""Shadow-mode steering evaluation for Model-Scope artifacts."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from steering_policy import ModelScopeSteeringPolicy


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
