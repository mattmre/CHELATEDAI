"""Bounded shadow-policy training and promotion for Model-Scope replay bundles."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from checkpoint_manager import CheckpointManager


MODEL_SCOPE_TRAINER_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass
class ModelScopeTrainerConfig:
    """Boundaries for replay-driven shadow-policy training."""

    max_rules: int = 8
    min_examples: int = 4
    min_feature_gap: float = 0.15
    min_alignment_score: float = 0.55
    min_rule_support: int = 1
    rule_threshold_fraction: float = 0.6

    def __post_init__(self) -> None:
        if self.max_rules < 1:
            raise ValueError("max_rules must be >= 1")
        if self.min_examples < 1:
            raise ValueError("min_examples must be >= 1")
        if self.min_feature_gap < 0:
            raise ValueError("min_feature_gap must be >= 0")
        if not 0.0 <= self.min_alignment_score <= 1.0:
            raise ValueError("min_alignment_score must be in [0, 1]")
        if self.min_rule_support < 1:
            raise ValueError("min_rule_support must be >= 1")
        if not 0.0 < self.rule_threshold_fraction <= 1.0:
            raise ValueError("rule_threshold_fraction must be in (0, 1]")


def _extract_label(entry: Mapping[str, Any]) -> str | None:
    metadata = entry.get("metadata", {})
    label = metadata.get("label") if isinstance(metadata, Mapping) else None
    if label is not None:
        normalized = str(label).strip().lower()
        if normalized in {"positive", "negative"}:
            return normalized
    comparison = entry.get("expectation_comparison")
    if isinstance(comparison, Mapping) and "passed" in comparison:
        return "positive" if bool(comparison.get("passed")) else "negative"
    return None


def _extract_feature_rows(artifact: Mapping[str, Any]) -> List[Tuple[int, str, str, float]]:
    rows: List[Tuple[int, str, str, float]] = []
    for observation in artifact.get("capture", {}).get("observations", []):
        layer_index = int(observation.get("layer_index", -1))
        feature_summary = observation.get("feature_summary")
        if not isinstance(feature_summary, Mapping):
            continue
        feature_space = str(feature_summary.get("feature_space") or "unknown_feature_space")
        for item in feature_summary.get("active_features", []):
            feature_id = item.get("feature_id")
            if feature_id is None:
                continue
            rows.append((layer_index, feature_space, str(feature_id), float(item.get("value", 0.0))))
    return rows


class ModelScopeShadowPolicyTrainer:
    """Learn advisory shadow policies from replay bundles without mutating base weights."""

    def __init__(self, config: ModelScopeTrainerConfig | None = None):
        self.config = config or ModelScopeTrainerConfig()

    def _empty_candidate(self, candidate_id: str, *, reasons: Iterable[str], example_counts: Mapping[str, int]) -> Dict[str, Any]:
        return {
            "schema_version": MODEL_SCOPE_TRAINER_SCHEMA_VERSION,
            "artifact_type": "model_scope_overlay_candidate",
            "candidate_type": "shadow_policy",
            "candidate_id": candidate_id,
            "feature_space": None,
            "policy": {
                "name": candidate_id,
                "deployment_mode": "shadow_mode",
                "feature_space": None,
                "rules": [],
            },
            "training_summary": {
                "positive_examples": int(example_counts.get("positive", 0)),
                "negative_examples": int(example_counts.get("negative", 0)),
                "labeled_examples": int(sum(example_counts.values())),
            },
            "artifact_manifest": {
                "artifact_class": "shadow_policy_overlay",
                "base_weights_mutated": False,
                "base_model": None,
                "training_evidence_bundle_id": None,
                "replay_bundle_id": None,
                "evaluator_report_id": None,
                "safety_report_id": None,
                "rollback_metadata": {"delete_promoted_policy_file": True},
            },
            "promotion_gate": {
                "promotion_ready": False,
                "alignment_score": 0.0,
                "reasons": list(reasons),
                "rule_count": 0,
                "example_counts": dict(example_counts),
            },
        }

    def train_shadow_policy(
        self,
        replay_bundle: Mapping[str, Any],
        *,
        candidate_id: str = "model_scope_shadow_policy_v1",
    ) -> Dict[str, Any]:
        labeled_entries = []
        for entry in replay_bundle.get("entries", []):
            label = _extract_label(entry)
            artifact = entry.get("artifact")
            if label is None or not isinstance(artifact, Mapping):
                continue
            feature_rows = _extract_feature_rows(artifact)
            if not feature_rows:
                continue
            labeled_entries.append((label, feature_rows))

        example_counts = {
            "positive": sum(1 for label, _rows in labeled_entries if label == "positive"),
            "negative": sum(1 for label, _rows in labeled_entries if label == "negative"),
        }
        if len(labeled_entries) < self.config.min_examples:
            return self._empty_candidate(
                candidate_id,
                reasons=["insufficient_labeled_examples"],
                example_counts=example_counts,
            )
        if example_counts["positive"] == 0 or example_counts["negative"] == 0:
            return self._empty_candidate(
                candidate_id,
                reasons=["class_imbalance_no_positive_or_negative_examples"],
                example_counts=example_counts,
            )

        stats = defaultdict(
            lambda: {
                "layer_index": None,
                "feature_space": None,
                "feature_id": None,
                "positive_sum": 0.0,
                "positive_count": 0,
                "negative_sum": 0.0,
                "negative_count": 0,
            }
        )
        feature_space_counts = defaultdict(int)
        for label, feature_rows in labeled_entries:
            for layer_index, feature_space, feature_id, value in feature_rows:
                key = (layer_index, feature_id)
                stat = stats[key]
                stat["layer_index"] = layer_index
                stat["feature_space"] = feature_space
                stat["feature_id"] = feature_id
                if label == "positive":
                    stat["positive_sum"] += value
                    stat["positive_count"] += 1
                else:
                    stat["negative_sum"] += value
                    stat["negative_count"] += 1
                feature_space_counts[feature_space] += 1

        feature_space = None
        if feature_space_counts:
            feature_space = max(feature_space_counts.items(), key=lambda item: (item[1], item[0]))[0]

        amplify_candidates = []
        suppress_candidates = []
        for stat in stats.values():
            positive_support = int(stat["positive_count"])
            negative_support = int(stat["negative_count"])
            positive_mean = stat["positive_sum"] / positive_support if positive_support > 0 else 0.0
            negative_mean = stat["negative_sum"] / negative_support if negative_support > 0 else 0.0
            positive_gap = positive_mean - negative_mean
            negative_gap = negative_mean - positive_mean
            if positive_support >= self.config.min_rule_support and positive_gap >= self.config.min_feature_gap:
                amplify_candidates.append((positive_gap, positive_mean, stat))
            if negative_support >= self.config.min_rule_support and negative_gap >= self.config.min_feature_gap:
                suppress_candidates.append((negative_gap, negative_mean, stat))

        amplify_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        suppress_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        rule_budget = self.config.max_rules
        amplify_budget = max(1, rule_budget // 2) if amplify_candidates else 0
        suppress_budget = rule_budget - amplify_budget if suppress_candidates else 0
        rules = []
        for gap, mean_value, stat in amplify_candidates[:amplify_budget]:
            rules.append(
                {
                    "feature_id": stat["feature_id"],
                    "min_value": float(max(mean_value * self.config.rule_threshold_fraction, self.config.min_feature_gap)),
                    "layer_index": int(stat["layer_index"]),
                    "action_type": "amplify",
                    "strength": float(gap),
                    "description": "Replay-derived positive-support steering rule",
                }
            )
        for gap, mean_value, stat in suppress_candidates[:suppress_budget]:
            rules.append(
                {
                    "feature_id": stat["feature_id"],
                    "min_value": float(max(mean_value * self.config.rule_threshold_fraction, self.config.min_feature_gap)),
                    "layer_index": int(stat["layer_index"]),
                    "action_type": "suppress",
                    "strength": float(gap),
                    "description": "Replay-derived negative-support steering rule",
                }
            )

        candidate = {
            "schema_version": MODEL_SCOPE_TRAINER_SCHEMA_VERSION,
            "artifact_type": "model_scope_overlay_candidate",
            "candidate_type": "shadow_policy",
            "candidate_id": candidate_id,
            "feature_space": feature_space,
            "policy": {
                "name": candidate_id,
                "deployment_mode": "shadow_mode",
                "feature_space": feature_space,
                "rules": rules,
            },
            "training_summary": {
                "positive_examples": int(example_counts["positive"]),
                "negative_examples": int(example_counts["negative"]),
                "labeled_examples": int(sum(example_counts.values())),
                "candidate_rule_count": len(rules),
                "feature_space": feature_space,
            },
            "artifact_manifest": {
                "artifact_class": "shadow_policy_overlay",
                "base_weights_mutated": False,
                "base_model": replay_bundle.get("metadata", {}).get("base_model")
                if isinstance(replay_bundle.get("metadata"), Mapping)
                else None,
                "training_evidence_bundle_id": replay_bundle.get("evidence_bundle_id"),
                "replay_bundle_id": replay_bundle.get("bundle_id"),
                "evaluator_report_id": replay_bundle.get("evaluator_report_id"),
                "safety_report_id": replay_bundle.get("safety_report_id"),
                "rollback_metadata": {
                    "delete_promoted_policy_file": True,
                    "deployment_mode": "shadow_mode",
                },
            },
        }
        candidate["promotion_gate"] = self.evaluate_shadow_policy_candidate(candidate, replay_bundle)
        return candidate

    def evaluate_shadow_policy_candidate(
        self,
        candidate: Mapping[str, Any],
        replay_bundle: Mapping[str, Any],
    ) -> Dict[str, Any]:
        rules = list(candidate.get("policy", {}).get("rules", []))
        labeled_total = 0
        matched_good = 0
        matched_bad = 0
        neutral = 0
        positive_examples = 0
        negative_examples = 0

        for entry in replay_bundle.get("entries", []):
            label = _extract_label(entry)
            artifact = entry.get("artifact")
            if label is None or not isinstance(artifact, Mapping):
                continue
            if label == "positive":
                positive_examples += 1
            else:
                negative_examples += 1
            labeled_total += 1
            feature_values = {
                (layer_index, feature_id): value
                for layer_index, _feature_space, feature_id, value in _extract_feature_rows(artifact)
            }
            amplify_hit = False
            suppress_hit = False
            for rule in rules:
                key = (int(rule.get("layer_index", -1)), str(rule.get("feature_id")))
                value = feature_values.get(key)
                if value is None or value < float(rule.get("min_value", 0.0)):
                    continue
                if str(rule.get("action_type")) == "amplify":
                    amplify_hit = True
                elif str(rule.get("action_type")) == "suppress":
                    suppress_hit = True
            if label == "positive":
                if amplify_hit:
                    matched_good += 1
                if suppress_hit:
                    matched_bad += 1
                if not amplify_hit and not suppress_hit:
                    neutral += 1
            else:
                if suppress_hit:
                    matched_good += 1
                if amplify_hit:
                    matched_bad += 1
                if not amplify_hit and not suppress_hit:
                    neutral += 1

        precision = matched_good / max(matched_good + matched_bad, 1)
        coverage = matched_good / max(labeled_total, 1)
        alignment_score = (precision + coverage) / 2.0 if labeled_total > 0 else 0.0
        reasons = []
        if labeled_total < self.config.min_examples:
            reasons.append("insufficient_labeled_examples")
        if positive_examples == 0 or negative_examples == 0:
            reasons.append("class_imbalance_no_positive_or_negative_examples")
        if len(rules) == 0:
            reasons.append("no_rules_generated")
        if alignment_score < self.config.min_alignment_score:
            reasons.append("alignment_below_threshold")
        return {
            "promotion_ready": len(reasons) == 0,
            "alignment_score": float(alignment_score),
            "precision": float(precision),
            "coverage": float(coverage),
            "matched_good": int(matched_good),
            "matched_bad": int(matched_bad),
            "neutral_examples": int(neutral),
            "rule_count": int(len(rules)),
            "example_counts": {
                "positive": int(positive_examples),
                "negative": int(negative_examples),
                "total": int(labeled_total),
            },
            "reasons": reasons,
            "config": asdict(self.config),
        }

    def promote_candidate(
        self,
        candidate: Mapping[str, Any],
        target_path: str | Path,
        *,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> Dict[str, Any]:
        gate = candidate.get("promotion_gate", {})
        if not gate.get("promotion_ready"):
            return {
                "promoted": False,
                "path": str(target_path),
                "checkpoint_id": None,
                "reasons": list(gate.get("reasons", ["promotion_gate_failed"])),
            }
        output_path = Path(target_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_id = None
        if checkpoint_manager is not None and output_path.exists():
            checkpoint_id = checkpoint_manager.create_checkpoint(
                name="before_model_scope_shadow_policy",
                adapter_path=output_path,
                description="Automatic backup before Model-Scope shadow-policy promotion",
            )
        output_path.write_text(json.dumps(_json_safe(candidate.get("policy", {})), indent=2), encoding="utf-8")
        return {
            "promoted": True,
            "path": str(output_path),
            "checkpoint_id": checkpoint_id,
            "reasons": [],
        }
