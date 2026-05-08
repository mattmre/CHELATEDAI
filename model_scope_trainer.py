"""Bounded shadow-policy training and promotion for Model-Scope replay bundles."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from checkpoint_manager import CheckpointManager
from expectation_comparator import ExpectationComparator
from model_scope_artifacts import ArtifactStore
from model_scope_features import SparseFeatureEvent
from model_scope_memory import MemoryManager


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


# ---------------------------------------------------------------------------
# Slice-17: OverlayTrainer — iterative overlay trainer + evidence-gated promotion
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OverlayConfig:
    """Configuration for one iterative overlay training campaign."""

    overlay_id: str
    policy_id: str
    learning_rate: float = 0.001
    max_epochs: int = 10
    patience: int = 3
    min_improvement: float = 0.01
    promotion_threshold: float = 0.05
    enabled: bool = True


@dataclass
class TrainingRecord:
    """Result of a single training epoch."""

    overlay_id: str
    epoch: int
    loss: float
    improved: bool
    checkpoint_path: Optional[str]
    created_at: str


@dataclass
class PromotionDecision:
    """Evidence-gated promotion decision for a trained overlay."""

    overlay_id: str
    promoted: bool
    reason: str
    baseline_score: float
    candidate_score: float
    delta: float
    decided_at: str


class OverlayTrainer:
    """Iterative overlay trainer with evidence-gated promotion, early stopping, and rollback.

    Per-feature scaling weights are initialised to 1.0 (identity) and updated with a
    simple first-order rule each epoch:
        weight[k] += lr * (target[k] - weight[k] * input[k])

    Promotion compares two scores computed from (baseline_events, candidate_events) pairs:
      * baseline_score — fraction of pairs where all comparator rules pass for the
        *unscaled* input features vs the target features.
      * candidate_score — same pairs, but the current overlay weights are applied to the
        input features before comparison.
    """

    def __init__(
        self,
        config: OverlayConfig,
        memory: MemoryManager,
        comparator: ExpectationComparator,
        artifact_store: ArtifactStore,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.comparator = comparator
        self.artifact_store = artifact_store
        self._checkpoint_dir: Any = (
            Path(checkpoint_dir) if checkpoint_dir is not None else artifact_store._base_dir / "checkpoints"
        )
        self._weights: Dict[str, float] = {}
        self._epoch_counter: int = 0
        self._best_loss: Optional[float] = None

    def _get_weight(self, key: str) -> float:
        return self._weights.get(key, 1.0)

    def _to_comparison_dict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Build a comparator-compatible dict from a raw feature map."""
        vals = list(features.values())
        mean_act = sum(vals) / len(vals) if vals else 0.0
        return {"features": features, "mean_activation": mean_act}

    def train_epoch(
        self,
        features: List[SparseFeatureEvent],
        target_features: List[SparseFeatureEvent],
    ) -> TrainingRecord:
        """Compute one gradient step and return a TrainingRecord.

        Loss is MSE over all feature keys across all event pairs.
        Weight update: weight[k] += lr * (target[k] - weight[k] * input[k]).
        The record is stored in episodic memory with tags=["training"].
        """
        total_sq_error = 0.0
        total_count = 0

        for inp, tgt in zip(features, target_features):
            all_keys = set(inp.features) | set(tgt.features)
            for k in all_keys:
                inp_val = inp.features.get(k, 0.0)
                tgt_val = tgt.features.get(k, 0.0)
                w = self._get_weight(k)
                total_sq_error += (tgt_val - w * inp_val) ** 2
                self._weights[k] = w + self.config.learning_rate * (tgt_val - w * inp_val)
                total_count += 1

        loss = total_sq_error / max(total_count, 1)
        epoch = self._epoch_counter
        self._epoch_counter += 1

        improved = self._best_loss is None or (self._best_loss - loss) >= self.config.min_improvement
        if improved:
            self._best_loss = loss

        self.memory.episodic.store(
            key=f"epoch_{epoch}",
            value={
                "overlay_id": self.config.overlay_id,
                "epoch": epoch,
                "loss": loss,
                "improved": improved,
            },
            tags=["training"],
        )

        return TrainingRecord(
            overlay_id=self.config.overlay_id,
            epoch=epoch,
            loss=loss,
            improved=improved,
            checkpoint_path=None,
            created_at=_utcnow_iso(),
        )

    def run_campaign(
        self,
        episodes: List[tuple],
    ) -> List[TrainingRecord]:
        """Run epochs 0..max_epochs-1 with early stopping and per-improvement checkpointing.

        Each epoch trains on all events from all episodes (concatenated).
        Early stopping halts after `patience` consecutive non-improving epochs.
        A checkpoint is saved to artifact_store after each improving epoch.
        Returns all TrainingRecords produced during the campaign.
        """
        all_inputs: List[SparseFeatureEvent] = []
        all_targets: List[SparseFeatureEvent] = []
        for inp_list, tgt_list in episodes:
            all_inputs.extend(inp_list)
            all_targets.extend(tgt_list)

        records: List[TrainingRecord] = []
        no_improve_count = 0

        for _ in range(self.config.max_epochs):
            record = self.train_epoch(all_inputs, all_targets)
            if record.improved:
                no_improve_count = 0
                record.checkpoint_path = self._save_checkpoint(record.epoch)
            else:
                no_improve_count += 1

            records.append(record)

            if no_improve_count >= self.config.patience:
                break

        return records

    def _save_checkpoint(self, epoch: int) -> str:
        """Write current weights to a checkpoint JSON file; return its path string."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_dir / f"overlay_{self.config.overlay_id}_epoch_{epoch}.json"
        path.write_text(
            json.dumps({"overlay_id": self.config.overlay_id, "epoch": epoch, "weights": dict(self._weights)}, indent=2),
            encoding="utf-8",
        )
        return str(path)

    def evaluate_promotion(
        self,
        baseline_events: List[SparseFeatureEvent],
        candidate_events: List[SparseFeatureEvent],
    ) -> PromotionDecision:
        """Evaluate whether the trained overlay warrants promotion.

        baseline_score: fraction of pairs where all comparator rules pass comparing
        raw input features (baseline_events[i]) against target features (candidate_events[i]).

        candidate_score: same pairs but overlay weights are applied to the input features,
        measuring what the trained overlay actually produces vs the target.

        promoted = True if (candidate_score - baseline_score) >= promotion_threshold.
        The decision is persisted to PersistentMemory.
        """
        pairs = list(zip(baseline_events, candidate_events))

        if pairs:
            baseline_score = sum(
                float(
                    self.comparator.all_passed(
                        self._to_comparison_dict(b.features),
                        self._to_comparison_dict(c.features),
                    )
                )
                for b, c in pairs
            ) / len(pairs)
        else:
            baseline_score = 0.0

        if pairs:
            candidate_score = sum(
                float(
                    self.comparator.all_passed(
                        self._to_comparison_dict({k: self._get_weight(k) * v for k, v in b.features.items()}),
                        self._to_comparison_dict(c.features),
                    )
                )
                for b, c in pairs
            ) / len(pairs)
        else:
            candidate_score = 0.0

        delta = candidate_score - baseline_score
        promoted = delta >= self.config.promotion_threshold
        reason = (
            f"delta={delta:.6f} >= threshold={self.config.promotion_threshold}"
            if promoted
            else f"delta={delta:.6f} < threshold={self.config.promotion_threshold}"
        )

        decision = PromotionDecision(
            overlay_id=self.config.overlay_id,
            promoted=promoted,
            reason=reason,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            delta=delta,
            decided_at=_utcnow_iso(),
        )

        self.memory.persistent.save(
            key=f"promotion_{self.config.overlay_id}",
            value={
                "overlay_id": decision.overlay_id,
                "promoted": decision.promoted,
                "reason": decision.reason,
                "baseline_score": decision.baseline_score,
                "candidate_score": decision.candidate_score,
                "delta": decision.delta,
                "decided_at": decision.decided_at,
            },
            tags=["promotion_decision"],
        )

        return decision

    def rollback(self) -> None:
        """Reset all feature weights to 1.0 and log a rollback event in working memory."""
        self._weights = {}
        self._best_loss = None
        self.memory.working.store(
            key="rollback",
            value={"overlay_id": self.config.overlay_id, "event": "rollback", "rolled_back_at": _utcnow_iso()},
            tags=["rollback"],
        )

    def save_weights(self, path: str) -> None:
        """Serialize current feature weights to a JSON file."""
        Path(path).write_text(json.dumps({"weights": dict(self._weights)}, indent=2), encoding="utf-8")

    def load_weights(self, path: str) -> None:
        """Load feature weights from a JSON file produced by save_weights."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._weights = {str(k): float(v) for k, v in data.get("weights", {}).items()}
