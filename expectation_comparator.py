"""Offline expectation comparison for Model-Scope artifacts and replay bundles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Dict, List, Mapping


MODEL_SCOPE_EXPECTATION_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class ExpectationComparatorConfig:
    """Thresholds for Model-Scope expectation comparison."""

    min_layer_overlap: float = 0.5
    min_feature_jaccard: float = 0.2
    max_token_count_delta: int | None = None
    max_mean_value_drift: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_layer_overlap <= 1.0:
            raise ValueError("min_layer_overlap must be in [0, 1]")
        if not 0.0 <= self.min_feature_jaccard <= 1.0:
            raise ValueError("min_feature_jaccard must be in [0, 1]")
        if self.max_token_count_delta is not None and self.max_token_count_delta < 0:
            raise ValueError("max_token_count_delta must be >= 0")
        if self.max_mean_value_drift is not None and self.max_mean_value_drift < 0:
            raise ValueError("max_mean_value_drift must be >= 0")


def _extract_feature_layers(artifact: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    layers: Dict[int, Dict[str, Any]] = {}
    observations = artifact.get("capture", {}).get("observations", [])
    for observation in observations:
        layer_index = int(observation.get("layer_index", -1))
        feature_summary = observation.get("feature_summary")
        features: Dict[str, float] = {}
        feature_space = None
        if isinstance(feature_summary, Mapping):
            feature_space = feature_summary.get("feature_space")
            for item in feature_summary.get("active_features", []):
                feature_id = item.get("feature_id")
                if feature_id is None:
                    continue
                features[str(feature_id)] = float(item.get("value", 0.0))
        if not features and isinstance(observation.get("top_dimensions"), list):
            feature_space = feature_space or "activation_top_dimensions"
            for item in observation.get("top_dimensions", []):
                if "dimension" not in item:
                    continue
                features[f"dim_{int(item['dimension'])}"] = float(item.get("mean_abs_activation", 0.0))
        layers[layer_index] = {
            "layer_index": layer_index,
            "feature_space": feature_space,
            "features": features,
        }
    return layers


def _extract_profile_layers(profile: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    layers: Dict[int, Dict[str, Any]] = {}
    for item in profile.get("layers", []):
        layer_index = int(item.get("layer_index", -1))
        features = {
            str(feature_id): float(value)
            for feature_id, value in dict(item.get("features", {})).items()
        }
        layers[layer_index] = {
            "layer_index": layer_index,
            "feature_space": item.get("feature_space"),
            "features": features,
        }
    return layers


class ModelScopeExpectationComparator:
    """Build expectation profiles and compare new artifacts against them."""

    def __init__(self, config: ExpectationComparatorConfig | None = None):
        self.config = config or ExpectationComparatorConfig()

    def build_expectation_profile(
        self,
        artifact: Mapping[str, Any],
        *,
        profile_id: str | None = None,
        description: str = "",
    ) -> Dict[str, Any]:
        capture = artifact.get("capture", {})
        runtime = artifact.get("runtime", {})
        layers = _extract_feature_layers(artifact)
        resolved_profile_id = str(
            profile_id
            or capture.get("prompt_hash")
            or _stable_hash(
                {
                    "model_name": runtime.get("model_name"),
                    "token_count": capture.get("token_count"),
                    "layers": layers,
                }
            )
        )
        return {
            "schema_version": MODEL_SCOPE_EXPECTATION_SCHEMA_VERSION,
            "artifact_type": "model_scope_expectation_profile",
            "profile_id": resolved_profile_id,
            "description": str(description),
            "model_name": runtime.get("model_name"),
            "prompt_hash": capture.get("prompt_hash"),
            "token_count": capture.get("token_count"),
            "captured_layer_count": capture.get("captured_layer_count"),
            "layers": [
                {
                    "layer_index": layer_index,
                    "feature_space": layer["feature_space"],
                    "features": layer["features"],
                }
                for layer_index, layer in sorted(layers.items())
            ],
        }

    def compare_to_profile(
        self,
        artifact: Mapping[str, Any],
        profile: Mapping[str, Any],
        *,
        candidate_id: str | None = None,
    ) -> Dict[str, Any]:
        observed_layers = _extract_feature_layers(artifact)
        expected_layers = _extract_profile_layers(profile)
        layer_reports: List[Dict[str, Any]] = []
        present_count = 0
        passed_layers = 0
        jaccards: List[float] = []
        drifts: List[float] = []

        for layer_index, expected in sorted(expected_layers.items()):
            observed = observed_layers.get(layer_index)
            if observed is None:
                layer_reports.append(
                    {
                        "layer_index": layer_index,
                        "status": "missing",
                        "feature_jaccard": 0.0,
                        "mean_value_drift": None,
                    }
                )
                continue
            present_count += 1
            expected_features = expected["features"]
            observed_features = observed["features"]
            expected_ids = set(expected_features)
            observed_ids = set(observed_features)
            union = expected_ids | observed_ids
            intersection = expected_ids & observed_ids
            jaccard = 1.0 if not union else len(intersection) / len(union)
            all_ids = sorted(union)
            mean_drift = (
                sum(abs(observed_features.get(feature_id, 0.0) - expected_features.get(feature_id, 0.0)) for feature_id in all_ids)
                / len(all_ids)
                if all_ids
                else 0.0
            )
            passes_thresholds = jaccard >= self.config.min_feature_jaccard
            if self.config.max_mean_value_drift is not None:
                passes_thresholds = passes_thresholds and mean_drift <= self.config.max_mean_value_drift
            if passes_thresholds:
                passed_layers += 1
            jaccards.append(jaccard)
            drifts.append(mean_drift)
            layer_reports.append(
                {
                    "layer_index": layer_index,
                    "status": "matched" if passes_thresholds else "drifted",
                    "feature_jaccard": float(jaccard),
                    "mean_value_drift": float(mean_drift),
                    "expected_feature_count": len(expected_ids),
                    "observed_feature_count": len(observed_ids),
                }
            )

        expected_layer_total = max(len(expected_layers), 1)
        layer_overlap_ratio = present_count / expected_layer_total
        passed_layer_ratio = passed_layers / expected_layer_total
        token_count = artifact.get("capture", {}).get("token_count")
        token_count_delta = (
            None
            if token_count is None or profile.get("token_count") is None
            else abs(int(token_count) - int(profile.get("token_count")))
        )
        mean_feature_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0
        mean_value_drift = sum(drifts) / len(drifts) if drifts else 0.0
        score = max(
            0.0,
            min(
                1.0,
                (0.7 * mean_feature_jaccard) + (0.3 * passed_layer_ratio) - min(mean_value_drift, 1.0) * 0.1,
            ),
        )
        passes = passed_layer_ratio >= self.config.min_layer_overlap
        reasons = []
        if layer_overlap_ratio < self.config.min_layer_overlap:
            passes = False
            reasons.append("layer_overlap_below_threshold")
        if self.config.max_token_count_delta is not None and token_count_delta is not None:
            if token_count_delta > self.config.max_token_count_delta:
                passes = False
                reasons.append("token_count_delta_above_threshold")
        if mean_feature_jaccard < self.config.min_feature_jaccard:
            passes = False
            reasons.append("feature_jaccard_below_threshold")
        if self.config.max_mean_value_drift is not None and mean_value_drift > self.config.max_mean_value_drift:
            passes = False
            reasons.append("mean_value_drift_above_threshold")
        return {
            "schema_version": MODEL_SCOPE_EXPECTATION_SCHEMA_VERSION,
            "comparison_type": "artifact_vs_expectation_profile",
            "candidate_id": candidate_id,
            "profile_id": profile.get("profile_id"),
            "passed": bool(passes),
            "reasons": reasons,
            "score": float(score),
            "layer_overlap_ratio": float(layer_overlap_ratio),
            "passed_layer_ratio": float(passed_layer_ratio),
            "mean_feature_jaccard": float(mean_feature_jaccard),
            "mean_value_drift": float(mean_value_drift),
            "token_count_delta": token_count_delta,
            "layer_reports": layer_reports,
            "config": asdict(self.config),
        }

    def compare_artifacts(
        self,
        baseline_artifact: Mapping[str, Any],
        candidate_artifact: Mapping[str, Any],
        *,
        candidate_id: str | None = None,
    ) -> Dict[str, Any]:
        profile = self.build_expectation_profile(baseline_artifact)
        return self.compare_to_profile(candidate_artifact, profile, candidate_id=candidate_id)

    def compare_replay_bundles(
        self,
        reference_bundle: Mapping[str, Any],
        candidate_bundle: Mapping[str, Any],
    ) -> Dict[str, Any]:
        reference_by_query: Dict[str, Mapping[str, Any]] = {}
        for entry in reference_bundle.get("entries", []):
            query_hash = entry.get("query_hash")
            artifact = entry.get("artifact")
            if query_hash is None or not isinstance(artifact, Mapping):
                continue
            reference_by_query.setdefault(str(query_hash), artifact)
        comparisons = []
        for entry in candidate_bundle.get("entries", []):
            query_hash = entry.get("query_hash")
            artifact = entry.get("artifact")
            if query_hash is None or not isinstance(artifact, Mapping):
                continue
            reference_artifact = reference_by_query.get(str(query_hash))
            if reference_artifact is None:
                continue
            comparisons.append(
                self.compare_artifacts(
                    reference_artifact,
                    artifact,
                    candidate_id=str(entry.get("entry_id")),
                )
            )
        mean_score = (
            sum(float(item.get("score", 0.0)) for item in comparisons) / len(comparisons)
            if comparisons
            else 0.0
        )
        return {
            "schema_version": MODEL_SCOPE_EXPECTATION_SCHEMA_VERSION,
            "comparison_type": "replay_bundle_vs_replay_bundle",
            "comparison_count": len(comparisons),
            "passed_count": sum(1 for item in comparisons if item.get("passed")),
            "failed_count": sum(1 for item in comparisons if not item.get("passed")),
            "mean_score": float(mean_score),
            "comparisons": comparisons,
            "config": asdict(self.config),
        }


# ---------------------------------------------------------------------------
# Slice-16: ComparatorRule API
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Result of a single comparator rule evaluation."""

    rule_name: str
    passed: bool
    delta: float
    threshold: float
    detail: str


class ComparatorRule(ABC):
    """Abstract base for expectation comparator rules."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def evaluate(self, baseline: dict, candidate: dict) -> ComparisonResult: ...


class MeanActivationRule(ComparatorRule):
    """Compare mean activation values between baseline and candidate."""

    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = float(threshold)

    @property
    def name(self) -> str:
        return "mean_activation"

    def evaluate(self, baseline: dict, candidate: dict) -> ComparisonResult:
        baseline_mean = float(baseline.get("mean_activation", 0.0))
        candidate_mean = float(candidate.get("mean_activation", 0.0))
        delta = abs(candidate_mean - baseline_mean) / (abs(baseline_mean) + 1e-9)
        passed = delta <= self.threshold
        detail = (
            f"baseline={baseline_mean:.6f}, candidate={candidate_mean:.6f}, delta={delta:.6f}"
        )
        return ComparisonResult(
            rule_name=self.name,
            passed=passed,
            delta=delta,
            threshold=self.threshold,
            detail=detail,
        )


class FeatureOverlapRule(ComparatorRule):
    """Compare feature overlap (Jaccard on nonzero keys) between baseline and candidate."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    @property
    def name(self) -> str:
        return "feature_overlap"

    def evaluate(self, baseline: dict, candidate: dict) -> ComparisonResult:
        baseline_features: dict = baseline.get("features", {})
        candidate_features: dict = candidate.get("features", {})
        baseline_keys = {k for k, v in baseline_features.items() if float(v) != 0.0}
        candidate_keys = {k for k, v in candidate_features.items() if float(v) != 0.0}
        union = baseline_keys | candidate_keys
        intersection = baseline_keys & candidate_keys
        overlap = len(intersection) / (len(union) + 1e-9)
        passed = overlap >= self.threshold
        detail = (
            f"|intersection|={len(intersection)}, |union|={len(union)}, overlap={overlap:.6f}"
        )
        return ComparisonResult(
            rule_name=self.name,
            passed=passed,
            delta=overlap,
            threshold=self.threshold,
            detail=detail,
        )


class InterventionCountRule(ComparatorRule):
    """Ensure candidate intervention count does not exceed a maximum."""

    def __init__(self, max_interventions: int = 5) -> None:
        self.max_interventions = int(max_interventions)

    @property
    def name(self) -> str:
        return "intervention_count"

    def evaluate(self, baseline: dict, candidate: dict) -> ComparisonResult:
        count = float(candidate.get("intervention_count", 0))
        passed = count <= self.max_interventions
        detail = f"intervention_count={count}, max={self.max_interventions}"
        return ComparisonResult(
            rule_name=self.name,
            passed=passed,
            delta=count,
            threshold=float(self.max_interventions),
            detail=detail,
        )


class ExpectationComparator:
    """Run a list of ComparatorRules and aggregate results."""

    def __init__(self) -> None:
        self.rules: List[ComparatorRule] = []

    def add_rule(self, rule: ComparatorRule) -> None:
        self.rules.append(rule)

    def compare(self, baseline: dict, candidate: dict) -> List[ComparisonResult]:
        return [rule.evaluate(baseline, candidate) for rule in self.rules]

    def all_passed(self, baseline: dict, candidate: dict) -> bool:
        return all(r.passed for r in self.compare(baseline, candidate))

    def summary(self, baseline: dict, candidate: dict) -> dict:
        results = self.compare(baseline, candidate)
        return {
            "passed": all(r.passed for r in results),
            "results": [asdict(r) for r in results],
            "rule_count": len(results),
        }


class ReplaySetGenerator:
    """Generate comparison sets from an EpisodicMemory replay bundle."""

    def __init__(self, episodic_memory: Any) -> None:
        self.episodic_memory = episodic_memory

    def generate(
        self,
        episode_id: str,
        comparator: ExpectationComparator,
        baseline_key: str,
    ) -> List[dict]:
        bundle = self.episodic_memory.replay_bundle(episode_id)
        baseline_entry = next((e for e in bundle if e.key == baseline_key), None)
        if baseline_entry is None:
            return []
        baseline = baseline_entry.value
        results = []
        for entry in bundle:
            if entry.key == baseline_key:
                continue
            comparison = comparator.summary(baseline, entry.value)
            results.append({"entry": entry, "comparison": comparison})
        return results
