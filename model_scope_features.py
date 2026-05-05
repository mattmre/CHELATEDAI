"""Sparse feature extraction surfaces for Model-Scope captures."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import torch


MODEL_SCOPE_FEATURE_SCORECARD_SCHEMA_VERSION = 1


def _artifact_feature_rows(artifact: Mapping[str, Any]):
    for observation in artifact.get("capture", {}).get("observations", []) or []:
        if not isinstance(observation, Mapping):
            continue
        layer_index = int(observation.get("layer_index", -1))
        feature_summary = observation.get("feature_summary")
        if not isinstance(feature_summary, Mapping):
            continue
        feature_space = str(feature_summary.get("feature_space") or "unknown")
        for item in feature_summary.get("active_features", []) or []:
            if not isinstance(item, Mapping) or item.get("feature_id") is None:
                continue
            yield {
                "layer_index": layer_index,
                "feature_space": feature_space,
                "feature_id": str(item.get("feature_id")),
                "value": float(item.get("value", 0.0)),
            }


def build_feature_scorecard(entries: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Score activation features by support, stability, and intervention risk."""

    by_feature: Dict[str, Dict[str, Any]] = {}
    entry_count = 0
    for entry in entries:
        entry_count += 1
        label = str(entry.get("label") or entry.get("metadata", {}).get("label") or "unknown")
        artifact = entry.get("artifact") if isinstance(entry.get("artifact"), Mapping) else entry
        seen_in_entry = set()
        for row in _artifact_feature_rows(artifact):
            key = f"{row['feature_space']}|L{row['layer_index']}|{row['feature_id']}"
            record = by_feature.setdefault(
                key,
                {
                    "feature_key": key,
                    "feature_space": row["feature_space"],
                    "layer_index": row["layer_index"],
                    "feature_id": row["feature_id"],
                    "entry_support": 0,
                    "positive_support": 0,
                    "negative_support": 0,
                    "unknown_support": 0,
                    "values": [],
                },
            )
            if key not in seen_in_entry:
                record["entry_support"] += 1
                if label == "positive":
                    record["positive_support"] += 1
                elif label == "negative":
                    record["negative_support"] += 1
                else:
                    record["unknown_support"] += 1
                seen_in_entry.add(key)
            record["values"].append(float(row["value"]))

    features = []
    for record in by_feature.values():
        values = record.pop("values")
        mean_value = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean_value) ** 2 for value in values) / len(values) if values else 0.0
        support_ratio = record["entry_support"] / max(entry_count, 1)
        positive = int(record["positive_support"])
        negative = int(record["negative_support"])
        polarity = (positive - negative) / max(positive + negative, 1)
        intervention_risk = min(
            1.0,
            (negative / max(record["entry_support"], 1)) + min(variance, 1.0) * 0.25,
        )
        features.append(
            {
                **record,
                "mean_value": float(mean_value),
                "value_variance": float(variance),
                "support_ratio": float(support_ratio),
                "polarity": float(polarity),
                "intervention_risk": float(intervention_risk),
                "recommended_posture": (
                    "candidate_amplify"
                    if polarity > 0.5 and intervention_risk < 0.5
                    else "candidate_suppress"
                    if polarity < -0.5
                    else "observe_only"
                ),
            }
        )
    features.sort(key=lambda item: (-abs(float(item["polarity"])), -int(item["entry_support"]), item["feature_key"]))
    return {
        "schema_version": MODEL_SCOPE_FEATURE_SCORECARD_SCHEMA_VERSION,
        "artifact_type": "model_scope_feature_scorecard",
        "entry_count": entry_count,
        "feature_count": len(features),
        "features": features,
    }


class FallbackActivationFeatureExtractor:
    """Feature summaries derived directly from activation dimensions."""

    def __init__(self, top_dimensions: int = 8):
        self.top_dimensions = int(top_dimensions)

    def summarize(self, *, layer_index: int, activation: torch.Tensor):
        value = activation.detach().float().cpu()
        if value.ndim == 3:
            last_token = value[0, -1]
        elif value.ndim == 2:
            last_token = value[-1]
        else:
            last_token = value.reshape(-1)
        magnitudes = last_token.abs()
        top_k = min(max(self.top_dimensions, 0), magnitudes.numel())
        if top_k > 0:
            top_vals, top_idx = torch.topk(magnitudes, k=top_k, dim=-1)
            features = [
                {
                    "feature_id": f"dim_{int(index)}",
                    "value": float(score),
                }
                for index, score in zip(top_idx.tolist(), top_vals.tolist())
            ]
        else:
            features = []
        return {
            "feature_space": "activation_dimension_fallback",
            "layer_index": int(layer_index),
            "token_selector": "last_token",
            "active_feature_count": len(features),
            "active_features": features,
        }


class QwenScopeFeatureExtractor:
    """Extract sparse feature summaries using official Qwen-Scope SAEs when available."""

    def __init__(self, sae_layers: Mapping[int, object], *, top_features: int = 8, fallback=None):
        self.sae_layers = {int(layer_index): sae for layer_index, sae in sae_layers.items()}
        self.top_features = int(top_features)
        self.fallback = fallback

    def summarize(self, *, layer_index: int, activation: torch.Tensor):
        sae = self.sae_layers.get(int(layer_index))
        if sae is not None:
            return sae.summarize_last_token(activation, top_features=self.top_features)
        if self.fallback is not None:
            return self.fallback.summarize(layer_index=layer_index, activation=activation)
        return None
