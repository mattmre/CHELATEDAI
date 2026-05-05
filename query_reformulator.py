"""Lightweight query reformulation scaffolding for future adaptive RAG loops."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import re
from typing import Any, List

from chelation_logger import get_logger


@dataclass
class QueryReformulation:
    """One reformulated query candidate."""

    text: str
    strategy: str


class QueryReformulator:
    """Generate deterministic query variants without external LLM dependencies."""

    def __init__(self, stopwords: Iterable[str] | None = None, logger=None):
        self.stopwords = set(stopwords or {"the", "a", "an", "of", "to", "and", "or", "for", "in"})
        self.logger = logger or get_logger()

    def reformulate(self, query: str, max_variants: int = 3) -> List[QueryReformulation]:
        if max_variants < 1:
            raise ValueError("max_variants must be >= 1")
        normalized = " ".join(re.findall(r"[A-Za-z0-9_+-]+", query.lower()))
        tokens = normalized.split()
        if not tokens:
            raise ValueError("query must contain at least one token")

        variants = [QueryReformulation(text=query.strip(), strategy="original")]
        keyword_tokens = [token for token in tokens if token not in self.stopwords]
        if keyword_tokens:
            variants.append(QueryReformulation(text=" ".join(keyword_tokens), strategy="stopword_removed"))
        if len(keyword_tokens) > 2:
            variants.append(QueryReformulation(text=" ".join(keyword_tokens[:2]), strategy="focused_prefix"))

        deduped = []
        seen = set()
        for variant in variants:
            key = variant.text.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(variant)
            if len(deduped) >= max_variants:
                break
        self.logger.log_event(
            "query_reformulated",
            "Generated query reformulation variants",
            variant_count=len(deduped),
            strategies=[variant.strategy for variant in deduped],
            level="DEBUG",
        )
        return deduped


def query_lexical_features(query: str) -> dict:
    """Compute lightweight query-shape features usable before retrieval."""

    normalized = " ".join(re.findall(r"[A-Za-z0-9_+-]+", query.lower()))
    tokens = normalized.split()
    stopwords = {"the", "a", "an", "of", "to", "and", "or", "for", "in"}
    negations = {"no", "not", "without", "lack", "lacks", "never"}
    claim_cues = {
        "increase",
        "increased",
        "decrease",
        "decreased",
        "risk",
        "treat",
        "treats",
        "used",
        "associated",
        "affect",
        "facilitates",
    }
    token_count = len(tokens)
    return {
        "token_count": token_count,
        "char_count": len(query),
        "stopword_ratio": (sum(token in stopwords for token in tokens) / token_count) if token_count else 0.0,
        "numeric_token_count": sum(any(char.isdigit() for char in token) for token in tokens),
        "negation_count": sum(token in negations for token in tokens),
        "claim_cue_count": sum(token in claim_cues for token in tokens),
    }


def _query_policy_features(query: str) -> dict[str, float]:
    """Expose lexical features with both raw and query_-prefixed names."""

    features = query_lexical_features(query)
    return {
        **features,
        **{f"query_{name}": value for name, value in features.items()},
    }


def _linear_policy_score(features: Mapping[str, float], policy: Mapping[str, Any]) -> float | None:
    weights = policy.get("weights")
    if not isinstance(weights, Mapping) or not weights:
        return None
    try:
        score = float(policy.get("bias", policy.get("intercept", 0.0)))
    except (TypeError, ValueError):
        return None
    for name, weight in weights.items():
        if name not in features:
            return None
        try:
            score += float(weight) * float(features[name])
        except (TypeError, ValueError):
            return None
    return score


def _linear_classifier_policy_score(features: Mapping[str, float], policy: Mapping[str, Any]) -> float | None:
    feature_names = policy.get("features")
    means = policy.get("means")
    scales = policy.get("scales")
    weights = policy.get("weights")
    if not all(isinstance(values, list) for values in (feature_names, means, scales, weights)):
        return None
    lengths = {len(feature_names), len(means), len(scales), len(weights)}
    if len(lengths) != 1 or 0 in lengths:
        return None
    try:
        score = float(policy.get("intercept", policy.get("bias", 0.0)))
    except (TypeError, ValueError):
        return None
    for name, mean, scale, weight in zip(feature_names, means, scales, weights):
        if name not in features:
            return None
        try:
            normalized = (float(features[name]) - float(mean)) / max(float(scale), 1e-12)
            score += normalized * float(weight)
        except (TypeError, ValueError):
            return None
    score_transform = policy.get("score_transform", "sigmoid")
    if score_transform == "identity":
        return score
    if score_transform != "sigmoid":
        return None
    clipped = max(min(score, 40.0), -40.0)
    return float(1.0 / (1.0 + math.exp(-clipped)))


def _evaluate_structured_reformulation_policy(query: str, policy: Mapping[str, Any]) -> bool:
    """Return a fail-closed decision for structured learned gate configs."""

    deployment_mode = str(policy.get("deployment_mode", "runtime_enabled")).strip().lower()
    if deployment_mode in {"advisory", "advisory_only", "shadow", "shadow_only", "offline_only"}:
        return False
    if policy.get("runtime_compatible") is False:
        return False
    policy_type = str(policy.get("type", "")).strip().lower()
    features = _query_policy_features(query)
    default_threshold = 0.0
    if policy_type == "linear":
        score = _linear_policy_score(features, policy)
    elif policy_type == "linear_classifier":
        score = _linear_classifier_policy_score(features, policy)
        default_threshold = 0.5
    else:
        return False
    if score is None:
        return False
    try:
        threshold = float(policy.get("threshold", default_threshold))
    except (TypeError, ValueError):
        return False
    operator = policy.get("operator", ">=")
    if operator == ">=":
        return score >= threshold
    if operator == "<=":
        return score <= threshold
    return False


def should_apply_reformulation(query: str, policy: str | Mapping[str, Any] = "always") -> bool:
    """Return whether query reformulation should run for a query under a policy."""

    features = query_lexical_features(query)
    token_count = int(features["token_count"])
    stopword_ratio = float(features["stopword_ratio"])
    if isinstance(policy, str):
        if policy == "always":
            return True
        if policy == "never":
            return False
        if policy == "selective_low_specificity":
            if token_count == 0:
                return False
            return token_count <= 6 or stopword_ratio >= 0.35
        if policy == "selective_high_specificity":
            return token_count >= 8 and stopword_ratio <= 0.25
        if policy == "selective_claim_cue":
            return bool(features["claim_cue_count"] or features["negation_count"] or features["numeric_token_count"])
        raise ValueError(f"unsupported query reformulation policy: {policy}")
    if isinstance(policy, Mapping):
        return _evaluate_structured_reformulation_policy(query, policy)
    raise ValueError(f"unsupported query reformulation policy: {policy}")

