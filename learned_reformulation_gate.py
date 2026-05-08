"""Train fail-closed learned query-reformulation gates from tuning artifacts.

Two training paths are provided:

train_reformulation_gate(rows)
    Classic path: rows come from load_query_attribution_rows() + enrich/filter.
    Positive label = delta_ndcg_at_10 > positive_delta.

train_gate_from_pool(pool)
    Attribution-pool path: accepts the dict produced by build_attribution_pool().
    Positive label = action == positive_action (default "REFORMULATE").
    This uses action-signal learning: learns which query patterns historically route to
    reformulation regardless of whether that routing improved NDCG.

predict_single(feature_dict, gate_config)
    Run inference for a single query feature vector against a saved gate config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from benchmark_utils import canonicalize_id, load_mteb_data
from engine_scope import build_engine_scope_rows, load_engine_scope_rows
from query_reformulator import query_lexical_features


REFORMULATION_RUNTIME_FEATURES = [
    "query_token_count",
    "query_char_count",
    "query_stopword_ratio",
    "query_numeric_token_count",
    "query_negation_count",
    "query_claim_cue_count",
]

REFORMULATION_ENGINE_SCOPE_FEATURES = [
    "top10_overlap_with_baseline",
    "top_doc_changed",
    "global_variance",
    "jaccard",
    "mask_density",
    "reformulation_variant_count",
    "reformulation_changed",
]

REFORMULATION_FEATURES = [
    *REFORMULATION_RUNTIME_FEATURES,
    *REFORMULATION_ENGINE_SCOPE_FEATURES,
]

_QUERY_PROFILE_ROW_TYPE = "query_profile"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def _iter_artifact_paths(paths: Sequence[str | Path]) -> List[Path]:
    artifact_paths: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            artifact_paths.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            artifact_paths.append(path)
        else:
            raise FileNotFoundError(f"artifact path not found: {path}")
    if not artifact_paths:
        raise ValueError("no artifact files found")
    deduped: List[Path] = []
    seen = set()
    for path in artifact_paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def load_query_attribution_rows(paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    """Load reformulation training rows from Engine-Scope or legacy artifacts."""

    combined = [
        dict(row)
        for row in load_engine_scope_rows(paths)
        if str(row.get("row_type") or _QUERY_PROFILE_ROW_TYPE) == _QUERY_PROFILE_ROW_TYPE
    ]
    if not combined:
        raise ValueError("no reformulation training rows found in the supplied artifacts")
    return combined


def _task_query_texts(task: str) -> Dict[str, str]:
    _corpus, queries, _qrels = load_mteb_data(task)
    return {
        canonicalize_id(query_id): str(query_text)
        for query_id, query_text in queries.items()
    }


def enrich_query_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure pooled rows have lexical query features and query text when possible."""

    query_cache: Dict[str, Dict[str, str]] = {}
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        task = str(item.get("task") or "")
        query_id = str(item.get("query_id") or "")
        if task and query_id and "query_text" not in item:
            if task not in query_cache:
                query_cache[task] = _task_query_texts(task)
            query_text = query_cache[task].get(query_id)
            if query_text is not None:
                item["query_text"] = query_text
        if task and query_id and any(feature not in item for feature in REFORMULATION_FEATURES):
            if task not in query_cache:
                query_cache[task] = _task_query_texts(task)
            query_text = item.get("query_text") or query_cache[task].get(query_id, "")
            if query_text:
                lexical = query_lexical_features(str(query_text))
                item.setdefault("query_text", str(query_text))
                item.setdefault("query_token_count", lexical["token_count"])
                item.setdefault("query_char_count", lexical["char_count"])
                item.setdefault("query_stopword_ratio", lexical["stopword_ratio"])
                item.setdefault("query_numeric_token_count", lexical["numeric_token_count"])
                item.setdefault("query_negation_count", lexical["negation_count"])
                item.setdefault("query_claim_cue_count", lexical["claim_cue_count"])
        item.setdefault("row_type", _QUERY_PROFILE_ROW_TYPE)
        item["top_doc_changed"] = 1.0 if bool(item.get("top_doc_changed", False)) else 0.0
        item["reformulation_changed"] = 1.0 if bool(item.get("reformulation_changed", False)) else 0.0
        for feature in (
            "top10_overlap_with_baseline",
            "global_variance",
            "jaccard",
            "mask_density",
            "reformulation_variant_count",
        ):
            if not _is_number(item.get(feature)):
                item[feature] = 0.0
        enriched.append(item)
    return enriched


def filter_reformulation_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    profile: str = "reform_rrf_v2",
) -> List[Dict[str, Any]]:
    """Return only rows for the target reformulation profile with usable features."""

    filtered = []
    for row in rows:
        item = dict(row)
        if str(item.get("row_type") or _QUERY_PROFILE_ROW_TYPE) != _QUERY_PROFILE_ROW_TYPE:
            continue
        if str(item.get("profile")) != profile:
            continue
        enriched_row = enrich_query_rows([item])[0]
        if not all(_is_number(enriched_row.get(feature)) for feature in REFORMULATION_FEATURES):
            continue
        if not _is_number(enriched_row.get("delta_ndcg_at_10")):
            continue
        filtered.append(enriched_row)
    return filtered


def _classifier_matrix(rows: List[Dict[str, Any]], features: Sequence[str]) -> np.ndarray:
    def _safe_float(value: Any) -> float:
        return float(value) if value is not None else 0.0

    return np.array(
        [[_safe_float(row.get(feature)) for feature in features] for row in rows],
        dtype=float,
    )


def _train_logistic_weights(
    rows: List[Dict[str, Any]],
    *,
    features: Sequence[str],
    positive_delta: float,
    l2_penalty: float,
    learning_rate: float = 0.08,
    epochs: int = 300,
) -> Dict[str, Any] | None:
    matrix = _classifier_matrix(rows, features)
    labels = np.array(
        [1.0 if float(row["delta_ndcg_at_10"]) > positive_delta else 0.0 for row in rows],
        dtype=float,
    )
    positive_count = int(np.sum(labels))
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    normalized = (matrix - means) / np.maximum(scales, 1e-12)
    weights = np.zeros(normalized.shape[1], dtype=float)
    intercept = 0.0
    sample_weights = np.where(
        labels > 0.0,
        len(labels) / (2.0 * positive_count),
        len(labels) / (2.0 * negative_count),
    )
    for _epoch in range(epochs):
        logits = np.clip(normalized @ weights + intercept, -40.0, 40.0)
        predictions = 1.0 / (1.0 + np.exp(-logits))
        errors = (predictions - labels) * sample_weights
        weights -= learning_rate * ((normalized.T @ errors) / len(labels) + l2_penalty * weights)
        intercept -= learning_rate * float(np.mean(errors))
    return {
        "features": list(features),
        "means": [float(value) for value in means],
        "scales": [float(value) for value in np.maximum(scales, 1e-12)],
        "weights": [float(value) for value in weights],
        "intercept": float(intercept),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def _classifier_score(row: Dict[str, Any], gate: Dict[str, Any]) -> float:
    values = np.array(
        [float(row.get(feature) if row.get(feature) is not None else 0.0) for feature in gate["features"]],
        dtype=float,
    )
    means = np.array(gate["means"], dtype=float)
    scales = np.array(gate["scales"], dtype=float)
    weights = np.array(gate["weights"], dtype=float)
    normalized = (values - means) / np.maximum(scales, 1e-12)
    logit = float(normalized @ weights + float(gate["intercept"]))
    clipped = np.clip(logit, -40.0, 40.0)
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _group_key(row: Dict[str, Any]) -> str:
    return f"{row.get('task', '')}:{row.get('query_id', '')}"


def _is_holdout_row(row: Dict[str, Any], holdout_modulo: int, holdout_remainder: int) -> bool:
    digest = hashlib.sha256(_group_key(row).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % holdout_modulo == holdout_remainder


def _summarize_rows(rows: List[Dict[str, Any]], gate: Dict[str, Any] | None) -> Dict[str, Any]:
    if gate is None:
        matched = []
    else:
        matched = [row for row in rows if _classifier_score(row, gate) >= float(gate["threshold"])]
    deltas = [float(row["delta_ndcg_at_10"]) for row in matched]
    tasks = sorted({str(row.get("task")) for row in matched})
    return {
        "rows": len(rows),
        "matched": len(matched),
        "tasks": tasks,
        "mean_delta_ndcg_at_10": float(np.mean(deltas)) if deltas else 0.0,
        "best_delta_ndcg_at_10": float(np.max(deltas)) if deltas else 0.0,
        "worst_delta_ndcg_at_10": float(np.min(deltas)) if deltas else 0.0,
        "positive_examples": sum(delta > 0.001 for delta in deltas),
        "negative_examples": sum(delta < -0.001 for delta in deltas),
        "neutral_examples": sum(abs(delta) <= 0.001 for delta in deltas),
    }


def train_reformulation_gate(
    rows: Iterable[Dict[str, Any]],
    *,
    features: Sequence[str] = REFORMULATION_FEATURES,
    holdout_modulo: int = 5,
    holdout_remainder: int = 0,
    min_train_support: int = 2,
    min_holdout_support: int = 1,
    min_mean_delta: float = 0.0,
    max_negative_examples: int = 0,
    positive_delta: float = 0.001,
    min_positive_examples: int = 2,
    l2_penalty: float = 0.08,
) -> Dict[str, Any]:
    """Train a fail-closed learned reformulation gate from pooled query rows."""

    all_rows = [dict(row) for row in rows]
    runtime_compatible = set(features).issubset(set(REFORMULATION_RUNTIME_FEATURES))
    deployment_mode = "runtime_enabled" if runtime_compatible else "advisory_only"
    if holdout_modulo <= 1:
        raise ValueError("holdout_modulo must be > 1")
    train_rows = [
        row for row in all_rows
        if not _is_holdout_row(row, holdout_modulo, holdout_remainder)
    ]
    holdout_rows = [
        row for row in all_rows
        if _is_holdout_row(row, holdout_modulo, holdout_remainder)
    ]
    model = _train_logistic_weights(
        train_rows,
        features=features,
        positive_delta=positive_delta,
        l2_penalty=l2_penalty,
    )
    criteria = {
        "holdout_modulo": holdout_modulo,
        "holdout_remainder": holdout_remainder,
        "min_train_support": min_train_support,
        "min_holdout_support": min_holdout_support,
        "min_mean_delta": min_mean_delta,
        "max_negative_examples": max_negative_examples,
        "positive_delta": positive_delta,
        "min_positive_examples": min_positive_examples,
        "l2_penalty": l2_penalty,
    }
    if model is None:
        return {
            "version": 1,
            "policy": "query_reformulation_linear_classifier",
            "feature_space": "engine_scope",
            "deployment_mode": deployment_mode,
            "runtime_compatible": runtime_compatible,
            "gate": None,
            "accepted": [],
            "rejected_count": 0,
            "criteria": criteria,
            "training_summary": {
                "train_rows": len(train_rows),
                "holdout_rows": len(holdout_rows),
                "positive_examples": sum(float(row["delta_ndcg_at_10"]) > positive_delta for row in train_rows),
            },
        }
    if model["positive_count"] < min_positive_examples:
        return {
            "version": 1,
            "policy": "query_reformulation_linear_classifier",
            "feature_space": "engine_scope",
            "deployment_mode": deployment_mode,
            "runtime_compatible": runtime_compatible,
            "gate": None,
            "accepted": [],
            "rejected_count": 0,
            "criteria": criteria,
            "training_summary": {
                "train_rows": len(train_rows),
                "holdout_rows": len(holdout_rows),
                "positive_examples": model["positive_count"],
                "negative_examples": model["negative_count"],
            },
        }
    train_scores = [
        _classifier_score(
            row,
            {"threshold": 0.0, **model},
        )
        for row in train_rows
    ]
    thresholds = sorted(set(np.quantile(train_scores, [0.5, 0.65, 0.8, 0.9]).tolist()), reverse=True)
    accepted = []
    rejected_count = 0
    for threshold in thresholds:
        gate = {
            "type": "linear_classifier",
            "operator": ">=",
            "score_transform": "sigmoid",
            "feature_space": "engine_scope",
            "deployment_mode": deployment_mode,
            "runtime_compatible": runtime_compatible,
            "threshold": float(threshold),
            **model,
        }
        train_summary = _summarize_rows(train_rows, gate)
        holdout_summary = _summarize_rows(holdout_rows, gate)
        train_passed = (
            train_summary["matched"] >= min_train_support
            and train_summary["mean_delta_ndcg_at_10"] >= min_mean_delta
            and train_summary["negative_examples"] <= max_negative_examples
        )
        holdout_passed = (
            holdout_summary["matched"] >= min_holdout_support
            and holdout_summary["mean_delta_ndcg_at_10"] >= min_mean_delta
            and holdout_summary["negative_examples"] <= max_negative_examples
        )
        record = {
            "gate": gate,
            "train": train_summary,
            "holdout": holdout_summary,
            "accepted": train_passed and holdout_passed,
        }
        if record["accepted"]:
            accepted.append(record)
        else:
            rejected_count += 1
    accepted.sort(
        key=lambda item: (
            -item["holdout"]["mean_delta_ndcg_at_10"],
            -item["train"]["mean_delta_ndcg_at_10"],
            -item["holdout"]["matched"],
            -item["train"]["matched"],
        )
    )
    return {
        "version": 1,
        "policy": "query_reformulation_linear_classifier",
        "feature_space": "engine_scope",
        "deployment_mode": deployment_mode,
        "runtime_compatible": runtime_compatible,
        "gate": accepted[0]["gate"] if accepted else None,
        "accepted": accepted[:5],
        "rejected_count": rejected_count,
        "criteria": criteria,
        "training_summary": {
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "positive_examples": model["positive_count"],
            "negative_examples": model["negative_count"],
        },
    }


def write_gate_config(config: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Attribution-pool training path (Slice 2)
# ---------------------------------------------------------------------------

def _train_action_gate_weights(
    rows: List[Dict[str, Any]],
    *,
    features: Sequence[str],
    positive_action: str = "REFORMULATE",
    l2_penalty: float = 0.08,
    learning_rate: float = 0.08,
    epochs: int = 300,
) -> Dict[str, Any] | None:
    """Train logistic regression using action type as the positive label.

    Returns None (fail-closed) when either class is absent.
    """
    matrix = _classifier_matrix(rows, features)
    labels = np.array(
        [1.0 if str(row.get("action", "")) == positive_action else 0.0 for row in rows],
        dtype=float,
    )
    positive_count = int(np.sum(labels))
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    normalized = (matrix - means) / np.maximum(scales, 1e-12)
    weights = np.zeros(normalized.shape[1], dtype=float)
    intercept = 0.0
    sample_weights = np.where(
        labels > 0.0,
        len(labels) / (2.0 * positive_count),
        len(labels) / (2.0 * negative_count),
    )
    for _epoch in range(epochs):
        logits = np.clip(normalized @ weights + intercept, -40.0, 40.0)
        predictions = 1.0 / (1.0 + np.exp(-logits))
        errors = (predictions - labels) * sample_weights
        weights -= learning_rate * ((normalized.T @ errors) / len(labels) + l2_penalty * weights)
        intercept -= learning_rate * float(np.mean(errors))
    return {
        "features": list(features),
        "means": [float(v) for v in means],
        "scales": [float(v) for v in np.maximum(scales, 1e-12)],
        "weights": [float(v) for v in weights],
        "intercept": float(intercept),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_action": positive_action,
    }


def load_attribution_pool(path: str | Path) -> Dict[str, Any]:
    """Load an attribution pool JSON produced by build_attribution_pool."""
    pool_path = Path(path)
    if not pool_path.is_file():
        raise FileNotFoundError(f"attribution pool file not found: {pool_path}")
    return json.loads(pool_path.read_text(encoding="utf-8"))


def train_gate_from_pool(
    pool: Dict[str, Any],
    *,
    features: Sequence[str] = REFORMULATION_FEATURES,
    positive_action: str = "REFORMULATE",
    holdout_modulo: int = 5,
    holdout_remainder: int = 0,
    min_train_support: int = 2,
    min_holdout_support: int = 1,
    l2_penalty: float = 0.08,
) -> Dict[str, Any]:
    """Train a fail-closed gate from an attribution pool using action as the label.

    The pool dict is produced by ``build_attribution_pool()``.  Rows with
    ``action == positive_action`` are treated as positives; all other actions are
    negatives.  The returned config follows the same schema as
    ``train_reformulation_gate`` so it can be passed to ``predict_single``.
    """
    raw_rows: List[Dict[str, Any]] = [
        dict(r) for r in pool.get("query_attribution_rows", []) if isinstance(r, dict)
    ]
    if not raw_rows:
        raise ValueError("pool contains no query_attribution_rows")

    # Convert to engine-scope format so enrich_query_rows can fill lexical features.
    scope_rows = build_engine_scope_rows(
        query_attribution_rows=raw_rows,
        source_family="attribution_pool",
    )
    all_rows = enrich_query_rows(scope_rows)

    if holdout_modulo <= 1:
        raise ValueError("holdout_modulo must be > 1")
    train_rows = [r for r in all_rows if not _is_holdout_row(r, holdout_modulo, holdout_remainder)]
    holdout_rows = [r for r in all_rows if _is_holdout_row(r, holdout_modulo, holdout_remainder)]

    runtime_compatible = set(features).issubset(set(REFORMULATION_RUNTIME_FEATURES))
    deployment_mode = "runtime_enabled" if runtime_compatible else "advisory_only"

    criteria = {
        "holdout_modulo": holdout_modulo,
        "holdout_remainder": holdout_remainder,
        "min_train_support": min_train_support,
        "min_holdout_support": min_holdout_support,
        "positive_action": positive_action,
        "l2_penalty": l2_penalty,
    }

    model = _train_action_gate_weights(
        train_rows,
        features=features,
        positive_action=positive_action,
        l2_penalty=l2_penalty,
    )

    def _action_summary(rows: List[Dict[str, Any]], gate: Dict[str, Any] | None) -> Dict[str, Any]:
        if gate is None:
            matched: List[Dict[str, Any]] = []
        else:
            matched = [r for r in rows if _classifier_score(r, gate) >= float(gate["threshold"])]
        true_positives = sum(1 for r in matched if str(r.get("action", "")) == positive_action)
        false_positives = len(matched) - true_positives
        return {
            "rows": len(rows),
            "matched": len(matched),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "tasks": sorted({str(r.get("task")) for r in matched}),
        }

    no_support_result: Dict[str, Any] = {
        "version": 1,
        "policy": "query_reformulation_action_classifier",
        "feature_space": "attribution_pool",
        "deployment_mode": deployment_mode,
        "runtime_compatible": runtime_compatible,
        "gate": None,
        "accepted": [],
        "rejected_count": 0,
        "criteria": criteria,
        "training_summary": {
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "positive_action": positive_action,
            "positive_examples": sum(1 for r in train_rows if str(r.get("action", "")) == positive_action),
        },
    }

    if model is None:
        return no_support_result
    if model["positive_count"] < min_train_support:
        no_support_result["training_summary"]["positive_examples"] = model["positive_count"]
        no_support_result["training_summary"]["negative_examples"] = model["negative_count"]
        return no_support_result

    train_scores = [
        _classifier_score(r, {"threshold": 0.0, **model}) for r in train_rows
    ]
    thresholds = sorted(set(np.quantile(train_scores, [0.5, 0.65, 0.8, 0.9]).tolist()), reverse=True)
    accepted = []
    rejected_count = 0
    for threshold in thresholds:
        gate: Dict[str, Any] = {
            "type": "linear_classifier",
            "operator": ">=",
            "score_transform": "sigmoid",
            "feature_space": "attribution_pool",
            "deployment_mode": deployment_mode,
            "runtime_compatible": runtime_compatible,
            "threshold": float(threshold),
            **model,
        }
        train_summary = _action_summary(train_rows, gate)
        holdout_summary = _action_summary(holdout_rows, gate)
        train_passed = (
            train_summary["matched"] >= min_train_support
            and train_summary["true_positives"] > 0
        )
        holdout_passed = (
            holdout_summary["matched"] >= min_holdout_support
            and holdout_summary["true_positives"] > 0
        )
        record = {
            "gate": gate,
            "train": train_summary,
            "holdout": holdout_summary,
            "accepted": train_passed and holdout_passed,
        }
        if record["accepted"]:
            accepted.append(record)
        else:
            rejected_count += 1

    accepted.sort(
        key=lambda item: (
            -item["holdout"]["true_positives"],
            -item["train"]["true_positives"],
            -item["holdout"]["matched"],
            -item["train"]["matched"],
        )
    )
    return {
        "version": 1,
        "policy": "query_reformulation_action_classifier",
        "feature_space": "attribution_pool",
        "deployment_mode": deployment_mode,
        "runtime_compatible": runtime_compatible,
        "gate": accepted[0]["gate"] if accepted else None,
        "accepted": accepted[:5],
        "rejected_count": rejected_count,
        "criteria": criteria,
        "training_summary": {
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "positive_action": positive_action,
            "positive_examples": model["positive_count"],
            "negative_examples": model["negative_count"],
        },
    }


def predict_single(
    feature_dict: Dict[str, Any],
    gate_config: Dict[str, Any],
) -> bool:
    """Return True if the gate predicts this query should be reformulated.

    Fail-closed: returns False when ``gate_config["gate"]`` is None or when
    the computed score is below the stored threshold.

    ``feature_dict`` must contain the same keys listed in
    ``gate_config["gate"]["features"]``; missing keys default to 0.0.
    """
    gate = gate_config.get("gate")
    if gate is None:
        return False
    score = _classifier_score(feature_dict, gate)
    return score >= float(gate["threshold"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a fail-closed learned reformulation gate from artifacts")
    parser.add_argument("--artifact", action="append")
    parser.add_argument("--from-pool", metavar="POOL_JSON", help="Train from attribution pool JSON instead of raw artifacts")
    parser.add_argument("--profile", default="reform_rrf_v2")
    parser.add_argument("--positive-action", default="REFORMULATE", help="Action label treated as positive (pool mode only)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout-modulo", type=int, default=5)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument("--min-train-support", type=int, default=2)
    parser.add_argument("--min-holdout-support", type=int, default=1)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--max-negative-examples", type=int, default=0)
    parser.add_argument("--positive-delta", type=float, default=0.001)
    parser.add_argument("--min-positive-examples", type=int, default=2)
    parser.add_argument("--l2-penalty", type=float, default=0.08)
    args = parser.parse_args()

    if args.from_pool:
        pool = load_attribution_pool(args.from_pool)
        config = train_gate_from_pool(
            pool,
            positive_action=args.positive_action,
            holdout_modulo=args.holdout_modulo,
            holdout_remainder=args.holdout_remainder,
            min_train_support=args.min_train_support,
            min_holdout_support=args.min_holdout_support,
            l2_penalty=args.l2_penalty,
        )
        config["pool_source"] = args.from_pool
        config["positive_action"] = args.positive_action
        write_gate_config(config, args.output)
        print(json.dumps({
            "output": args.output,
            "mode": "attribution_pool",
            "positive_action": args.positive_action,
            "pool_rows": config["training_summary"]["train_rows"] + config["training_summary"]["holdout_rows"],
            "gate_present": config["gate"] is not None,
            "accepted": len(config["accepted"]),
            "rejected_count": config["rejected_count"],
        }, indent=2))
        return 0

    if not args.artifact:
        parser.error("one of --artifact or --from-pool is required")
    rows = enrich_query_rows(load_query_attribution_rows(args.artifact))
    filtered = filter_reformulation_rows(rows, profile=args.profile)
    config = train_reformulation_gate(
        filtered,
        holdout_modulo=args.holdout_modulo,
        holdout_remainder=args.holdout_remainder,
        min_train_support=args.min_train_support,
        min_holdout_support=args.min_holdout_support,
        min_mean_delta=args.min_mean_delta,
        max_negative_examples=args.max_negative_examples,
        positive_delta=args.positive_delta,
        min_positive_examples=args.min_positive_examples,
        l2_penalty=args.l2_penalty,
    )
    config["profile"] = args.profile
    config["artifact_count"] = len(_iter_artifact_paths(args.artifact))
    config["pooled_rows"] = len(filtered)
    write_gate_config(config, args.output)
    print(json.dumps({
        "output": args.output,
        "profile": args.profile,
        "artifact_count": config["artifact_count"],
        "pooled_rows": config["pooled_rows"],
        "gate_present": config["gate"] is not None,
        "accepted": len(config["accepted"]),
        "rejected_count": config["rejected_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
