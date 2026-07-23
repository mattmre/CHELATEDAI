"""Offline sequential runner for the lean Regime-C D2 kill-screen."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from research.drift_recovery.artifacts import D2CellPack, EmbeddingPack  # noqa: E402
from research.drift_recovery.d2.decisions import (  # noqa: E402
    FROZEN_CONTRASTS,
    apply_kill_screen,
    evaluate_g2,
)
from research.drift_recovery.d2.detector_evaluation import (  # noqa: E402
    average_precision,
    cluster_counterfactual_harm,
    cluster_geometry_features,
    evaluate_detector,
)
from research.drift_recovery.harness_bridge import (  # noqa: E402
    aggregate_ndcg,
    assert_harness_parity,
    assert_score_transform_parity,
    encode_clean_retrieval_case,
    load_retrieval_evalsplit,
    per_query_ndcg,
    per_query_ndcg_from_rankings,
)
from research.drift_recovery.methods import (  # noqa: E402
    AffineRidgeAdapter,
    AllButTopAdapter,
    CBIEAdapter,
    DetectorGatedChelationAdapter,
    HubnessScoreScaling,
    SoftRoutedLocalAdapter,
    ZCAAdapter,
)
from research.drift_recovery.regimes.synthetic_collapse import (  # noqa: E402
    deterministic_kmeans,
    inject_semantic_collapse,
)
from research.drift_recovery.stats.multiple_testing import holm_adjust  # noqa: E402
from research.drift_recovery.stats.paired_bootstrap import BootstrapDraws  # noqa: E402


METHODS = (
    "global_ridge",
    "paired_local_adapter",
    "cbie",
    "hubness",
    "zca",
    "all_but_top",
    "chelation_primary",
    "chelation_alpha_0_05",
)
VECTOR_METHODS = tuple(method for method in METHODS if method != "hubness")
CONTRAST_BASELINES = {
    "chelation_vs_cbie": "cbie",
    "chelation_vs_hubness": "hubness",
    "chelation_vs_global_ridge": "global_ridge",
}


class D2AvailabilityError(RuntimeError):
    """The frozen offline dataset/model cache is unavailable."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def load_protocol(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol_name") != "lean-d2-regime-c-crossover":
        raise ValueError("unexpected or malformed D2 protocol")
    return payload


def _positive_document_ids(*qrel_partitions: Mapping[str, Mapping[str, float]]) -> set:
    return {
        str(doc_id)
        for qrels in qrel_partitions
        for relevance in qrels.values()
        for doc_id, score in relevance.items()
        if float(score) > 0.0
    }


def build_leakage_safe_anchor_indices(
    doc_ids: Sequence[str],
    validation_qrels: Mapping[str, Mapping[str, float]],
    eval_qrels: Mapping[str, Mapping[str, float]],
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Freeze paired-fit rows after excluding every known positive document."""

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("anchor fraction must be in (0, 1]")
    positives = _positive_document_ids(validation_qrels, eval_qrels)
    safe = np.asarray(
        [index for index, doc_id in enumerate(doc_ids) if str(doc_id) not in positives],
        dtype=np.int64,
    )
    if len(safe) < 2:
        raise ValueError("fewer than two leakage-safe anchor documents remain")
    count = max(2, int(math.floor(float(fraction) * len(safe))))
    order = np.random.default_rng(int(seed) ^ 0xA11CE).permutation(len(safe))
    selected = np.sort(safe[order[:count]])
    if any(str(doc_ids[index]) in positives for index in selected):
        raise AssertionError("qrel-positive document leaked into the fit pool")
    return selected


def audit_anchor_indices(
    indices: np.ndarray,
    doc_ids: Sequence[str],
    validation_qrels: Mapping[str, Mapping[str, float]],
    eval_qrels: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Any]:
    positives = _positive_document_ids(validation_qrels, eval_qrels)
    contaminated = [str(doc_ids[index]) for index in np.asarray(indices, dtype=np.int64) if str(doc_ids[index]) in positives]
    return {
        "fit_document_count": int(len(indices)),
        "known_positive_document_count": int(len(positives)),
        "positive_documents_in_fit": int(len(contaminated)),
        "contaminated_ids": contaminated[:10],
    }


def cosine_score_matrix(query_vectors: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    queries = np.asarray(query_vectors, dtype=np.float64)
    documents = np.asarray(document_vectors, dtype=np.float64)
    if queries.ndim != 2 or documents.ndim != 2 or queries.shape[1] != documents.shape[1]:
        raise ValueError("query/document matrices must share a dimension")
    query_norm = np.linalg.norm(queries, axis=1, keepdims=True)
    document_norm = np.linalg.norm(documents, axis=1, keepdims=True)
    normalized_queries = queries / np.where(query_norm > 0.0, query_norm, 1.0)
    normalized_documents = documents / np.where(document_norm > 0.0, document_norm, 1.0)
    return normalized_queries @ normalized_documents.T


def build_detector_assignments(
    corrupted_documents: np.ndarray,
    cluster_count: int,
    seed: int,
) -> np.ndarray:
    """Cluster only corrupted vectors for detector features and routing."""

    assignments, _centroids = deterministic_kmeans(
        np.asarray(corrupted_documents, dtype=np.float64),
        cluster_count=int(cluster_count),
        seed=int(seed) ^ 0xD37EC7,
        iterations=25,
    )
    return assignments


def _score_documents(
    documents: np.ndarray,
    queries: np.ndarray,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    k: int,
) -> np.ndarray:
    return per_query_ndcg(queries, query_ids, documents, doc_ids, qrels, k=k)


def _validation_score(
    documents: np.ndarray,
    queries: np.ndarray,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    k: int,
) -> float:
    return aggregate_ndcg(per_query_ndcg(queries, query_ids, documents, doc_ids, qrels, k=k))


def _select_vector_candidate(
    candidates: Sequence[float],
    build_documents: Callable[[float], Tuple[Any, np.ndarray]],
    validation_queries: np.ndarray,
    validation_query_ids: Sequence[str],
    doc_ids: Sequence[str],
    validation_qrels: Mapping[str, Mapping[str, float]],
    k: int,
) -> Tuple[float, Any, np.ndarray, float]:
    best: Optional[Tuple[float, float, Any, np.ndarray]] = None
    for value in sorted(map(float, candidates)):
        adapter, documents = build_documents(value)
        score = _validation_score(
            documents,
            validation_queries,
            validation_query_ids,
            doc_ids,
            validation_qrels,
            k,
        )
        if best is None or score > best[0] + 1e-15:
            best = (score, value, adapter, documents)
    if best is None:
        raise ValueError("candidate grid cannot be empty")
    return best[1], best[2], best[3], best[0]


def _run_seed(
    *,
    case: Mapping[str, Any],
    encoded: Mapping[str, Any],
    protocol: Mapping[str, Any],
    seed: int,
    collapse_cluster_count: int,
) -> Mapping[str, Any]:
    clean = np.asarray(encoded["documents"], dtype=np.float64)
    validation_queries = np.asarray(encoded["validation_queries"], dtype=np.float64)
    eval_queries = np.asarray(encoded["eval_queries"], dtype=np.float64)
    doc_ids = list(case["doc_ids"])
    validation_query_ids = list(case["validation_query_ids"])
    eval_query_ids = list(case["eval_query_ids"])
    validation_qrels = case["validation_qrels"]
    eval_qrels = case["eval_qrels"]
    retrieval = protocol["retrieval"]
    scope = protocol["scope"]
    methods_protocol = protocol["methods"]
    detector_protocol = protocol["detector"]
    k = int(retrieval["ndcg_k"])
    total_clusters = int(scope["total_clusters"])

    collapse = inject_semantic_collapse(
        clean,
        collapse_cluster_count=int(collapse_cluster_count),
        beta=float(scope["beta"]),
        seed=int(seed),
        total_clusters=total_clusters,
    )
    corrupted = collapse.corrupted
    detector_assignments = build_detector_assignments(
        corrupted, cluster_count=total_clusters, seed=int(seed)
    )
    fit_indices = build_leakage_safe_anchor_indices(
        doc_ids,
        validation_qrels,
        eval_qrels,
        fraction=float(protocol["leakage"]["fit_anchor_fraction_of_safe_documents"]),
        seed=int(seed),
    )
    leakage_audit = audit_anchor_indices(
        fit_indices, doc_ids, validation_qrels, eval_qrels
    )
    if leakage_audit["positive_documents_in_fit"] != 0:
        raise AssertionError("leakage guard failed")

    features = cluster_geometry_features(
        corrupted, detector_assignments, cluster_count=total_clusters
    )
    validation_harm = cluster_counterfactual_harm(
        clean,
        corrupted,
        detector_assignments,
        validation_queries,
        validation_query_ids,
        doc_ids,
        validation_qrels,
        k=k,
        cluster_count=total_clusters,
    )
    eval_harm = cluster_counterfactual_harm(
        clean,
        corrupted,
        detector_assignments,
        eval_queries,
        eval_query_ids,
        doc_ids,
        eval_qrels,
        k=k,
        cluster_count=total_clusters,
    )
    detector = evaluate_detector(
        features=features,
        validation_harm_losses=validation_harm,
        eval_harm_losses=eval_harm,
        validation_query_ids=validation_query_ids,
        eval_query_ids=eval_query_ids,
        gate_candidates=detector_protocol["gate_threshold_candidates"],
        harm_threshold=float(detector_protocol["harm_threshold"]),
    )
    probabilities = np.asarray(detector["probabilities"], dtype=np.float64)
    gate = float(detector["selected_gate_threshold"])

    dimension = clean.shape[1]
    source_fit = corrupted[fit_indices]
    target_fit = clean[fit_indices]
    shared = methods_protocol["shared_adapter_budget"]
    global_rank = int(shared["global_rank"])
    local_rank = int(shared["local_rank"])
    routing_temperature = float(shared["routing_temperature"])
    selected_hyperparameters: Dict[str, Any] = {"detector_gate_threshold": gate}
    validation_scores: Dict[str, float] = {}
    method_documents: Dict[str, np.ndarray] = {}

    ridge_grid = methods_protocol["global_ridge"]["regularization_candidates"]

    def build_ridge(regularization: float) -> Tuple[Any, np.ndarray]:
        adapter = AffineRidgeAdapter(regularization=regularization).fit(source_fit, target_fit)
        return adapter, adapter.transform(corrupted)

    ridge_value, _ridge, ridge_documents, ridge_validation = _select_vector_candidate(
        ridge_grid,
        build_ridge,
        validation_queries,
        validation_query_ids,
        doc_ids,
        validation_qrels,
        k,
    )
    selected_hyperparameters["global_ridge_regularization"] = ridge_value
    validation_scores["global_ridge"] = ridge_validation
    method_documents["global_ridge"] = ridge_documents

    local_grid = methods_protocol["paired_local_adapter"]["regularization_candidates"]

    def build_local(regularization: float) -> Tuple[Any, np.ndarray]:
        adapter = SoftRoutedLocalAdapter(
            dimension=dimension,
            global_rank=global_rank,
            local_rank=local_rank,
            cluster_count=total_clusters,
            regularization=regularization,
            routing_temperature=routing_temperature,
            kmeans_iterations=25,
        ).fit(source_fit, target_fit)
        return adapter, adapter.transform(corrupted)

    local_value, local_adapter, local_documents, local_validation = _select_vector_candidate(
        local_grid,
        build_local,
        validation_queries,
        validation_query_ids,
        doc_ids,
        validation_qrels,
        k,
    )
    selected_hyperparameters["paired_local_regularization"] = local_value
    validation_scores["paired_local_adapter"] = local_validation
    method_documents["paired_local_adapter"] = local_documents

    cbie_config = methods_protocol["cbie"]
    cbie = CBIEAdapter(
        cluster_count=int(cbie_config["cluster_count"]),
        epsilon=float(cbie_config["epsilon"]),
        routing_temperature=float(cbie_config["routing_temperature"]),
        kmeans_iterations=25,
    ).fit(corrupted)
    method_documents["cbie"] = cbie.transform(corrupted)
    validation_scores["cbie"] = _validation_score(
        method_documents["cbie"], validation_queries, validation_query_ids, doc_ids, validation_qrels, k
    )

    zca = ZCAAdapter(epsilon=float(methods_protocol["zca"]["epsilon"])).fit(corrupted)
    method_documents["zca"] = zca.transform(corrupted)
    validation_scores["zca"] = _validation_score(
        method_documents["zca"], validation_queries, validation_query_ids, doc_ids, validation_qrels, k
    )
    all_but_top = AllButTopAdapter(
        components=int(methods_protocol["all_but_top"]["component_count"])
    ).fit(corrupted)
    method_documents["all_but_top"] = all_but_top.transform(corrupted)
    validation_scores["all_but_top"] = _validation_score(
        method_documents["all_but_top"], validation_queries, validation_query_ids, doc_ids, validation_qrels, k
    )

    gain_grid = methods_protocol["chelation_primary"]["radial_gain_candidates"]

    def build_chelation(radial_gain: float) -> Tuple[Any, np.ndarray]:
        adapter = DetectorGatedChelationAdapter(
            dimension=dimension,
            global_rank=global_rank,
            local_rank=local_rank,
            cluster_count=total_clusters,
            alpha=None,
            radial_gain=radial_gain,
            detector_threshold=gate,
            kmeans_iterations=25,
        ).fit(
            corrupted,
            cluster_assignments=detector_assignments,
            detector_probabilities=probabilities,
        )
        return adapter, adapter.transform(corrupted)

    gain, chelation, chelation_documents, chelation_validation = _select_vector_candidate(
        gain_grid,
        build_chelation,
        validation_queries,
        validation_query_ids,
        doc_ids,
        validation_qrels,
        k,
    )
    selected_hyperparameters["chelation_primary_radial_gain"] = gain
    validation_scores["chelation_primary"] = chelation_validation
    method_documents["chelation_primary"] = chelation_documents
    validation_positive_clusters = int(
        np.count_nonzero(validation_harm > float(detector_protocol["harm_threshold"]))
    )
    if validation_positive_clusters > 0 and float(
        (chelation.last_diagnostics_ or {}).get("moved_fraction", 0.0)
    ) <= 0.0:
        raise AssertionError(
            "primary chelation did not move any documents despite positive validation harm"
        )
    bounded = DetectorGatedChelationAdapter(
        dimension=dimension,
        global_rank=global_rank,
        local_rank=local_rank,
        cluster_count=total_clusters,
        alpha=float(methods_protocol["chelation_alpha_0_05"]["alpha"]),
        radial_gain=gain,
        detector_threshold=gate,
        kmeans_iterations=25,
    ).fit(
        corrupted,
        cluster_assignments=detector_assignments,
        detector_probabilities=probabilities,
    )
    method_documents["chelation_alpha_0_05"] = bounded.transform(corrupted)
    validation_scores["chelation_alpha_0_05"] = _validation_score(
        method_documents["chelation_alpha_0_05"],
        validation_queries,
        validation_query_ids,
        doc_ids,
        validation_qrels,
        k,
    )
    if local_adapter.allocated_parameter_count != chelation.allocated_parameter_count:
        raise AssertionError("paired local and primary chelation parameter budgets differ")

    validation_raw_scores = cosine_score_matrix(validation_queries, corrupted)
    eval_raw_scores = cosine_score_matrix(eval_queries, corrupted)
    hubness_grid = methods_protocol["hubness"]["strength_candidates"]
    best_hubness = None
    for strength in sorted(map(float, hubness_grid)):
        candidate = HubnessScoreScaling(reference_k=k, strength=strength).fit(validation_raw_scores)
        validation_rankings = candidate.rank(validation_raw_scores, k=k)
        values = per_query_ndcg_from_rankings(
            validation_rankings, validation_query_ids, doc_ids, validation_qrels, k=k
        )
        score = aggregate_ndcg(values)
        if best_hubness is None or score > best_hubness[0] + 1e-15:
            best_hubness = (score, strength, candidate)
    assert best_hubness is not None
    validation_scores["hubness"] = float(best_hubness[0])
    selected_hyperparameters["hubness_strength"] = float(best_hubness[1])
    hubness_rankings = best_hubness[2].rank(eval_raw_scores, k=k)
    hubness_scores = per_query_ndcg_from_rankings(
        hubness_rankings, eval_query_ids, doc_ids, eval_qrels, k=k
    )
    scores: Dict[str, np.ndarray] = {}
    scores["floor"] = _score_documents(
        corrupted, eval_queries, eval_query_ids, doc_ids, eval_qrels, k
    )
    scores["oracle"] = _score_documents(
        clean, eval_queries, eval_query_ids, doc_ids, eval_qrels, k
    )
    for method in VECTOR_METHODS:
        scores[method] = _score_documents(
            method_documents[method], eval_queries, eval_query_ids, doc_ids, eval_qrels, k
        )
    scores["hubness"] = hubness_scores
    vector_reference_names = ("floor", "oracle") + VECTOR_METHODS
    parity_pack = EmbeddingPack(
        Do=corrupted,
        Dor=clean,
        Qd=eval_queries,
        doc_ids=np.asarray(doc_ids, dtype=str),
        query_ids=np.asarray(eval_query_ids, dtype=str),
        qrels=eval_qrels,
        fit_idx=np.asarray(fit_indices, dtype=np.int64),
        per_query_scores={name: scores[name] for name in vector_reference_names},
        extra_arrays={
            f"method_documents__{method}": method_documents[method]
            for method in VECTOR_METHODS
        },
        metadata={
            "k": k,
            "harness_aggregate_ndcg": {
                name: aggregate_ndcg(scores[name]) for name in vector_reference_names
            },
        },
    )
    parity_pack.validate()
    parity = assert_harness_parity(parity_pack, atol=1e-12)
    score_parity = assert_score_transform_parity(
        hubness_rankings,
        hubness_scores,
        aggregate_ndcg(hubness_scores),
        eval_query_ids,
        doc_ids,
        eval_qrels,
        k=k,
        atol=1e-12,
    )
    parity["hubness_per_query"] = score_parity["per_query_max_abs"]
    parity["hubness_aggregate"] = score_parity["aggregate_abs"]

    oracle_gap = float(scores["oracle"].mean() - scores["floor"].mean())
    return {
        "record_type": "d2_atomic_seed",
        "seed": int(seed),
        "collapse_cluster_count": int(collapse_cluster_count),
        "scores": scores,
        "oracle_gap": oracle_gap,
        "detector": detector,
        "selected_hyperparameters": selected_hyperparameters,
        "validation_ndcg": validation_scores,
        "leakage_audit": leakage_audit,
        "parameter_budget": {
            "paired_local_allocated_low_rank_scalars": int(
                local_adapter.allocated_parameter_count
            ),
            "chelation_allocated_low_rank_scalars": int(
                chelation.allocated_parameter_count
            ),
            "allocated_low_rank_scalar_budget_matched": True,
            "full_pipeline_free_parameter_match_claimed": False,
            "not_counted": [
                "routing centroids",
                "chelation cluster probabilities",
                "harm scorer coefficients",
                "tied PCA factor degrees of freedom",
            ],
        },
        "displacement": {
            "corruption_mean": collapse.mean_displacement,
            "chelation_primary": dict(chelation.last_diagnostics_ or {}),
            "chelation_alpha_0_05": dict(bounded.last_diagnostics_ or {}),
        },
        "harness_parity_max_abs": parity,
        "transformed_document_count": {method: len(corrupted) for method in VECTOR_METHODS},
        "collapsed_clusters": collapse.collapsed_clusters,
    }


def _metric(estimate: Optional[float], samples: np.ndarray) -> Mapping[str, Any]:
    if estimate is None or len(samples) == 0:
        return {"estimate": estimate, "ci_low": None, "ci_high": None, "half_width": None}
    low, high = np.quantile(np.asarray(samples, dtype=np.float64), [0.025, 0.975])
    return {
        "estimate": float(estimate),
        "ci_low": float(low),
        "ci_high": float(high),
        "half_width": float((high - low) / 2.0),
    }


def _sign_p_value(samples: np.ndarray) -> float:
    non_positive = int(np.count_nonzero(samples <= 0.0))
    non_negative = int(np.count_nonzero(samples >= 0.0))
    return float(min(1.0, 2.0 * (min(non_positive, non_negative) + 1.0) / (len(samples) + 1.0)))


def bootstrap_cell(
    pack: D2CellPack,
    draws: int,
    seed: int,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Bootstrap averaged queries while retaining five seed signs separately."""

    pack.validate()
    averaged = {
        method: np.asarray(values, dtype=np.float64).mean(axis=0)
        for method, values in pack.per_seed_scores.items()
    }
    paired = BootstrapDraws.create(len(pack.query_ids), draws=int(draws), seed=int(seed))
    indices = paired.indices
    resampled = {method: values[indices].mean(axis=1) for method, values in averaged.items()}
    floor_draw = resampled["floor"]
    oracle_draw = resampled["oracle"]
    gap_draw = oracle_draw - floor_draw
    valid_gap = gap_draw > 1e-12
    invalid_fraction = float(np.mean(~valid_gap))
    point_gap = float(averaged["oracle"].mean() - averaged["floor"].mean())
    methods = {}
    for method in METHODS:
        ndcg_point = float(averaged[method].mean())
        delta_point = ndcg_point - float(averaged["floor"].mean())
        recovery_valid = invalid_fraction <= 0.01 and point_gap > 1e-12
        recovery_point = delta_point / point_gap if recovery_valid else None
        recovery_samples = (
            (resampled[method][valid_gap] - floor_draw[valid_gap]) / gap_draw[valid_gap]
            if recovery_valid
            else np.asarray([], dtype=np.float64)
        )
        recovery_metric = dict(_metric(recovery_point, recovery_samples))
        recovery_metric["status"] = (
            "ok" if recovery_valid else "blocked_invalid_or_nonpositive_oracle_gap"
        )
        methods[method] = {
            "ndcg": _metric(ndcg_point, resampled[method]),
            "delta_ndcg": _metric(delta_point, resampled[method] - floor_draw),
            "recovery": recovery_metric,
        }
    contrasts: Dict[str, Any] = {}
    for name, baseline in CONTRAST_BASELINES.items():
        delta_samples = resampled["chelation_primary"] - resampled[baseline]
        delta_point = float(averaged["chelation_primary"].mean() - averaged[baseline].mean())
        recovery_valid = invalid_fraction <= 0.01 and point_gap > 1e-12
        if recovery_valid:
            recovery_point = delta_point / point_gap
        else:
            recovery_point = None
        recovery_samples = (
            delta_samples[valid_gap] / gap_draw[valid_gap]
            if recovery_valid
            else np.asarray([], dtype=np.float64)
        )
        recovery_metric = dict(_metric(recovery_point, recovery_samples))
        recovery_metric["status"] = (
            "ok" if recovery_valid else "blocked_invalid_or_nonpositive_oracle_gap"
        )
        seed_delta = (
            np.asarray(pack.per_seed_scores["chelation_primary"]).mean(axis=1)
            - np.asarray(pack.per_seed_scores[baseline]).mean(axis=1)
        )
        seed_gap = (
            np.asarray(pack.per_seed_scores["oracle"]).mean(axis=1)
            - np.asarray(pack.per_seed_scores["floor"]).mean(axis=1)
        )
        contrasts[name] = {
            "left": "chelation_primary",
            "right": baseline,
            "delta_ndcg": _metric(delta_point, delta_samples),
            "recovery_point_advantage": recovery_metric,
            "p_value": _sign_p_value(delta_samples),
            "seed_delta_ndcg": seed_delta.tolist(),
            "seed_recovery_advantage": [
                float(delta / gap) if gap > 1e-12 else None
                for delta, gap in zip(seed_delta, seed_gap)
            ],
        }
    holm = holm_adjust({name: row["p_value"] for name, row in contrasts.items()})
    for name, row in contrasts.items():
        row["multiple_testing"] = holm[name]
        row["half_width"] = row["delta_ndcg"]["half_width"]
    bootstrap = {
        "record_type": "d2_grouped_paired_query_bootstrap",
        "seed_aggregation": "mean_each_query_across_seeds",
        "seed_count": len(pack.seed_ids),
        "query_count": len(pack.query_ids),
        "draws": int(draws),
        "seed": int(seed),
        "floor_ndcg": float(averaged["floor"].mean()),
        "oracle_ndcg": float(averaged["oracle"].mean()),
        "oracle_gap": point_gap,
        "invalid_oracle_gap_draws": int(np.count_nonzero(~valid_gap)),
        "invalid_oracle_gap_fraction": invalid_fraction,
        "methods": methods,
    }
    return bootstrap, contrasts


def summarize_cell(
    *,
    dataset: str,
    collapse_cluster_count: int,
    seed_rows: Sequence[Mapping[str, Any]],
    query_ids: Sequence[str],
    protocol: Mapping[str, Any],
    output_dir: Path,
    run_fingerprint: Mapping[str, str],
    data_fingerprint: Mapping[str, Any],
) -> Mapping[str, Any]:
    stats_protocol = protocol["statistics"]
    detector_protocol = protocol["detector"]
    score_names = ("floor", "oracle") + METHODS
    per_seed_scores = {
        method: np.vstack([np.asarray(row["scores"][method], dtype=np.float64) for row in seed_rows])
        for method in score_names
    }
    probabilities = np.vstack(
        [np.asarray(row["detector"]["probabilities"], dtype=np.float64) for row in seed_rows]
    )
    eval_harm = np.vstack(
        [np.asarray(row["detector"]["eval_harm_losses"], dtype=np.float64) for row in seed_rows]
    )
    pack = D2CellPack(
        seed_ids=[int(row["seed"]) for row in seed_rows],
        query_ids=list(query_ids),
        per_seed_scores=per_seed_scores,
        detector_probabilities=probabilities,
        detector_harm_losses=eval_harm,
        metadata={
            "dataset": dataset,
            "collapse_cluster_count": int(collapse_cluster_count),
            "beta": float(protocol["scope"]["beta"]),
            "method_labels": protocol["methods"],
            "run_fingerprint": dict(run_fingerprint),
            "data_fingerprint": dict(data_fingerprint),
        },
    )
    pack_paths = pack.save(output_dir / "cell_pack")
    reloaded = D2CellPack.load(output_dir / "cell_pack")
    for method in score_names:
        if not np.array_equal(reloaded.per_seed_scores[method], pack.per_seed_scores[method]):
            raise AssertionError("D2 cell pack score reload changed values")

    bootstrap, contrasts = bootstrap_cell(
        reloaded,
        draws=int(stats_protocol["bootstrap_draws"]),
        seed=int(stats_protocol["bootstrap_seed"]),
    )
    g2 = evaluate_g2(
        contrasts,
        query_count=len(query_ids),
        threshold=float(stats_protocol["g2_median_half_width_threshold"]),
        family=FROZEN_CONTRASTS,
    )
    aggregate_auprc, aggregate_auprc_status = average_precision(
        reloaded.detector_harm_losses.reshape(-1),
        reloaded.detector_probabilities.reshape(-1),
        harm_threshold=float(detector_protocol["harm_threshold"]),
    )
    min_oracle_gap = float(min(float(row["oracle_gap"]) for row in seed_rows))
    primary_movement_valid = any(
        float(row["displacement"]["chelation_primary"].get("moved_fraction", 0.0)) > 0.0
        for row in seed_rows
    )
    decision = apply_kill_screen(
        g2=g2,
        contrasts=contrasts,
        detector_auprc=aggregate_auprc,
        min_oracle_gap=min_oracle_gap,
        seed_count=int(protocol["g3"]["required_seed_count"]),
        auprc_threshold=float(protocol["g3"]["require_detector_auprc"]),
        oracle_gap_threshold=float(protocol["g3"]["minimum_oracle_gap_each_seed"]),
        invalid_oracle_gap_fraction=float(bootstrap["invalid_oracle_gap_fraction"]),
        invalid_oracle_gap_limit=float(
            stats_protocol["invalid_oracle_gap_draw_fraction_limit"]
        ),
        primary_movement_valid=primary_movement_valid,
        family=FROZEN_CONTRASTS,
    )
    detector_summary = {
        "record_type": "d2_cell_detector",
        "dataset": dataset,
        "collapse_cluster_count": int(collapse_cluster_count),
        "auprc": aggregate_auprc,
        "auprc_status": aggregate_auprc_status,
        "positive_clusters": int(
            np.count_nonzero(
                reloaded.detector_harm_losses
                > float(detector_protocol["harm_threshold"])
            )
        ),
        "cluster_predictions": int(reloaded.detector_harm_losses.size),
        "per_seed": [
            {
                "seed": int(row["seed"]),
                "auprc": row["detector"]["auprc"],
                "auprc_status": row["detector"]["auprc_status"],
                "eval_positive_clusters": row["detector"]["eval_positive_clusters"],
                "selected_gate_threshold": row["detector"]["selected_gate_threshold"],
            }
            for row in seed_rows
        ],
    }
    cell_result = {
        "record_type": "d2_decision_cell",
        "run_fingerprint": dict(run_fingerprint),
        "data_fingerprint": dict(data_fingerprint),
        "dataset": dataset,
        "collapse_cluster_count": int(collapse_cluster_count),
        "seeds": [int(row["seed"]) for row in seed_rows],
        "min_oracle_gap": min_oracle_gap,
        "bootstrap_invalid_oracle_gap_fraction": float(
            bootstrap["invalid_oracle_gap_fraction"]
        ),
        "primary_movement_valid": primary_movement_valid,
        "g2": g2,
        "contrasts": contrasts,
        "detector": detector_summary,
        "decision": decision,
        "pack": pack_paths,
        "leakage_audits": [row["leakage_audit"] for row in seed_rows],
        "parameter_budgets": [row["parameter_budget"] for row in seed_rows],
        "displacement": [row["displacement"] for row in seed_rows],
        "selected_hyperparameters": [row["selected_hyperparameters"] for row in seed_rows],
        "harness_parity_max_abs": float(
            max(max(row["harness_parity_max_abs"].values()) for row in seed_rows)
        ),
    }
    _write_json(output_dir / "bootstrap_cis.json", {"bootstrap": bootstrap, "contrasts": contrasts, "g2": g2})
    _write_json(output_dir / "detector_auprc.json", detector_summary)
    _write_json(output_dir / "decision.json", cell_result)
    return cell_result


def _format_optional(value: Any, digits: int = 3) -> str:
    return "undefined" if value is None else f"{float(value):.{digits}f}"


def render_decision_markdown(cells: Sequence[Mapping[str, Any]], overall: str) -> str:
    lines = [
        "# D2 decision",
        "",
        f"Overall: **{overall}**.",
        "",
        "G2 is applied before G3. `NO_G3_VERDICT` is not a corrector kill claim.",
        "",
        "| Cell | G2 half-width | Invalid gap draws | Primary moved fraction | Harm AUPRC | G3 verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for cell in cells:
        lines.append(
            "| "
            f"{cell['dataset']} / {cell['collapse_cluster_count']} clusters | "
            f"{cell['g2']['median_half_width']:.4f} | "
            f"{cell['bootstrap_invalid_oracle_gap_fraction']:.2%} | "
            f"{np.mean([row['chelation_primary'].get('moved_fraction', 0.0) for row in cell['displacement']]):.3f} | "
            f"{_format_optional(cell['detector']['auprc'])} | "
            f"{cell['decision']['verdict']} |"
        )
    lines.extend(["", "## Cell evidence", ""])
    for cell in cells:
        lines.append(
            f"- {cell['dataset']} / {cell['collapse_cluster_count']}: minimum oracle gap "
            f"{cell['min_oracle_gap']:.4f}; decision `{cell['decision']['verdict']}`."
        )
        for contrast_name in FROZEN_CONTRASTS:
            contrast = cell["contrasts"][contrast_name]
            ndcg = contrast["delta_ndcg"]
            recovery = contrast["recovery_point_advantage"]
            lines.append(
                f"  - {contrast_name}: ΔNDCG {ndcg['estimate']:.4f} "
                f"(95% CI {ndcg['ci_low']:.4f}, {ndcg['ci_high']:.4f}); "
                f"recovery advantage {_format_optional(recovery['estimate'])} "
                f"(95% CI {_format_optional(recovery['ci_low'])}, "
                f"{_format_optional(recovery['ci_high'])}); seed Δ signs "
                f"{[float(value) for value in contrast['seed_delta_ndcg']]}; "
                f"Holm reject={contrast['multiple_testing']['reject']}."
            )
        if not cell["decision"]["g3_interpreted"]:
            suffix = (
                f" Rough G2 query budget: {cell['g2']['rough_query_count_for_threshold']}."
                if not cell["g2"]["pass"]
                else ""
            )
            lines.append(
                f"  G3 was not interpreted: {cell['decision']['reason']}.{suffix}"
            )
    return "\n".join(lines) + "\n"


def render_report(cells: Sequence[Mapping[str, Any]], overall: str) -> str:
    lines = [
        "# D2 kill-screen report",
        "",
        f"**Verdict:** {overall}.",
        "",
        "The only judged correction is `chelation_primary` (unbounded displacement, detector-gated, unpaired). "
        "The α=0.05 arm is a separate design-premise stress test; the paired local adapter is diagnostic only.",
        "",
        "| Cell | G2 / gap validity | AUPRC | Primary movement | Judged result |",
        "|---|---|---:|---:|---|",
    ]
    for cell in cells:
        g2 = cell["g2"]
        lines.append(
            f"| {cell['dataset']} / {cell['collapse_cluster_count']} | "
            f"{'PASS' if g2['pass'] else 'FAIL'} ({g2['median_half_width']:.4f}) / "
            f"{cell['bootstrap_invalid_oracle_gap_fraction']:.2%} invalid | "
            f"{_format_optional(cell['detector']['auprc'])} | "
            f"{np.mean([row['chelation_primary'].get('moved_fraction', 0.0) for row in cell['displacement']]):.3f} | "
            f"{cell['decision']['verdict']} |"
        )
        if cell["decision"]["win"]:
            for contrast_name in FROZEN_CONTRASTS:
                contrast = cell["contrasts"][contrast_name]
                ndcg = contrast["delta_ndcg"]
                recovery = contrast["recovery_point_advantage"]
                lines.append(
                    f"  - {contrast_name}: ΔNDCG {ndcg['estimate']:.4f} "
                    f"[{ndcg['ci_low']:.4f}, {ndcg['ci_high']:.4f}], recovery "
                    f"{recovery['estimate']:.3f} [{recovery['ci_low']:.3f}, "
                    f"{recovery['ci_high']:.3f}], seeds {contrast['seed_delta_ndcg']}."
                )
    caveat = (
        "Protocol v2 was locked only after an excluded SciFact pilot exposed invalid "
        "bootstrap, tied-AUPRC, gating, and routing implementations; the archived pilot is not evidence."
    )
    lines.extend(["", f"**Single most important caveat:** {caveat}"])
    return "\n".join(lines) + "\n"


def _repo_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_sha256(package_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in Path(package_root).rglob("*.py")
        if "tests" not in path.parts and "out" not in path.parts
    )
    for path in paths:
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _identity_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(map(str, values)), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _prepare_output_dir(output_dir: Path, force: bool) -> Optional[Path]:
    output_dir = Path(output_dir)
    allowed_root = Path("research/drift_recovery/out/d2").resolve()
    resolved_output = output_dir.resolve()
    if allowed_root != resolved_output:
        raise ValueError(
            f"D2 output must be exactly {allowed_root}; received {resolved_output}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"refusing to mix D2 artifacts in non-empty {output_dir}; rerun with --force to archive it"
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = output_dir.with_name(f"{output_dir.name}_invalidated_{stamp}")
        counter = 1
        while archive.exists():
            archive = output_dir.with_name(
                f"{output_dir.name}_invalidated_{stamp}_{counter}"
            )
            counter += 1
        shutil.move(str(output_dir), str(archive))
        output_dir.mkdir(parents=True, exist_ok=False)
        return archive
    output_dir.mkdir(parents=True, exist_ok=True)
    return None


def run_d2(
    *,
    output_dir: Path = Path("research/drift_recovery/out/d2"),
    protocol_path: Path = Path("research/drift_recovery/protocols/d2_crossover.yaml"),
    device: str = "cuda",
    command: str = "python research/drift_recovery/run_d2.py --device cuda",
    force: bool = False,
) -> Mapping[str, Any]:
    protocol = load_protocol(protocol_path)
    output_dir = Path(output_dir)
    workspace_state_before_run = subprocess.check_output(
        ["git", "status", "--short", "--", "research/drift_recovery"], text=True
    ).splitlines()
    archived_output = _prepare_output_dir(output_dir, force=bool(force))
    protocol_sha = _file_sha256(protocol_path)
    source_sha = _source_sha256(Path("research/drift_recovery"))
    harness_bridge_sha = _file_sha256(Path("research/drift_recovery/harness_bridge.py"))
    run_fingerprint = {
        "protocol_sha256": protocol_sha,
        "d2_source_sha256": source_sha,
        "harness_bridge_sha256": harness_bridge_sha,
        "repo_sha": _repo_sha(),
        "model": str(protocol["retrieval"]["model"]),
    }
    cells = []
    unavailable = []
    model_revisions: Dict[str, str] = {}
    for dataset in protocol["scope"]["datasets"]:
        try:
            sample_docs = int(
                protocol["retrieval"].get("dataset_sample_docs", {}).get(
                    dataset, protocol["retrieval"]["sample_docs"]
                )
            )
            case = load_retrieval_evalsplit(
                dataset,
                seed=42,
                anchor_fraction=float(protocol["retrieval"]["anchor_validation_fraction"]),
                max_queries=int(protocol["retrieval"]["max_queries"]),
                sample_docs=sample_docs,
            )
            encoded = encode_clean_retrieval_case(
                case,
                device=device,
                model_name=str(protocol["retrieval"]["model"]),
            )
            model_revisions[str(dataset)] = str(encoded.get("model_revision", "unknown"))
        except (FileNotFoundError, OSError) as error:
            unavailable.append({"dataset": dataset, "reason": f"{type(error).__name__}: {error}"})
            continue
        except RuntimeError as error:
            message = str(error).lower()
            if "offline" not in message and "unavailable" not in message and "not found in cache" not in message:
                raise
            unavailable.append({"dataset": dataset, "reason": f"{type(error).__name__}: {error}"})
            continue
        data_fingerprint = {
            "document_ids_sha256": _identity_sha256(case["doc_ids"]),
            "validation_query_ids_sha256": _identity_sha256(case["validation_query_ids"]),
            "eval_query_ids_sha256": _identity_sha256(case["eval_query_ids"]),
            "document_count": len(case["doc_ids"]),
            "validation_query_count": len(case["validation_query_ids"]),
            "eval_query_count": len(case["eval_query_ids"]),
        }
        cell_run_fingerprint = dict(run_fingerprint)
        cell_run_fingerprint["model_revision"] = model_revisions[str(dataset)]
        for collapse_count in protocol["scope"]["collapsed_cluster_counts"]:
            seed_rows = [
                _run_seed(
                    case=case,
                    encoded=encoded,
                    protocol=protocol,
                    seed=int(seed),
                    collapse_cluster_count=int(collapse_count),
                )
                for seed in protocol["scope"]["seeds"]
            ]
            cell_slug = f"{str(dataset).lower()}_c{int(collapse_count)}"
            cell_dir = output_dir / "cells" / cell_slug
            for row in seed_rows:
                seed_payload = {key: value for key, value in row.items() if key != "scores"}
                seed_payload["aggregate_ndcg"] = {
                    method: float(np.asarray(values).mean()) for method, values in row["scores"].items()
                }
                _write_json(cell_dir / f"seed_{row['seed']}.json", seed_payload)
            cells.append(
                summarize_cell(
                    dataset=str(dataset),
                    collapse_cluster_count=int(collapse_count),
                    seed_rows=seed_rows,
                    query_ids=case["eval_query_ids"],
                    protocol=protocol,
                    output_dir=cell_dir,
                    run_fingerprint=cell_run_fingerprint,
                    data_fingerprint=data_fingerprint,
                )
            )
    if unavailable or any(not cell["decision"]["g3_interpreted"] for cell in cells):
        overall = "NO_GLOBAL_G3_VERDICT_UNDERPOWERED_INVALID_OR_UNAVAILABLE"
    elif cells and all(cell["decision"]["win"] for cell in cells):
        overall = "SURPRISING_CHELATION_WIN_ALL_CELLS"
    elif cells:
        overall = "KILL_CORRECTOR_CLAIM"
    else:
        overall = "NO_RESULT_OFFLINE_INPUTS_UNAVAILABLE"
    decision_markdown = render_decision_markdown(cells, overall)
    report_markdown = render_report(cells, overall)
    (output_dir / "d2_decision.md").write_text(decision_markdown, encoding="utf-8")
    (output_dir / "D2_REPORT.md").write_text(report_markdown, encoding="utf-8")
    protocol_bytes = Path(protocol_path).read_bytes()
    manifest = {
        "record_type": "d2_run_manifest",
        "repo_sha": _repo_sha(),
        "harness_sha": harness_bridge_sha,
        "protocol_sha": hashlib.sha256(protocol_bytes).hexdigest(),
        "d2_source_sha256": source_sha,
        "run_fingerprint": run_fingerprint,
        "protocol_path": str(protocol_path),
        "command": command,
        "environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
        },
        "device": device,
        "cells_completed": len(cells),
        "archived_previous_output": str(archived_output) if archived_output else None,
        "model_revisions": model_revisions,
        "workspace_state_before_run": workspace_state_before_run,
        "datasets_unavailable": unavailable,
        "overall_verdict": overall,
        "outputs": sorted(
            {
                str(path.relative_to(output_dir))
                for path in output_dir.rglob("*")
                if path.is_file()
            }.union({"run_manifest.json", "d2_summary.json"})
        ),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    summary = {
        "record_type": "d2_summary",
        "overall_verdict": overall,
        "cells": cells,
        "datasets_unavailable": unavailable,
    }
    _write_json(output_dir / "d2_summary.json", summary)
    return summary
