"""Build a non-promotional BCC-1 pack from recovered frozen embeddings.

This builder deliberately accepts only the already-consumed D3/D1 embedding
packs.  They use sampled corpora and therefore cannot produce SELECT or REPORT
evidence.  Their purpose is to test whether bidirectional compatibility has
enough per-query complementarity to justify a new full-corpus experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "chelatedai.bcc1.pack.v1"
MANIFEST_SCHEMA_VERSION = "chelatedai.bcc1.manifest.v1"
RIDGE_REGULARIZATION = 1.0
TOP_K = 10
MIN_VECTOR_L2_NORM = 1e-12
REVERSE_ROUTE_ROLE_VALIDATION = "UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY"
METRIC_NAME = "ndcg_at_10"
METRIC_GAIN = "binary_positive_qrel"
METRIC_IDCG_POPULATION = "all_positive_qrels_for_query"
METRIC_QUERY_INCLUSION = "queries_with_at_least_one_positive_qrel"
METRIC_RANKING_TIE_BREAK = "score_desc_document_id_asc"
ANCHOR_STRATEGY: Mapping[str, Any] = {
    "strategy_id": "dataset-doc-id-sha256-half-corpus-v1",
    "selection_count": "ceil(document_count/2)",
    "stable_inputs": ["dataset_family_id", "doc_id"],
    "anchor_representation_role": "document",
    "reverse_application_role": "query",
    "reverse_role_validation": REVERSE_ROUTE_ROLE_VALIDATION,
    "qrels_used": False,
    "inherited_fit_idx_used": False,
}


def _feature(name: str, source: str, availability_stage: str) -> Mapping[str, Any]:
    if availability_stage == "PRE_SEARCH":
        execution_cost_class = "zero_search"
    elif availability_stage == "DUAL_READ_METHOD_DEV":
        execution_cost_class = "dual_search_probe"
    else:
        raise ValueError(f"unsupported feature availability stage: {availability_stage}")
    return {
        "name": name,
        "source": source,
        "availability_stage": availability_stage,
        "execution_cost_class": execution_cost_class,
    }


FEATURES: Tuple[Mapping[str, Any], ...] = (
    _feature("forward_top1", "forward resident-corpus cosine scores", "DUAL_READ_METHOD_DEV"),
    _feature(
        "forward_top1_margin",
        "forward resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "forward_top10_mean",
        "forward resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "forward_top10_std",
        "forward resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "forward_top10_entropy",
        "forward resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature("reverse_top1", "reverse resident-corpus cosine scores", "DUAL_READ_METHOD_DEV"),
    _feature(
        "reverse_top1_margin",
        "reverse resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "reverse_top10_mean",
        "reverse resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "reverse_top10_std",
        "reverse resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "reverse_top10_entropy",
        "reverse resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "top1_direction_gap",
        "forward and reverse resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "top10_mean_direction_gap",
        "forward and reverse resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "top10_jaccard",
        "forward and reverse resident-index candidate IDs",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "forward_choice_reverse_score",
        "reverse resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "reverse_choice_forward_score",
        "forward resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "forward_choice_reverse_regret",
        "reverse resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "reverse_choice_forward_regret",
        "forward resident-corpus cosine scores",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "query_cycle_l2",
        "qrels-independent frozen bridge matrices and query vector",
        "PRE_SEARCH",
    ),
    _feature(
        "reverse_query_norm_ratio",
        "qrels-independent frozen bridge matrix and query vector",
        "PRE_SEARCH",
    ),
    _feature(
        "new_anchor_max_cosine",
        "deterministic qrels-independent half-resident-corpus anchor vectors",
        "DUAL_READ_METHOD_DEV",
    ),
    _feature(
        "old_anchor_max_cosine",
        "deterministic qrels-independent half-resident-corpus anchor vectors",
        "DUAL_READ_METHOD_DEV",
    ),
)


@dataclass(frozen=True)
class FrozenEmbeddingPack:
    prefix: Path
    do: np.ndarray
    dor: np.ndarray
    qd: np.ndarray
    doc_ids: np.ndarray
    query_ids: np.ndarray
    qrels: Mapping[str, Mapping[str, float]]
    fit_idx: np.ndarray
    metadata: Mapping[str, Any]
    arrays_sha256: str
    metadata_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    matrix = _as_matrix(value, "embedding matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    invalid_rows = np.flatnonzero(norms[:, 0] <= MIN_VECTOR_L2_NORM)
    if invalid_rows.size:
        raise ValueError(
            "embedding matrix contains zero or near-zero L2 norm row(s): {}".format(invalid_rows.astype(int).tolist())
        )
    return matrix / norms


def _load_pack(prefix: Path) -> FrozenEmbeddingPack:
    prefix = Path(prefix)
    arrays_path = prefix.with_suffix(".npz")
    metadata_path = prefix.with_suffix(".json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_arrays_sha256 = _sha256_file(arrays_path)
    if actual_arrays_sha256 != payload.get("arrays_sha256"):
        raise ValueError(f"array checksum mismatch for {prefix}")
    with np.load(arrays_path, allow_pickle=False) as stored:
        required = {"Do", "Dor", "Qd", "doc_ids", "query_ids"}
        missing = required.difference(stored.files)
        if missing:
            raise ValueError(f"{prefix} is missing arrays: {sorted(missing)}")
        if "extra__leakage_safe_fit_idx" in stored.files:
            fit_idx = stored["extra__leakage_safe_fit_idx"].astype(np.int64).copy()
        elif "fit_idx" in stored.files:
            fit_idx = stored["fit_idx"].astype(np.int64).copy()
        else:
            raise ValueError(f"{prefix} has no fit index")
        pack = FrozenEmbeddingPack(
            prefix=prefix,
            do=_as_matrix(stored["Do"].copy(), "Do"),
            dor=_as_matrix(stored["Dor"].copy(), "Dor"),
            qd=_as_matrix(stored["Qd"].copy(), "Qd"),
            doc_ids=stored["doc_ids"].astype(str).copy(),
            query_ids=stored["query_ids"].astype(str).copy(),
            qrels=payload["qrels"],
            fit_idx=fit_idx,
            metadata=payload.get("metadata", {}),
            arrays_sha256=actual_arrays_sha256,
            metadata_sha256=_sha256_file(metadata_path),
        )
    _validate_pack(pack)
    return pack


def _validate_pack(pack: FrozenEmbeddingPack) -> None:
    if pack.do.shape != pack.dor.shape:
        raise ValueError(f"{pack.prefix}: Do and Dor shapes differ")
    if pack.qd.shape[1] != pack.do.shape[1]:
        raise ValueError(f"{pack.prefix}: Qd dimension differs from document dimension")
    if len(pack.doc_ids) != len(pack.do) or len(pack.query_ids) != len(pack.qd):
        raise ValueError(f"{pack.prefix}: embedding and ID counts differ")
    if pack.fit_idx.ndim != 1 or not len(pack.fit_idx):
        raise ValueError(f"{pack.prefix}: fit index must be a non-empty vector")
    if np.any(pack.fit_idx < 0) or np.any(pack.fit_idx >= len(pack.doc_ids)):
        raise ValueError(f"{pack.prefix}: fit index is out of bounds")
    if len(np.unique(pack.fit_idx)) != len(pack.fit_idx):
        raise ValueError(f"{pack.prefix}: fit index contains duplicates")
    if len(np.unique(pack.doc_ids.astype(str))) != len(pack.doc_ids):
        raise ValueError(f"{pack.prefix}: document IDs contain duplicates")
    if len(np.unique(pack.query_ids.astype(str))) != len(pack.query_ids):
        raise ValueError(f"{pack.prefix}: query IDs contain duplicates")
    document_ids = set(pack.doc_ids.astype(str))
    for query_id in pack.query_ids.astype(str):
        relevant_ids = _positive_relevant_ids(pack, query_id)
        missing = sorted(set(relevant_ids).difference(document_ids))
        if missing:
            raise ValueError(
                f"{pack.prefix}: query {query_id} has positive qrels outside the sampled corpus: " f"{missing}"
            )


def _positive_relevant_ids(pack: FrozenEmbeddingPack, query_id: str) -> List[str]:
    relevance = pack.qrels.get(str(query_id))
    if not isinstance(relevance, Mapping):
        raise ValueError(f"{pack.prefix}: query {query_id} has no qrel mapping")
    try:
        relevant_ids = [str(doc_id) for doc_id, score in relevance.items() if float(score) > 0.0]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{pack.prefix}: query {query_id} has invalid qrel scores") from exc
    if not relevant_ids:
        raise ValueError(f"{pack.prefix}: query {query_id} has no positive qrels")
    return relevant_ids


def _normalized_evaluated_qrels(pack: FrozenEmbeddingPack) -> Mapping[str, Mapping[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}
    for query_id in pack.query_ids.astype(str):
        relevance = pack.qrels.get(str(query_id))
        if not isinstance(relevance, Mapping):
            raise ValueError(f"{pack.prefix}: query {query_id} has no qrel mapping")
        row: Dict[str, float] = {}
        for raw_doc_id, raw_score in relevance.items():
            doc_id = str(raw_doc_id)
            if doc_id in row:
                raise ValueError(f"{pack.prefix}: query {query_id} has duplicate normalized qrel ID {doc_id}")
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{pack.prefix}: query {query_id} has invalid qrel scores") from exc
            if not math.isfinite(score):
                raise ValueError(f"{pack.prefix}: query {query_id} has non-finite qrel scores")
            row[doc_id] = score
        normalized[str(query_id)] = row
    return normalized


def _qrels_sha256(packs: Sequence[FrozenEmbeddingPack]) -> str:
    blocks = []
    for pack in packs:
        blocks.append(
            {
                "block_id": _block_id(pack),
                "qrels": _normalized_evaluated_qrels(pack),
            }
        )
    blocks.sort(key=lambda item: str(item["block_id"]))
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema_version": "chelatedai.bcc1.qrels-binding.v1",
                "blocks": blocks,
            }
        )
    )


def _binary_ndcg_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Compute binary nDCG with IDCG from every positive qrel, not retrieved positives."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    normalized_ranked_ids = [str(doc_id) for doc_id in ranked_ids[:k]]
    if len(set(normalized_ranked_ids)) != len(normalized_ranked_ids):
        raise ValueError("ranked_ids must not contain duplicate document IDs")
    positives = {str(doc_id) for doc_id in relevant_ids}
    if not positives:
        raise ValueError("binary nDCG requires at least one positive qrel")
    dcg = sum(1.0 / math.log2(rank + 2.0) for rank, doc_id in enumerate(normalized_ranked_ids) if doc_id in positives)
    ideal_count = min(k, len(positives))
    idcg = sum(1.0 / math.log2(rank + 2.0) for rank in range(ideal_count))
    return float(dcg / idcg)


def _rank_order(
    scores: np.ndarray,
    document_ids: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    normalized_scores = np.asarray(scores, dtype=np.float64)
    normalized_ids = np.asarray(document_ids).astype(str)
    if normalized_scores.ndim != 1 or normalized_ids.ndim != 1:
        raise ValueError("ranking requires one-dimensional scores and document IDs")
    if len(normalized_scores) != len(normalized_ids) or not len(normalized_scores):
        raise ValueError("ranking score and document ID counts must be equal and non-empty")
    if not np.all(np.isfinite(normalized_scores)):
        raise ValueError("ranking scores must be finite")
    if len(np.unique(normalized_ids)) != len(normalized_ids):
        raise ValueError("ranking document IDs must be unique")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return np.lexsort((normalized_ids, -normalized_scores))[: min(k, len(normalized_scores))]


def _dataset_family_id(pack: FrozenEmbeddingPack) -> str:
    return _slug(str(pack.metadata.get("dataset", "SciFact")))


def _transition_family_id(pack: FrozenEmbeddingPack) -> str:
    models = pack.metadata.get("models", {})
    old = str(models.get("old", pack.metadata.get("old_model", "unknown-old")))
    swap = str(models.get("swap", pack.metadata.get("swap_model", "unknown-new")))
    return f"{_slug(old)}-to-{_slug(swap)}"


def _anchor_indices(dataset_family_id: str, doc_ids: np.ndarray) -> np.ndarray:
    """Select a qrels-independent, order-independent half-corpus anchor set."""

    normalized_ids = np.asarray(doc_ids).astype(str)
    if normalized_ids.ndim != 1 or not len(normalized_ids):
        raise ValueError("anchor selection requires a non-empty document ID vector")
    if len(np.unique(normalized_ids)) != len(normalized_ids):
        raise ValueError("anchor selection requires unique document IDs")
    ranked: List[Tuple[bytes, str, int]] = []
    for index, doc_id in enumerate(normalized_ids):
        stable_key = f"{dataset_family_id}\0{doc_id}".encode("utf-8")
        ranked.append((hashlib.sha256(stable_key).digest(), doc_id, index))
    anchor_count = max(1, (len(ranked) + 1) // 2)
    selected = sorted(ranked)[:anchor_count]
    return np.asarray(sorted(item[2] for item in selected), dtype=np.int64)


def _fit_ridge(source: np.ndarray, target: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
    selected_source = np.asarray(source, dtype=np.float64)[fit_idx]
    selected_target = np.asarray(target, dtype=np.float64)[fit_idx]
    dimension = selected_source.shape[1]
    return np.linalg.solve(
        selected_source.T @ selected_source + RIDGE_REGULARIZATION * np.eye(dimension, dtype=np.float64),
        selected_source.T @ selected_target,
    )


def _per_query_ndcg(
    query_vectors: np.ndarray,
    document_vectors: np.ndarray,
    pack: FrozenEmbeddingPack,
) -> np.ndarray:
    queries = _normalize_rows(query_vectors)
    documents = _normalize_rows(document_vectors)
    rows: List[float] = []
    for query_id, query in zip(pack.query_ids, queries):
        relevant_ids = _positive_relevant_ids(pack, str(query_id))
        similarities = documents @ query
        order = _rank_order(similarities, pack.doc_ids, k=TOP_K)
        ranked_ids = pack.doc_ids[order].astype(str).tolist()
        rows.append(_binary_ndcg_at_k(ranked_ids, relevant_ids, k=TOP_K))
    return np.asarray(rows, dtype=np.float64)


def _top_summary(
    scores: np.ndarray,
    document_ids: np.ndarray,
) -> Tuple[np.ndarray, float, float, float, float, float]:
    order = _rank_order(scores, document_ids, k=TOP_K)
    top = np.asarray(scores[order], dtype=np.float64)
    margin = float(top[0] - top[1]) if len(top) > 1 else 0.0
    shifted = top - float(np.max(top))
    weights = np.exp(shifted)
    probabilities = weights / float(np.sum(weights))
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15))))
    return order, float(top[0]), margin, float(np.mean(top)), float(np.std(top)), entropy


def _features_for_query(
    query_new_raw: np.ndarray,
    query_reverse_raw: np.ndarray,
    forward_documents: np.ndarray,
    old_documents: np.ndarray,
    forward_bridge: np.ndarray,
    new_anchors: np.ndarray,
    old_anchors: np.ndarray,
    document_ids: np.ndarray,
) -> Dict[str, float]:
    query_new = _normalize_rows(query_new_raw[None, :])[0]
    query_reverse = _normalize_rows(query_reverse_raw[None, :])[0]
    forward_scores = forward_documents @ query_new
    reverse_scores = old_documents @ query_reverse
    f_order, f_top1, f_margin, f_mean, f_std, f_entropy = _top_summary(
        forward_scores,
        document_ids,
    )
    r_order, r_top1, r_margin, r_mean, r_std, r_entropy = _top_summary(
        reverse_scores,
        document_ids,
    )
    f_first = int(f_order[0])
    r_first = int(r_order[0])
    intersection = len(set(map(int, f_order)).intersection(map(int, r_order)))
    query_cycle = _normalize_rows((query_reverse_raw[None, :] @ forward_bridge))[0]
    new_norm = float(np.linalg.norm(query_new_raw))
    reverse_norm = float(np.linalg.norm(query_reverse_raw))
    values = {
        "forward_top1": f_top1,
        "forward_top1_margin": f_margin,
        "forward_top10_mean": f_mean,
        "forward_top10_std": f_std,
        "forward_top10_entropy": f_entropy,
        "reverse_top1": r_top1,
        "reverse_top1_margin": r_margin,
        "reverse_top10_mean": r_mean,
        "reverse_top10_std": r_std,
        "reverse_top10_entropy": r_entropy,
        "top1_direction_gap": f_top1 - r_top1,
        "top10_mean_direction_gap": f_mean - r_mean,
        "top10_jaccard": float(intersection / max(1, len(set(f_order).union(r_order)))),
        "forward_choice_reverse_score": float(reverse_scores[f_first]),
        "reverse_choice_forward_score": float(forward_scores[r_first]),
        "forward_choice_reverse_regret": float(reverse_scores[f_first] - r_top1),
        "reverse_choice_forward_regret": float(forward_scores[r_first] - f_top1),
        "query_cycle_l2": float(np.linalg.norm(query_cycle - query_new)),
        "reverse_query_norm_ratio": reverse_norm / max(new_norm, 1e-15),
        "new_anchor_max_cosine": float(np.max(new_anchors @ query_new)),
        "old_anchor_max_cosine": float(np.max(old_anchors @ query_reverse)),
    }
    expected = {str(item["name"]) for item in FEATURES}
    if set(values) != expected or not all(math.isfinite(value) for value in values.values()):
        raise ValueError("feature construction produced an invalid signature or non-finite value")
    return values


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return cleaned or "unknown"


def _block_id(pack: FrozenEmbeddingPack) -> str:
    models = pack.metadata.get("models", {})
    swap = str(models.get("swap", pack.metadata.get("swap_model", "all-mpnet-base-v2")))
    return f"{_dataset_family_id(pack)}--{_slug(swap)}"


def _independence_group_id(dataset_family_id: str, raw_query_id: str) -> str:
    raw_value = str(raw_query_id)
    query_digest = _sha256_bytes(raw_value.encode("utf-8"))[:16]
    return f"{dataset_family_id}::query/{_slug(raw_value)[:80]}-{query_digest}"


def _rows_for_pack(
    pack: FrozenEmbeddingPack,
) -> Tuple[str, str, str, List[Dict[str, Any]]]:
    block_id = _block_id(pack)
    dataset_family_id = _dataset_family_id(pack)
    transition_family_id = _transition_family_id(pack)
    anchor_idx = _anchor_indices(dataset_family_id, pack.doc_ids)
    forward_bridge = _fit_ridge(pack.do, pack.dor, anchor_idx)
    reverse_bridge = _fit_ridge(pack.dor, pack.do, anchor_idx)
    forward_documents_raw = pack.do @ forward_bridge
    reverse_queries_raw = pack.qd @ reverse_bridge
    forward_scores = _per_query_ndcg(pack.qd, forward_documents_raw, pack)
    reverse_scores = _per_query_ndcg(reverse_queries_raw, pack.do, pack)
    native_new_scores = _per_query_ndcg(pack.qd, pack.dor, pack)
    mismatch_scores = _per_query_ndcg(pack.qd, pack.do, pack)
    forward_documents = _normalize_rows(forward_documents_raw)
    old_documents = _normalize_rows(pack.do)
    new_anchors = _normalize_rows(pack.dor[anchor_idx])
    old_anchors = _normalize_rows(pack.do[anchor_idx])
    rows: List[Dict[str, Any]] = []
    for index, raw_query_id in enumerate(pack.query_ids):
        query_id = f"{block_id}::{raw_query_id}"
        rows.append(
            {
                "query_id": query_id,
                "independence_group_id": _independence_group_id(dataset_family_id, str(raw_query_id)),
                "block_id": block_id,
                "scores": {
                    "reverse": float(reverse_scores[index]),
                    "forward": float(forward_scores[index]),
                    "native_new": float(native_new_scores[index]),
                    "mismatch": float(mismatch_scores[index]),
                },
                "features": _features_for_query(
                    pack.qd[index],
                    reverse_queries_raw[index],
                    forward_documents,
                    old_documents,
                    forward_bridge,
                    new_anchors,
                    old_anchors,
                    pack.doc_ids,
                ),
            }
        )
    return block_id, dataset_family_id, transition_family_id, rows


def _feature_declarations(source_digest: str, implementation_sha256: str) -> List[Dict[str, Any]]:
    declarations: List[Dict[str, Any]] = []
    for item in FEATURES:
        name = str(item["name"])
        source = str(item["source"])
        availability_stage = str(item["availability_stage"])
        execution_cost_class = str(item["execution_cost_class"])
        provenance = {
            "feature_name": name,
            "feature_source": source,
            "availability_stage": availability_stage,
            "execution_cost_class": execution_cost_class,
            "source_digest": source_digest,
            "anchor_strategy": ANCHOR_STRATEGY,
            "implementation_sha256": implementation_sha256,
        }
        declarations.append(
            {
                "name": name,
                "qrels_free": True,
                "inference_available": True,
                "source": source,
                "availability_stage": availability_stage,
                "execution_cost_class": execution_cost_class,
                "implementation_sha256": implementation_sha256,
                "provenance_sha256": _sha256_bytes(_canonical_json_bytes(provenance)),
            }
        )
    return declarations


def _discover_prefixes(source_root: Path) -> List[Path]:
    source_root = Path(source_root)
    estimator_dir = source_root / "research" / "drift_recovery" / "out" / "estimator" / "packs"
    prefixes = sorted(path.with_suffix("") for path in estimator_dir.glob("*_pack.json"))
    d1_prefix = source_root / "research" / "drift_recovery" / "out" / "d1" / "scifact_evalsplit_pack"
    if d1_prefix.with_suffix(".json").is_file() and d1_prefix.with_suffix(".npz").is_file():
        prefixes.append(d1_prefix)
    if not prefixes:
        raise ValueError(f"no recovered embedding packs found below {source_root}")
    return prefixes


def _derivation_sha256(source_digest: str, implementation_sha256: str) -> str:
    """Bind the derived dataset identity to every result-changing input."""
    derivation = {
        "schema_version": SCHEMA_VERSION,
        "evidence_mode": "METHOD_DEV",
        "source_digest": source_digest,
        "implementation_sha256": implementation_sha256,
        "anchor_strategy": ANCHOR_STRATEGY,
        "ridge_regularization": RIDGE_REGULARIZATION,
        "top_k": TOP_K,
        "minimum_vector_l2_norm": MIN_VECTOR_L2_NORM,
        "metric": {
            "metric_name": METRIC_NAME,
            "gain": METRIC_GAIN,
            "idcg_population": METRIC_IDCG_POPULATION,
            "query_inclusion": METRIC_QUERY_INCLUSION,
            "ranking_tie_break": METRIC_RANKING_TIE_BREAK,
        },
        "features": FEATURES,
    }
    return _sha256_bytes(_canonical_json_bytes(derivation))


def build_method_dev_pack(source_root: Path, pack_output: Path, manifest_output: Path) -> Dict[str, Any]:
    pack_output = Path(pack_output)
    manifest_output = Path(manifest_output)
    if pack_output.exists() or manifest_output.exists():
        raise FileExistsError("refusing to overwrite an existing BCC-1 artifact")
    packs = [_load_pack(prefix) for prefix in _discover_prefixes(source_root)]
    source_fingerprints = sorted(f"{pack.prefix.name}:{pack.arrays_sha256}:{pack.metadata_sha256}" for pack in packs)
    source_digest = _sha256_bytes("\n".join(source_fingerprints).encode("utf-8"))
    implementation_sha256 = _sha256_file(Path(__file__))
    derivation_sha256 = _derivation_sha256(source_digest, implementation_sha256)
    pack_id = f"bcc1-method-dev-derivation-{derivation_sha256}"
    rows: List[Dict[str, Any]] = []
    query_ids_by_block: Dict[str, List[str]] = {}
    block_families: Dict[str, Tuple[str, str]] = {}
    for frozen_pack in packs:
        block_id, dataset_family_id, transition_family_id, block_rows = _rows_for_pack(frozen_pack)
        if block_id in query_ids_by_block:
            raise ValueError(f"duplicate structural block: {block_id}")
        rows.extend(block_rows)
        query_ids_by_block[block_id] = [str(row["query_id"]) for row in block_rows]
        block_families[block_id] = (dataset_family_id, transition_family_id)
    rows.sort(key=lambda row: (str(row["block_id"]), str(row["query_id"])))
    pack_payload = {
        "schema_version": SCHEMA_VERSION,
        "pack_id": pack_id,
        "evidence_mode": "METHOD_DEV",
        "sampled": True,
        "metric": {
            "name": METRIC_NAME,
            "higher_is_better": True,
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "rows": rows,
    }
    pack_bytes = _canonical_json_bytes(pack_payload)
    pack_sha256 = _sha256_bytes(pack_bytes)
    qrels_sha256 = _qrels_sha256(packs)
    metric_binding_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema_version": "chelatedai.bcc1.metric-binding.v1",
                "pack_sha256": pack_sha256,
                "qrels_sha256": qrels_sha256,
            }
        )
    )
    pack_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    pack_output.write_bytes(pack_bytes)
    blocks = [
        {
            "block_id": block_id,
            "dataset_family_id": block_families[block_id][0],
            "transition_family_id": block_families[block_id][1],
            "role": "METHOD_DEV",
            "query_ids_sha256": _sha256_bytes(_canonical_json_bytes(sorted(query_ids_by_block[block_id]))),
            "consumed": False,
        }
        for block_id in sorted(query_ids_by_block)
    ]
    manifest_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pack_id": pack_id,
        "evidence_mode": "METHOD_DEV",
        "sampled": True,
        "pack_sha256": pack_sha256,
        "metric_contract": {
            "metric_name": METRIC_NAME,
            "implementation_sha256": implementation_sha256,
            "gain": METRIC_GAIN,
            "idcg_population": METRIC_IDCG_POPULATION,
            "cutoff": TOP_K,
            "query_inclusion": METRIC_QUERY_INCLUSION,
            "qrels_sha256": qrels_sha256,
            "pack_qrels_binding_sha256": metric_binding_sha256,
            "ranking_tie_break": METRIC_RANKING_TIE_BREAK,
        },
        "reverse_route_role_validation": REVERSE_ROUTE_ROLE_VALIDATION,
        "features": _feature_declarations(source_digest, implementation_sha256),
        "blocks": blocks,
        "soft_fusion": None,
    }
    manifest_output.write_bytes(_canonical_json_bytes(manifest_payload))
    return {
        "pack_id": pack_id,
        "pack_path": str(pack_output),
        "manifest_path": str(manifest_output),
        "pack_sha256": manifest_payload["pack_sha256"],
        "source_digest": source_digest,
        "derivation_sha256": derivation_sha256,
        "implementation_sha256": implementation_sha256,
        "blocks": len(blocks),
        "queries": len(rows),
        "classification": "exploratory_non_promotional",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pack-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = build_method_dev_pack(args.source_root, args.pack_output, args.manifest_output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
