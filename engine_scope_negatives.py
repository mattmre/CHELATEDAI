"""Mine replayable hard-negative families from Engine-Scope rows."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from benchmark_utils import canonicalize_id
from engine_scope import summarize_engine_scope_rows
from engine_scope_coverage import row_feature_tokens, summarize_engine_scope_coverage


_QUERY_PROFILE_FAILURES = {
    "actuator_active_negative": 0,
    "actuator_active_neutral": 1,
}

_QUERY_PROFILE_SIGNATURE_PREFIXES = {
    "row_type",
    "profile",
    "action",
    "fault",
    "delta",
    "q.tokens",
    "q.stopwords",
    "q.numeric",
    "q.negation",
    "q.claim",
    "overlap",
    "jaccard",
    "mask_density",
    "variants",
    "reform_changed",
}

_MASK_PROBE_SIGNATURE_PREFIXES = {
    "row_type",
    "delta",
    "q.tokens",
    "q.stopwords",
    "q.numeric",
    "q.negation",
    "q.claim",
    "margin",
    "top_score",
    "query_norm",
}


def _signature_tokens(row: Dict[str, Any]) -> List[str]:
    row_type = str(row.get("row_type") or "")
    allowed = (
        _QUERY_PROFILE_SIGNATURE_PREFIXES
        if row_type == "query_profile"
        else _MASK_PROBE_SIGNATURE_PREFIXES
    )
    return sorted(
        token
        for token in row_feature_tokens(row)
        if token.split(":", 1)[0] in allowed
    )


def _query_profile_negative_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for row in rows:
        item = dict(row)
        if str(item.get("row_type") or "") != "query_profile":
            continue
        if str(item.get("source_family") or "").startswith("hard_negative_"):
            continue
        if str(item.get("profile") or "") == "baseline":
            continue
        fault_class = str(item.get("fault_class") or "")
        if fault_class not in _QUERY_PROFILE_FAILURES:
            continue
        try:
            delta = float(item.get("delta_ndcg_at_10", 0.0))
        except (TypeError, ValueError):
            continue
        rank_delta = item.get("rank_delta")
        non_improving_rank = rank_delta is None
        try:
            if rank_delta is not None:
                non_improving_rank = int(rank_delta) >= 0
        except (TypeError, ValueError):
            non_improving_rank = True
        if delta > 0.001 and not non_improving_rank:
            continue
        if delta > 0.001:
            continue
        filtered.append(item)
    return filtered


def _mask_probe_negative_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for row in rows:
        item = dict(row)
        if str(item.get("row_type") or "") != "mask_probe":
            continue
        if str(item.get("source_family") or "").startswith("hard_negative_"):
            continue
        try:
            delta = float(item.get("delta_ndcg_at_10", 0.0))
        except (TypeError, ValueError):
            continue
        if delta > 0.001:
            continue
        filtered.append(item)
    return filtered


def _query_profile_family_sort_key(family: Dict[str, Any]) -> Tuple[int, float, int, str, str]:
    primary_fault = str(family.get("primary_fault_class") or "")
    return (
        _QUERY_PROFILE_FAILURES.get(primary_fault, 99),
        float(family.get("mean_delta_ndcg_at_10", 0.0)),
        -int(family.get("query_count", 0)),
        str(family.get("task") or ""),
        str(family.get("signature") or ""),
    )


def _mask_probe_family_sort_key(family: Dict[str, Any]) -> Tuple[float, int, str, str]:
    return (
        float(family.get("mean_delta_ndcg_at_10", 0.0)),
        -int(family.get("query_count", 0)),
        str(family.get("task") or ""),
        str(family.get("signature") or ""),
    )


def _group_rows_into_families(
    rows: Iterable[Dict[str, Any]],
    *,
    family_kind: str,
    min_family_size: int,
    max_queries_per_family: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        signature_tokens = _signature_tokens(item)
        signature = "|".join(signature_tokens)
        key = (str(item.get("task") or ""), signature)
        item["_signature_tokens"] = signature_tokens
        grouped.setdefault(key, []).append(item)

    families = []
    for (task, signature), group_rows in grouped.items():
        if len(group_rows) < min_family_size:
            continue
        sorted_rows = sorted(
            group_rows,
            key=lambda row: (
                float(row.get("delta_ndcg_at_10", 0.0)),
                str(row.get("query_id") or ""),
                str(row.get("profile") or ""),
            ),
        )
        deltas = [float(row.get("delta_ndcg_at_10", 0.0)) for row in sorted_rows]
        query_ids = [str(row.get("query_id") or "") for row in sorted_rows]
        negative_source_type = (
            "engine_scope_query_profile"
            if family_kind == "query_profile_failure"
            else "engine_scope_mask_probe"
        )
        family = {
            "family_kind": family_kind,
            "task": task,
            "signature": signature,
            "signature_tokens": list(sorted_rows[0]["_signature_tokens"]),
            "query_count": len(sorted_rows),
            "mean_delta_ndcg_at_10": float(np.mean(deltas)) if deltas else 0.0,
            "worst_delta_ndcg_at_10": float(np.min(deltas)) if deltas else 0.0,
            "best_delta_ndcg_at_10": float(np.max(deltas)) if deltas else 0.0,
            "query_ids": query_ids,
            "replay_query_ids": query_ids[:max_queries_per_family],
            "examples": [
                {
                    "query_id": str(row.get("query_id") or ""),
                    "query_text": row.get("query_text"),
                    "profile": row.get("profile"),
                    "action": row.get("action"),
                    "fault_class": row.get("fault_class"),
                    "delta_ndcg_at_10": float(row.get("delta_ndcg_at_10", 0.0)),
                    "source_family": row.get("source_family"),
                    "query_offset": row.get("query_offset"),
                    "seed": row.get("seed"),
                }
                for row in sorted_rows[:max_queries_per_family]
            ],
            "negative_source_type": negative_source_type,
            "ambiguity_grade": "unknown",
            "generator": "engine_scope_negatives.mine_hard_negative_families",
            "verifier": "delta_ndcg_at_10_non_positive",
            "false_negative_risk": "unscored",
            "synthetic_depth": int(sorted_rows[0].get("synthetic_depth", 0) or 0),
            "source_lineage": [
                str(row.get("_artifact_path") or row.get("source_family") or "unknown")
                for row in sorted_rows[:max_queries_per_family]
            ],
        }
        if family_kind == "query_profile_failure":
            fault_counts: Dict[str, int] = {}
            profile_counts: Dict[str, int] = {}
            action_counts: Dict[str, int] = {}
            for row in sorted_rows:
                fault = str(row.get("fault_class") or "unknown")
                profile = str(row.get("profile") or "unknown")
                action = str(row.get("action") or "unknown")
                fault_counts[fault] = fault_counts.get(fault, 0) + 1
                profile_counts[profile] = profile_counts.get(profile, 0) + 1
                action_counts[action] = action_counts.get(action, 0) + 1
            primary_fault = sorted(
                fault_counts.items(),
                key=lambda item: (-item[1], _QUERY_PROFILE_FAILURES.get(item[0], 99), item[0]),
            )[0][0]
            family["fault_counts"] = fault_counts
            family["primary_fault_class"] = primary_fault
            family["profile_counts"] = profile_counts
            family["action_counts"] = action_counts
        families.append(family)
    return families


def mine_hard_negative_families(
    rows: Iterable[Dict[str, Any]],
    *,
    min_family_size: int = 2,
    max_query_profile_families: int = 5,
    max_mask_probe_families: int = 5,
    max_queries_per_family: int = 12,
    max_synthetic_depth: int = 1,
) -> Dict[str, Any]:
    """Mine replayable failure families from pooled Engine-Scope rows.

    **Grouping algorithm**: rows are partitioned into families by a deterministic
    fault-class grouping — each (task, signature-token-set) pair becomes exactly
    one family, with the signature derived from a fixed prefix whitelist per
    ``row_type``.  This is NOT algorithmic clustering (no distance metric, no
    centroid, no k-means or DBSCAN pass).  The word "cluster" in any prior
    description was inaccurate; "fault-class group" or "signature group" is
    the correct term.  The grouping is fully deterministic: given the same input
    rows in any order, the output families and their ``family_id`` assignments
    are identical.
    """

    row_list = [dict(row) for row in rows]
    filtered_rows = [
        row
        for row in row_list
        if int(row.get("synthetic_depth", 0) or 0) <= max_synthetic_depth
    ]
    recursive_blocked_count = len(row_list) - len(filtered_rows)
    query_profile_families = _group_rows_into_families(
        _query_profile_negative_rows(filtered_rows),
        family_kind="query_profile_failure",
        min_family_size=min_family_size,
        max_queries_per_family=max_queries_per_family,
    )
    query_profile_families.sort(key=_query_profile_family_sort_key)
    mask_probe_families = _group_rows_into_families(
        _mask_probe_negative_rows(filtered_rows),
        family_kind="mask_probe_failure",
        min_family_size=min_family_size,
        max_queries_per_family=max_queries_per_family,
    )
    mask_probe_families.sort(key=_mask_probe_family_sort_key)
    selected_query_profile = query_profile_families[:max_query_profile_families]
    for index, family in enumerate(selected_query_profile, start=1):
        family["family_id"] = f"qneg_{index:03d}"
    selected_mask_probe = mask_probe_families[:max_mask_probe_families]
    for index, family in enumerate(selected_mask_probe, start=1):
        family["family_id"] = f"mneg_{index:03d}"
    return {
        "query_profile_failure_count": len(query_profile_families),
        "mask_probe_failure_count": len(mask_probe_families),
        "recursive_blocked_count": recursive_blocked_count,
        "max_synthetic_depth": int(max_synthetic_depth),
        "query_profile_families": selected_query_profile,
        "mask_probe_families": selected_mask_probe,
    }


def build_hard_negative_replay_artifact(
    rows: Iterable[Dict[str, Any]],
    families: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a deterministic replay artifact from selected hard-negative families."""

    row_list = [dict(row) for row in rows]
    selected_families = [copy.deepcopy(dict(family)) for family in families]
    family_index = {
        (str(family.get("task") or ""), str(family.get("signature") or "")): family
        for family in selected_families
    }
    family_query_ids = {
        key: {str(query_id) for query_id in family.get("query_ids", []) if str(query_id)}
        for key, family in family_index.items()
    }
    selected_rows = []
    for row in row_list:
        key = (
            str(row.get("task") or ""),
            "|".join(_signature_tokens(row)),
        )
        family = family_index.get(key)
        if family is None:
            continue
        query_id = str(row.get("query_id") or "")
        if family_query_ids.get(key) and query_id not in family_query_ids[key]:
            continue
        item = dict(row)
        item["family_id"] = family.get("family_id")
        item["family_kind"] = family.get("family_kind")
        item["family_signature"] = family.get("signature")
        selected_rows.append(item)
    selected_rows.sort(
        key=lambda row: (
            str(row.get("family_id") or ""),
            str(row.get("task") or ""),
            str(row.get("query_id") or ""),
            str(row.get("profile") or ""),
            str(row.get("row_type") or ""),
        )
    )
    return {
        "family_count": len(selected_families),
        "families": selected_families,
        "engine_scope_rows": selected_rows,
        "summary": summarize_engine_scope_rows(selected_rows),
        "coverage_summary": summarize_engine_scope_coverage(selected_rows),
    }


def select_query_subset_by_ids(
    corpus: Mapping[Any, str],
    queries: Mapping[Any, str],
    qrels: Mapping[Any, Mapping[Any, float]],
    query_ids: Sequence[str],
    *,
    sample_docs: int,
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, float]]]:
    """Select an explicit judged query subset and preserve all relevant docs."""

    ordered_ids = []
    seen = set()
    for query_id in query_ids:
        canonical_query_id = canonicalize_id(query_id)
        if canonical_query_id not in seen:
            seen.add(canonical_query_id)
            ordered_ids.append(canonical_query_id)

    query_texts = {
        canonicalize_id(query_id): str(query_text)
        for query_id, query_text in queries.items()
    }
    qrels_by_query = {
        canonicalize_id(query_id): {
            canonicalize_id(doc_id): float(score)
            for doc_id, score in relevance.items()
            if float(score) > 0
        }
        for query_id, relevance in qrels.items()
    }
    selected_queries = {
        query_id: query_texts[query_id]
        for query_id in ordered_ids
        if query_id in query_texts and qrels_by_query.get(query_id)
    }
    if not selected_queries:
        raise ValueError("no judged query ids from the requested family were available")
    selected_qrels = {query_id: qrels_by_query[query_id] for query_id in selected_queries}

    corpus_by_id = {canonicalize_id(doc_id): str(text) for doc_id, text in corpus.items()}
    required_ids = {
        doc_id
        for relevance in selected_qrels.values()
        for doc_id in relevance
        if doc_id in corpus_by_id
    }
    remaining = [doc_id for doc_id in corpus_by_id if doc_id not in required_ids]
    rng = np.random.RandomState(seed)
    extra_budget = max(0, sample_docs - len(required_ids))
    if extra_budget < len(remaining):
        sampled_extra = set(rng.choice(remaining, size=extra_budget, replace=False))
    else:
        sampled_extra = set(remaining)
    keep_ids = sorted(required_ids | sampled_extra)
    selected_corpus = {doc_id: corpus_by_id[doc_id] for doc_id in keep_ids}
    return selected_corpus, selected_queries, selected_qrels
