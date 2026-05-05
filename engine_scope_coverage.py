"""Feature-footprint coverage summaries for pooled Engine-Scope rows."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


_CONTEXT_FIELDS: Sequence[str] = (
    "source_family",
    "task",
    "query_offset",
    "seed",
    "split",
    "loop",
    "window",
    "global_window",
)


def _context_value(row: Dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return str(value)


def _sort_component(row: Dict[str, Any], key: str) -> Tuple[int, Any]:
    value = row.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 0, float(value)
    try:
        if value is not None and str(value).strip() != "":
            return 0, float(value)
    except (TypeError, ValueError):
        pass
    return 1, _context_value(row, key)


def _window_sort_key(row: Dict[str, Any]) -> Tuple[Tuple[int, Any], ...]:
    return tuple(_sort_component(row, key) for key in _CONTEXT_FIELDS)


def _window_key(row: Dict[str, Any]) -> str:
    parts = []
    for key in _CONTEXT_FIELDS:
        value = row.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")
    return "|".join(parts) or "window=unknown"


def _bucket(value: Any, *, low: float, high: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if number <= low:
        return "low"
    if number <= high:
        return "mid"
    return "high"


def _count_bucket(value: Any, *, low: int, high: int) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if number <= low:
        return "low"
    if number <= high:
        return "mid"
    return "high"


def _delta_bucket(value: Any) -> str:
    try:
        delta = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if delta > 0.001:
        return "positive"
    if delta < -0.001:
        return "negative"
    return "neutral"


def _bool_bucket(value: Any) -> str:
    return "yes" if bool(value) else "no"


def row_feature_tokens(row: Dict[str, Any]) -> Set[str]:
    """Map an Engine-Scope row to a compact, comparable feature-token set."""

    row_type = str(row.get("row_type") or "unknown")
    tokens = {
        f"row_type:{row_type}",
        f"profile:{row.get('profile') or 'unknown'}",
        f"delta:{_delta_bucket(row.get('delta_ndcg_at_10'))}",
        f"q.tokens:{_count_bucket(row.get('query_token_count'), low=4, high=8)}",
        f"q.length:{_bucket(row.get('query_char_count'), low=30.0, high=80.0)}",
        f"q.stopwords:{_bucket(row.get('query_stopword_ratio'), low=0.15, high=0.35)}",
        f"q.numeric:{_count_bucket(row.get('query_numeric_token_count'), low=0, high=1)}",
        f"q.negation:{_count_bucket(row.get('query_negation_count'), low=0, high=1)}",
        f"q.claim:{_count_bucket(row.get('query_claim_cue_count'), low=0, high=1)}",
    }
    if row_type == "query_profile":
        tokens.update({
            f"action:{row.get('action') or 'unknown'}",
            f"fault:{row.get('fault_class') or 'unknown'}",
            f"overlap:{_bucket(row.get('top10_overlap_with_baseline'), low=3.0, high=7.0)}",
            f"topdoc_changed:{_bool_bucket(row.get('top_doc_changed'))}",
            f"variance:{_bucket(row.get('global_variance'), low=0.05, high=0.2)}",
            f"jaccard:{_bucket(row.get('jaccard'), low=0.25, high=0.6)}",
            f"mask_density:{_bucket(row.get('mask_density'), low=0.2, high=0.8)}",
            f"variants:{_count_bucket(row.get('reformulation_variant_count'), low=0, high=2)}",
            f"reform_changed:{_bool_bucket(row.get('reformulation_changed'))}",
        })
    elif row_type == "mask_probe":
        tokens.update({
            f"margin:{_bucket(row.get('baseline_score_margin'), low=0.05, high=0.2)}",
            f"top_score:{_bucket(row.get('baseline_top_score'), low=0.25, high=0.75)}",
            f"query_norm:{_bucket(row.get('query_norm'), low=0.8, high=1.2)}",
        })
    return tokens


def build_window_footprints(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in sorted((dict(item) for item in rows), key=_window_sort_key):
        key = _window_key(row)
        footprint = grouped.setdefault(
            key,
            {
                "window_key": key,
                "source_family": row.get("source_family"),
                "task": row.get("task"),
                "query_offset": row.get("query_offset"),
                "seed": row.get("seed"),
                "split": row.get("split"),
                "loop": row.get("loop"),
                "window": row.get("window"),
                "global_window": row.get("global_window"),
                "row_count": 0,
                "row_types": {},
                "profiles": {},
                "tokens": set(),
            },
        )
        footprint["row_count"] += 1
        row_type = str(row.get("row_type") or "unknown")
        profile = str(row.get("profile") or "unknown")
        footprint["row_types"][row_type] = footprint["row_types"].get(row_type, 0) + 1
        footprint["profiles"][profile] = footprint["profiles"].get(profile, 0) + 1
        footprint["tokens"].update(row_feature_tokens(row))

    results = []
    for footprint in grouped.values():
        token_set = sorted(footprint.pop("tokens"))
        results.append({
            **footprint,
            "token_count": len(token_set),
            "feature_tokens": token_set,
        })
    return sorted(results, key=lambda item: item["window_key"])


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def summarize_engine_scope_coverage(
    rows: Iterable[Dict[str, Any]],
    *,
    novelty_threshold: float = 0.35,
    redundancy_threshold: float = 0.85,
    top_pair_count: int = 10,
) -> Dict[str, Any]:
    footprints = build_window_footprints(rows)
    prior_universe: Set[str] = set()
    window_records: List[Dict[str, Any]] = []
    redundant_windows: List[Dict[str, Any]] = []
    novel_windows: List[Dict[str, Any]] = []
    overlap_pairs: List[Dict[str, Any]] = []

    for index, footprint in enumerate(footprints):
        tokens = footprint["feature_tokens"]
        token_set = set(tokens)
        novel_tokens = sorted(token_set - prior_universe)
        max_overlap = 0.0
        closest_window = None
        for other in footprints[:index]:
            overlap = _jaccard(tokens, other["feature_tokens"])
            if overlap > max_overlap:
                max_overlap = overlap
                closest_window = other["window_key"]
        novelty = (len(novel_tokens) / len(token_set)) if token_set else 0.0
        record = {
            **footprint,
            "novel_token_count": len(novel_tokens),
            "novelty_ratio": float(novelty),
            "max_overlap_with_previous": float(max_overlap),
            "closest_previous_window": closest_window,
        }
        window_records.append(record)
        if novelty >= novelty_threshold:
            novel_windows.append(record)
        if max_overlap >= redundancy_threshold and closest_window is not None:
            redundant_windows.append(record)
        prior_universe.update(token_set)

    for index, left in enumerate(footprints):
        for right in footprints[index + 1:]:
            overlap_pairs.append({
                "left_window": left["window_key"],
                "right_window": right["window_key"],
                "overlap": float(_jaccard(left["feature_tokens"], right["feature_tokens"])),
            })
    overlap_pairs.sort(
        key=lambda item: (-item["overlap"], item["left_window"], item["right_window"])
    )

    task_offset_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in window_records:
        group_key = (str(record.get("task") or ""), str(record.get("query_offset") or ""))
        group = task_offset_groups.setdefault(
            group_key,
            {
                "task": record.get("task"),
                "query_offset": record.get("query_offset"),
                "window_count": 0,
                "source_families": set(),
                "row_count": 0,
                "token_union": set(),
                "max_overlap_with_previous": 0.0,
                "max_novelty_ratio": 0.0,
            },
        )
        group["window_count"] += 1
        group["row_count"] += int(record["row_count"])
        if record.get("source_family"):
            group["source_families"].add(str(record["source_family"]))
        group["token_union"].update(record["feature_tokens"])
        group["max_overlap_with_previous"] = max(
            group["max_overlap_with_previous"],
            float(record["max_overlap_with_previous"]),
        )
        group["max_novelty_ratio"] = max(
            group["max_novelty_ratio"],
            float(record["novelty_ratio"]),
        )

    task_offset_summary = sorted(
        [
            {
                "task": group["task"],
                "query_offset": group["query_offset"],
                "window_count": group["window_count"],
                "row_count": group["row_count"],
                "token_count": len(group["token_union"]),
                "source_families": sorted(group["source_families"]),
                "max_overlap_with_previous": float(group["max_overlap_with_previous"]),
                "max_novelty_ratio": float(group["max_novelty_ratio"]),
            }
            for group in task_offset_groups.values()
        ],
        key=lambda item: (
            -item["max_novelty_ratio"],
            item["max_overlap_with_previous"],
            str(item["task"] or ""),
            str(item["query_offset"] or ""),
        ),
    )

    return {
        "window_count": len(footprints),
        "token_universe_count": len(prior_universe),
        "novel_window_count": len(novel_windows),
        "redundant_window_count": len(redundant_windows),
        "windows": window_records,
        "top_overlap_pairs": overlap_pairs[:top_pair_count],
        "novel_windows": [record["window_key"] for record in novel_windows],
        "redundant_windows": [record["window_key"] for record in redundant_windows],
        "task_offset_summary": task_offset_summary,
    }


def _task_offset_key(task: Any, query_offset: Any) -> Tuple[str, str]:
    return str(task or ""), str(query_offset if query_offset is not None else "")


def rank_task_offset_candidates(
    candidates: Sequence[Tuple[str, int]],
    coverage_summary: Dict[str, Any] | None,
    *,
    rotation_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Rank task/offset candidates by novelty and overlap from a coverage summary."""

    task_offset_rows = list((coverage_summary or {}).get("task_offset_summary", []))
    exact_index = {
        _task_offset_key(row.get("task"), row.get("query_offset")): dict(row)
        for row in task_offset_rows
    }
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for row in task_offset_rows:
        by_task.setdefault(str(row.get("task") or ""), []).append(dict(row))

    ranked = []
    candidate_count = len(candidates)
    shift = rotation_offset % candidate_count if candidate_count else 0
    for original_index, (task, query_offset) in enumerate(candidates):
        rotated_index = (original_index - shift) % candidate_count if candidate_count else 0
        exact = exact_index.get(_task_offset_key(task, query_offset))
        task_rows = by_task.get(str(task or ""), [])
        task_seen = bool(task_rows)
        exact_seen = exact is not None
        reference_rows = task_rows if task_rows else []
        exact_row_count = int(exact.get("row_count", 0)) if exact else 0
        exact_window_count = int(exact.get("window_count", 0)) if exact else 0
        exact_overlap = float(exact.get("max_overlap_with_previous", 0.0)) if exact else 0.0
        exact_novelty = float(exact.get("max_novelty_ratio", 1.0 if not task_seen else 0.0)) if exact else (1.0 if not task_seen else 0.0)
        task_row_count = sum(int(row.get("row_count", 0)) for row in reference_rows)
        task_window_count = sum(int(row.get("window_count", 0)) for row in reference_rows)
        task_overlap = max((float(row.get("max_overlap_with_previous", 0.0)) for row in reference_rows), default=0.0)
        task_novelty = max((float(row.get("max_novelty_ratio", 0.0)) for row in reference_rows), default=0.0)
        ranked.append({
            "task": task,
            "query_offset": query_offset,
            "exact_seen": exact_seen,
            "task_seen": task_seen,
            "exact_row_count": exact_row_count,
            "exact_window_count": exact_window_count,
            "exact_max_overlap_with_previous": exact_overlap,
            "exact_max_novelty_ratio": exact_novelty,
            "task_row_count": task_row_count,
            "task_window_count": task_window_count,
            "task_max_overlap_with_previous": task_overlap,
            "task_max_novelty_ratio": task_novelty,
            "selection_priority": (
                0 if not exact_seen else 1,
                0 if not task_seen else 1,
                -exact_novelty,
                exact_overlap,
                exact_row_count,
                task_row_count,
                rotated_index,
            ),
        })

    ranked.sort(key=lambda item: item["selection_priority"])
    return ranked


def select_task_offset_candidates(
    candidates: Sequence[Tuple[str, int]],
    coverage_summary: Dict[str, Any] | None,
    *,
    count: int,
    rotation_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Select the highest-priority candidates for the next collection slice."""

    if count <= 0:
        return []
    ranked = rank_task_offset_candidates(
        candidates,
        coverage_summary,
        rotation_offset=rotation_offset,
    )
    return ranked[:count]
