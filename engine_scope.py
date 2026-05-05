"""Normalized Engine-Scope row contract and artifact loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from evidence_contract import evidence_event_from_engine_scope_row


ENGINE_SCOPE_SCHEMA_VERSION = 1

_CONTEXT_KEYS = (
    "task",
    "seed",
    "query_offset",
    "split",
    "loop",
    "window",
    "global_window",
    "profile",
    "source_family",
)


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


def _merged_context(row: Dict[str, Any], context: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(context or {})
    for key in _CONTEXT_KEYS:
        if key in row and row[key] is not None:
            merged[key] = row[key]
    return merged


def _normalize_engine_scope_row(
    row: Dict[str, Any],
    *,
    artifact_path: Path | None = None,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    merged = _merged_context(row, context)
    item = dict(row)
    item.setdefault("schema_version", ENGINE_SCOPE_SCHEMA_VERSION)
    item.setdefault("source_family", merged.get("source_family"))
    if artifact_path is not None and not item.get("_artifact_path"):
        item["_artifact_path"] = str(artifact_path)
    else:
        item.setdefault("_artifact_path", str(artifact_path) if artifact_path is not None else None)
    for key in _CONTEXT_KEYS:
        item.setdefault(key, merged.get(key))
    return item


def _query_attribution_to_engine_scope_row(
    row: Dict[str, Any],
    *,
    source_family: str,
    artifact_path: Path | None = None,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    merged = _merged_context(row, context)
    return {
        "schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
        "row_type": "query_profile",
        "source_family": source_family,
        "_artifact_path": str(artifact_path) if artifact_path is not None else None,
        "task": merged.get("task"),
        "seed": merged.get("seed"),
        "query_offset": merged.get("query_offset"),
        "loop": merged.get("loop"),
        "window": merged.get("window"),
        "global_window": merged.get("global_window"),
        "split": merged.get("split"),
        "profile": merged.get("profile"),
        "query_id": row.get("query_id"),
        "query_text": row.get("query_text"),
        "query_token_count": row.get("query_token_count"),
        "query_char_count": row.get("query_char_count"),
        "query_stopword_ratio": row.get("query_stopword_ratio"),
        "query_numeric_token_count": row.get("query_numeric_token_count"),
        "query_negation_count": row.get("query_negation_count"),
        "query_claim_cue_count": row.get("query_claim_cue_count"),
        "delta_ndcg_at_10": row.get("delta_ndcg_at_10"),
        "delta_mrr": row.get("delta_mrr"),
        "delta_recall_at_10": row.get("delta_recall_at_10"),
        "rank_delta": row.get("rank_delta"),
        "baseline_rank": row.get("baseline_rank"),
        "candidate_rank": row.get("candidate_rank"),
        "top10_overlap_with_baseline": row.get("top10_overlap_with_baseline"),
        "top_doc_changed": row.get("top_doc_changed"),
        "action": row.get("action"),
        "global_variance": row.get("global_variance"),
        "jaccard": row.get("jaccard"),
        "mask_density": row.get("mask_density"),
        "reformulation_variant_count": row.get("reformulation_variant_count"),
        "reformulation_changed": row.get("reformulation_changed"),
        "fault_class": row.get("fault_class"),
        "baseline_score_margin": row.get("baseline_score_margin"),
        "baseline_top_score": row.get("baseline_top_score"),
        "query_norm": row.get("query_norm"),
        "masked_rank": row.get("masked_rank"),
        "gate_applied": row.get("gate_applied"),
        "selected_delta_ndcg_at_10": row.get("selected_delta_ndcg_at_10"),
        "route": row.get("route"),
        "retrieval_intervention": row.get("retrieval_intervention"),
        "gate_decision": row.get("gate_decision"),
        "evaluator_outcome": row.get("evaluator_outcome"),
        "safety_outcome": row.get("safety_outcome"),
        "promotion_blocker": row.get("promotion_blocker"),
        "decision": row.get("decision"),
    }


def _mask_example_to_engine_scope_row(
    row: Dict[str, Any],
    *,
    source_family: str,
    artifact_path: Path | None = None,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    merged = _merged_context(row, context)
    return {
        "schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
        "row_type": "mask_probe",
        "source_family": source_family,
        "_artifact_path": str(artifact_path) if artifact_path is not None else None,
        "task": merged.get("task"),
        "seed": merged.get("seed"),
        "query_offset": merged.get("query_offset"),
        "loop": merged.get("loop"),
        "window": merged.get("window"),
        "global_window": merged.get("global_window"),
        "split": merged.get("split"),
        "profile": merged.get("profile"),
        "query_id": row.get("query_id"),
        "query_text": row.get("query_text"),
        "query_token_count": row.get("query_token_count"),
        "query_char_count": row.get("query_char_count"),
        "query_stopword_ratio": row.get("query_stopword_ratio"),
        "query_numeric_token_count": row.get("query_numeric_token_count"),
        "query_negation_count": row.get("query_negation_count"),
        "query_claim_cue_count": row.get("query_claim_cue_count"),
        "delta_ndcg_at_10": row.get("delta_ndcg_at_10"),
        "delta_mrr": row.get("delta_mrr"),
        "delta_recall_at_10": row.get("delta_recall_at_10"),
        "rank_delta": row.get("rank_delta"),
        "baseline_rank": row.get("baseline_rank"),
        "candidate_rank": row.get("candidate_rank"),
        "top10_overlap_with_baseline": row.get("top10_overlap_with_baseline"),
        "top_doc_changed": row.get("top_doc_changed"),
        "action": row.get("action"),
        "global_variance": row.get("global_variance"),
        "jaccard": row.get("jaccard"),
        "mask_density": row.get("mask_density"),
        "reformulation_variant_count": row.get("reformulation_variant_count"),
        "reformulation_changed": row.get("reformulation_changed"),
        "fault_class": row.get("fault_class"),
        "baseline_score_margin": row.get("baseline_score_margin"),
        "baseline_top_score": row.get("baseline_top_score"),
        "query_norm": row.get("query_norm"),
        "masked_rank": row.get("masked_rank"),
        "gate_applied": row.get("gate_applied"),
        "selected_delta_ndcg_at_10": row.get("selected_delta_ndcg_at_10"),
        "route": row.get("route"),
        "retrieval_intervention": row.get("retrieval_intervention"),
        "gate_decision": row.get("gate_decision"),
        "evaluator_outcome": row.get("evaluator_outcome"),
        "safety_outcome": row.get("safety_outcome"),
        "promotion_blocker": row.get("promotion_blocker"),
        "decision": row.get("decision"),
    }


def build_engine_scope_rows(
    *,
    query_attribution_rows: Iterable[Dict[str, Any]] | None = None,
    mask_example_rows: Iterable[Dict[str, Any]] | None = None,
    source_family: str,
    artifact_path: Path | None = None,
    context: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in query_attribution_rows or ():
        rows.append(
            _query_attribution_to_engine_scope_row(
                dict(row),
                source_family=source_family,
                artifact_path=artifact_path,
                context=context,
            )
        )
    for row in mask_example_rows or ():
        rows.append(
            _mask_example_to_engine_scope_row(
                dict(row),
                source_family=source_family,
                artifact_path=artifact_path,
                context=context,
            )
        )
    return rows


def _extract_engine_scope_rows(
    payload: Any,
    *,
    artifact_path: Path,
    context: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Extract legacy-derived Engine-Scope rows from artifacts without direct rows."""

    active_context = dict(context or {})
    rows: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in _CONTEXT_KEYS:
            if key in payload and payload[key] is not None:
                active_context[key] = payload[key]
        query_rows = payload.get("query_attribution_rows")
        if isinstance(query_rows, list):
            rows.extend(
                build_engine_scope_rows(
                    query_attribution_rows=[row for row in query_rows if isinstance(row, dict)],
                    source_family=str(active_context.get("source_family") or "legacy_query_attribution"),
                    artifact_path=artifact_path,
                    context=active_context,
                )
            )
        mask_rows = payload.get("mask_example_rows")
        if isinstance(mask_rows, list):
            rows.extend(
                build_engine_scope_rows(
                    mask_example_rows=[row for row in mask_rows if isinstance(row, dict)],
                    source_family=str(active_context.get("source_family") or "legacy_mask_probe"),
                    artifact_path=artifact_path,
                    context=active_context,
                )
            )
        for key, value in payload.items():
            if key in {"engine_scope_rows", "query_attribution_rows", "mask_example_rows"}:
                continue
            rows.extend(_extract_engine_scope_rows(value, artifact_path=artifact_path, context=active_context))
    elif isinstance(payload, list):
        for item in payload:
            rows.extend(_extract_engine_scope_rows(item, artifact_path=artifact_path, context=active_context))
    return rows


def _extract_direct_engine_scope_rows(
    payload: Any,
    *,
    artifact_path: Path,
    context: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    active_context = dict(context or {})
    rows: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in _CONTEXT_KEYS:
            if key in payload and payload[key] is not None:
                active_context[key] = payload[key]
        direct_rows = payload.get("engine_scope_rows")
        if isinstance(direct_rows, list):
            for row in direct_rows:
                if isinstance(row, dict):
                    rows.append(
                        _normalize_engine_scope_row(
                            row,
                            artifact_path=artifact_path,
                            context=active_context,
                        )
                    )
        for key, value in payload.items():
            if key == "engine_scope_rows":
                continue
            rows.extend(_extract_direct_engine_scope_rows(value, artifact_path=artifact_path, context=active_context))
    elif isinstance(payload, list):
        for item in payload:
            rows.extend(_extract_direct_engine_scope_rows(item, artifact_path=artifact_path, context=active_context))
    return rows


def load_engine_scope_rows(paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    """Load engine_scope_rows from modern or legacy artifacts."""

    combined: List[Dict[str, Any]] = []
    for path in _iter_artifact_paths(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        direct_rows = _extract_direct_engine_scope_rows(payload, artifact_path=path)
        if direct_rows:
            combined.extend(direct_rows)
            continue
        combined.extend(_extract_engine_scope_rows(payload, artifact_path=path))
    if not combined:
        raise ValueError("no engine_scope_rows found in the supplied artifacts")
    return combined


def summarize_engine_scope_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows_list = [dict(row) for row in rows]
    by_source_family: Dict[str, int] = {}
    by_row_type: Dict[str, int] = {}
    tasks = set()
    for row in rows_list:
        source_family = str(row.get("source_family") or "unknown")
        row_type = str(row.get("row_type") or "unknown")
        by_source_family[source_family] = by_source_family.get(source_family, 0) + 1
        by_row_type[row_type] = by_row_type.get(row_type, 0) + 1
        if row.get("task"):
            tasks.add(str(row["task"]))
    return {
        "schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
        "row_count": len(rows_list),
        "source_families": by_source_family,
        "row_types": by_row_type,
        "tasks": sorted(tasks),
    }


def engine_scope_rows_to_evidence_events(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Engine-Scope rows into the shared evidence-event contract."""

    return [evidence_event_from_engine_scope_row(row) for row in rows]
