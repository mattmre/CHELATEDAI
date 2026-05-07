"""Build the query-level attribution pool from road-course artifacts.

Reads all artifact JSONs from ``experiment_runs/roadcourse-small/`` and
normalizes four data sources into a unified pool artifact written to
``experiment_runs/attribution-pool/latest/attribution_pool.json``:

1. ``query_attribution_rows`` – per-query signal rows (delta NDCG, action,
   fault class, variance, jaccard, etc.) from probe artifacts.
2. ``gate_feature_rows`` – per-window/profile aggregates (chelate/reform/fast
   counts, variance means, promotion blockers) from gate-learning artifacts.
3. ``attnres_profile_rows`` – per-task × seed × profile retrieval metrics from
   attnres road-course artifacts.
4. ``mask_probe_rows`` – train / holdout / conditional split outcomes from
   classifier mask-probe artifacts.

The pool is the training surface for the Slice 2 learned reformulation gate.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, List, Optional

_DEFAULT_ARTIFACT_DIR = pathlib.Path("experiment_runs/roadcourse-small")
_DEFAULT_OUTPUT_DIR = pathlib.Path("experiment_runs/attribution-pool/latest")


# ---------------------------------------------------------------------------
# Internal collection helpers
# ---------------------------------------------------------------------------


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    """Load a JSON artifact from *path*."""
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_query_attribution_rows(
    artifact_dir: pathlib.Path,
) -> List[Dict[str, Any]]:
    """Gather ``query_attribution_rows`` from probe artifacts.

    Enriches each row with ``source_artifact`` and ``strategy`` fields
    derived from the artifact filename stem.  Rows from artifacts with
    ≤ 1 entries are skipped (single-row sentinel / research-pathway
    meta files are not real query data).

    Args:
        artifact_dir: Root directory to scan for ``*.json`` files.

    Returns:
        Flat list of enriched query attribution row dicts.
    """
    rows: List[Dict[str, Any]] = []
    for path in sorted(artifact_dir.glob("*.json")):
        try:
            art = _load_json(path)
        except Exception:
            continue
        raw = art.get("query_attribution_rows")
        if not isinstance(raw, list) or len(raw) <= 1:
            continue
        strategy = path.stem
        seed: Optional[int] = art.get("seed")
        for row in raw:
            enriched: Dict[str, Any] = {
                "source_artifact": path.name,
                "strategy": strategy,
            }
            enriched.update(row)
            enriched.setdefault("seed", seed)
            # Normalise optional rich-schema fields added in reform_policy probes
            enriched.setdefault("query_text", None)
            enriched.setdefault("query_token_count", None)
            enriched.setdefault("query_char_count", None)
            enriched.setdefault("query_stopword_ratio", None)
            enriched.setdefault("query_numeric_token_count", None)
            enriched.setdefault("query_negation_count", None)
            enriched.setdefault("query_claim_cue_count", None)
            rows.append(enriched)
    return rows


def _collect_gate_feature_rows(
    artifact_dir: pathlib.Path,
) -> List[Dict[str, Any]]:
    """Gather ``gate_feature_rows`` from gate-learning and probe artifacts.

    Args:
        artifact_dir: Root directory to scan for ``*.json`` files.

    Returns:
        Flat list of enriched gate feature row dicts.
    """
    rows: List[Dict[str, Any]] = []
    for path in sorted(artifact_dir.glob("*.json")):
        try:
            art = _load_json(path)
        except Exception:
            continue
        raw = art.get("gate_feature_rows")
        if not isinstance(raw, list) or len(raw) <= 1:
            continue
        strategy = path.stem
        seed: Optional[int] = art.get("seed")
        for row in raw:
            enriched: Dict[str, Any] = {
                "source_artifact": path.name,
                "strategy": strategy,
            }
            enriched.update(row)
            enriched.setdefault("seed", seed)
            rows.append(enriched)
    return rows


def _collect_attnres_profile_rows(
    artifact_dir: pathlib.Path,
) -> List[Dict[str, Any]]:
    """Gather per-seed × per-profile retrieval metrics from attnres artifacts.

    Sources: ``attnres_*.json`` files with a ``profile_results`` list.

    Args:
        artifact_dir: Root directory to scan for ``attnres_*.json`` files.

    Returns:
        Flat list of attnres profile row dicts.
    """
    rows: List[Dict[str, Any]] = []
    for path in sorted(artifact_dir.glob("attnres_*.json")):
        try:
            art = _load_json(path)
        except Exception:
            continue
        profile_results = art.get("profile_results")
        if not isinstance(profile_results, list):
            continue
        task: Optional[str] = art.get("task")
        seed: Optional[int] = art.get("seed")
        corpus_size: Optional[int] = art.get("corpus_size")
        query_count: Optional[int] = art.get("query_count")
        seed_gate: Any = art.get("seed_gate")
        quantization_survival: Any = art.get("quantization_survival")
        for pr in profile_results:
            metrics = pr.get("metrics") or {}
            action_mix = pr.get("action_mix") or {}
            rows.append(
                {
                    "source_artifact": path.name,
                    "task": task,
                    "seed": seed,
                    "corpus_size": corpus_size,
                    "query_count": query_count,
                    "profile": pr.get("profile"),
                    "ndcg_at_10": metrics.get("ndcg_at_10"),
                    "map_at_10": metrics.get("map_at_10"),
                    "mrr": metrics.get("mrr"),
                    "recall_at_10": metrics.get("recall_at_10"),
                    "evaluated_queries": metrics.get("evaluated_queries"),
                    "latency_ms_mean": pr.get("latency_ms_mean"),
                    "action_fast": action_mix.get("FAST", 0),
                    "action_chelate": action_mix.get("CHELATE", 0),
                    "action_reform": action_mix.get("REFORM", 0),
                    "seed_gate": seed_gate,
                    "quantization_survival": quantization_survival,
                }
            )
    return rows


def _extract_split_metrics(
    split_data: Any,
    split_name: str,
    source_artifact: str,
    task: Optional[str],
    seed: Optional[int],
    max_queries: Optional[int],
    mask: Any,
    promotion_candidate: bool,
    conditional_promotion_candidate: bool,
) -> List[Dict[str, Any]]:
    """Extract metric rows from a single train / holdout / conditional split.

    Args:
        split_data: Raw value of the split field (list or dict or None).
        split_name: Name of the split (``"train"``, ``"holdout"``, etc.).
        source_artifact: Filename of the originating artifact.
        task: Dataset task name.
        seed: Artifact seed.
        max_queries: Max queries used in the artifact run.
        mask: Raw mask value from the artifact.
        promotion_candidate: Whether the artifact is a promotion candidate.
        conditional_promotion_candidate: Conditional promotion flag.

    Returns:
        List of row dicts for this split.
    """
    rows: List[Dict[str, Any]] = []
    if split_data is None:
        return rows
    items: List[Any] = split_data if isinstance(split_data, list) else [split_data]
    mask_list = mask if isinstance(mask, list) else []
    mask_dims = len(mask_list) if mask_list else None
    mask_density = (sum(mask_list) / len(mask_list)) if mask_list else None
    for item in items:
        if not isinstance(item, dict):
            continue
        row: Dict[str, Any] = {
            "source_artifact": source_artifact,
            "task": task,
            "seed": seed,
            "max_queries": max_queries,
            "split": split_name,
            "promotion_candidate": promotion_candidate,
            "conditional_promotion_candidate": conditional_promotion_candidate,
            "mask_dims": mask_dims,
            "mask_density": mask_density,
        }
        for key in ("baseline", "masked"):
            sub = item.get(key) or {}
            for mkey in ("ndcg_at_10", "map_at_10", "mrr", "recall_at_10"):
                row[f"{key}_{mkey}"] = sub.get(mkey) if isinstance(sub, dict) else None
        row["delta_ndcg_at_10"] = item.get("delta_ndcg_at_10")
        rows.append(row)
    return rows


def _collect_mask_probe_rows(
    artifact_dir: pathlib.Path,
) -> List[Dict[str, Any]]:
    """Gather train / holdout / conditional outcomes from mask-probe artifacts.

    Sources: ``classifier_conditional_mask_*.json`` files.

    Args:
        artifact_dir: Root directory to scan.

    Returns:
        Flat list of mask probe row dicts.
    """
    rows: List[Dict[str, Any]] = []
    for path in sorted(artifact_dir.glob("classifier_conditional_mask_*.json")):
        try:
            art = _load_json(path)
        except Exception:
            continue
        task: Optional[str] = art.get("task")
        seed: Optional[int] = art.get("seed")
        max_queries: Optional[int] = art.get("max_queries")
        mask: Any = art.get("mask")
        promotion_candidate: bool = bool(art.get("promotion_candidate", False))
        cond_promotion: bool = bool(
            art.get("conditional_promotion_candidate", False)
        )
        kwargs = dict(
            source_artifact=path.name,
            task=task,
            seed=seed,
            max_queries=max_queries,
            mask=mask,
            promotion_candidate=promotion_candidate,
            conditional_promotion_candidate=cond_promotion,
        )
        for split_name in ("train", "holdout", "conditional"):
            rows.extend(
                _extract_split_metrics(
                    art.get(split_name), split_name, **kwargs
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_attribution_pool(
    artifact_dir: pathlib.Path = _DEFAULT_ARTIFACT_DIR,
    output_dir: pathlib.Path = _DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Build and write the attribution pool artifact.

    Collects all four data source tables from *artifact_dir* and writes a
    single ``attribution_pool.json`` artifact to *output_dir*.

    Args:
        artifact_dir: Directory containing road-course artifact ``*.json``
            files.  Defaults to ``experiment_runs/roadcourse-small``.
        output_dir: Output directory.  Defaults to
            ``experiment_runs/attribution-pool/latest``.

    Returns:
        The attribution pool dict (also written to disk).
    """
    query_rows = _collect_query_attribution_rows(artifact_dir)
    gate_rows = _collect_gate_feature_rows(artifact_dir)
    attnres_rows = _collect_attnres_profile_rows(artifact_dir)
    mask_probe_rows = _collect_mask_probe_rows(artifact_dir)

    tasks = sorted({r["task"] for r in query_rows if r.get("task")})
    profiles = sorted({r["profile"] for r in query_rows if r.get("profile")})
    fault_classes = sorted(
        {r["fault_class"] for r in query_rows if r.get("fault_class")}
    )

    pool: Dict[str, Any] = {
        "record_type": "attribution_pool",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_dir": str(artifact_dir),
        "query_attribution_rows": query_rows,
        "gate_feature_rows": gate_rows,
        "attnres_profile_rows": attnres_rows,
        "mask_probe_rows": mask_probe_rows,
        "summary": {
            "query_attribution_count": len(query_rows),
            "gate_feature_count": len(gate_rows),
            "attnres_profile_count": len(attnres_rows),
            "mask_probe_count": len(mask_probe_rows),
            "tasks": tasks,
            "profiles": profiles,
            "fault_classes": fault_classes,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "attribution_pool.json"
    output_path.write_text(
        json.dumps(pool, indent=2, default=str), encoding="utf-8"
    )
    return pool


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pool = build_attribution_pool()
    s = pool["summary"]
    print("Attribution pool built:")
    print(f"  query_attribution_rows : {s['query_attribution_count']}")
    print(f"  gate_feature_rows      : {s['gate_feature_count']}")
    print(f"  attnres_profile_rows   : {s['attnres_profile_count']}")
    print(f"  mask_probe_rows        : {s['mask_probe_count']}")
    print(f"  tasks                  : {s['tasks']}")
    print(f"  profiles               : {s['profiles']}")
    print(f"  fault_classes          : {s['fault_classes']}")
    print(f"Written to: {_DEFAULT_OUTPUT_DIR / 'attribution_pool.json'}")
