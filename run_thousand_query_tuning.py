"""Adaptive road-course tuning cycle with 50-query checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from benchmark_utils import (
    canonicalize_id,
    isolated_adapter_state,
    load_mteb_data,
    map_predicted_ids,
    mean_average_precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from run_road_course_campaign import RoadCourseProfile, _profile_engine, evaluate_rankings
from run_road_course_tuning_loop import classify_profile_behavior, profile_summary
from query_reformulator import query_lexical_features


@dataclass(frozen=True)
class LoopSpec:
    task: str
    seed: int
    query_offset: int = 0


DEFAULT_LOOP_SPECS = [
    LoopSpec("SciFact", seed=46, query_offset=0),
    LoopSpec("NFCorpus", seed=46, query_offset=0),
    LoopSpec("FiQA2018", seed=46, query_offset=0),
    LoopSpec("FiQA2018", seed=47, query_offset=200),
    LoopSpec("FiQA2018", seed=48, query_offset=400),
]

PHASE_LOOP_TEMPLATES = [
    ("SciFact", 0),
    ("SciFact", 100),
    ("NFCorpus", 0),
    ("NFCorpus", 100),
    ("FiQA2018", 0),
    ("FiQA2018", 200),
    ("FiQA2018", 400),
]

BASELINE = RoadCourseProfile("baseline")
GUARD = RoadCourseProfile("guard_p85_t0.01", use_quantization=True, chelation_p=85, chelation_threshold=0.01)

PROFILE_LIBRARY = {
    "adaptive_p85_t0.0015": RoadCourseProfile(
        "adaptive_p85_t0.0015",
        use_quantization=True,
        chelation_p=85,
        chelation_threshold=0.0015,
    ),
    "adaptive_p85_t0.002": RoadCourseProfile(
        "adaptive_p85_t0.002",
        use_quantization=True,
        chelation_p=85,
        chelation_threshold=0.002,
    ),
    "adaptive_p85_t0.0025": RoadCourseProfile(
        "adaptive_p85_t0.0025",
        use_quantization=True,
        chelation_p=85,
        chelation_threshold=0.0025,
    ),
    "adaptive_p90_t0.002": RoadCourseProfile(
        "adaptive_p90_t0.002",
        use_quantization=True,
        chelation_p=90,
        chelation_threshold=0.002,
    ),
    "adaptive_p85_t0.003": RoadCourseProfile(
        "adaptive_p85_t0.003",
        use_quantization=True,
        chelation_p=85,
        chelation_threshold=0.003,
    ),
    "adaptive_p50_t0.002": RoadCourseProfile(
        "adaptive_p50_t0.002",
        use_quantization=True,
        chelation_p=50,
        chelation_threshold=0.002,
    ),
    "adaptive_p95_t0.002": RoadCourseProfile(
        "adaptive_p95_t0.002",
        use_quantization=True,
        chelation_p=95,
        chelation_threshold=0.002,
    ),
    "adaptive_p99_t0.0015": RoadCourseProfile(
        "adaptive_p99_t0.0015",
        use_quantization=True,
        chelation_p=99,
        chelation_threshold=0.0015,
    ),
    "adaptive_p99_t0.002": RoadCourseProfile(
        "adaptive_p99_t0.002",
        use_quantization=True,
        chelation_p=99,
        chelation_threshold=0.002,
    ),
    "adaptive_p99_t0.0025": RoadCourseProfile(
        "adaptive_p99_t0.0025",
        use_quantization=True,
        chelation_p=99,
        chelation_threshold=0.0025,
    ),
    "reform_rrf_v2": RoadCourseProfile("reform_rrf_v2", query_reformulation_variants=2),
    "selective_reform_rrf_v2": RoadCourseProfile(
        "selective_reform_rrf_v2",
        query_reformulation_variants=2,
        query_reformulation_policy="selective_low_specificity",
    ),
    "reform_high_specificity_rrf_v2": RoadCourseProfile(
        "reform_high_specificity_rrf_v2",
        query_reformulation_variants=2,
        query_reformulation_policy="selective_high_specificity",
    ),
    "reform_claim_cue_rrf_v2": RoadCourseProfile(
        "reform_claim_cue_rrf_v2",
        query_reformulation_variants=2,
        query_reformulation_policy="selective_claim_cue",
    ),
    "adaptive_p85_t0.002_reform_rrf_v2": RoadCourseProfile(
        "adaptive_p85_t0.002_reform_rrf_v2",
        use_quantization=True,
        chelation_p=85,
        chelation_threshold=0.002,
        query_reformulation_variants=2,
    ),
    "adaptive_p85_t0.0025_reform_rrf_v2": RoadCourseProfile(
        "adaptive_p85_t0.0025_reform_rrf_v2",
        use_quantization=True,
        chelation_p=85,
        chelation_threshold=0.0025,
        query_reformulation_variants=2,
    ),
}

FAULT_AWARE_ACTIVE_PROBES = {
    "SciFact": [
        "adaptive_p99_t0.002",
        "adaptive_p95_t0.002",
        "adaptive_p99_t0.0015",
        "adaptive_p85_t0.0025",
    ],
    "NFCorpus": [
        "adaptive_p99_t0.0025",
        "adaptive_p99_t0.002",
        "adaptive_p85_t0.0025",
        "adaptive_p85_t0.003",
    ],
    "FiQA2018": [
        "adaptive_p99_t0.002",
        "adaptive_p99_t0.0015",
        "adaptive_p85_t0.002",
        "adaptive_p85_t0.0025",
    ],
}

FAULT_AWARE_REFORM_PROBES = {
    "SciFact": ["reform_rrf_v2"],
    "NFCorpus": ["reform_rrf_v2"],
    "FiQA2018": ["reform_rrf_v2"],
}

GATE_LEARNING_ACTIVE_PROBES = {
    "SciFact": ["adaptive_p99_t0.0015", "adaptive_p95_t0.002", "adaptive_p85_t0.002"],
    "NFCorpus": ["adaptive_p99_t0.0015", "adaptive_p99_t0.0025", "adaptive_p85_t0.0025"],
    "FiQA2018": ["adaptive_p99_t0.0015", "adaptive_p85_t0.002", "adaptive_p99_t0.002"],
}


def select_query_window(
    corpus: Mapping[Any, str],
    queries: Mapping[Any, str],
    qrels: Mapping[Any, Mapping[Any, float]],
    query_offset: int,
    max_queries: int,
    sample_docs: int,
    seed: int,
) -> tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, float]]]:
    """Select a deterministic judged query window and preserve all relevant docs."""

    judged_queries = []
    for query_id, query_text in queries.items():
        canonical_query_id = canonicalize_id(query_id)
        relevance = {
            canonicalize_id(doc_id): float(score)
            for doc_id, score in qrels.get(query_id, qrels.get(canonical_query_id, {})).items()
            if float(score) > 0
        }
        if relevance:
            judged_queries.append((canonical_query_id, str(query_text), relevance))

    if query_offset >= len(judged_queries):
        raise ValueError(f"query_offset {query_offset} exceeds judged query count {len(judged_queries)}")

    selected = judged_queries[query_offset: query_offset + max_queries]
    if len(selected) < max_queries:
        raise ValueError(
            f"requested {max_queries} queries at offset {query_offset}, but only {len(selected)} judged queries are available"
        )

    selected_queries = {query_id: query_text for query_id, query_text, _relevance in selected}
    selected_qrels = {query_id: relevance for query_id, _query_text, relevance in selected}

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


def split_query_windows(
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, float]],
    window_queries: int,
) -> List[tuple[Dict[str, str], Dict[str, Dict[str, float]]]]:
    items = list(queries.items())
    if len(items) % window_queries != 0:
        raise ValueError("selected query count must divide evenly into validation windows")

    windows = []
    for start in range(0, len(items), window_queries):
        window_items = items[start: start + window_queries]
        window_query_ids = {query_id for query_id, _query_text in window_items}
        windows.append((
            dict(window_items),
            {query_id: qrels[query_id] for query_id in window_query_ids},
        ))
    return windows


def _profile_by_name(name: str) -> RoadCourseProfile:
    return PROFILE_LIBRARY[name]


def _same_task_profile_rows(
    previous_windows: List[Dict[str, Any]] | None,
    task: str | None,
) -> Dict[str, List[Dict[str, Any]]]:
    rows: Dict[str, List[Dict[str, Any]]] = {}
    if not previous_windows or task is None:
        return rows
    for window in previous_windows:
        if window.get("task") != task:
            continue
        for row in window.get("summary", {}).get("ranked_profiles", []):
            rows.setdefault(row["profile"], []).append(row)
    return rows


def _fault_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    positives = 0
    negatives = 0
    deltas = []
    for row in rows:
        deltas.append(float(row.get("delta_vs_baseline", 0.0)))
        fault = row.get("fault_classification") or classify_profile_behavior(row)
        if fault["fault_class"] == "actuator_active_positive":
            positives += 1
        elif fault["fault_class"] == "actuator_active_negative":
            negatives += 1
    return {
        "positives": positives,
        "negatives": negatives,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
    }


def _fault_aware_probe(
    candidates: List[str],
    profile_rows: Dict[str, List[Dict[str, Any]]],
    window_index: int,
) -> str:
    scored = []
    for index, candidate in enumerate(candidates):
        rows = profile_rows.get(candidate, [])
        stats = _fault_stats(rows)
        if rows and stats["positives"] > stats["negatives"] and stats["mean_delta"] > 0:
            scored.append((0, -stats["mean_delta"], index, candidate))
        elif rows and stats["negatives"] > stats["positives"]:
            scored.append((2, stats["negatives"] - stats["positives"], index, candidate))
        else:
            scored.append((1, len(rows), index, candidate))
    scored.sort()
    positive = [candidate for rank, _score, _index, candidate in scored if rank == 0]
    if positive:
        return positive[0]
    viable = [
        candidate for rank, _score, _index, candidate in scored
        if rank == 1
    ]
    if viable:
        return viable[(window_index - 1) % len(viable)]
    return candidates[(window_index - 1) % len(candidates)]


def _profile_cache_key(profile: RoadCourseProfile) -> str:
    return json.dumps(asdict(profile), sort_keys=True)


def _dedupe_profiles(profiles: Iterable[RoadCourseProfile]) -> List[RoadCourseProfile]:
    deduped = []
    seen = set()
    for profile in profiles:
        if profile.name not in seen:
            seen.add(profile.name)
            deduped.append(profile)
    return deduped


def query_metric_row(
    query_id: str,
    retrieved_ids: List[str],
    relevance: Mapping[str, float],
    *,
    k: int = 10,
) -> Dict[str, Any]:
    """Compute per-query retrieval metrics for attribution rows."""

    relevant_ids = {doc_id for doc_id, score in relevance.items() if score > 0}
    retrieved = [canonicalize_id(doc_id) for doc_id in retrieved_ids[:k]]
    relevance_by_rank = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved]
    first_relevant_rank = next(
        (rank for rank, doc_id in enumerate(retrieved, start=1) if doc_id in relevant_ids),
        None,
    )
    return {
        "query_id": query_id,
        "ndcg_at_10": float(ndcg_at_k(relevance_by_rank, k)),
        "map_at_10": float(mean_average_precision_at_k(retrieved, relevant_ids, k)),
        "mrr": float(mean_reciprocal_rank(retrieved, relevant_ids)),
        "recall_at_10": float(recall_at_k(retrieved, relevant_ids, k)),
        "first_relevant_rank": first_relevant_rank,
        "relevant_count": len(relevant_ids),
        "retrieved_relevant_count": sum(1 for doc_id in retrieved if doc_id in relevant_ids),
        "top_doc_id": retrieved[0] if retrieved else None,
    }


def load_gate_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"rules": []}
    gate_config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(gate_config.get("rules"), list):
        raise ValueError("gate config must contain a rules list")
    return gate_config


def _profiles_from_gate_config(gate_config: Dict[str, Any], task: str | None) -> List[RoadCourseProfile]:
    selected = []
    for rule in gate_config.get("rules", []):
        profile = rule.get("profile")
        if profile not in PROFILE_LIBRARY:
            continue
        rule_task = rule.get("task")
        if rule_task is None or rule_task == task:
            selected.append(_profile_by_name(profile))
    return _dedupe_profiles(selected)


def evaluate_profile_with_cache(
    profile: RoadCourseProfile,
    model_name: str,
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, float]],
    engine_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a profile against a query window while reusing its loop-local engine."""

    start = time.perf_counter()
    key = _profile_cache_key(profile)
    if key not in engine_cache:
        engine_cache[key] = _profile_engine(profile, model_name, corpus)
    engine = engine_cache[key]

    rankings: Dict[str, list[str]] = {}
    query_diagnostics: Dict[str, Dict[str, Any]] = {}
    action_mix: Dict[str, int] = {}
    latencies = []
    variances = []
    jaccards = []
    mask_densities = []
    reformulation_variant_counts = []
    reformulation_changed_count = 0
    for query_id, query_text in queries.items():
        query_start = time.perf_counter()
        _std_top, final_top, mask, jaccard = engine.run_inference(query_text)
        latencies.append((time.perf_counter() - query_start) * 1000.0)
        rankings[query_id] = map_predicted_ids(engine, final_top[:10])
        diagnostics = engine.get_last_runtime_diagnostics() or {}
        runtime = diagnostics.get("runtime", {})
        action = runtime.get("action", "unknown")
        action_mix[action] = action_mix.get(action, 0) + 1
        if runtime.get("global_variance") is not None:
            variances.append(float(runtime["global_variance"]))
        if jaccard is not None:
            jaccards.append(float(jaccard))
        if mask is not None:
            mask_densities.append(float(np.mean(mask)))
        reformulation = diagnostics.get("query_reformulation") or {}
        if reformulation:
            reformulation_variant_counts.append(float(reformulation.get("variant_count", 0)))
            fusion = reformulation.get("fusion") or {}
            if fusion.get("fused_changed"):
                reformulation_changed_count += 1
        query_diagnostics[query_id] = {
            "action": action,
            "latency_ms": latencies[-1],
            "global_variance": runtime.get("global_variance"),
            "jaccard": float(jaccard) if jaccard is not None else None,
            "mask_density": float(np.mean(mask)) if mask is not None else None,
            "reformulation_variant_count": (
                int(reformulation.get("variant_count", 0))
                if reformulation
                else 0
            ),
            "reformulation_changed": bool((reformulation.get("fusion") or {}).get("fused_changed"))
            if reformulation
            else False,
        }

    metrics = evaluate_rankings(rankings, qrels)
    query_metrics = {
        query_id: {
            **query_metric_row(query_id, rankings.get(query_id, []), relevance),
            **query_diagnostics.get(query_id, {}),
        }
        for query_id, relevance in qrels.items()
    }
    return {
        "profile": profile.name,
        "controls": asdict(profile),
        "metrics": metrics,
        "rankings": rankings,
        "query_metrics": query_metrics,
        "action_mix": action_mix,
        "control_diagnostics": {
            "variance_min": float(np.min(variances)) if variances else None,
            "variance_mean": float(np.mean(variances)) if variances else None,
            "variance_max": float(np.max(variances)) if variances else None,
            "jaccard_mean": float(np.mean(jaccards)) if jaccards else None,
            "mask_density_mean": float(np.mean(mask_densities)) if mask_densities else None,
            "reformulation_variant_count_mean": (
                float(np.mean(reformulation_variant_counts))
                if reformulation_variant_counts
                else None
            ),
            "reformulation_changed_count": reformulation_changed_count,
        },
        "latency_ms_mean": float(np.mean(latencies)) if latencies else 0.0,
        "elapsed_seconds": time.perf_counter() - start,
        "telemetry": engine.get_runtime_telemetry(),
    }


def choose_window_profiles(
    previous_summaries: List[Dict[str, Any]],
    window_index: int,
    task: str | None = None,
    previous_windows: List[Dict[str, Any]] | None = None,
    strategy: str = "adaptive",
    gate_config: Dict[str, Any] | None = None,
) -> List[RoadCourseProfile]:
    """Choose a compact adaptive profile set for the next 50-query window."""

    profiles = [BASELINE, GUARD]
    if strategy == "confirm_fiqa":
        if task == "FiQA2018":
            return profiles + [
                _profile_by_name("adaptive_p85_t0.002"),
                _profile_by_name("adaptive_p85_t0.002_reform_rrf_v2"),
            ]
        if task == "SciFact":
            probe = "adaptive_p95_t0.002" if window_index % 2 == 0 else "reform_rrf_v2"
        else:
            probe = "adaptive_p85_t0.002" if window_index % 2 == 0 else "reform_rrf_v2"
        return profiles + [_profile_by_name(probe)]

    if strategy == "fault_aware":
        profile_rows = _same_task_profile_rows(previous_windows, task)
        active_candidates = FAULT_AWARE_ACTIVE_PROBES.get(task or "", FAULT_AWARE_ACTIVE_PROBES["SciFact"])
        profiles.append(_profile_by_name(_fault_aware_probe(active_candidates, profile_rows, window_index)))
        if window_index % 3 == 0:
            reform_candidates = FAULT_AWARE_REFORM_PROBES.get(task or "", ["reform_rrf_v2"])
            profiles.append(_profile_by_name(_fault_aware_probe(reform_candidates, profile_rows, window_index)))
        return _dedupe_profiles(profiles)

    if strategy == "gate_learning":
        profile_rows = _same_task_profile_rows(previous_windows, task)
        active_candidates = GATE_LEARNING_ACTIVE_PROBES.get(task or "", GATE_LEARNING_ACTIVE_PROBES["SciFact"])
        profiles.append(_profile_by_name("adaptive_p99_t0.0015"))
        profiles.append(_profile_by_name(_fault_aware_probe(active_candidates, profile_rows, window_index)))
        if window_index % 2 == 0:
            profiles.append(_profile_by_name("adaptive_p85_t0.002"))
        if window_index % 3 == 0:
            profiles.append(_profile_by_name("reform_rrf_v2"))
        return _dedupe_profiles(profiles)

    if strategy == "learned_gate":
        learned_profiles = _profiles_from_gate_config(gate_config or {"rules": []}, task)
        return _dedupe_profiles(profiles + learned_profiles)

    if strategy == "selective_reform":
        return _dedupe_profiles(profiles + [
            _profile_by_name("reform_rrf_v2"),
            _profile_by_name("selective_reform_rrf_v2"),
        ])

    if strategy == "reform_policy_search":
        return _dedupe_profiles(profiles + [
            _profile_by_name("reform_rrf_v2"),
            _profile_by_name("selective_reform_rrf_v2"),
            _profile_by_name("reform_high_specificity_rrf_v2"),
            _profile_by_name("reform_claim_cue_rrf_v2"),
        ])

    if not previous_summaries:
        return profiles + [
            _profile_by_name("adaptive_p85_t0.002"),
            _profile_by_name("reform_rrf_v2"),
        ]

    source_summary = previous_summaries[-1]
    if task is not None and previous_windows:
        same_task_windows = [
            window for window in previous_windows
            if window["task"] == task
        ]
        if same_task_windows:
            source_summary = same_task_windows[-1]["summary"]

    last_rows = source_summary["ranked_profiles"]
    active_rows = [
        row for row in last_rows
        if row["action_mix"].get("CHELATE", 0) > 0
    ]
    reform_rows = [
        row for row in last_rows
        if row["action_mix"].get("REFORMULATE", 0) > 0
    ]

    best_active_delta = max((row["delta_vs_baseline"] for row in active_rows), default=None)
    best_reform_delta = max((row["delta_vs_baseline"] for row in reform_rows), default=None)

    if best_active_delta is None:
        active_probe = "adaptive_p85_t0.002"
    elif best_active_delta > 0.001:
        active_probe = "adaptive_p85_t0.0025" if window_index % 2 == 0 else "adaptive_p90_t0.002"
    elif best_active_delta < -0.01:
        active_probe = "adaptive_p85_t0.003"
    elif best_active_delta < -0.002:
        active_probe = "adaptive_p95_t0.002"
    else:
        active_probe = "adaptive_p85_t0.002" if window_index % 2 == 0 else "adaptive_p50_t0.002"
    profiles.append(_profile_by_name(active_probe))

    if best_reform_delta is None:
        reform_probe = "reform_rrf_v2"
    elif best_reform_delta >= 0.001:
        reform_probe = "adaptive_p85_t0.0025_reform_rrf_v2"
    elif best_reform_delta >= -0.002:
        reform_probe = "adaptive_p85_t0.002_reform_rrf_v2"
    elif window_index % 4 == 0:
        reform_probe = "adaptive_p85_t0.002_reform_rrf_v2"
    else:
        reform_probe = "reform_rrf_v2"
    profiles.append(_profile_by_name(reform_probe))

    return _dedupe_profiles(profiles)


def build_gate_feature_rows(window_results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten per-window profile diagnostics into gate-training rows."""

    feature_rows = []
    for window in window_results:
        for row in window["summary"]["ranked_profiles"]:
            diagnostics = row.get("control_diagnostics") or {}
            fault = row.get("fault_classification") or classify_profile_behavior(row)
            action_mix = row.get("action_mix") or {}
            feature_rows.append({
                "loop": window["loop"],
                "window": window["window"],
                "global_window": window["global_window"],
                "task": window["task"],
                "profile": row["profile"],
                "delta_vs_baseline": float(row["delta_vs_baseline"]),
                "ndcg_at_10": float(row["ndcg_at_10"]),
                "fault_class": fault["fault_class"],
                "promotion_blocker": bool(fault["promotion_blocker"]),
                "chelate_count": int(action_mix.get("CHELATE", 0)) + int(action_mix.get("CHELATE_ALWAYS", 0)),
                "reformulate_count": int(action_mix.get("REFORMULATE", 0)),
                "fast_count": int(action_mix.get("FAST", 0)),
                "variance_mean": diagnostics.get("variance_mean"),
                "jaccard_mean": diagnostics.get("jaccard_mean"),
                "mask_density_mean": diagnostics.get("mask_density_mean"),
                "reformulation_changed_count": int(diagnostics.get("reformulation_changed_count") or 0),
            })
    return feature_rows


def _rank_delta(candidate_rank: int | None, baseline_rank: int | None) -> int | None:
    if candidate_rank is None and baseline_rank is None:
        return None
    if candidate_rank is None:
        return 999
    if baseline_rank is None:
        return -999
    return candidate_rank - baseline_rank


def build_query_attribution_rows(window_results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten per-query profile outcomes for learned gates and actuator debugging."""

    attribution_rows = []
    for window in window_results:
        baseline_result = next(
            (result for result in window.get("profile_results", []) if result.get("profile") == "baseline"),
            None,
        )
        baseline_query_metrics = (baseline_result or {}).get("query_metrics", {})
        for result in window.get("profile_results", []):
            profile = result["profile"]
            fault_by_profile = {
                row["profile"]: row.get("fault_classification") or classify_profile_behavior(row)
                for row in window["summary"]["ranked_profiles"]
            }
            for query_id, query_row in result.get("query_metrics", {}).items():
                query_text = (window.get("queries") or {}).get(query_id, "")
                lexical_features = query_lexical_features(query_text)
                baseline_row = baseline_query_metrics.get(query_id, {})
                baseline_ranking = (baseline_result or {}).get("rankings", {}).get(query_id, [])
                candidate_ranking = result.get("rankings", {}).get(query_id, [])
                top_overlap = len(set(candidate_ranking[:10]) & set(baseline_ranking[:10]))
                attribution_rows.append({
                    "loop": window["loop"],
                    "window": window["window"],
                    "global_window": window["global_window"],
                    "task": window["task"],
                    "query_id": query_id,
                    "query_text": query_text,
                    "query_token_count": lexical_features["token_count"],
                    "query_char_count": lexical_features["char_count"],
                    "query_stopword_ratio": lexical_features["stopword_ratio"],
                    "query_numeric_token_count": lexical_features["numeric_token_count"],
                    "query_negation_count": lexical_features["negation_count"],
                    "query_claim_cue_count": lexical_features["claim_cue_count"],
                    "profile": profile,
                    "delta_ndcg_at_10": (
                        float(query_row["ndcg_at_10"]) - float(baseline_row.get("ndcg_at_10", 0.0))
                    ),
                    "delta_mrr": float(query_row["mrr"]) - float(baseline_row.get("mrr", 0.0)),
                    "delta_recall_at_10": (
                        float(query_row["recall_at_10"]) - float(baseline_row.get("recall_at_10", 0.0))
                    ),
                    "rank_delta": _rank_delta(
                        query_row.get("first_relevant_rank"),
                        baseline_row.get("first_relevant_rank"),
                    ),
                    "top10_overlap_with_baseline": top_overlap,
                    "top_doc_changed": query_row.get("top_doc_id") != baseline_row.get("top_doc_id"),
                    "action": query_row.get("action"),
                    "global_variance": query_row.get("global_variance"),
                    "jaccard": query_row.get("jaccard"),
                    "mask_density": query_row.get("mask_density"),
                    "reformulation_variant_count": query_row.get("reformulation_variant_count", 0),
                    "reformulation_changed": bool(query_row.get("reformulation_changed", False)),
                    "fault_class": fault_by_profile.get(profile, {}).get("fault_class"),
                })
    return attribution_rows


def _gate_subset_summary(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    deltas = [float(row["delta_vs_baseline"]) for row in rows]
    fault_counts: Dict[str, int] = {}
    for row in rows:
        fault = row["fault_class"]
        fault_counts[fault] = fault_counts.get(fault, 0) + 1
    return {
        "gate": name,
        "windows": len(rows),
        "mean_delta_vs_baseline": float(np.mean(deltas)) if deltas else 0.0,
        "best_delta_vs_baseline": float(np.max(deltas)) if deltas else 0.0,
        "worst_delta_vs_baseline": float(np.min(deltas)) if deltas else 0.0,
        "improved": sum(delta > 0.001 for delta in deltas),
        "tied": sum(abs(delta) <= 0.001 for delta in deltas),
        "regressed": sum(delta < -0.001 for delta in deltas),
        "fault_counts": fault_counts,
        "shippable_gate_candidate": (
            len(rows) >= 3
            and bool(deltas)
            and float(np.mean(deltas)) > 0.001
            and all(delta >= -0.001 for delta in deltas)
            and fault_counts.get("actuator_active_negative", 0) == 0
            and fault_counts.get("metric_changed_without_actuator", 0) == 0
        ),
    }


def summarize_gate_candidates(feature_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Build simple post-hoc diagnostic gates from campaign feature rows."""

    rows_by_profile: Dict[str, List[Dict[str, Any]]] = {}
    for row in feature_rows:
        if row["profile"] in {"baseline", "guard_p85_t0.01"}:
            continue
        rows_by_profile.setdefault(row["profile"], []).append(row)

    report: Dict[str, Any] = {}
    for profile, rows in rows_by_profile.items():
        subsets = [_gate_subset_summary("all", rows)]
        for task in sorted({row["task"] for row in rows}):
            task_rows = [row for row in rows if row["task"] == task]
            subsets.append(_gate_subset_summary(f"task:{task}", task_rows))

        numeric_gates = [
            ("jaccard_mean", 0.9),
            ("mask_density_mean", 0.98),
            ("variance_mean", 0.002),
        ]
        for key, threshold in numeric_gates:
            values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
            if not values:
                continue
            low_rows = [row for row in rows if isinstance(row.get(key), (int, float)) and float(row[key]) < threshold]
            high_rows = [row for row in rows if isinstance(row.get(key), (int, float)) and float(row[key]) >= threshold]
            if low_rows:
                subsets.append(_gate_subset_summary(f"{key}< {threshold:g}", low_rows))
            if high_rows:
                subsets.append(_gate_subset_summary(f"{key}>= {threshold:g}", high_rows))

        changed_rows = [row for row in rows if row["reformulation_changed_count"] > 0]
        unchanged_rows = [row for row in rows if row["reformulation_changed_count"] == 0]
        if changed_rows:
            subsets.append(_gate_subset_summary("reformulation_changed", changed_rows))
        if unchanged_rows:
            subsets.append(_gate_subset_summary("reformulation_unchanged", unchanged_rows))

        subsets.sort(key=lambda item: (not item["shippable_gate_candidate"], -item["mean_delta_vs_baseline"]))
        report[profile] = {
            "best_gate": subsets[0],
            "candidate_gates": subsets,
        }
    return report


def summarize_profile_outcomes(window_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate profile directional outcomes across validation windows."""

    profile_rows: Dict[str, Dict[str, Any]] = {}
    for window in window_results:
        for row in window["summary"]["ranked_profiles"]:
            profile = row["profile"]
            current = profile_rows.setdefault(
                profile,
                {
                    "windows": 0,
                    "improved": 0,
                    "tied": 0,
                    "regressed": 0,
                    "deltas": [],
                    "action_mix": {},
                    "mask_density_values": [],
                    "reformulation_changed_total": 0,
                    "fault_counts": {},
                    "tasks": set(),
                },
            )
            delta = float(row["delta_vs_baseline"])
            current["windows"] += 1
            current["deltas"].append(delta)
            current["tasks"].add(window["task"])
            if delta > 0.001:
                current["improved"] += 1
            elif delta < -0.001:
                current["regressed"] += 1
            else:
                current["tied"] += 1
            for action, count in row["action_mix"].items():
                current["action_mix"][action] = current["action_mix"].get(action, 0) + int(count)
            diagnostics = row.get("control_diagnostics") or {}
            if diagnostics.get("mask_density_mean") is not None:
                current["mask_density_values"].append(float(diagnostics["mask_density_mean"]))
            current["reformulation_changed_total"] += int(diagnostics.get("reformulation_changed_count") or 0)
            fault = row.get("fault_classification") or classify_profile_behavior(row)
            fault_class = fault["fault_class"]
            current["fault_counts"][fault_class] = current["fault_counts"].get(fault_class, 0) + 1

    summary = {}
    for profile, row in profile_rows.items():
        deltas = row.pop("deltas")
        mask_values = row.pop("mask_density_values")
        tasks = sorted(row.pop("tasks"))
        summary[profile] = {
            **row,
            "tasks": tasks,
            "mean_delta_vs_baseline": float(np.mean(deltas)) if deltas else 0.0,
            "best_delta_vs_baseline": float(np.max(deltas)) if deltas else 0.0,
            "worst_delta_vs_baseline": float(np.min(deltas)) if deltas else 0.0,
            "mask_density_mean": float(np.mean(mask_values)) if mask_values else None,
        }
    return summary


def run_thousand_query_cycle(
    model: str,
    loop_specs: List[LoopSpec],
    loop_queries: int,
    window_queries: int,
    sample_docs: int,
    checkpoint_path: Path | None = None,
    strategy: str = "adaptive",
    gate_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if loop_queries % window_queries != 0:
        raise ValueError("loop_queries must divide evenly by window_queries")
    if strategy not in {
        "adaptive",
        "confirm_fiqa",
        "fault_aware",
        "gate_learning",
        "learned_gate",
        "selective_reform",
        "reform_policy_search",
    }:
        raise ValueError(
            "strategy must be 'adaptive', 'confirm_fiqa', 'fault_aware', 'gate_learning', "
            "'learned_gate', 'selective_reform', or 'reform_policy_search'"
        )

    all_window_results = []
    loops = []
    previous_summaries: List[Dict[str, Any]] = []
    for loop_index, spec in enumerate(loop_specs, start=1):
        corpus, queries, qrels = load_mteb_data(spec.task)
        if not corpus or not queries or not qrels:
            raise RuntimeError(f"Failed to load task {spec.task}")
        sliced_corpus, sliced_queries, sliced_qrels = select_query_window(
            corpus,
            queries,
            qrels,
            query_offset=spec.query_offset,
            max_queries=loop_queries,
            sample_docs=sample_docs,
            seed=spec.seed,
        )
        windows = split_query_windows(sliced_queries, sliced_qrels, window_queries)
        loop_windows = []
        with isolated_adapter_state():
            engine_cache: Dict[str, Any] = {}
            try:
                for local_window_index, (window_queries_map, window_qrels) in enumerate(windows, start=1):
                    global_window_index = len(all_window_results) + 1
                    profiles = choose_window_profiles(
                        previous_summaries,
                        global_window_index,
                        task=spec.task,
                        previous_windows=all_window_results,
                        strategy=strategy,
                        gate_config=gate_config,
                    )
                    profile_results = [
                        evaluate_profile_with_cache(
                            profile,
                            model,
                            sliced_corpus,
                            window_queries_map,
                            window_qrels,
                            engine_cache,
                        )
                        for profile in profiles
                    ]
                    summary = profile_summary(profile_results)
                    window_record = {
                        "loop": loop_index,
                        "task": spec.task,
                        "seed": spec.seed,
                        "query_offset": spec.query_offset,
                        "window": local_window_index,
                        "global_window": global_window_index,
                        "query_count": len(window_queries_map),
                        "corpus_size": len(sliced_corpus),
                        "queries": window_queries_map,
                        "profiles_evaluated": [asdict(profile) for profile in profiles],
                        "summary": summary,
                        "profile_results": profile_results,
                    }
                    previous_summaries.append(summary)
                    all_window_results.append(window_record)
                    loop_windows.append(window_record)
                    if checkpoint_path is not None:
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint_path.write_text(
                            json.dumps({
                                "status": "running",
                                "completed_windows": len(all_window_results),
                                "completed_queries": sum(window["query_count"] for window in all_window_results),
                                "loops": loops + [{
                                    "loop": loop_index,
                                    "task": spec.task,
                                    "seed": spec.seed,
                                    "query_offset": spec.query_offset,
                                    "query_count": sum(window["query_count"] for window in loop_windows),
                                    "windows": loop_windows,
                                }],
                                "profile_outcomes": summarize_profile_outcomes(all_window_results),
                                "gate_feature_rows": build_gate_feature_rows(all_window_results),
                                "query_attribution_rows": build_query_attribution_rows(all_window_results),
                                "gate_candidate_report": summarize_gate_candidates(
                                    build_gate_feature_rows(all_window_results)
                                ),
                            }, indent=2),
                            encoding="utf-8",
                        )
                        print(json.dumps({
                            "completed_windows": len(all_window_results),
                            "completed_queries": sum(window["query_count"] for window in all_window_results),
                            "task": spec.task,
                            "window": local_window_index,
                            "best_profile": summary["best_profile"],
                            "best_delta": summary["best_delta_vs_baseline"],
                        }))
            finally:
                for engine in engine_cache.values():
                    engine.close()
        loops.append({
            "loop": loop_index,
            "task": spec.task,
            "seed": spec.seed,
            "query_offset": spec.query_offset,
            "query_count": sum(window["query_count"] for window in loop_windows),
            "windows": loop_windows,
        })

    profile_outcomes = summarize_profile_outcomes(all_window_results)
    gate_feature_rows = build_gate_feature_rows(all_window_results)
    query_attribution_rows = build_query_attribution_rows(all_window_results)
    gate_candidate_report = summarize_gate_candidates(gate_feature_rows)
    directional_candidates = [
        profile
        for profile, outcome in profile_outcomes.items()
        if profile != "baseline" and outcome["best_delta_vs_baseline"] > 0.001
    ]
    promotable_profiles = [
        profile
        for profile, outcome in profile_outcomes.items()
        if (
            profile != "baseline"
            and outcome["windows"] >= 3
            and outcome["mean_delta_vs_baseline"] > 0.001
            and outcome["regressed"] == 0
        )
    ]
    default_promotable_profiles = [
        profile
        for profile, outcome in profile_outcomes.items()
        if (
            profile != "baseline"
            and len(outcome["tasks"]) >= 3
            and outcome["windows"] >= 10
            and outcome["mean_delta_vs_baseline"] > 0.001
            and outcome["regressed"] == 0
            and outcome.get("fault_counts", {}).get("actuator_active_negative", 0) == 0
            and outcome.get("fault_counts", {}).get("metric_changed_without_actuator", 0) == 0
        )
    ]
    golden_candidate_profiles = [
        profile
        for profile, outcome in profile_outcomes.items()
        if (
            profile != "baseline"
            and len(outcome["tasks"]) >= 3
            and outcome["windows"] >= 10
            and outcome["mean_delta_vs_baseline"] >= 0.01
            and outcome["regressed"] == 0
            and outcome.get("fault_counts", {}).get("actuator_active_negative", 0) == 0
            and outcome.get("fault_counts", {}).get("metric_changed_without_actuator", 0) == 0
        )
    ]
    return {
        "status": "completed",
        "model": model,
        "loop_queries": loop_queries,
        "window_queries": window_queries,
        "sample_docs": sample_docs,
        "strategy": strategy,
        "gate_config": gate_config if strategy == "learned_gate" else None,
        "total_queries": sum(loop["query_count"] for loop in loops),
        "completed_windows": len(all_window_results),
        "completed_queries": sum(window["query_count"] for window in all_window_results),
        "loops": loops,
        "profile_outcomes": profile_outcomes,
        "gate_feature_rows": gate_feature_rows,
        "query_attribution_rows": query_attribution_rows,
        "gate_candidate_report": gate_candidate_report,
        "recommendation": {
            "default_change_allowed": False,
            "directional_candidate_profiles": directional_candidates,
            "promotable_profiles": promotable_profiles,
            "default_promotable_profiles": default_promotable_profiles,
            "golden_candidate_profiles": golden_candidate_profiles,
            "shippable_gate_candidates": [
                {
                    "profile": profile,
                    "gate": report["best_gate"]["gate"],
                    "mean_delta_vs_baseline": report["best_gate"]["mean_delta_vs_baseline"],
                    "windows": report["best_gate"]["windows"],
                }
                for profile, report in gate_candidate_report.items()
                if report["best_gate"]["shippable_gate_candidate"]
            ],
            "reason": (
                "requires repeatable positive deltas across tasks, no active-negative fault blockers, "
                "and promotion gates; "
                "this runner reports directional actuator evidence only"
            ),
        },
    }


def parse_loop_specs(raw_specs: str) -> List[LoopSpec]:
    specs = []
    for raw in raw_specs.split(","):
        task, seed, offset = raw.split(":")
        specs.append(LoopSpec(task=task, seed=int(seed), query_offset=int(offset)))
    return specs


def build_phase_loop_specs(phase_queries: int, loop_queries: int, base_seed: int = 46) -> List[LoopSpec]:
    """Build enough loop specs to reach a requested phase size."""

    if phase_queries <= 0:
        raise ValueError("phase_queries must be positive")
    if loop_queries <= 0:
        raise ValueError("loop_queries must be positive")
    if phase_queries % loop_queries != 0:
        raise ValueError("phase_queries must divide evenly by loop_queries")

    loop_count = phase_queries // loop_queries
    specs = []
    for index in range(loop_count):
        task, offset = PHASE_LOOP_TEMPLATES[index % len(PHASE_LOOP_TEMPLATES)]
        seed = base_seed + (index // len(PHASE_LOOP_TEMPLATES))
        specs.append(LoopSpec(task=task, seed=seed, query_offset=offset))
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an adaptive road-course tuning phase")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--phase-queries", type=int, default=None)
    parser.add_argument("--loop-queries", type=int, default=200)
    parser.add_argument("--window-queries", type=int, default=50)
    parser.add_argument("--sample-docs", type=int, default=800)
    parser.add_argument("--base-seed", type=int, default=46)
    parser.add_argument(
        "--strategy",
        choices=[
            "adaptive",
            "confirm_fiqa",
            "fault_aware",
            "gate_learning",
            "learned_gate",
            "selective_reform",
            "reform_policy_search",
        ],
        default="adaptive",
    )
    parser.add_argument("--gate-config", "--gate-artifact", dest="gate_config", default=None)
    parser.add_argument(
        "--loop-specs",
        default=None,
        help=(
            "Comma-separated task:seed:query_offset specs. If omitted, --phase-queries builds specs by cycling "
            "SciFact/NFCorpus/FiQA windows; otherwise defaults to the original 1,000-query phase."
        ),
    )
    parser.add_argument(
        "--output",
        default="experiment_runs/roadcourse-small/adaptive_thousand_query_tuning.json",
    )
    args = parser.parse_args()
    if args.loop_specs:
        loop_specs = parse_loop_specs(args.loop_specs)
    elif args.phase_queries:
        loop_specs = build_phase_loop_specs(args.phase_queries, args.loop_queries, base_seed=args.base_seed)
    else:
        loop_specs = DEFAULT_LOOP_SPECS

    result = run_thousand_query_cycle(
        model=args.model,
        loop_specs=loop_specs,
        loop_queries=args.loop_queries,
        window_queries=args.window_queries,
        sample_docs=args.sample_docs,
        checkpoint_path=Path(args.output),
        strategy=args.strategy,
        gate_config=load_gate_config(args.gate_config),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "total_queries": result["total_queries"],
        "window_queries": result["window_queries"],
        "loops": len(result["loops"]),
        "recommendation": result["recommendation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
