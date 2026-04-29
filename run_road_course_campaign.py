"""Small-model road-course benchmark campaign for ChelatedAI defaults."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from antigravity_engine import AntigravityEngine
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
from config import ChelationConfig
from quantization_promotion_gate import QuantizationPromotionGate
from reproducibility_context import evaluate_seed_scores, stable_hash


@dataclass(frozen=True)
class RoadCourseProfile:
    name: str
    use_centering: bool = False
    use_quantization: bool = False
    chelation_p: int = ChelationConfig.DEFAULT_CHELATION_P
    chelation_threshold: float = ChelationConfig.DEFAULT_CHELATION_THRESHOLD
    temperature: float = 1.0
    query_reformulation_variants: int = 0
    query_reformulation_policy: str = "always"


DEFAULT_PROFILE_GRID = [
    RoadCourseProfile("baseline"),
    RoadCourseProfile("adaptive_p50_t0.0004", use_quantization=True, chelation_p=50, chelation_threshold=0.0004),
    RoadCourseProfile("adaptive_p75_t0.0004", use_quantization=True, chelation_p=75, chelation_threshold=0.0004),
    RoadCourseProfile("adaptive_p85_t0.0004", use_quantization=True, chelation_p=85, chelation_threshold=0.0004),
    RoadCourseProfile("adaptive_p95_t0.0004", use_quantization=True, chelation_p=95, chelation_threshold=0.0004),
    RoadCourseProfile("adaptive_p85_t0.001", use_quantization=True, chelation_p=85, chelation_threshold=0.001),
    RoadCourseProfile("adaptive_p85_t0.01", use_quantization=True, chelation_p=85, chelation_threshold=0.01),
    RoadCourseProfile("fast_guard_p85_t999", use_quantization=True, chelation_p=85, chelation_threshold=999.0),
    RoadCourseProfile("centered_p85_temp1", use_centering=True, chelation_p=85, chelation_threshold=0.0),
    RoadCourseProfile("centered_p50_temp1", use_centering=True, chelation_p=50, chelation_threshold=0.0),
]


def select_road_course_slice(
    corpus: Mapping[Any, str],
    queries: Mapping[Any, str],
    qrels: Mapping[Any, Mapping[Any, float]],
    max_queries: int,
    sample_docs: int,
    seed: int,
) -> tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, float]]]:
    """Select a deterministic small slice while preserving relevant documents."""

    selected_queries: Dict[str, str] = {}
    selected_qrels: Dict[str, Dict[str, float]] = {}
    for query_id, query_text in queries.items():
        canonical_query_id = canonicalize_id(query_id)
        relevance = {
            canonicalize_id(doc_id): float(score)
            for doc_id, score in qrels.get(query_id, qrels.get(canonical_query_id, {})).items()
            if float(score) > 0
        }
        if not relevance:
            continue
        selected_queries[canonical_query_id] = str(query_text)
        selected_qrels[canonical_query_id] = relevance
        if len(selected_queries) >= max_queries:
            break

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


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    k: int = 10,
) -> Dict[str, float]:
    ndcg_scores = []
    map_scores = []
    mrr_scores = []
    recall_scores = []
    for query_id, relevance in qrels.items():
        relevant_ids = {doc_id for doc_id, score in relevance.items() if score > 0}
        retrieved = [canonicalize_id(doc_id) for doc_id in rankings.get(query_id, [])[:k]]
        relevance_by_rank = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved]
        ndcg_scores.append(float(ndcg_at_k(relevance_by_rank, k)))
        map_scores.append(float(mean_average_precision_at_k(retrieved, relevant_ids, k)))
        mrr_scores.append(float(mean_reciprocal_rank(retrieved, relevant_ids)))
        recall_scores.append(float(recall_at_k(retrieved, relevant_ids, k)))
    return {
        "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "map_at_10": float(np.mean(map_scores)) if map_scores else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "recall_at_10": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "evaluated_queries": float(len(ndcg_scores)),
    }


def _profile_engine(
    profile: RoadCourseProfile,
    model_name: str,
    corpus: Mapping[str, str],
):
    engine = AntigravityEngine(
        qdrant_location=":memory:",
        model_name=model_name,
        use_centering=profile.use_centering,
        use_quantization=profile.use_quantization,
        store_full_text_payload=True,
    )
    engine.chelation_p = profile.chelation_p
    engine.chelation_threshold = profile.chelation_threshold
    if profile.temperature != 1.0:
        engine.set_temperature(profile.temperature)
    if profile.query_reformulation_variants > 0:
        engine.enable_query_reformulation(
            max_variants=profile.query_reformulation_variants,
            policy=profile.query_reformulation_policy,
        )
    doc_ids = list(corpus.keys())
    engine.ingest([corpus[doc_id] for doc_id in doc_ids], [{"doc_id": doc_id} for doc_id in doc_ids])
    return engine


def evaluate_profile(
    profile: RoadCourseProfile,
    model_name: str,
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    start = time.perf_counter()
    with isolated_adapter_state():
        engine = _profile_engine(profile, model_name, corpus)
        try:
            rankings: Dict[str, list[str]] = {}
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
            metrics = evaluate_rankings(rankings, qrels)
            telemetry = engine.get_runtime_telemetry()
        finally:
            engine.close()
    elapsed = time.perf_counter() - start
    return {
        "profile": profile.name,
        "controls": asdict(profile),
        "metrics": metrics,
        "rankings": rankings,
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
        "elapsed_seconds": elapsed,
        "telemetry": telemetry,
    }


def quantization_survival_check(
    profile: RoadCourseProfile,
    baseline_ndcg: float,
    fp32_ndcg: float,
    model_name: str,
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    """Evaluate candidate with INT8-simulated embeddings and run the promotion gate."""

    quantized_profile = RoadCourseProfile(
        name=f"{profile.name}_quantized_embeddings",
        use_centering=profile.use_centering,
        use_quantization=profile.use_quantization,
        chelation_p=profile.chelation_p,
        chelation_threshold=profile.chelation_threshold,
        temperature=profile.temperature,
        query_reformulation_variants=profile.query_reformulation_variants,
        query_reformulation_policy=profile.query_reformulation_policy,
    )
    with isolated_adapter_state():
        engine = _profile_engine(quantized_profile, model_name, corpus)
        try:
            engine._simulate_embedding_quantization = True
            rankings: Dict[str, list[str]] = {}
            for query_id, query_text in queries.items():
                _std_top, final_top, _mask, _jaccard = engine.run_inference(query_text)
                rankings[query_id] = map_predicted_ids(engine, final_top[:10])
            metrics = evaluate_rankings(rankings, qrels)
        finally:
            engine.close()
    return {
        "metrics": metrics,
        "gate": QuantizationPromotionGate(0.8).evaluate(
            fp32_fitness=fp32_ndcg,
            quantized_fitness=metrics["ndcg_at_10"],
            baseline_fitness=baseline_ndcg,
        ).to_dict(),
    }


def run_campaign(
    task: str,
    model: str,
    max_queries: int,
    sample_docs: int,
    seed: int,
    profiles: Iterable[RoadCourseProfile] = DEFAULT_PROFILE_GRID,
) -> Dict[str, Any]:
    corpus, queries, qrels = load_mteb_data(task)
    if not corpus or not queries or not qrels:
        raise RuntimeError(f"Failed to load task {task}")
    sliced_corpus, sliced_queries, sliced_qrels = select_road_course_slice(
        corpus,
        queries,
        qrels,
        max_queries=max_queries,
        sample_docs=sample_docs,
        seed=seed,
    )
    profile_results = [
        evaluate_profile(profile, model, sliced_corpus, sliced_queries, sliced_qrels)
        for profile in profiles
    ]
    baseline = next(result for result in profile_results if result["profile"] == "baseline")
    best = max(profile_results, key=lambda result: result["metrics"]["ndcg_at_10"])
    quantization = quantization_survival_check(
        RoadCourseProfile(**best["controls"]),
        baseline_ndcg=baseline["metrics"]["ndcg_at_10"],
        fp32_ndcg=best["metrics"]["ndcg_at_10"],
        model_name=model,
        corpus=sliced_corpus,
        queries=sliced_queries,
        qrels=sliced_qrels,
    )
    seed_gate = evaluate_seed_scores([result["metrics"]["ndcg_at_10"] for result in profile_results], tolerance=1.0)
    default_change = {
        "recommended_profile": best["profile"],
        "baseline_ndcg_at_10": baseline["metrics"]["ndcg_at_10"],
        "best_ndcg_at_10": best["metrics"]["ndcg_at_10"],
        "delta_vs_baseline": best["metrics"]["ndcg_at_10"] - baseline["metrics"]["ndcg_at_10"],
        "default_change_allowed": best["profile"] != "baseline" and best["metrics"]["ndcg_at_10"] > baseline["metrics"]["ndcg_at_10"],
        "reason": (
            "baseline_remains_best"
            if best["profile"] == "baseline"
            else "candidate_beats_baseline_on_small_road_course"
        ),
    }
    return {
        "task": task,
        "model": model,
        "seed": seed,
        "dataset_hash": stable_hash({"corpus": sliced_corpus, "qrels": sliced_qrels}),
        "query_hash": stable_hash(sliced_queries),
        "corpus_size": len(sliced_corpus),
        "query_count": len(sliced_queries),
        "profile_results": profile_results,
        "quantization_survival": quantization,
        "seed_gate": seed_gate.to_dict(),
        "default_recommendation": default_change,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run small-model ChelatedAI road-course profile campaign")
    parser.add_argument("--task", default="SciFact")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--sample-docs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiment_runs/roadcourse-small/roadcourse_profile_grid.json")
    args = parser.parse_args()

    result = run_campaign(
        task=args.task,
        model=args.model,
        max_queries=args.max_queries,
        sample_docs=args.sample_docs,
        seed=args.seed,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "task": result["task"],
        "model": result["model"],
        "corpus_size": result["corpus_size"],
        "query_count": result["query_count"],
        "default_recommendation": result["default_recommendation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
