"""Run drift-recovery experiments for the June 2026 harness."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmark_utils import canonicalize_id, isolated_adapter_state, load_mteb_data, map_predicted_ids
from chelation_adapter import create_adapter
from drift_injector import DriftInjector
from drift_recovery_metrics import RecoveryTracker, ndcg_at_k
from run_road_course_campaign import select_road_course_slice


CONDITIONS = ("C0", "C1", "C2", "C2O", "C3", "C4")
DRIFT_MODES = ("rotation", "noise", "query_encoder_swap")


@dataclass(frozen=True)
class DriftRecoveryConfig:
    task: str
    condition: str
    drift: str
    fraction: float
    angle: float
    sigma: float
    cycles: int
    seed: int
    max_queries: int
    sample_docs: int
    output: str
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    k: int = 10
    injection_index: Optional[int] = None
    device: Optional[str] = None
    bound_epsilon: float = 0.01
    trigger_threshold: float = 0.0
    max_temperature: float = 1.0
    epochs_scale: float = 1.0
    swap_model: str = "all-mpnet-base-v2"


def run_experiment(
    config: DriftRecoveryConfig,
    corpus: Optional[Mapping[Any, str]] = None,
    queries: Optional[Mapping[Any, str]] = None,
    qrels: Optional[Mapping[Any, Mapping[Any, float]]] = None,
) -> Dict[str, Any]:
    """Run one configured drift-recovery experiment and write its JSON artifact."""

    _validate_config(config)
    _seed_everything(config.seed)
    started = time.perf_counter()

    if corpus is None or queries is None or qrels is None:
        corpus, queries, qrels = load_mteb_data(config.task)
    if corpus is None or queries is None or qrels is None:
        raise RuntimeError(f"Failed to load task data for {config.task}")

    sliced_corpus, sliced_queries, sliced_qrels = select_road_course_slice(
        corpus,
        queries,
        qrels,
        max_queries=config.max_queries,
        sample_docs=config.sample_docs,
        seed=config.seed,
    )
    if not sliced_corpus or not sliced_queries:
        raise ValueError("Selected experiment slice is empty")

    with isolated_adapter_state():
        engine, doc_ids = _build_engine(config, sliced_corpus)
        try:
            baseline_ndcg, baseline_details = evaluate_engine(engine, sliced_queries, sliced_qrels, config.k)
            _prime_correction_log(engine, sliced_queries)

            if config.condition == "C1":
                engine.run_sedimentation_cycle(threshold=1, learning_rate=0.001, epochs=1)

            query_drift = None
            drifted_query_vectors: Optional[Dict[str, np.ndarray]] = None
            if config.drift == "query_encoder_swap":
                # Query-encoder upgrade: the store is NOT mutated. Instead, eval
                # queries are re-embedded with the swapped encoder and used for
                # every post-drift measurement. The drift object carries the
                # frozen seeded projection that C2O reuses to re-embed docs.
                query_drift, drifted_query_vectors = _build_query_encoder_drift(
                    engine, sliced_queries, config
                )
                manifest = query_drift.manifest()
                manifest["injection_index"] = 0
            else:
                injector = DriftInjector(engine, seed=config.seed)
                manifest = _inject_drift(injector, config)
            run_config = asdict(config)
            run_config["injection_index"] = manifest["injection_index"]
            run_config["device"] = config.device or _detect_device()
            run_config["corpus_size"] = len(sliced_corpus)
            run_config["query_count"] = len(sliced_queries)

            tracker = RecoveryTracker(baseline_ndcg=baseline_ndcg)
            correction_norms = []

            for cycle_index in range(1, config.cycles + 1):
                metadata: Dict[str, Any] = {"condition": config.condition}
                metadata.update(
                    _run_condition_cycle(
                        engine,
                        config.condition,
                        sliced_queries,
                        manifest,
                        run_config,
                        query_drift=query_drift,
                    )
                )
                if drifted_query_vectors is not None:
                    ndcg, details = evaluate_engine_with_query_vectors(
                        engine, drifted_query_vectors, sliced_qrels, config.k
                    )
                else:
                    ndcg, details = evaluate_engine(engine, sliced_queries, sliced_qrels, config.k)
                metadata["query_ndcg"] = details
                metadata["evaluated_queries"] = len(details)
                if config.condition in {"C3", "C4"}:
                    norm_stats = _correction_norm_stats(engine)
                    metadata["correction_norm_stats"] = norm_stats
                    correction_norms.extend(norm_stats["sample_norms"])
                tracker.record_cycle(cycle_index, ndcg, metadata)

            result = {
                "record_type": "drift_recovery_experiment",
                "config": run_config,
                "drift_manifest": manifest,
                "baseline": {
                    "ndcg_at_10": baseline_ndcg,
                    "query_ndcg": baseline_details,
                },
                "recovery": tracker.to_json(),
                "correction_norm_stats": _aggregate_norm_stats(correction_norms),
                "cycle_errors": [],
                "wall_clock_seconds": time.perf_counter() - started,
                "doc_ids": doc_ids,
            }
        finally:
            if hasattr(engine, "close"):
                engine.close()

    _write_json(config.output, result)
    return result


def evaluate_engine(engine, queries: Mapping[str, str], qrels: Mapping[str, Mapping[str, float]], k: int = 10) -> Tuple[float, list]:
    rows = []
    for query_id, query_text in queries.items():
        relevance = qrels.get(query_id, qrels.get(canonicalize_id(query_id), {}))
        relevant_ids = [canonicalize_id(doc_id) for doc_id, score in relevance.items() if float(score) > 0.0]
        if not relevant_ids:
            continue
        _std_top, chelated_top, _mask, _jaccard = engine.run_inference(str(query_text))
        ranked = map_predicted_ids(engine, chelated_top[:k])
        score = ndcg_at_k(ranked, relevant_ids, k=k)
        rows.append(
            {
                "query_id": canonicalize_id(query_id),
                "ndcg": score,
                "ranked_ids": ranked,
                "relevant_ids": relevant_ids,
            }
        )
    mean_ndcg = float(statistics.fmean(row["ndcg"] for row in rows)) if rows else 0.0
    return mean_ndcg, rows


def evaluate_engine_with_query_vectors(
    engine,
    query_vectors_by_id: Mapping[str, np.ndarray],
    qrels: Mapping[str, Mapping[str, float]],
    k: int = 10,
) -> Tuple[float, list]:
    """Score retrieval using precomputed query vectors instead of engine embeddings.

    Mirrors ``evaluate_engine`` exactly (same row shape, same ID mapping, same
    ndcg_at_k scoring) but searches the store by a caller-supplied vector. Used
    for the query-encoder-swap arena, where eval queries live in the swapped
    encoder's space while the cached doc vectors stay in the original space.
    """
    rows = []
    for query_id, query_vector in query_vectors_by_id.items():
        relevance = qrels.get(query_id, qrels.get(canonicalize_id(query_id), {}))
        relevant_ids = [canonicalize_id(doc_id) for doc_id, score in relevance.items() if float(score) > 0.0]
        if not relevant_ids:
            continue
        hits = engine.qdrant.query_points(
            collection_name=engine.collection_name,
            query=np.asarray(query_vector, dtype=np.float32),
            limit=k,
            with_payload=False,
            with_vectors=False,
        ).points
        ranked = map_predicted_ids(engine, [hit.id for hit in hits])
        score = ndcg_at_k(ranked, relevant_ids, k=k)
        rows.append(
            {
                "query_id": canonicalize_id(query_id),
                "ndcg": score,
                "ranked_ids": ranked,
                "relevant_ids": relevant_ids,
            }
        )
    mean_ndcg = float(statistics.fmean(row["ndcg"] for row in rows)) if rows else 0.0
    return mean_ndcg, rows


def _build_query_encoder_drift(engine, queries: Mapping[str, str], config: DriftRecoveryConfig):
    """Construct the query-encoder-swap drift and embed the eval queries with it.

    Returns ``(query_drift, drifted_query_vectors_by_id)``. The drift object owns
    the frozen seeded projection; ``manifest()`` is only valid after this call.
    """
    from query_encoder_drift import QueryEncoderDrift

    query_drift = QueryEncoderDrift(
        store_dim=engine.vector_size,
        swap_model_name=config.swap_model,
        seed=config.seed,
    )
    query_ids = list(queries.keys())
    drifted = query_drift.embed_queries([str(queries[query_id]) for query_id in query_ids])
    drifted_query_vectors = {
        canonicalize_id(query_id): drifted[index] for index, query_id in enumerate(query_ids)
    }
    return query_drift, drifted_query_vectors


def _build_engine(config: DriftRecoveryConfig, corpus: Mapping[str, str]):
    from antigravity_engine import AntigravityEngine

    engine = AntigravityEngine(
        qdrant_location=":memory:",
        model_name=config.model,
        store_full_text_payload=True,
    )
    if config.condition in {"C3", "C4"}:
        engine.adapter = create_adapter(
            "mlp",
            input_dim=engine.vector_size,
            bounded=(config.condition == "C3"),
            min_correction=config.bound_epsilon,
            max_correction=0.5,
        )
    doc_ids = list(corpus.keys())
    payloads = [{"doc_id": canonicalize_id(doc_id)} for doc_id in doc_ids]
    engine.ingest([corpus[doc_id] for doc_id in doc_ids], payloads)
    return engine, [canonicalize_id(doc_id) for doc_id in doc_ids]


def _inject_drift(injector: DriftInjector, config: DriftRecoveryConfig) -> dict:
    if config.drift == "rotation":
        return injector.inject_rotation_drift(fraction=config.fraction, angle_degrees=config.angle)
    if config.drift == "noise":
        return injector.inject_noise_drift(fraction=config.fraction, sigma=config.sigma)
    raise ValueError(f"Unsupported drift mode: {config.drift}")


def _run_condition_cycle(
    engine,
    condition: str,
    queries: Mapping[str, str],
    drift_manifest: Mapping[str, Any],
    run_config: Optional[Mapping[str, Any]] = None,
    query_drift: Any = None,
) -> Dict[str, Any]:
    config = run_config or {}
    if condition == "C0":
        return {"action": "none"}
    if condition == "C1":
        return {"action": "static_adapter_frozen"}
    if condition == "C2":
        # Re-embed docs with the ORIGINAL engine model. For store-mutating drift
        # this refreshes the affected vectors; for query_encoder_swap it is a
        # proven no-op (re-embedding unchanged text reproduces cached vectors,
        # which remain in the original space and stay misaligned with the swapped
        # query space) — so all docs are refreshed to make that no-op observable.
        if drift_manifest.get("drift") == "query_encoder_swap":
            return {
                "action": "maintenance_reembed_original_model",
                "refresh": _reembed_all_docs_with_original_model(engine),
            }
        return {
            "action": "maintenance_reindex_affected",
            "refresh": _refresh_affected_corpus_vectors(engine, drift_manifest["affected_ids"]),
        }
    if condition == "C2O":
        if query_drift is None:
            raise ValueError("C2O requires a query_encoder_swap drift object")
        return {
            "action": "oracle_reembed_swap_model",
            "refresh": _reembed_all_docs_with_swap_model(engine, query_drift),
        }
    if condition in {"C3", "C4"}:
        controller = getattr(engine, "_annealing_controller", None)
        if controller is None:
            controller = engine.enable_annealing_controller(
                trigger_threshold=float(config.get("trigger_threshold", 0.0)),
                max_temperature=float(config.get("max_temperature", 1.0)),
                cooling_rate=0.5,
            )
        engine._annealing_epochs_scale = float(config.get("epochs_scale", 1.0))
        _prime_correction_log(engine, queries)
        observation = engine.observe_annealing_drift()
        should_correct = bool(controller.should_correct())
        checksum_before = _vector_store_checksum(engine)
        sedimentation_attempted = False
        correction_applied = False
        if controller.should_correct():
            sedimentation_attempted = True
            engine.run_sedimentation_cycle(threshold=1, learning_rate=0.001, epochs=1)
            correction_applied = checksum_before != _vector_store_checksum(engine)
        metadata = {
            "action": "detection_triggered_sedimentation",
            "bounded": condition == "C3",
            "detector_source": "AntigravityEngine._compute_annealing_drift_magnitude",
            "should_correct": should_correct,
            "sedimentation_attempted": sedimentation_attempted,
            "correction_applied": correction_applied,
            "annealing_observation": observation,
            "annealing_settings": getattr(engine, "_last_annealing_settings", None),
        }
        knobs = {
            "bound_epsilon": float(config.get("bound_epsilon", 0.01)),
            "trigger_threshold": float(config.get("trigger_threshold", 0.0)),
            "max_temperature": float(config.get("max_temperature", 1.0)),
            "epochs_scale": float(config.get("epochs_scale", 1.0)),
        }
        if knobs != {
            "bound_epsilon": 0.01,
            "trigger_threshold": 0.0,
            "max_temperature": 1.0,
            "epochs_scale": 1.0,
        }:
            metadata["knobs"] = knobs
        return metadata
    raise ValueError(f"Unsupported condition: {condition}")


def _refresh_affected_corpus_vectors(engine, affected_ids: Sequence[Any]) -> Dict[str, int]:
    from qdrant_client.models import PointStruct

    if not affected_ids:
        return {"updated": 0, "failed": 0}
    points = engine.qdrant.retrieve(
        collection_name=engine.collection_name,
        ids=list(affected_ids),
        with_payload=True,
    )
    missing_text = [
        point.id
        for point in points
        if not isinstance(getattr(point, "payload", None), dict) or "text" not in point.payload
    ]
    if missing_text:
        raise ValueError(f"Cannot refresh affected vectors without text payloads; missing IDs: {missing_text[:5]}")
    vectors = np.asarray(engine.embed([point.payload["text"] for point in points]), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(points):
        raise ValueError(f"Embedding refresh returned invalid shape {vectors.shape}; expected {len(points)} rows")
    upserts = [
        PointStruct(
            id=point.id,
            vector=vectors[index],
            payload=point.payload,
        )
        for index, point in enumerate(points)
    ]
    engine.qdrant.upsert(collection_name=engine.collection_name, points=upserts)
    return {"updated": len(upserts), "failed": 0}


def _load_all_doc_points(engine) -> list:
    """Scroll the whole store and return points carrying a 'text' payload."""
    points = []
    offset = None
    while True:
        batch, next_offset = engine.qdrant.scroll(
            collection_name=engine.collection_name,
            limit=256,
            with_vectors=False,
            with_payload=True,
            offset=offset,
        )
        points.extend(batch)
        if next_offset is None or not batch:
            break
        offset = next_offset
    points.sort(key=lambda item: repr(item.id))
    missing_text = [
        point.id
        for point in points
        if not isinstance(getattr(point, "payload", None), dict) or "text" not in point.payload
    ]
    if missing_text:
        raise ValueError(f"Cannot re-embed docs without text payloads; missing IDs: {missing_text[:5]}")
    return points


def _reembed_all_docs_with_original_model(engine) -> Dict[str, int]:
    """C2 for query_encoder_swap: re-embed every doc with the ORIGINAL engine model.

    The store is not mutated by query-side drift, so re-embedding the unchanged
    doc text with the frozen original encoder reproduces the cached vectors and
    leaves them in the original space — a proven no-op against swapped queries.
    """
    from qdrant_client.models import PointStruct

    points = _load_all_doc_points(engine)
    if not points:
        return {"updated": 0, "failed": 0}
    vectors = np.asarray(engine.embed([point.payload["text"] for point in points]), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(points):
        raise ValueError(f"Embedding refresh returned invalid shape {vectors.shape}; expected {len(points)} rows")
    upserts = [
        PointStruct(id=point.id, vector=vectors[index], payload=point.payload)
        for index, point in enumerate(points)
    ]
    engine.qdrant.upsert(collection_name=engine.collection_name, points=upserts)
    return {"updated": len(upserts), "failed": 0}


def _reembed_all_docs_with_swap_model(engine, query_drift: Any) -> Dict[str, int]:
    """C2-oracle: re-embed every doc with the SWAP model + the SAME frozen seeded
    projection used for the query drift, so docs land in the drifted query space.

    This is the expensive upper bound that recovers retrieval. It reuses the
    query_drift object directly so the projection (and its checksum) is identical
    to the one applied to the eval queries.
    """
    from qdrant_client.models import PointStruct

    points = _load_all_doc_points(engine)
    if not points:
        return {"updated": 0, "failed": 0}
    vectors = np.asarray(
        query_drift.embed_queries([str(point.payload["text"]) for point in points]),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(points):
        raise ValueError(f"Swap re-embed returned invalid shape {vectors.shape}; expected {len(points)} rows")
    upserts = [
        PointStruct(id=point.id, vector=vectors[index], payload=point.payload)
        for index, point in enumerate(points)
    ]
    engine.qdrant.upsert(collection_name=engine.collection_name, points=upserts)
    return {"updated": len(upserts), "failed": 0}


def _prime_correction_log(engine, queries: Mapping[str, str]) -> None:
    for query_text in queries.values():
        engine.run_inference(str(query_text))


def _correction_norm_stats(engine, sample_limit: int = 128) -> Dict[str, Any]:
    import torch

    vectors = []
    offset = None
    while len(vectors) < sample_limit:
        points, next_offset = engine.qdrant.scroll(
            collection_name=engine.collection_name,
            limit=min(64, sample_limit - len(vectors)),
            with_vectors=True,
            with_payload=False,
            offset=offset,
        )
        for point in points:
            vectors.append(_point_vector(point.vector))
        if next_offset is None or not points:
            break
        offset = next_offset
    if not vectors:
        return _aggregate_norm_stats([])

    with torch.no_grad():
        tensor = torch.tensor(np.asarray(vectors, dtype=np.float32), dtype=torch.float32)
        adapted = engine.adapter(tensor).detach().cpu().numpy()
    norms = np.linalg.norm(adapted - np.asarray(vectors, dtype=np.float32), axis=1)
    return _aggregate_norm_stats([float(value) for value in norms])


def _vector_store_checksum(engine) -> str:
    import hashlib

    digest = hashlib.sha256()
    offset = None
    while True:
        points, next_offset = engine.qdrant.scroll(
            collection_name=engine.collection_name,
            limit=256,
            with_vectors=True,
            with_payload=False,
            offset=offset,
        )
        for point in sorted(points, key=lambda item: repr(item.id)):
            digest.update(repr(point.id).encode("utf-8"))
            digest.update(_point_vector(point.vector).tobytes())
        if next_offset is None or not points:
            break
        offset = next_offset
    return digest.hexdigest()


def _aggregate_norm_stats(norms: Sequence[float]) -> Dict[str, Any]:
    values = [float(value) for value in norms]
    if not values:
        return {"count": 0, "mean": None, "max": None, "sample_norms": []}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "max": float(max(values)),
        "sample_norms": values[:128],
    }


def _point_vector(vector: Any) -> np.ndarray:
    if isinstance(vector, dict):
        vector = next(iter(vector.values()))
    return np.asarray(vector, dtype=np.float32)


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def _validate_config(config: DriftRecoveryConfig) -> None:
    if config.condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if config.drift not in DRIFT_MODES:
        raise ValueError(f"drift must be one of {DRIFT_MODES}")
    if config.condition == "C2O" and config.drift != "query_encoder_swap":
        raise ValueError("condition C2O is only valid with drift=query_encoder_swap")
    if config.drift == "query_encoder_swap" and not str(config.swap_model).strip():
        raise ValueError("swap_model must be a non-empty model name")
    if not 0.0 <= float(config.fraction) <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if config.cycles < 1:
        raise ValueError("cycles must be >= 1")
    if config.max_queries < 1:
        raise ValueError("max_queries must be >= 1")
    if config.sample_docs < 1:
        raise ValueError("sample_docs must be >= 1")
    if not math.isfinite(float(config.angle)):
        raise ValueError("angle must be finite")
    if not math.isfinite(float(config.sigma)) or float(config.sigma) < 0.0:
        raise ValueError("sigma must be a non-negative finite number")
    if not math.isfinite(float(config.bound_epsilon)) or float(config.bound_epsilon) <= 0.0:
        raise ValueError("bound_epsilon must be a positive finite number")
    if not math.isfinite(float(config.trigger_threshold)) or float(config.trigger_threshold) < 0.0:
        raise ValueError("trigger_threshold must be a non-negative finite number")
    if not math.isfinite(float(config.max_temperature)) or float(config.max_temperature) <= 0.0:
        raise ValueError("max_temperature must be a positive finite number")
    if not math.isfinite(float(config.epochs_scale)) or float(config.epochs_scale) <= 0.0:
        raise ValueError("epochs_scale must be a positive finite number")


def _write_json(path: str, data: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a drift-recovery experiment")
    parser.add_argument("--task", default="SciFact")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--sample-docs", type=int, default=1200)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--drift", choices=DRIFT_MODES, default="rotation")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--angle", type=float, default=25.0)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--bound-epsilon", type=float, default=0.01)
    parser.add_argument("--trigger-threshold", type=float, default=0.0)
    parser.add_argument("--max-temperature", type=float, default=1.0)
    parser.add_argument("--epochs-scale", type=float, default=1.0)
    parser.add_argument("--swap-model", default="all-mpnet-base-v2")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = DriftRecoveryConfig(
        task=args.task,
        condition=args.condition,
        drift=args.drift,
        fraction=args.fraction,
        angle=args.angle,
        sigma=args.sigma,
        cycles=args.cycles,
        seed=args.seed,
        max_queries=args.max_queries,
        sample_docs=args.sample_docs,
        model=args.model,
        output=args.output,
        device=_detect_device(),
        bound_epsilon=args.bound_epsilon,
        trigger_threshold=args.trigger_threshold,
        max_temperature=args.max_temperature,
        epochs_scale=args.epochs_scale,
        swap_model=args.swap_model,
    )
    result = run_experiment(config)
    print(json.dumps({"output": args.output, "final_ndcg": result["recovery"]["trajectory"][-1]["ndcg"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
