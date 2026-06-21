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


CONDITIONS = ("C0", "C1", "C2", "C2O", "C3", "C4", "C3a", "C4a")
DRIFT_MODES = ("rotation", "noise", "query_encoder_swap")

# Conditions whose correction is the supervised anchor-pair InfoNCE closed loop
# (PR-A2b). Unlike C3/C4 (unsupervised homeostatic sedimentation, which never
# fired in v1), these train the adapter so adapted cached doc vectors realign
# with the NEW drifted query space, supervised by held-out pre-drift anchors.
SUPERVISED_CONDITIONS = ("C3a", "C4a")


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
    anchor_fraction: float = 0.0


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
            # Anchor/eval split (PR-A2b): when the supervised closed loop needs
            # held-out anchors, partition the sliced queries into a disjoint
            # ANCHOR set (supervision) and EVAL set (scoring). When inactive
            # (anchor_fraction == 0), the eval set IS every sliced query, so the
            # existing arena behavior is byte-for-byte preserved.
            split_active = (
                config.drift == "query_encoder_swap" and float(config.anchor_fraction) > 0.0
            )
            anchor_ids, eval_ids = _split_anchor_eval(
                sliced_queries, config.anchor_fraction, config.seed, active=split_active
            )
            assert set(anchor_ids).isdisjoint(set(eval_ids)), "anchor/eval id sets overlap"
            eval_queries = {qid: sliced_queries[qid] for qid in eval_ids}

            baseline_ndcg, baseline_details = evaluate_engine(engine, eval_queries, sliced_qrels, config.k)
            _prime_correction_log(engine, sliced_queries)

            if config.condition == "C1":
                engine.run_sedimentation_cycle(threshold=1, learning_rate=0.001, epochs=1)

            query_drift = None
            drifted_query_vectors: Optional[Dict[str, np.ndarray]] = None
            anchor_pairs: Optional[list] = None
            original_doc_points: Optional[list] = None
            if config.drift == "query_encoder_swap":
                # Query-encoder upgrade: the store is NOT mutated. Instead, eval
                # queries are re-embedded with the swapped encoder and used for
                # every post-drift measurement. The drift object carries the
                # frozen seeded projection that C2O reuses to re-embed docs.
                query_drift, drifted_query_vectors = _build_query_encoder_drift(
                    engine, eval_queries, config
                )
                manifest = query_drift.manifest()
                manifest["injection_index"] = 0
                if split_active:
                    anchor_pairs = _build_anchor_pairs(
                        engine, query_drift, sliced_queries, sliced_qrels, anchor_ids
                    )
                    # Snapshot the ORIGINAL cached doc vectors before any
                    # supervised correction mutates the store, so each cycle
                    # applies the adapter to the same fixed input (the adapter is
                    # trained as adapter(original_doc) -> query; applying it to an
                    # already-mutated store across cycles would compound).
                    original_doc_points = _snapshot_doc_points(engine)
            else:
                injector = DriftInjector(engine, seed=config.seed)
                manifest = _inject_drift(injector, config)
            run_config = asdict(config)
            run_config["injection_index"] = manifest["injection_index"]
            run_config["device"] = config.device or _detect_device()
            run_config["corpus_size"] = len(sliced_corpus)
            run_config["query_count"] = len(sliced_queries)
            run_config["anchor_count"] = len(anchor_ids)
            run_config["eval_count"] = len(eval_ids)

            tracker = RecoveryTracker(baseline_ndcg=baseline_ndcg)
            correction_norms = []

            for cycle_index in range(1, config.cycles + 1):
                metadata: Dict[str, Any] = {"condition": config.condition}
                metadata.update(
                    _run_condition_cycle(
                        engine,
                        config.condition,
                        eval_queries,
                        manifest,
                        run_config,
                        query_drift=query_drift,
                        drifted_eval_vectors=drifted_query_vectors,
                        eval_qrels=sliced_qrels,
                        anchor_pairs=anchor_pairs,
                        baseline_ndcg=baseline_ndcg,
                        original_doc_points=original_doc_points,
                    )
                )
                if drifted_query_vectors is not None:
                    ndcg, details = evaluate_engine_with_query_vectors(
                        engine, drifted_query_vectors, sliced_qrels, config.k
                    )
                else:
                    ndcg, details = evaluate_engine(engine, eval_queries, sliced_qrels, config.k)
                metadata["query_ndcg"] = details
                metadata["evaluated_queries"] = len(details)
                if config.condition in {"C3", "C4"}:
                    # C3/C4 never mutate the store, so adapter(stored)-stored is a
                    # valid post-cycle correction-magnitude probe.
                    norm_stats = _correction_norm_stats(engine)
                    metadata["correction_norm_stats"] = norm_stats
                    correction_norms.extend(norm_stats["sample_norms"])
                elif config.condition in {"C3a", "C4a"}:
                    # C3a/C4a already recorded the correction ACTUALLY written
                    # (adapted - original) inside the supervised cycle; do not
                    # re-probe a mutated store.
                    norm_stats = metadata.get("correction_norm_stats")
                    if isinstance(norm_stats, dict):
                        correction_norms.extend(norm_stats.get("sample_norms", []))
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
                "anchor_eval_split": {
                    "active": split_active,
                    "anchor_count": len(anchor_ids),
                    "eval_count": len(eval_ids),
                    "anchor_ids": list(anchor_ids),
                    "eval_ids": list(eval_ids),
                },
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


def _split_anchor_eval(
    queries: Mapping[str, str],
    anchor_fraction: float,
    seed: int,
    active: bool,
) -> Tuple[list, list]:
    """Deterministically split query ids into disjoint (anchor, eval) lists.

    When ``active`` is False the anchor list is empty and every query is an eval
    query — preserving the pre-PR-A2b behavior bit-for-bit. When active, ids are
    sorted then shuffled by a ``seed``-derived RNG so the split is reproducible
    and independent of dict iteration order. At least one anchor and one eval id
    are guaranteed when there are >= 2 queries.
    """
    ordered_ids = sorted(canonicalize_id(qid) for qid in queries.keys())
    if not active or float(anchor_fraction) <= 0.0:
        return [], ordered_ids
    rng = random.Random(seed)
    shuffled = list(ordered_ids)
    rng.shuffle(shuffled)
    total = len(shuffled)
    anchor_n = int(round(total * float(anchor_fraction)))
    anchor_n = max(1, min(anchor_n, total - 1)) if total >= 2 else 0
    anchor_ids = sorted(shuffled[:anchor_n])
    eval_ids = sorted(shuffled[anchor_n:])
    return anchor_ids, eval_ids


def _build_anchor_pairs(
    engine,
    query_drift: Any,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, float]],
    anchor_ids: Sequence[str],
) -> list:
    """Build supervised anchor pairs (cached doc vector, drifted query vector).

    For every anchor query and each of its relevant docs, the positive pair is
    the doc's CACHED stored vector (fetched from the store by canonical doc_id)
    and the anchor query's DRIFTED vector (re-embedded with the swap encoder).
    These pairs encode the pre-drift relevance the adapter must re-align toward.
    """
    doc_vectors = _doc_vectors_by_id(engine)
    anchor_id_list = list(anchor_ids)
    anchor_texts = [str(queries[qid]) for qid in anchor_id_list]
    if not anchor_texts:
        return []
    drifted = query_drift.embed_queries(anchor_texts)
    drifted_by_id = {
        canonicalize_id(qid): drifted[index] for index, qid in enumerate(anchor_id_list)
    }
    pairs = []
    for qid in anchor_id_list:
        canonical_qid = canonicalize_id(qid)
        relevance = qrels.get(qid, qrels.get(canonical_qid, {}))
        query_vector = drifted_by_id[canonical_qid]
        for doc_id, score in relevance.items():
            if float(score) <= 0.0:
                continue
            doc_vector = doc_vectors.get(canonicalize_id(doc_id))
            if doc_vector is None:
                continue
            pairs.append(
                {
                    "query_id": canonical_qid,
                    "doc_id": canonicalize_id(doc_id),
                    "doc_vector": np.asarray(doc_vector, dtype=np.float32),
                    "query_vector": np.asarray(query_vector, dtype=np.float32),
                }
            )
    return pairs


def _doc_vectors_by_id(engine) -> Dict[str, np.ndarray]:
    """Map canonical doc_id -> cached stored vector for every doc in the store."""
    vectors: Dict[str, np.ndarray] = {}
    offset = None
    while True:
        points, next_offset = engine.qdrant.scroll(
            collection_name=engine.collection_name,
            limit=256,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        for point in points:
            payload = getattr(point, "payload", None) or {}
            doc_id = payload.get("doc_id", payload.get("original_id", point.id))
            vectors[canonicalize_id(doc_id)] = _point_vector(point.vector)
        if next_offset is None or not points:
            break
        offset = next_offset
    return vectors


def _build_engine(config: DriftRecoveryConfig, corpus: Mapping[str, str]):
    from antigravity_engine import AntigravityEngine

    engine = AntigravityEngine(
        qdrant_location=":memory:",
        model_name=config.model,
        store_full_text_payload=True,
    )
    if config.condition in {"C3", "C4", "C3a", "C4a"}:
        engine.adapter = create_adapter(
            "mlp",
            input_dim=engine.vector_size,
            bounded=(config.condition in {"C3", "C3a"}),
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
    drifted_eval_vectors: Optional[Mapping[str, np.ndarray]] = None,
    eval_qrels: Optional[Mapping[str, Mapping[str, float]]] = None,
    anchor_pairs: Optional[Sequence[Mapping[str, Any]]] = None,
    baseline_ndcg: float = 0.0,
    original_doc_points: Optional[Sequence[Any]] = None,
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
    if condition in SUPERVISED_CONDITIONS:
        return _supervised_anchor_cycle(
            engine,
            condition,
            config,
            drifted_eval_vectors=drifted_eval_vectors,
            eval_qrels=eval_qrels,
            anchor_pairs=anchor_pairs,
            baseline_ndcg=baseline_ndcg,
            original_doc_points=original_doc_points,
        )
    raise ValueError(f"Unsupported condition: {condition}")


def _supervised_anchor_cycle(
    engine,
    condition: str,
    config: Mapping[str, Any],
    drifted_eval_vectors: Optional[Mapping[str, np.ndarray]],
    eval_qrels: Optional[Mapping[str, Mapping[str, float]]],
    anchor_pairs: Optional[Sequence[Mapping[str, Any]]],
    baseline_ndcg: float,
    original_doc_points: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Supervised anchor-pair InfoNCE closed loop (PR-A2b).

    This is the CRUX of PR-A2b and is deliberately NOT run_sedimentation_cycle
    (whose unsupervised homeostatic targets never fired in v1). The actuator
    fires when an NDCG-drop trigger crosses threshold, trains the MLP adapter so
    adapted cached doc vectors realign with the drifted query space using
    held-out anchor (query, relevant-doc) pairs, then applies the trained adapter
    to ALL stored docs and writes them back (store mutates -> correction_applied).
    """
    if drifted_eval_vectors is None or eval_qrels is None:
        raise ValueError(f"{condition} requires drifted eval vectors and eval qrels")

    controller = getattr(engine, "_annealing_controller", None)
    if controller is None:
        controller = engine.enable_annealing_controller(
            trigger_threshold=float(config.get("trigger_threshold", 0.0)),
            max_temperature=float(config.get("max_temperature", 1.0)),
            cooling_rate=0.5,
        )

    # Trigger (P1): an explicit NDCG-drop magnitude drives the controller, unlike
    # C3/C4's internal structural-report magnitude that stayed flat in v1.
    current_ndcg, _ = evaluate_engine_with_query_vectors(
        engine, drifted_eval_vectors, eval_qrels, int(config.get("k", 10))
    )
    ndcg_drop = max(0.0, float(baseline_ndcg) - float(current_ndcg))
    observation = engine.observe_annealing_drift(drift_magnitude=ndcg_drop)
    should_correct = bool(controller.should_correct())

    anchor_count = len(anchor_pairs) if anchor_pairs else 0
    eval_count = len(drifted_eval_vectors)
    checksum_before = _vector_store_checksum(engine)
    correction_attempted = False
    correction_applied = False
    # Norm of the correction ACTUALLY written to the store (adapted - original).
    # For C3a/C4a the store is mutated, so measuring adapter(stored)-stored after
    # the fact would re-apply the adapter and mislead; capture the real delta here.
    correction_norm_stats = _aggregate_norm_stats([])

    if should_correct and anchor_pairs:
        import torch

        correction_attempted = True
        # Re-initialize the adapter from scratch each supervised cycle so the
        # correction is a pure function of (original doc snapshot, anchor pairs,
        # seed) — i.e. genuinely idempotent across cycles. Without this, re-
        # training from already-accumulated weights makes the trajectory drift
        # cycle-over-cycle (Tier B found norm 0.136->0.497->0.443). Seed BEFORE
        # create_adapter so the random weight init is identical every cycle (the
        # global torch RNG state otherwise differs after the prior cycle's
        # training, making cycle 1 != cycle 2). Each cycle is then a clean,
        # reproducible one-shot supervised correction from the fixed originals.
        torch.manual_seed(int(config.get("seed", 0)))
        engine.adapter = create_adapter(
            "mlp",
            input_dim=engine.vector_size,
            bounded=(condition == "C3a"),
            min_correction=float(config.get("bound_epsilon", 0.01)),
            max_correction=0.5,
        )
        _train_adapter_on_anchor_pairs(
            engine,
            anchor_pairs,
            seed=int(config.get("seed", 0)),
        )
        applied = _apply_adapter_to_all_docs(engine, original_points=original_doc_points)
        correction_norm_stats = applied["correction_norm_stats"]
        correction_applied = checksum_before != _vector_store_checksum(engine)

    return {
        "action": "supervised_anchor_infonce_correction",
        "bounded": condition == "C3a",
        "detector_source": "ndcg_drop_vs_baseline",
        "should_correct": should_correct,
        "sedimentation_attempted": correction_attempted,
        "correction_applied": correction_applied,
        "anchor_count": anchor_count,
        "eval_count": eval_count,
        "baseline_ndcg": float(baseline_ndcg),
        "pre_correction_ndcg": float(current_ndcg),
        "ndcg_drop": float(ndcg_drop),
        "annealing_observation": observation,
        "correction_norm_stats": correction_norm_stats,
    }


def _train_adapter_on_anchor_pairs(
    engine,
    anchor_pairs: Sequence[Mapping[str, Any]],
    seed: int,
    steps: int = 30,
    learning_rate: float = 0.01,
) -> None:
    """Train engine.adapter so adapted cached doc vectors match drifted queries.

    Loss is InfoNCE between adapter(doc_vectors) and the drifted anchor query
    vectors (positives are the matching rows). For the bounded adapter (C3a) the
    forward pass already applies the magnitude bound, so the learned realignment
    is capped — this is the honest capacity limit under test.

    SCOPE CAVEAT (PR-A4 must sweep this): the default 30 steps / lr 0.01 budget
    under-fits even in-sample on the tiny stub geometry (Tier B measured ~0.515
    in-sample NDCG with ALL anchors at this budget vs 1.0 at 2000 steps/lr 0.10).
    So the stub negative reflects training budget + sparse-anchor generalization,
    not purely generalization. The PR-A4 real-data campaign MUST sweep steps/lr
    before concluding the supervised loop does not help on real data.
    """
    import torch

    from sedimentation_loss import SedimentationInfoNCELoss

    doc_tensor = torch.tensor(
        np.asarray([pair["doc_vector"] for pair in anchor_pairs], dtype=np.float32),
        dtype=torch.float32,
    )
    query_tensor = torch.tensor(
        np.asarray([pair["query_vector"] for pair in anchor_pairs], dtype=np.float32),
        dtype=torch.float32,
    )
    loss_fn = SedimentationInfoNCELoss()
    optimizer = torch.optim.Adam(engine.adapter.parameters(), lr=float(learning_rate))
    torch.manual_seed(int(seed))
    engine.adapter.train()
    for _ in range(int(steps)):
        optimizer.zero_grad()
        adapted = engine.adapter(doc_tensor)
        loss = loss_fn(adapted, query_tensor)
        loss.backward()
        optimizer.step()
    engine.adapter.eval()


def _snapshot_doc_points(engine) -> list:
    """Return a frozen list of (id, original_vector, payload) for every doc.

    Captured once before any supervised correction so each cycle re-derives the
    corrected store from the SAME fixed inputs (adapter trained as
    adapter(original_doc) -> query) instead of compounding across cycles.
    """
    snapshot = []
    offset = None
    while True:
        batch, next_offset = engine.qdrant.scroll(
            collection_name=engine.collection_name,
            limit=256,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        for point in batch:
            snapshot.append(
                {
                    "id": point.id,
                    "vector": _point_vector(point.vector),
                    "payload": getattr(point, "payload", None),
                }
            )
        if next_offset is None or not batch:
            break
        offset = next_offset
    return snapshot


def _apply_adapter_to_all_docs(engine, original_points: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """Apply the trained adapter to every doc vector and upsert it back.

    When ``original_points`` (the pre-correction snapshot) is supplied, the
    adapter is applied to those fixed original vectors so re-running across cycles
    is idempotent. Otherwise it falls back to scrolling the live store.
    """
    import torch

    from qdrant_client.models import PointStruct

    if original_points is not None:
        points = [
            type("P", (), {"id": item["id"], "vector": item["vector"], "payload": item["payload"]})()
            for item in original_points
        ]
    else:
        points = []
        offset = None
        while True:
            batch, next_offset = engine.qdrant.scroll(
                collection_name=engine.collection_name,
                limit=256,
                with_vectors=True,
                with_payload=True,
                offset=offset,
            )
            points.extend(batch)
            if next_offset is None or not batch:
                break
            offset = next_offset
    if not points:
        return {"updated": 0, "failed": 0, "correction_norm_stats": _aggregate_norm_stats([])}
    vectors = np.asarray([_point_vector(point.vector) for point in points], dtype=np.float32)
    with torch.no_grad():
        adapted = engine.adapter(torch.tensor(vectors, dtype=torch.float32)).detach().cpu().numpy()
    adapted = np.asarray(adapted, dtype=np.float32)
    correction_norms = np.linalg.norm(adapted - vectors, axis=1)
    upserts = [
        PointStruct(id=point.id, vector=adapted[index], payload=point.payload)
        for index, point in enumerate(points)
    ]
    engine.qdrant.upsert(collection_name=engine.collection_name, points=upserts)
    return {
        "updated": len(upserts),
        "failed": 0,
        "correction_norm_stats": _aggregate_norm_stats([float(value) for value in correction_norms]),
    }


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
    if config.condition in SUPERVISED_CONDITIONS:
        if config.drift != "query_encoder_swap":
            raise ValueError(
                f"condition {config.condition} is only valid with drift=query_encoder_swap"
            )
        if not (math.isfinite(float(config.anchor_fraction)) and 0.0 < float(config.anchor_fraction) < 1.0):
            raise ValueError(
                f"condition {config.condition} requires 0 < anchor_fraction < 1"
            )
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
    if not (math.isfinite(float(config.anchor_fraction)) and 0.0 <= float(config.anchor_fraction) < 1.0):
        raise ValueError("anchor_fraction must be a finite number in [0, 1)")


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
    parser.add_argument("--anchor-fraction", type=float, default=0.0)
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
        anchor_fraction=args.anchor_fraction,
    )
    result = run_experiment(config)
    print(json.dumps({"output": args.output, "final_ndcg": result["recovery"]["trajectory"][-1]["ndcg"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
