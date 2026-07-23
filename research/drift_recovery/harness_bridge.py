"""The single import boundary between D1 and the merged CHELATEDAI harness.

No other module under :mod:`research.drift_recovery` may import root-level
harness modules. Statistical work consumes a frozen :class:`EmbeddingPack`.
"""

from __future__ import annotations

import os
import statistics
from typing import Any, Dict, Mapping, Sequence

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Merged-harness imports: this is intentionally the only module containing them.
from benchmark_utils import canonicalize_id, load_mteb_data  # noqa: E402
from chelation_adapter import create_adapter  # noqa: E402
from drift_recovery_metrics import ndcg_at_k  # noqa: E402
from query_encoder_drift import QueryEncoderDrift  # noqa: E402
from run_drift_recovery_experiment import _split_anchor_eval  # noqa: E402
from run_road_course_campaign import select_road_course_slice  # noqa: E402
from sedimentation_loss import SedimentationInfoNCELoss  # noqa: E402


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0.0, norms, 1.0)


def per_query_ndcg(
    query_vectors: np.ndarray,
    query_ids: Sequence[str],
    doc_vectors: np.ndarray,
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    k: int = 10,
) -> np.ndarray:
    """Rank by cosine and score with the merged harness's binary NDCG."""

    if len(query_vectors) != len(query_ids):
        raise ValueError("query vector/id length mismatch")
    if len(doc_vectors) != len(doc_ids):
        raise ValueError("document vector/id length mismatch")
    docs = _normalize(np.asarray(doc_vectors, dtype=np.float64))
    queries = _normalize(np.asarray(query_vectors, dtype=np.float64))
    canonical_docs = np.asarray([canonicalize_id(value) for value in doc_ids], dtype=str)
    rows = []
    for query_id, vector in zip(query_ids, queries):
        qid = canonicalize_id(query_id)
        relevance = qrels.get(query_id, qrels.get(qid, {}))
        relevant = [canonicalize_id(doc_id) for doc_id, score in relevance.items() if float(score) > 0.0]
        if not relevant:
            raise ValueError(f"eval query {qid} has no positive qrels")
        similarities = docs @ vector
        # Stable full ordering makes the otherwise-unspecified equal-score case deterministic.
        ranked_idx = np.argsort(-similarities, kind="mergesort")[: min(k, len(canonical_docs))]
        ranked = canonical_docs[ranked_idx].tolist()
        rows.append(ndcg_at_k(ranked, relevant, k=k))
    return np.asarray(rows, dtype=np.float64)


def per_query_ndcg_from_rankings(
    ranked_indices: np.ndarray,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    k: int = 10,
) -> np.ndarray:
    """Score frozen score-space rankings through the merged NDCG boundary."""

    rankings = np.asarray(ranked_indices, dtype=np.int64)
    if rankings.ndim != 2 or rankings.shape[0] != len(query_ids):
        raise ValueError("ranked index/query shape mismatch")
    if rankings.shape[1] < min(int(k), len(doc_ids)):
        raise ValueError("rankings do not contain enough documents for k")
    if np.any(rankings < 0) or np.any(rankings >= len(doc_ids)):
        raise ValueError("ranked document index is out of bounds")
    canonical_docs = np.asarray([canonicalize_id(value) for value in doc_ids], dtype=str)
    rows = []
    for row_index, query_id in enumerate(query_ids):
        qid = canonicalize_id(query_id)
        relevance = qrels.get(query_id, qrels.get(qid, {}))
        relevant = [canonicalize_id(doc_id) for doc_id, score in relevance.items() if float(score) > 0.0]
        if not relevant:
            raise ValueError(f"eval query {qid} has no positive qrels")
        ranked = canonical_docs[rankings[row_index, : int(k)]].tolist()
        rows.append(ndcg_at_k(ranked, relevant, k=int(k)))
    return np.asarray(rows, dtype=np.float64)


def assert_score_transform_parity(
    ranked_indices: np.ndarray,
    stored_scores: np.ndarray,
    expected_aggregate: float,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    k: int = 10,
    atol: float = 1e-12,
) -> Dict[str, float]:
    """Prove a score-space ranking reproduces merged-harness NDCG."""

    recomputed = per_query_ndcg_from_rankings(
        ranked_indices, query_ids, doc_ids, qrels, k=int(k)
    )
    stored = np.asarray(stored_scores, dtype=np.float64)
    if stored.shape != recomputed.shape:
        raise AssertionError("score-transform stored/recomputed shape mismatch")
    per_query_difference = float(np.max(np.abs(recomputed - stored)))
    aggregate_difference = abs(aggregate_ndcg(recomputed) - float(expected_aggregate))
    if per_query_difference > float(atol) or aggregate_difference > float(atol):
        raise AssertionError(
            "score-transform harness parity failed: "
            f"per-query={per_query_difference:.3g}, aggregate={aggregate_difference:.3g}, "
            f"atol={float(atol):.3g}"
        )
    return {
        "per_query_max_abs": per_query_difference,
        "aggregate_abs": aggregate_difference,
    }


def aggregate_ndcg(scores: Sequence[float]) -> float:
    """Match the merged evaluator's ``statistics.fmean`` aggregation."""

    return float(statistics.fmean(float(value) for value in scores))


def assert_harness_parity(pack: Any, atol: float = 1e-12) -> Dict[str, float]:
    """Prove frozen per-query scores reproduce merged-harness aggregate NDCG."""

    references = dict(pack.metadata.get("harness_aggregate_ndcg", {}))
    if not references:
        raise AssertionError("pack has no harness aggregate references")
    differences: Dict[str, float] = {}
    document_matrices = {"floor": pack.Do, "oracle": pack.Dor}
    document_matrices.update(
        {
            name[len("method_documents__") :]: values
            for name, values in pack.extra_arrays.items()
            if name.startswith("method_documents__")
        }
    )
    for method, expected in references.items():
        if method not in pack.per_query_scores:
            raise AssertionError(f"pack is missing reference method {method}")
        if method not in document_matrices:
            raise AssertionError(f"pack is missing frozen document vectors for {method}")
        recomputed = per_query_ndcg(
            pack.Qd,
            pack.query_ids,
            document_matrices[method],
            pack.doc_ids,
            pack.qrels,
            k=int(pack.metadata.get("k", 10)),
        )
        stored = np.asarray(pack.per_query_scores[method], dtype=np.float64)
        per_query_difference = float(np.max(np.abs(recomputed - stored)))
        if per_query_difference > atol:
            raise AssertionError(
                f"{method} per-query pack/harness parity failed: max difference "
                f"{per_query_difference:.3g} > {atol:.3g}"
            )
        actual = aggregate_ndcg(recomputed)
        difference = abs(actual - float(expected))
        differences[method] = difference
        if difference > atol:
            raise AssertionError(
                f"{method} pack/harness parity failed: {actual:.17g} vs {float(expected):.17g} "
                f"(difference {difference:.3g} > {atol:.3g})"
            )
    return differences


def load_scifact_evalsplit(seed: int = 42) -> Dict[str, Any]:
    """Load the exact waypoint eval-split recipe and leakage-free curve pools."""

    corpus, queries, qrels = load_mteb_data("SciFact")
    if corpus is None or queries is None or qrels is None:
        raise RuntimeError("offline SciFact cache is unavailable")
    sliced_corpus, sliced_queries, sliced_qrels = select_road_course_slice(
        corpus,
        queries,
        qrels,
        max_queries=100,
        sample_docs=1200,
        seed=seed,
    )
    anchor_ids, eval_ids = _split_anchor_eval(sliced_queries, 0.4, seed, active=True)
    eval_set = set(map(canonicalize_id, eval_ids))
    anchor_set = set(map(canonicalize_id, anchor_ids))
    if eval_set.intersection(anchor_set):
        raise AssertionError("merged split leaked anchor IDs into eval IDs")

    doc_ids = [canonicalize_id(value) for value in sliced_corpus]
    eval_query_ids = [
        canonicalize_id(value)
        for value in sliced_queries
        if sliced_qrels.get(value) and canonicalize_id(value) in eval_set
    ]
    canonical_anchor_ids = [
        canonicalize_id(value) for value in sliced_queries if canonicalize_id(value) in anchor_set
    ]
    query_by_id = {canonicalize_id(key): str(value) for key, value in queries.items()}
    corpus_by_id = {canonicalize_id(key): str(value) for key, value in corpus.items()}
    qrels_by_id = {
        canonicalize_id(query_id): {
            canonicalize_id(doc_id): float(score) for doc_id, score in rels.items()
        }
        for query_id, rels in qrels.items()
    }
    sliced_qrels_by_id = {
        canonicalize_id(query_id): {
            canonicalize_id(doc_id): float(score) for doc_id, score in rels.items()
        }
        for query_id, rels in sliced_qrels.items()
    }

    # Canonical C3a pool: every positive pair from the exact 40-query anchor split.
    selected_doc_set = set(doc_ids)
    canonical_pairs = []
    for query_id in canonical_anchor_ids:
        for doc_id, score in sorted(qrels_by_id.get(query_id, {}).items()):
            if float(score) > 0.0 and doc_id in selected_doc_set:
                canonical_pairs.append((query_id, doc_id))

    # Learning-curve query pool: one positive document per non-eval SciFact
    # query. This reaches 128 distinct query anchors without touching eval qrels.
    curve_pairs = []
    for query_id in sorted(query_by_id):
        if query_id in eval_set:
            continue
        positives = sorted(
            doc_id for doc_id, score in qrels_by_id.get(query_id, {}).items() if float(score) > 0.0
        )
        if positives:
            curve_pairs.append((query_id, positives[0]))
    if len(curve_pairs) < 128:
        raise RuntimeError(f"only {len(curve_pairs)} leakage-free C3a curve anchors; need 128")

    return {
        "corpus_texts": [str(sliced_corpus[key]) for key in sliced_corpus],
        "doc_ids": doc_ids,
        "eval_query_texts": [query_by_id[key] for key in eval_query_ids],
        "eval_query_ids": eval_query_ids,
        "eval_qrels": {key: sliced_qrels_by_id[key] for key in eval_query_ids},
        "anchor_ids": canonical_anchor_ids,
        "canonical_pair_query_ids": [item[0] for item in canonical_pairs],
        "canonical_pair_doc_ids": [item[1] for item in canonical_pairs],
        "canonical_pair_query_texts": [query_by_id[item[0]] for item in canonical_pairs],
        "curve_pair_query_ids": [item[0] for item in curve_pairs],
        "curve_pair_doc_ids": [item[1] for item in curve_pairs],
        "curve_pair_query_texts": [query_by_id[item[0]] for item in curve_pairs],
        "curve_pair_doc_texts": [corpus_by_id[item[1]] for item in curve_pairs],
        "all_query_count": len(queries),
    }


def load_retrieval_evalsplit(
    dataset: str,
    seed: int = 42,
    anchor_fraction: float = 0.4,
    max_queries: int = 100,
    sample_docs: int = 1200,
) -> Dict[str, Any]:
    """Load a deterministic, generic frozen-pack case through merged helpers.

    Unlike :func:`load_scifact_evalsplit`, this generic path does not construct
    C3a or learning-curve supervision.  It is intentionally limited to the
    document/query/qrels arrays needed by the D3 ridge estimator regimes.
    """

    corpus, queries, qrels = load_mteb_data(str(dataset))
    if corpus is None or queries is None or qrels is None:
        raise RuntimeError(f"offline {dataset} cache is unavailable")
    sliced_corpus, sliced_queries, sliced_qrels = select_road_course_slice(
        corpus,
        queries,
        qrels,
        max_queries=int(max_queries),
        sample_docs=int(sample_docs),
        seed=int(seed),
    )
    anchor_ids, eval_ids = _split_anchor_eval(
        sliced_queries,
        float(anchor_fraction),
        int(seed),
        active=True,
    )
    anchor_set = set(map(canonicalize_id, anchor_ids))
    eval_set = set(map(canonicalize_id, eval_ids))
    if anchor_set.intersection(eval_set):
        raise AssertionError("merged split leaked anchor IDs into eval IDs")

    query_by_id = {
        canonicalize_id(query_id): str(text) for query_id, text in sliced_queries.items()
    }
    qrels_by_id = {
        canonicalize_id(query_id): {
            canonicalize_id(doc_id): float(score) for doc_id, score in relevance.items()
        }
        for query_id, relevance in sliced_qrels.items()
    }
    eval_query_ids = [
        query_id
        for query_id in sorted(query_by_id)
        if query_id in eval_set and qrels_by_id.get(query_id)
    ]
    if not eval_query_ids:
        raise RuntimeError(f"{dataset} split produced no evaluable queries")
    validation_query_ids = [
        query_id
        for query_id in sorted(query_by_id)
        if query_id in anchor_set and qrels_by_id.get(query_id)
    ]
    if not validation_query_ids:
        raise RuntimeError(f"{dataset} split produced no anchor-validation queries")
    if set(validation_query_ids).intersection(eval_query_ids):
        raise AssertionError("anchor-validation queries leaked into eval queries")
    return {
        "dataset": str(dataset),
        "corpus_texts": [str(text) for text in sliced_corpus.values()],
        "doc_ids": [canonicalize_id(value) for value in sliced_corpus.keys()],
        "eval_query_texts": [query_by_id[query_id] for query_id in eval_query_ids],
        "eval_query_ids": eval_query_ids,
        "eval_qrels": {query_id: qrels_by_id[query_id] for query_id in eval_query_ids},
        "validation_query_texts": [query_by_id[query_id] for query_id in validation_query_ids],
        "validation_query_ids": validation_query_ids,
        "validation_qrels": {
            query_id: qrels_by_id[query_id] for query_id in validation_query_ids
        },
        "anchor_ids": list(anchor_ids),
        "anchor_fraction": float(anchor_fraction),
        "max_queries": int(max_queries),
        "sample_docs": int(sample_docs),
    }


def encode_clean_retrieval_case(
    case: Mapping[str, Any],
    device: str = "cuda",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Encode Regime-C documents and both query partitions with one model."""

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_name), device=str(device))
    documents = np.asarray(
        model.encode(
            list(case["corpus_texts"]),
            batch_size=int(batch_size),
            show_progress_bar=True,
        ),
        dtype=np.float64,
    )
    validation_queries = np.asarray(
        model.encode(
            list(case["validation_query_texts"]),
            batch_size=int(batch_size),
            show_progress_bar=False,
        ),
        dtype=np.float64,
    )
    eval_queries = np.asarray(
        model.encode(
            list(case["eval_query_texts"]),
            batch_size=int(batch_size),
            show_progress_bar=False,
        ),
        dtype=np.float64,
    )
    if documents.shape[1] != validation_queries.shape[1] or documents.shape[1] != eval_queries.shape[1]:
        raise AssertionError("same-model Regime-C embeddings changed dimension")
    first_module = model[0] if len(model) else None
    auto_model = getattr(first_module, "auto_model", None)
    model_config = getattr(auto_model, "config", None)
    model_revision = getattr(model_config, "_commit_hash", None)
    if not model_revision:
        model_revision = getattr(model_config, "_name_or_path", None)
    if not model_revision:
        model_revision = getattr(getattr(model, "model_card_data", None), "base_model_revision", None)
    return {
        "documents": documents,
        "validation_queries": validation_queries,
        "eval_queries": eval_queries,
        "model": str(model_name),
        "model_revision": str(model_revision or "offline-cache-revision-unavailable"),
        "dimension": int(documents.shape[1]),
    }


def encode_retrieval_case(
    case: Mapping[str, Any],
    swap_model: str,
    seed: int = 42,
    device: str = "cuda",
    old_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Encode one generic D3 regime with cached local models only."""

    import gc

    from sentence_transformers import SentenceTransformer

    old = SentenceTransformer(str(old_model), device=device)
    Do = np.asarray(
        old.encode(
            list(case["corpus_texts"]),
            batch_size=int(batch_size),
            show_progress_bar=True,
        ),
        dtype=np.float64,
    )
    drift = QueryEncoderDrift(
        store_dim=int(Do.shape[1]),
        swap_model_name=str(swap_model),
        seed=int(seed),
    )
    # The first call fixes the seeded frozen projection.  The document oracle
    # positions reuse that exact projection instance.
    Qd = np.asarray(drift.embed_queries(list(case["eval_query_texts"])), dtype=np.float64)
    Dor = np.asarray(drift.embed_queries(list(case["corpus_texts"])), dtype=np.float64)
    manifest = drift.manifest()
    del old
    del drift
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return {
        "Do": Do,
        "Dor": Dor,
        "Qd": Qd,
        "drift_manifest": manifest,
        "old_model": str(old_model),
        "swap_model": str(swap_model),
    }


def encode_scifact_case(case: Mapping[str, Any], seed: int = 42, device: str = "cuda") -> Dict[str, Any]:
    """Run both cached encoders once and return arrays ready for freezing."""

    from sentence_transformers import SentenceTransformer

    old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    Do = np.asarray(
        old.encode(case["corpus_texts"], batch_size=128, show_progress_bar=True), dtype=np.float64
    )
    curve_source = np.asarray(
        old.encode(case["curve_pair_doc_texts"], batch_size=128, show_progress_bar=True), dtype=np.float64
    )
    drift = QueryEncoderDrift(store_dim=Do.shape[1], swap_model_name="all-mpnet-base-v2", seed=seed)
    # One call fixes the projection; all later calls reuse exactly that instance.
    Qd = np.asarray(drift.embed_queries(list(case["eval_query_texts"])), dtype=np.float64)
    Dor = np.asarray(drift.embed_queries(list(case["corpus_texts"])), dtype=np.float64)
    canonical_target = np.asarray(
        drift.embed_queries(list(case["canonical_pair_query_texts"])), dtype=np.float64
    )
    curve_target = np.asarray(
        drift.embed_queries(list(case["curve_pair_query_texts"])), dtype=np.float64
    )
    doc_lookup = {doc_id: index for index, doc_id in enumerate(case["doc_ids"])}
    canonical_source = np.asarray(
        [Do[doc_lookup[doc_id]] for doc_id in case["canonical_pair_doc_ids"]], dtype=np.float64
    )
    return {
        "Do": Do,
        "Dor": Dor,
        "Qd": Qd,
        "canonical_source": canonical_source,
        "canonical_target": canonical_target,
        "curve_source": curve_source,
        "curve_target": curve_target,
        "drift_manifest": drift.manifest(),
        "old_model": "sentence-transformers/all-MiniLM-L6-v2",
        "swap_model": "all-mpnet-base-v2",
    }


def train_c3a_documents(
    all_documents: np.ndarray,
    anchor_documents: np.ndarray,
    anchor_queries: np.ndarray,
    seed: int,
    steps: int = 30,
    learning_rate: float = 0.01,
    min_correction: float = 0.01,
    max_correction: float = 0.5,
) -> np.ndarray:
    """Run the merged C3a bounded adapter and InfoNCE training path."""

    import torch

    torch.manual_seed(int(seed))
    adapter = create_adapter(
        "mlp",
        input_dim=int(all_documents.shape[1]),
        bounded=True,
        min_correction=float(min_correction),
        max_correction=float(max_correction),
    )
    source = torch.tensor(np.asarray(anchor_documents, dtype=np.float32), dtype=torch.float32)
    target = torch.tensor(np.asarray(anchor_queries, dtype=np.float32), dtype=torch.float32)
    loss_fn = SedimentationInfoNCELoss()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=float(learning_rate))
    # Mirrors the merged harness, including the post-construction seed call.
    torch.manual_seed(int(seed))
    adapter.train()
    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = loss_fn(adapter(source), target)
        loss.backward()
        optimizer.step()
    adapter.eval()
    with torch.no_grad():
        output = adapter(torch.tensor(np.asarray(all_documents, dtype=np.float32), dtype=torch.float32))
    return np.asarray(output.detach().cpu().numpy(), dtype=np.float64)


def system_under_test_spec() -> Dict[str, Any]:
    """Return exact code-level constants used by the Section 3 renderer."""

    from config import ChelationConfig

    return {
        "bounded_defaults": {
            "min_correction": ChelationConfig.BOUNDED_ADAPTER_MIN_CORRECTION,
            "max_correction": ChelationConfig.BOUNDED_ADAPTER_MAX_CORRECTION,
            "int8_floor": 1.0 / 128.0,
        },
        "controller_defaults": {
            "initial_temperature": 0.0,
            "cooling_rate": 0.7,
            "trigger_threshold": 0.15,
            "max_temperature": 1.0,
            "epsilon": 1e-6,
        },
        "supervised_cycle_controller": {
            "cooling_rate": 0.5,
            "drift_signal": "max(0, baseline_ndcg - current_ndcg)",
        },
    }
