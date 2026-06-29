"""Post-bank runtime glue (Phase II — H5b / S2b runtime).

Ties the decoupled builder (``post_bank_correction.build_post_bank``) to the
engine's vector store:

- ``build_post_bank_from_anchor_pairs`` extracts the (doc, drifted-query) vectors
  from the harness ``anchor_pairs`` and builds a ``SteeringPostBank`` via a
  caller-supplied ``make_post`` factory (the C5/C5s/C5r conditions supply a real
  bounded-adapter trainer).
- ``apply_post_bank_to_store`` routes every stored doc to its nearest post,
  applies that post's correction, and writes the corrected vectors back — returning
  the updated count, per-post hit counts, and correction-norm stats (the
  store-mutation evidence).

``make_post`` is injected, so this module is numpy-only and unit-testable with stub
corrections + a fake store. The C5/C5s/C5r dispatch (S2b) supplies the real adapter
training and the live engine; the GPU head-to-head campaign is deferred.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from post_bank_correction import MakePost, build_post_bank
from steering_post_bank import SteeringPostBank


def _norm_stats(norms: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(norms), dtype=float)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def build_post_bank_from_anchor_pairs(
    anchor_pairs: Sequence[Dict[str, Any]],
    vector_size: int,
    k: int,
    make_post: MakePost,
    seed: int,
    **bank_kwargs: Any,
) -> SteeringPostBank:
    """Build a post-bank from the harness anchor pairs.

    Each pair carries ``doc_vector`` (the cached original doc embedding) and
    ``query_vector`` (the drifted anchor-query embedding) — the same keys the
    single-adapter supervised cycle trains on. Clusters the doc vectors and builds
    one post per cluster via ``make_post``.
    """
    if not anchor_pairs:
        raise ValueError("anchor_pairs must be non-empty to build a post bank")
    doc_vectors = np.asarray([p["doc_vector"] for p in anchor_pairs], dtype=float)
    query_vectors = np.asarray([p["query_vector"] for p in anchor_pairs], dtype=float)
    if doc_vectors.ndim != 2 or doc_vectors.shape[1] != int(vector_size):
        raise ValueError(
            f"anchor doc vectors dim {doc_vectors.shape[1] if doc_vectors.ndim == 2 else doc_vectors.shape} "
            f"!= vector_size {vector_size}"
        )
    return build_post_bank(doc_vectors, query_vectors, k, make_post, seed, **bank_kwargs)


def apply_post_bank_to_store(
    engine: Any, bank: SteeringPostBank, original_points: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Route each stored doc to its post, apply the correction, and upsert it back.

    ``original_points``: a pre-correction snapshot, each ``{"id", "vector",
    "payload"}`` (the same shape ``_snapshot_doc_points`` returns). Applying to the
    fixed snapshot keeps the correction idempotent across cycles. Returns the
    updated count, per-post hits, and correction-norm stats (store-mutation
    evidence).
    """
    from qdrant_client.models import PointStruct

    if not original_points:
        return {"updated": 0, "per_post_hits": {}, "correction_norm_stats": _norm_stats([])}

    vectors = np.asarray([np.asarray(p["vector"], dtype=float) for p in original_points], dtype=float)
    corrected = bank.apply(vectors)  # routes each doc to its post + applies; logs the mutation
    norms = np.linalg.norm(corrected - vectors, axis=1)
    upserts = [
        PointStruct(
            id=original_points[i]["id"],
            vector=corrected[i].tolist(),
            payload=original_points[i].get("payload"),
        )
        for i in range(len(original_points))
    ]
    engine.qdrant.upsert(collection_name=engine.collection_name, points=upserts)

    apply_logs = [e for e in bank.lifecycle_log if e["action"] == "apply"]
    per_post_hits = apply_logs[-1]["per_post_hits"] if apply_logs else {}
    return {
        "updated": len(upserts),
        "per_post_hits": per_post_hits,
        "correction_norm_stats": _norm_stats(norms),
    }
