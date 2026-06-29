"""Build a steering-post bank from supervised anchor pairs (Phase II — H5b / S2a).

The supervised loop (C3a) trains ONE bounded adapter on ALL held-out anchor
(doc, drifted-query) pairs. The post-bank thesis instead partitions the document
space into clusters and gives each cluster its own *post* — a correction trained
only on that cluster's anchors, keyed by the cluster centroid. A stored doc is
then routed to its nearest post and corrected locally.

This module owns the **bank-from-anchors construction**: seeded k-means over the
anchor doc vectors + assembly of a ``SteeringPostBank`` whose posts carry the
per-cluster corrections. It is **decoupled from the engine and from torch**: the
caller supplies a ``make_post`` factory that turns a cluster's (doc, query) pairs
into a correction callable (in production: train a bounded adapter; in tests: a
stub). So the clustering + assembly is unit-testable without a GPU. The harness
conditions C5 / C5s / C5r (slice S2b) supply the real adapter factory and the
store read/write, and drive the lifecycle (anneal/prune/re-anneal).
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from steering_post_bank import SteeringPostBank

Correction = Callable[[np.ndarray], np.ndarray]
# (cluster_doc_vectors, cluster_query_vectors) -> correction callable for that cluster.
MakePost = Callable[[np.ndarray, np.ndarray], Correction]


def cluster_vectors(
    vectors: np.ndarray, k: int, seed: int, iters: int = 25
) -> Tuple[np.ndarray, np.ndarray]:
    """Seeded k-means. Returns ``(labels (N,), centroids (k, d))``.

    Deterministic for a fixed ``(vectors, k, seed)``. ``k`` is clamped to
    ``[1, N]``. Empty clusters are re-seeded by stealing the worst-fit member of
    the currently-largest cluster (counts recomputed per empty), so every returned
    centroid is backed by >= 1 point and labels are always in ``[0, k)`` — this is
    always achievable because ``k <= N``.
    """
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2:
        raise ValueError("vectors must be 2D (N, d)")
    n = arr.shape[0]
    if n == 0:
        raise ValueError("vectors must be non-empty")
    if int(k) <= 0:
        raise ValueError("k must be a positive integer")
    k = min(int(k), n)

    rng = np.random.default_rng(int(seed))
    # k-means++-lite seeding: pick distinct initial centroids deterministically.
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = arr[init_idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max(1, int(iters))):
        # Assign: nearest centroid by squared Euclidean distance.
        dists = np.linalg.norm(arr[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        # Re-seed empty clusters: steal the worst-fit member from the CURRENTLY
        # largest cluster (counts recomputed each time), so we never create a new
        # empty. With k <= N the largest cluster always has >= 2 members while any
        # cluster is empty (pigeonhole), so the donor keeps >= 1 — every cluster
        # ends non-empty. (The prior version recomputed `worst` from a constant
        # `dists` and overwrote the same point, leaving >=2 simultaneous empties
        # unfilled on duplicate/degenerate input.)
        for c in range(k):
            if not np.any(new_labels == c):
                counts = np.bincount(new_labels, minlength=k)
                donor = int(np.argmax(counts))
                members = np.where(new_labels == donor)[0]
                worst_member = int(members[np.argmax(np.min(dists[members], axis=1))])
                new_labels[worst_member] = c
        moved = not np.array_equal(new_labels, labels)
        labels = new_labels
        new_centroids = np.stack(
            [arr[labels == c].mean(axis=0) if np.any(labels == c) else centroids[c] for c in range(k)]
        )
        centroid_shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        if not moved and centroid_shift < 1e-12:
            break
    return labels, centroids


def build_post_bank(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    make_post: MakePost,
    seed: int,
    prune_below: float = 0.0,
    min_posts: int = 1,
    temperature: float = 0.0,
) -> SteeringPostBank:
    """Cluster the anchor doc vectors into ``k`` groups and assemble a post-bank.

    For each cluster, ``make_post(cluster_doc_vectors, cluster_query_vectors)``
    returns the correction callable for that cluster's region; it is registered as
    a post keyed by the cluster centroid. The number of posts equals the number of
    non-empty clusters (== ``min(k, n_anchors)`` after re-seeding).
    """
    docs = np.asarray(doc_vectors, dtype=float)
    queries = np.asarray(query_vectors, dtype=float)
    if docs.ndim != 2 or queries.ndim != 2:
        raise ValueError("doc_vectors and query_vectors must be 2D (N, d)")
    if docs.shape[0] != queries.shape[0]:
        raise ValueError("doc_vectors and query_vectors must have the same N")
    dim = docs.shape[1]

    labels, centroids = cluster_vectors(docs, k, seed)
    bank = SteeringPostBank(
        input_dim=dim, prune_below=prune_below, min_posts=min_posts, temperature=temperature
    )
    for c in range(centroids.shape[0]):
        mask = labels == c
        correction = make_post(docs[mask], queries[mask])
        bank.register_post(f"post:{c}", centroids[c], correction)
    return bank
