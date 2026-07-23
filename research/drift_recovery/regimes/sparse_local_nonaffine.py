"""CPU-only sparse, cluster-local, non-affine retrieval drift regime.

The regime deliberately keeps data generation separate from recovery.  True
cluster identities are returned for diagnostics and relevance construction,
but a recovery method must infer its own partitions from ``corrupted_documents``.

Two deterministic warp families are available:

``quadratic`` (F1)
    ``x -> x + gamma * ((x-mu).T A (x-mu) / scale) * u``

``soft_fold`` (F2)
    A smooth hinge of the projection on one local axis, displaced along an
    orthogonal axis.  This is piecewise-like, non-affine, and non-radial.

Only NumPy is required.  The balanced Gaussian mixture, queries, anchor
indices, cluster-specific warp parameters, and corruptions are all seeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


_FAMILY_ALIASES = {
    "f1": "quadratic",
    "quadratic": "quadratic",
    "quadratic_form": "quadratic",
    "f2": "soft_fold",
    "soft-fold": "soft_fold",
    "soft_fold": "soft_fold",
    "piecewise": "soft_fold",
}


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return np.asarray(vector, dtype=np.float64) / norm


def _orthogonal_unit(rng: np.random.Generator, axis: np.ndarray) -> np.ndarray:
    """Draw a unit direction orthogonal to ``axis`` deterministically."""

    axis = _unit(axis)
    for _ in range(32):
        candidate = rng.normal(size=len(axis))
        candidate = candidate - float(candidate @ axis) * axis
        if np.linalg.norm(candidate) > 1e-10:
            return _unit(candidate)
    # This is practically unreachable, but gives a deterministic fallback.
    basis = np.zeros_like(axis)
    basis[int(np.argmin(np.abs(axis)))] = 1.0
    return _unit(basis - float(basis @ axis) * axis)


def _canonical_family(family: str) -> str:
    try:
        return _FAMILY_ALIASES[str(family).strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(_FAMILY_ALIASES))
        raise ValueError(f"unknown warp family {family!r}; expected one of {choices}") from exc


@dataclass(frozen=True)
class ClusterWarp:
    """Parameters for one harmed cluster's warp.

    ``matrix`` is populated only for F1.  ``axis`` and ``temperature`` are
    populated only for F2.  ``direction`` is the displacement direction.
    ``normalizer`` makes gamma an interpretable corpus-scale maximum: on the
    clean documents used to create the parameter, displacement is at most
    gamma (up to floating-point error).
    """

    cluster_id: int
    family: str
    center: np.ndarray
    direction: np.ndarray
    normalizer: float
    matrix: np.ndarray
    axis: np.ndarray
    projection_scale: float
    temperature: float


@dataclass(frozen=True)
class SparseLocalNonAffineData:
    """Complete synthetic preflight cell.

    ``true_cluster_ids`` and ``harmed_cluster_ids`` are diagnostic truth.  They
    are intentionally named as truth rather than inferred assignments so a
    recovery implementation cannot plausibly mistake them for fit inputs.
    """

    clean_documents: np.ndarray
    corrupted_documents: np.ndarray
    clean_queries: np.ndarray
    query_cluster_ids: np.ndarray
    query_doc_indices: np.ndarray
    qrels: np.ndarray
    true_cluster_ids: np.ndarray
    harmed_cluster_ids: np.ndarray
    anchor_indices: np.ndarray
    anchor_cluster_ids: np.ndarray
    clean_cluster_centers: np.ndarray
    within_cluster_covariances: np.ndarray
    warp_parameters: Tuple[ClusterWarp, ...]
    displacements: np.ndarray
    family: str
    gamma: float
    requested_sparsity: float
    seed: int

    # Stable, compact compatibility names used by the preflight and tests.
    @property
    def clean_docs(self) -> np.ndarray:
        return self.clean_documents

    @property
    def corrupted_docs(self) -> np.ndarray:
        return self.corrupted_documents

    @property
    def queries(self) -> np.ndarray:
        return self.clean_queries

    @property
    def relevance(self) -> np.ndarray:
        return self.qrels

    @property
    def clean_cluster_ids(self) -> np.ndarray:
        return self.true_cluster_ids

    @property
    def harmed_clusters(self) -> np.ndarray:
        return self.harmed_cluster_ids

    @property
    def warp_family(self) -> str:
        return self.family

    @property
    def changed_mask(self) -> np.ndarray:
        return np.isin(self.true_cluster_ids, self.harmed_cluster_ids)

    @property
    def displacement_norms(self) -> np.ndarray:
        return np.linalg.norm(self.displacements, axis=1)

    @property
    def displacement_mean(self) -> float:
        return float(self.displacement_norms.mean())

    @property
    def displacement_max(self) -> float:
        return float(self.displacement_norms.max(initial=0.0))

    @property
    def harmed_displacement_mean(self) -> float:
        norms = self.displacement_norms[self.changed_mask]
        return float(norms.mean()) if len(norms) else 0.0

    @property
    def actual_sparsity(self) -> float:
        return float(len(self.harmed_cluster_ids) / len(self.clean_cluster_centers))

    @property
    def warp_metadata(self) -> Dict[str, object]:
        """Return compact, JSON-safe metadata without the dense F1 matrices."""

        return {
            "family": self.family,
            "gamma": self.gamma,
            "requested_sparsity": self.requested_sparsity,
            "actual_sparsity": self.actual_sparsity,
            "harmed_cluster_ids": self.harmed_cluster_ids.tolist(),
            "cluster_normalizers": {
                str(parameter.cluster_id): parameter.normalizer
                for parameter in self.warp_parameters
            },
            "mean_displacement_all_docs": self.displacement_mean,
            "mean_displacement_harmed_docs": self.harmed_displacement_mean,
            "max_displacement": self.displacement_max,
        }


def apply_cluster_warp(
    vectors: np.ndarray,
    parameters: ClusterWarp,
    gamma: Optional[float] = None,
) -> np.ndarray:
    """Apply one cluster-specific warp to vectors from that cluster.

    This function does not inspect cluster labels.  It is public so tests can
    evaluate the warp at interpolated/off-sample points and directly falsify
    claims of affinity or radiality.
    """

    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
        squeeze = True
    elif matrix.ndim == 2:
        squeeze = False
    else:
        raise ValueError("vectors must be a vector or 2D matrix")
    if matrix.shape[1] != len(parameters.center):
        raise ValueError("vector dimension does not match warp parameters")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("vectors contain non-finite values")

    strength = 1.0 if gamma is None else float(gamma)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    local = matrix - parameters.center[None, :]
    if parameters.family == "quadratic":
        amplitude = np.einsum("ni,ij,nj->n", local, parameters.matrix, local)
    elif parameters.family == "soft_fold":
        projected = (local @ parameters.axis) / parameters.projection_scale
        temperature = parameters.temperature
        # logaddexp is the stable softplus.  Subtracting its value at zero
        # keeps the cluster center fixed without removing the hinge curvature.
        amplitude = temperature * np.logaddexp(0.0, projected / temperature)
        amplitude -= temperature * np.log(2.0)
    else:  # Guard hand-constructed parameter objects.
        raise ValueError(f"unsupported parameter family: {parameters.family!r}")
    amplitude = strength * amplitude / parameters.normalizer
    warped = matrix + amplitude[:, None] * parameters.direction[None, :]
    return warped[0] if squeeze else warped


def _gaussian_mixture(
    rng: np.random.Generator,
    n_clusters: int,
    docs_per_cluster: int,
    dimension: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create balanced separated clusters with anisotropic full covariance."""

    raw_centers = rng.normal(size=(n_clusters, dimension))
    raw_centers /= np.linalg.norm(raw_centers, axis=1, keepdims=True)
    # A fixed moderate radius keeps clusters separated without making a
    # bounded local displacement invisible under cosine retrieval.
    centers = raw_centers * 3.5
    documents = []
    labels = []
    covariances = []
    for cluster_id in range(n_clusters):
        basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        # A rotating eigenspectrum creates correlated, realistically
        # anisotropic local clouds without ill-conditioned near-zero axes.
        eigenvalues = np.exp(rng.uniform(np.log(0.02), np.log(0.18), size=dimension))
        covariance = (basis * eigenvalues[None, :]) @ basis.T
        samples = rng.multivariate_normal(
            centers[cluster_id], covariance, size=docs_per_cluster
        )
        documents.append(samples)
        labels.append(np.full(docs_per_cluster, cluster_id, dtype=np.int64))
        covariances.append(covariance)
    return (
        np.vstack(documents),
        np.concatenate(labels),
        centers,
        np.stack(covariances),
    )


def _make_queries(
    rng: np.random.Generator,
    documents: np.ndarray,
    cluster_ids: np.ndarray,
    centers: np.ndarray,
    covariances: np.ndarray,
    queries_per_cluster: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    queries = []
    query_clusters = []
    positives = []
    for cluster_id in range(len(centers)):
        candidates = np.flatnonzero(cluster_ids == cluster_id)
        chosen = rng.choice(candidates, size=queries_per_cluster, replace=True)
        noise_covariance = 0.001 * covariances[cluster_id]
        noise = rng.multivariate_normal(
            np.zeros(documents.shape[1]), noise_covariance, size=queries_per_cluster
        )
        for local_index, document_index in enumerate(chosen):
            # Alternate member-like and centroid-like queries.  Their planted
            # positives are fixed from the clean corpus below, before any warp
            # exists, so small within-cluster rank changes remain measurable.
            if local_index % 2 == 0:
                member = documents[document_index]
                member_norm = max(float(np.linalg.norm(member)), 1e-12)
                candidate_vectors = documents[candidates]
                similarities = candidate_vectors @ member / (
                    np.maximum(np.linalg.norm(candidate_vectors, axis=1), 1e-12)
                    * member_norm
                )
                similarities[candidates == document_index] = -np.inf
                neighbor = documents[candidates[int(np.argmax(similarities))]]
                # A near-boundary perturbed member makes rank damage visible
                # without changing the warp or choosing relevance post-warp.
                base = 0.5001 * member + 0.4999 * neighbor
            else:
                base = centers[cluster_id]
            queries.append(base + noise[local_index])
            query_clusters.append(cluster_id)
            positives.append(int(document_index))
    query_matrix = np.asarray(queries, dtype=np.float64)
    query_cluster_array = np.asarray(query_clusters, dtype=np.int64)
    positive_array = np.asarray(positives, dtype=np.int64)
    qrels = np.zeros((len(query_matrix), len(documents)), dtype=np.float64)
    document_norms = np.maximum(np.linalg.norm(documents, axis=1), 1e-12)
    for query_index, _cluster_id in enumerate(query_cluster_array):
        query = query_matrix[query_index]
        similarities = documents @ query / (
            document_norms * max(float(np.linalg.norm(query)), 1e-12)
        )
        positive_array[query_index] = int(np.argmax(similarities))
        qrels[query_index, positive_array[query_index]] = 1.0
    return query_matrix, query_cluster_array, positive_array, qrels


def _make_warp(
    rng: np.random.Generator,
    family: str,
    cluster_id: int,
    cluster_documents: np.ndarray,
    center: np.ndarray,
) -> ClusterWarp:
    dimension = cluster_documents.shape[1]
    local = cluster_documents - center[None, :]
    if family == "quadratic":
        raw = rng.normal(size=(dimension, dimension))
        quadratic = 0.5 * (raw + raw.T)
        quadratic -= np.trace(quadratic) / dimension * np.eye(dimension)
        quadratic /= max(float(np.linalg.norm(quadratic, ord="fro")), 1e-12)
        direction = _unit(rng.normal(size=dimension))
        raw_amplitude = np.einsum("ni,ij,nj->n", local, quadratic, local)
        normalizer = max(float(np.max(np.abs(raw_amplitude))), 1e-12)
        return ClusterWarp(
            cluster_id=cluster_id,
            family=family,
            center=center.copy(),
            direction=direction,
            normalizer=normalizer,
            matrix=quadratic,
            axis=np.zeros(dimension, dtype=np.float64),
            projection_scale=1.0,
            temperature=1.0,
        )

    axis = _unit(rng.normal(size=dimension))
    direction = _orthogonal_unit(rng, axis)
    projected_unscaled = local @ axis
    projection_scale = max(float(np.std(projected_unscaled, ddof=1)), 1e-8)
    temperature = float(rng.uniform(0.28, 0.55))
    projected = projected_unscaled / projection_scale
    raw_amplitude = temperature * np.logaddexp(0.0, projected / temperature)
    raw_amplitude -= temperature * np.log(2.0)
    normalizer = max(float(np.max(np.abs(raw_amplitude))), 1e-12)
    return ClusterWarp(
        cluster_id=cluster_id,
        family=family,
        center=center.copy(),
        direction=direction,
        normalizer=normalizer,
        matrix=np.zeros((dimension, dimension), dtype=np.float64),
        axis=axis,
        projection_scale=projection_scale,
        temperature=temperature,
    )


def generate_sparse_local_nonaffine(
    family: str = "quadratic",
    gamma: float = 0.20,
    sparsity: float = 0.15,
    anchors_per_harmed_cluster: int = 12,
    seed: int = 42,
    n_clusters: int = 20,
    docs_per_cluster: int = 48,
    dimension: int = 16,
    queries_per_cluster: int = 5,
) -> SparseLocalNonAffineData:
    """Generate one deterministic ``(family, gamma, sparsity, anchors)`` cell.

    Anchor selection is a data-generation responsibility only.  The returned
    anchors are paired clean/corrupted indices for harmed clusters; no dev/eval
    split or model fitting occurs here.
    """

    canonical_family = _canonical_family(family)
    gamma = float(gamma)
    sparsity = float(sparsity)
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    if not any(np.isclose(sparsity, allowed, rtol=0.0, atol=1e-12) for allowed in (0.15, 0.25)):
        raise ValueError("sparsity must be 0.15 or 0.25")
    if int(n_clusters) < 4:
        raise ValueError("n_clusters must be at least 4")
    if int(docs_per_cluster) < 4:
        raise ValueError("docs_per_cluster must be at least 4")
    if int(dimension) < 2:
        raise ValueError("dimension must be at least 2")
    if int(queries_per_cluster) < 1:
        raise ValueError("queries_per_cluster must be positive")
    if not 1 <= int(anchors_per_harmed_cluster) <= int(docs_per_cluster):
        raise ValueError("anchors_per_harmed_cluster must be in [1, docs_per_cluster]")
    if not 0 <= int(seed) < 2**32:
        raise ValueError("seed must be in [0, 2**32)")

    # Independent streams are load-bearing for the sweep: changing anchor
    # count must not silently change the corpus, harmed clusters, or warps.
    corpus_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC0A5]))
    query_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x0E17]))
    harm_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xA11D]))
    clean, cluster_ids, centers, covariances = _gaussian_mixture(
        corpus_rng,
        n_clusters=int(n_clusters),
        docs_per_cluster=int(docs_per_cluster),
        dimension=int(dimension),
    )
    queries, query_clusters, positives, qrels = _make_queries(
        query_rng,
        clean,
        cluster_ids,
        centers,
        covariances,
        queries_per_cluster=int(queries_per_cluster),
    )

    harmed_count = max(1, int(np.floor(int(n_clusters) * sparsity + 0.5)))
    harmed_clusters = np.sort(
        harm_rng.choice(int(n_clusters), size=harmed_count, replace=False).astype(np.int64)
    )
    corrupted = clean.copy()
    parameters = []
    anchor_indices = []
    anchor_cluster_ids = []
    for cluster_id in harmed_clusters:
        indices = np.flatnonzero(cluster_ids == cluster_id)
        family_code = 1 if canonical_family == "quadratic" else 2
        warp_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0x7A2F, family_code, int(cluster_id)])
        )
        anchor_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0xA4C0, int(cluster_id)])
        )
        parameter = _make_warp(
            warp_rng,
            canonical_family,
            int(cluster_id),
            clean[indices],
            centers[cluster_id],
        )
        parameters.append(parameter)
        corrupted[indices] = apply_cluster_warp(clean[indices], parameter, gamma=gamma)
        chosen_anchors = np.sort(
            anchor_rng.permutation(indices)[: int(anchors_per_harmed_cluster)]
        )
        anchor_indices.extend(chosen_anchors.tolist())
        anchor_cluster_ids.extend([int(cluster_id)] * len(chosen_anchors))

    displacements = corrupted - clean
    return SparseLocalNonAffineData(
        clean_documents=clean,
        corrupted_documents=corrupted,
        clean_queries=queries,
        query_cluster_ids=query_clusters,
        query_doc_indices=positives,
        qrels=qrels,
        true_cluster_ids=cluster_ids,
        harmed_cluster_ids=harmed_clusters,
        anchor_indices=np.asarray(anchor_indices, dtype=np.int64),
        anchor_cluster_ids=np.asarray(anchor_cluster_ids, dtype=np.int64),
        clean_cluster_centers=centers,
        within_cluster_covariances=covariances,
        warp_parameters=tuple(parameters),
        displacements=displacements,
        family=canonical_family,
        gamma=gamma,
        requested_sparsity=sparsity,
        seed=int(seed),
    )


def generate_sparse_local_regime(**kwargs: object) -> SparseLocalNonAffineData:
    """Compatibility alias for :func:`generate_sparse_local_nonaffine`."""

    return generate_sparse_local_nonaffine(**kwargs)


__all__ = [
    "ClusterWarp",
    "SparseLocalNonAffineData",
    "apply_cluster_warp",
    "generate_sparse_local_nonaffine",
    "generate_sparse_local_regime",
]
