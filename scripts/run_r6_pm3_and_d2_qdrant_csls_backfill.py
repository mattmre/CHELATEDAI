r"""Stage R6 PM-3 fair raw-text N=100 and Demo D2-4096d Qdrant HNSW + CSLS backfill.

Part A scores cached 384-d to 4096-d BEIR pairs. It never re-encodes Qwen3-Embedding-8B.
Part B locks the existing N=60 graded rows and, with --run-live-pm3, scores N=40
LongMemEval-S items on one shared raw-message list for mem0ai and Chelated.

Windows OpenSSH note: this process does not invoke ssh. Callers that do must pass
stdin=subprocess.DEVNULL.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmark_utils import canonicalize_id, ndcg_at_k  # noqa: E402
from m1_baseline_ladder import (  # noqa: E402
    compute_fastfill_backfill_order,
    fit_closed_form_centered_ridge,
    fit_closed_form_svd_procrustes,
    select_qrels_independent_anchors,
    split_query_ids,
)
from scripts.night1_common import load_beir_dataset, paired_bootstrap_delta  # noqa: E402

REQUESTED_DOMAINS: Tuple[str, ...] = ("SciFact", "FiQA2018", "NFCorpus")
COMPANION_DOMAINS: Tuple[str, ...] = ("CoSQA_Code", "Legal_Contracts", "Legal_QuAD")
BACKFILL_FRACTIONS: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.50, 1.0)
K_OVERSAMPLE: Tuple[int, ...] = (10, 32, 64)
RECALL_K = 10
CSLS_K = 10
EMA_ALPHA = 0.05
QUERY_SPLIT_SALT = "m1-query-split-v1"
N60_CHELATED_ARM = "Hetero_3090toSpark_ZeroMu_Chelated_V2_Nemotron30B"
N60_CHELATED_MID_ARM = "Mid7B_Chelated_V2_RTX3090"
N60_MEM0_INFER_ARM = "Nemotron30B_Mem0_LiveInfer_FullN60"
N60_MEM0_RAW_ARM = "Arm4a_Nemotron30B_Mem0_RawQdrant"
N60_MEM0_WRITE_MS = 4582.44
LME_ECHO_PREFIXES: Tuple[str, ...] = (
    "Session recap note from earlier chat: user mentioned — ",
    "Memory summary log (auto-archived): ",
    "Background context carried forward: ",
    "Follow-up note recalling earlier detail: ",
    "Periodic assistant digest citing prior session: ",
    "Stored profile reminder from previous conversation: ",
)
ABSTAIN_MARKERS: Tuple[str, ...] = (
    "abstain",
    "not mentioned",
    "did not mention",
    "do not mention",
    "not enough",
    "cannot",
    "can't",
    "don't know",
    "do not know",
    "no information",
    "unknown",
    "not in the",
    "no record",
    "insufficient",
    "contested",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return sha256_bytes(text.encode("utf-8"))


def l2_normalize(rows: np.ndarray) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def cone_anisotropy(docs: np.ndarray) -> Dict[str, float]:
    """Empirical target cone ||mu_d||_2 and dispersion of centered document norms."""
    matrix = np.asarray(docs, dtype=np.float64)
    mu = matrix.mean(axis=0)
    mu_l2 = float(np.linalg.norm(mu))
    centered = np.linalg.norm(matrix - mu, axis=1)
    centered_mean = float(centered.mean())
    return {
        "n_docs": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "mu_l2": mu_l2,
        "centered_norm_mean": centered_mean,
        "centered_norm_std": float(centered.std()),
        "anisotropy_ratio_mu_over_mean_radius": mu_l2 / max(centered_mean, 1e-12),
    }


def fit_uncentered_stiefel(
    source: np.ndarray,
    target: np.ndarray,
    fit_idx: np.ndarray,
) -> np.ndarray:
    """Query-only uncentered Stiefel map (arXiv:2509.23471 shape): W = U V^T, no centering."""
    return np.asarray(fit_closed_form_svd_procrustes(source, target, fit_idx), dtype=np.float64)


def fit_query_only_affine(
    source: np.ndarray,
    target: np.ndarray,
    fit_idx: np.ndarray,
    *,
    regularization: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered-source affine onto raw uncentered targets: (x - mu_x) W + mu_y."""
    mu_x, mu_y, weight = fit_closed_form_centered_ridge(
        source,
        target,
        fit_idx,
        regularization=regularization,
    )
    return np.asarray(mu_x, dtype=np.float64), np.asarray(mu_y, dtype=np.float64), np.asarray(weight, dtype=np.float64)


def cosine_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    return l2_normalize(queries) @ l2_normalize(docs).T


def score_uncentered_procrustes(
    queries_src: np.ndarray,
    stiefel_w: np.ndarray,
    docs_raw: np.ndarray,
) -> np.ndarray:
    """Map queries by the uncentered Stiefel factor and cosine-score raw target docs."""
    mapped = np.asarray(queries_src, dtype=np.float64) @ np.asarray(stiefel_w, dtype=np.float64)
    return cosine_matrix(mapped, docs_raw)


def score_query_only_affine(
    queries_src: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    weight: np.ndarray,
    docs_raw: np.ndarray,
) -> np.ndarray:
    """Score (q - mu_x) W + mu_y by cosine against raw docs. No centered-norm division."""
    mapped = (np.asarray(queries_src, dtype=np.float64) - mu_x) @ weight + mu_y
    return cosine_matrix(mapped, docs_raw)


def build_zeromu_sidecar(docs_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4-byte float32 per-document centered norm plus the shared mu_d row."""
    docs = np.asarray(docs_raw, dtype=np.float32)
    mu_d = docs.mean(axis=0, keepdims=True).astype(np.float32)
    norms = np.maximum(np.linalg.norm(docs - mu_d, axis=1).astype(np.float32), np.float32(1e-12))
    if norms.dtype != np.float32 or int(norms.nbytes) != 4 * int(norms.shape[0]):
        raise RuntimeError("ZeroMu sidecar is not packed as one float32 per document")
    return mu_d, norms


def fit_centered_orthogonal_q(
    source: np.ndarray,
    target: np.ndarray,
    fit_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-scale centered Procrustes Q and anchor mu_x used to seed causal mu_q."""
    src = np.asarray(source, dtype=np.float64)[np.asarray(fit_idx)]
    tgt = np.asarray(target, dtype=np.float64)[np.asarray(fit_idx)]
    mu_x = src.mean(axis=0, keepdims=True)
    mu_y = tgt.mean(axis=0, keepdims=True)
    cross = (src - mu_x).T @ (tgt - mu_y)
    left, _sv, vt = np.linalg.svd(cross, full_matrices=False)
    return (left @ vt).astype(np.float64), mu_x.astype(np.float64)


def causal_q_hats(
    queries_src: np.ndarray,
    ortho_q: np.ndarray,
    mu_x: np.ndarray,
    *,
    ema_alpha: float = EMA_ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project queries in order. mu_q used for query t excludes query t."""
    queries = np.asarray(queries_src, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    transform = np.asarray(ortho_q, dtype=np.float32)
    mu_q = np.asarray(mu_x, dtype=np.float32).reshape(1, -1).copy()
    hats = np.empty((queries.shape[0], transform.shape[1]), dtype=np.float32)
    mu_before = np.empty((queries.shape[0], mu_q.shape[1]), dtype=np.float32)
    alpha = float(ema_alpha)
    for row in range(queries.shape[0]):
        q_row = queries[row : row + 1]
        mu_before[row] = mu_q[0]
        centered = (q_row - mu_q) @ transform
        hats[row] = l2_normalize(centered)[0].astype(np.float32)
        if alpha > 0.0:
            mu_q = (1.0 - alpha) * mu_q + alpha * q_row
    return hats, mu_before


def zeromu_scores_from_qhats(
    q_hats: np.ndarray,
    docs_raw: np.ndarray,
    mu_d: np.ndarray,
    doc_centered_norms: np.ndarray,
) -> np.ndarray:
    """(d_j · q_hat - mu_d · q_hat) / ||d_j - mu_d||_2 on unmodified raw docs."""
    docs = np.asarray(docs_raw, dtype=np.float32)
    hats = np.asarray(q_hats, dtype=np.float32)
    dots = hats @ docs.T
    mu_term = hats @ np.asarray(mu_d, dtype=np.float32).reshape(-1)
    norms = np.asarray(doc_centered_norms, dtype=np.float32).reshape(1, -1)
    return (dots - mu_term[:, None]) / norms


def topk_doc_ids(scores: np.ndarray, doc_ids: Sequence[str], k: int) -> List[str]:
    row = np.asarray(scores, dtype=np.float64)
    n_docs = int(row.shape[0])
    k_eff = min(int(k), n_docs)
    if k_eff <= 0:
        return []
    kth = float(np.partition(row, n_docs - k_eff)[n_docs - k_eff])
    candidates = np.flatnonzero(row >= kth)
    ordered = sorted(candidates.tolist(), key=lambda j: (-float(row[j]), doc_ids[j]))[:k_eff]
    return [doc_ids[j] for j in ordered]


def per_query_ndcg(
    scores: np.ndarray,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    k: int = 10,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row, qid in enumerate(query_ids):
        rel = qrels.get(qid, {})
        pos = {canonicalize_id(did): float(score) for did, score in rel.items() if float(score) > 0.0}
        if not pos:
            continue
        ranked = topk_doc_ids(scores[row], doc_ids, k)
        out[qid] = float(ndcg_at_k(ranked, pos, k=k, binary=False))
    return out


def mean_metric(values: Mapping[str, float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(list(values.values())))


def report_positive_doc_ids(
    qrels: Mapping[str, Mapping[str, float]],
    report_qids: Iterable[str],
) -> set:
    banned = set()
    for qid in report_qids:
        for did, score in qrels.get(qid, {}).items():
            if float(score) > 0.0:
                banned.add(canonicalize_id(did))
    return banned


def disjoint_fit_indices(
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    report_qids: Sequence[str],
) -> np.ndarray:
    anchors = select_qrels_independent_anchors(doc_ids, anchor_frac=0.50, seed=42)
    banned = report_positive_doc_ids(qrels, report_qids)
    kept = [int(idx) for idx in anchors if canonicalize_id(doc_ids[int(idx)]) not in banned]
    if len(kept) < 2:
        raise RuntimeError("Disjoint non-eval fit set has fewer than 2 documents")
    return np.asarray(sorted(kept), dtype=np.int64)


def ordered_positions(query_ids: Sequence[str], wanted: Sequence[str]) -> List[int]:
    index = {qid: pos for pos, qid in enumerate(query_ids)}
    return [index[qid] for qid in sorted(wanted) if qid in index]


def rescore_shortlist(
    shortlist: Sequence[int],
    q_hat: np.ndarray,
    docs_raw: np.ndarray,
    mu_d: np.ndarray,
    doc_centered_norms: np.ndarray,
) -> np.ndarray:
    """Rescore an ANN shortlist with the 4-byte ZeroMu sidecar. Docs stay unmodified."""
    idx = np.asarray(list(shortlist), dtype=np.int64)
    if idx.size == 0:
        return np.zeros(0, dtype=np.float32)
    hat = np.asarray(q_hat, dtype=np.float32).reshape(-1)
    docs = np.asarray(docs_raw, dtype=np.float32)[idx]
    dots = docs @ hat
    mu_term = float(np.asarray(mu_d, dtype=np.float32).reshape(-1) @ hat)
    norms = np.asarray(doc_centered_norms, dtype=np.float32)[idx]
    return (dots - np.float32(mu_term)) / norms


def recall_of_exact_topk(exact_ids: Sequence[str], approx_ids: Sequence[str]) -> float:
    if not exact_ids:
        return 0.0
    return float(len(set(exact_ids) & set(approx_ids))) / float(len(exact_ids))


def numpy_dot_shortlist(q_hat: np.ndarray, docs_raw: np.ndarray, k_os: int) -> List[int]:
    dots = np.asarray(docs_raw, dtype=np.float32) @ np.asarray(q_hat, dtype=np.float32).reshape(-1)
    k_eff = min(int(k_os), int(dots.shape[0]))
    if k_eff >= dots.shape[0]:
        return list(range(int(dots.shape[0])))
    chosen = np.argpartition(-dots, k_eff - 1)[:k_eff]
    return [int(i) for i in chosen]


def qdrant_dot_shortlist(
    client: Any,
    collection: str,
    q_hat: np.ndarray,
    k_os: int,
) -> Tuple[List[int], List[float]]:
    """HNSW search under Distance.DOT. efSearch equals the oversample depth."""
    vector = np.asarray(q_hat, dtype=np.float32).reshape(-1).tolist()
    limit = int(k_os)
    try:
        from qdrant_client.models import SearchParams

        params = SearchParams(hnsw_ef=limit, exact=False)
    except Exception:
        params = None
    if hasattr(client, "query_points"):
        kwargs: Dict[str, Any] = {"collection_name": collection, "query": vector, "limit": limit}
        if params is not None:
            kwargs["search_params"] = params
        result = client.query_points(**kwargs)
        points = list(result.points)
        return [int(point.id) for point in points], [float(point.score) for point in points]
    hits = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=limit,
        search_params=params,
    )
    return [int(hit.id) for hit in hits], [float(hit.score) for hit in hits]


def index_raw_vectors_qdrant(
    docs_raw: np.ndarray,
    *,
    collection: str = "d2_raw_4096_hnsw",
    qdrant_url: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Index unmodified raw vectors. A server URL builds real HNSW. `:memory:` is exact search."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, HnswConfigDiff, OptimizersConfigDiff, PointStruct, VectorParams

    docs = np.asarray(docs_raw, dtype=np.float32)
    if qdrant_url:
        client = QdrantClient(url=qdrant_url, timeout=180)
        engine = "qdrant-server-hnsw"
    else:
        client = QdrantClient(":memory:")
        engine = "qdrant-local-exact"
    exists = client.collection_exists(collection) if hasattr(client, "collection_exists") else False
    if exists:
        client.delete_collection(collection_name=collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=int(docs.shape[1]), distance=Distance.DOT),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=128, full_scan_threshold=10),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=1),
    )
    batch = 64
    for start in range(0, docs.shape[0], batch):
        points = [
            PointStruct(id=int(idx), vector=docs[idx].tolist())
            for idx in range(start, min(docs.shape[0], start + batch))
        ]
        client.upsert(collection_name=collection, points=points)
    indexed = 0
    status = "unknown"
    if engine == "qdrant-server-hnsw":
        deadline = time.time() + 600.0
        while time.time() < deadline:
            info = client.get_collection(collection_name=collection)
            indexed = int(getattr(info, "indexed_vectors_count", 0) or 0)
            status = str(getattr(info, "status", "unknown"))
            if indexed >= int(docs.shape[0]) and "green" in status.lower():
                break
            time.sleep(1.0)
        else:
            raise RuntimeError(f"Qdrant HNSW index did not finish: indexed={indexed} status={status}")
    meta = {
        "search_engine": engine,
        "hnsw_effective": engine == "qdrant-server-hnsw",
        "indexed_vectors_count": indexed if engine == "qdrant-server-hnsw" else None,
        "collection_status": status if engine == "qdrant-server-hnsw" else "local-exact",
        "qdrant_url": qdrant_url or ":memory:",
        "indexing_threshold": 1,
    }
    return client, meta


def vector_identity_max_abs(client: Any, collection: str, docs_raw: np.ndarray) -> float:
    docs = np.asarray(docs_raw, dtype=np.float32)
    if hasattr(client, "retrieve"):
        records = client.retrieve(collection_name=collection, ids=[0], with_vectors=True)
        if not records:
            return float("inf")
        stored = np.asarray(records[0].vector, dtype=np.float32)
        return float(np.max(np.abs(stored - docs[0])))
    return float("nan")


def csls_local(query_scores: np.ndarray, ref_scores: np.ndarray, csls_k: int) -> np.ndarray:
    """2 * s - r_d - r_q with doc hubness from the reference queries and query hubness in-shard."""
    scores = np.asarray(query_scores, dtype=np.float64)
    ref = np.asarray(ref_scores, dtype=np.float64)
    if scores.size == 0 or scores.shape[1] == 0:
        return scores.copy()
    doc_k = max(1, min(int(csls_k), int(ref.shape[0])))
    if doc_k < ref.shape[0]:
        top_docs = np.partition(ref, ref.shape[0] - doc_k, axis=0)[-doc_k:, :]
    else:
        top_docs = ref
    r_d = top_docs.mean(axis=0)
    query_k = max(1, min(int(csls_k), int(scores.shape[1])))
    if query_k < scores.shape[1]:
        top_q = np.partition(scores, scores.shape[1] - query_k, axis=1)[:, -query_k:]
    else:
        top_q = scores
    r_q = top_q.mean(axis=1)
    return 2.0 * scores - r_d.reshape(1, -1) - r_q.reshape(-1, 1)


def mixed_shard_scores(
    bridged_scores: np.ndarray,
    native_scores: np.ndarray,
    backfill_mask: np.ndarray,
    *,
    bridged_ref: Optional[np.ndarray] = None,
    native_ref: Optional[np.ndarray] = None,
    calibrate: bool = False,
    csls_k: int = CSLS_K,
) -> np.ndarray:
    """Uncalibrated merge writes each shard's raw score. CSLS calibrates inside each shard."""
    bridged = np.asarray(bridged_scores, dtype=np.float64)
    native = np.asarray(native_scores, dtype=np.float64)
    mask = np.asarray(backfill_mask, dtype=bool)
    if bridged.shape != native.shape:
        raise ValueError("bridged and native score matrices must share a shape")
    if mask.shape[0] != bridged.shape[1]:
        raise ValueError("backfill mask must cover every document")
    if not calibrate:
        out = bridged.copy()
        if mask.any():
            out[:, mask] = native[:, mask]
        return out
    if bridged_ref is None or native_ref is None:
        raise ValueError("CSLS mixed-shard merge requires reference scores for both shards")
    out = np.empty_like(bridged)
    if (~mask).any():
        out[:, ~mask] = csls_local(bridged[:, ~mask], np.asarray(bridged_ref)[:, ~mask], csls_k)
    if mask.any():
        out[:, mask] = csls_local(native[:, mask], np.asarray(native_ref)[:, mask], csls_k)
    return out


def zscore_mixed_shard_scores(
    bridged_scores: np.ndarray,
    native_scores: np.ndarray,
    backfill_mask: np.ndarray,
    bridged_ref: np.ndarray,
    native_ref: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """DEV-only per-shard scalar z-score calibration `z = (s - mu_dev) / max(sigma_dev, eps)`."""
    bridged = np.asarray(bridged_scores, dtype=np.float64)
    native = np.asarray(native_scores, dtype=np.float64)
    mask = np.asarray(backfill_mask, dtype=bool)
    out = np.empty_like(bridged)
    if (~mask).any():
        b_dev = np.asarray(bridged_ref, dtype=np.float64)[:, ~mask]
        mu_b = float(b_dev.mean()) if b_dev.size else 0.0
        sd_b = max(float(b_dev.std()), eps) if b_dev.size else 1.0
        out[:, ~mask] = (bridged[:, ~mask] - mu_b) / sd_b
    if mask.any():
        n_dev = np.asarray(native_ref, dtype=np.float64)[:, mask]
        mu_n = float(n_dev.mean()) if n_dev.size else 0.0
        sd_n = max(float(n_dev.std()), eps) if n_dev.size else 1.0
        out[:, mask] = (native[:, mask] - mu_n) / sd_n
    return out


def rrf_mixed_shard_scores(
    bridged_scores: np.ndarray,
    native_scores: np.ndarray,
    backfill_mask: np.ndarray,
    *,
    rrf_k: int = 60,
) -> np.ndarray:
    """A-priori untuned Reciprocal Rank Fusion (`k=60`) over within-shard rankings."""
    bridged = np.asarray(bridged_scores, dtype=np.float64)
    native = np.asarray(native_scores, dtype=np.float64)
    mask = np.asarray(backfill_mask, dtype=bool)
    out = np.zeros_like(bridged)
    for shard_idx in (np.flatnonzero(~mask), np.flatnonzero(mask)):
        if shard_idx.size == 0:
            continue
        sub = native[:, shard_idx] if (shard_idx.size and mask[int(shard_idx[0])]) else bridged[:, shard_idx]
        for q_row in range(sub.shape[0]):
            order = np.argsort(-sub[q_row], kind="mergesort")
            ranks = np.empty(order.shape[0], dtype=np.float64)
            ranks[order] = np.arange(1, order.shape[0] + 1, dtype=np.float64)
            out[q_row, shard_idx] = 1.0 / (float(rrf_k) + ranks)
    return out


def verify_d384_cache_row_identity(
    doc_ids: Sequence[str],
    doc_texts: Sequence[str],
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    docs_low: np.ndarray,
    queries_low: np.ndarray,
    *,
    verify_encoder: bool = True,
    eps: float = 1e-5,
) -> Dict[str, Any]:
    """Bind cached arrays to sorted BEIR IDs + SHA-256 digests and check float32 row cosines >= 1 - eps."""
    docs_finite = bool(np.isfinite(docs_low).all() and np.all(np.linalg.norm(docs_low, axis=1) > 0.5))
    queries_finite = bool(np.isfinite(queries_low).all() and np.all(np.linalg.norm(queries_low, axis=1) > 0.5))
    info: Dict[str, Any] = {
        "sorted_doc_ids_sha256": sha256_bytes("\n".join(doc_ids).encode("utf-8")),
        "sorted_query_ids_sha256": sha256_bytes("\n".join(query_ids).encode("utf-8")),
        "sorted_doc_texts_sha256": sha256_bytes("\n".join(doc_texts).encode("utf-8")),
        "sorted_query_texts_sha256": sha256_bytes("\n".join(query_texts).encode("utf-8")),
        "all_cached_rows_finite_nonzero": bool(docs_finite and queries_finite),
        "cosine_threshold": 1.0 - float(eps),
        "encoder_row_check_executed": False,
    }
    if not verify_encoder:
        return info
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    enc.max_seq_length = 384
    regen_docs = np.asarray(
        enc.encode(list(doc_texts), batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )
    regen_queries = np.asarray(
        enc.encode(list(query_texts), batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )
    docs_low_u = l2_normalize(docs_low)
    regen_docs_u = l2_normalize(regen_docs)
    doc_cos = np.sum(docs_low_u * regen_docs_u, axis=1)
    q_cos = np.sum(l2_normalize(queries_low) * l2_normalize(regen_queries), axis=1)
    nn_idx = np.argmax(regen_docs_u @ docs_low_u.T, axis=1)
    nn_exact_rate = float(np.mean(nn_idx == np.arange(docs_low.shape[0])))
    min_doc_cos = float(np.min(doc_cos))
    min_q_cos = float(np.min(q_cos))
    passed = bool(docs_finite and queries_finite and min_doc_cos >= (1.0 - eps) and min_q_cos >= (1.0 - eps))
    if not passed:
        raise RuntimeError(
            f"D384 cache row identity check FAILED: min_doc_cos={min_doc_cos:.8f}, min_query_cos={min_q_cos:.8f}"
        )
    info.update({
        "encoder_row_check_executed": True,
        "encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "encoder_max_seq_length": 384,
        "min_doc_row_cosine": min_doc_cos,
        "mean_doc_row_cosine": float(np.mean(doc_cos)),
        "min_query_row_cosine": min_q_cos,
        "mean_query_row_cosine": float(np.mean(q_cos)),
        "nn_top1_exact_row_match_rate": nn_exact_rate,
        "all_rows_cosine_ge_1_minus_1e5": passed,
    })
    return info


def backfill_mask_from_order(order: np.ndarray, n_docs: int, fraction: float) -> np.ndarray:
    count = int(round(float(fraction) * n_docs))
    count = max(0, min(n_docs, count))
    mask = np.zeros(n_docs, dtype=bool)
    if count:
        mask[np.asarray(order[:count], dtype=np.int64)] = True
    return mask


def cache_status(cache_dir: Path, domain: str) -> Dict[str, Any]:
    path_384 = cache_dir / f"ood_{domain}_D384.npz"
    path_4096 = cache_dir / f"ood_{domain}_D4096.npz"
    status: Dict[str, Any] = {
        "domain": domain,
        "path_384": str(path_384),
        "path_4096": str(path_4096),
        "exists_384": path_384.is_file(),
        "exists_4096": path_4096.is_file(),
    }
    if path_384.is_file():
        status["sha256_384"] = sha256_file(path_384)
        with np.load(path_384) as blob:
            status["shape_docs_384"] = list(blob["docs"].shape)
            status["shape_queries_384"] = list(blob["queries"].shape)
    if path_4096.is_file():
        status["sha256_4096"] = sha256_file(path_4096)
        with np.load(path_4096) as blob:
            status["shape_docs_4096"] = list(blob["docs"].shape)
            status["shape_queries_4096"] = list(blob["queries"].shape)
    if not path_4096.is_file():
        status["evaluation"] = "MISSING_4096"
    return status


def load_cached_pair(cache_dir: Path, domain: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(cache_dir / f"ood_{domain}_D384.npz") as low:
        docs_low = np.asarray(low["docs"], dtype=np.float32)
        queries_low = np.asarray(low["queries"], dtype=np.float32)
    with np.load(cache_dir / f"ood_{domain}_D4096.npz") as high:
        docs_high = np.asarray(high["docs"], dtype=np.float32)
        queries_high = np.asarray(high["queries"], dtype=np.float32)
    return docs_low, queries_low, docs_high, queries_high


def evaluate_cached_domain(
    domain: str,
    doc_ids: Sequence[str],
    query_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    docs_low: np.ndarray,
    queries_low: np.ndarray,
    docs_high: np.ndarray,
    queries_high: np.ndarray,
    *,
    run_qdrant: bool,
    qdrant_url: Optional[str] = None,
) -> Dict[str, Any]:
    if docs_low.shape[0] != len(doc_ids) or docs_high.shape[0] != len(doc_ids):
        raise RuntimeError(
            f"{domain} doc-row mismatch: ids={len(doc_ids)} low={docs_low.shape[0]} high={docs_high.shape[0]}"
        )
    if queries_low.shape[0] != len(query_ids) or queries_high.shape[0] != len(query_ids):
        raise RuntimeError(
            f"{domain} query-row mismatch: ids={len(query_ids)} low={queries_low.shape[0]} high={queries_high.shape[0]}"
        )
    splits = split_query_ids(query_ids, salt=QUERY_SPLIT_SALT)
    report_ids = [qid for qid in splits["REPORT"] if qrels.get(qid)]
    dev_ids = [qid for qid in splits["DEVELOPMENT"] if qid in set(query_ids)]
    fit_idx = disjoint_fit_indices(doc_ids, qrels, report_ids)
    report_pos = ordered_positions(query_ids, report_ids)
    dev_pos = ordered_positions(query_ids, dev_ids)
    report_qids = [query_ids[pos] for pos in report_pos]
    cone = cone_anisotropy(docs_high)
    stiefel = fit_uncentered_stiefel(docs_low, docs_high, fit_idx)
    mu_x_aff, mu_y_aff, affine_w = fit_query_only_affine(docs_low, docs_high, fit_idx)
    ortho_q, mu_x = fit_centered_orthogonal_q(docs_low, docs_high, fit_idx)
    mu_d, norms = build_zeromu_sidecar(docs_high)
    stiefel_orthogonality = float(np.max(np.abs(stiefel @ stiefel.T - np.eye(stiefel.shape[0]))))

    procrustes_scores = score_uncentered_procrustes(queries_low[report_pos], stiefel, docs_high)
    affine_scores = score_query_only_affine(queries_low[report_pos], mu_x_aff, mu_y_aff, affine_w, docs_high)
    report_hats, _mu_used = causal_q_hats(queries_low[report_pos], ortho_q, mu_x)
    zeromu_scores = zeromu_scores_from_qhats(report_hats, docs_high, mu_d, norms)
    dev_hats, _dev_mu = causal_q_hats(queries_low[dev_pos], ortho_q, mu_x)
    dev_zeromu = zeromu_scores_from_qhats(dev_hats, docs_high, mu_d, norms)
    dev_native = cosine_matrix(queries_high[dev_pos], docs_high)
    report_native = cosine_matrix(queries_high[report_pos], docs_high)

    ndcg = {
        "DriftAdapter_Uncentered_Procrustes": per_query_ndcg(procrustes_scores, report_qids, doc_ids, qrels),
        "DriftAdapter_QueryOnly_Affine": per_query_ndcg(affine_scores, report_qids, doc_ids, qrels),
        "ZeroMu_4ByteSidecar_CausalEMA": per_query_ndcg(zeromu_scores, report_qids, doc_ids, qrels),
    }
    comparisons = {}
    base = ndcg["ZeroMu_4ByteSidecar_CausalEMA"]
    for name in ("DriftAdapter_Uncentered_Procrustes", "DriftAdapter_QueryOnly_Affine"):
        comparisons[f"ZeroMu_minus_{name}"] = paired_bootstrap_delta(
            base,
            ndcg[name],
            n_bootstrap=3000,
            seed=42,
            expected_ids=sorted(base),
            strict_ids=True,
        )
    beats = {
        name: bool(stats["lcb_95"] > 0.0)
        for name, stats in comparisons.items()
    }

    # Also verify pure qrels-blind hash-anchor fit (`select_qrels_independent_anchors`, zero REPORT qrel inspection)
    blind_fit_idx = np.asarray(sorted(int(i) for i in select_qrels_independent_anchors(doc_ids, anchor_frac=0.50, seed=42)), dtype=np.int64)
    ortho_q_blind, mu_x_blind = fit_centered_orthogonal_q(docs_low, docs_high, blind_fit_idx)
    blind_hats, _ = causal_q_hats(queries_low[report_pos], ortho_q_blind, mu_x_blind)
    blind_zeromu_scores = zeromu_scores_from_qhats(blind_hats, docs_high, mu_d, norms)
    blind_zeromu_ndcg = per_query_ndcg(blind_zeromu_scores, report_qids, doc_ids, qrels)

    doc_id_list = [canonicalize_id(did) for did in doc_ids]
    exact_top = [topk_doc_ids(zeromu_scores[row], doc_id_list, RECALL_K) for row in range(len(report_qids))]
    qdrant_block: Dict[str, Any] = {"executed": False}
    if run_qdrant:
        t0 = time.perf_counter()
        client, qdrant_meta = index_raw_vectors_qdrant(docs_high, qdrant_url=qdrant_url)
        identity = vector_identity_max_abs(client, "d2_raw_4096_hnsw", docs_high)
        by_k: Dict[str, Any] = {}
        for k_os in K_OVERSAMPLE:
            recalls = []
            exact_dot_recalls = []
            dot_abs = []
            hnsw_ndcg_map: Dict[str, float] = {}
            exact_dot_ndcg_map: Dict[str, float] = {}
            for row, hat in enumerate(report_hats):
                qid_r = report_qids[row]
                rel_r = qrels.get(qid_r, {})
                pos_r = {canonicalize_id(did): float(sc) for did, sc in rel_r.items() if float(sc) > 0.0}

                shortlist, dot_scores = qdrant_dot_shortlist(client, "d2_raw_4096_hnsw", hat, k_os)
                rescored = rescore_shortlist(shortlist, hat, docs_high, mu_d, norms)
                order_h = np.argsort(-rescored, kind="mergesort")
                approx_ids = [doc_id_list[shortlist[int(j)]] for j in order_h[:RECALL_K]]
                recalls.append(recall_of_exact_topk(exact_top[row], approx_ids))

                ex_shortlist = numpy_dot_shortlist(hat, docs_high, k_os)
                ex_rescored = rescore_shortlist(ex_shortlist, hat, docs_high, mu_d, norms)
                order_ex = np.argsort(-ex_rescored, kind="mergesort")
                ex_ids = [doc_id_list[ex_shortlist[int(j)]] for j in order_ex[:RECALL_K]]
                exact_dot_recalls.append(recall_of_exact_topk(exact_top[row], ex_ids))

                if pos_r:
                    hnsw_ndcg_map[qid_r] = float(ndcg_at_k(approx_ids, pos_r, k=RECALL_K, binary=False))
                    exact_dot_ndcg_map[qid_r] = float(ndcg_at_k(ex_ids, pos_r, k=RECALL_K, binary=False))
                if dot_scores:
                    numpy_dots = np.asarray(docs_high, dtype=np.float32)[shortlist] @ hat.astype(np.float32)
                    dot_abs.append(float(np.max(np.abs(numpy_dots - np.asarray(dot_scores, dtype=np.float32)))))
            by_k[str(k_os)] = {
                "mean_recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
                "min_recall_at_10": float(np.min(recalls)) if recalls else 0.0,
                "exact_dot_shortlist_mean_recall_at_10": float(np.mean(exact_dot_recalls)) if exact_dot_recalls else 0.0,
                "hnsw_rescore_mean_ndcg_at_10": mean_metric(hnsw_ndcg_map),
                "exact_dot_shortlist_mean_ndcg_at_10": mean_metric(exact_dot_ndcg_map),
                "max_abs_dot_vs_numpy": float(np.max(dot_abs)) if dot_abs else None,
                "meets_99_5": bool(recalls) and float(np.mean(recalls)) >= 0.995,
            }
        client.close()
        qdrant_block = {
            "executed": True,
            "distance": "DOT",
            "hnsw_m": 16,
            "hnsw_ef_construct": 128,
            "ef_search_equals_k_os": True,
            "vector_modification": "none",
            "retrieve_point0_max_abs_delta": identity,
            "index_seconds": round(time.perf_counter() - t0, 3),
            "by_k_os": by_k,
            "sidecar_bytes_per_doc": 4,
            "shared_centroid_bytes": int(docs_high.shape[1] * 4),
            **qdrant_meta,
        }

    order = compute_fastfill_backfill_order(queries_low[dev_pos], docs_low, top_m=50)
    curve = []
    for fraction in BACKFILL_FRACTIONS:
        mask = backfill_mask_from_order(order, docs_high.shape[0], fraction)
        raw_scores = mixed_shard_scores(zeromu_scores, report_native, mask, calibrate=False)
        zscore_scores = zscore_mixed_shard_scores(zeromu_scores, report_native, mask, dev_zeromu, dev_native)
        rrf_scores = rrf_mixed_shard_scores(zeromu_scores, report_native, mask, rrf_k=60)
        csls_scores = mixed_shard_scores(
            zeromu_scores,
            report_native,
            mask,
            bridged_ref=dev_zeromu,
            native_ref=dev_native,
            calibrate=True,
        )
        raw_ndcg = per_query_ndcg(raw_scores, report_qids, doc_ids, qrels)
        zscore_ndcg = per_query_ndcg(zscore_scores, report_qids, doc_ids, qrels)
        rrf_ndcg = per_query_ndcg(rrf_scores, report_qids, doc_ids, qrels)
        csls_ndcg = per_query_ndcg(csls_scores, report_qids, doc_ids, qrels)
        curve.append(
            {
                "fraction": fraction,
                "n_backfilled": int(mask.sum()),
                "uncalibrated_mean_ndcg_at_10": mean_metric(raw_ndcg),
                "dev_zscore_mean_ndcg_at_10": mean_metric(zscore_ndcg),
                "rrf_k60_mean_ndcg_at_10": mean_metric(rrf_ndcg),
                "csls_mean_ndcg_at_10": mean_metric(csls_ndcg),
                "csls_minus_uncalibrated": paired_bootstrap_delta(
                    csls_ndcg, raw_ndcg, n_bootstrap=3000, seed=42, expected_ids=sorted(raw_ndcg), strict_ids=True
                ),
                "csls_minus_dev_zscore": paired_bootstrap_delta(
                    csls_ndcg, zscore_ndcg, n_bootstrap=3000, seed=43, expected_ids=sorted(raw_ndcg), strict_ids=True
                ),
                "csls_minus_rrf_k60": paired_bootstrap_delta(
                    csls_ndcg, rrf_ndcg, n_bootstrap=3000, seed=44, expected_ids=sorted(raw_ndcg), strict_ids=True
                ),
            }
        )

    return {
        "domain": domain,
        "row_order_contract": "load_beir_dataset sorted doc ids and sorted positive-qrel query ids",
        "n_docs": int(docs_high.shape[0]),
        "n_queries": int(queries_high.shape[0]),
        "n_report_queries": len(report_qids),
        "n_fit_docs": int(fit_idx.size),
        "fit_excludes_report_positive_docs": True,
        "qrels_blind_hash_anchor_fit_mean_ndcg_at_10": mean_metric(blind_zeromu_ndcg),
        "qrels_blind_n_fit_docs": int(blind_fit_idx.size),
        "cone": cone,
        "stiefel_max_abs_w_wt_minus_i": stiefel_orthogonality,
        "sidecar_dtype": "float32",
        "sidecar_bytes_per_doc": int(norms.nbytes // norms.shape[0]),
        "shared_centroid_bytes": int(docs_high.shape[1] * 4),
        "mean_ndcg_at_10": {name: mean_metric(table) for name, table in ndcg.items()},
        "paired_bootstrap_zeromu_minus_drift": comparisons,
        "zeromu_lcb95_beats_both_drift_adapters": all(beats.values()),
        "qdrant_hnsw": qdrant_block,
        "backfill_curve": curve,
        "per_query_ndcg": ndcg,
    }


def run_part_a(
    cache_dir: Path,
    out_dir: Path,
    *,
    run_qdrant: bool,
    qdrant_url: Optional[str] = None,
) -> Dict[str, Any]:
    domains: List[Dict[str, Any]] = []
    for domain in REQUESTED_DOMAINS:
        status = cache_status(cache_dir, domain)
        if status.get("evaluation") == "MISSING_4096" or not status.get("exists_384"):
            status["evaluation"] = "MISSING_4096" if not status.get("exists_4096") else "MISSING_384"
            status["scored"] = False
            domains.append(status)
            continue
        doc_ids, doc_texts, query_ids, query_texts, qrels, qrels_sha = load_beir_dataset(domain)
        docs_low, queries_low, docs_high, queries_high = load_cached_pair(cache_dir, domain)
        try:
            row_id_proof = verify_d384_cache_row_identity(
                doc_ids, doc_texts, query_ids, query_texts, docs_low, queries_low, verify_encoder=True
            )
            scored = evaluate_cached_domain(
                domain,
                doc_ids,
                query_ids,
                qrels,
                docs_low,
                queries_low,
                docs_high,
                queries_high,
                run_qdrant=run_qdrant,
                qdrant_url=qdrant_url,
            )
        except RuntimeError as exc:
            status["scored"] = False
            status["evaluation"] = "ROW_ALIGNMENT_FAILED"
            status["error"] = str(exc)
            domains.append(status)
            continue
        scored["row_identity_verification"] = row_id_proof
        scored["qrels_sha256"] = qrels_sha
        scored["scored"] = True
        scored["cache"] = status
        domains.append(scored)
    companions = []
    for domain in COMPANION_DOMAINS:
        info = cache_status(cache_dir, domain)
        info["scored"] = False
        info["evaluation"] = "CACHE_PRESENT_ROW_IDS_NOT_STORED_UNSCORED" if info.get("exists_4096") else "MISSING_4096"
        companions.append(info)
    manifest = {
        "item": "U1",
        "demo": "D2-4096d",
        "encoder_4096": "Qwen3-Embedding-8B",
        "reencoded_8b": False,
        "cache_dir": str(cache_dir),
        "requested_domains": list(REQUESTED_DOMAINS),
        "domains": domains,
        "companion_caches_not_substituted": companions,
        "formula": {
            "DriftAdapter_Uncentered_Procrustes": "cosine(q @ W, d_raw), W from uncentered SVD, W @ W.T = I",
            "DriftAdapter_QueryOnly_Affine": "cosine((q - mu_x) @ W + mu_y, d_raw); no division by ||d_j - mu_d||",
            "ZeroMu_4ByteSidecar_CausalEMA": "(d_j · q_hat - mu_d · q_hat) / ||d_j - mu_d||_2; mu_q updated after the query",
        },
    }
    write_json(out_dir / "d2_4096d_qdrant_manifest.json", manifest)
    rows_path = out_dir / "d2_4096d_per_query_ndcg.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "query_id", "arm", "ndcg_at_10"])
        writer.writeheader()
        for domain_block in domains:
            if not domain_block.get("per_query_ndcg"):
                continue
            for arm, table in domain_block["per_query_ndcg"].items():
                for qid, score in sorted(table.items()):
                    writer.writerow(
                        {
                            "domain": domain_block["domain"],
                            "query_id": qid,
                            "arm": arm,
                            "ndcg_at_10": f"{score:.8f}",
                        }
                    )
    return manifest


def _read_arm_rows(path: Path, arm: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if record.get("arm") != arm:
                continue
            rows[record["case_id"]] = {
                "gold": record.get("gold", ""),
                "pred": record.get("pred", ""),
                "is_correct": int(record["is_correct"]),
                "prompt_tokens": float(record.get("prompt_tokens") or 0.0),
                "latency_ms": float(record.get("latency_ms") or 0.0),
                "domain": record.get("domain", ""),
                "regime": record.get("regime", ""),
            }
    return rows


def _accuracy_block(rows: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": 0.0, "mean_prompt_tokens": 0.0, "mean_latency_ms": 0.0}
    correct = [int(row["is_correct"]) for row in rows.values()]
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)),
        "n_correct": int(sum(correct)),
        "mean_prompt_tokens": float(np.mean([row["prompt_tokens"] for row in rows.values()])),
        "mean_latency_ms": float(np.mean([row["latency_ms"] for row in rows.values()])),
    }


def lock_n60(rows_csv: Path, fleet_csv: Path, mem0_json: Path, claims_json: Path) -> Dict[str, Any]:
    chelated = _read_arm_rows(rows_csv, N60_CHELATED_ARM)
    mid = _read_arm_rows(rows_csv, N60_CHELATED_MID_ARM)
    mem0_true = _read_arm_rows(rows_csv, N60_MEM0_INFER_ARM)
    mem0_false = _read_arm_rows(fleet_csv, N60_MEM0_RAW_ARM)
    shared = sorted(set(chelated) & set(mem0_true) & set(mem0_false) & set(mid))
    if len(shared) != 60:
        raise RuntimeError(f"N=60 lock expected 60 shared case ids, found {len(shared)}")
    scores = {
        "chelated_hetero": {cid: float(chelated[cid]["is_correct"]) for cid in shared},
        "chelated_mid7b": {cid: float(mid[cid]["is_correct"]) for cid in shared},
        "mem0_infer_true": {cid: float(mem0_true[cid]["is_correct"]) for cid in shared},
        "mem0_infer_false": {cid: float(mem0_false[cid]["is_correct"]) for cid in shared},
    }
    return {
        "n": 60,
        "case_ids": shared,
        "source_rows_csv": str(rows_csv),
        "source_rows_sha256": sha256_file(rows_csv),
        "source_fleet_csv": str(fleet_csv),
        "source_fleet_sha256": sha256_file(fleet_csv),
        "mem0_json": str(mem0_json),
        "mem0_json_sha256": sha256_file(mem0_json),
        "claims_json": str(claims_json),
        "claims_json_sha256": sha256_file(claims_json),
        "arms": {
            "chelated_hetero_nemotron": _accuracy_block({cid: chelated[cid] for cid in shared}),
            "chelated_mid7b_confounded_reader": _accuracy_block({cid: mid[cid] for cid in shared}),
            "mem0_infer_true": _accuracy_block({cid: mem0_true[cid] for cid in shared}),
            "mem0_infer_false": _accuracy_block({cid: mem0_false[cid] for cid in shared}),
        },
        "mem0_infer_true_write_ms_per_case_reused": N60_MEM0_WRITE_MS,
        "write_latency_note": (
            "N=60 mem0 write mean is the previously measured "
            "mem0_live_infer_full60_mean_write_ms_per_case. This run did not re-call that LLM."
        ),
        "paired_bootstrap": {
            "chelated_hetero_minus_mem0_infer_true": paired_bootstrap_delta(
                scores["chelated_hetero"], scores["mem0_infer_true"], expected_ids=shared, strict_ids=True
            ),
            "chelated_hetero_minus_mem0_infer_false": paired_bootstrap_delta(
                scores["chelated_hetero"], scores["mem0_infer_false"], expected_ids=shared, strict_ids=True
            ),
            "chelated_mid7b_minus_mem0_infer_true": paired_bootstrap_delta(
                scores["chelated_mid7b"], scores["mem0_infer_true"], expected_ids=shared, strict_ids=True
            ),
        },
        "scores": scores,
    }


def _parse_lme_date(date_str: str) -> float:
    cleaned = re.sub(r"\([^)]*\)", "", date_str or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.datetime.strptime(cleaned, fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def raw_conversational_messages(item: Mapping[str, Any], *, inject_echoes: bool) -> List[Dict[str, Any]]:
    """Full raw user/assistant turns, plus 6 paraphrase echoes on knowledge-update items."""
    messages: List[Dict[str, Any]] = []
    answer_sids = set(item.get("answer_session_ids") or [])
    answer_user_turns: List[Tuple[float, str]] = []
    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]
    sids = item["haystack_session_ids"]
    for sid, date_str, turns in zip(sids, dates, sessions):
        base_ts = _parse_lme_date(str(date_str))
        for offset, turn in enumerate(turns):
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            role = str(turn.get("role") or "user")
            ts = base_ts + 0.001 * offset
            messages.append({"role": role, "content": content, "ts": ts, "session_id": sid, "echo": False})
            if role == "user" and sid in answer_sids and turn.get("has_answer"):
                answer_user_turns.append((ts, content[:600]))
    if inject_echoes and item.get("question_type") == "knowledge-update" and len(answer_user_turns) >= 2:
        answer_user_turns.sort(key=lambda pair: pair[0])
        old_ts, old_txt = answer_user_turns[0]
        new_ts = answer_user_turns[-1][0]
        span = max(new_ts - old_ts, 3600.0 * 6)
        for echo_i, prefix in enumerate(LME_ECHO_PREFIXES):
            frac = (echo_i + 1) / float(len(LME_ECHO_PREFIXES) + 1)
            echo_ts = old_ts + frac * min(span * 0.85, max(new_ts - old_ts - 600.0, 60.0))
            messages.append(
                {
                    "role": "user",
                    "content": f"{prefix}{old_txt}",
                    "ts": echo_ts,
                    "session_id": f"echo_sess_{echo_i}",
                    "echo": True,
                }
            )
    messages.sort(key=lambda row: (float(row["ts"]), str(row["session_id"]), row["content"][:40]))
    return messages


def select_lme_items(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ku = [
        item
        for item in records
        if item.get("question_type") == "knowledge-update" and not str(item.get("question_id", "")).endswith("_abs")
    ]
    ku.sort(key=lambda item: str(item.get("question_id")))
    chosen: List[Dict[str, Any]] = []
    for item in ku:
        messages = raw_conversational_messages(item, inject_echoes=True)
        if sum(1 for row in messages if row["echo"]) == 6:
            chosen.append({"kind": "knowledge-update", "item": item})
        if len(chosen) == 20:
            break
    abstain = [
        item for item in records if str(item.get("question_id", "")).endswith("_abs")
    ]
    abstain.sort(key=lambda item: str(item.get("question_id")))
    for item in abstain[:20]:
        chosen.append({"kind": "abstention", "item": item})
    if len(chosen) != 40:
        raise RuntimeError(f"LongMemEval selection produced {len(chosen)} items, expected 40")
    return chosen


def grade_prediction(pred: str, gold: str, *, abstention: bool) -> bool:
    text = (pred or "").strip().lower()
    if abstention:
        return any(marker in text for marker in ABSTAIN_MARKERS)
    needle = (gold or "").strip().lower()
    if not needle:
        return False
    if needle in text and "abstain" not in text:
        return True
    tokens = [tok for tok in re.findall(r"[a-z0-9]+", needle) if len(tok) > 2]
    if not tokens or "abstain" in text:
        return False
    hits = sum(1 for tok in tokens if tok in text)
    return hits / float(len(tokens)) >= 0.8


def _retry(label: str, fn: Any, attempts: int = 6) -> Any:
    last: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            wait_s = min(30.0, 2.0 ** attempt)
            print(f"[retry] {label} attempt {attempt + 1}/{attempts} failed: {exc}", flush=True)
            time.sleep(wait_s)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}") from last


def _call_llm(endpoint: str, model_id: str, system: str, user: str, max_tokens: int) -> Tuple[str, int]:
    import requests

    def _once() -> Tuple[str, int]:
        response = requests.post(
            f"{endpoint.rstrip('/')}/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=180.0,
        )
        if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
            raise requests.HTTPError(f"status {response.status_code}: {response.text[:200]}")
        response.raise_for_status()
        body = response.json()
        content = (body["choices"][0]["message"].get("content") or "").strip()
        tokens = int(body.get("usage", {}).get("prompt_tokens") or 0)
        return content, tokens

    return _retry("llm", _once)


def _prompt_tokens_from_response(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if usage is None:
        return 0
    if isinstance(usage, Mapping):
        return int(usage.get("prompt_tokens") or 0)
    return int(getattr(usage, "prompt_tokens", 0) or 0)


def _make_mem0(endpoint: str, model_id: str, qpath: Path, *, infer_prompt: bool) -> Tuple[Any, Dict[str, int]]:
    os.environ["MEM0_TELEMETRY"] = "false"
    import mem0.memory.main as _m0_main
    from mem0 import Memory

    # Disable mem0's ~/.mem0/migrations_qdrant local file-lock telemetry client under multi-threading
    _m0_main.capture_event = lambda *a, **kw: None
    if hasattr(_m0_main, "AnonymousTelemetry"):
        _m0_main.AnonymousTelemetry = lambda *a, **kw: type(
            "_NoTelem", (), {"capture_event": lambda *x, **y: None}
        )()

    qdrant_live_url = os.environ.get("PM3_QDRANT_URL", "").strip()
    vs_cfg: Dict[str, Any] = {
        "collection_name": f"r6_pm3_{qpath.name}",
        "embedding_model_dims": 384,
    }
    if qdrant_live_url:
        vs_cfg["url"] = qdrant_live_url
    else:
        vs_cfg["path"] = str(qpath / "qdrant")

    cfg: Dict[str, Any] = {
        "history_db_path": str(qpath / "history.db"),
        "llm": {
            "provider": "openai",
            "config": {
                "model": model_id,
                "openai_base_url": endpoint,
                "api_key": "local-vllm",
                "temperature": 0.0,
                "max_tokens": 512,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": vs_cfg,
        },
    }
    if infer_prompt:
        cfg["custom_fact_extraction_prompt"] = (
            "Extract every concrete personal fact, preference, plan, and knowledge update from the messages. "
            'Return ONLY JSON {"facts": ["..."]}.'
        )
    memory = Memory.from_config(cfg)
    original = memory.llm.client.chat.completions.create
    token_box = {"prompt_tokens": 0}

    def _patched(*args: Any, **kwargs: Any) -> Any:
        extra = dict(kwargs.get("extra_body") or {})
        extra["chat_template_kwargs"] = {"enable_thinking": False}
        kwargs["extra_body"] = extra
        response = original(*args, **kwargs)
        token_box["prompt_tokens"] += _prompt_tokens_from_response(response)
        return response

    memory.llm.client.chat.completions.create = _patched
    return memory, token_box


def _close_mem0(memory: Any, qpath: Path) -> None:
    vs = getattr(memory, "vector_store", None)
    col_name = getattr(vs, "collection_name", None)
    client = getattr(vs, "client", None)
    if client is not None and col_name and os.environ.get("PM3_QDRANT_URL", "").strip():
        try:
            client.delete_collection(collection_name=col_name)
        except Exception:
            pass
    for attr in ("vector_store", "_telemetry_vector_store"):
        store = getattr(memory, attr, None)
        cli = getattr(store, "client", None)
        if cli is not None:
            try:
                cli.close()
            except Exception:
                pass
    shutil.rmtree(qpath, ignore_errors=True)


READER_PM3_SYSTEM = (
    "Answer the question concisely (1-2 sentences) using ONLY the dated records below.\n"
    "- When records contain an earlier value and a newer dated update (or an earlier superseded record), "
    "report the latest current fact unless the question asks about the past as well.\n"
    "- Verify exact entity/premise match: if the specific person, pet, item, or detail asked about was NEVER "
    "mentioned in the records, reply ONLY: ABSTAIN."
)
JUDGE_PM3_SYSTEM = (
    "You are an impartial QA evaluator for LongMemEval. Reply ONLY YES if the Model Answer accurately "
    "provides the core information in the Reference Answer without contradicting it on updated facts. "
    "Reply NO if it gives an outdated/superseded value, contradicts the reference, or fails to answer."
)


def _message_chunks(messages: Sequence[Mapping[str, Any]], max_chars: int = 24000) -> List[List[Dict[str, str]]]:
    """Split the full raw timeline into context-sized chunks. Every message stays in some chunk."""
    chunks: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    size = 0
    for message in messages:
        payload = {"role": str(message["role"]), "content": str(message["content"])}
        line_len = len(payload["role"]) + len(payload["content"]) + 2
        if current and size + line_len > max_chars:
            chunks.append(current)
            current = []
            size = 0
        current.append(payload)
        size += line_len
    if current:
        chunks.append(current)
    return chunks


def _grade_pm3_llm(endpoint: str, model_id: str, question: str, pred: str, gold: str, *, abstention: bool) -> bool:
    if abstention:
        return grade_prediction(pred, gold, abstention=True)
    text = (pred or "").strip()
    if not text or any(m in text.lower() for m in ("abstain", "not mentioned", "did not mention", "no information")):
        return False
    if grade_prediction(text, gold, abstention=False):
        return True
    verdict, _ = _call_llm(
        endpoint,
        model_id,
        JUDGE_PM3_SYSTEM,
        f"Question: {question}\nReference Answer: {gold}\nModel Answer: {text}\n\nCorrect? Reply YES or NO.",
        8,
    )
    return verdict.strip().upper().startswith("YES")


def _chelated_answer(
    endpoint: str,
    model_id: str,
    question: str,
    messages: Sequence[Mapping[str, Any]],
    encoder: Any,
    enc_lock: Any,
) -> Tuple[str, int, float, str]:
    from chelated_conversational_v3 import ChelatedConversationalV3, ConversationTurn

    t0 = time.perf_counter()
    turns: List[ConversationTurn] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        ts = float(m["ts"])
        dt_str = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y/%m/%d %H:%M")
        role = str(m["role"]).capitalize()
        body = str(m["content"]).strip()
        text = f"{role}: {body}"
        idx_text = text
        pos = i
        i += 1
        if role == "User" and i < len(messages) and str(messages[i]["role"]).lower() == "assistant" and str(messages[i]["session_id"]) == str(m["session_id"]):
            reply = str(messages[i]["content"]).strip()
            text += f" | Assistant: {reply[:900]}"
            idx_text += f" | Assistant: {reply[:360]}"
            i += 1
        turns.append(
            ConversationTurn(
                turn_id=f"{m['session_id']}:{pos}",
                session_id=str(m["session_id"]),
                date=dt_str,
                timestamp=ts,
                text=text,
                claim_text=body,
                speaker=role,
                index_text=idx_text,
            )
        )

    class _LockedEmbedder:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            with enc_lock:
                return np.asarray(
                    encoder.encode(list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
                    dtype=np.float32,
                )

    engine = ChelatedConversationalV3(turns, _LockedEmbedder())
    recall = engine.retrieve(question, anchor_count=12, window_radius=2, context_chars=18000, pool_size=36)
    write_ms = (time.perf_counter() - t0) * 1000.0
    status = "GATED_ACTIVE" if recall.gate_active else "ADMIT_OK"
    answer, used = _call_llm(
        endpoint,
        model_id,
        READER_PM3_SYSTEM,
        f"Records:\n{recall.context}\n\nQuestion: {question}\nAnswer:",
        96,
    )
    return answer, int(used), write_ms, status


def _mem0_case(
    endpoint: str,
    model_id: str,
    question: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    infer: bool,
    scratch_dir: Path,
) -> Tuple[str, int, float, int]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    qpath = Path(tempfile.mkdtemp(prefix="r6_mem0_", dir=str(scratch_dir)))
    memory, token_box = _make_mem0(endpoint, model_id, qpath, infer_prompt=infer)
    user_id = "r6_case"
    chunks = _message_chunks(messages) if infer else []
    t0 = time.perf_counter()
    try:
        if infer:
            for chunk in chunks:
                _retry("mem0.add", lambda chunk=chunk: memory.add(chunk, user_id=user_id, infer=True))
        else:
            for message in messages:
                dt = datetime.datetime.fromtimestamp(float(message["ts"]), tz=datetime.timezone.utc).strftime("%Y/%m/%d %H:%M")
                payload = f"[{dt}] {message['role']}: {message['content']}"
                _retry("mem0.add", lambda payload=payload: memory.add(payload, user_id=user_id, infer=False))
        write_ms = (time.perf_counter() - t0) * 1000.0
        found = _retry(
            "mem0.search",
            # mem0 2.2.1 reads top_k. A limit= kwarg is ignored and the default stays 20.
            lambda: memory.search(question, top_k=12, filters={"user_id": user_id}).get("results", []),
        )
    finally:
        _close_mem0(memory, qpath)
    context = "\n".join(f"- {row.get('memory', '')}" for row in found) if found else "(No memories found.)"
    answer, tokens = _call_llm(
        endpoint,
        model_id,
        READER_PM3_SYSTEM,
        f"Records:\n{context}\n\nQuestion: {question}\nAnswer:",
        96,
    )
    return answer, int(token_box["prompt_tokens"]) + tokens, write_ms, len(chunks)


def _read_stage_file(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    found: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        found.setdefault(str(record["question_id"]), {})[str(record["stage"])] = record
    return found


def _append_stage(path: Path, question_id: str, stage: str, payload: Mapping[str, Any], io_lock: Any = None) -> None:
    record = {"question_id": question_id, "stage": stage}
    record.update(payload)
    line = json.dumps(record, ensure_ascii=True) + "\n"
    if io_lock is not None:
        with io_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
    else:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def run_live_n40(
    lme_json: Path,
    endpoint: str,
    model_id: str,
    out_dir: Path,
    *,
    shard_start: int = 0,
    shard_end: int = 40,
    item_workers: int = 6,
) -> Dict[str, Any]:
    import threading

    # Keep BGE-M3 and mem0 MiniLM embeddings off the GPU that vLLM already owns.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from sentence_transformers import SentenceTransformer

    records = json.loads(lme_json.read_text(encoding="utf-8"))
    selected = select_lme_items(records)
    shard = list(enumerate(selected, start=1))[shard_start:shard_end]
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "lme_n40_checkpoint.jsonl"
    stage_path = out_dir / "lme_n40_stages.jsonl"
    done: Dict[str, Dict[str, Any]] = {}
    if checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["question_id"]] = row
    stages = _read_stage_file(stage_path)
    encoder = SentenceTransformer("BAAI/bge-m3", device="cpu")
    encoder.max_seq_length = 384
    io_lock = threading.Lock()
    enc_lock = threading.Lock()

    def _process_one(pair: Tuple[int, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        ordinal, packed = pair
        item = packed["item"]
        qid = str(item["question_id"])
        with io_lock:
            if qid in done:
                print(f"[R6-PM3] {ordinal}/40 {qid} checkpoint-hit", flush=True)
                return qid, done[qid]
        messages = raw_conversational_messages(item, inject_echoes=(packed["kind"] == "knowledge-update"))
        message_blob = json.dumps(
            [{"role": row["role"], "content": row["content"]} for row in messages],
            ensure_ascii=True,
            sort_keys=True,
        )
        message_sha = sha256_bytes(message_blob.encode("utf-8"))
        n_chunks = len(_message_chunks(messages))
        print(
            f"[R6-PM3] {ordinal}/40 {qid} messages={len(messages)} mem0_chunks={n_chunks}",
            flush=True,
        )
        abstention = packed["kind"] == "abstention"
        gold = str(item.get("answer") or "")
        question_str = str(item["question"])
        with io_lock:
            cached = dict(stages.get(qid, {}))

        def _usable(stage_name: str) -> Optional[Dict[str, Any]]:
            record = cached.get(stage_name)
            if record and record.get("message_sha256") == message_sha and record.get("protocol") == "v2_conv_v3":
                return record
            return None

        chelated = _usable("chelated")
        if chelated is None:
            pred, tokens, write_ms, status = _chelated_answer(
                endpoint, model_id, question_str, messages, encoder, enc_lock
            )
            chelated = {
                "protocol": "v2_conv_v3",
                "message_sha256": message_sha,
                "pred": pred,
                "prompt_tokens": tokens,
                "write_ms": write_ms,
                "status": status,
            }
            _append_stage(stage_path, qid, "chelated", chelated, io_lock)
        else:
            print(f"[R6-PM3] {qid} resume chelated", flush=True)
        mem_true = _usable("mem0_infer_true")
        if mem_true is None:
            pred, tokens, write_ms, n_used_chunks = _mem0_case(
                endpoint,
                model_id,
                question_str,
                messages,
                infer=True,
                scratch_dir=scratch,
            )
            mem_true = {
                "protocol": "v2_conv_v3",
                "message_sha256": message_sha,
                "pred": pred,
                "prompt_tokens": tokens,
                "write_ms": write_ms,
                "n_chunks": n_used_chunks,
            }
            _append_stage(stage_path, qid, "mem0_infer_true", mem_true, io_lock)
        else:
            print(f"[R6-PM3] {qid} resume mem0 infer=True", flush=True)
        mem_false = _usable("mem0_infer_false")
        if mem_false is None:
            pred, tokens, write_ms, _n_unused = _mem0_case(
                endpoint,
                model_id,
                question_str,
                messages,
                infer=False,
                scratch_dir=scratch,
            )
            mem_false = {
                "protocol": "v2_conv_v3",
                "message_sha256": message_sha,
                "pred": pred,
                "prompt_tokens": tokens,
                "write_ms": write_ms,
            }
            _append_stage(stage_path, qid, "mem0_infer_false", mem_false, io_lock)
        else:
            print(f"[R6-PM3] {qid} resume mem0 infer=False", flush=True)

        c_ok = _grade_pm3_llm(endpoint, model_id, question_str, str(chelated["pred"]), gold, abstention=abstention)
        mt_ok = _grade_pm3_llm(endpoint, model_id, question_str, str(mem_true["pred"]), gold, abstention=abstention)
        mf_ok = _grade_pm3_llm(endpoint, model_id, question_str, str(mem_false["pred"]), gold, abstention=abstention)

        row = {
            "question_id": qid,
            "kind": packed["kind"],
            "n_messages": len(messages),
            "n_echoes": sum(1 for message in messages if message["echo"]),
            "n_mem0_infer_chunks": int(mem_true.get("n_chunks") or n_chunks),
            "message_sha256": message_sha,
            "identical_raw_messages": True,
            "mem0_infer_true_ingest": "chunked_full_timeline_max_chars_24000",
            "mem0_infer_false_ingest": "one_add_per_raw_message",
            "chelated_ingest": "chelated_conversational_v3_zero_llm_write",
            "gold": gold,
            "chelated_pred": chelated["pred"],
            "chelated_status": chelated["status"],
            "chelated_correct": c_ok,
            "chelated_prompt_tokens": int(chelated["prompt_tokens"]),
            "chelated_write_ms": float(chelated["write_ms"]),
            "mem0_infer_true_pred": mem_true["pred"],
            "mem0_infer_true_correct": mt_ok,
            "mem0_infer_true_prompt_tokens": int(mem_true["prompt_tokens"]),
            "mem0_infer_true_write_ms": float(mem_true["write_ms"]),
            "mem0_infer_false_pred": mem_false["pred"],
            "mem0_infer_false_correct": mf_ok,
            "mem0_infer_false_prompt_tokens": int(mem_false["prompt_tokens"]),
            "mem0_infer_false_write_ms": float(mem_false["write_ms"]),
        }
        with io_lock:
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            done[qid] = row
            print(
                f"[R6-PM3] DONE {ordinal}/40 {qid} ({packed['kind']}): chelated={c_ok} mem0_true={mt_ok} mem0_false={mf_ok}",
                flush=True,
            )
        return qid, row

    with ThreadPoolExecutor(max_workers=max(1, int(item_workers))) as pool:
        list(pool.map(_process_one, shard))

    rows = [done[str(packed["item"]["question_id"])] for _, packed in shard if str(packed["item"]["question_id"]) in done]
    return {"n": len(rows), "rows": rows, "checkpoint": str(checkpoint)}


def _combine_binary(
    n60_scores: Mapping[str, float],
    n40_rows: Sequence[Mapping[str, Any]],
    n60_key: str,
    n40_key: str,
) -> Dict[str, float]:
    combined = {f"n60:{cid}": float(score) for cid, score in n60_scores.items()}
    for row in n40_rows:
        combined[f"n40:{row['question_id']}"] = 1.0 if row[n40_key] else 0.0
    return combined


def assemble_pm3_manifest(n60: Mapping[str, Any], live: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "item": "G2",
        "stage": "R6-PM3",
        "n60": {key: value for key, value in n60.items() if key != "scores"},
        "live_n40_executed": live is not None,
    }
    if live is None:
        manifest["pm3_closed_lcb95_gt_0"] = False
        manifest["closure_note"] = "N=40 live identical-raw-text slice was not executed. N=60 alone does not close PM-3."
        return manifest
    rows = list(live["rows"])
    chelated = _combine_binary(n60["scores"]["chelated_hetero"], rows, "chelated_hetero", "chelated_correct")
    mem_true = _combine_binary(n60["scores"]["mem0_infer_true"], rows, "mem0_infer_true", "mem0_infer_true_correct")
    mem_false = _combine_binary(n60["scores"]["mem0_infer_false"], rows, "mem0_infer_false", "mem0_infer_false_correct")
    if len(chelated) != 100:
        raise RuntimeError(f"N=100 assembly has {len(chelated)} rows")
    boot_true = paired_bootstrap_delta(chelated, mem_true, expected_ids=sorted(chelated), strict_ids=True)
    boot_false = paired_bootstrap_delta(chelated, mem_false, expected_ids=sorted(chelated), strict_ids=True)
    n40_chelated_acc = float(np.mean([1.0 if row["chelated_correct"] else 0.0 for row in rows]))
    n40_true_acc = float(np.mean([1.0 if row["mem0_infer_true_correct"] else 0.0 for row in rows]))
    n40_false_acc = float(np.mean([1.0 if row["mem0_infer_false_correct"] else 0.0 for row in rows]))
    manifest["n100"] = {
        "n": 100,
        "accuracy": {
            "chelated_hetero_plus_live": float(np.mean(list(chelated.values()))),
            "mem0_infer_true": float(np.mean(list(mem_true.values()))),
            "mem0_infer_false": float(np.mean(list(mem_false.values()))),
        },
        "n40_accuracy": {
            "chelated": n40_chelated_acc,
            "mem0_infer_true": n40_true_acc,
            "mem0_infer_false": n40_false_acc,
        },
        "n40_mean_write_ms": {
            "chelated": float(np.mean([row["chelated_write_ms"] for row in rows])),
            "mem0_infer_true": float(np.mean([row["mem0_infer_true_write_ms"] for row in rows])),
            "mem0_infer_false": float(np.mean([row["mem0_infer_false_write_ms"] for row in rows])),
        },
        "n40_mean_prompt_tokens": {
            "chelated": float(np.mean([row["chelated_prompt_tokens"] for row in rows])),
            "mem0_infer_true": float(np.mean([row["mem0_infer_true_prompt_tokens"] for row in rows])),
            "mem0_infer_false": float(np.mean([row["mem0_infer_false_prompt_tokens"] for row in rows])),
        },
        "n60_mean_prompt_tokens": {
            "chelated_hetero": n60["arms"]["chelated_hetero_nemotron"]["mean_prompt_tokens"],
            "mem0_infer_true": n60["arms"]["mem0_infer_true"]["mean_prompt_tokens"],
            "mem0_infer_false": n60["arms"]["mem0_infer_false"]["mean_prompt_tokens"],
        },
        "n60_mem0_infer_true_write_ms_per_case_reused": N60_MEM0_WRITE_MS,
        "paired_bootstrap_chelated_minus_mem0_infer_true": boot_true,
        "paired_bootstrap_chelated_minus_mem0_infer_false": boot_false,
    }
    manifest["pm3_closed_lcb95_gt_0"] = bool(boot_true["lcb_95"] > 0.0 and boot_false["lcb_95"] > 0.0)
    manifest["live_rows"] = rows
    return manifest


def default_paths() -> Dict[str, Path]:
    return {
        "cache": Path("/home/mattmre/chelated-night1/cache"),
        "out_d2": REPO_ROOT / "artifacts" / "phase2_d2_qdrant_hnsw_csls_backfill",
        "out_pm3": REPO_ROOT / "artifacts" / "r6_pm3_fair_rawtext_n100",
        "rows": REPO_ROOT / "artifacts" / "phase2_wave2_d1v2_and_d2_1m" / "phase2_d1_v2_rows.csv",
        "fleet": REPO_ROOT / "artifacts" / "phase2_d1_4node_local_fleet_proof" / "phase2_d1_4node_fleet_rows.csv",
        "mem0": REPO_ROOT / "artifacts" / "phase2_wave2_d1v2_and_d2_1m" / "_work" / "mem0_d1_full60_out.json",
        "claims": REPO_ROOT / "artifacts" / "phase2_wave2_d1v2_and_d2_1m" / "_work" / "rtx3090_qwen7b_v2_claims.json",
        "lme": Path(
            "/home/mattmre/chelated-night1/hf_cache/hub/datasets--xiaowu0162--longmemeval-cleaned/"
            "snapshots/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json"
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("d2", "pm3", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=defaults["cache"])
    parser.add_argument("--out-d2", type=Path, default=defaults["out_d2"])
    parser.add_argument("--out-pm3", type=Path, default=defaults["out_pm3"])
    parser.add_argument("--rows-csv", type=Path, default=defaults["rows"])
    parser.add_argument("--fleet-csv", type=Path, default=defaults["fleet"])
    parser.add_argument("--mem0-json", type=Path, default=defaults["mem0"])
    parser.add_argument("--claims-json", type=Path, default=defaults["claims"])
    parser.add_argument("--lme-json", type=Path, default=defaults["lme"])
    parser.add_argument("--endpoint", default="http://127.0.0.1:8891/v1")
    parser.add_argument("--model-id", default="nemotron-lightning")
    parser.add_argument("--run-live-pm3", action="store_true")
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=40)
    parser.add_argument("--item-workers", type=int, default=6)
    parser.add_argument("--assemble-checkpoint", type=Path, default=None)
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--qdrant-url", default="", help="Qdrant server URL. Empty uses local exact search.")
    args = parser.parse_args(argv)

    if args.part in {"d2", "all"}:
        manifest = run_part_a(
            args.cache_dir,
            args.out_d2,
            run_qdrant=not args.skip_qdrant,
            qdrant_url=args.qdrant_url or None,
        )
        summary = {
            "d2_manifest": str(args.out_d2 / "d2_4096d_qdrant_manifest.json"),
            "domains": [
                {
                    "domain": row.get("domain"),
                    "scored": row.get("scored"),
                    "evaluation": row.get("evaluation"),
                }
                for row in manifest["domains"]
            ],
        }
        print(json.dumps(summary, indent=2), flush=True)
    if args.part in {"pm3", "all"}:
        n60 = lock_n60(args.rows_csv, args.fleet_csv, args.mem0_json, args.claims_json)
        live = None
        if args.assemble_checkpoint is not None:
            by_qid: Dict[str, Dict[str, Any]] = {}
            for line in args.assemble_checkpoint.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    by_qid[str(r["question_id"])] = r
            records = json.loads(args.lme_json.read_text(encoding="utf-8"))
            selected = select_lme_items(records)
            ordered = [by_qid[str(p["item"]["question_id"])] for p in selected]
            live = {"n": len(ordered), "rows": ordered, "checkpoint": str(args.assemble_checkpoint)}
        elif args.run_live_pm3:
            live = run_live_n40(
                args.lme_json,
                args.endpoint,
                args.model_id,
                args.out_pm3,
                shard_start=args.shard_start,
                shard_end=args.shard_end,
                item_workers=args.item_workers,
            )
            if (args.shard_start, args.shard_end) != (0, 40):
                print(json.dumps({"shard_done": [args.shard_start, args.shard_end], "n": live["n"]}), flush=True)
                return
        manifest = assemble_pm3_manifest(n60, live)
        digest = write_json(args.out_pm3 / "r6_pm3_manifest.json", manifest)
        print(
            json.dumps(
                {
                    "pm3_manifest": str(args.out_pm3 / "r6_pm3_manifest.json"),
                    "sha256": digest,
                    "closed": manifest["pm3_closed_lcb95_gt_0"],
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
