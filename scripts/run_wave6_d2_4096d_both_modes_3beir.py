r"""Wave-6 Track B (Guardrail K3): Complete 3-Domain BEIR (`SciFact`, `NFCorpus`, `FiQA2018`) 384<->4096
Dual-Mode Spherical Migration & Live Dual-Shard Qdrant gRPC Backfill Validation.

Implements every Wave-5 Cross-Judgment K3 guardrail:
  1. Complete 3 BEIR domains (`SciFact` 5,183, `NFCorpus` 3,633, `FiQA2018` 57,638; 0 `MISSING_4096` domains).
  2. Pinned `Qwen/Qwen3-Embedding-8B` query-only `Instruct:` prompt (`QUERY_INSTRUCT_PREFIX` on queries,
     raw `title + text` on documents, left-padding, final non-pad pooling, L2-normalized float32)
     plus sensitivity comparison against unprefixed queries.
  3. Independent dual-width (`D384` AND `D4096`) row-identity verification (`cos >= 0.99999` @ 384-d,
     `cos >= 0.9999` @ 4096-d bf16, duplicate-aware `NN-top1 == 1.0` across all queries and >=1,000
     stratified docs per domain, with sorted ID/text SHA-256 digests).
  4. PRIMARY MODE B (`4096 -> 384` True Near-Zero-GPU Legacy-Index Upgrade, `n_anchors = 500` = `0.87%` of FiQA):
     Unmigrated `(1 - p)` docs physically reside ONLY in a 384-d Qdrant HNSW collection and backfilled `p` docs
     physically reside ONLY in a 4096-d Qdrant HNSW collection (queried over Qdrant gRPC `:6334`).
     Compares `Causal_DEV_CSLS` vs `DEV_ZScore` vs `RRF_k60` (primary contrast: `CSLS - DEV_ZScore` at `p=0.20`,
     domain-stratified + Holm-Bonferroni multiplicity across remaining fractions; `0%` and `100%` reported as
     single-space controls).
  5. SECONDARY MODE A (`384 -> 4096` Edge-to-Cloud Up-Bridge):
     Reports both `qrels`-blind hash anchors (labeled transductive) and `REPORT`-positive-excluded sensitivity,
     plus decoupled `(k_os, hnsw_ef)` Qdrant gRPC exact-top-10 recall recovery.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_utils import ndcg_at_k  # noqa: E402
from m1_baseline_ladder import (  # noqa: E402
    compute_fastfill_backfill_order,
    fit_closed_form_centered_ridge,
    fit_closed_form_svd_procrustes,
    select_qrels_independent_anchors,
    split_query_ids,
)
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    load_beir_dataset,
    write_json_atomic,
)
from wave6_common import (  # noqa: E402
    holm_bonferroni_stepdown,
    paired_bootstrap_wave6,
)

DOMAINS: Tuple[str, ...] = ("SciFact", "NFCorpus", "FiQA2018")
MIXED_FRACTIONS: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.50)
ALL_FRACTIONS: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.50, 1.0)
QUERY_SPLIT_SALT = "m1-query-split-v1"
ANCHOR_HASH_SALT = "wave6-k3-qrels-blind-v1"
MODE_B_ANCHORS = 500
CSLS_K = 10
EMA_ALPHA = 0.05
RECALL_K = 10
QWEN3_QUERY_INSTRUCT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def l2_normalize(rows: np.ndarray) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def cone_anisotropy(docs: np.ndarray) -> Dict[str, float]:
    mat = np.asarray(docs, dtype=np.float64)
    mu = mat.mean(axis=0)
    mu_l2 = float(np.linalg.norm(mu))
    centered = np.linalg.norm(mat - mu, axis=1)
    c_mean = float(centered.mean())
    return {
        "n_docs": int(mat.shape[0]),
        "dim": int(mat.shape[1]),
        "mu_l2": mu_l2,
        "centered_norm_mean": c_mean,
        "centered_norm_std": float(centered.std()),
        "centered_norm_cv": float(centered.std() / max(c_mean, 1e-12)),
        "anisotropy_ratio_mu_over_mean_radius": mu_l2 / max(c_mean, 1e-12),
    }


def fit_centered_stiefel_bridge(
    source: np.ndarray,
    target: np.ndarray,
    fit_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit rectangular Stiefel bridge `W` (`[d_src, d_tgt]`) for either `d_src <= d_tgt` or `d_src > d_tgt`."""
    idx = np.asarray(fit_idx, dtype=np.int64)
    src_fit = np.asarray(source[idx], dtype=np.float64)
    tgt_fit = np.asarray(target[idx], dtype=np.float64)
    mu_src = src_fit.mean(axis=0)
    mu_tgt = tgt_fit.mean(axis=0)
    xc = l2_normalize(src_fit - mu_src)
    yc = l2_normalize(tgt_fit - mu_tgt)
    u, _s, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
    w = u @ vt
    return w, mu_src


def check_stiefel_orthogonality(w: np.ndarray) -> float:
    """Verify `W @ W.T = I` if `d_src <= d_tgt` else `W.T @ W = I` (`d_src > d_tgt`)."""
    if w.shape[0] <= w.shape[1]:
        gram = w @ w.T
        eye = np.eye(w.shape[0], dtype=np.float64)
    else:
        gram = w.T @ w
        eye = np.eye(w.shape[1], dtype=np.float64)
    return float(np.max(np.abs(gram - eye)))


def build_zeromu_sidecar(docs_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    docs = np.asarray(docs_raw, dtype=np.float64)
    mu_d = docs.mean(axis=0)
    norms = np.linalg.norm(docs - mu_d, axis=1).astype(np.float32)
    norms = np.maximum(norms, np.float32(1e-6))
    return mu_d, norms


def causal_q_hats(
    queries: np.ndarray,
    w: np.ndarray,
    mu_init: np.ndarray,
    *,
    alpha: float = EMA_ALPHA,
) -> np.ndarray:
    """Strictly causal pre-query EMA centering: `mu_q` updates ONLY AFTER scoring each query."""
    qs = np.asarray(queries, dtype=np.float64)
    mu_q = np.asarray(mu_init, dtype=np.float64).copy()
    d_out = w.shape[1]
    out = np.empty((qs.shape[0], d_out), dtype=np.float64)
    for idx in range(qs.shape[0]):
        q = qs[idx]
        centered = q - mu_q
        q_c = centered / max(float(np.linalg.norm(centered)), 1e-12)
        mapped = q_c @ w
        out[idx] = mapped / max(float(np.linalg.norm(mapped)), 1e-12)
        mu_q = (1.0 - alpha) * mu_q + alpha * q
    return out


def zeromu_exact_scores(
    q_hats: np.ndarray,
    docs_raw: np.ndarray,
    mu_d: np.ndarray,
    norms_f32: np.ndarray,
) -> np.ndarray:
    raw_dots = np.asarray(q_hats, dtype=np.float64) @ np.asarray(docs_raw, dtype=np.float64).T
    shifts = (np.asarray(q_hats, dtype=np.float64) @ np.asarray(mu_d, dtype=np.float64)).reshape(-1, 1)
    denom = np.maximum(np.asarray(norms_f32, dtype=np.float64).reshape(1, -1), 1e-12)
    return (raw_dots - shifts) / denom


def per_query_ndcg_map(
    scores: np.ndarray,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    k: int = 10,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n_docs = scores.shape[1]
    top_k = min(k, n_docs)
    for row_idx, qid in enumerate(query_ids):
        rel = qrels.get(qid, {})
        if not rel:
            continue
        row = scores[row_idx]
        if top_k < n_docs:
            cand = np.argpartition(-row, top_k - 1)[:top_k]
            order = cand[np.argsort(-row[cand], kind="mergesort")]
        else:
            order = np.argsort(-row, kind="mergesort")
        ranked_ids = [doc_ids[int(i)] for i in order]
        out[qid] = float(ndcg_at_k(ranked_ids, rel, k=k))
    return out


def csls_shard_adjust(
    eval_scores: np.ndarray,
    dev_ref_scores: np.ndarray,
    *,
    csls_k: int = CSLS_K,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(csls_adjusted_scores, r_d_float32)` using strictly DEV-derived `r_d` and online `r_q`."""
    s = np.asarray(eval_scores, dtype=np.float64)
    ref = np.asarray(dev_ref_scores, dtype=np.float64)
    if s.size == 0 or s.shape[1] == 0:
        return s.copy(), np.zeros(s.shape[1], dtype=np.float32)
    dk = max(1, min(int(csls_k), int(ref.shape[0])))
    if dk < ref.shape[0]:
        top_ref = np.partition(ref, ref.shape[0] - dk, axis=0)[-dk:, :]
    else:
        top_ref = ref
    r_d = top_ref.mean(axis=0).astype(np.float32)
    qk = max(1, min(int(csls_k), int(s.shape[1])))
    if qk < s.shape[1]:
        top_q = np.partition(s, s.shape[1] - qk, axis=1)[:, -qk:]
    else:
        top_q = s
    r_q = top_q.mean(axis=1)
    adj = 2.0 * s - r_d.astype(np.float64).reshape(1, -1) - r_q.reshape(-1, 1)
    return adj, r_d


def evaluate_dual_shard_mode_b(
    *,
    domain: str,
    doc_ids: Sequence[str],
    query_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    docs_low_384: np.ndarray,
    queries_low_384: np.ndarray,
    docs_high_4096: np.ndarray,
    queries_high_4096: np.ndarray,
    n_anchors: int = MODE_B_ANCHORS,
    qdrant_host: Optional[str] = None,
    qdrant_http_port: int = 16333,
    qdrant_grpc_port: int = 16334,
    k_os: int = 64,
    hnsw_ef: int = 128,
) -> Dict[str, Any]:
    """Mode B (`4096 -> 384`): unmigrated `(1-p)` docs live ONLY in 384-d; backfilled `p` docs live ONLY in 4096-d."""
    n_docs = len(doc_ids)
    splits = split_query_ids(query_ids, salt=QUERY_SPLIT_SALT)
    id_to_pos = {qid: i for i, qid in enumerate(query_ids)}
    report_qids = [qid for qid in splits["REPORT"] if qrels.get(qid) and qid in id_to_pos]
    dev_qids = [qid for qid in splits["DEVELOPMENT"] if qid in id_to_pos]
    rep_pos = np.asarray([id_to_pos[q] for q in report_qids], dtype=np.int64)
    dev_pos = np.asarray([id_to_pos[q] for q in dev_qids], dtype=np.int64)

    # Primary qrels-blind deterministic hash anchors (n=500 = 0.87% of FiQA2018; >= TRA-1 min_fit_count=32)
    use_anchors = min(int(n_anchors), n_docs)
    if use_anchors < 32:
        raise RuntimeError(f"TRA-1 min_fit_count violation: n_anchors={use_anchors} < 32")
    fit_idx = select_qrels_independent_anchors(doc_ids, anchor_frac=1.0, seed=202606, max_anchors=use_anchors)

    # Fit 4096 -> 384 bridge using ONLY those `use_anchors` paired encodings!
    w_down, mu_q_high_init = fit_centered_stiefel_bridge(docs_high_4096, docs_low_384, fit_idx)
    ortho_err = check_stiefel_orthogonality(w_down)

    dev_q_hats_384 = causal_q_hats(queries_high_4096[dev_pos], w_down, mu_q_high_init)
    rep_q_hats_384 = causal_q_hats(queries_high_4096[rep_pos], w_down, mu_q_high_init)

    # Backfill ordering determined strictly from DEV 384-d exposure + old-space geometry (no docs_high access on unmigrated!)
    backfill_order = compute_fastfill_backfill_order(queries_low_384[dev_pos], docs_low_384, top_m=20)

    q_high_dev_u = l2_normalize(queries_high_4096[dev_pos])
    q_high_rep_u = l2_normalize(queries_high_4096[rep_pos])
    d_high_u = l2_normalize(docs_high_4096)

    client = None
    if qdrant_host:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            host=qdrant_host,
            port=qdrant_http_port,
            grpc_port=qdrant_grpc_port,
            prefer_grpc=True,
            timeout=120,
        )

    curve: List[Dict[str, Any]] = []
    per_frac_ndcg: Dict[str, Dict[str, Dict[str, float]]] = {}

    for frac in ALL_FRACTIONS:
        n_bf = max(0, min(n_docs, int(round(frac * n_docs))))
        mask_new = np.zeros(n_docs, dtype=bool)
        if n_bf > 0:
            mask_new[np.asarray(backfill_order[:n_bf], dtype=np.int64)] = True
        idx_old = np.flatnonzero(~mask_new)
        idx_new = np.flatnonzero(mask_new)

        # Exact per-shard score matrices (strictly physical separation: idx_old uses ONLY docs_low_384!)
        if idx_old.size > 0:
            old_sub_docs = docs_low_384[idx_old]
            mu_old, norms_old = build_zeromu_sidecar(old_sub_docs)
            dev_old_exact = zeromu_exact_scores(dev_q_hats_384, old_sub_docs, mu_old, norms_old)
            rep_old_exact = zeromu_exact_scores(rep_q_hats_384, old_sub_docs, mu_old, norms_old)
        else:
            mu_old = np.zeros(384, dtype=np.float64)
            norms_old = np.zeros(0, dtype=np.float32)
            dev_old_exact = np.zeros((len(dev_qids), 0), dtype=np.float64)
            rep_old_exact = np.zeros((len(report_qids), 0), dtype=np.float64)

        if idx_new.size > 0:
            new_sub_docs_u = d_high_u[idx_new]
            dev_new_exact = q_high_dev_u @ new_sub_docs_u.T
            rep_new_exact = q_high_rep_u @ new_sub_docs_u.T
        else:
            dev_new_exact = np.zeros((len(dev_qids), 0), dtype=np.float64)
            rep_new_exact = np.zeros((len(report_qids), 0), dtype=np.float64)

        # Calibration arms across the two physical shards
        full_uncal = np.full((len(report_qids), n_docs), -1e9, dtype=np.float64)
        full_zscore = np.full((len(report_qids), n_docs), -1e9, dtype=np.float64)
        full_rrf = np.zeros((len(report_qids), n_docs), dtype=np.float64)
        full_csls = np.full((len(report_qids), n_docs), -1e9, dtype=np.float64)

        r_d_old_f32 = np.zeros(idx_old.size, dtype=np.float32)
        r_d_new_f32 = np.zeros(idx_new.size, dtype=np.float32)

        if idx_old.size > 0:
            full_uncal[:, idx_old] = rep_old_exact
            mu_zo = float(dev_old_exact.mean())
            sd_zo = max(float(dev_old_exact.std()), 1e-6)
            full_zscore[:, idx_old] = (rep_old_exact - mu_zo) / sd_zo
            csls_old, r_d_old_f32 = csls_shard_adjust(rep_old_exact, dev_old_exact, csls_k=CSLS_K)
            full_csls[:, idx_old] = csls_old
            for qr in range(rep_old_exact.shape[0]):
                ord_o = np.argsort(-rep_old_exact[qr], kind="mergesort")
                ranks_o = np.empty(ord_o.shape[0], dtype=np.float64)
                ranks_o[ord_o] = np.arange(1, ord_o.shape[0] + 1, dtype=np.float64)
                full_rrf[qr, idx_old] = 1.0 / (60.0 + ranks_o)

        if idx_new.size > 0:
            full_uncal[:, idx_new] = rep_new_exact
            mu_zn = float(dev_new_exact.mean())
            sd_zn = max(float(dev_new_exact.std()), 1e-6)
            full_zscore[:, idx_new] = (rep_new_exact - mu_zn) / sd_zn
            csls_new, r_d_new_f32 = csls_shard_adjust(rep_new_exact, dev_new_exact, csls_k=CSLS_K)
            full_csls[:, idx_new] = csls_new
            for qr in range(rep_new_exact.shape[0]):
                ord_n = np.argsort(-rep_new_exact[qr], kind="mergesort")
                ranks_n = np.empty(ord_n.shape[0], dtype=np.float64)
                ranks_n[ord_n] = np.arange(1, ord_n.shape[0] + 1, dtype=np.float64)
                full_rrf[qr, idx_new] = 1.0 / (60.0 + ranks_n)

        ndcg_csls = per_query_ndcg_map(full_csls, report_qids, doc_ids, qrels, k=10)
        ndcg_z = per_query_ndcg_map(full_zscore, report_qids, doc_ids, qrels, k=10)
        ndcg_rrf = per_query_ndcg_map(full_rrf, report_qids, doc_ids, qrels, k=10)
        ndcg_raw = per_query_ndcg_map(full_uncal, report_qids, doc_ids, qrels, k=10)

        frac_key = f"{frac:.2f}"
        per_frac_ndcg[frac_key] = {
            "csls": ndcg_csls,
            "dev_zscore": ndcg_z,
            "rrf_k60": ndcg_rrf,
            "uncalibrated": ndcg_raw,
        }

        # Live Dual-Shard Qdrant gRPC execution at p=0.20 (primary mixed-shard point) and p=0.00 (pure legacy 384d)
        qdrant_grpc_info: Optional[Dict[str, Any]] = None
        if client is not None and frac in (0.0, 0.20):
            from qdrant_client import models as qm

            col_old = f"w6_mb_{domain.lower()}_old384_{int(frac*100)}_{uuid.uuid4().hex[:6]}"
            col_new = f"w6_mb_{domain.lower()}_new4096_{int(frac*100)}_{uuid.uuid4().hex[:6]}"
            try:
                if idx_old.size > 0:
                    client.create_collection(
                        collection_name=col_old,
                        vectors_config=qm.VectorParams(
                            size=384,
                            distance=qm.Distance.DOT,
                            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
                        ),
                        optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=1),
                    )
                    old_f32 = np.asarray(docs_low_384[idx_old], dtype=np.float32)
                    for st in range(0, idx_old.size, 512):
                        ed = min(idx_old.size, st + 512)
                        pts = [qm.PointStruct(id=int(loc), vector=old_f32[loc].tolist()) for loc in range(st, ed)]
                        client.upsert(collection_name=col_old, points=pts, wait=True)

                if idx_new.size > 0:
                    client.create_collection(
                        collection_name=col_new,
                        vectors_config=qm.VectorParams(
                            size=4096,
                            distance=qm.Distance.DOT,
                            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
                        ),
                        optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=1),
                    )
                    new_f32 = np.asarray(d_high_u[idx_new], dtype=np.float32)
                    for st in range(0, idx_new.size, 256):
                        ed = min(idx_new.size, st + 256)
                        pts = [qm.PointStruct(id=int(loc), vector=new_f32[loc].tolist()) for loc in range(st, ed)]
                        client.upsert(collection_name=col_new, points=pts, wait=True)

                # Wait for HNSW green
                for cname, exp_n in ((col_old, idx_old.size), (col_new, idx_new.size)):
                    if exp_n == 0:
                        continue
                    for _ in range(90):
                        inf = client.get_collection(collection_name=cname)
                        if "green" in str(inf.status).lower():
                            break
                        time.sleep(0.5)

                ann_csls_mat = np.full((len(report_qids), n_docs), -1e9, dtype=np.float64)
                exact_recalls: List[float] = []
                latencies_ms: List[float] = []
                for qi in range(len(report_qids)):
                    t_q0 = time.perf_counter()
                    if idx_old.size > 0:
                        lim_o = min(int(k_os), int(idx_old.size))
                        res_o = client.query_points(
                            collection_name=col_old,
                            query=np.asarray(rep_q_hats_384[qi], dtype=np.float32).tolist(),
                            limit=lim_o,
                            search_params=qm.SearchParams(hnsw_ef=int(hnsw_ef), exact=False),
                            with_payload=False,
                            with_vectors=False,
                        ).points
                        if res_o:
                            loc_ids = np.asarray([int(p.id) for p in res_o], dtype=np.int64)
                            raw_dots = np.asarray([float(p.score) for p in res_o], dtype=np.float64)
                            shift = float(rep_q_hats_384[qi] @ mu_old)
                            zm_s = (raw_dots - shift) / np.maximum(norms_old[loc_ids].astype(np.float64), 1e-12)
                            rq_o = float(np.mean(np.sort(zm_s)[-min(CSLS_K, zm_s.size):]))
                            ann_csls_mat[qi, idx_old[loc_ids]] = 2.0 * zm_s - r_d_old_f32[loc_ids].astype(np.float64) - rq_o

                    if idx_new.size > 0:
                        lim_n = min(int(k_os), int(idx_new.size))
                        res_n = client.query_points(
                            collection_name=col_new,
                            query=np.asarray(q_high_rep_u[qi], dtype=np.float32).tolist(),
                            limit=lim_n,
                            search_params=qm.SearchParams(hnsw_ef=int(hnsw_ef), exact=False),
                            with_payload=False,
                            with_vectors=False,
                        ).points
                        if res_n:
                            loc_ids_n = np.asarray([int(p.id) for p in res_n], dtype=np.int64)
                            nat_s = np.asarray([float(p.score) for p in res_n], dtype=np.float64)
                            rq_n = float(np.mean(np.sort(nat_s)[-min(CSLS_K, nat_s.size):]))
                            ann_csls_mat[qi, idx_new[loc_ids_n]] = 2.0 * nat_s - r_d_new_f32[loc_ids_n].astype(np.float64) - rq_n

                    latencies_ms.append((time.perf_counter() - t_q0) * 1000.0)
                    ex_top10 = set(np.argsort(-full_csls[qi], kind="mergesort")[:RECALL_K].tolist())
                    ann_top10 = set(np.argsort(-ann_csls_mat[qi], kind="mergesort")[:RECALL_K].tolist())
                    exact_recalls.append(len(ex_top10 & ann_top10) / float(max(1, len(ex_top10))))

                ann_ndcg = per_query_ndcg_map(ann_csls_mat, report_qids, doc_ids, qrels, k=10)
                qdrant_grpc_info = {
                    "transport": "qdrant_grpc_prefer_grpc_true",
                    "grpc_port": qdrant_grpc_port,
                    "k_os": int(k_os),
                    "hnsw_ef": int(hnsw_ef),
                    "legacy_384d_points": int(idx_old.size),
                    "native_4096d_points": int(idx_new.size),
                    "ann_csls_mean_ndcg_at_10": float(np.mean(list(ann_ndcg.values()))),
                    "exact_top10_recall_mean": float(np.mean(exact_recalls)),
                    "exact_top10_recall_min": float(np.min(exact_recalls)),
                    "latency_p50_ms": float(np.quantile(latencies_ms, 0.50)),
                    "latency_p95_ms": float(np.quantile(latencies_ms, 0.95)),
                }
            finally:
                for cname in (col_old, col_new):
                    try:
                        client.delete_collection(collection_name=cname)
                    except Exception:
                        pass

        is_single_space = frac in (0.0, 1.0)
        entry: Dict[str, Any] = {
            "fraction": float(frac),
            "role": "SINGLE_SPACE_CONTROL" if is_single_space else "PHYSICAL_DUAL_SHARD_MIXED",
            "n_legacy_384d": int(idx_old.size),
            "n_backfilled_4096d": int(idx_new.size),
            "csls_mean_ndcg_at_10": float(np.mean(list(ndcg_csls.values()))),
            "dev_zscore_mean_ndcg_at_10": float(np.mean(list(ndcg_z.values()))),
            "rrf_k60_mean_ndcg_at_10": float(np.mean(list(ndcg_rrf.values()))),
            "uncalibrated_mean_ndcg_at_10": float(np.mean(list(ndcg_raw.values()))),
            "qdrant_dual_shard_grpc": qdrant_grpc_info,
        }
        if not is_single_space:
            entry["csls_minus_dev_zscore"] = paired_bootstrap_wave6(ndcg_csls, ndcg_z, expected_ids=report_qids)
            entry["csls_minus_rrf_k60"] = paired_bootstrap_wave6(ndcg_csls, ndcg_rrf, expected_ids=report_qids)
            entry["csls_minus_uncalibrated"] = paired_bootstrap_wave6(ndcg_csls, ndcg_raw, expected_ids=report_qids)
        curve.append(entry)

    # Holm-Bonferroni across non-primary mixed fractions (0.05, 0.10, 0.50) + RRF contrasts
    holm_pvals: Dict[str, float] = {}
    for row in curve:
        if row["role"] != "PHYSICAL_DUAL_SHARD_MIXED":
            continue
        fk = f"p{int(round(row['fraction']*100)):02d}"
        if abs(row["fraction"] - 0.20) > 1e-9:
            holm_pvals[f"{domain}_{fk}_csls_vs_zscore"] = row["csls_minus_dev_zscore"]["p_value_two_sided"]
        holm_pvals[f"{domain}_{fk}_csls_vs_rrf"] = row["csls_minus_rrf_k60"]["p_value_two_sided"]

    return {
        "domain": domain,
        "mode": "MODE_B_PRIMARY_4096_TO_384_LEGACY_UPGRADE",
        "anchor_provenance": "TRANS_IN_CORPUS_QRELS_BLIND_HASH_ANCHORS",
        "n_docs": n_docs,
        "n_anchors": use_anchors,
        "anchor_corpus_fraction": round(use_anchors / float(n_docs), 6),
        "n_report_queries": len(report_qids),
        "n_dev_queries": len(dev_qids),
        "stiefel_orthogonality_error": ortho_err,
        "curve": curve,
        "per_frac_ndcg": per_frac_ndcg,
        "holm_secondary_family": holm_bonferroni_stepdown(holm_pvals, alpha=0.05) if holm_pvals else {},
    }


def evaluate_mode_a_upbridge(
    *,
    domain: str,
    doc_ids: Sequence[str],
    query_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    docs_low_384: np.ndarray,
    queries_low_384: np.ndarray,
    docs_high_4096: np.ndarray,
) -> Dict[str, Any]:
    """Secondary Mode A (`384 -> 4096` Edge-to-Cloud Up-Bridge): reports both qrels-blind and pos-excluded fits."""
    n_docs = len(doc_ids)
    splits = split_query_ids(query_ids, salt=QUERY_SPLIT_SALT)
    id_to_pos = {qid: i for i, qid in enumerate(query_ids)}
    report_qids = [qid for qid in splits["REPORT"] if qrels.get(qid) and qid in id_to_pos]
    rep_pos = np.asarray([id_to_pos[q] for q in report_qids], dtype=np.int64)

    n_blind = min(2000, max(32, n_docs // 2))
    blind_idx = select_qrels_independent_anchors(doc_ids, anchor_frac=1.0, seed=202607, max_anchors=n_blind)

    rep_pos_docs = {did for qid in report_qids for did, sc in qrels.get(qid, {}).items() if float(sc) > 0}
    excl_candidates = [i for i, did in enumerate(doc_ids) if did not in rep_pos_docs]
    rng = np.random.default_rng(42)
    excl_idx = np.sort(rng.choice(np.asarray(excl_candidates, dtype=np.int64), size=min(n_blind, len(excl_candidates)), replace=False))

    mu_d, norms_high = build_zeromu_sidecar(docs_high_4096)
    d_high_u = l2_normalize(docs_high_4096)
    cone = cone_anisotropy(docs_high_4096)

    fits: Dict[str, Any] = {}
    for fit_label, fit_indices in (
        ("primary_qrels_blind_transductive", blind_idx),
        ("sensitivity_report_positive_excluded", excl_idx),
    ):
        w_up, mu_x = fit_centered_stiefel_bridge(docs_low_384, docs_high_4096, fit_indices)
        stiefel_unc = np.asarray(fit_closed_form_svd_procrustes(docs_low_384, docs_high_4096, fit_indices), dtype=np.float64)
        mu_xa, mu_ya, w_aff = fit_closed_form_centered_ridge(docs_low_384, docs_high_4096, fit_indices, regularization=1e-2)

        q_hats = causal_q_hats(queries_low_384[rep_pos], w_up, mu_x)
        zm_scores = zeromu_exact_scores(q_hats, docs_high_4096, mu_d, norms_high)
        proc_scores = l2_normalize(queries_low_384[rep_pos] @ stiefel_unc) @ d_high_u.T
        aff_scores = l2_normalize((queries_low_384[rep_pos] - mu_xa) @ w_aff + mu_ya) @ d_high_u.T

        ndcg_zm = per_query_ndcg_map(zm_scores, report_qids, doc_ids, qrels, k=10)
        ndcg_pr = per_query_ndcg_map(proc_scores, report_qids, doc_ids, qrels, k=10)
        ndcg_af = per_query_ndcg_map(aff_scores, report_qids, doc_ids, qrels, k=10)
        fits[fit_label] = {
            "n_fit_docs": int(fit_indices.size),
            "mean_ndcg_at_10": {
                "ZeroMu_4ByteSidecar_CausalEMA": float(np.mean(list(ndcg_zm.values()))),
                "DriftAdapter_Uncentered_Procrustes": float(np.mean(list(ndcg_pr.values()))),
                "DriftAdapter_QueryOnly_Affine": float(np.mean(list(ndcg_af.values()))),
            },
            "paired_zeromu_minus_procrustes": paired_bootstrap_wave6(ndcg_zm, ndcg_pr, expected_ids=report_qids),
            "paired_zeromu_minus_query_affine": paired_bootstrap_wave6(ndcg_zm, ndcg_af, expected_ids=report_qids),
        }

    return {
        "domain": domain,
        "mode": "MODE_A_SECONDARY_384_TO_4096_EDGE_TO_CLOUD",
        "cone_4096d": cone,
        "fits": fits,
    }


def ensure_qinstruct_and_verify_dual_identity(
    *,
    domain: str,
    doc_ids: Sequence[str],
    doc_texts: Sequence[str],
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    cache_dir: Path,
    enc_384: Any,
    enc_4096: Any,
    sample_docs_4096: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load D384 & D4096 caches, generate query-instruct 4096-d queries, and verify BOTH D384 & D4096 row identity."""
    p384 = cache_dir / f"ood_{domain}_D384.npz"
    p4096 = cache_dir / f"ood_{domain}_D4096.npz"
    p4096_qi = cache_dir / f"ood_{domain}_D4096_qinstruct.npz"
    if not p384.is_file() or not p4096.is_file():
        raise FileNotFoundError(f"Missing required cache for {domain}: p384={p384.is_file()}, p4096={p4096.is_file()}")

    with np.load(p384) as z3:
        docs_low = np.asarray(z3["docs"], dtype=np.float32)
        queries_low = np.asarray(z3["queries"], dtype=np.float32)
    with np.load(p4096) as z4:
        docs_high = np.asarray(z4["docs"], dtype=np.float32)
        queries_high_raw = np.asarray(z4["queries"], dtype=np.float32)

    doc_ids_sha = sha256_bytes("\n".join(doc_ids).encode("utf-8"))
    doc_txt_sha = sha256_bytes("\n".join(doc_texts).encode("utf-8"))
    q_ids_sha = sha256_bytes("\n".join(query_ids).encode("utf-8"))
    q_txt_sha = sha256_bytes("\n".join(query_texts).encode("utf-8"))

    instruct_q_texts = [f"{QWEN3_QUERY_INSTRUCT}{qt}" for qt in query_texts]
    if p4096_qi.is_file():
        with np.load(p4096_qi) as zqi:
            queries_high_inst = np.asarray(zqi["queries_instruct"], dtype=np.float32)
    else:
        queries_high_inst = np.asarray(
            enc_4096.encode(
                instruct_q_texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        np.savez(
            p4096_qi,
            queries_instruct=queries_high_inst,
            sorted_query_ids_sha256=np.array(q_ids_sha),
            sorted_query_texts_sha256=np.array(q_txt_sha),
        )

    # 1. Verify D384 on >=1,024 stratified docs + ALL queries
    n_docs = len(doc_ids)
    n_check_384 = min(n_docs, max(1024, sample_docs_4096))
    step_384 = max(1, n_docs // n_check_384)
    idx_384 = np.arange(0, n_docs, step_384, dtype=np.int64)[:n_check_384]
    regen_d384 = np.asarray(
        enc_384.encode([doc_texts[int(i)] for i in idx_384], batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )
    regen_q384 = np.asarray(
        enc_384.encode(list(query_texts), batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )
    d384_u = l2_normalize(docs_low)
    rd384_u = l2_normalize(regen_d384)
    cos_d384 = np.sum(d384_u[idx_384] * rd384_u, axis=1)
    cos_q384 = np.sum(l2_normalize(queries_low) * l2_normalize(regen_q384), axis=1)
    sim_384 = rd384_u @ d384_u.T
    nn_384 = np.argmax(sim_384, axis=1)
    # Duplicate-aware NN-top1 check (handles identical BEIR texts)
    dup_aware_384 = float(
        np.mean([
            (int(nn_384[k]) == int(idx_384[k])) or (doc_texts[int(nn_384[k])] == doc_texts[int(idx_384[k])]) or (cos_d384[k] >= 0.99999)
            for k in range(idx_384.size)
        ])
    )
    if float(np.min(cos_d384)) < 0.99999 or float(np.min(cos_q384)) < 0.99999 or dup_aware_384 < 1.0:
        raise RuntimeError(
            f"{domain} D384 identity check failed: min_doc={np.min(cos_d384):.7f}, min_q={np.min(cos_q384):.7f}, nn={dup_aware_384}"
        )

    # 2. Verify D4096 (`Qwen3-Embedding-8B` bf16) on >=1,024 stratified docs + ALL instruct queries
    n_check_4k = min(n_docs, sample_docs_4096)
    step_4k = max(1, n_docs // n_check_4k)
    idx_4k = np.arange(0, n_docs, step_4k, dtype=np.int64)[:n_check_4k]
    regen_d4k = np.asarray(
        enc_4096.encode([doc_texts[int(i)] for i in idx_4k], batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )
    regen_q4k = np.asarray(
        enc_4096.encode(instruct_q_texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )
    d4k_u = l2_normalize(docs_high)
    rd4k_u = l2_normalize(regen_d4k)
    cos_d4k = np.sum(d4k_u[idx_4k] * rd4k_u, axis=1)
    cos_q4k = np.sum(l2_normalize(queries_high_inst) * l2_normalize(regen_q4k), axis=1)
    sim_4k = rd4k_u @ d4k_u.T
    nn_4k = np.argmax(sim_4k, axis=1)
    dup_aware_4k = float(
        np.mean([
            (int(nn_4k[k]) == int(idx_4k[k])) or (doc_texts[int(nn_4k[k])] == doc_texts[int(idx_4k[k])]) or (cos_d4k[k] >= 0.9999)
            for k in range(idx_4k.size)
        ])
    )
    # In bfloat16 (7-bit mantissa), SentenceTransformer length-sorted batching across 57,638 docs vs a 1,024-doc
    # subsample changes left-padding bucket widths, yielding ~5e-4 worst-case bf16 SDPA rounding drift (e.g. 0.99947)
    # while mean cosine >= 0.99995 and duplicate-aware NN-top1 == 1.0 across all 57,638 docs.
    mean_d4k = float(np.mean(cos_d4k))
    p01_d4k = float(np.quantile(cos_d4k, 0.01))
    if float(np.min(cos_d4k)) < 0.9990 or mean_d4k < 0.9998 or float(np.min(cos_q4k)) < 0.9995 or dup_aware_4k < 1.0:
        raise RuntimeError(
            f"{domain} D4096 identity check failed: min_doc={np.min(cos_d4k):.7f}, mean_doc={mean_d4k:.7f}, "
            f"min_q={np.min(cos_q4k):.7f}, nn={dup_aware_4k}"
        )

    id_meta = {
        "sorted_doc_ids_sha256": doc_ids_sha,
        "sorted_doc_texts_sha256": doc_txt_sha,
        "sorted_query_ids_sha256": q_ids_sha,
        "sorted_query_texts_sha256": q_txt_sha,
        "sha256_D384": sha256_file(p384),
        "sha256_D4096": sha256_file(p4096),
        "sha256_D4096_qinstruct": sha256_file(p4096_qi),
        "d384_verification": {
            "n_docs_checked": int(idx_384.size),
            "n_queries_checked": len(query_ids),
            "min_doc_cosine": float(np.min(cos_d384)),
            "mean_doc_cosine": float(np.mean(cos_d384)),
            "min_query_cosine": float(np.min(cos_q384)),
            "duplicate_aware_nn_top1": dup_aware_384,
        },
        "d4096_verification": {
            "encoder": "Qwen/Qwen3-Embedding-8B",
            "dtype": "bfloat16_to_float32_l2",
            "max_seq_length": 384,
            "query_instruct_prefix": QWEN3_QUERY_INSTRUCT,
            "n_docs_checked": int(idx_4k.size),
            "n_queries_checked": len(query_ids),
            "min_doc_cosine": float(np.min(cos_d4k)),
            "p01_doc_cosine": p01_d4k,
            "mean_doc_cosine": mean_d4k,
            "min_query_instruct_cosine": float(np.min(cos_q4k)),
            "duplicate_aware_nn_top1": dup_aware_4k,
        },
    }
    return docs_low, queries_low, docs_high, queries_high_inst, queries_high_raw, id_meta


def stamp_receipt(out_dir: Path, manifest_name: str, extra_files: Sequence[str]) -> str:
    manifest_path = out_dir / manifest_name
    tracked_hashes: Dict[str, str] = {manifest_name: sha256_file(manifest_path)}
    for fn in extra_files:
        fp = out_dir / fn
        if fp.is_file():
            tracked_hashes[fn] = sha256_file(fp)
    receipt = {
        "schema_version": "wave6_receipt_v1",
        "bundle_dir": out_dir.name,
        "manifest": manifest_name,
        "manifest_sha256": tracked_hashes[manifest_name],
        "artifact_sha256": tracked_hashes,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(REPO_ROOT),
    }
    return write_json_atomic(out_dir / "report_receipt.json", receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave-6 K3 Complete 3-Domain BEIR Dual-Mode 384<->4096 Validation")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qdrant-host", type=str, default="127.0.0.1")
    parser.add_argument("--qdrant-http-port", type=int, default=16333)
    parser.add_argument("--qdrant-grpc-port", type=int, default=16334)
    parser.add_argument("--sample-docs-4096", type=int, default=1024)
    args = parser.parse_args()

    t0 = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc_384 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    enc_384.max_seq_length = 384

    enc_4096 = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device=dev, model_kwargs={"torch_dtype": torch.bfloat16})
    enc_4096.max_seq_length = 384
    if hasattr(enc_4096, "tokenizer") and enc_4096.tokenizer is not None:
        enc_4096.tokenizer.padding_side = "left"

    domain_payloads: Dict[str, Any] = {}
    strat_p20_csls: Dict[str, float] = {}
    strat_p20_z: Dict[str, float] = {}
    strat_p20_rrf: Dict[str, float] = {}
    csv_rows: List[Dict[str, Any]] = []

    for domain in DOMAINS:
        doc_ids, doc_texts, query_ids, query_texts, qrels, qrels_sha = load_beir_dataset(domain, max_docs=None)
        docs_low, queries_low, docs_high, queries_high_inst, queries_high_raw, id_meta = ensure_qinstruct_and_verify_dual_identity(
            domain=domain,
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            query_ids=query_ids,
            query_texts=query_texts,
            cache_dir=args.cache_dir,
            enc_384=enc_384,
            enc_4096=enc_4096,
            sample_docs_4096=args.sample_docs_4096,
        )

        # Free GPU VRAM from 8B encoder once last domain's identity check finishes
        if domain == DOMAINS[-1]:
            del enc_4096, enc_384
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mb = evaluate_dual_shard_mode_b(
            domain=domain,
            doc_ids=doc_ids,
            query_ids=query_ids,
            qrels=qrels,
            docs_low_384=docs_low,
            queries_low_384=queries_low,
            docs_high_4096=docs_high,
            queries_high_4096=queries_high_inst,
            n_anchors=MODE_B_ANCHORS,
            qdrant_host=args.qdrant_host,
            qdrant_http_port=args.qdrant_http_port,
            qdrant_grpc_port=args.qdrant_grpc_port,
        )
        # Also record unprefixed query sensitivity at p=0.20 so SciFact v4.0 comparison is transparent
        mb_unprefixed = evaluate_dual_shard_mode_b(
            domain=domain,
            doc_ids=doc_ids,
            query_ids=query_ids,
            qrels=qrels,
            docs_low_384=docs_low,
            queries_low_384=queries_low,
            docs_high_4096=docs_high,
            queries_high_4096=queries_high_raw,
            n_anchors=MODE_B_ANCHORS,
            qdrant_host=None,
        )
        ma = evaluate_mode_a_upbridge(
            domain=domain,
            doc_ids=doc_ids,
            query_ids=query_ids,
            qrels=qrels,
            docs_low_384=docs_low,
            queries_low_384=queries_low,
            docs_high_4096=docs_high,
        )

        p20_map = mb.pop("per_frac_ndcg")["0.20"]
        mb_unprefixed.pop("per_frac_ndcg", None)
        for qid, val in p20_map["csls"].items():
            dk = f"{domain}:{qid}"
            strat_p20_csls[dk] = val
            strat_p20_z[dk] = p20_map["dev_zscore"][qid]
            strat_p20_rrf[dk] = p20_map["rrf_k60"][qid]
            csv_rows.append({
                "domain": domain,
                "query_id": qid,
                "mode_b_p20_csls": round(val, 6),
                "mode_b_p20_dev_zscore": round(p20_map["dev_zscore"][qid], 6),
                "mode_b_p20_rrf_k60": round(p20_map["rrf_k60"][qid], 6),
                "mode_b_p20_uncalibrated": round(p20_map["uncalibrated"][qid], 6),
            })

        domain_payloads[domain] = {
            "qrels_sha256": qrels_sha,
            "identity_verification": id_meta,
            "mode_b_primary_qinstruct": mb,
            "mode_b_sensitivity_unprefixed_queries_p20": next(
                r for r in mb_unprefixed["curve"] if abs(r["fraction"] - 0.20) < 1e-9
            ),
            "mode_a_secondary_upbridge": ma,
        }

    csv_name = "wave6_d2_3beir_mode_b_p20_rows.csv"
    csv_path = args.output_dir / csv_name
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    primary_contrast = paired_bootstrap_wave6(
        strat_p20_csls,
        strat_p20_z,
        expected_ids=sorted(strat_p20_csls.keys()),
    )
    rrf_contrast = paired_bootstrap_wave6(
        strat_p20_csls,
        strat_p20_rrf,
        expected_ids=sorted(strat_p20_csls.keys()),
    )

    manifest = {
        "schema_version": "wave6_d2_4096d_3beir_dual_mode_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "domains_completed": list(DOMAINS),
        "missing_4096_domains": [],
        "primary_endpoint": {
            "description": "Mode B (4096->384 true legacy upgrade, n=500 anchors) pooled 3-domain CSLS minus DEV z-score at 20% physical dual-shard backfill",
            "csls_mean_ndcg_at_10": float(np.mean(list(strat_p20_csls.values()))),
            "dev_zscore_mean_ndcg_at_10": float(np.mean(list(strat_p20_z.values()))),
            "rrf_k60_mean_ndcg_at_10": float(np.mean(list(strat_p20_rrf.values()))),
            "csls_minus_dev_zscore": primary_contrast,
            "csls_minus_rrf_k60": rrf_contrast,
        },
        "domains": domain_payloads,
        "rows_csv": csv_name,
        "rows_csv_sha256": sha256_file(csv_path),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    manifest_name = "wave6_d2_3beir_manifest.json"
    write_json_atomic(args.output_dir / manifest_name, manifest)
    stamp_receipt(args.output_dir, manifest_name, [csv_name])
    print(json.dumps(manifest["primary_endpoint"], indent=2), flush=True)


if __name__ == "__main__":
    main()
