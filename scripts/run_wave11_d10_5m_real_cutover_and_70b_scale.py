r"""Wave 11 Lane 5 (`Demo D10`, Guardrail `L5`):
Full Unsubsampled `8.84M` (`BeIR/msmarco`) & `9.47M` 7-Domain Real-Document Zero-Downtime Cutover
+ Local `70B` (`unsloth/Llama-3.3-70B-Instruct-bnb-4bit`, 4-bit NF4) Frontier Scale Sweep.

Enforces every Wave-10 Panel-of-Experts `L5` guardrail (`Fable 5.2`, `Carbon 4`, `Sol 6.2`, `Neon 4`):
  1. 100% Real Passages, Zero Synthetic Vectors, Zero MSMARCO Subsampling:
     - Evaluates BOTH standalone full `BeIR/msmarco` (`8,841,823` passages) for apples-to-apples
       MSMARCO dev comparability AND the merged 7-domain enterprise index (`9,465,539` real passages =
       `MSMARCO` + `Quora` + `FiQA2018` + `SciDocs` + `ArguAna` + `SciFact` + `NFCorpus`).
     - Global `{dataset}:{orig_id}` namespacing (`assert len(unique_ids) == total_docs`), ArguAna
       self-match exclusion (`doc_id == query_id`), and per-domain unpooled `nDCG@10` / `Recall@10`.
  2. Streamed `65,536`-Row `np.memmap` Sidecar & Exact Binary (`MiB`/`GiB`) Sizing:
     - `compute_chunked_zeromu_csls_sidecar` streams float16/float32 disk memmaps in `65,536`-row
       chunks (`0` full-matrix float64 allocation!).
     - Exact binary ratios: `8 B/doc` sidecar vs `1024-d` float32 (`4,096 B/doc` = `512.0x` exact)
       and `4096-d` float32 (`16,384 B/doc` = `2,048.0x` exact).
     - Explicit FAISS index factory disclosure (`IndexHNSWSQ` `QT_fp16` `M=16` `METRIC_INNER_PRODUCT`),
       sampled exact-Flat recall verification, peak RSS gate (`<= 44 GiB`), `EK5 v3` packed epoch
       admission fence (`0.00%` stale resurrection), and `del` + `malloc_trim(0)` cleanup.
  3. Single-Locked `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` (`4-bit NF4` Disclosed Everywhere):
     - Evaluates `Bare Llama-3.3-70B (4-bit NF4)` at both `k=5` (HybridRAG parity) and `k=17`
       (`100%` recall control) alongside `Bare Nemotron-30B`, `Bare Qwen2.5-7B`,
       and Chelated readers at 1.5B, 7B, and 70B under the fixed Nemotron-30B extractor
       and governance pipeline on `Suite A (N=60)`.
     - Deterministic `strict_exact_match` across all arms + zero-violation cross-family blind
       judging (`Nemotron-30B` judging `Llama` & `Qwen`; `Qwen2.5-7B` judging `Llama` & `Nemotron`,
       reporting inter-judge Cohen's kappa on `Llama-70B` predictions).
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable
from heapq import nsmallest
from tempfile import NamedTemporaryFile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_utils import ndcg_at_k  # noqa: E402
from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore  # noqa: E402
from m1_baseline_ladder import (  # noqa: E402
    select_qrels_independent_anchors,
    split_query_ids,
)
from prospective_realworld_drift_corpus import (  # noqa: E402
    build_prospective_realworld_drift_suite,
)
from scripts.night1_common import (  # noqa: E402
    FastBM25Index,
    current_git_sha,
    write_json_atomic,
)
from wave6_common import (  # noqa: E402
    BlindCrossFamilySemanticJudge,
    holm_bonferroni_stepdown,
    paired_bootstrap_wave6,
)

# ---------------------------------------------------------------------------
# Constants & Strict Binary Unit Helpers (`L5`)
# ---------------------------------------------------------------------------
MIB = float(1024 ** 2)
GIB = float(1024 ** 3)
SIDECAR_BYTES_PER_DOC = 8  # 4 B float32 ||d - mu_d||_2 + 4 B float32 causal-DEV CSLS r_d
CHUNK_ROWS = 65_536
QUERY_SPLIT_SALT = "m1-query-split-v1"
ANCHOR_SALT = "wave11-d10-qrels-blind-v1"
CSLS_K = 10
EMA_ALPHA = 0.05
FLAT_ORACLE_SAMPLE_SIZE = 50
FLAT_ORACLE_RECALL_FLOOR = 0.85
RETRACTION_ORACLE_SAMPLE_SIZE = 25
TOP10_TIE_BREAKING = "(-score, doc_id)"

PART_A_SUMMARY = "part_a_cutover_summary.json"
PART_B_SUMMARY = "part_b_70b_sweep_summary.json"
BARE_NEMOTRON_ARM = "Bare_Nemotron_30B_NVFP4_HybridRAG_k5"
SMALL_CHELATED_ARM = "Small_Qwen_Plus_Chelated_k5"
MID_CHELATED_ARM = "Mid_Qwen2_5_7B_Plus_Chelated_k5"
BARE_70B_K5_ARM = "Bare_Llama_3_3_70B_4bit_HybridRAG_k5"
BARE_70B_K17_ARM = "Bare_Llama_3_3_70B_4bit_FullContext_k17"
CHELATED_70B_ARM = "Llama_3_3_70B_4bit_Chelated"
CHELATED_70B_LABEL = "Llama-3.3-70B (4-bit NF4) Reader + Chelated (Nemotron-30B Extractor)"

LOCKED_70B_MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
LOCKED_70B_QUANT_DISCLOSURE = "4-bit NF4 (bitsandbytes Linear4bit, bfloat16 compute)"

SEVEN_BEIR_DATASETS: Tuple[Tuple[str, str], ...] = (
    ("msmarco", "MSMARCO"),
    ("quora", "Quora"),
    ("fiqa", "FiQA2018"),
    ("scidocs", "SciDocs"),
    ("arguana", "ArguAna"),
    ("scifact", "SciFact"),
    ("nfcorpus", "NFCorpus"),
)

ENTERPRISE_SCALE_SYSTEM_PROMPT = (
    "You are an enterprise verification assistant. Answer the question concisely using ONLY the provided dated records.\n"
    "- Higher-authority official filings, signed security advisories, and postmortems supersede preliminary notes and unverified rumors.\n"
    "- If equal-authority official sources mutually conflict with no authoritative resolution (CONTESTED), or if the verified answer is absent, reply ONLY: ABSTAIN.\n"
    "- Otherwise output ONLY the concise current verified value/identifier (1 short line, no preamble)."
)

_NEGATION_RE = re.compile(
    r"\b(?:not|never|no\s+longer|other\s+than|except|cannot\s+be|isn't|is\s+not|neither|nor)\b",
    re.IGNORECASE,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_exact_binary_footprint(n_docs: int) -> Dict[str, float]:
    """Compute exact binary (`MiB` / `GiB`) sidecar vs full-index rewrite footprints (`L5` Defect #3 fix)."""
    if n_docs <= 0:
        raise ValueError("n_docs must be positive")
    sidecar_bytes = int(n_docs) * SIDECAR_BYTES_PER_DOC
    f32_384_bytes = int(n_docs) * 384 * 4
    f32_1024_bytes = int(n_docs) * 1024 * 4
    f32_4096_bytes = int(n_docs) * 4096 * 4
    return {
        "n_docs": int(n_docs),
        "sidecar_bytes_per_doc": SIDECAR_BYTES_PER_DOC,
        "sidecar_total_bytes": sidecar_bytes,
        "sidecar_mib": round(sidecar_bytes / MIB, 4),
        "full_rewrite_384d_f32_gib": round(f32_384_bytes / GIB, 4),
        "full_rewrite_1024d_f32_gib": round(f32_1024_bytes / GIB, 4),
        "full_rewrite_4096d_f32_gib": round(f32_4096_bytes / GIB, 4),
        "exact_ratio_vs_384d_f32": float(f32_384_bytes) / float(sidecar_bytes),
        "exact_ratio_vs_1024d_f32": float(f32_1024_bytes) / float(sidecar_bytes),
        "exact_ratio_vs_4096d_f32": float(f32_4096_bytes) / float(sidecar_bytes),
    }


def namespace_corpus_and_qrels(
    corpora_ids: Mapping[str, Sequence[str]],
    queries_by_domain: Mapping[str, Mapping[str, str]],
    qrels_by_domain: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> Tuple[List[str], Dict[str, Dict[str, str]], Dict[str, Dict[str, Dict[str, int]]]]:
    """Prefix every doc_id, query_id, and qrels entry with `{domain}:` and assert 100% uniqueness & resolution."""
    global_doc_ids: List[str] = []
    seen_docs: Set[str] = set()
    doc_set_by_dom: Dict[str, Set[str]] = {}

    for dom in sorted(corpora_ids.keys()):
        if ":" in dom:
            raise ValueError(f"Domain prefix cannot contain ':': {dom!r}")
        dom_set: Set[str] = set()
        for raw_did in corpora_ids[dom]:
            gid = f"{dom}:{raw_did}"
            if gid in seen_docs:
                raise ValueError(f"Duplicate global doc_id detected: {gid!r}")
            seen_docs.add(gid)
            dom_set.add(gid)
            global_doc_ids.append(gid)
        doc_set_by_dom[dom] = dom_set

    ns_queries: Dict[str, Dict[str, str]] = {}
    ns_qrels: Dict[str, Dict[str, Dict[str, int]]] = {}
    for dom, qmap in queries_by_domain.items():
        if dom not in doc_set_by_dom:
            raise KeyError(f"Query domain {dom!r} missing from corpora_ids")
        ns_queries[dom] = {}
        ns_qrels[dom] = {}
        raw_qr = qrels_by_domain.get(dom, {})
        for raw_qid, qtext in qmap.items():
            gqid = f"{dom}:{raw_qid}"
            ns_queries[dom][gqid] = str(qtext)
            if raw_qid in raw_qr:
                mapped_rels: Dict[str, int] = {}
                for raw_did, rel in raw_qr[raw_qid].items():
                    if int(rel) <= 0:
                        continue
                    gdid = f"{dom}:{raw_did}"
                    # Exclude ArguAna self-match (`raw_did == raw_qid`)
                    if dom.lower() == "arguana" and str(raw_did) == str(raw_qid):
                        continue
                    if gdid not in doc_set_by_dom[dom]:
                        raise KeyError(f"Unresolved qrels doc_id {gdid!r} in domain {dom!r}")
                    mapped_rels[gdid] = int(rel)
                if mapped_rels:
                    ns_qrels[dom][gqid] = mapped_rels

    return global_doc_ids, ns_queries, ns_qrels


def release_native_heap() -> None:
    """Force Python GC + glibc `malloc_trim(0)` on Linux before loading 70B weights (`L5`)."""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except (OSError, AttributeError) as exc:
            print(f"[D10] glibc heap trim unavailable: {exc}", file=sys.stderr, flush=True)


def mem_available_gib() -> float:
    """Read Linux system-wide available memory from /proc/meminfo."""
    with Path("/proc/meminfo").open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                _, kib, unit = line.split()
                if unit != "kB":
                    raise ValueError(f"Unexpected MemAvailable unit: {unit!r}")
                return int(kib) / (1024 ** 2)
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _resolve_hf_device_map(required_gib: float) -> Any:
    """Place directly on `cuda:0` on Linux UMA (DGX Spark GB10) when `/proc/meminfo` `MemAvailable`
    suffices, avoiding `accelerate` `device_map='auto'` false disk offload caused by file page cache.
    """
    meminfo = Path("/proc/meminfo")
    if sys.platform.startswith("linux") and meminfo.is_file():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    kib = int(line.split()[1])
                    if (kib / (1024 ** 2)) >= float(required_gib):
                        return {"": 0}
                    break
        except OSError:
            pass
    return "auto"


def rss_gib() -> float:
    """Return current process RSS in binary `GiB`."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        kb = float(line.split()[1])
                        return kb / (1024.0 * 1024.0)
        except OSError as exc:
            print(f"[D10] Unable to read /proc/self/status, using psutil: {exc}", file=sys.stderr, flush=True)
    import psutil  # type: ignore
    return float(psutil.Process(os.getpid()).memory_info().rss) / GIB


def l2_normalize_f64(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def compute_chunked_zeromu_csls_sidecar(
    docs_matrix: np.ndarray,
    dev_queries_in_shard_space: np.ndarray,
    *,
    chunk_rows: int = CHUNK_ROWS,
    csls_k: int = CSLS_K,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Streamed `65,536`-row chunked ZeroMu + causal-DEV CSLS sidecar builder (`0` full-matrix float64 copy!).

    Accepts an in-memory `ndarray` or disk-backed `np.memmap` (`float16` or `float32`) of shape `[N, d]`.
    Returns `(mu_d_f64 [d], centered_norms_f32 [N], r_d_f32 [N])`, where `(norms_f32, r_d_f32)` is `8 B/doc`.
    """
    n_docs, dim = int(docs_matrix.shape[0]), int(docs_matrix.shape[1])
    if n_docs == 0:
        raise ValueError("docs_matrix cannot be empty")

    # Pass 1: streaming float64 mean `mu_d`
    sum_vec = np.zeros(dim, dtype=np.float64)
    for start in range(0, n_docs, chunk_rows):
        end = min(n_docs, start + chunk_rows)
        blk = np.asarray(docs_matrix[start:end], dtype=np.float32)
        sum_vec += blk.sum(axis=0, dtype=np.float64)
    mu_d = sum_vec / float(n_docs)
    mu_f32 = mu_d.astype(np.float32)

    # Pre-normalize DEV hubness probes (`[Q_dev, d]`)
    q_dev = l2_normalize_f64(dev_queries_in_shard_space).astype(np.float32)
    q_shifts = (q_dev.astype(np.float64) @ mu_d).astype(np.float32)
    k_eff = min(int(csls_k), int(q_dev.shape[0]))
    if k_eff <= 0:
        raise ValueError("dev_queries_in_shard_space must contain >= 1 query")

    norms_f32 = np.empty(n_docs, dtype=np.float32)
    r_d_f32 = np.empty(n_docs, dtype=np.float32)

    # Pass 2: streaming centered L2 norm + top-k DEV ZeroMu hubness penalty per chunk
    for start in range(0, n_docs, chunk_rows):
        end = min(n_docs, start + chunk_rows)
        blk = np.asarray(docs_matrix[start:end], dtype=np.float32)
        diff = blk - mu_f32
        c_norms = np.linalg.norm(diff, axis=1).astype(np.float32)
        c_norms = np.maximum(c_norms, np.float32(1e-6))
        norms_f32[start:end] = c_norms

        # ZeroMu cosine wrt DEV probes: `(blk @ q_dev.T - q_shifts) / c_norms[:, None]`
        dots = blk @ q_dev.T  # [B, Q_dev]
        zm_sims = (dots - q_shifts.reshape(1, -1)) / c_norms.reshape(-1, 1)
        if k_eff == zm_sims.shape[1]:
            topk_mean = zm_sims.mean(axis=1, dtype=np.float64)
        else:
            part = np.partition(zm_sims, zm_sims.shape[1] - k_eff, axis=1)[:, -k_eff:]
            topk_mean = part.mean(axis=1, dtype=np.float64)
        r_d_f32[start:end] = topk_mean.astype(np.float32)

    return mu_d, norms_f32, r_d_f32


def fit_rectangular_stiefel_bridge(
    src_anchors: np.ndarray,
    tgt_anchors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit centered rectangular Stiefel bridge `W` (`[d_src, d_tgt]`) from paired anchor matrices."""
    src = np.asarray(src_anchors, dtype=np.float64)
    tgt = np.asarray(tgt_anchors, dtype=np.float64)
    if src.shape[0] != tgt.shape[0] or src.shape[0] < 32:
        raise ValueError(f"Expected >= 32 paired anchors, got {src.shape} vs {tgt.shape}")
    mu_src = src.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)
    xc = l2_normalize_f64(src - mu_src)
    yc = l2_normalize_f64(tgt - mu_tgt)
    u, _s, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
    w = u @ vt
    return w, mu_src


def causal_stream_bridge_queries(
    queries_native: np.ndarray,
    w: np.ndarray,
    mu_init: np.ndarray,
    *,
    alpha: float = EMA_ALPHA,
) -> np.ndarray:
    """Strictly causal streaming query projection (`mu_q` updated AFTER scoring each query)."""
    qs = np.asarray(queries_native, dtype=np.float64)
    mu_q = np.asarray(mu_init, dtype=np.float64).copy()
    out = np.empty((qs.shape[0], w.shape[1]), dtype=np.float32)
    for i in range(qs.shape[0]):
        q = qs[i]
        c = q - mu_q
        c_norm = c / max(float(np.linalg.norm(c)), 1e-12)
        mapped = c_norm @ w
        out[i] = (mapped / max(float(np.linalg.norm(mapped)), 1e-12)).astype(np.float32)
        mu_q = (1.0 - alpha) * mu_q + alpha * q
    return out


def pack_epoch_admitted_bitset(admitted_flags: np.ndarray) -> bytes:
    """Pack boolean admission vector of length `N` into `ceil(N / 8)` bytes (`1.13 MiB` @ `9.47M` docs)."""
    bools = np.asarray(admitted_flags, dtype=np.uint8)
    return np.packbits(bools, bitorder="little").tobytes()


def is_doc_admitted_in_bitset(packed_bitset: bytes, doc_idx: int) -> bool:
    byte_pos = doc_idx >> 3
    bit_pos = doc_idx & 7
    if byte_pos < 0 or byte_pos >= len(packed_bitset):
        return False
    return bool((packed_bitset[byte_pos] >> bit_pos) & 1)


def score_dual_shard_shortlists(
    *,
    legacy_ids: Sequence[str],
    legacy_idx: np.ndarray,
    legacy_raw_dots: np.ndarray,
    q_bridged_f32: np.ndarray,
    leg_mu_d: np.ndarray,
    leg_norms_f32: np.ndarray,
    leg_rd_f32: np.ndarray,
    native_ids: Sequence[str],
    native_idx: np.ndarray,
    native_raw_dots: np.ndarray,
    q_native_f32: np.ndarray,
    nat_mu_d: np.ndarray,
    nat_norms_f32: np.ndarray,
    nat_rd_f32: np.ndarray,
    dev_z_stats: Optional[Mapping[str, float]] = None,
    admitted_leg_bitset: Optional[bytes] = None,
    admitted_nat_bitset: Optional[bytes] = None,
    top_k: int = 10,
    csls_k: int = CSLS_K,
) -> Dict[str, List[str]]:
    """Merge dual-shard (`legacy_384d` + `native_1024d`) shortlists under 4 methods with optional epoch bitset fence."""
    # Filter by epoch-pinned admission bitset if provided
    valid_leg = [
        pos for pos, idx in enumerate(legacy_idx.tolist())
        if idx >= 0 and (admitted_leg_bitset is None or is_doc_admitted_in_bitset(admitted_leg_bitset, int(idx)))
    ]
    valid_nat = [
        pos for pos, idx in enumerate(native_idx.tolist())
        if idx >= 0 and (admitted_nat_bitset is None or is_doc_admitted_in_bitset(admitted_nat_bitset, int(idx)))
    ]

    l_idx = legacy_idx[valid_leg].astype(np.int64) if valid_leg else np.empty(0, dtype=np.int64)
    l_dots = legacy_raw_dots[valid_leg].astype(np.float64) if valid_leg else np.empty(0, dtype=np.float64)
    n_idx = native_idx[valid_nat].astype(np.int64) if valid_nat else np.empty(0, dtype=np.int64)
    n_dots = native_raw_dots[valid_nat].astype(np.float64) if valid_nat else np.empty(0, dtype=np.float64)

    l_doc_ids = [legacy_ids[int(i)] for i in l_idx]
    n_doc_ids = [native_ids[int(i)] for i in n_idx]

    # 1. Uncalibrated Raw Dot Merge
    raw_pairs = list(zip(l_doc_ids, l_dots.tolist())) + list(zip(n_doc_ids, n_dots.tolist()))
    raw_pairs.sort(key=lambda kv: (-kv[1], kv[0]))

    # ZeroMu rescoring from 8 B/doc sidecars (`0 B/doc` vector egress!)
    q_leg_shift = float(np.dot(q_bridged_f32.astype(np.float64), leg_mu_d))
    l_zm = (l_dots - q_leg_shift) / np.maximum(leg_norms_f32[l_idx].astype(np.float64), 1e-12) if l_idx.size else l_dots

    q_nat_shift = float(np.dot(q_native_f32.astype(np.float64), nat_mu_d))
    n_zm = (n_dots - q_nat_shift) / np.maximum(nat_norms_f32[n_idx].astype(np.float64), 1e-12) if n_idx.size else n_dots

    # 2. DEV Z-Score Merge on ZeroMu cosines
    stats = dev_z_stats or {"leg_mean": 0.0, "leg_std": 1.0, "nat_mean": 0.0, "nat_std": 1.0}
    l_z = (l_zm - float(stats["leg_mean"])) / max(float(stats["leg_std"]), 1e-6)
    n_z = (n_zm - float(stats["nat_mean"])) / max(float(stats["nat_std"]), 1e-6)
    z_pairs = list(zip(l_doc_ids, l_z.tolist())) + list(zip(n_doc_ids, n_z.tolist()))
    z_pairs.sort(key=lambda kv: (-kv[1], kv[0]))

    # 3. RRF (k=60) over ZeroMu per-shard rankings
    rrf_scores: Dict[str, float] = {}
    l_order = np.argsort(-l_zm, kind="mergesort")
    for rank_pos, p in enumerate(l_order.tolist()):
        did = l_doc_ids[p]
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (60.0 + rank_pos + 1.0)
    n_order = np.argsort(-n_zm, kind="mergesort")
    for rank_pos, p in enumerate(n_order.tolist()):
        did = n_doc_ids[p]
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (60.0 + rank_pos + 1.0)
    rrf_pairs = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))

    # 4. Chelated ZeroMu + Per-Shard CSLS (`2 * s_zm - r_d - r_q`)
    rq_leg = float(np.sort(l_zm)[-min(csls_k, l_zm.size):].mean()) if l_zm.size else 0.0
    rq_nat = float(np.sort(n_zm)[-min(csls_k, n_zm.size):].mean()) if n_zm.size else 0.0
    l_csls = 2.0 * l_zm - leg_rd_f32[l_idx].astype(np.float64) - rq_leg if l_idx.size else l_zm
    n_csls = 2.0 * n_zm - nat_rd_f32[n_idx].astype(np.float64) - rq_nat if n_idx.size else n_zm
    csls_pairs = list(zip(l_doc_ids, l_csls.tolist())) + list(zip(n_doc_ids, n_csls.tolist()))
    csls_pairs.sort(key=lambda kv: (-kv[1], kv[0]))

    def _dedup_topk(pairs: Sequence[Tuple[str, float]]) -> List[str]:
        out_ids: List[str] = []
        seen: Set[str] = set()
        for did, _sc in pairs:
            if did not in seen:
                seen.add(did)
                out_ids.append(did)
                if len(out_ids) >= top_k:
                    break
        return out_ids

    return {
        "Uncalibrated_Raw_Merge": _dedup_topk(raw_pairs),
        "DEV_ZScore_ZeroMu": _dedup_topk(z_pairs),
        "RRF_k60_ZeroMu": _dedup_topk(rrf_pairs),
        "Chelated_ZeroMu_CSLS_8B": _dedup_topk(csls_pairs),
    }


def exact_chunk_top10(
    queries: np.ndarray,
    docs: np.ndarray,
    doc_ids: Sequence[str],
    *,
    admitted_flags: Optional[np.ndarray] = None,
    sidecar: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    chunk_rows: int = CHUNK_ROWS,
) -> List[List[Tuple[float, str]]]:
    """Scan a shard for exact raw-IP or ZeroMu/CSLS top 10, ordered by (-score, doc_id)."""
    ranked: List[List[Tuple[float, str]]] = [[] for _ in queries]
    if sidecar is not None:
        mu_d, norms, r_d = sidecar
        shifts = queries.astype(np.float64) @ mu_d
        best_zeromu = np.empty((len(queries), 0), dtype=np.float64)

    for start in range(0, len(docs), chunk_rows):
        end = min(len(docs), start + chunk_rows)
        rows = np.arange(start, end)
        raw_dots = queries @ np.asarray(docs[start:end], dtype=np.float32).T
        if admitted_flags is not None:
            admitted = admitted_flags[start:end].astype(bool)
            rows = rows[admitted]
            raw_dots = raw_dots[:, admitted]
        if rows.size == 0:
            continue

        scores = raw_dots
        if sidecar is not None:
            zeromu = (raw_dots - shifts[:, None]) / np.maximum(norms[rows].astype(np.float64), 1e-12)
            combined = np.concatenate((best_zeromu, zeromu), axis=1)
            cutoff = max(0, combined.shape[1] - CSLS_K)
            best_zeromu = np.partition(combined, cutoff, axis=1)[:, cutoff:].copy()
            scores = 2.0 * zeromu - r_d[rows].astype(np.float64)

        cutoff = max(0, rows.size - 10)
        for query_row, score_row in enumerate(scores):
            floor = np.partition(score_row, cutoff)[cutoff]
            candidates = (
                (-float(score_row[i]), doc_ids[int(rows[i])])
                for i in np.flatnonzero(score_row >= floor)
            )
            ranked[query_row] = nsmallest(10, (*ranked[query_row], *nsmallest(10, candidates)))

    if sidecar is not None and best_zeromu.shape[1]:
        # The query penalty is constant within a shard and only affects the cross-shard merge.
        return [
            [(neg_score + float(penalty), doc_id) for neg_score, doc_id in hits]
            for hits, penalty in zip(ranked, best_zeromu.mean(axis=1))
        ]
    return ranked


def require_fenced_top10(fnc: Sequence[str], query_id: str) -> None:
    assert len(fnc) == 10 and len(set(fnc)) == 10, (
        f"Expected 10 distinct fenced hits for {query_id}, got {len(fnc)} hits and {len(set(fnc))} distinct IDs"
    )


def score_strict_suite_a_prediction(
    prediction: str,
    expected_answer: str,
    stale_traps: Sequence[str],
    abstention_required: bool,
) -> Dict[str, bool]:
    """Deterministic canonical grader for `Suite A (N=60)` (`strict_exact_match`, `stale_leak`, `false_commit_on_u`)."""
    raw = (prediction or "").strip()
    low = raw.lower()
    abstain_hit = (
        low == "abstain"
        or low.startswith("abstain")
        or "cannot determine" in low
        or "unresolved conflict" in low
        or "mutually conflict" in low
        or "insufficient verified" in low
    )
    has_stale = any(t.strip().lower() in low for t in stale_traps if t and t.strip())
    if abstention_required:
        # Must abstain AND not commit to any candidate trap value
        exact_ok = bool(abstain_hit and not has_stale)
        return {
            "strict_exact_match": exact_ok,
            "stale_leak": has_stale,
            "false_commit_on_u": not abstain_hit,
            "abstained": abstain_hit,
        }

    exp = expected_answer.strip().lower()
    has_exp = bool(exp and exp in low)
    has_neg = bool(_NEGATION_RE.search(raw))
    exact_ok = bool(has_exp and not has_stale and not abstain_hit and not has_neg)
    return {
        "strict_exact_match": exact_ok,
        "stale_leak": has_stale,
        "false_commit_on_u": False,
        "abstained": abstain_hit,
    }


def compute_cohens_kappa_binary(labels_1: Sequence[int], labels_2: Sequence[int]) -> float:
    """Compute unweighted Cohen's kappa between two binary `{0, 1}` judge vectors."""
    if len(labels_1) != len(labels_2) or len(labels_1) == 0:
        raise ValueError("Judge vectors must have equal positive length")
    a = np.asarray(labels_1, dtype=np.int64)
    b = np.asarray(labels_2, dtype=np.int64)
    p_o = float(np.mean(a == b))
    p_a1 = float(np.mean(a == 1))
    p_b1 = float(np.mean(b == 1))
    p_e = p_a1 * p_b1 + (1.0 - p_a1) * (1.0 - p_b1)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(1.0 - p_o) < 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Worker Subcommands for Distributed 4-GPU Encoding, Spark #1 FAISS, & Spark #2 70B
# ---------------------------------------------------------------------------
def cmd_encode_msmarco_shard(args: argparse.Namespace) -> None:
    """Encode a contiguous slice `[start_idx, end_idx)` of `BeIR/msmarco` into `fp16` `384-d` (and optional `1024-d`)."""
    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    out_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = f"shard_{args.shard_id:02d}_of_{args.num_shards:02d}"
    done_flag = out_dir / f"{shard_tag}.done.json"
    if done_flag.exists():
        print(f"[SKIP] {shard_tag} already encoded at {done_flag}", flush=True)
        return

    t0 = time.perf_counter()
    ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    total_n = len(ds)
    per_shard = math.ceil(total_n / args.num_shards)
    s_idx = args.shard_id * per_shard
    e_idx = min(total_n, (args.shard_id + 1) * per_shard)
    sub = ds.select(range(s_idx, e_idx))
    n_sub = len(sub)
    print(f"[ENCODE] {shard_tag}: rows [{s_idx}, {e_idx}) = {n_sub} / {total_n}", flush=True)

    ids_path = out_dir / f"{shard_tag}_ids.json"
    emb384_path = out_dir / f"{shard_tag}_e384_fp16.npy"

    doc_ids: List[str] = []
    texts: List[str] = []
    for row in sub:
        doc_ids.append(str(row["_id"]))
        title = str(row.get("title") or "").strip()
        txt = str(row.get("text") or "").strip()
        texts.append(f"{title} {txt}".strip() if title else txt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_384 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    model_384.max_seq_length = 160
    if device == "cuda":
        model_384.half()

    mm384 = np.lib.format.open_memmap(emb384_path, mode="w+", dtype=np.float16, shape=(n_sub, 384))
    batch_sz = int(args.batch_size)
    step_chunk = 65_536
    for c_start in range(0, n_sub, step_chunk):
        c_end = min(n_sub, c_start + step_chunk)
        vecs = model_384.encode(
            texts[c_start:c_end],
            batch_size=batch_sz,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        mm384[c_start:c_end] = vecs.astype(np.float16)
        print(f"  [{shard_tag} 384-d] {c_end}/{n_sub} ({time.perf_counter() - t0:.1f}s)", flush=True)
    mm384.flush()
    del mm384, model_384
    if device == "cuda":
        torch.cuda.empty_cache()

    # Deterministic 20% upgraded subset (`hash(msmarco:{id}) % 5 == 0`) encoded in 1024-d (`BAAI/bge-large-en-v1.5`)
    upg_local_idx: List[int] = []
    for idx_pos, did in enumerate(doc_ids):
        h = int.from_bytes(hashlib.sha256(f"msmarco:{did}|upg20".encode("utf-8")).digest()[:4], "little")
        if (h % 5) == 0:
            upg_local_idx.append(idx_pos)

    emb1024_path = out_dir / f"{shard_tag}_upg20_e1024_fp16.npy"
    upg_idx_path = out_dir / f"{shard_tag}_upg20_local_idx.npy"
    np.save(upg_idx_path, np.asarray(upg_local_idx, dtype=np.int32))

    model_1024 = SentenceTransformer(args.upgrade_model_1024, device=device)
    model_1024.max_seq_length = 160
    if device == "cuda":
        model_1024.half()

    n_upg = len(upg_local_idx)
    mm1024 = np.lib.format.open_memmap(emb1024_path, mode="w+", dtype=np.float16, shape=(n_upg, 1024))
    upg_texts = [texts[i] for i in upg_local_idx]
    for c_start in range(0, n_upg, 32_768):
        c_end = min(n_upg, c_start + 32_768)
        vecs1024 = model_1024.encode(
            upg_texts[c_start:c_end],
            batch_size=max(128, batch_sz // 4),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        mm1024[c_start:c_end] = vecs1024.astype(np.float16)
        print(f"  [{shard_tag} 1024-d upg20] {c_end}/{n_upg} ({time.perf_counter() - t0:.1f}s)", flush=True)
    mm1024.flush()
    del mm1024, model_1024
    if device == "cuda":
        torch.cuda.empty_cache()

    ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")
    meta = {
        "shard_tag": shard_tag,
        "total_msmarco_docs": total_n,
        "start_idx": s_idx,
        "end_idx": e_idx,
        "n_docs": n_sub,
        "n_upgraded_1024d": n_upg,
        "upgrade_model_1024": args.upgrade_model_1024,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "e384_sha256": sha256_file(emb384_path),
        "e1024_upg_sha256": sha256_file(emb1024_path),
    }
    write_json_atomic(done_flag, meta)
    print(f"[DONE] {shard_tag} in {meta['wall_seconds']}s", flush=True)


def cmd_run_part_a_cutover(args: argparse.Namespace) -> None:
    """Execute Part A: Full `8,841,823` `BeIR/msmarco` + `623,716` 6-BEIR = `9,465,539` Real-Doc Cutover (`Spark #1`)."""
    pre_run_mem_available_gib = mem_available_gib() if sys.platform.startswith("linux") else None
    if pre_run_mem_available_gib is not None:
        assert pre_run_mem_available_gib >= 48.0, (
            f"D10-A needs at least 48 GiB MemAvailable, found {pre_run_mem_available_gib:.2f} GiB"
        )

    import faiss  # type: ignore
    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    t_start = time.perf_counter()
    work_dir = Path(args.work_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify all 4 MSMARCO shards are present (`8,841,823` real passages, 0% subsampling)
    msmarco_ids: List[str] = []
    shard_metas: List[Dict[str, Any]] = []
    shard_384_paths: List[Path] = []
    shard_upg_idx_paths: List[Path] = []
    shard_1024_paths: List[Path] = []

    for sid in range(int(args.num_shards)):
        stag = f"shard_{sid:02d}_of_{args.num_shards:02d}"
        meta_p = work_dir / f"{stag}.done.json"
        if not meta_p.exists():
            raise FileNotFoundError(f"Missing MSMARCO shard metadata: {meta_p}")
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        shard_metas.append(meta)
        s_ids = json.loads((work_dir / f"{stag}_ids.json").read_text(encoding="utf-8"))
        msmarco_ids.extend(s_ids)
        shard_384_paths.append(work_dir / f"{stag}_e384_fp16.npy")
        shard_upg_idx_paths.append(work_dir / f"{stag}_upg20_local_idx.npy")
        shard_1024_paths.append(work_dir / f"{stag}_upg20_e1024_fp16.npy")

    n_msmarco = len(msmarco_ids)
    assert n_msmarco >= 5_000_000, f"Expected >= 5M MSMARCO docs, got {n_msmarco}"

    # 2. Load MSMARCO queries + dev qrels (`BeIR/msmarco` + `BeIR/msmarco-qrels`)
    q_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    raw_msmarco_queries = {str(r["_id"]): str(r["text"]) for r in q_ds}
    qr_ds = load_dataset("BeIR/msmarco-qrels", split="validation")
    raw_msmarco_qrels: Dict[str, Dict[str, int]] = {}
    for r in qr_ds:
        qid_s = str(r["query-id"])
        did_s = str(r["corpus-id"])
        sc_i = int(r["score"])
        if sc_i > 0 and qid_s in raw_msmarco_queries:
            raw_msmarco_qrels.setdefault(qid_s, {})[did_s] = sc_i

    # Deterministic DEV (fit/calibration) vs REPORT split on MSMARCO dev queries (cap REPORT at 500 for fast exact CIs)
    all_ms_qids = sorted(raw_msmarco_qrels.keys())
    ms_split = split_query_ids(all_ms_qids, salt=QUERY_SPLIT_SALT)
    ms_dev_qids = sorted(ms_split["DEVELOPMENT"])[:200]
    ms_rep_qids = sorted(ms_split["REPORT"])[:500]

    # 3. Also load cached 6-BEIR corpora (`SciFact`, `NFCorpus`, `FiQA2018`, `ArguAna`, `SCIDOCS`, `Quora`)
    from scripts.night1_common import load_beir_dataset

    extra_beir_domains: Tuple[Tuple[str, str], ...] = (
        ("scifact", "SciFact"),
        ("nfcorpus", "NFCorpus"),
        ("fiqa", "FiQA2018"),
        ("arguana", "ArguAna"),
        ("scidocs", "SCIDOCS"),
        ("quora", "Quora"),
    )
    corpora_ids_map: Dict[str, List[str]] = {"msmarco": msmarco_ids}
    queries_by_dom: Dict[str, Dict[str, str]] = {
        "msmarco": {q: raw_msmarco_queries[q] for q in (ms_dev_qids + ms_rep_qids)}
    }
    qrels_by_dom: Dict[str, Dict[str, Dict[str, int]]] = {
        "msmarco": {q: raw_msmarco_qrels[q] for q in (ms_dev_qids + ms_rep_qids)}
    }
    beir_texts_by_dom: Dict[str, List[str]] = {}

    for key_low, bdom in extra_beir_domains:
        d_ids, d_texts, q_ids, q_texts, qrels_d, _qsha = load_beir_dataset(bdom)
        corpora_ids_map[key_low] = list(d_ids)
        beir_texts_by_dom[key_low] = list(d_texts)
        q_lookup = dict(zip(q_ids, q_texts))
        valid_q_keys = sorted(q for q in q_ids if q in qrels_d and qrels_d[q])
        if len(valid_q_keys) > 500:
            valid_q_keys = valid_q_keys[:500]
        queries_by_dom[key_low] = {q: q_lookup[q] for q in valid_q_keys}
        qrels_by_dom[key_low] = {
            q: {did: int(round(sc)) for did, sc in qrels_d[q].items() if float(sc) > 0.0}
            for q in valid_q_keys
        }

    global_doc_ids, ns_queries, ns_qrels = namespace_corpus_and_qrels(
        corpora_ids_map, queries_by_dom, qrels_by_dom
    )
    sampled_oracle_qids = sorted(ns_queries["scifact"])[:FLAT_ORACLE_SAMPLE_SIZE]
    assert len(sampled_oracle_qids) == FLAT_ORACLE_SAMPLE_SIZE, "Expected 50 SciFact Flat-oracle query IDs"
    sampled_oracle_qids_sha256 = sha256_bytes(json.dumps(sampled_oracle_qids).encode("utf-8"))
    total_merged_docs = len(global_doc_ids)
    print(f"[D10-A] Verified {total_merged_docs:,} unique real global doc IDs (MSMARCO={n_msmarco:,})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_384 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    enc_384.max_seq_length = 160
    if device == "cuda":
        enc_384.half()
    enc_1024 = SentenceTransformer(args.upgrade_model_1024, device=device)
    enc_1024.max_seq_length = 160
    if device == "cuda":
        enc_1024.half()

    # Encode 6-BEIR docs (`623,716` docs) if not already cached in work_dir
    beir_384_cache = work_dir / "extra_6beir_e384_fp16.npy"
    beir_1024_upg_cache = work_dir / "extra_6beir_upg20_e1024_fp16.npy"
    beir_upg_idx_cache = work_dir / "extra_6beir_upg20_idx.npy"
    beir_ids_ordered: List[str] = []
    beir_texts_ordered: List[str] = []
    for kl, _bdom in extra_beir_domains:
        for did, txt in zip(corpora_ids_map[kl], beir_texts_by_dom[kl]):
            beir_ids_ordered.append(f"{kl}:{did}")
            beir_texts_ordered.append(txt)

    n_extra = len(beir_ids_ordered)
    if not (beir_384_cache.exists() and beir_1024_upg_cache.exists() and beir_upg_idx_cache.exists()):
        mm_b384 = np.lib.format.open_memmap(beir_384_cache, mode="w+", dtype=np.float16, shape=(n_extra, 384))
        for cs in range(0, n_extra, 32_768):
            ce = min(n_extra, cs + 32_768)
            mm_b384[cs:ce] = enc_384.encode(
                beir_texts_ordered[cs:ce],
                batch_size=int(args.batch_size),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float16)
        mm_b384.flush()
        del mm_b384

        upg_b_idx = [
            idx_p for idx_p, gdid in enumerate(beir_ids_ordered)
            if (int.from_bytes(hashlib.sha256(f"{gdid}|upg20".encode("utf-8")).digest()[:4], "little") % 5) == 0
        ]
        np.save(beir_upg_idx_cache, np.asarray(upg_b_idx, dtype=np.int32))
        mm_b1024 = np.lib.format.open_memmap(
            beir_1024_upg_cache, mode="w+", dtype=np.float16, shape=(len(upg_b_idx), 1024)
        )
        upg_b_txts = [beir_texts_ordered[i] for i in upg_b_idx]
        for cs in range(0, len(upg_b_idx), 16_384):
            ce = min(len(upg_b_idx), cs + 16_384)
            mm_b1024[cs:ce] = enc_1024.encode(
                upg_b_txts[cs:ce],
                batch_size=max(128, int(args.batch_size) // 4),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float16)
        mm_b1024.flush()
        del mm_b1024
    del beir_texts_by_dom, beir_texts_ordered
    gc.collect()

    # Encode all domain queries in both 384-d and 1024-d
    q_embs_384: Dict[str, Tuple[List[str], np.ndarray]] = {}
    q_embs_1024: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for dom, qmap in ns_queries.items():
        q_list = sorted(qmap.keys())
        q_txts = [qmap[q] for q in q_list]
        v384 = enc_384.encode(q_txts, batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        v1024 = enc_1024.encode(q_txts, batch_size=128, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        q_embs_384[dom] = (q_list, v384)
        q_embs_1024[dom] = (q_list, v1024)

    del enc_384, enc_1024
    if device == "cuda":
        torch.cuda.empty_cache()

    # 4. Assemble contiguous disk-backed memmaps for `legacy_384d` (80%) and `native_1024d` (20%)
    # First collect paired anchors (`n=2,000` qrels-blind anchors from upgraded docs) to fit Stiefel bridge `1024 -> 384`
    leg_ids: List[str] = []
    nat_ids: List[str] = []
    anchor_384_list: List[np.ndarray] = []
    anchor_1024_list: List[np.ndarray] = []

    # Count exact legacy/native sizes across MSMARCO + 6-BEIR
    shard_upg_sets: List[np.ndarray] = [np.load(p) for p in shard_upg_idx_paths]
    extra_upg_arr = np.load(beir_upg_idx_cache)
    n_nat_msmarco = sum(int(a.shape[0]) for a in shard_upg_sets)
    n_leg_msmarco = n_msmarco - n_nat_msmarco
    n_nat_total = n_nat_msmarco + int(extra_upg_arr.shape[0])
    n_leg_total = n_leg_msmarco + (n_extra - int(extra_upg_arr.shape[0]))

    leg_mm_path = work_dir / "merged_leg80_e384_fp16.npy"
    nat_mm_path = work_dir / "merged_nat20_e1024_fp16.npy"
    leg_mm = np.lib.format.open_memmap(leg_mm_path, mode="w+", dtype=np.float16, shape=(n_leg_total, 384))
    nat_mm = np.lib.format.open_memmap(nat_mm_path, mode="w+", dtype=np.float16, shape=(n_nat_total, 1024))

    leg_cursor = 0
    nat_cursor = 0
    ms_offset = 0
    for sid in range(int(args.num_shards)):
        stag = f"shard_{sid:02d}_of_{args.num_shards:02d}"
        s_ids = json.loads((work_dir / f"{stag}_ids.json").read_text(encoding="utf-8"))
        s_384 = np.load(shard_384_paths[sid], mmap_mode="r")
        s_1024 = np.load(shard_1024_paths[sid], mmap_mode="r")
        upg_local = shard_upg_sets[sid]
        upg_mask = np.zeros(len(s_ids), dtype=bool)
        upg_mask[upg_local] = True

        # Sample up to 400 strictly qrels-independent anchors per shard (`1,600` total from MSMARCO)
        pool_n = min(len(upg_local), 8192)
        cand_pos = select_qrels_independent_anchors(
            [f"msmarco:{s_ids[int(upg_local[pos])]}" for pos in range(pool_n)],
            anchor_frac=1.0,
            seed=202600 + sid,
            max_anchors=400,
        ).tolist()
        if cand_pos:
            anchor_384_list.append(np.asarray(s_384[upg_local[cand_pos]], dtype=np.float32))
            anchor_1024_list.append(np.asarray(s_1024[cand_pos], dtype=np.float32))

        leg_local = np.flatnonzero(~upg_mask)
        n_l = int(leg_local.shape[0])
        n_u = int(upg_local.shape[0])
        for cs in range(0, n_l, CHUNK_ROWS):
            ce = min(n_l, cs + CHUNK_ROWS)
            leg_mm[leg_cursor + cs : leg_cursor + ce] = s_384[leg_local[cs:ce]]
        for cs in range(0, n_u, CHUNK_ROWS):
            ce = min(n_u, cs + CHUNK_ROWS)
            nat_mm[nat_cursor + cs : nat_cursor + ce] = s_1024[cs:ce]

        leg_ids.extend(f"msmarco:{s_ids[int(i)]}" for i in leg_local.tolist())
        nat_ids.extend(f"msmarco:{s_ids[int(i)]}" for i in upg_local.tolist())
        leg_cursor += n_l
        nat_cursor += n_u
        ms_offset += len(s_ids)
        del s_384, s_1024

    # Record MSMARCO-only boundary (`leg_msmarco_end = n_leg_msmarco`, `nat_msmarco_end = n_nat_msmarco`)
    b_384 = np.load(beir_384_cache, mmap_mode="r")
    b_1024 = np.load(beir_1024_upg_cache, mmap_mode="r")
    b_upg_mask = np.zeros(n_extra, dtype=bool)
    b_upg_mask[extra_upg_arr] = True
    b_leg_local = np.flatnonzero(~b_upg_mask)

    # Also add 400 qrels-independent anchors from 6-BEIR
    beir_upg_ids = [beir_ids_ordered[int(i)] for i in extra_upg_arr.tolist()]
    b_anc_pos_arr = select_qrels_independent_anchors(
        beir_upg_ids,
        anchor_frac=1.0,
        seed=202610,
        max_anchors=400,
    )
    b_anc_pos = b_anc_pos_arr.tolist()
    if b_anc_pos:
        anchor_384_list.append(np.asarray(b_384[extra_upg_arr[b_anc_pos]], dtype=np.float32))
        anchor_1024_list.append(np.asarray(b_1024[b_anc_pos], dtype=np.float32))

    leg_mm[leg_cursor : leg_cursor + len(b_leg_local)] = b_384[b_leg_local]
    nat_mm[nat_cursor : nat_cursor + len(extra_upg_arr)] = b_1024[:]
    del b_384, b_1024
    leg_ids.extend(beir_ids_ordered[int(i)] for i in b_leg_local.tolist())
    nat_ids.extend(beir_ids_ordered[int(i)] for i in extra_upg_arr.tolist())
    leg_mm.flush()
    nat_mm.flush()
    release_native_heap()

    assert len(leg_ids) == n_leg_total and len(nat_ids) == n_nat_total
    assert len(leg_ids) + len(nat_ids) == total_merged_docs

    # 5. Fit centered Stiefel bridge `1024 -> 384` (`n=2,000` anchors = `0.021%` of `9.47M` corpus!)
    anc_384 = np.vstack(anchor_384_list)
    anc_1024 = np.vstack(anchor_1024_list)
    w_bridge, mu_q_init = fit_rectangular_stiefel_bridge(anc_1024, anc_384)
    ortho_err = float(np.max(np.abs(w_bridge.T @ w_bridge - np.eye(384))))

    # DEV query set in shard spaces (`ms_dev_qids` mapped to `msmarco:qid`)
    ms_q_order, ms_q1024_mat = q_embs_1024["msmarco"]
    ms_qid_to_row = {qid: r for r, qid in enumerate(ms_q_order)}
    dev_rows = [ms_qid_to_row[f"msmarco:{q}"] for q in ms_dev_qids if f"msmarco:{q}" in ms_qid_to_row]
    q_dev_1024 = ms_q1024_mat[dev_rows]
    q_dev_bridged_384 = causal_stream_bridge_queries(q_dev_1024, w_bridge, mu_q_init)

    # Streamed `8 B/doc` sidecars (`65,536`-row chunks, zero float64 copy!)
    t_sc = time.perf_counter()
    leg_mu_d, leg_norms_f32, leg_rd_f32 = compute_chunked_zeromu_csls_sidecar(leg_mm, q_dev_bridged_384)
    nat_mu_d, nat_norms_f32, nat_rd_f32 = compute_chunked_zeromu_csls_sidecar(nat_mm, q_dev_1024)
    sidecar_build_s = time.perf_counter() - t_sc

    # 6. Build FAISS `IndexHNSWSQ(d, QT_fp16, M=32, METRIC_INNER_PRODUCT)` on BOTH standalone 8.84M MSMARCO & 9.47M Merged!
    faiss.omp_set_num_threads(min(16, os.cpu_count() or 8))
    faiss_factory_desc = "IndexHNSWSQ(QT_fp16, M=32, efConstruction=128, efSearch=512, METRIC_INNER_PRODUCT)"

    def _build_hnsw_sq_fp16(mm_src: np.ndarray, n_rows: int, dim: int) -> Any:
        idx = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_fp16, 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 128
        idx.hnsw.efSearch = 512
        # Train SQ fp16 (trivial min/max) on first chunk
        train_slice = np.asarray(mm_src[: min(n_rows, CHUNK_ROWS)], dtype=np.float32)
        idx.train(train_slice)
        for s in range(0, n_rows, CHUNK_ROWS):
            e = min(n_rows, s + CHUNK_ROWS)
            idx.add(np.asarray(mm_src[s:e], dtype=np.float32))
        return idx

    print(f"[D10-A] Building FAISS {faiss_factory_desc} over {total_merged_docs:,} real docs...", flush=True)
    t_idx = time.perf_counter()
    idx_leg = _build_hnsw_sq_fp16(leg_mm, n_leg_total, 384)
    idx_nat = _build_hnsw_sq_fp16(nat_mm, n_nat_total, 1024)
    index_build_s = time.perf_counter() - t_idx
    peak_rss = rss_gib()
    print(f"[D10-A] Built dual FAISS indexes in {index_build_s:.1f}s (RSS={peak_rss:.2f} GiB)", flush=True)
    assert peak_rss <= 44.0, f"RSS exceeded 44 GiB safety cap: {peak_rss:.2f} GiB"

    # Calibrate DEV Z-Score mean/std from `q_dev_1024` shortlists
    d_leg_dots, d_leg_i = idx_leg.search(q_dev_bridged_384, 64)
    d_nat_dots, d_nat_i = idx_nat.search(q_dev_1024, 64)
    dev_leg_zm_vals: List[float] = []
    dev_nat_zm_vals: List[float] = []
    for qi in range(q_dev_1024.shape[0]):
        ls = float(np.dot(q_dev_bridged_384[qi].astype(np.float64), leg_mu_d))
        ns = float(np.dot(q_dev_1024[qi].astype(np.float64), nat_mu_d))
        vi_l = d_leg_i[qi][d_leg_i[qi] >= 0]
        vi_n = d_nat_i[qi][d_nat_i[qi] >= 0]
        if vi_l.size:
            dev_leg_zm_vals.extend(((d_leg_dots[qi][: vi_l.size] - ls) / leg_norms_f32[vi_l]).tolist())
        if vi_n.size:
            dev_nat_zm_vals.extend(((d_nat_dots[qi][: vi_n.size] - ns) / nat_norms_f32[vi_n]).tolist())

    dev_z_stats = {
        "leg_mean": float(np.mean(dev_leg_zm_vals)),
        "leg_std": float(np.std(dev_leg_zm_vals)),
        "nat_mean": float(np.mean(dev_nat_zm_vals)),
        "nat_std": float(np.std(dev_nat_zm_vals)),
    }

    # Score the frozen SciFact sample against the entire native shard.
    sf_qids, sf_q1024 = q_embs_1024["scifact"]
    sf_qid_to_row = {qid: row for row, qid in enumerate(sf_qids)}
    sample_q = sf_q1024[[sf_qid_to_row[qid] for qid in sampled_oracle_qids]]
    _, hnsw_i = idx_nat.search(sample_q, 10)
    flat_top10 = exact_chunk_top10(sample_q, nat_mm, nat_ids)
    ann_recalls = []
    for ann_rows, flat_hits in zip(hnsw_i, flat_top10):
        assert len(flat_hits) == 10, "Expected 10 exact Flat-oracle hits"
        ann_ids = {nat_ids[int(i)] for i in ann_rows if i >= 0}
        ann_recalls.append(len(ann_ids.intersection(did for _, did in flat_hits)) / 10.0)
    ann_vs_flat_recall10 = float(np.mean(ann_recalls))
    assert ann_vs_flat_recall10 >= FLAT_ORACLE_RECALL_FLOOR, (
        f"ANN vs Flat recall@10 {ann_vs_flat_recall10:.4f} is below {FLAT_ORACLE_RECALL_FLOOR:.2f}"
    )

    # 7. Evaluate per-domain unpooled `nDCG@10` & `Recall@10` across all 7 real BEIR domains + Standalone 8.84M MSMARCO!
    domain_results: Dict[str, Any] = {}
    search_latencies_ms: List[float] = []

    # Standalone 8.84M MSMARCO admission bitsets (masking out the 623,716 extra BEIR docs gives exact 8,841,823 MSMARCO!)
    ms_only_leg_flags = np.zeros(n_leg_total, dtype=np.uint8)
    ms_only_leg_flags[:n_leg_msmarco] = 1
    ms_only_nat_flags = np.zeros(n_nat_total, dtype=np.uint8)
    ms_only_nat_flags[:n_nat_msmarco] = 1
    ms_only_leg_bset = pack_epoch_admitted_bitset(ms_only_leg_flags)
    ms_only_nat_bset = pack_epoch_admitted_bitset(ms_only_nat_flags)

    eval_modes: List[Tuple[str, str, Optional[bytes], Optional[bytes]]] = [
        ("msmarco_standalone_8_84m", "msmarco", ms_only_leg_bset, ms_only_nat_bset),
    ] + [(f"merged_9_47m_{dom_k}", dom_k, None, None) for dom_k, _label in SEVEN_BEIR_DATASETS]

    for mode_name, dom_k, leg_bset, nat_bset in eval_modes:
        q_order, q1024_all = q_embs_1024[dom_k]
        if dom_k == "msmarco":
            rep_set = {f"msmarco:{q}" for q in ms_rep_qids}
            eval_rows = [r for r, qid in enumerate(q_order) if qid in rep_set]
        else:
            eval_rows = [r for r, qid in enumerate(q_order) if qid in ns_qrels[dom_k]]

        q_eval_1024 = q1024_all[eval_rows]
        q_eval_ids = [q_order[r] for r in eval_rows]
        q_eval_bridged = causal_stream_bridge_queries(q_eval_1024, w_bridge, mu_q_init)

        per_arm_ndcg: Dict[str, Dict[str, float]] = {
            "Uncalibrated_Raw_Merge": {},
            "DEV_ZScore_ZeroMu": {},
            "RRF_k60_ZeroMu": {},
            "Chelated_ZeroMu_CSLS_8B": {},
        }
        per_arm_rec10: Dict[str, List[float]] = {k: [] for k in per_arm_ndcg}

        for qi, qid in enumerate(q_eval_ids):
            t_q = time.perf_counter()
            ld, li = idx_leg.search(q_eval_bridged[qi : qi + 1], 64)
            nd, ni = idx_nat.search(q_eval_1024[qi : qi + 1], 64)
            merged_lists = score_dual_shard_shortlists(
                legacy_ids=leg_ids,
                legacy_idx=li[0],
                legacy_raw_dots=ld[0],
                q_bridged_f32=q_eval_bridged[qi],
                leg_mu_d=leg_mu_d,
                leg_norms_f32=leg_norms_f32,
                leg_rd_f32=leg_rd_f32,
                native_ids=nat_ids,
                native_idx=ni[0],
                native_raw_dots=nd[0],
                q_native_f32=q_eval_1024[qi],
                nat_mu_d=nat_mu_d,
                nat_norms_f32=nat_norms_f32,
                nat_rd_f32=nat_rd_f32,
                dev_z_stats=dev_z_stats,
                admitted_leg_bitset=leg_bset,
                admitted_nat_bitset=nat_bset,
                top_k=10,
            )
            if mode_name == "merged_9_47m_msmarco":
                search_latencies_ms.append((time.perf_counter() - t_q) * 1000.0)

            rel_map = ns_qrels[dom_k][qid]
            n_pos = max(1, min(10, len(rel_map)))
            for arm_k, top_ids in merged_lists.items():
                per_arm_ndcg[arm_k][qid] = float(ndcg_at_k(top_ids, rel_map, k=10))
                hits_pos = sum(1 for did in top_ids[:10] if rel_map.get(did, 0) > 0)
                per_arm_rec10[arm_k].append(float(hits_pos) / float(n_pos))

        c_vs_raw = paired_bootstrap_wave6(
            per_arm_ndcg["Chelated_ZeroMu_CSLS_8B"], per_arm_ndcg["Uncalibrated_Raw_Merge"], expected_ids=q_eval_ids
        )
        c_vs_z = paired_bootstrap_wave6(
            per_arm_ndcg["Chelated_ZeroMu_CSLS_8B"], per_arm_ndcg["DEV_ZScore_ZeroMu"], expected_ids=q_eval_ids
        )
        c_vs_rrf = paired_bootstrap_wave6(
            per_arm_ndcg["Chelated_ZeroMu_CSLS_8B"], per_arm_ndcg["RRF_k60_ZeroMu"], expected_ids=q_eval_ids
        )
        domain_results[mode_name] = {
            "n_queries": len(q_eval_ids),
            "means_ndcg_at_10": {k: round(float(np.mean(list(v.values()))), 5) for k, v in per_arm_ndcg.items()},
            "means_recall_at_10": {k: round(float(np.mean(v)), 5) for k, v in per_arm_rec10.items()},
            "contrasts": {
                "chelated_minus_uncalibrated_raw": c_vs_raw,
                "chelated_minus_dev_zscore": c_vs_z,
                "chelated_minus_rrf_k60": c_vs_rrf,
            },
        }

    # 8. `9.47M`-Scale `EK5 v3` Epoch-Pinned Retraction Fence Proof (`200` MSMARCO top-1 hits retracted via `REFUTE`)
    retract_store = StratifiedEvidenceStore(strict_dag_supersede=False)
    retraction_qids = [f"msmarco:{q}" for q in ms_rep_qids[:200]]
    sampled_retraction_qids = retraction_qids[:RETRACTION_ORACLE_SAMPLE_SIZE]
    assert len(sampled_retraction_qids) == RETRACTION_ORACLE_SAMPLE_SIZE, "Expected 25 retraction oracle queries"
    q_ret_1024 = ms_q1024_mat[[ms_qid_to_row[qid] for qid in retraction_qids]]
    q_ret_bridged = causal_stream_bridge_queries(q_ret_1024, w_bridge, mu_q_init)
    leg_adm_flags = np.ones(n_leg_total, dtype=np.uint8)
    nat_adm_flags = np.ones(n_nat_total, dtype=np.uint8)

    retracted_target_ids: Set[str] = set()
    ret_events: List[StratifiedEvent] = []
    ret_edges: List[EdgeAttestation] = []
    for qi in range(q_ret_1024.shape[0]):
        ld, li = idx_leg.search(q_ret_bridged[qi : qi + 1], 10)
        nd, ni = idx_nat.search(q_ret_1024[qi : qi + 1], 10)
        pre_hits = score_dual_shard_shortlists(
            legacy_ids=leg_ids, legacy_idx=li[0], legacy_raw_dots=ld[0], q_bridged_f32=q_ret_bridged[qi],
            leg_mu_d=leg_mu_d, leg_norms_f32=leg_norms_f32, leg_rd_f32=leg_rd_f32,
            native_ids=nat_ids, native_idx=ni[0], native_raw_dots=nd[0], q_native_f32=q_ret_1024[qi],
            nat_mu_d=nat_mu_d, nat_norms_f32=nat_norms_f32, nat_rd_f32=nat_rd_f32, dev_z_stats=dev_z_stats, top_k=1,
        )["Chelated_ZeroMu_CSLS_8B"]
        if pre_hits:
            tid = pre_hits[0]
            if tid not in retracted_target_ids:
                retracted_target_ids.add(tid)
                ret_events.append(StratifiedEvent(event_id=tid, entity=tid, slot="status", claim_value="v1", authority_tier=2))
                rid = f"retract::{tid}"
                ret_events.append(StratifiedEvent(event_id=rid, entity=tid, slot="status", claim_value="revoked", authority_tier=4))
                ret_edges.append(EdgeAttestation(attestation_id=f"att::{tid}", src_id=rid, dst_id=tid, edge_type="REFUTE"))

    t_ek5 = time.perf_counter()
    retract_store.apply_mutation(ret_events, ret_edges)
    blocked_ids = {tid for tid in retracted_target_ids if retract_store.eligibility.get(tid) != "ADMIT"}
    leg_adm_flags[[i for i, did in enumerate(leg_ids) if did in blocked_ids]] = 0
    nat_adm_flags[[i for i, did in enumerate(nat_ids) if did in blocked_ids]] = 0
    fenced_leg_bset = pack_epoch_admitted_bitset(leg_adm_flags)
    fenced_nat_bset = pack_epoch_admitted_bitset(nat_adm_flags)
    ek5_fence_ms = (time.perf_counter() - t_ek5) * 1000.0

    unmasked_leaks = 0
    fenced_leaks = 0
    sampled_fenced_hits: List[List[str]] = []
    for qi, query_id in enumerate(retraction_qids):
        ld, li = idx_leg.search(q_ret_bridged[qi : qi + 1], 32)
        nd, ni = idx_nat.search(q_ret_1024[qi : qi + 1], 32)
        unm = score_dual_shard_shortlists(
            legacy_ids=leg_ids, legacy_idx=li[0], legacy_raw_dots=ld[0], q_bridged_f32=q_ret_bridged[qi],
            leg_mu_d=leg_mu_d, leg_norms_f32=leg_norms_f32, leg_rd_f32=leg_rd_f32,
            native_ids=nat_ids, native_idx=ni[0], native_raw_dots=nd[0], q_native_f32=q_ret_1024[qi],
            nat_mu_d=nat_mu_d, nat_norms_f32=nat_norms_f32, nat_rd_f32=nat_rd_f32, dev_z_stats=dev_z_stats, top_k=10,
        )["Chelated_ZeroMu_CSLS_8B"]
        fnc = score_dual_shard_shortlists(
            legacy_ids=leg_ids, legacy_idx=li[0], legacy_raw_dots=ld[0], q_bridged_f32=q_ret_bridged[qi],
            leg_mu_d=leg_mu_d, leg_norms_f32=leg_norms_f32, leg_rd_f32=leg_rd_f32,
            native_ids=nat_ids, native_idx=ni[0], native_raw_dots=nd[0], q_native_f32=q_ret_1024[qi],
            nat_mu_d=nat_mu_d, nat_norms_f32=nat_norms_f32, nat_rd_f32=nat_rd_f32, dev_z_stats=dev_z_stats,
            admitted_leg_bitset=fenced_leg_bset, admitted_nat_bitset=fenced_nat_bset, top_k=10,
        )["Chelated_ZeroMu_CSLS_8B"]
        require_fenced_top10(fnc, query_id)
        if qi < RETRACTION_ORACLE_SAMPLE_SIZE:
            sampled_fenced_hits.append(fnc)
        if any(d in retracted_target_ids for d in unm):
            unmasked_leaks += 1
        if any(d in retracted_target_ids for d in fnc):
            fenced_leaks += 1

    assert unmasked_leaks > 0, "Expected at least one unmasked retraction leak at 9.47M scale"
    assert fenced_leaks == 0, f"Expected 0 fenced leaks at 9.47M scale, got {fenced_leaks}"

    exact_admitted_leg = exact_chunk_top10(
        q_ret_bridged[:RETRACTION_ORACLE_SAMPLE_SIZE], leg_mm, leg_ids,
        admitted_flags=leg_adm_flags, sidecar=(leg_mu_d, leg_norms_f32, leg_rd_f32),
    )
    exact_admitted_nat = exact_chunk_top10(
        q_ret_1024[:RETRACTION_ORACLE_SAMPLE_SIZE], nat_mm, nat_ids,
        admitted_flags=nat_adm_flags, sidecar=(nat_mu_d, nat_norms_f32, nat_rd_f32),
    )
    admitted_recalls = []
    for qid, fnc, leg_hits, nat_hits in zip(
        sampled_retraction_qids, sampled_fenced_hits, exact_admitted_leg, exact_admitted_nat,
    ):
        oracle_hits = [did for _, did in nsmallest(10, (*leg_hits, *nat_hits))]
        require_fenced_top10(oracle_hits, qid)
        admitted_recalls.append(len(set(fnc).intersection(oracle_hits)) / 10.0)
    fenced_admitted_recall_at_10 = float(np.mean(admitted_recalls))
    assert fenced_admitted_recall_at_10 >= FLAT_ORACLE_RECALL_FLOOR, (
        f"Fenced ANN recall@10 {fenced_admitted_recall_at_10:.4f} is below {FLAT_ORACLE_RECALL_FLOOR}"
    )

    # Release FAISS native heap (`L5`)
    del idx_leg, idx_nat, leg_mm, nat_mm
    release_native_heap()

    part_a_payload = {
        "total_real_documents_merged": total_merged_docs,
        "standalone_msmarco_real_documents": n_msmarco,
        "extra_6beir_real_documents": n_extra,
        "synthetic_vector_fraction": 0.0,
        "msmarco_subsampled": False,
        "legacy_384d_shard_docs": n_leg_total,
        "upgraded_1024d_shard_docs": n_nat_total,
        "upgrade_model_1024": args.upgrade_model_1024,
        "stiefel_bridge_anchors": int(anc_384.shape[0]),
        "stiefel_anchor_fraction_of_corpus": round(float(anc_384.shape[0]) / float(total_merged_docs), 7),
        "stiefel_orthogonality_max_err": ortho_err,
        "faiss_index_factory": faiss_factory_desc,
        "sampled_oracle_qids": sampled_oracle_qids,
        "sampled_oracle_qids_sha256": sampled_oracle_qids_sha256,
        "sampled_oracle_n_queries": len(sampled_oracle_qids),
        "sampled_oracle_tie_breaking": TOP10_TIE_BREAKING,
        "sampled_oracle_recall_at_10_floor": FLAT_ORACLE_RECALL_FLOOR,
        "sampled_ann_vs_exact_flat_recall_at_10": round(ann_vs_flat_recall10, 5),
        "pre_run_mem_available_gib": pre_run_mem_available_gib,
        "peak_rss_gib": round(peak_rss, 3),
        "binary_storage_footprint_merged_9_47m": compute_exact_binary_footprint(total_merged_docs),
        "binary_storage_footprint_msmarco_8_84m": compute_exact_binary_footprint(n_msmarco),
        "timings": {
            "sidecar_stream_build_seconds": round(sidecar_build_s, 3),
            "faiss_hnsw_build_seconds": round(index_build_s, 3),
            "dual_shard_query_p50_ms": round(float(np.percentile(search_latencies_ms, 50)), 3),
            "dual_shard_query_p95_ms": round(float(np.percentile(search_latencies_ms, 95)), 3),
            "dual_shard_query_p99_ms": round(float(np.percentile(search_latencies_ms, 99)), 3),
            "ek5_bitset_fence_compile_ms": round(ek5_fence_ms, 3),
        },
        "retraction_fence_9_47m_proof": {
            "n_queries": int(q_ret_1024.shape[0]),
            "n_retracted_docs": len(retracted_target_ids),
            "unmasked_leaks": unmasked_leaks,
            "fenced_leaks": fenced_leaks,
            "fill_at_10": 1.0,
            "fenced_admitted_recall_at_10": round(fenced_admitted_recall_at_10, 5),
            "fenced_admitted_recall_n_queries": len(admitted_recalls),
            "fenced_admitted_recall_query_ids": sampled_retraction_qids,
            "packed_bitset_bytes_total": len(fenced_leg_bset) + len(fenced_nat_bset),
            "unmasked_stale_resurrection_rate": round(float(unmasked_leaks) / float(q_ret_1024.shape[0]), 5),
            "chelated_fenced_stale_resurrection_rate": round(float(fenced_leaks) / float(q_ret_1024.shape[0]), 5),
        },
        "evaluations_by_corpus": domain_results,
        "wall_seconds": round(time.perf_counter() - t_start, 2),
    }
    write_json_atomic(out_dir / PART_A_SUMMARY, part_a_payload)
    print("[D10-A] Saved part_a_cutover_summary.json", flush=True)


def _suite_a_reader_predictions(
    cases: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, str],
    tok: Any,
    model: Any,
    inference_mode: Callable[[], Any],
    *,
    gated_decisions: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, str], List[float]]:
    """Use the same governance short-circuit and decoding for every HF reader."""
    preds: Dict[str, str] = {}
    lat_ms: List[float] = []
    for case in cases:
        cid = str(case["case_id"])
        tq = time.perf_counter()
        if gated_decisions is not None and gated_decisions[cid] == "CONTESTED_ABSTAIN":
            preds[cid] = "ABSTAIN"
        else:
            messages = [
                {"role": "system", "content": ENTERPRISE_SCALE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Records:\n{contexts[cid]}\n\nQuestion: {case['query_text']}"},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
            text = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            preds[cid] = text.splitlines()[0].strip() if text else "ABSTAIN"
        lat_ms.append((time.perf_counter() - tq) * 1000.0)
    return preds, lat_ms


def _is_70b_resource_error(exc: Exception, torch_oom_type: type[Exception]) -> bool:
    """Recognize 70B OOM and bitsandbytes backend failures without swallowing unrelated errors."""
    if isinstance(exc, (MemoryError, torch_oom_type)):
        return True
    message = str(exc).lower()
    if any(marker in message for marker in (
        "out of memory",
        "cannot allocate memory",
        "cuda_error_out_of_memory",
        "cublas_status_alloc_failed",
        "defaultcpuallocator: can't allocate memory",
        "bitsandbytes",
        "no kernel image is available for execution on the device",
        "some modules are dispatched on the cpu or the disk",
    )):
        return True
    traceback = exc.__traceback__
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get("__name__", "")
        if module == "bitsandbytes" or module.startswith("bitsandbytes."):
            return True
        traceback = traceback.tb_next
    return False


def cmd_run_part_b_70b_sweep(args: argparse.Namespace) -> None:
    """Evaluate varying reader sizes under fixed Chelated governance alongside bare HybridRAG readers."""
    import requests
    import torch
    import transformers as _tf
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from chelated_epistemic_proposer_v2 import EXTRACTION_SYSTEM_PROMPT_V2, AuthorityDominatedExocortexV2
    from chelated_raw_text_exocortex import parse_extracted_claim_json

    t_start = time.perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    release_native_heap()

    _tf_major = int(str(_tf.__version__).split(".", 1)[0])
    osv_path = Path(args.osv_cache) if args.osv_cache else (REPO_ROOT / "cache" / "osv_cve_seeds.json")
    cases, docs_list, suite_a_sha = build_prospective_realworld_drift_suite(osv_path)
    docs_by_id = {d.doc_id: d for d in docs_list}

    # Build Hybrid BM25 + BGE-M3 dense top-k contexts (`k=5` and `k=17` full-recall control) per case
    embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
    bare_k5_ctx: Dict[str, str] = {}
    bare_k17_ctx: Dict[str, str] = {}
    chelated_ctx: Dict[str, str] = {}
    chelated_decisions: Dict[str, str] = {}
    nemotron_extract_total_ms = 0.0

    # Use Nemotron-30B (`:8890`) for Chelated's structured claim extraction
    vllm_url = args.vllm_endpoint.rstrip("/")

    def _call_nemotron(sys_p: str, usr_p: str, max_toks: int = 256) -> str:
        r = requests.post(
            f"{vllm_url}/chat/completions",
            json={
                "model": "nemotron-lightning",
                "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                "temperature": 0.0,
                "max_tokens": max_toks,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120.0,
        )
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"] or "").strip()

    for idx_c, c in enumerate(cases):
        cid = str(c["case_id"])
        q_txt = str(c["query_text"])
        c_docs = sorted(
            [docs_by_id[did] for did in c["doc_ids"]],
            key=lambda d: (d.ingested_ts, d.published_ts, d.doc_id),
        )
        bm = FastBM25Index([d.doc_id for d in c_docs], [d.text for d in c_docs])
        bm_sc = bm.score_query(q_txt)
        d_vecs = np.asarray(
            embedder.encode([d.text for d in c_docs], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )
        q_vec = np.asarray(
            embedder.encode([q_txt], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )
        dn_sc = d_vecs @ q_vec

        # RRF k=60 hybrid ranking
        bm_ord = np.argsort(-bm_sc, kind="mergesort")
        dn_ord = np.argsort(-dn_sc, kind="mergesort")
        rrf = np.zeros(len(c_docs), dtype=np.float64)
        for rk, idx_i in enumerate(bm_ord.tolist()):
            rrf[idx_i] += 1.0 / (60.0 + rk + 1.0)
        for rk, idx_i in enumerate(dn_ord.tolist()):
            rrf[idx_i] += 1.0 / (60.0 + rk + 1.0)
        top_ord = np.argsort(-rrf, kind="mergesort").tolist()

        def _fmt_docs(idxs: Sequence[int], _cdocs: Sequence[Any] = c_docs) -> str:
            lines = []
            for i in idxs:
                d = _cdocs[i]
                lines.append(f"[{d.doc_id} | channel={d.source_channel} | published_day={d.published_ts}] {d.text}")
            return "\n".join(lines)

        bare_k5_ctx[cid] = _fmt_docs(top_ord[:5])
        bare_k17_ctx[cid] = _fmt_docs(list(range(len(c_docs))))

        raw_claims = []
        for d in c_docs:
            u_prompt = f"Source channel: {d.source_channel}\nPublished day: {d.published_ts}\nDocument:\n{d.text}"
            t_extract = time.perf_counter()
            raw_json = _call_nemotron(EXTRACTION_SYSTEM_PROMPT_V2, u_prompt, 256)
            raw_claims.append(parse_extracted_claim_json(d.doc_id, raw_json))
            nemotron_extract_total_ms += (time.perf_counter() - t_extract) * 1000.0

        s_vecs = np.asarray(
            embedder.encode(
                [f"{cl.entity} {cl.slot}" for cl in raw_claims],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        exo = AuthorityDominatedExocortexV2(c_docs, raw_claims, d_vecs, s_vecs)
        q_res = exo.select_context_for_query(rrf, q_txt, top_k=5)
        chelated_decisions[cid] = str(q_res.get("status", "ADMIT_OK"))
        chelated_ctx[cid] = str(
            q_res.get("formatted_context")
            or "(CONTESTED_ABSTAIN: Equal-authority official sources are in unresolved mutual conflict. Reply ABSTAIN.)"
        )
        if (idx_c + 1) % 10 == 0:
            print(f"[D10-B] Extracted & compiled {idx_c + 1}/{len(cases)} cases...", flush=True)

    nemotron_extract_mean_ms_per_case = nemotron_extract_total_ms / len(cases)
    del embedder
    torch.cuda.empty_cache()
    release_native_heap()

    def _eval_hf_reader(
        model_repo: str,
        ctx_map: Mapping[str, str],
        gated_decisions: Optional[Mapping[str, str]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tok = AutoTokenizer.from_pretrained(model_repo)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        _dt_kw = {"dtype": torch.bfloat16} if _tf_major >= 5 else {"torch_dtype": torch.bfloat16}
        mdl = AutoModelForCausalLM.from_pretrained(model_repo, device_map=_resolve_hf_device_map(18.0), **_dt_kw)
        mdl.eval()
        n_params = sum(int(p.numel()) for p in mdl.parameters())
        preds, lat_ms = _suite_a_reader_predictions(
            cases, ctx_map, tok, mdl, torch.inference_mode, gated_decisions=gated_decisions,
        )
        mean_query_ms = float(np.mean(lat_ms))
        peak_gb = float(torch.cuda.max_memory_allocated()) / GIB if torch.cuda.is_available() else 0.0
        del mdl, tok
        torch.cuda.empty_cache()
        release_native_heap()
        telemetry = {
            "model_repo": model_repo,
            "stored_tensor_numel": n_params,
            "peak_cuda_allocated_gib": round(peak_gb, 3),
            "mean_query_ms": round(mean_query_ms, 2),
            "p95_query_ms": round(float(np.percentile(lat_ms, 95)), 2),
            "wall_seconds": round(time.perf_counter() - t0, 2),
        }
        if gated_decisions is not None:
            telemetry["mean_reader_plus_extraction_ms_per_case"] = round(
                mean_query_ms + nemotron_extract_mean_ms_per_case, 2,
            )
        return preds, telemetry

    # Evaluate Bare Nemotron-30B first via `:8890`
    print("[D10-B] Evaluating Bare Nemotron-30B (k=5)...", flush=True)
    preds_nem30b: Dict[str, str] = {}
    nem_lat: List[float] = []
    for c in cases:
        cid = str(c["case_id"])
        tq = time.perf_counter()
        out = _call_nemotron(ENTERPRISE_SCALE_SYSTEM_PROMPT, f"Records:\n{bare_k5_ctx[cid]}\n\nQuestion: {c['query_text']}", 64)
        preds_nem30b[cid] = out.splitlines()[0].strip() if out else "ABSTAIN"
        nem_lat.append((time.perf_counter() - tq) * 1000.0)

    # Evaluate compact Qwen 1.5B & 7B arms (`args.small_reader_model`, `Qwen/Qwen2.5-7B-Instruct`)
    print(f"[D10-B] Evaluating compact readers ({args.small_reader_model} & Qwen2.5-7B-Instruct)...", flush=True)
    preds_bare_7b, meta_bare_7b = _eval_hf_reader("Qwen/Qwen2.5-7B-Instruct", bare_k5_ctx)
    preds_chel_small, meta_chel_small = _eval_hf_reader(args.small_reader_model, chelated_ctx, gated_decisions=chelated_decisions)
    preds_chel_7b, meta_chel_7b = _eval_hf_reader("Qwen/Qwen2.5-7B-Instruct", chelated_ctx, gated_decisions=chelated_decisions)

    all_preds = {
        BARE_NEMOTRON_ARM: preds_nem30b,
        "Bare_Qwen2_5_7B_bf16_HybridRAG_k5": preds_bare_7b,
        SMALL_CHELATED_ARM: preds_chel_small,
        MID_CHELATED_ARM: preds_chel_7b,
    }
    exact_scores: Dict[str, Dict[str, float]] = {}
    stale_leaks: Dict[str, List[int]] = {}
    u_false_commits: Dict[str, List[int]] = {}
    nem_judge_scores: Dict[str, Dict[str, float]] = {}
    qwen_judge_scores: Dict[str, Dict[str, float]] = {}

    def _grade_arms(
        predictions: Mapping[str, Mapping[str, str]],
        *,
        nemotron_arms: Sequence[str],
        qwen_arms: Sequence[str],
    ) -> None:
        judge_nem = BlindCrossFamilySemanticJudge(
            judge_family="nemotron",
            generator_family="llama_or_qwen",
            completion_fn=lambda sp, up: _call_nemotron(sp, up, 32),
        )
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        dtype_kw = {"dtype": torch.bfloat16} if _tf_major >= 5 else {"torch_dtype": torch.bfloat16}
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map=_resolve_hf_device_map(18.0), **dtype_kw)
        model.eval()

        def _qwen_judge_fn(system_prompt: str, user_prompt: str) -> str:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        judge_qwen = BlindCrossFamilySemanticJudge(
            judge_family="qwen",
            generator_family="llama_or_nemotron",
            completion_fn=_qwen_judge_fn,
        )

        for arm in predictions:
            exact_scores[arm] = {}
            stale_leaks[arm] = []
            u_false_commits[arm] = []
        nem_judge_scores.update({arm: {} for arm in nemotron_arms})
        qwen_judge_scores.update({arm: {} for arm in qwen_arms})

        for case in cases:
            cid = str(case["case_id"])
            question = str(case["query_text"])
            expected = str(case["gold"])
            traps = list(case.get("stale_or_false_values") or [])
            require_abstain = bool(case.get("abstention_required", expected.strip().upper() == "ABSTAIN"))
            for arm, preds in predictions.items():
                grade = score_strict_suite_a_prediction(preds[cid], expected, traps, require_abstain)
                exact_scores[arm][cid] = float(grade["strict_exact_match"])
                stale_leaks[arm].append(int(grade["stale_leak"]))
                if require_abstain:
                    u_false_commits[arm].append(int(grade["false_commit_on_u"]))

            for judge, arms, scores in (
                (judge_nem, nemotron_arms, nem_judge_scores),
                (judge_qwen, qwen_arms, qwen_judge_scores),
            ):
                grades = judge.grade_arms_blind(
                    item_id=cid,
                    question=question,
                    gold_reference=expected,
                    arm_predictions={arm: predictions[arm][cid] for arm in arms},
                    abstention_required=require_abstain,
                )
                for arm, grade in grades.items():
                    scores[arm][cid] = float(grade["correct"])

    # Finish and grade the smaller arms before risking the 70B CUDA context.
    _grade_arms(
        all_preds,
        nemotron_arms=tuple(arm for arm in all_preds if arm != BARE_NEMOTRON_ARM),
        qwen_arms=(BARE_NEMOTRON_ARM,),
    )
    torch.cuda.empty_cache()

    def _eval_70b_readers() -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tok = AutoTokenizer.from_pretrained(LOCKED_70B_MODEL_ID)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        t_load = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(LOCKED_70B_MODEL_ID, device_map=_resolve_hf_device_map(44.0))
        model.eval()
        load_seconds = time.perf_counter() - t_load
        preds = {}
        mean_lat_ms = {}
        for arm, ctx_map, decisions in (
            (BARE_70B_K5_ARM, bare_k5_ctx, None),
            (BARE_70B_K17_ARM, bare_k17_ctx, None),
            (CHELATED_70B_ARM, chelated_ctx, chelated_decisions),
        ):
            print(f"[D10-B] Evaluating {arm}...", flush=True)
            preds[arm], lat_ms = _suite_a_reader_predictions(
                cases, ctx_map, tok, model, torch.inference_mode, gated_decisions=decisions,
            )
            mean_lat_ms[arm] = float(np.mean(lat_ms))
        peak_cuda_gib = float(torch.cuda.max_memory_allocated()) / GIB if torch.cuda.is_available() else 0.0
        return preds, {
            "load_seconds": round(load_seconds, 2),
            "peak_cuda_allocated_gib": round(peak_cuda_gib, 3),
            "mean_query_ms_k5": round(mean_lat_ms[BARE_70B_K5_ARM], 2),
            "mean_query_ms_k17": round(mean_lat_ms[BARE_70B_K17_ARM], 2),
            "mean_query_ms_chelated": round(mean_lat_ms[CHELATED_70B_ARM], 2),
            "mean_reader_plus_extraction_ms_per_case_chelated": round(
                mean_lat_ms[CHELATED_70B_ARM] + nemotron_extract_mean_ms_per_case, 2,
            ),
        }

    release_native_heap()
    pre_70b_mem_available_gib = mem_available_gib() if sys.platform.startswith("linux") else None
    print(
        f"[D10-B] Loading {LOCKED_70B_MODEL_ID} ({LOCKED_70B_QUANT_DISCLOSURE}) "
        f"with pre_70b_mem_available_gib={pre_70b_mem_available_gib}",
        flush=True,
    )
    model_70b_meta = {
        "model_id": LOCKED_70B_MODEL_ID,
        "nominal_parameters_b": 70.6,
        "quantization": LOCKED_70B_QUANT_DISCLOSURE,
        "load_seconds": None,
        "peak_cuda_allocated_gib": None,
        "mean_query_ms_k5": None,
        "mean_query_ms_k17": None,
        "mean_query_ms_chelated": None,
        "mean_reader_plus_extraction_ms_per_case_chelated": None,
    }
    try:
        preds_70b, telemetry_70b = _eval_70b_readers()
    except (RuntimeError, ValueError, ImportError, OSError, MemoryError) as exc:
        if not _is_70b_resource_error(exc, torch.cuda.OutOfMemoryError):
            raise
        status_70b = "NOT_RUN"
        reason_70b = str(exc)[:400]
        preds_70b = {}
        print(f"[D10-B] 70B NOT_RUN: {reason_70b}", flush=True)
    else:
        status_70b = "COMPLETED"
        reason_70b = None
        model_70b_meta.update(telemetry_70b)

    release_native_heap()
    if status_70b == "COMPLETED":
        torch.cuda.empty_cache()
        all_preds.update(preds_70b)
        _grade_arms(preds_70b, nemotron_arms=tuple(preds_70b), qwen_arms=tuple(preds_70b))
        torch.cuda.empty_cache()
        release_native_heap()

    cids_sorted = sorted(str(c["case_id"]) for c in cases)

    def _mean_score(scores: Sequence[float]) -> Optional[float]:
        return round(float(np.mean(scores)), 5) if scores else None

    def _agreement(arms: Sequence[str]) -> Dict[str, Any]:
        nem = [int(nem_judge_scores[arm][cid]) for arm in arms for cid in cids_sorted]
        qwen = [int(qwen_judge_scores[arm][cid]) for arm in arms for cid in cids_sorted]
        return {
            "n_rows": len(nem),
            "cohens_kappa": round(compute_cohens_kappa_binary(nem, qwen), 5),
            "raw_agreement": round(float(np.mean(np.asarray(nem) == np.asarray(qwen))), 5),
        }

    def _contrast(arm_a: str, arm_b: str) -> Dict[str, Any]:
        return paired_bootstrap_wave6(nem_judge_scores[arm_a], nem_judge_scores[arm_b], expected_ids=cids_sorted)

    contrasts_nem_judge = {}
    descriptive_contrasts = {}
    bare_agreement = None
    all_70b_agreement = None
    if status_70b == "COMPLETED":
        bare_agreement = _agreement((BARE_70B_K5_ARM, BARE_70B_K17_ARM))
        all_70b_agreement = _agreement(tuple(preds_70b))
        contrasts_nem_judge = {
            "Small_Chelated_minus_Bare_Llama70B_4bit_k5": _contrast(SMALL_CHELATED_ARM, BARE_70B_K5_ARM),
            "Mid_7B_Chelated_minus_Bare_Llama70B_4bit_k5": _contrast(MID_CHELATED_ARM, BARE_70B_K5_ARM),
        }
        descriptive_contrasts = {
            "Small_Chelated_minus_Bare_Llama70B_4bit_k17": _contrast(SMALL_CHELATED_ARM, BARE_70B_K17_ARM),
            "Mid_7B_Chelated_minus_Bare_Llama70B_4bit_k17": _contrast(MID_CHELATED_ARM, BARE_70B_K17_ARM),
            "Chelated_70B_minus_Bare_70B_k5": _contrast(CHELATED_70B_ARM, BARE_70B_K5_ARM),
            "Chelated_70B_minus_Bare_70B_k17": _contrast(CHELATED_70B_ARM, BARE_70B_K17_ARM),
        }
    holm_nem = holm_bonferroni_stepdown({k: v["p_value_two_sided"] for k, v in contrasts_nem_judge.items()})

    arm_metrics = {}
    for arm in all_preds:
        arm_metrics[arm] = {
            "strict_exact_match": _mean_score(list(exact_scores[arm].values())),
            "nemotron_30b_cross_family_blind_acc": _mean_score(list(nem_judge_scores.get(arm, {}).values())),
            "qwen_7b_cross_family_blind_acc": _mean_score(list(qwen_judge_scores.get(arm, {}).values())),
            "stale_leak_rate": _mean_score(stale_leaks[arm]),
            "false_commit_rate_on_u": _mean_score(u_false_commits[arm]),
        }

    for arm in (SMALL_CHELATED_ARM, MID_CHELATED_ARM, CHELATED_70B_ARM):
        if arm not in arm_metrics:
            continue
        for cohort, gate_decided in (("gate_decided_cases", True), ("reader_decided_cases", False)):
            ids = [cid for cid in cids_sorted if (chelated_decisions[cid] == "CONTESTED_ABSTAIN") == gate_decided]
            arm_metrics[arm][cohort] = {
                "n_cases": len(ids),
                "strict_exact_match": _mean_score([exact_scores[arm][cid] for cid in ids]),
                "nemotron_30b_cross_family_blind_acc": _mean_score([nem_judge_scores[arm][cid] for cid in ids]),
                "qwen_7b_cross_family_blind_acc": (
                    _mean_score([qwen_judge_scores[arm][cid] for cid in ids]) if arm in qwen_judge_scores else None
                ),
            }
    if CHELATED_70B_ARM in arm_metrics:
        arm_metrics[CHELATED_70B_ARM]["label"] = CHELATED_70B_LABEL

    part_b_payload = {
        "status": status_70b,
        "reason": reason_70b,
        "suite": "prospective_realworld_drift_suite_n60",
        "suite_sha256": suite_a_sha,
        "n_cases": len(cids_sorted),
        "design_scope": "reader_size_varied_under_fixed_nemotron30b_extractor_governance_pipeline",
        "pre_70b_mem_available_gib": pre_70b_mem_available_gib,
        "nemotron_extract_total_ms": round(nemotron_extract_total_ms, 2),
        "nemotron_extract_mean_ms_per_case": round(nemotron_extract_mean_ms_per_case, 2),
        "locked_frontier_70b_model": model_70b_meta,
        "small_and_mid_reader_telemetry": {
            "small_chelated": meta_chel_small,
            "mid_7b_chelated": meta_chel_7b,
            "bare_7b": meta_bare_7b,
            "bare_nemotron_30b_mean_ms": round(float(np.mean(nem_lat)), 2),
        },
        "per_case_predictions": all_preds,
        "arm_metrics": arm_metrics,
        "inter_judge_agreement_llama70b_120_rows": bare_agreement,
        "inter_judge_agreement_llama70b": all_70b_agreement,
        "paired_contrasts_nemotron_blind_judge": contrasts_nem_judge,
        "descriptive_contrasts": descriptive_contrasts,
        "holm_bonferroni_family": holm_nem,
        "wall_seconds": round(time.perf_counter() - t_start, 2),
    }
    write_json_atomic(out_dir / PART_B_SUMMARY, part_b_payload)
    print("[D10-B] Saved part_b_70b_sweep_summary.json", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wave 11 Demo D10: 8.84M/9.47M Real Corpus Cutover + 70B Scale Sweep")
    p.add_argument(
        "--mode",
        choices=("verify-only", "encode-msmarco-shard", "run-part-a-cutover", "run-part-b-70b-sweep", "assemble-manifest"),
        default="verify-only",
    )
    p.add_argument("--work-dir", default="/tmp/wave11_d10_work")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--upgrade-model-1024", default="BAAI/bge-m3")
    p.add_argument("--small-reader-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--vllm-endpoint", default="http://127.0.0.1:8890/v1")
    p.add_argument("--osv-cache", default="")
    p.add_argument("--part-a-json", default="")
    p.add_argument("--part-b-json", default="")
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "wave11_d10_5m_real_cutover_and_70b_scale"),
    )
    return p


def cmd_verify_only() -> None:
    fp = compute_exact_binary_footprint(9_465_539)
    assert fp["exact_ratio_vs_1024d_f32"] == 512.0
    assert fp["exact_ratio_vs_4096d_f32"] == 2048.0
    print("D10 verify-only OK:", json.dumps(fp, indent=2))


def _write_summary_bytes_atomic(path: Path, data: bytes) -> None:
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.close()
            tmp_path.replace(path)
        finally:
            tmp.close()
            tmp_path.unlink(missing_ok=True)


def compute_part_a_multiplicity_and_loss_disclosure(part_a: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute Holm-Bonferroni step-down across all Part A domain x challenger contrasts and surface all Chelated losses."""
    raw_p: Dict[str, float] = {}
    losses: List[Dict[str, Any]] = []
    dom_res = part_a.get("domain_evaluation_results") or {}
    for dom_name, dom_block in sorted(dom_res.items()):
        contrasts = (dom_block or {}).get("contrasts") or {}
        for cname, cstats in sorted(contrasts.items()):
            key = f"{dom_name}::{cname}"
            if isinstance(cstats, Mapping) and "p_value_two_sided" in cstats:
                raw_p[key] = float(cstats["p_value_two_sided"])
                delta = float(cstats.get("mean_delta", 0.0))
                ci_hi = float(cstats.get("ci95_high", 0.0))
                if delta < 0.0:
                    losses.append(
                        {
                            "contrast": key,
                            "mean_delta_ndcg_at_10": round(delta, 5),
                            "ci95_low": round(float(cstats.get("ci95_low", 0.0)), 5),
                            "ci95_high": round(ci_hi, 5),
                            "statistically_significant_loss": bool(ci_hi < 0.0),
                        }
                    )
        rec = (dom_block or {}).get("means_recall_at_10") or {}
        chel_r = rec.get("Chelated_ZeroMu_CSLS_8B")
        if chel_r is not None:
            for other_arm, other_r in sorted(rec.items()):
                if other_arm != "Chelated_ZeroMu_CSLS_8B" and float(other_r) > float(chel_r):
                    losses.append(
                        {
                            "contrast": f"{dom_name}::recall_at_10_vs_{other_arm}",
                            "chelated_recall_at_10": round(float(chel_r), 5),
                            "challenger_recall_at_10": round(float(other_r), 5),
                            "mean_delta_recall_at_10": round(float(chel_r) - float(other_r), 5),
                        }
                    )
    holm = holm_bonferroni_stepdown(raw_p) if raw_p else {}
    return {
        "n_contrasts_in_family": len(raw_p),
        "holm_bonferroni_24_contrasts": holm,
        "disclosed_chelated_point_or_ci_losses": losses,
    }


def cmd_assemble_manifest(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    part_a_path = Path(args.part_a_json) if args.part_a_json else (out_dir / PART_A_SUMMARY)
    part_b_path = Path(args.part_b_json) if args.part_b_json else (out_dir / PART_B_SUMMARY)
    part_a_bytes = part_a_path.read_bytes()
    part_b_bytes = part_b_path.read_bytes()
    part_a = json.loads(part_a_bytes)
    part_b = json.loads(part_b_bytes)
    if "ek5_retraction_fence_9_47m" in part_a:
        fenced_rec = float((part_a.get("ek5_retraction_fence_9_47m") or {}).get("fenced_admitted_recall_at_10", 0.0))
        if fenced_rec < FLAT_ORACLE_RECALL_FLOOR:
            raise RuntimeError(f"Part A fenced_admitted_recall_at_10 {fenced_rec:.4f} < {FLAT_ORACLE_RECALL_FLOOR}")
    part_a_multiplicity = compute_part_a_multiplicity_and_loss_disclosure(part_a)
    input_part_a_sha256 = sha256_bytes(part_a_bytes)
    input_part_b_sha256 = sha256_bytes(part_b_bytes)
    input_hashes = {
        "input_part_a_sha256": input_part_a_sha256,
        "input_part_b_sha256": input_part_b_sha256,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    side_hashes = {}
    for name, input_path, data, digest in (
        (PART_A_SUMMARY, part_a_path, part_a_bytes, input_part_a_sha256),
        (PART_B_SUMMARY, part_b_path, part_b_bytes, input_part_b_sha256),
    ):
        canonical_path = out_dir / name
        if Path(os.path.abspath(input_path)) != Path(os.path.abspath(canonical_path)):
            _write_summary_bytes_atomic(canonical_path, data)
        if sha256_file(canonical_path) != digest:
            raise RuntimeError(f"D10 summary bytes changed while assembling: {canonical_path}")
        side_hashes[name] = digest

    manifest = {
        "bundle_id": "wave11_d10_5m_real_cutover_and_70b_scale",
        "part": "PART_VII_TIER1_AND_TIER2_FLAGSHIPS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(REPO_ROOT),
        "wall_seconds": round(float(part_a.get("wall_seconds", 0.0)) + float(part_b.get("wall_seconds", 0.0)), 2),
        "locked_70b_model": {
            "model_id": LOCKED_70B_MODEL_ID,
            "quantization_disclosure": LOCKED_70B_QUANT_DISCLOSURE,
        },
        **input_hashes,
        "part_a_multiplicity_and_loss_disclosure": part_a_multiplicity,
        "part_a_real_corpus_cutover": part_a,
        "part_b_70b_frontier_scale_sweep": part_b,
    }
    mpath = out_dir / "wave11_d10_scale_manifest.json"
    m_sha = write_json_atomic(mpath, manifest)

    receipt = {
        "bundle_id": "wave11_d10_5m_real_cutover_and_70b_scale",
        "manifest_file": mpath.name,
        "manifest_sha256": m_sha,
        **input_hashes,
        "artifact_hashes": side_hashes,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    r_sha = write_json_atomic(out_dir / "report_receipt.json", receipt)
    print(f"Sealed D10 bundle: manifest_sha256={m_sha[:12]}, receipt_sha256={r_sha[:12]}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "verify-only":
        cmd_verify_only()
    elif args.mode == "encode-msmarco-shard":
        cmd_encode_msmarco_shard(args)
    elif args.mode == "run-part-a-cutover":
        cmd_run_part_a_cutover(args)
    elif args.mode == "run-part-b-70b-sweep":
        cmd_run_part_b_70b_sweep(args)
    elif args.mode == "assemble-manifest":
        cmd_assemble_manifest(args)


if __name__ == "__main__":
    main()
