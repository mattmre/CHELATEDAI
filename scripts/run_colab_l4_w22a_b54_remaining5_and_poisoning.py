#!/usr/bin/env python3
"""Colab L4 Wave 22A: Complete 12/12 Unburned Public-Domain Closure, 20-Draw Anchor Variance,
0/1/5/10/20% Anchor Contamination Sweep, Real Prompt-Shift, & Pooled Power-Sized Tango NI.

Encodes the 5 remaining unburned MTEB/BEIR domains (`dbpedia`, `climate-fever`,
`cqadupstack-english`, `cqadupstack-gis`, `cqadupstack-tex`) across all 5 real neural encoders
(`MiniLM-384`, `bge-small-384`, `mpnet-768`, `bge-m3-1024`, `Qwen/Qwen3-Embedding-8B` `4096-d`)
plus real prompt-wrapper-shifted `Qwen3-4096` queries, then:
  1. Runs the complete 5x3 Bridge x Merge factorial (`Mode B` 20% backfill `4096 -> 384`),
     real prompt-shift contrast (without refitting), and 4-hop chain (`384 -> ... -> 4096`).
  2. Runs 20 deterministic seeded 500-anchor draws per domain (`sample_anchor_draws` +
     `summarize_anchor_draw_variance`), cleanly separating within-domain anchor SD from cross-domain SD.
  3. Runs the progressive anchor contamination sweep (`0%, 1%, 5%, 10%, 20%`) across
     `mismatched`, `duplicate`, and `off_domain` modes with `TRA-1` health diagnostics.
  4. Pools with the 7 unburned domains in `artifacts/pv7_colab_l4_closure_20261125/colab_l4_closure_manifest.json`
     to emit the complete 12/12 real-neural-encoder unburned public-corpus adjudication for `#54`
     (`H6B_merge`, `H6B_bridge_static`, `H6B_4hop`) and `#50` (pooled exact `Tango (1998)` NI, `N > 2,200`).
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_utils import canonicalize_id  # noqa: E402
from chelated_loft_and_zero_write_ni import PREREGISTERED_NI_MARGIN, compute_exact_tango_ni  # noqa: E402
from chelated_pv7_spine.bridge_merge_harness import (  # noqa: E402
    CSLSMerge,
    CenteredProcrustes,
    UncenteredProcrustes,
    ZeroMuPolarBridge,
    evaluate_4hop_chain,
    evaluate_item6b_hypotheses,
    evaluate_score_matrix,
    exact_binomial_upper_tail_p,
    l2_normalize,
    paired_bootstrap_ci,
    run_contamination_sweep,
    run_factorial_5x3,
    sample_anchor_draws,
    summarize_anchor_draw_variance,
)
from chelated_pv7_spine.burn_ledger import BURNED_CORPORA, normalize_dataset_id, text_fingerprint  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402

ENCODER_SPECS: Tuple[Tuple[str, str, int, int], ...] = (
    ("MiniLM-384", "sentence-transformers/all-MiniLM-L6-v2", 384, 256),
    ("bge-small-384", "BAAI/bge-small-en-v1.5", 384, 256),
    ("mpnet-768", "sentence-transformers/all-mpnet-base-v2", 768, 128),
    ("bge-m3-1024", "BAAI/bge-m3", 1024, 96),
    ("Qwen3-4096", "Qwen/Qwen3-Embedding-8B", 4096, 64),
)

REMAINING5_DOMAIN_PLAN: Tuple[Dict[str, Any], ...] = (
    {
        "domain_id": "dbpedia",
        "hf_repo": "mteb/dbpedia",
        "max_docs": 750,
        "max_eval_q": 260,
        "max_dev_q": 80,
        "anchor_pool": 620,
    },
    {
        "domain_id": "climate-fever",
        "hf_repo": "mteb/climate-fever",
        "max_docs": 750,
        "max_eval_q": 260,
        "max_dev_q": 80,
        "anchor_pool": 620,
    },
    {
        "domain_id": "cqadupstack-english",
        "hf_repo": "mteb/cqadupstack-english",
        "max_docs": 750,
        "max_eval_q": 260,
        "max_dev_q": 80,
        "anchor_pool": 620,
    },
    {
        "domain_id": "cqadupstack-gis",
        "hf_repo": "mteb/cqadupstack-gis",
        "max_docs": 750,
        "max_eval_q": 260,
        "max_dev_q": 80,
        "anchor_pool": 620,
    },
    {
        "domain_id": "cqadupstack-tex",
        "hf_repo": "mteb/cqadupstack-tex",
        "max_docs": 750,
        "max_eval_q": 260,
        "max_dev_q": 80,
        "anchor_pool": 620,
    },
)

PROMPT_SHIFT_PREFIX = (
    "Instruct: Given a technical search query, retrieve authoritative supporting passages "
    "under cross-encoder migration.\nQuery: "
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_beir_slice_from_hf(
    *,
    domain_id: str,
    hf_repo: str,
    cache_dir: Path,
    max_docs: int,
    max_eval_q: int,
    max_dev_q: int,
    anchor_pool: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Stream-load a deterministic unburned MTEB/BEIR slice (memory-safe on multi-GB corpora)."""
    from huggingface_hub import hf_hub_download

    norm_id = normalize_dataset_id(domain_id)
    if norm_id in BURNED_CORPORA:
        raise RuntimeError(f"Burned corpus forbidden: {domain_id!r} normalized to {norm_id!r}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching {hf_repo} JSONL/TSV files via HuggingFace Hub ...", flush=True)
    qrels_path = Path(hf_hub_download(hf_repo, "qrels/test.tsv", repo_type="dataset", cache_dir=str(cache_dir)))
    queries_path = Path(hf_hub_download(hf_repo, "queries.jsonl", repo_type="dataset", cache_dir=str(cache_dir)))
    corpus_path = Path(hf_hub_download(hf_repo, "corpus.jsonl", repo_type="dataset", cache_dir=str(cache_dir)))
    file_digests = {
        "qrels/test.tsv": sha256_file(qrels_path),
        "queries.jsonl": sha256_file(queries_path),
        "corpus.jsonl": sha256_file(corpus_path),
    }

    qrels_all: Dict[str, Dict[str, float]] = {}
    with qrels_path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter="\t")
        _header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score_s = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                sc = float(score_s)
            except ValueError:
                continue
            if sc > 0.0:
                qrels_all.setdefault(qid, {})[did] = sc

    queries_raw: Dict[str, str] = {}
    with queries_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj["_id"]).strip()
            txt = str(obj.get("text") or "").strip()
            if qid in qrels_all and txt:
                queries_raw[qid] = txt

    # Cap per-query positive docs to top-3 by grade so dense corpora stay compact and well-balanced
    capped_qrels: Dict[str, Dict[str, float]] = {}
    for qid, dmap in qrels_all.items():
        if qid not in queries_raw:
            continue
        top_dids = sorted(dmap.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        capped_qrels[qid] = dict(top_dids)

    # Deterministically pre-select candidate queries BEFORE scanning multi-GB corpus.jsonl
    ordered_qids = sorted(
        capped_qrels.keys(),
        key=lambda q: hashlib.sha256(f"{seed}:q:{domain_id}:{q}".encode("utf-8")).hexdigest(),
    )
    cand_q_limit = min(len(ordered_qids), int((max_eval_q + max_dev_q) * 1.4) + 40)
    cand_qids = ordered_qids[:cand_q_limit]
    needed_pos_docs = {did for qid in cand_qids for did in capped_qrels[qid]}

    # Stream corpus.jsonl with bounded memory: collect needed_pos_docs + deterministic reservoir of distractors
    corpus_raw: Dict[str, str] = {}
    reservoir_cap = max_docs + anchor_pool + 500
    reservoir: List[Tuple[str, str, str]] = []
    with corpus_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_idx, line in enumerate(fh):
            if not line.strip():
                continue
            # Fast pre-check when reservoir is already full and only missing positive docs remain
            if len(reservoir) >= reservoir_cap and len(corpus_raw) >= len(needed_pos_docs):
                break
            obj = json.loads(line)
            did = str(obj["_id"]).strip()
            title = str(obj.get("title") or "").strip()
            body = str(obj.get("text") or "").strip()
            full_txt = (f"{title}\n{body}".strip() if title else body)[:1400]
            if not full_txt:
                continue
            if did in needed_pos_docs:
                corpus_raw[did] = full_txt
            elif len(reservoir) < reservoir_cap:
                h_key = hashlib.sha256(f"{seed}:{domain_id}:{did}".encode("utf-8")).hexdigest()
                reservoir.append((h_key, did, full_txt))
            if line_idx > 0 and line_idx % 1_000_000 == 0:
                print(
                    f"[corpus-scan] {domain_id}: scanned {line_idx:,} rows, "
                    f"pos={len(corpus_raw)}/{len(needed_pos_docs)}, res={len(reservoir)}",
                    flush=True,
                )

    valid_qids = [qid for qid in cand_qids if any(d in corpus_raw for d in capped_qrels[qid])]
    n_eval = min(int(max_eval_q), max(10, int(len(valid_qids) * 0.75)))
    n_dev = min(int(max_dev_q), max(10, len(valid_qids) - n_eval))
    eval_qids = valid_qids[:n_eval]
    dev_qids = valid_qids[n_eval : n_eval + n_dev]

    active_pos_docs = sorted(
        {did for qid in (eval_qids + dev_qids) for did in capped_qrels[qid] if did in corpus_raw}
    )
    reservoir.sort(key=lambda item: item[0])

    eval_doc_ids = list(active_pos_docs)
    extra_pool: List[Tuple[str, str]] = [(did, txt) for _, did, txt in reservoir if did not in corpus_raw]
    cursor = 0
    while len(eval_doc_ids) < max_docs and cursor < len(extra_pool):
        did, txt = extra_pool[cursor]
        corpus_raw[did] = txt
        eval_doc_ids.append(did)
        cursor += 1

    seen_doc_fps = {text_fingerprint(corpus_raw[d]) for d in eval_doc_ids}
    anchor_doc_ids: List[str] = []
    anchor_doc_texts: List[str] = []
    while len(anchor_doc_ids) < anchor_pool and cursor < len(extra_pool):
        did, txt = extra_pool[cursor]
        cursor += 1
        fp = text_fingerprint(txt)
        if fp in seen_doc_fps:
            continue
        seen_doc_fps.add(fp)
        anchor_doc_ids.append(did)
        anchor_doc_texts.append(txt)

    dev_ref_doc_ids: List[str] = []
    dev_ref_doc_texts: List[str] = []
    while len(dev_ref_doc_ids) < 180 and cursor < len(extra_pool):
        did, txt = extra_pool[cursor]
        cursor += 1
        fp = text_fingerprint(txt)
        if fp in seen_doc_fps:
            continue
        seen_doc_fps.add(fp)
        dev_ref_doc_ids.append(did)
        dev_ref_doc_texts.append(txt)

    if len(anchor_doc_ids) < 520 or len(dev_ref_doc_ids) < 80:
        raise RuntimeError(
            f"Domain {domain_id} insufficient background docs: anchors={len(anchor_doc_ids)}, dev_ref={len(dev_ref_doc_ids)}"
        )

    eval_qrels = {
        qid: {did: sc for did, sc in capped_qrels[qid].items() if did in corpus_raw}
        for qid in eval_qids
    }
    dev_qrels = {
        qid: {did: sc for did, sc in capped_qrels[qid].items() if did in corpus_raw}
        for qid in dev_qids
    }

    return {
        "domain_id": domain_id,
        "normalized_burn_id": norm_id,
        "eval_doc_ids": eval_doc_ids,
        "eval_doc_texts": [corpus_raw[d] for d in eval_doc_ids],
        "anchor_doc_ids": anchor_doc_ids,
        "anchor_doc_texts": anchor_doc_texts,
        "dev_ref_doc_ids": dev_ref_doc_ids,
        "dev_ref_doc_texts": dev_ref_doc_texts,
        "eval_query_ids": eval_qids,
        "eval_query_texts": [queries_raw[q] for q in eval_qids],
        "dev_query_ids": dev_qids,
        "dev_query_texts": [queries_raw[q] for q in dev_qids],
        "shifted_eval_query_texts": [PROMPT_SHIFT_PREFIX + queries_raw[q] for q in eval_qids],
        "shifted_dev_query_texts": [PROMPT_SHIFT_PREFIX + queries_raw[q] for q in dev_qids],
        "eval_qrels": eval_qrels,
        "dev_qrels": dev_qrels,
    }, file_digests


def encode_all_texts_on_gpu(
    model_id: str,
    texts: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    import torch
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model_kwargs: Dict[str, Any] = {}
    if "Qwen3-Embedding" in model_id:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16

    model = SentenceTransformer(model_id, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = 256
    hf_info: Dict[str, Any] = {"model_id": model_id, "max_seq_length": int(model.max_seq_length)}
    vecs = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    arr = np.asarray(vecs, dtype=np.float32)
    wall_s = time.perf_counter() - t0
    peak_vram_mib = (
        round(float(torch.cuda.max_memory_allocated() / (1024.0 ** 2)), 2)
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )
    del model
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    hf_info.update(
        {
            "encoded_count": int(arr.shape[0]),
            "embedding_dim": int(arr.shape[1]),
            "embeddings_sha256": sha256_bytes(np.ascontiguousarray(arr).tobytes()),
            "wall_seconds": round(wall_s, 2),
            "peak_vram_mib": peak_vram_mib,
        }
    )
    return arr, hf_info


def scrub_embedding_row_collisions_globally(domain_bundles: List[Dict[str, Any]]) -> Dict[str, int]:
    """Eliminate all post-truncation float64 row collisions across auxiliary roles and domains."""
    enc_names = list(domain_bundles[0]["emb"].keys())
    forbidden_by_enc: Dict[str, set] = {e: set() for e in enc_names}

    for dom in domain_bundles:
        for e in enc_names:
            for mat_key in ("eval_docs", "eval_queries", "shifted_eval_queries"):
                if mat_key not in dom["emb"][e]:
                    continue
                mat = np.ascontiguousarray(np.asarray(dom["emb"][e][mat_key], dtype=np.float64))
                for row in mat:
                    forbidden_by_enc[e].add(row.tobytes())

    removed = {"dev_queries": 0, "anchor_docs": 0, "dev_ref_docs": 0}

    for dom in domain_bundles:
        n_dq = len(dom["dev_query_ids"])
        keep_dq = []
        for i in range(n_dq):
            collided = False
            for e in enc_names:
                b1 = np.ascontiguousarray(np.asarray(dom["emb"][e]["dev_queries"][i], dtype=np.float64)).tobytes()
                if b1 in forbidden_by_enc[e]:
                    collided = True
                    break
                if "shifted_dev_queries" in dom["emb"][e]:
                    b2 = np.ascontiguousarray(
                        np.asarray(dom["emb"][e]["shifted_dev_queries"][i], dtype=np.float64)
                    ).tobytes()
                    if b2 in forbidden_by_enc[e]:
                        collided = True
                        break
            if collided:
                removed["dev_queries"] += 1
                continue
            keep_dq.append(i)
            for e in enc_names:
                forbidden_by_enc[e].add(
                    np.ascontiguousarray(np.asarray(dom["emb"][e]["dev_queries"][i], dtype=np.float64)).tobytes()
                )
                if "shifted_dev_queries" in dom["emb"][e]:
                    forbidden_by_enc[e].add(
                        np.ascontiguousarray(
                            np.asarray(dom["emb"][e]["shifted_dev_queries"][i], dtype=np.float64)
                        ).tobytes()
                    )
        dom["dev_query_ids"] = [dom["dev_query_ids"][i] for i in keep_dq]
        for e in enc_names:
            dom["emb"][e]["dev_queries"] = dom["emb"][e]["dev_queries"][keep_dq]
            if "shifted_dev_queries" in dom["emb"][e]:
                dom["emb"][e]["shifted_dev_queries"] = dom["emb"][e]["shifted_dev_queries"][keep_dq]

    for dom in domain_bundles:
        n_a = len(dom["anchor_doc_ids"])
        keep_a = []
        for i in range(n_a):
            if any(
                np.ascontiguousarray(np.asarray(dom["emb"][e]["anchor_docs"][i], dtype=np.float64)).tobytes()
                in forbidden_by_enc[e]
                for e in enc_names
            ):
                removed["anchor_docs"] += 1
                continue
            keep_a.append(i)
            for e in enc_names:
                forbidden_by_enc[e].add(
                    np.ascontiguousarray(np.asarray(dom["emb"][e]["anchor_docs"][i], dtype=np.float64)).tobytes()
                )
        dom["anchor_doc_ids"] = [dom["anchor_doc_ids"][i] for i in keep_a]
        for e in enc_names:
            dom["emb"][e]["anchor_docs"] = dom["emb"][e]["anchor_docs"][keep_a]

    for dom in domain_bundles:
        n_r = len(dom["dev_ref_doc_ids"])
        keep_r = []
        for i in range(n_r):
            if any(
                np.ascontiguousarray(np.asarray(dom["emb"][e]["dev_ref_docs"][i], dtype=np.float64)).tobytes()
                in forbidden_by_enc[e]
                for e in enc_names
            ):
                removed["dev_ref_docs"] += 1
                continue
            keep_r.append(i)
            for e in enc_names:
                forbidden_by_enc[e].add(
                    np.ascontiguousarray(np.asarray(dom["emb"][e]["dev_ref_docs"][i], dtype=np.float64)).tobytes()
                )
        dom["dev_ref_doc_ids"] = [dom["dev_ref_doc_ids"][i] for i in keep_r]
        for e in enc_names:
            dom["emb"][e]["dev_ref_docs"] = dom["emb"][e]["dev_ref_docs"][keep_r]

    return removed


def evaluate_remaining5_and_pool_12(
    domain_bundles: Sequence[Dict[str, Any]],
    *,
    prior7_manifest_path: Optional[Path] = None,
    n_anchor_draws: int = 20,
    anchors_per_draw: int = 500,
) -> Dict[str, Any]:
    """Evaluate 5x3 factorial, real prompt-shift, 4-hop, 20-draw variance, contamination sweep, & 12-domain pool."""
    enc_names = [spec[0] for spec in ENCODER_SPECS]
    per_domain_summary: Dict[str, Any] = {}
    csls_minus_zscore: Dict[str, float] = {}
    zeromu_minus_uncentered_static: Dict[str, float] = {}
    zeromu_minus_centered_static: Dict[str, float] = {}
    shift_zeromu_minus_uncentered: Dict[str, float] = {}
    shift_zeromu_minus_centered: Dict[str, float] = {}
    prefix_disp_norms: Dict[str, float] = {}
    fourhop_zeromu_minus_uncentered: Dict[str, float] = {}
    fourhop_direct_v1_v5_delta: Dict[str, float] = {}

    draw_scores_zeromu: Dict[str, List[float]] = {}
    draw_scores_uncentered: Dict[str, List[float]] = {}
    actual_anchors_by_dom: Dict[str, int] = {}

    contamination_by_dom: Dict[str, Any] = {}

    pooled_zw_dual_20pct_hits: List[int] = []
    pooled_zw_pure_0pct_hits: List[int] = []
    pooled_zw_1024_to_384_hits: List[int] = []
    pooled_native_4096_hits: List[int] = []
    pooled_native_384_hits: List[int] = []

    for dom in domain_bundles:
        d_id = dom["domain_id"]
        emb = dom["emb"]
        n_docs = len(dom["eval_doc_ids"])

        rng_bf = np.random.default_rng(int.from_bytes(hashlib.sha256(f"bf20:{d_id}".encode("utf-8")).digest()[:4], "big"))
        bf_mask = np.zeros(n_docs, dtype=bool)
        bf_mask[rng_bf.choice(n_docs, size=max(1, int(round(0.20 * n_docs))), replace=False)] = True

        src_anc_all = emb["Qwen3-4096"]["anchor_docs"].astype(np.float64)
        tgt_anc_all = emb["MiniLM-384"]["anchor_docs"].astype(np.float64)
        n_fit_primary = min(anchors_per_draw, src_anc_all.shape[0])
        src_anc = src_anc_all[:n_fit_primary]
        tgt_anc = tgt_anc_all[:n_fit_primary]

        q_eval_4096 = emb["Qwen3-4096"]["eval_queries"].astype(np.float64)
        q_dev_4096 = emb["Qwen3-4096"]["dev_queries"].astype(np.float64)
        d_old_384 = emb["MiniLM-384"]["eval_docs"].astype(np.float64)
        d_new_4096 = emb["Qwen3-4096"]["eval_docs"].astype(np.float64)
        d_ref_384 = emb["MiniLM-384"]["dev_ref_docs"].astype(np.float64)

        # 1. Full 5x3 Factorial (Mode B 20% backfill)
        fact = run_factorial_5x3(
            domain=d_id,
            src_anchors=src_anc,
            tgt_anchors=tgt_anc,
            eval_queries_src=q_eval_4096,
            eval_queries_tgt=q_eval_4096,
            docs_old=d_old_384,
            docs_new=d_new_4096,
            query_ids=dom["eval_query_ids"],
            doc_ids=dom["eval_doc_ids"],
            qrels=dom["eval_qrels"],
            dev_queries_src=q_dev_4096,
            dev_queries_tgt=q_dev_4096,
            dev_docs=d_ref_384,
            backfill_mask=bf_mask,
        )
        zm_csls = fact[("ZeroMuPolarBridge", "CSLSMerge")]
        zm_zsc = fact[("ZeroMuPolarBridge", "CalibratedZScoreMerge")]
        zm_raw = fact[("ZeroMuPolarBridge", "RawCosineMerge")]
        up_csls = fact[("UncenteredProcrustes", "CSLSMerge")]
        cp_csls = fact[("CenteredProcrustes", "CSLSMerge")]

        csls_minus_zscore[d_id] = round(zm_csls.mean_ndcg_at_10 - zm_zsc.mean_ndcg_at_10, 6)
        zeromu_minus_uncentered_static[d_id] = round(zm_csls.mean_ndcg_at_10 - up_csls.mean_ndcg_at_10, 6)
        zeromu_minus_centered_static[d_id] = round(zm_csls.mean_ndcg_at_10 - cp_csls.mean_ndcg_at_10, 6)

        # 2. Real Prompt-Wrapper Shift (without refitting bridges!)
        q_sh_eval = emb["Qwen3-4096"]["shifted_eval_queries"].astype(np.float64)
        q_sh_dev = emb["Qwen3-4096"]["shifted_dev_queries"].astype(np.float64)
        disp_norm = float(np.linalg.norm(q_sh_eval.mean(axis=0) - q_eval_4096.mean(axis=0)))
        prefix_disp_norms[d_id] = round(disp_norm, 6)

        fact_shift = run_factorial_5x3(
            domain=f"{d_id}::prompt_shift",
            src_anchors=src_anc,
            tgt_anchors=tgt_anc,
            eval_queries_src=q_sh_eval,
            eval_queries_tgt=q_sh_eval,
            docs_old=d_old_384,
            docs_new=d_new_4096,
            query_ids=dom["eval_query_ids"],
            doc_ids=dom["eval_doc_ids"],
            qrels=dom["eval_qrels"],
            dev_queries_src=q_sh_dev,
            dev_queries_tgt=q_sh_dev,
            dev_docs=d_ref_384,
            backfill_mask=bf_mask,
            bridge_names=("UncenteredProcrustes", "CenteredProcrustes", "ZeroMuPolarBridge"),
            merge_names=("CSLSMerge",),
        )
        sh_zm = fact_shift[("ZeroMuPolarBridge", "CSLSMerge")].mean_ndcg_at_10
        sh_up = fact_shift[("UncenteredProcrustes", "CSLSMerge")].mean_ndcg_at_10
        sh_cp = fact_shift[("CenteredProcrustes", "CSLSMerge")].mean_ndcg_at_10
        shift_zeromu_minus_uncentered[d_id] = round(sh_zm - sh_up, 6)
        shift_zeromu_minus_centered[d_id] = round(sh_zm - sh_cp, 6)

        # 3. Real 5-Encoder / 4-Hop Chain (`MiniLM-384 -> ... -> Qwen3-4096`)
        pos_doc_set = {canonicalize_id(did) for rel in dom["eval_qrels"].values() for did, sc in rel.items() if sc > 0}
        unlabeled_in_pool = [i for i, did in enumerate(dom["eval_doc_ids"]) if canonicalize_id(did) not in pos_doc_set]
        anc_in_pool = unlabeled_in_pool[: min(500, len(unlabeled_in_pool))]
        doc_seq = [emb[e]["eval_docs"].astype(np.float64) for e in enc_names]
        q_seq = [emb[e]["eval_queries"].astype(np.float64) for e in enc_names]
        dev_q_seq = [emb[e]["dev_queries"].astype(np.float64) for e in enc_names]
        dev_d_seq = [emb[e]["dev_ref_docs"].astype(np.float64) for e in enc_names]

        hop4_zm = evaluate_4hop_chain(
            bridge_name="ZeroMuPolarBridge",
            encoder_names=enc_names,
            doc_embeddings_seq=doc_seq,
            query_embeddings_seq=q_seq,
            anchor_indices=anc_in_pool,
            query_ids=dom["eval_query_ids"],
            doc_ids=dom["eval_doc_ids"],
            qrels=dom["eval_qrels"],
            dev_query_embeddings_seq=dev_q_seq,
            dev_docs_seq=dev_d_seq,
        )
        hop4_up = evaluate_4hop_chain(
            bridge_name="UncenteredProcrustes",
            encoder_names=enc_names,
            doc_embeddings_seq=doc_seq,
            query_embeddings_seq=q_seq,
            anchor_indices=anc_in_pool,
            query_ids=dom["eval_query_ids"],
            doc_ids=dom["eval_doc_ids"],
            qrels=dom["eval_qrels"],
            dev_query_embeddings_seq=dev_q_seq,
            dev_docs_seq=dev_d_seq,
        )
        fourhop_zeromu_minus_uncentered[d_id] = round(
            hop4_zm.final_composed_ndcg_at_10 - hop4_up.final_composed_ndcg_at_10, 6
        )
        fourhop_direct_v1_v5_delta[d_id] = round(
            hop4_zm.final_direct_ndcg_at_10 - hop4_up.final_direct_ndcg_at_10, 6
        )

        # 4. 20-Draw 500-Anchor Variance (`ZeroMu + CSLS` vs `UncenteredProcrustes + CSLS`, 0%-write 4096->384)
        draws = sample_anchor_draws(
            domain=d_id,
            doc_ids=dom["anchor_doc_ids"],
            n_draws=n_anchor_draws,
            anchors_per_draw=min(anchors_per_draw, max(2, len(dom["anchor_doc_ids"]) - 20)),
            base_seed=2200,
        )
        actual_anchors_by_dom[d_id] = int(draws[0].actual_anchors)
        csls_merger = CSLSMerge(csls_k=10)
        zero_mask = np.zeros(n_docs, dtype=bool)
        zm_draws_list: List[float] = []
        up_draws_list: List[float] = []
        for dr in draws:
            xs_d = src_anc_all[dr.indices]
            yt_d = tgt_anc_all[dr.indices]
            b_zm = ZeroMuPolarBridge.fit(xs_d, yt_d, scoring_mode="dual_adap")
            b_up = UncenteredProcrustes.fit(xs_d, yt_d)
            r_zm = b_zm.score_matrix(q_eval_4096, d_old_384, dev_docs=d_ref_384)
            ref_zm = b_zm.score_matrix(q_dev_4096, d_old_384, dev_docs=d_ref_384)
            s_zm = csls_merger.merge(
                r_zm, np.zeros_like(r_zm), zero_mask, bridged_ref=ref_zm, native_ref=np.zeros_like(ref_zm), strict_disjoint_dev=True
            )
            r_up = b_up.score_matrix(q_eval_4096, d_old_384)
            ref_up = b_up.score_matrix(q_dev_4096, d_old_384)
            s_up = csls_merger.merge(
                r_up, np.zeros_like(r_up), zero_mask, bridged_ref=ref_up, native_ref=np.zeros_like(ref_up), strict_disjoint_dev=True
            )
            n_zm, _ = evaluate_score_matrix(s_zm, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10)
            n_up, _ = evaluate_score_matrix(s_up, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10)
            zm_draws_list.append(float(np.mean(list(n_zm.values()))))
            up_draws_list.append(float(np.mean(list(n_up.values()))))
        draw_scores_zeromu[d_id] = zm_draws_list
        draw_scores_uncentered[d_id] = up_draws_list

        # 5. Progressive Anchor Contamination Sweep (0%, 1%, 5%, 10%, 20% x 3 modes) with TRA-1 Diagnostics
        sweep_pts = run_contamination_sweep(
            src_anchors=src_anc,
            tgt_anchors=tgt_anc,
            eval_queries_src=q_eval_4096,
            docs_old=d_old_384,
            query_ids=dom["eval_query_ids"],
            doc_ids=dom["eval_doc_ids"],
            qrels=dom["eval_qrels"],
            bridge_names=("UncenteredProcrustes", "CenteredProcrustes", "ZeroMuPolarBridge"),
            merge_name="CSLSMerge",
            rates=(0.0, 0.01, 0.05, 0.10, 0.20),
            modes=("mismatched", "duplicate", "off_domain"),
            seed=2201,
            dev_queries_src=q_dev_4096,
            dev_docs=d_ref_384,
        )
        contamination_by_dom[d_id] = [
            {
                "bridge": pt.bridge_name,
                "mode": pt.contamination_mode,
                "rate": pt.contamination_rate,
                "ndcg10": round(pt.mean_ndcg_at_10, 6),
                "recall10": round(pt.mean_recall_at_10, 6),
                "tra1_status": pt.tra1.tra1_health_status,
                "outlier_fraction": round(pt.tra1.outlier_fraction, 6),
                "duplicate_fraction": round(pt.tra1.duplicate_pair_fraction, 6),
                "rank_ratio": round(pt.tra1.rank_ratio, 6),
                "residual_cosine_mean": round(pt.tra1.residual_cosine_mean, 6),
            }
            for pt in sweep_pts
        ]

        # 6. Matched-pair Hit@10 for Tango NI (#50)
        nat4096_scores = l2_normalize(q_eval_4096) @ l2_normalize(d_new_4096).T
        nat384_scores = l2_normalize(emb["MiniLM-384"]["eval_queries"].astype(np.float64)) @ l2_normalize(d_old_384).T
        zm_0pct = ZeroMuPolarBridge.fit(src_anc, tgt_anc, scoring_mode="dual_adap")
        zw_0pct_scores = zm_0pct.score_matrix(q_eval_4096, d_old_384, dev_docs=d_ref_384)

        zm_1024_384 = ZeroMuPolarBridge.fit(
            emb["bge-m3-1024"]["anchor_docs"].astype(np.float64)[:n_fit_primary],
            tgt_anc,
            scoring_mode="dual_adap",
        )
        zw_1024_384_scores = zm_1024_384.score_matrix(
            emb["bge-m3-1024"]["eval_queries"].astype(np.float64),
            d_old_384,
            dev_docs=d_ref_384,
        )
        ndcg_nat4096, rec_nat4096 = evaluate_score_matrix(
            nat4096_scores, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10
        )
        ndcg_nat384, rec_nat384 = evaluate_score_matrix(
            nat384_scores, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10
        )
        _, rec_zw0 = evaluate_score_matrix(
            zw_0pct_scores, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10
        )
        _, rec_zw1024 = evaluate_score_matrix(
            zw_1024_384_scores, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10
        )
        for raw_qid in dom["eval_query_ids"]:
            qid = canonicalize_id(raw_qid)
            if qid not in rec_nat4096:
                continue
            pooled_zw_dual_20pct_hits.append(1 if zm_csls.per_query_recall.get(qid, 0.0) > 0.0 else 0)
            pooled_zw_pure_0pct_hits.append(1 if rec_zw0.get(qid, 0.0) > 0.0 else 0)
            pooled_zw_1024_to_384_hits.append(1 if rec_zw1024.get(qid, 0.0) > 0.0 else 0)
            pooled_native_4096_hits.append(1 if rec_nat4096.get(qid, 0.0) > 0.0 else 0)
            pooled_native_384_hits.append(1 if rec_nat384.get(qid, 0.0) > 0.0 else 0)

        per_domain_summary[d_id] = {
            "n_eval_docs": n_docs,
            "n_eval_queries": len(rec_nat4096),
            "native_minilm_384_ndcg10": round(float(np.mean(list(ndcg_nat384.values()))), 6),
            "native_qwen3_4096_ndcg10": round(float(np.mean(list(ndcg_nat4096.values()))), 6),
            "factorial_mode_b_20pct_backfill_ndcg10": {
                f"{b}__{m}": round(cell.mean_ndcg_at_10, 6) for (b, m), cell in sorted(fact.items())
            },
            "csls_minus_zscore_zeromu_ndcg10": csls_minus_zscore[d_id],
            "csls_minus_raw_zeromu_ndcg10": round(zm_csls.mean_ndcg_at_10 - zm_raw.mean_ndcg_at_10, 6),
            "zeromu_minus_uncentered_csls_ndcg10": zeromu_minus_uncentered_static[d_id],
            "zeromu_minus_centered_csls_ndcg10": zeromu_minus_centered_static[d_id],
            "prompt_shift_real_wrapper": {
                "prefix_displacement_norm": prefix_disp_norms[d_id],
                "zeromu_csls_ndcg10": round(sh_zm, 6),
                "uncentered_csls_ndcg10": round(sh_up, 6),
                "centered_csls_ndcg10": round(sh_cp, 6),
                "zeromu_minus_uncentered_ndcg10": shift_zeromu_minus_uncentered[d_id],
                "zeromu_minus_centered_ndcg10": shift_zeromu_minus_centered[d_id],
            },
            "four_hop_384_to_4096": {
                "zeromu_composed_ndcg10": round(hop4_zm.final_composed_ndcg_at_10, 6),
                "zeromu_direct_ndcg10": round(hop4_zm.final_direct_ndcg_at_10, 6),
                "zeromu_nonzero_kappa": round(hop4_zm.transitions[-1].composed_nonzero_kappa, 6),
                "uncentered_composed_ndcg10": round(hop4_up.final_composed_ndcg_at_10, 6),
                "uncentered_direct_ndcg10": round(hop4_up.final_direct_ndcg_at_10, 6),
                "zeromu_minus_uncentered_composed_ndcg10": fourhop_zeromu_minus_uncentered[d_id],
            },
        }

    # Summarize 20-draw anchor variance
    var_zm = summarize_anchor_draw_variance(
        draw_scores_zeromu, requested_anchors=anchors_per_draw, actual_anchors_by_domain=actual_anchors_by_dom
    )
    var_up = summarize_anchor_draw_variance(
        draw_scores_uncentered, requested_anchors=anchors_per_draw, actual_anchors_by_domain=actual_anchors_by_dom
    )

    # Aggregate contamination summary across the 5 domains by (bridge, mode, rate)
    contam_agg: Dict[str, Dict[str, Any]] = {}
    grouped_c: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for rows in contamination_by_dom.values():
        for row in rows:
            grouped_c.setdefault((row["bridge"], row["mode"], float(row["rate"])), []).append(row)
    for (b_name, mode, rate), rows in sorted(grouped_c.items()):
        key = f"{b_name}__{mode}__{int(round(rate * 100))}pct"
        contam_agg[key] = {
            "bridge": b_name,
            "mode": mode,
            "rate": rate,
            "mean_ndcg10": round(float(np.mean([r["ndcg10"] for r in rows])), 6),
            "mean_outlier_fraction": round(float(np.mean([r["outlier_fraction"] for r in rows])), 6),
            "mean_duplicate_fraction": round(float(np.mean([r["duplicate_fraction"] for r in rows])), 6),
            "mean_rank_ratio": round(float(np.mean([r["rank_ratio"] for r in rows])), 6),
            "tra1_statuses": [r["tra1_status"] for r in rows],
            "warn_or_fail_fraction": round(
                float(np.mean([1.0 if r["tra1_status"] != "PASS" else 0.0 for r in rows])), 4
            ),
        }

    # Evaluate Item 6B decomposed hypotheses on the 5 new domains (full including real prompt-shift)
    h6b_5dom = evaluate_item6b_hypotheses(
        clean_1hop_csls_vs_zscore_by_domain=csls_minus_zscore,
        static_zeromu_vs_uncentered_at_csls_by_domain=zeromu_minus_uncentered_static,
        static_zeromu_vs_centered_at_csls_by_domain=zeromu_minus_centered_static,
        shift_zeromu_vs_uncentered_at_csls_by_domain=shift_zeromu_minus_uncentered,
        fourhop_zeromu_vs_uncentered_at_csls_by_domain=fourhop_zeromu_minus_uncentered,
        prefix_displacement_norms_by_domain=prefix_disp_norms,
        direct_v1_v5_delta_by_domain=fourhop_direct_v1_v5_delta,
    )

    tango_5dom = {
        "zero_write_4096_to_384_vs_legacy_native_384": compute_exact_tango_ni(
            pooled_zw_pure_0pct_hits, pooled_native_384_hits, ni_margin=PREREGISTERED_NI_MARGIN
        ).to_dict(),
        "zero_write_1024_to_384_vs_legacy_native_384": compute_exact_tango_ni(
            pooled_zw_1024_to_384_hits, pooled_native_384_hits, ni_margin=PREREGISTERED_NI_MARGIN
        ).to_dict(),
        "dual_shard_20pct_4096_to_384_vs_legacy_native_384": compute_exact_tango_ni(
            pooled_zw_dual_20pct_hits, pooled_native_384_hits, ni_margin=PREREGISTERED_NI_MARGIN
        ).to_dict(),
        "zero_write_4096_to_384_vs_full_reindex_4096_ceiling": compute_exact_tango_ni(
            pooled_zw_pure_0pct_hits, pooled_native_4096_hits, ni_margin=PREREGISTERED_NI_MARGIN
        ).to_dict(),
        "dual_shard_20pct_4096_to_384_vs_full_reindex_4096_ceiling": compute_exact_tango_ni(
            pooled_zw_dual_20pct_hits, pooled_native_4096_hits, ni_margin=PREREGISTERED_NI_MARGIN
        ).to_dict(),
    }

    # Pool with prior 7 unburned domains (if present) for the complete 12/12 real-neural-encoder synthesis!
    pooled_12: Optional[Dict[str, Any]] = None
    if prior7_manifest_path is not None and prior7_manifest_path.is_file():
        prior_raw = json.loads(prior7_manifest_path.read_text(encoding="utf-8"))
        prior_tb = prior_raw["track_b_pv7_5encoder_public_corpus_graduation"]
        prior_per_dom: Mapping[str, Any] = prior_tb["per_domain"]

        all12_csls = {**{d: float(v["csls_minus_zscore_zeromu_ndcg10"]) for d, v in prior_per_dom.items()}, **csls_minus_zscore}
        all12_zm_up = {
            **{d: float(v["zeromu_minus_uncentered_csls_ndcg10"]) for d, v in prior_per_dom.items()},
            **zeromu_minus_uncentered_static,
        }
        all12_zm_cp = {
            **{d: float(v["zeromu_minus_centered_csls_ndcg10"]) for d, v in prior_per_dom.items()},
            **zeromu_minus_centered_static,
        }
        all12_4hop = {
            **{d: float(v["four_hop_384_to_4096"]["zeromu_minus_uncentered_composed_ndcg10"]) for d, v in prior_per_dom.items()},
            **fourhop_zeromu_minus_uncentered,
        }

        def _pool_tango_from_contingency(key: str) -> Dict[str, Any]:
            p_t = prior_tb["tango_ni_power_sized_pooled"][key]
            c_t = tango_5dom[key]
            n11 = int(p_t["n11"]) + int(c_t["n11"])
            n10 = int(p_t["n10"]) + int(c_t["n10"])
            n01 = int(p_t["n01"]) + int(c_t["n01"])
            n00 = int(p_t["n00"]) + int(c_t["n00"])
            zw_vec = [1] * n11 + [1] * n10 + [0] * n01 + [0] * n00
            nat_vec = [1] * n11 + [0] * n10 + [1] * n01 + [0] * n00
            return compute_exact_tango_ni(zw_vec, nat_vec, ni_margin=PREREGISTERED_NI_MARGIN).to_dict()

        m_vals = [all12_csls[d] for d in sorted(all12_csls)]
        m_mean, m_lcb, m_ucb = paired_bootstrap_ci(m_vals, n_resamples=2000, alpha=0.05, seed=1601)
        m_pos = sum(1 for v in m_vals if v > 0.0)
        m_p = exact_binomial_upper_tail_p(m_pos, len(m_vals), 0.5)

        su_vals = [all12_zm_up[d] for d in sorted(all12_zm_up)]
        su_mean, su_lcb, su_ucb = paired_bootstrap_ci(su_vals, n_resamples=2000, alpha=0.05, seed=1602)

        sc_vals = [all12_zm_cp[d] for d in sorted(all12_zm_cp)]
        sc_mean, sc_lcb, sc_ucb = paired_bootstrap_ci(sc_vals, n_resamples=2000, alpha=0.05, seed=1603)

        fh_vals = [all12_4hop[d] for d in sorted(all12_4hop)]
        fh_mean, fh_lcb, fh_ucb = paired_bootstrap_ci(fh_vals, n_resamples=2000, alpha=0.05, seed=1604)
        fh_pos = sum(1 for v in fh_vals if v > 0.0)

        pooled_12 = {
            "n_unburned_domains": len(all12_csls),
            "domains": sorted(all12_csls.keys()),
            "prior7_manifest_sha256": sha256_file(prior7_manifest_path),
            "total_pooled_eval_queries": int(prior_tb["total_pooled_eval_queries"]) + len(pooled_native_4096_hits),
            "h6b_merge_12dom": {
                "positive_domains": m_pos,
                "required_positive_domains": 11,
                "sign_test_count": f"{m_pos}/{len(m_vals)}",
                "mean_delta": round(m_mean, 6),
                "lcb_95": round(m_lcb, 6),
                "ucb_95": round(m_ucb, 6),
                "exact_binomial_p": round(m_p, 8),
                "passed_11_of_12_and_lcb_gt_0": bool(m_pos >= 11 and m_lcb > 0.0),
                "per_domain": all12_csls,
            },
            "h6b_bridge_static_12dom": {
                "zeromu_vs_uncentered_csls": {
                    "mean_delta": round(su_mean, 6),
                    "lcb_95": round(su_lcb, 6),
                    "ucb_95": round(su_ucb, 6),
                    "positive_domains": sum(1 for v in su_vals if v > 0.0),
                    "clears_retirement_floor_m010": bool(su_lcb > -0.010),
                    "clears_tight_floor_m005": bool(su_lcb > -0.005),
                    "per_domain": all12_zm_up,
                },
                "zeromu_vs_centered_csls": {
                    "mean_delta": round(sc_mean, 6),
                    "lcb_95": round(sc_lcb, 6),
                    "ucb_95": round(sc_ucb, 6),
                    "per_domain": all12_zm_cp,
                },
            },
            "h6b_4hop_12dom": {
                "positive_domains": fh_pos,
                "sign_test_count": f"{fh_pos}/{len(fh_vals)}",
                "mean_delta": round(fh_mean, 6),
                "lcb_95": round(fh_lcb, 6),
                "ucb_95": round(fh_ucb, 6),
                "per_domain": all12_4hop,
            },
            "tango_ni_12dom_pooled": {
                k: _pool_tango_from_contingency(k)
                for k in (
                    "zero_write_4096_to_384_vs_legacy_native_384",
                    "zero_write_1024_to_384_vs_legacy_native_384",
                    "dual_shard_20pct_4096_to_384_vs_legacy_native_384",
                    "zero_write_4096_to_384_vs_full_reindex_4096_ceiling",
                    "dual_shard_20pct_4096_to_384_vs_full_reindex_4096_ceiling",
                )
            },
        }

    return {
        "n_new_unburned_domains": len(per_domain_summary),
        "new_eval_queries_total": len(pooled_native_4096_hits),
        "per_domain_remaining5": per_domain_summary,
        "item6b_remaining5_full_report": {
            "n_domains": h6b_5dom.n_domains,
            "h6b_merge": h6b_5dom.h6b_merge,
            "h6b_bridge_static": h6b_5dom.h6b_bridge_static,
            "h6b_bridge_shift_and_4hop": h6b_5dom.h6b_bridge_shift_and_4hop,
            "holm_adjusted_p_values": h6b_5dom.holm_adjusted_p_values,
            "overall_verdict": h6b_5dom.overall_verdict,
            "report_sha256": h6b_5dom.report_sha256,
        },
        "anchor_draw_variance_20x500": {
            "zeromu_polar_csls": {
                "cross_domain_n": var_zm.cross_domain_n,
                "cross_domain_mean": round(var_zm.cross_domain_mean, 6),
                "cross_domain_sd": round(var_zm.cross_domain_sd, 6),
                "mean_within_domain_anchor_sd": round(var_zm.mean_within_domain_anchor_sd, 6),
                "per_domain": {
                    d: {
                        "n_draws": s.n_draws,
                        "actual_anchors": s.actual_anchors,
                        "mean": round(s.within_domain_mean, 6),
                        "within_domain_sd": round(s.within_domain_anchor_sd, 6),
                        "min": round(s.min_score, 6),
                        "max": round(s.max_score, 6),
                    }
                    for d, s in var_zm.per_domain.items()
                },
            },
            "uncentered_procrustes_csls": {
                "cross_domain_n": var_up.cross_domain_n,
                "cross_domain_mean": round(var_up.cross_domain_mean, 6),
                "cross_domain_sd": round(var_up.cross_domain_sd, 6),
                "mean_within_domain_anchor_sd": round(var_up.mean_within_domain_anchor_sd, 6),
                "per_domain": {
                    d: {
                        "n_draws": s.n_draws,
                        "actual_anchors": s.actual_anchors,
                        "mean": round(s.within_domain_mean, 6),
                        "within_domain_sd": round(s.within_domain_anchor_sd, 6),
                        "min": round(s.min_score, 6),
                        "max": round(s.max_score, 6),
                    }
                    for d, s in var_up.per_domain.items()
                },
            },
        },
        "contamination_sweep_aggregate": contam_agg,
        "contamination_sweep_per_domain": contamination_by_dom,
        "tango_ni_remaining5": tango_5dom,
        "pooled_12_unburned_domains": pooled_12,
    }


def run_self_test() -> Dict[str, Any]:
    rng = np.random.default_rng(2200)
    fake_bundles: List[Dict[str, Any]] = []
    for plan in REMAINING5_DOMAIN_PLAN[:2]:
        d_id = plan["domain_id"]
        assert normalize_dataset_id(d_id) not in BURNED_CORPORA
        n_d, n_eq, n_dq, n_anc = 120, 40, 16, 80
        doc_ids = [f"{d_id}_d{i}" for i in range(n_d)]
        eq_ids = [f"{d_id}_eq{i}" for i in range(n_eq)]
        dq_ids = [f"{d_id}_dq{i}" for i in range(n_dq)]
        eval_qrels = {eq_ids[i]: {doc_ids[i]: 2.0} for i in range(n_eq)}
        dev_qrels = {dq_ids[i]: {doc_ids[(i + 50) % n_d]: 1.0} for i in range(n_dq)}

        lat_eval_d = l2_normalize(rng.normal(size=(n_d, 24)))
        lat_anc_d = l2_normalize(rng.normal(size=(n_anc, 24)))
        lat_ref_d = l2_normalize(rng.normal(size=(60, 24)))
        lat_eq = l2_normalize(np.vstack([lat_eval_d[i] + 0.25 * rng.normal(size=24) for i in range(n_eq)]))
        lat_dq = l2_normalize(np.vstack([lat_eval_d[(i + 50) % n_d] + 0.25 * rng.normal(size=24) for i in range(n_dq)]))

        emb_map: Dict[str, Dict[str, np.ndarray]] = {}
        for enc_name, _mid, dim, _bs in ENCODER_SPECS:
            d_eff = min(dim, 48)
            proj = rng.normal(size=(24, d_eff)) / math.sqrt(d_eff)
            bias = rng.normal(scale=0.35, size=(1, d_eff))
            d_entry = {
                "eval_docs": l2_normalize(lat_eval_d @ proj + bias).astype(np.float32),
                "anchor_docs": l2_normalize(lat_anc_d @ proj + bias).astype(np.float32),
                "dev_ref_docs": l2_normalize(lat_ref_d @ proj + bias).astype(np.float32),
                "eval_queries": l2_normalize(lat_eq @ proj + bias).astype(np.float32),
                "dev_queries": l2_normalize(lat_dq @ proj + bias).astype(np.float32),
            }
            if enc_name == "Qwen3-4096":
                shift = rng.normal(scale=0.6, size=(1, d_eff))
                d_entry["shifted_eval_queries"] = l2_normalize(lat_eq @ proj + bias + shift).astype(np.float32)
                d_entry["shifted_dev_queries"] = l2_normalize(lat_dq @ proj + bias + shift).astype(np.float32)
            emb_map[enc_name] = d_entry
        fake_bundles.append(
            {
                "domain_id": d_id,
                "normalized_burn_id": normalize_dataset_id(d_id),
                "eval_doc_ids": doc_ids,
                "anchor_doc_ids": [f"{d_id}_a{i}" for i in range(n_anc)],
                "dev_ref_doc_ids": [f"{d_id}_r{i}" for i in range(60)],
                "eval_query_ids": eq_ids,
                "dev_query_ids": dq_ids,
                "eval_qrels": eval_qrels,
                "dev_qrels": dev_qrels,
                "emb": emb_map,
            }
        )
    scrubbed = scrub_embedding_row_collisions_globally(fake_bundles)
    prior7 = REPO_ROOT / "artifacts" / "pv7_colab_l4_closure_20261125" / "colab_l4_closure_manifest.json"
    res = evaluate_remaining5_and_pool_12(
        fake_bundles,
        prior7_manifest_path=prior7 if prior7.is_file() else None,
        n_anchor_draws=4,
        anchors_per_draw=48,
    )
    return {
        "self_test": "PASS",
        "scrubbed": scrubbed,
        "pooled_domains": res["pooled_12_unburned_domains"]["n_unburned_domains"] if res["pooled_12_unburned_domains"] else 0,
        "contam_points": len(res["contamination_sweep_aggregate"]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=Path("/content/beir_cache_w22a"))
    ap.add_argument("--output-dir", type=Path, default=Path("/content/out/pv7_w22a_b54_12dom_complete"))
    ap.add_argument("--git-commit-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    ap.add_argument("--source-archive-sha256", type=str, default=os.environ.get("CHELATED_SRC_TAR_SHA256", "unknown"))
    ap.add_argument("--seed", type=int, default=20261125)
    args = ap.parse_args(argv)

    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Wave 22A Colab L4 execution")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    vram_mib = round(float(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 2)), 2)
    hardware_cohort = f"colab:{gpu_name}"
    print(f"[w22a] hardware_cohort={hardware_cohort} vram_mib={vram_mib} torch={torch.__version__}", flush=True)

    hf_digests: Dict[str, Dict[str, str]] = {}
    domain_bundles: List[Dict[str, Any]] = []
    for plan in REMAINING5_DOMAIN_PLAN:
        sliced, digests = load_beir_slice_from_hf(
            domain_id=plan["domain_id"],
            hf_repo=plan["hf_repo"],
            cache_dir=args.cache_dir,
            max_docs=plan["max_docs"],
            max_eval_q=plan["max_eval_q"],
            max_dev_q=plan["max_dev_q"],
            anchor_pool=plan["anchor_pool"],
            seed=args.seed,
        )
        hf_digests[plan["hf_repo"]] = digests
        sliced["emb"] = {}
        domain_bundles.append(sliced)
        print(
            f"[w22a] Loaded {plan['domain_id']}: eval_docs={len(sliced['eval_doc_ids'])}, "
            f"anchors={len(sliced['anchor_doc_ids'])}, eval_q={len(sliced['eval_query_ids'])}, dev_q={len(sliced['dev_query_ids'])}",
            flush=True,
        )

    # Batched encoding: base texts for all 5 encoders, plus shifted queries for Qwen3-4096
    base_texts: List[str] = []
    base_slices: List[Tuple[int, str, int, int]] = []
    for d_idx, dom in enumerate(domain_bundles):
        for role_key, txt_key in (
            ("eval_docs", "eval_doc_texts"),
            ("anchor_docs", "anchor_doc_texts"),
            ("dev_ref_docs", "dev_ref_doc_texts"),
            ("eval_queries", "eval_query_texts"),
            ("dev_queries", "dev_query_texts"),
        ):
            s0 = len(base_texts)
            base_texts.extend(dom[txt_key])
            base_slices.append((d_idx, role_key, s0, len(base_texts)))

    shift_texts: List[str] = []
    shift_slices: List[Tuple[int, str, int, int]] = []
    for d_idx, dom in enumerate(domain_bundles):
        for role_key, txt_key in (
            ("shifted_eval_queries", "shifted_eval_query_texts"),
            ("shifted_dev_queries", "shifted_dev_query_texts"),
        ):
            s0 = len(shift_texts)
            shift_texts.extend(dom[txt_key])
            shift_slices.append((d_idx, role_key, s0, len(shift_texts)))

    print(
        f"[w22a] Total base segments per encoder: {len(base_texts)}; Qwen3 prompt-shifted queries: {len(shift_texts)}",
        flush=True,
    )

    encoder_prov: Dict[str, Any] = {}
    for enc_name, model_id, expected_dim, batch_sz in ENCODER_SPECS:
        texts_to_encode = base_texts + shift_texts if enc_name == "Qwen3-4096" else base_texts
        print(f"[w22a] Encoding {enc_name} ({model_id}, N={len(texts_to_encode)}, bs={batch_sz}) ...", flush=True)
        vecs, info = encode_all_texts_on_gpu(model_id, texts_to_encode, batch_size=batch_sz, device="cuda")
        if int(vecs.shape[1]) != int(expected_dim):
            raise RuntimeError(f"Dim mismatch on {enc_name}: {vecs.shape[1]} != {expected_dim}")
        encoder_prov[enc_name] = info
        for d_idx, role_key, s0, s1 in base_slices:
            domain_bundles[d_idx]["emb"].setdefault(enc_name, {})[role_key] = vecs[s0:s1]
        if enc_name == "Qwen3-4096":
            offset = len(base_texts)
            for d_idx, role_key, s0, s1 in shift_slices:
                domain_bundles[d_idx]["emb"][enc_name][role_key] = vecs[offset + s0 : offset + s1]
        print(f"[w22a] Finished {enc_name} in {info['wall_seconds']}s (vram={info['peak_vram_mib']} MiB)", flush=True)

    scrubbed = scrub_embedding_row_collisions_globally(domain_bundles)
    print(f"[w22a] Scrubbed post-encode row collisions: {scrubbed}", flush=True)

    prior7_path = REPO_ROOT / "artifacts" / "pv7_colab_l4_closure_20261125" / "colab_l4_closure_manifest.json"
    print(f"[w22a] Evaluating 5x3 factorial, prompt-shift, 4-hop, 20x500 draws, contamination, & 12-domain pool ...", flush=True)
    results = evaluate_remaining5_and_pool_12(
        domain_bundles,
        prior7_manifest_path=prior7_path if prior7_path.is_file() else None,
        n_anchor_draws=20,
        anchors_per_draw=500,
    )

    wall_total = round(time.perf_counter() - t0, 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "colab_l4_w22a_b54_12dom_complete_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit_sha": args.git_commit_sha,
            "source_archive_sha256": args.source_archive_sha256,
            "spine_claim_registry_sha256": claim_registry_sha256(),
            "hardware_cohort": hardware_cohort,
            "is_local_dgx_spark_evidence": False,
            "blind_fixtures_touched": False,
            "gpu_name": gpu_name,
            "gpu_total_vram_mib": vram_mib,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "hf_dataset_file_sha256": hf_digests,
            "row_collisions_scrubbed": scrubbed,
            "encoders": encoder_prov,
            "wall_seconds_total": wall_total,
        },
        "results": results,
    }
    m_path = args.output_dir / "w22a_b54_12dom_manifest.json"
    m_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    m_path.write_bytes(m_bytes)
    m_sha = sha256_bytes(m_bytes)

    receipt = {
        "schema_version": "colab_l4_burst_receipt_v1",
        "shard_id": "w22a_b54_12dom",
        "hardware_cohort": hardware_cohort,
        "git_commit_sha": args.git_commit_sha,
        "source_archive_sha256": args.source_archive_sha256,
        "manifest_file": m_path.name,
        "manifest_sha256": m_sha,
        "exit_code": 0,
        "wall_seconds_total": wall_total,
    }
    (args.output_dir / "w22a_b54_12dom_receipt.json").write_bytes(
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(f"[w22a] DONE sha256={m_sha[:12]} wall={wall_total}s", flush=True)
    if results["pooled_12_unburned_domains"]:
        print(
            json.dumps(
                {
                    "h6b_merge_12dom": {
                        k: v for k, v in results["pooled_12_unburned_domains"]["h6b_merge_12dom"].items() if k != "per_domain"
                    },
                    "h6b_static_12dom": {
                        k: v
                        for k, v in results["pooled_12_unburned_domains"]["h6b_bridge_static_12dom"][
                            "zeromu_vs_uncentered_csls"
                        ].items()
                        if k != "per_domain"
                    },
                    "anchor_variance_zm": {
                        k: v for k, v in results["anchor_draw_variance_20x500"]["zeromu_polar_csls"].items() if k != "per_domain"
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
