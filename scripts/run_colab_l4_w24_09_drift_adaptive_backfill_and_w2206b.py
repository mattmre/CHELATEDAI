#!/usr/bin/env python3
"""Colab L4 Wave 24-09 (Lane 4): 12-Domain Online Prompt-Shift (ZeroMu EMA vs Frozen CenteredProcrustes),
Hybrid CSLS + Margin/Hubness Adaptive Backfill (#50/#54), and W22-06(b) 3 Additional Encoders.

Tests the three high-upside migration lanes that static offline batch tests left open:
  Part A — All 12/12 Unburned Public MTEB/BEIR Domains under Real Online Query / Prompt-Wrapper Drift
           (without refitting the bridge): compares causal online `ZeroMu` query-EMA adaptation against
           frozen `CenteredProcrustes` and `UncenteredProcrustes` (all with CSLS merge).
  Part B — Hybrid `CSLS` + Unlabeled Hubness/Margin-Triggered Adaptive Backfill vs Uniform Random 20%
           Backfill at the EXACT same 20% re-encoding budget (including `dbpedia` & `climate-fever`),
           testing exact pooled `Tango (1998)` NI 95% LCB against -0.0500 and paired nDCG@10 gain.
  Part C — `W22-06(b)` 3 Additional Neural Encoders (`BAAI/bge-base-en-v1.5` 768-d,
           `BAAI/bge-large-en-v1.5` 1024-d, `intfloat/e5-large-v2` 1024-d) measuring CSLS merge gain
           and bridge parity across 3 new encoder architectures on unburned domains.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
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
    evaluate_score_matrix,
    exact_binomial_upper_tail_p,
    l2_normalize,
    paired_bootstrap_ci,
)
from chelated_pv7_spine.burn_ledger import BURNED_CORPORA, normalize_dataset_id  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402

ALL_12_DOMAINS: Tuple[Dict[str, Any], ...] = (
    {"domain_id": "trec-covid", "hf_repo": "mteb/trec-covid", "max_docs": 600, "max_eval_q": 35, "max_dev_q": 15},
    {"domain_id": "touche-2020", "hf_repo": "mteb/touche2020", "max_docs": 600, "max_eval_q": 34, "max_dev_q": 15},
    {"domain_id": "dbpedia", "hf_repo": "mteb/dbpedia", "max_docs": 600, "max_eval_q": 180, "max_dev_q": 60},
    {"domain_id": "climate-fever", "hf_repo": "mteb/climate-fever", "max_docs": 600, "max_eval_q": 180, "max_dev_q": 60},
    {"domain_id": "cqadupstack-android", "hf_repo": "mteb/cqadupstack-android", "max_docs": 600, "max_eval_q": 180, "max_dev_q": 60},
    {"domain_id": "cqadupstack-english", "hf_repo": "mteb/cqadupstack-english", "max_docs": 600, "max_eval_q": 180, "max_dev_q": 60},
    {"domain_id": "cqadupstack-gis", "hf_repo": "mteb/cqadupstack-gis", "max_docs": 600, "max_eval_q": 180, "max_dev_q": 60},
    {"domain_id": "cqadupstack-physics", "hf_repo": "mteb/cqadupstack-physics", "max_docs": 600, "max_eval_q": 180, "max_dev_q": 60},
    {"domain_id": "cqadupstack-programmers", "hf_repo": "mteb/cqadupstack-programmers", "max_docs": 600, "max_eval_q": 180, "max_dev_q": 60},
    {"domain_id": "cqadupstack-stats", "hf_repo": "mteb/cqadupstack-stats", "max_docs": 600, "max_eval_q": 180, "max_dev_q": 60},
    {"domain_id": "cqadupstack-tex", "hf_repo": "mteb/cqadupstack-tex", "max_docs": 600, "max_eval_q": 180, "max_dev_q": 60},
    {"domain_id": "cqadupstack-unix", "hf_repo": "mteb/cqadupstack-unix", "max_docs": 600, "max_eval_q": 180, "max_dev_q": 60},
)

W2206B_EXTRA_ENCODERS: Tuple[Tuple[str, str, int, int], ...] = (
    ("bge-base-768", "BAAI/bge-base-en-v1.5", 768, 128),
    ("bge-large-1024", "BAAI/bge-large-en-v1.5", 1024, 96),
    ("e5-large-1024", "intfloat/e5-large-v2", 1024, 96),
)

PROMPT_SHIFT_PREFIX = (
    "Instruct: Given a technical enterprise search query under live index migration, "
    "retrieve authoritative supporting passages.\nQuery: "
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select_adaptive_backfill_mask(
    old_docs: np.ndarray,
    anchor_old: np.ndarray,
    *,
    backfill_frac: float = 0.20,
    k_hub: int = 10,
) -> np.ndarray:
    """Strictly UNLABELED (zero query-label) adaptive backfill selector.

    Scores each document in the old index by combining two purely unlabeled geometric vulnerability signals:
      1. Local hubness density `r_d`: mean top-k cosine similarity to other old docs (hubs cause CSLS/projection crowding).
      2. Anchor-subspace residual: distance from the anchor-cloud centroid/subspace (OOD documents suffer highest
         bridge distortion when left un-reindexed).
    Selects the exact top `round(backfill_frac * N)` most vulnerable documents for re-indexing.
    """
    x = l2_normalize(old_docs)
    n = x.shape[0]
    n_sel = max(1, int(round(backfill_frac * n)))
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    k_eff = min(k_hub, max(1, n - 1))
    top_k = np.partition(sim, n - k_eff, axis=1)[:, -k_eff:]
    hubness = top_k.mean(axis=1)

    # Distance from anchor cloud mean (z-scored) + hubness (z-scored)
    a_norm = l2_normalize(anchor_old)
    a_mean = a_norm.mean(axis=0)
    a_mean = a_mean / max(float(np.linalg.norm(a_mean)), 1e-12)
    ood_dist = 1.0 - (x @ a_mean)

    def _z(v: np.ndarray) -> np.ndarray:
        s = float(np.std(v))
        return (v - float(np.mean(v))) / (s if s > 1e-12 else 1.0)

    vuln = _z(hubness) + _z(ood_dist)
    top_idx = np.argsort(-vuln, kind="mergesort")[:n_sel]
    mask = np.zeros(n, dtype=bool)
    mask[top_idx] = True
    return mask


def eval_dual_shard_csls(
    *,
    bridge: Any,
    old_docs: np.ndarray,
    new_docs: np.ndarray,
    dev_q_new: np.ndarray,
    eval_q_new: np.ndarray,
    backfill_mask: np.ndarray,
    doc_ids: Sequence[str],
    eval_qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> Dict[str, Any]:
    """Evaluate dual-shard retrieval (`backfill_mask` in native new shard, rest bridged via old shard) with CSLS."""
    s_dev_old = bridge.score_matrix(dev_q_new, old_docs)
    s_eval_old = bridge.score_matrix(eval_q_new, old_docs)

    d_new_n = l2_normalize(new_docs)
    s_dev_new = l2_normalize(dev_q_new) @ d_new_n.T
    s_eval_new = l2_normalize(eval_q_new) @ d_new_n.T

    merger = CSLSMerge(csls_k=10)
    full = merger.merge(
        s_eval_old,
        s_eval_new,
        backfill_mask,
        bridged_ref=s_dev_old,
        native_ref=s_dev_new,
    )
    ndcg_map, recall_map = evaluate_score_matrix(full, eval_qids, doc_ids, qrels, k=10)
    per_q_ndcg = [float(ndcg_map[canonicalize_id(q)]) for q in eval_qids if canonicalize_id(q) in ndcg_map]
    per_q_rec = [float(recall_map[canonicalize_id(q)]) for q in eval_qids if canonicalize_id(q) in recall_map]
    return {
        "ndcg10": float(np.mean(per_q_ndcg)) if per_q_ndcg else 0.0,
        "recall10": float(np.mean(per_q_rec)) if per_q_rec else 0.0,
        "per_query_ndcg10": per_q_ndcg,
    }


def load_mteb_domain(spec: Mapping[str, Any]) -> Dict[str, Any]:
    from datasets import load_dataset

    dom_id = spec["domain_id"]
    norm = normalize_dataset_id(dom_id)
    if norm in BURNED_CORPORA:
        raise RuntimeError(f"Refusing burned corpus {dom_id}")

    hf_repo = spec["hf_repo"]
    qrels_ds = load_dataset(hf_repo, "default", split="test")
    queries_ds = load_dataset(hf_repo, "queries", split="queries")
    corpus_ds = load_dataset(hf_repo, "corpus", split="corpus")

    qrels_all: Dict[str, Dict[str, int]] = {}
    for row in qrels_ds:
        score = int(row.get("score", 0))
        if score <= 0:
            continue
        qid = canonicalize_id(row["query-id"])
        did = canonicalize_id(row["corpus-id"])
        qrels_all.setdefault(qid, {})[did] = score

    q_text_map = {canonicalize_id(r["_id"]): str(r.get("text", "")).strip() for r in queries_ds}
    valid_qids = sorted(q for q in qrels_all if q_text_map.get(q))

    rng = np.random.default_rng(abs(hash(dom_id)) % (2**31 - 1))
    perm = rng.permutation(len(valid_qids))
    valid_qids = [valid_qids[i] for i in perm]

    n_eval = min(int(spec["max_eval_q"]), max(10, int(len(valid_qids) * 0.72)))
    n_dev = min(int(spec["max_dev_q"]), max(5, len(valid_qids) - n_eval))
    eval_qids = valid_qids[:n_eval]
    dev_qids = valid_qids[n_eval : n_eval + n_dev]

    needed_docs = set()
    for q in eval_qids + dev_qids:
        needed_docs.update(qrels_all[q].keys())

    doc_ids: List[str] = []
    doc_texts: List[str] = []
    seen = set()
    # First add all relevant docs, then pad up to max_docs
    corpus_rows = list(corpus_ds)
    for r in corpus_rows:
        did = canonicalize_id(r["_id"])
        if did in needed_docs and did not in seen:
            txt = (str(r.get("title", "")).strip() + " " + str(r.get("text", "")).strip()).strip()
            if txt:
                seen.add(did)
                doc_ids.append(did)
                doc_texts.append(txt[:1600])
    for r in corpus_rows:
        if len(doc_ids) >= int(spec["max_docs"]):
            break
        did = canonicalize_id(r["_id"])
        if did not in seen:
            txt = (str(r.get("title", "")).strip() + " " + str(r.get("text", "")).strip()).strip()
            if txt:
                seen.add(did)
                doc_ids.append(did)
                doc_texts.append(txt[:1600])

    # Keep only queries that have >=1 relevant doc in doc_ids
    doc_set = set(doc_ids)
    qrels = {q: {d: s for d, s in rel.items() if d in doc_set} for q, rel in qrels_all.items()}
    eval_qids = [q for q in eval_qids if qrels.get(q)]
    dev_qids = [q for q in dev_qids if qrels.get(q)]

    return {
        "domain_id": dom_id,
        "doc_ids": doc_ids,
        "doc_texts": doc_texts,
        "eval_qids": eval_qids,
        "eval_q_texts": [q_text_map[q] for q in eval_qids],
        "dev_qids": dev_qids,
        "dev_q_texts": [q_text_map[q] for q in dev_qids],
        "qrels": qrels,
    }


def encode_texts_st(model_id: str, texts: Sequence[str], batch_size: int) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    mdl = SentenceTransformer(
        model_id,
        device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16} if "Qwen3" in model_id else None,
    )
    arr = mdl.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    del mdl
    gc.collect()
    torch.cuda.empty_cache()
    return np.asarray(arr, dtype=np.float64)


def run_self_test() -> Dict[str, Any]:
    rng = np.random.default_rng(2409)
    n_docs, n_q, d_old, d_new = 120, 40, 32, 64
    q_mat = np.linalg.qr(rng.normal(size=(d_new, d_old)))[0][:, :d_old]
    old_d = l2_normalize(rng.normal(size=(n_docs, d_old)))
    new_d = l2_normalize(old_d @ q_mat.T + 0.05 * rng.normal(size=(n_docs, d_new)))
    # Add synthetic prompt-shift offset to eval queries
    shift = l2_normalize(rng.normal(size=(1, d_new))) * 0.45
    dev_q = l2_normalize(new_d[:20] + 0.1 * rng.normal(size=(20, d_new)))
    eval_q = l2_normalize(new_d[20:20 + n_q] + shift + 0.1 * rng.normal(size=(n_q, d_new)))
    mask = select_adaptive_backfill_mask(old_d, old_d[:80], backfill_frac=0.20)
    assert int(mask.sum()) == 24
    doc_ids = [f"d{i}" for i in range(n_docs)]
    qids = [f"q{i}" for i in range(n_q)]
    qrels = {f"q{i}": {f"d{20 + i}": 1} for i in range(n_q)}
    zm = ZeroMuPolarBridge.fit(new_d[:80], old_d[:80], scoring_mode="dual_adap")
    out = eval_dual_shard_csls(
        bridge=zm, old_docs=old_d, new_docs=new_d, dev_q_new=dev_q, eval_q_new=eval_q,
        backfill_mask=mask, doc_ids=doc_ids, eval_qids=qids, qrels=qrels,
    )
    assert out["ndcg10"] > 0.5
    return {"self_test": "PASS", "adaptive_selected": int(mask.sum()), "ndcg10": round(out["ndcg10"], 4)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("/content/out/pv7_w24_09_drift_adaptive_w2206b"))
    ap.add_argument("--git-commit-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    ap.add_argument("--source-archive-sha256", type=str, default=os.environ.get("CHELATED_SRC_TAR_SHA256", "unknown"))
    args = ap.parse_args(argv)

    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Wave 24-09")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    hardware_cohort = f"colab:{gpu_name}"
    print(f"[w24-09] hardware_cohort={hardware_cohort} torch={torch.__version__}", flush=True)

    # 1. Load all 12 unburned domains
    domains = []
    for spec in ALL_12_DOMAINS:
        dom = load_mteb_domain(spec)
        domains.append(dom)
        print(f"[w24-09] loaded {dom['domain_id']}: docs={len(dom['doc_ids'])} eval_q={len(dom['eval_qids'])} dev_q={len(dom['dev_qids'])}", flush=True)

    # Flatten texts per encoder so each encoder loads ONCE on GPU
    def encode_all_domains(model_id: str, batch_size: int, include_shifted: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        from sentence_transformers import SentenceTransformer

        print(f"[w24-09] Loading encoder {model_id} ...", flush=True)
        mdl = SentenceTransformer(
            model_id,
            device="cuda",
            model_kwargs={"torch_dtype": torch.bfloat16} if "Qwen3" in model_id else None,
        )
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for dom in domains:
            did = dom["domain_id"]
            d_emb = mdl.encode(dom["doc_texts"], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            dq_emb = mdl.encode(dom["dev_q_texts"], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            eq_emb = mdl.encode(dom["eval_q_texts"], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            entry = {
                "docs": np.asarray(d_emb, dtype=np.float64),
                "dev_q": np.asarray(dq_emb, dtype=np.float64),
                "eval_q": np.asarray(eq_emb, dtype=np.float64),
            }
            if include_shifted:
                sh_texts = [PROMPT_SHIFT_PREFIX + q for q in dom["eval_q_texts"]]
                sq_emb = mdl.encode(sh_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
                entry["eval_q_shifted"] = np.asarray(sq_emb, dtype=np.float64)
            out[did] = entry
        del mdl
        gc.collect()
        torch.cuda.empty_cache()
        return out

    old_embs = encode_all_domains("sentence-transformers/all-MiniLM-L6-v2", 256, include_shifted=False)
    new_embs = encode_all_domains("Qwen/Qwen3-Embedding-8B", 48, include_shifted=True)

    # Part A & Part B across all 12 domains
    part_a_domains: List[Dict[str, Any]] = []
    part_b_domains: List[Dict[str, Any]] = []
    pooled_unif_hits: List[int] = []
    pooled_adap_hits: List[int] = []
    pooled_oracle_hits: List[int] = []

    for dom in domains:
        did = dom["domain_id"]
        od = old_embs[did]["docs"]
        nd = new_embs[did]["docs"]
        dq = new_embs[did]["dev_q"]
        eq = new_embs[did]["eval_q"]
        sq = new_embs[did]["eval_q_shifted"]
        n_docs = len(dom["doc_ids"])

        # Fit bridges on 400 deterministic anchors (no query labels)
        rng = np.random.default_rng(1400 + abs(hash(did)) % 10000)
        n_anc = min(450, n_docs)
        anc_idx = rng.choice(n_docs, size=n_anc, replace=False)

        zm = ZeroMuPolarBridge.fit(nd[anc_idx], od[anc_idx], scoring_mode="dual_adap")
        cp = CenteredProcrustes.fit(nd[anc_idx], od[anc_idx])
        up = UncenteredProcrustes.fit(nd[anc_idx], od[anc_idx])

        # Uniform 20% random backfill mask vs Unlabeled Adaptive 20% backfill mask
        unif_mask = np.zeros(n_docs, dtype=bool)
        unif_mask[rng.choice(n_docs, size=max(1, int(round(0.20 * n_docs))), replace=False)] = True
        adap_mask = select_adaptive_backfill_mask(od, od[anc_idx], backfill_frac=0.20)

        # Part A: Online Prompt-Shift without refitting (ZeroMu causal EMA vs frozen CenteredProcrustes vs Uncentered)
        r_zm_shift = eval_dual_shard_csls(
            bridge=zm, old_docs=od, new_docs=nd, dev_q_new=dq, eval_q_new=sq,
            backfill_mask=unif_mask, doc_ids=dom["doc_ids"], eval_qids=dom["eval_qids"], qrels=dom["qrels"],
        )
        r_cp_shift = eval_dual_shard_csls(
            bridge=cp, old_docs=od, new_docs=nd, dev_q_new=dq, eval_q_new=sq,
            backfill_mask=unif_mask, doc_ids=dom["doc_ids"], eval_qids=dom["eval_qids"], qrels=dom["qrels"],
        )
        r_up_shift = eval_dual_shard_csls(
            bridge=up, old_docs=od, new_docs=nd, dev_q_new=dq, eval_q_new=sq,
            backfill_mask=unif_mask, doc_ids=dom["doc_ids"], eval_qids=dom["eval_qids"], qrels=dom["qrels"],
        )
        d_zm_cp = r_zm_shift["ndcg10"] - r_cp_shift["ndcg10"]
        d_zm_up = r_zm_shift["ndcg10"] - r_up_shift["ndcg10"]
        part_a_domains.append({
            "domain_id": did,
            "n_eval_q": len(dom["eval_qids"]),
            "zeromu_ema_shift_ndcg10": round(r_zm_shift["ndcg10"], 5),
            "centered_frozen_shift_ndcg10": round(r_cp_shift["ndcg10"], 5),
            "uncentered_frozen_shift_ndcg10": round(r_up_shift["ndcg10"], 5),
            "delta_zeromu_vs_centered_pp": round(d_zm_cp * 100.0, 3),
            "delta_zeromu_vs_uncentered_pp": round(d_zm_up * 100.0, 3),
        })

        # Part B: Adaptive 20% Backfill vs Uniform 20% Backfill vs Native Oracle (clean eval queries)
        r_unif = eval_dual_shard_csls(
            bridge=zm, old_docs=od, new_docs=nd, dev_q_new=dq, eval_q_new=eq,
            backfill_mask=unif_mask, doc_ids=dom["doc_ids"], eval_qids=dom["eval_qids"], qrels=dom["qrels"],
        )
        r_adap = eval_dual_shard_csls(
            bridge=zm, old_docs=od, new_docs=nd, dev_q_new=dq, eval_q_new=eq,
            backfill_mask=adap_mask, doc_ids=dom["doc_ids"], eval_qids=dom["eval_qids"], qrels=dom["qrels"],
        )
        oracle_scores = l2_normalize(eq) @ l2_normalize(nd).T
        o_ndcg_map, _ = evaluate_score_matrix(oracle_scores, dom["eval_qids"], dom["doc_ids"], dom["qrels"], k=10)
        o_per_q = [float(o_ndcg_map[canonicalize_id(q)]) for q in dom["eval_qids"] if canonicalize_id(q) in o_ndcg_map]
        o_mean = float(np.mean(o_per_q)) if o_per_q else 0.0
        # Binary hit@10 (ndcg10 > 0) for exact Tango NI paired test
        u_hits = [int(v > 0.0) for v in r_unif["per_query_ndcg10"]]
        a_hits = [int(v > 0.0) for v in r_adap["per_query_ndcg10"]]
        o_hits = [int(v > 0.0) for v in o_per_q]
        pooled_unif_hits.extend(u_hits)
        pooled_adap_hits.extend(a_hits)
        pooled_oracle_hits.extend(o_hits)

        part_b_domains.append({
            "domain_id": did,
            "n_eval_q": len(dom["eval_qids"]),
            "uniform_20_ndcg10": round(r_unif["ndcg10"], 5),
            "adaptive_20_ndcg10": round(r_adap["ndcg10"], 5),
            "oracle_ndcg10": round(o_mean, 5),
            "delta_adaptive_vs_uniform_pp": round((r_adap["ndcg10"] - r_unif["ndcg10"]) * 100.0, 3),
            "tango_ni_uniform": dataclasses.asdict(compute_exact_tango_ni(u_hits, o_hits, ni_margin=PREREGISTERED_NI_MARGIN)),
            "tango_ni_adaptive": dataclasses.asdict(compute_exact_tango_ni(a_hits, o_hits, ni_margin=PREREGISTERED_NI_MARGIN)),
        })
        print(
            f"[w24-09] {did:24s} | Shift ZM-CP={d_zm_cp*100:+.2f}pp ZM-UP={d_zm_up*100:+.2f}pp | "
            f"Backfill Adap-Unif={(r_adap['ndcg10']-r_unif['ndcg10'])*100:+.2f}pp",
            flush=True,
        )

    # Aggregate Part A & Part B
    zm_cp_deltas = [r["delta_zeromu_vs_centered_pp"] for r in part_a_domains]
    zm_up_deltas = [r["delta_zeromu_vs_uncentered_pp"] for r in part_a_domains]
    adap_deltas = [r["delta_adaptive_vs_uniform_pp"] for r in part_b_domains]

    part_a_summary = {
        "n_domains": len(part_a_domains),
        "zeromu_vs_centered_mean_pp": round(float(np.mean(zm_cp_deltas)), 3),
        "zeromu_vs_centered_pos_domains": sum(1 for d in zm_cp_deltas if d > 0.0),
        "zeromu_vs_centered_ci95_pp": [round(float(x), 4) for x in paired_bootstrap_ci(zm_cp_deltas, seed=24091)],
        "zeromu_vs_centered_binom_p": round(exact_binomial_upper_tail_p(sum(1 for d in zm_cp_deltas if d > 0.0), len(zm_cp_deltas)), 6),
        "zeromu_vs_uncentered_mean_pp": round(float(np.mean(zm_up_deltas)), 3),
        "zeromu_vs_uncentered_pos_domains": sum(1 for d in zm_up_deltas if d > 0.0),
        "zeromu_vs_uncentered_ci95_pp": [round(float(x), 4) for x in paired_bootstrap_ci(zm_up_deltas, seed=24092)],
        "domains": part_a_domains,
    }
    part_b_summary = {
        "n_domains": len(part_b_domains),
        "n_pooled_queries": len(pooled_oracle_hits),
        "adaptive_vs_uniform_mean_pp": round(float(np.mean(adap_deltas)), 3),
        "adaptive_vs_uniform_pos_domains": sum(1 for d in adap_deltas if d > 0.0),
        "adaptive_vs_uniform_ci95_pp": [round(float(x), 4) for x in paired_bootstrap_ci(adap_deltas, seed=24093)],
        "pooled_tango_ni_uniform": dataclasses.asdict(compute_exact_tango_ni(pooled_unif_hits, pooled_oracle_hits, ni_margin=PREREGISTERED_NI_MARGIN)),
        "pooled_tango_ni_adaptive": dataclasses.asdict(compute_exact_tango_ni(pooled_adap_hits, pooled_oracle_hits, ni_margin=PREREGISTERED_NI_MARGIN)),
        "domains": part_b_domains,
    }

    # Part C: W22-06(b) 3 Additional Encoders on 4 representative unburned domains (dbpedia, climate-fever, trec-covid, cqadupstack-unix)
    subset_ids = ("trec-covid", "dbpedia", "climate-fever", "cqadupstack-unix")
    subset_doms = [d for d in domains if d["domain_id"] in subset_ids]
    part_c_results: List[Dict[str, Any]] = []
    for enc_name, enc_id, dim, bs in W2206B_EXTRA_ENCODERS:
        from sentence_transformers import SentenceTransformer

        print(f"[w24-09:C] W22-06(b) encoding 4 domains with {enc_name} ({enc_id}) ...", flush=True)
        mdl = SentenceTransformer(enc_id, device="cuda")
        enc_rows = []
        for dom in subset_doms:
            did = dom["domain_id"]
            od = old_embs[did]["docs"]
            nd = np.asarray(mdl.encode(dom["doc_texts"], batch_size=bs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True), dtype=np.float64)
            dq = np.asarray(mdl.encode(dom["dev_q_texts"], batch_size=bs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True), dtype=np.float64)
            eq = np.asarray(mdl.encode(dom["eval_q_texts"], batch_size=bs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True), dtype=np.float64)

            n_docs = len(dom["doc_ids"])
            rng = np.random.default_rng(24060 + abs(hash(did + enc_name)) % 10000)
            anc_idx = rng.choice(n_docs, size=min(450, n_docs), replace=False)
            mask = np.zeros(n_docs, dtype=bool)
            mask[rng.choice(n_docs, size=max(1, int(round(0.20 * n_docs))), replace=False)] = True

            zm = ZeroMuPolarBridge.fit(nd[anc_idx], od[anc_idx], scoring_mode="dual_adap")
            cp = CenteredProcrustes.fit(nd[anc_idx], od[anc_idx])
            # Compare RawCos vs CSLS merge under ZeroMu, and ZeroMu vs Centered under CSLS
            s_eval_old = zm.score_matrix(eq, od)
            s_eval_new = l2_normalize(eq) @ l2_normalize(nd).T
            raw_full = np.where(mask[None, :], s_eval_new, s_eval_old)
            raw_ndcg_map, _ = evaluate_score_matrix(raw_full, dom["eval_qids"], dom["doc_ids"], dom["qrels"], k=10)
            raw_mean = float(np.mean(list(raw_ndcg_map.values()))) if raw_ndcg_map else 0.0
            r_zm_csls = eval_dual_shard_csls(
                bridge=zm, old_docs=od, new_docs=nd, dev_q_new=dq, eval_q_new=eq,
                backfill_mask=mask, doc_ids=dom["doc_ids"], eval_qids=dom["eval_qids"], qrels=dom["qrels"],
            )
            r_cp_csls = eval_dual_shard_csls(
                bridge=cp, old_docs=od, new_docs=nd, dev_q_new=dq, eval_q_new=eq,
                backfill_mask=mask, doc_ids=dom["doc_ids"], eval_qids=dom["eval_qids"], qrels=dom["qrels"],
            )
            enc_rows.append({
                "domain_id": did,
                "raw_ndcg10": round(raw_mean, 5),
                "zeromu_csls_ndcg10": round(r_zm_csls["ndcg10"], 5),
                "centered_csls_ndcg10": round(r_cp_csls["ndcg10"], 5),
                "csls_gain_pp": round((r_zm_csls["ndcg10"] - raw_mean) * 100.0, 3),
                "zeromu_vs_centered_pp": round((r_zm_csls["ndcg10"] - r_cp_csls["ndcg10"]) * 100.0, 3),
            })
        del mdl
        gc.collect()
        torch.cuda.empty_cache()
        part_c_results.append({
            "encoder_name": enc_name,
            "model_id": enc_id,
            "dim": dim,
            "mean_csls_gain_pp": round(float(np.mean([r["csls_gain_pp"] for r in enc_rows])), 3),
            "mean_zeromu_vs_centered_pp": round(float(np.mean([r["zeromu_vs_centered_pp"] for r in enc_rows])), 3),
            "domains": enc_rows,
        })
        print(
            f"[w24-09:C] {enc_name} DONE | mean_CSLS_gain={part_c_results[-1]['mean_csls_gain_pp']:+.2f}pp | "
            f"ZM-CP={part_c_results[-1]['mean_zeromu_vs_centered_pp']:+.2f}pp",
            flush=True,
        )

    wall_total = round(time.perf_counter() - t0, 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "colab_l4_w24_09_drift_adaptive_w2206b_v1",
        "wave_item_id": "W24-09",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit_sha": args.git_commit_sha,
            "source_archive_sha256": args.source_archive_sha256,
            "spine_claim_registry_sha256": claim_registry_sha256(),
            "hardware_cohort": hardware_cohort,
            "gpu_name": gpu_name,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "wall_seconds_total": wall_total,
        },
        "part_a_12domain_online_prompt_shift": part_a_summary,
        "part_b_adaptive_vs_uniform_backfill": part_b_summary,
        "part_c_w22_06b_three_extra_encoders": part_c_results,
    }
    m_path = args.output_dir / "w24_09_manifest.json"
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    m_path.write_bytes(raw)
    print(
        f"[w24-09] DONE sha256={sha256_bytes(raw)[:12]} wall={wall_total}s | "
        f"PartA_ZM_vs_CP={part_a_summary['zeromu_vs_centered_mean_pp']:+.2f}pp ({part_a_summary['zeromu_vs_centered_pos_domains']}/12) | "
        f"PartB_Adap_vs_Unif={part_b_summary['adaptive_vs_uniform_mean_pp']:+.2f}pp ({part_b_summary['adaptive_vs_uniform_pos_domains']}/12)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
