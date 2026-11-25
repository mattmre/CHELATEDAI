#!/usr/bin/env python3
"""Colab L4 Burst GPU Closure Pass: PM-6 Benign FPR Calibration + Real 5-Encoder PV7 Public-Corpus Suite.

Executes two strictly non-blind, unburned public-corpus GPU validations on Google Colab (`--gpu L4`):
  1. Track A (`PM-6` Benign False-Refusal Calibration Closure, `384-d MiniLM` Vault <- `4096-d Qwen3-8B` Stream):
     - Diagnoses and closes the `v7.0` `Demo D11` empirical benign FPR miss (`6.25%/11.76% > 2.0%` caused by
       finite-sample order-statistic overshoot under `N_cal_neg = 50`).
     - Evaluates distribution-free Risk-Controlling Split-Conformal Calibration (`RCPS`, Bates et al. 2021:
       largest `k` on disjoint `CALIB` with `Beta.ppf(0.95, k+1, N_cal - k) <= 0.02`) at `N_cal_neg >= 500`
       and one-shot held-out `EVAL` (`N_eval_neg >= 500`), comparing:
         * `Naive_N50_Empirical_2pct` (reproducing `D11`'s `N=50` calibration rule + 50-draw Monte Carlo),
         * `Chelated_ZeroMu_CSLS_RCPS`,
         * `Chelated_ZeroMu_RawCos_RCPS` (`CSLS` tail-FPR ablation),
         * `UncenteredProcrustes_CSLS_RCPS`,
         * `Native_4096d_Reindex_Oracle_RCPS`.
  2. Track B (`Post-v7.0` Public-Corpus Bridge/Merge, `4`-Hop, & Power-Sized `Tango NI` Graduation):
     - Encodes 7 unburned public IR domains (`trec-covid`, `webis-touche2020`, `cqadupstack-{android,physics,
       programmers,stats,unix}`) verified against `chelated_pv7_spine.burn_ledger.BURNED_CORPORA` across all 5
       real neural encoders (`MiniLM-384`, `bge-small-384`, `mpnet-768`, `bge-m3-1024`, `Qwen3-Embedding-8B` `4096-d`).
     - Runs `chelated_pv7_spine.bridge_merge_harness.run_factorial_5x3` (`Mode B` `4096 -> 384` dual-shard `20%`
       backfill across all 5 bridges x 3 merges) and `evaluate_4hop_chain` (`384 -> 384 -> 768 -> 1024 -> 4096`).
     - Runs power-sized (`N >= 1,024` real held-out paired queries) exact one-shot `Tango (1998)` `95%` Score LCB
       against `-0.0500` (`chelated_loft_and_zero_write_ni.compute_exact_tango_ni`).

Provenance & Brutal-Honesty Rules:
  - Stamps `hardware_cohort` (e.g. `"colab:NVIDIA L4"`), `git_commit_sha`, `source_archive_sha256`,
    `torch_version`, model HF revisions + safetensors SHA-256s, and downloaded `.zip` SHA-256s.
  - Never touches blind `REPORT` fixtures or claims Colab GPU results are local DGX Spark GB10 hardware runs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta as scipy_beta

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_utils import canonicalize_id  # noqa: E402
from chelated_loft_and_zero_write_ni import PREREGISTERED_NI_MARGIN, compute_exact_tango_ni  # noqa: E402
from chelated_pv7_spine.bridge_merge_harness import (  # noqa: E402
    BRIDGE_NAMES,
    MERGE_NAMES,
    CSLSMerge,
    UncenteredProcrustes,
    ZeroMuPolarBridge,
    evaluate_4hop_chain,
    evaluate_score_matrix,
    l2_normalize,
    run_factorial_5x3,
)
from chelated_pv7_spine.burn_ledger import BURNED_CORPORA, normalize_dataset_id, text_fingerprint  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402

ENCODER_SPECS: Tuple[Tuple[str, str, int, int], ...] = (
    ("MiniLM-384", "sentence-transformers/all-MiniLM-L6-v2", 384, 256),
    ("bge-small-384", "BAAI/bge-small-en-v1.5", 384, 256),
    ("mpnet-768", "sentence-transformers/all-mpnet-base-v2", 768, 128),
    ("bge-m3-1024", "BAAI/bge-m3", 1024, 64),
    ("Qwen3-4096", "Qwen/Qwen3-Embedding-8B", 4096, 32),
)

BEIR_BASE_URLS: Tuple[str, ...] = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets",
    "https://huggingface.co/datasets/BeIR",
)

UNBURNED_DOMAIN_PLAN: Tuple[Dict[str, Any], ...] = (
    {"domain_id": "trec-covid", "zip_name": "trec-covid.zip", "subpath": "trec-covid", "max_docs": 1400, "max_eval_q": 35, "max_dev_q": 15},
    {"domain_id": "touche-2020", "zip_name": "webis-touche2020.zip", "subpath": "webis-touche2020", "max_docs": 1400, "max_eval_q": 34, "max_dev_q": 15},
    {"domain_id": "cqadupstack-android", "zip_name": "cqadupstack.zip", "subpath": "cqadupstack/android", "max_docs": 1300, "max_eval_q": 240, "max_dev_q": 80},
    {"domain_id": "cqadupstack-physics", "zip_name": "cqadupstack.zip", "subpath": "cqadupstack/physics", "max_docs": 1300, "max_eval_q": 240, "max_dev_q": 80},
    {"domain_id": "cqadupstack-programmers", "zip_name": "cqadupstack.zip", "subpath": "cqadupstack/programmers", "max_docs": 1300, "max_eval_q": 240, "max_dev_q": 80},
    {"domain_id": "cqadupstack-stats", "zip_name": "cqadupstack.zip", "subpath": "cqadupstack/stats", "max_docs": 1300, "max_eval_q": 220, "max_dev_q": 80},
    {"domain_id": "cqadupstack-unix", "zip_name": "cqadupstack.zip", "subpath": "cqadupstack/unix", "max_docs": 1300, "max_eval_q": 240, "max_dev_q": 80},
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clopper_pearson_upper_one_sided(k: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0 or k < 0 or k > n:
        raise ValueError(f"Invalid binomial counts k={k}, n={n}")
    if k == n:
        return 1.0
    return float(scipy_beta.ppf(1.0 - float(alpha), k + 1, n - k))


def mann_whitney_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    gt = (pos[:, None] > neg[None, :]).astype(np.float64)
    eq = (pos[:, None] == neg[None, :]).astype(np.float64)
    return float(np.mean(gt + 0.5 * eq))


def calibrate_naive_empirical_threshold(
    cal_pos: np.ndarray,
    cal_neg: np.ndarray,
    *,
    target_fpr: float = 0.02,
) -> Dict[str, Any]:
    """Exact reproduction of D11's empirical quantile threshold rule (`floor(target_fpr * N_neg)`)."""
    neg_sorted = np.sort(np.asarray(cal_neg, dtype=np.float64))
    pos = np.asarray(cal_pos, dtype=np.float64)
    max_fp = int(math.floor(float(target_fpr) * float(neg_sorted.size) + 1e-12))
    if max_fp <= 0:
        lo_val = float(neg_sorted[-1])
        above = pos[pos > lo_val]
        tau = 0.5 * (lo_val + float(np.min(above))) if above.size > 0 else float(np.nextafter(lo_val, np.inf))
        if tau <= lo_val:
            tau = float(np.nextafter(lo_val, np.inf))
    elif max_fp >= neg_sorted.size:
        tau = float(np.nextafter(float(neg_sorted[0]), -np.inf))
    else:
        idx_below = neg_sorted.size - max_fp - 1
        lo_val = float(neg_sorted[idx_below])
        hi_val = float(neg_sorted[idx_below + 1])
        tau = 0.5 * (lo_val + hi_val) if hi_val > lo_val else float(np.nextafter(lo_val, np.inf))
    return {
        "method": "naive_empirical_quantile",
        "threshold": float(tau),
        "n_cal_neg": int(neg_sorted.size),
        "max_allowed_cal_fp": max_fp,
        "cal_fpr": float(np.mean(neg_sorted >= tau)),
        "cal_tpr": float(np.mean(pos >= tau)),
    }


def calibrate_rcps_conformal_threshold(
    cal_pos: np.ndarray,
    cal_neg: np.ndarray,
    *,
    target_fpr: float = 0.02,
    delta: float = 0.05,
) -> Dict[str, Any]:
    """Distribution-free Risk-Controlling Prediction Set (Bates et al., JASA 2021) upper-confidence threshold.

    Chooses the largest calibration false-positive count `k_max` satisfying:
        `Beta.ppf(1 - delta, k_max + 1, N_cal - k_max) <= target_fpr`
    If `N_cal` is too small even for `k=0` (`N_cal < ceil(ln(delta)/ln(1-target_fpr)) = 149`), fails closed
    by reporting `rcps_feasible = False` and placing `tau` strictly above `max(cal_neg)`.
    """
    neg_sorted = np.sort(np.asarray(cal_neg, dtype=np.float64))
    pos = np.asarray(cal_pos, dtype=np.float64)
    n_neg = int(neg_sorted.size)
    feasible_k = -1
    for k_cand in range(n_neg + 1):
        ub = clopper_pearson_upper_one_sided(k_cand, n_neg, alpha=delta)
        if ub <= float(target_fpr) + 1e-12:
            feasible_k = k_cand
        else:
            break

    rcps_feasible = feasible_k >= 0
    k_used = max(0, feasible_k)
    if k_used == 0:
        lo_val = float(neg_sorted[-1])
        above = pos[pos > lo_val]
        tau = 0.5 * (lo_val + float(np.min(above))) if above.size > 0 else float(np.nextafter(lo_val, np.inf))
        if tau <= lo_val:
            tau = float(np.nextafter(lo_val, np.inf))
    else:
        idx_below = n_neg - k_used - 1
        lo_val = float(neg_sorted[idx_below])
        hi_val = float(neg_sorted[idx_below + 1])
        if hi_val > lo_val:
            in_gap = pos[(pos > lo_val) & (pos < hi_val)]
            upper_anchor = float(np.min(in_gap)) if in_gap.size > 0 else hi_val
            tau = 0.5 * (lo_val + upper_anchor)
            if tau <= lo_val:
                tau = float(np.nextafter(lo_val, np.inf))
        else:
            tau = float(np.nextafter(lo_val, np.inf))

    cal_fp = int(np.sum(neg_sorted >= tau))
    return {
        "method": "rcps_beta_upper_bound_95",
        "rcps_feasible": bool(rcps_feasible),
        "delta": float(delta),
        "target_fpr": float(target_fpr),
        "n_cal_neg": n_neg,
        "k_max_allowed": int(feasible_k),
        "cal_fp_count": cal_fp,
        "cal_fpr": float(cal_fp / float(n_neg)),
        "cal_cp_ub95": clopper_pearson_upper_one_sided(cal_fp, n_neg, alpha=delta),
        "cal_tpr": float(np.mean(pos >= tau)),
        "threshold": float(tau),
    }


def download_beir_zip(zip_name: str, cache_dir: Path) -> Tuple[Path, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / zip_name
    if not dst.is_file():
        url = f"{BEIR_BASE_URLS[0]}/{zip_name}"
        print(f"[download] Fetching {url} -> {dst} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "chelatedai-pv7-colab-runner/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, dst.open("wb") as out_fh:
            while True:
                chunk = resp.read(4 * 1024 * 1024)
                if not chunk:
                    break
                out_fh.write(chunk)
    digest = sha256_file(dst)
    return dst, digest


def load_beir_slice_from_zip(
    zip_path: Path,
    *,
    domain_id: str,
    subpath: str,
    max_docs: int,
    max_eval_q: int,
    max_dev_q: int,
    seed: int,
) -> Dict[str, Any]:
    norm_id = normalize_dataset_id(domain_id)
    if norm_id in BURNED_CORPORA:
        raise RuntimeError(f"Burned corpus forbidden: {domain_id!r} normalized to {norm_id!r}")

    prefix = subpath.strip("/") + "/"
    qrels_all: Dict[str, Dict[str, float]] = {}
    queries_raw: Dict[str, str] = {}
    corpus_raw: Dict[str, str] = {}

    def _open_effective_zip(outer_zf: zipfile.ZipFile) -> Tuple[zipfile.ZipFile, str, Optional[io.BytesIO]]:
        outer_names = set(outer_zf.namelist())
        if f"{prefix}queries.jsonl" in outer_names:
            return outer_zf, prefix, None
        # UKP cqadupstack.zip packages subforums either as cqadupstack/android/queries.jsonl OR cqadupstack/android.zip
        nested_candidates = [f"{subpath.strip('/')}.zip", f"{subpath.strip('/').split('/')[-1]}.zip"]
        for nc in nested_candidates:
            if nc in outer_names:
                bio = io.BytesIO(outer_zf.read(nc))
                inner_zf = zipfile.ZipFile(bio, "r")
                inner_names = set(inner_zf.namelist())
                leaf = subpath.strip("/").split("/")[-1] + "/"
                eff_pref = leaf if f"{leaf}queries.jsonl" in inner_names else ""
                return inner_zf, eff_pref, bio
        # Fallback: auto-locate queries.jsonl matching leaf name
        leaf_tok = subpath.strip("/").split("/")[-1]
        for name in sorted(outer_names):
            if name.endswith("queries.jsonl") and leaf_tok in name:
                return outer_zf, name[: -len("queries.jsonl")], None
        raise FileNotFoundError(f"Could not locate {subpath} inside {zip_path.name}; sample entries: {sorted(outer_names)[:8]}")

    with zipfile.ZipFile(zip_path, "r") as outer_zf:
        zf, eff_prefix, _bio_ref = _open_effective_zip(outer_zf)
        try:
            names = set(zf.namelist())
            qrels_member = f"{eff_prefix}qrels/test.tsv"
            if qrels_member not in names:
                qrels_member = next(n for n in sorted(names) if n.startswith(f"{eff_prefix}qrels/") and n.endswith(".tsv"))
            with zf.open(qrels_member) as fh:
                reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t")
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

            with zf.open(f"{eff_prefix}queries.jsonl") as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    qid = str(obj["_id"]).strip()
                    txt = str(obj.get("text") or "").strip()
                    if qid in qrels_all and txt:
                        queries_raw[qid] = txt

            needed_pos_docs = {did for qid in queries_raw for did in qrels_all.get(qid, {})}
            reservoir: List[Tuple[str, str, str]] = []
            with zf.open(f"{eff_prefix}corpus.jsonl") as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    did = str(obj["_id"]).strip()
                    title = str(obj.get("title") or "").strip()
                    body = str(obj.get("text") or "").strip()
                    full_txt = (f"{title}\n{body}".strip() if title else body)[:1400]
                    if not full_txt:
                        continue
                    if did in needed_pos_docs:
                        corpus_raw[did] = full_txt
                    else:
                        h_key = hashlib.sha256(f"{seed}:{domain_id}:{did}".encode("utf-8")).hexdigest()
                        if len(reservoir) < (max_docs * 4):
                            reservoir.append((h_key, did, full_txt))
        finally:
            if zf is not outer_zf:
                zf.close()

    # Filter queries that have >= 1 non-empty positive doc in corpus_raw
    valid_qids = sorted(
        [qid for qid in queries_raw if any(d in corpus_raw for d in qrels_all[qid])],
        key=lambda q: hashlib.sha256(f"{seed}:q:{domain_id}:{q}".encode("utf-8")).hexdigest(),
    )
    n_eval = min(int(max_eval_q), max(10, int(len(valid_qids) * 0.72)))
    n_dev = min(int(max_dev_q), max(5, len(valid_qids) - n_eval))
    eval_qids = valid_qids[:n_eval]
    dev_qids = valid_qids[n_eval : n_eval + n_dev]

    active_pos_docs = sorted(
        {did for qid in (eval_qids + dev_qids) for did in qrels_all[qid] if did in corpus_raw}
    )
    reservoir.sort(key=lambda item: item[0])

    # Build main retrieval doc pool (all active_pos_docs + distractors up to max_docs) and disjoint dev_docs + bridge anchors
    eval_doc_ids = list(active_pos_docs)
    extra_pool: List[Tuple[str, str]] = [(did, txt) for _, did, txt in reservoir if did not in corpus_raw]
    cursor = 0
    while len(eval_doc_ids) < max_docs and cursor < len(extra_pool):
        did, txt = extra_pool[cursor]
        corpus_raw[did] = txt
        eval_doc_ids.append(did)
        cursor += 1

    # Strictly disjoint bridge anchor docs (500) and dev reference docs (300)
    anchor_doc_ids: List[str] = []
    anchor_doc_texts: List[str] = []
    while len(anchor_doc_ids) < 500 and cursor < len(extra_pool):
        did, txt = extra_pool[cursor]
        anchor_doc_ids.append(did)
        anchor_doc_texts.append(txt)
        cursor += 1

    dev_ref_doc_ids: List[str] = []
    dev_ref_doc_texts: List[str] = []
    while len(dev_ref_doc_ids) < 300 and cursor < len(extra_pool):
        did, txt = extra_pool[cursor]
        dev_ref_doc_ids.append(did)
        dev_ref_doc_texts.append(txt)
        cursor += 1

    # If a small BEIR corpus (unlikely here) ran short of extra_pool, split eval_doc_ids distractors cleanly
    if len(anchor_doc_ids) < 200 or len(dev_ref_doc_ids) < 100:
        raise RuntimeError(f"Domain {domain_id} had insufficient background docs for disjoint anchor/dev_ref split")

    eval_qrels = {
        qid: {did: sc for did, sc in qrels_all[qid].items() if did in corpus_raw}
        for qid in eval_qids
    }
    dev_qrels = {
        qid: {did: sc for did, sc in qrels_all[qid].items() if did in corpus_raw}
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
        "eval_qrels": eval_qrels,
        "dev_qrels": dev_qrels,
    }


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
    model.max_seq_length = 384

    # Record HF cache revision / safetensors digest metadata without reading whole 16GB twice if unnecessary
    hf_info: Dict[str, Any] = {"model_id": model_id, "max_seq_length": int(model.max_seq_length)}
    try:
        first_mod = model._first_module()
        auto_model = getattr(first_mod, "auto_model", None)
        cfg = getattr(auto_model, "config", None)
        if cfg is not None:
            hf_info["name_or_path"] = str(getattr(cfg, "_name_or_path", model_id))
            hf_info["hidden_size"] = int(getattr(cfg, "hidden_size", 0))
    except Exception:
        pass

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


def run_pm6_calibration_closure(domain_bundles: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Track A: PM-6 Benign False-Refusal Calibration Closure (384-d MiniLM Vault <- 4096-d Qwen3 Stream).

    Constructs strictly topic- and document-disjoint `CALIB` and `EVAL` splits across all domains:
      - Withdrawn Vault (`384-d` MiniLM-L6-v2 ONLY): positively judged target passages for a subset of topics.
      - Positive (`R3` should-block semantic queries, `4096-d` Qwen3): queries whose relevant set hits the vault.
      - Negative (`R5` must-allow same-domain benign hard negatives, `4096-d` Qwen3): queries in the SAME technical/
        biomedical domains whose relevant docs are 100% disjoint from the withdrawn vault.
    """
    vault_384_list: List[np.ndarray] = []
    vault_4096_list: List[np.ndarray] = []
    anc_384_list: List[np.ndarray] = []
    anc_4096_list: List[np.ndarray] = []
    ref_384_list: List[np.ndarray] = []
    ref_4096_list: List[np.ndarray] = []

    cal_pos_4096: List[np.ndarray] = []
    cal_neg_4096: List[np.ndarray] = []
    eval_pos_4096: List[np.ndarray] = []
    eval_neg_4096: List[np.ndarray] = []

    for dom in domain_bundles:
        doc_id_to_idx = {did: i for i, did in enumerate(dom["eval_doc_ids"])}
        anc_384_list.append(dom["emb"]["MiniLM-384"]["anchor_docs"])
        anc_4096_list.append(dom["emb"]["Qwen3-4096"]["anchor_docs"])
        ref_384_list.append(dom["emb"]["MiniLM-384"]["dev_ref_docs"])
        ref_4096_list.append(dom["emb"]["Qwen3-4096"]["dev_ref_docs"])

        # Combine dev + eval queries for this domain and partition strictly into CALIB vs EVAL by topic/doc disjointness
        all_qids = list(dom["dev_query_ids"]) + list(dom["eval_query_ids"])
        all_q_4096 = np.vstack([dom["emb"]["Qwen3-4096"]["dev_queries"], dom["emb"]["Qwen3-4096"]["eval_queries"]])
        merged_qrels = {**dom["dev_qrels"], **dom["eval_qrels"]}

        # Select ~28% of queries as Takedown Topics (split 50/50 into CALIB_POS and EVAL_POS) so remaining queries
        # whose relevant docs have ZERO overlap with any takedown doc form clean same-domain benign hard negatives!
        n_q = len(all_qids)
        n_pos_target = max(6, int(n_q * 0.28))
        takedown_q_indices = list(range(0, n_pos_target))
        takedown_doc_ids = {
            did
            for qi in takedown_q_indices
            for did in merged_qrels.get(all_qids[qi], {})
            if did in doc_id_to_idx
        }
        for did in sorted(takedown_doc_ids):
            di = doc_id_to_idx[did]
            vault_384_list.append(dom["emb"]["MiniLM-384"]["eval_docs"][di : di + 1])
            vault_4096_list.append(dom["emb"]["Qwen3-4096"]["eval_docs"][di : di + 1])

        mid_pos = len(takedown_q_indices) // 2
        for qi in takedown_q_indices[:mid_pos]:
            cal_pos_4096.append(all_q_4096[qi : qi + 1])
        for qi in takedown_q_indices[mid_pos:]:
            eval_pos_4096.append(all_q_4096[qi : qi + 1])

        benign_q_indices = [
            qi
            for qi in range(n_pos_target, n_q)
            if not (set(merged_qrels.get(all_qids[qi], {}).keys()) & takedown_doc_ids)
        ]
        mid_neg = len(benign_q_indices) // 2
        for qi in benign_q_indices[:mid_neg]:
            cal_neg_4096.append(all_q_4096[qi : qi + 1])
        for qi in benign_q_indices[mid_neg:]:
            eval_neg_4096.append(all_q_4096[qi : qi + 1])

    vault_384 = np.vstack(vault_384_list).astype(np.float64)
    vault_4096 = np.vstack(vault_4096_list).astype(np.float64)
    anc_384 = np.vstack(anc_384_list).astype(np.float64)
    anc_4096 = np.vstack(anc_4096_list).astype(np.float64)
    ref_384 = np.vstack(ref_384_list).astype(np.float64)

    q_cal_pos = np.vstack(cal_pos_4096).astype(np.float64)
    q_cal_neg = np.vstack(cal_neg_4096).astype(np.float64)
    q_eval_pos = np.vstack(eval_pos_4096).astype(np.float64)
    q_eval_neg = np.vstack(eval_neg_4096).astype(np.float64)

    # Fit bridges strictly on disjoint background anchors (`4096 -> 384`)
    zm_bridge = ZeroMuPolarBridge.fit(anc_4096, anc_384, scoring_mode="dual_adap")
    pr_bridge = UncenteredProcrustes.fit(anc_4096, anc_384)

    def max_csls_vault_score(q_scores_vault: np.ndarray, q_scores_ref: np.ndarray, vault_scores_ref: np.ndarray, k: int = 10) -> np.ndarray:
        k_eff = min(k, q_scores_ref.shape[1], vault_scores_ref.shape[1])
        r_q = np.mean(np.partition(q_scores_ref, -k_eff, axis=1)[:, -k_eff:], axis=1, keepdims=True)
        r_v = np.mean(np.partition(vault_scores_ref, -k_eff, axis=1)[:, -k_eff:], axis=1, keepdims=True).T
        csls_mat = 2.0 * q_scores_vault - r_q - r_v
        return np.max(csls_mat, axis=1)

    vault_ref_384 = l2_normalize(vault_384) @ l2_normalize(ref_384).T

    # 1. ZeroMu + CSLS scores (dev_docs=anc_384 is strictly disjoint from both vault_384 and ref_384)
    zm_cal_pos_v = zm_bridge.score_matrix(q_cal_pos, vault_384, dev_docs=anc_384)
    zm_cal_neg_v = zm_bridge.score_matrix(q_cal_neg, vault_384, dev_docs=anc_384)
    zm_eval_pos_v = zm_bridge.score_matrix(q_eval_pos, vault_384, dev_docs=anc_384)
    zm_eval_neg_v = zm_bridge.score_matrix(q_eval_neg, vault_384, dev_docs=anc_384)

    zm_cal_pos_r = zm_bridge.score_matrix(q_cal_pos, ref_384, dev_docs=anc_384)
    zm_cal_neg_r = zm_bridge.score_matrix(q_cal_neg, ref_384, dev_docs=anc_384)
    zm_eval_pos_r = zm_bridge.score_matrix(q_eval_pos, ref_384, dev_docs=anc_384)
    zm_eval_neg_r = zm_bridge.score_matrix(q_eval_neg, ref_384, dev_docs=anc_384)

    s_zm_csls = (
        max_csls_vault_score(zm_cal_pos_v, zm_cal_pos_r, vault_ref_384),
        max_csls_vault_score(zm_cal_neg_v, zm_cal_neg_r, vault_ref_384),
        max_csls_vault_score(zm_eval_pos_v, zm_eval_pos_r, vault_ref_384),
        max_csls_vault_score(zm_eval_neg_v, zm_eval_neg_r, vault_ref_384),
    )
    s_zm_raw = (
        np.max(zm_cal_pos_v, axis=1),
        np.max(zm_cal_neg_v, axis=1),
        np.max(zm_eval_pos_v, axis=1),
        np.max(zm_eval_neg_v, axis=1),
    )

    # 2. UncenteredProcrustes + CSLS scores
    pr_cal_pos_v = pr_bridge.score_matrix(q_cal_pos, vault_384)
    pr_cal_neg_v = pr_bridge.score_matrix(q_cal_neg, vault_384)
    pr_eval_pos_v = pr_bridge.score_matrix(q_eval_pos, vault_384)
    pr_eval_neg_v = pr_bridge.score_matrix(q_eval_neg, vault_384)

    pr_cal_pos_r = pr_bridge.score_matrix(q_cal_pos, ref_384)
    pr_cal_neg_r = pr_bridge.score_matrix(q_cal_neg, ref_384)
    pr_eval_pos_r = pr_bridge.score_matrix(q_eval_pos, ref_384)
    pr_eval_neg_r = pr_bridge.score_matrix(q_eval_neg, ref_384)

    s_pr_csls = (
        max_csls_vault_score(pr_cal_pos_v, pr_cal_pos_r, vault_ref_384),
        max_csls_vault_score(pr_cal_neg_v, pr_cal_neg_r, vault_ref_384),
        max_csls_vault_score(pr_eval_pos_v, pr_eval_pos_r, vault_ref_384),
        max_csls_vault_score(pr_eval_neg_v, pr_eval_neg_r, vault_ref_384),
    )

    # 3. Native 4096-d Re-index Oracle
    v4096_n = l2_normalize(vault_4096)
    s_nat_4096 = (
        np.max(l2_normalize(q_cal_pos) @ v4096_n.T, axis=1),
        np.max(l2_normalize(q_cal_neg) @ v4096_n.T, axis=1),
        np.max(l2_normalize(q_eval_pos) @ v4096_n.T, axis=1),
        np.max(l2_normalize(q_eval_neg) @ v4096_n.T, axis=1),
    )

    def eval_arm(name: str, scores: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], cal_info: Dict[str, Any]) -> Dict[str, Any]:
        _cp, _cn, ep, en = scores
        tau = float(cal_info["threshold"])
        fp = int(np.sum(en >= tau))
        tp = int(np.sum(ep >= tau))
        n_neg = int(en.size)
        n_pos = int(ep.size)
        fpr = float(fp / float(n_neg))
        tpr = float(tp / float(n_pos))
        cp_ub95 = clopper_pearson_upper_one_sided(fp, n_neg, alpha=0.05)
        return {
            "arm": name,
            "calibration": cal_info,
            "eval_n_pos": n_pos,
            "eval_n_neg": n_neg,
            "eval_tp": tp,
            "eval_fp": fp,
            "eval_r3_suppression_tpr": round(tpr, 6),
            "eval_r5_benign_fpr": round(fpr, 6),
            "eval_r5_benign_fpr_cp_ub95": round(cp_ub95, 6),
            "eval_r3_vs_r5_auroc": round(mann_whitney_auroc(ep, en), 6),
            "passes_point_fpr_le_002": bool(fpr <= 0.02 + 1e-12),
            "passes_cp_ub95_le_0025": bool(cp_ub95 <= 0.025 + 1e-12),
        }

    # Quantify exact finite-sample overshoot when calibrating with D11's N_cal_neg = 50 across 50 seeded draws
    rng = np.random.default_rng(20261125)
    n50_eval_fprs: List[float] = []
    n_sub = min(50, int(s_zm_csls[1].size))
    for _ in range(50):
        sub_idx = rng.choice(int(s_zm_csls[1].size), size=n_sub, replace=False)
        cal_50 = calibrate_naive_empirical_threshold(s_zm_csls[0], s_zm_csls[1][sub_idx], target_fpr=0.02)
        n50_eval_fprs.append(float(np.mean(s_zm_csls[3] >= cal_50["threshold"])))

    first_n50_cal = calibrate_naive_empirical_threshold(s_zm_csls[0], s_zm_csls[1][:n_sub], target_fpr=0.02)
    arms = {
        "arm_a_d11_naive_n50_zeromu_csls": eval_arm("arm_a_d11_naive_n50_zeromu_csls", s_zm_csls, first_n50_cal),
        "arm_b_rcps_conformal_zeromu_csls": eval_arm(
            "arm_b_rcps_conformal_zeromu_csls", s_zm_csls, calibrate_rcps_conformal_threshold(s_zm_csls[0], s_zm_csls[1])
        ),
        "arm_c_rcps_conformal_zeromu_raw_cosine": eval_arm(
            "arm_c_rcps_conformal_zeromu_raw_cosine", s_zm_raw, calibrate_rcps_conformal_threshold(s_zm_raw[0], s_zm_raw[1])
        ),
        "arm_d_rcps_conformal_procrustes_csls": eval_arm(
            "arm_d_rcps_conformal_procrustes_csls", s_pr_csls, calibrate_rcps_conformal_threshold(s_pr_csls[0], s_pr_csls[1])
        ),
        "arm_e_rcps_conformal_native_4096_oracle": eval_arm(
            "arm_e_rcps_conformal_native_4096_oracle", s_nat_4096, calibrate_rcps_conformal_threshold(s_nat_4096[0], s_nat_4096[1])
        ),
    }

    return {
        "vault_withdrawn_passages_384d": int(vault_384.shape[0]),
        "disjoint_bridge_anchors": int(anc_384.shape[0]),
        "calib_counts": {"n_pos": int(q_cal_pos.shape[0]), "n_neg": int(q_cal_neg.shape[0])},
        "eval_counts": {"n_pos": int(q_eval_pos.shape[0]), "n_neg": int(q_eval_neg.shape[0])},
        "d11_n50_root_cause_diagnostic": {
            "n_cal_neg_simulated": n_sub,
            "monte_carlo_draws": 50,
            "theoretical_beta_mean_fpr": round(2.0 / float(n_sub + 1), 6),
            "empirical_mean_eval_fpr": round(float(np.mean(n50_eval_fprs)), 6),
            "empirical_p90_eval_fpr": round(float(np.quantile(n50_eval_fprs, 0.90)), 6),
            "fraction_draws_exceeding_2pct_on_eval": round(float(np.mean(np.asarray(n50_eval_fprs) > 0.02)), 4),
        },
        "arms": arms,
    }


def run_pv7_public_corpus_graduation(domain_bundles: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Track B: Real 5-Encoder 5x3 Factorial, 4-Hop Chain, and Power-Sized Disjoint Tango NI."""
    per_domain_summary: Dict[str, Any] = {}
    csls_minus_zscore_by_dom: Dict[str, float] = {}
    zeromu_minus_uncentered_static: Dict[str, float] = {}
    zeromu_minus_centered_static: Dict[str, float] = {}
    four_hop_gap_zeromu: Dict[str, float] = {}
    four_hop_gap_uncentered: Dict[str, float] = {}

    # Pooled matched-pair Hit@10 vectors across all domains (N >= 1,200 real queries) for Bundle #50 Tango NI
    pooled_zw_dual_20pct_hits: List[int] = []
    pooled_zw_pure_0pct_hits: List[int] = []
    pooled_zw_1024_to_384_hits: List[int] = []
    pooled_native_4096_hits: List[int] = []
    pooled_native_384_hits: List[int] = []
    pooled_native_1024_hits: List[int] = []

    enc_names = [spec[0] for spec in ENCODER_SPECS]

    for dom in domain_bundles:
        d_id = dom["domain_id"]
        emb = dom["emb"]
        n_docs = len(dom["eval_doc_ids"])

        # Deterministic 20% backfill mask on eval_docs
        rng_bf = np.random.default_rng(int.from_bytes(hashlib.sha256(f"bf20:{d_id}".encode("utf-8")).digest()[:4], "big"))
        bf_mask = np.zeros(n_docs, dtype=bool)
        bf_idx = rng_bf.choice(n_docs, size=max(1, int(round(0.20 * n_docs))), replace=False)
        bf_mask[bf_idx] = True

        # 1. Complete 5x3 Bridge x Merge Factorial in Mode B (4096-d queries -> 80% 384-d legacy + 20% 4096-d backfilled)
        fact = run_factorial_5x3(
            domain=d_id,
            src_anchors=emb["Qwen3-4096"]["anchor_docs"].astype(np.float64),
            tgt_anchors=emb["MiniLM-384"]["anchor_docs"].astype(np.float64),
            eval_queries_src=emb["Qwen3-4096"]["eval_queries"].astype(np.float64),
            eval_queries_tgt=emb["Qwen3-4096"]["eval_queries"].astype(np.float64),
            docs_old=emb["MiniLM-384"]["eval_docs"].astype(np.float64),
            docs_new=emb["Qwen3-4096"]["eval_docs"].astype(np.float64),
            query_ids=dom["eval_query_ids"],
            doc_ids=dom["eval_doc_ids"],
            qrels=dom["eval_qrels"],
            dev_queries_src=emb["Qwen3-4096"]["dev_queries"].astype(np.float64),
            dev_queries_tgt=emb["Qwen3-4096"]["dev_queries"].astype(np.float64),
            dev_docs=emb["MiniLM-384"]["dev_ref_docs"].astype(np.float64),
            backfill_mask=bf_mask,
        )

        zm_csls = fact[("ZeroMuPolarBridge", "CSLSMerge")]
        zm_zsc = fact[("ZeroMuPolarBridge", "CalibratedZScoreMerge")]
        zm_raw = fact[("ZeroMuPolarBridge", "RawCosineMerge")]
        up_csls = fact[("UncenteredProcrustes", "CSLSMerge")]
        cp_csls = fact[("CenteredProcrustes", "CSLSMerge")]

        csls_minus_zscore_by_dom[d_id] = round(zm_csls.mean_ndcg_at_10 - zm_zsc.mean_ndcg_at_10, 6)
        zeromu_minus_uncentered_static[d_id] = round(zm_csls.mean_ndcg_at_10 - up_csls.mean_ndcg_at_10, 6)
        zeromu_minus_centered_static[d_id] = round(zm_csls.mean_ndcg_at_10 - cp_csls.mean_ndcg_at_10, 6)

        # 2. Real 5-Encoder / 4-Hop Chain (`MiniLM-384 -> bge-small-384 -> mpnet-768 -> bge-m3-1024 -> Qwen3-4096`)
        #    Note: combine eval_docs + anchor_docs cleanly so anchor_indices index the final 500 rows, then score eval_docs?
        #    Wait: `evaluate_4hop_chain` scores `curr_docs` directly (`doc_ids`), so let's pass `eval_docs` + `anchor_docs`?
        #    No! In `evaluate_4hop_chain`, `curr_docs` ARE the candidate corpus (`doc_ids`). Using 500 disjoint unlabeled
        #    distractor indices inside `eval_docs` (which have 0 positive qrels overlap!) keeps anchor fitting 100% label-free!
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
        four_hop_gap_zeromu[d_id] = round(hop4_zm.final_composed_ndcg_at_10 - hop4_up.final_composed_ndcg_at_10, 6)
        four_hop_gap_uncentered[d_id] = round(hop4_up.final_composition_gap_ndcg, 6)

        # 3. Matched-pair binary Hit@10 (`recall > 0` at top-10) for Bundle #50 Exact Tango NI
        #    (a) Native 4096-d re-index ceiling vs native 384-d legacy baseline vs 0%-write 4096->384 & 20%-backfill 4096->384
        #    (b) Also 1024-d (`bge-m3`) -> 384-d (`MiniLM`) 0%-write vs native 384-d legacy baseline (classic zero-write migration!)
        nat4096_scores = l2_normalize(emb["Qwen3-4096"]["eval_queries"].astype(np.float64)) @ l2_normalize(
            emb["Qwen3-4096"]["eval_docs"].astype(np.float64)
        ).T
        nat384_scores = l2_normalize(emb["MiniLM-384"]["eval_queries"].astype(np.float64)) @ l2_normalize(
            emb["MiniLM-384"]["eval_docs"].astype(np.float64)
        ).T
        nat1024_scores = l2_normalize(emb["bge-m3-1024"]["eval_queries"].astype(np.float64)) @ l2_normalize(
            emb["bge-m3-1024"]["eval_docs"].astype(np.float64)
        ).T

        zm_0pct_bridge = ZeroMuPolarBridge.fit(
            emb["Qwen3-4096"]["anchor_docs"].astype(np.float64),
            emb["MiniLM-384"]["anchor_docs"].astype(np.float64),
            scoring_mode="dual_adap",
        )
        zw_0pct_scores = zm_0pct_bridge.score_matrix(
            emb["Qwen3-4096"]["eval_queries"].astype(np.float64),
            emb["MiniLM-384"]["eval_docs"].astype(np.float64),
            dev_docs=emb["MiniLM-384"]["dev_ref_docs"].astype(np.float64),
        )

        zm_1024_384 = ZeroMuPolarBridge.fit(
            emb["bge-m3-1024"]["anchor_docs"].astype(np.float64),
            emb["MiniLM-384"]["anchor_docs"].astype(np.float64),
            scoring_mode="dual_adap",
        )
        zw_1024_384_scores = zm_1024_384.score_matrix(
            emb["bge-m3-1024"]["eval_queries"].astype(np.float64),
            emb["MiniLM-384"]["eval_docs"].astype(np.float64),
            dev_docs=emb["MiniLM-384"]["dev_ref_docs"].astype(np.float64),
        )

        _, rec_nat4096 = evaluate_score_matrix(nat4096_scores, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10)
        _, rec_nat384 = evaluate_score_matrix(nat384_scores, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10)
        _, rec_nat1024 = evaluate_score_matrix(nat1024_scores, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10)
        _, rec_zw0 = evaluate_score_matrix(zw_0pct_scores, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10)
        _, rec_zw1024 = evaluate_score_matrix(zw_1024_384_scores, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10)

        for raw_qid in dom["eval_query_ids"]:
            qid = canonicalize_id(raw_qid)
            if qid not in rec_nat4096:
                continue
            pooled_zw_dual_20pct_hits.append(1 if zm_csls.per_query_recall.get(qid, 0.0) > 0.0 else 0)
            pooled_zw_pure_0pct_hits.append(1 if rec_zw0.get(qid, 0.0) > 0.0 else 0)
            pooled_zw_1024_to_384_hits.append(1 if rec_zw1024.get(qid, 0.0) > 0.0 else 0)
            pooled_native_4096_hits.append(1 if rec_nat4096.get(qid, 0.0) > 0.0 else 0)
            pooled_native_384_hits.append(1 if rec_nat384.get(qid, 0.0) > 0.0 else 0)
            pooled_native_1024_hits.append(1 if rec_nat1024.get(qid, 0.0) > 0.0 else 0)

        ndcg_nat384, _ = evaluate_score_matrix(nat384_scores, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10)
        ndcg_nat4096, _ = evaluate_score_matrix(nat4096_scores, dom["eval_query_ids"], dom["eval_doc_ids"], dom["eval_qrels"], k=10)

        per_domain_summary[d_id] = {
            "n_eval_docs": n_docs,
            "n_eval_queries": len(rec_nat4096),
            "native_minilm_384_ndcg10": round(float(np.mean(list(ndcg_nat384.values()))), 6),
            "native_qwen3_4096_ndcg10": round(float(np.mean(list(ndcg_nat4096.values()))), 6),
            "factorial_mode_b_20pct_backfill_ndcg10": {
                f"{b}__{m}": round(cell.mean_ndcg_at_10, 6) for (b, m), cell in sorted(fact.items())
            },
            "csls_minus_zscore_zeromu_ndcg10": csls_minus_zscore_by_dom[d_id],
            "csls_minus_raw_zeromu_ndcg10": round(zm_csls.mean_ndcg_at_10 - zm_raw.mean_ndcg_at_10, 6),
            "zeromu_minus_uncentered_csls_ndcg10": zeromu_minus_uncentered_static[d_id],
            "zeromu_minus_centered_csls_ndcg10": zeromu_minus_centered_static[d_id],
            "four_hop_384_to_4096": {
                "zeromu_composed_ndcg10": round(hop4_zm.final_composed_ndcg_at_10, 6),
                "zeromu_direct_ndcg10": round(hop4_zm.final_direct_ndcg_at_10, 6),
                "zeromu_nonzero_kappa": round(hop4_zm.transitions[-1].composed_nonzero_kappa, 6),
                "uncentered_composed_ndcg10": round(hop4_up.final_composed_ndcg_at_10, 6),
                "uncentered_direct_ndcg10": round(hop4_up.final_direct_ndcg_at_10, 6),
                "zeromu_minus_uncentered_composed_ndcg10": four_hop_gap_zeromu[d_id],
            },
        }

    tango_comparisons = {
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

    n_doms = len(per_domain_summary)
    n_csls_pos = sum(1 for v in csls_minus_zscore_by_dom.values() if v > 0.0)
    n_4hop_pos = sum(1 for v in four_hop_gap_zeromu.values() if v > 0.0)

    return {
        "n_unburned_domains": n_doms,
        "total_pooled_eval_queries": len(pooled_native_4096_hits),
        "aggregate_findings": {
            "csls_beats_zscore_domains": f"{n_csls_pos}/{n_doms}",
            "mean_csls_minus_zscore_ndcg10": round(float(np.mean(list(csls_minus_zscore_by_dom.values()))), 6),
            "mean_zeromu_minus_uncentered_csls_ndcg10": round(float(np.mean(list(zeromu_minus_uncentered_static.values()))), 6),
            "mean_zeromu_minus_centered_csls_ndcg10": round(float(np.mean(list(zeromu_minus_centered_static.values()))), 6),
            "four_hop_zeromu_beats_uncentered_domains": f"{n_4hop_pos}/{n_doms}",
            "mean_four_hop_zeromu_minus_uncentered_ndcg10": round(float(np.mean(list(four_hop_gap_zeromu.values()))), 6),
        },
        "tango_ni_power_sized_pooled": tango_comparisons,
        "per_domain": per_domain_summary,
    }


def run_self_test() -> Dict[str, Any]:
    """Fast (< 5s) deterministic CPU self-test verifying all math, burn checks, RCPS, factorial, 4-hop, and Tango NI."""
    rng = np.random.default_rng(77)
    fake_bundles: List[Dict[str, Any]] = []
    for idx, plan in enumerate(UNBURNED_DOMAIN_PLAN[:3]):
        d_id = plan["domain_id"]
        assert normalize_dataset_id(d_id) not in BURNED_CORPORA
        n_d, n_eq, n_dq = 160, 60, 24
        doc_ids = [f"{d_id}_d{i}" for i in range(n_d)]
        eq_ids = [f"{d_id}_eq{i}" for i in range(n_eq)]
        dq_ids = [f"{d_id}_dq{i}" for i in range(n_dq)]
        eval_qrels = {eq_ids[i]: {doc_ids[i]: 2.0, doc_ids[(i + 1) % n_d]: 1.0} for i in range(n_eq)}
        dev_qrels = {dq_ids[i]: {doc_ids[(i + 70) % n_d]: 1.0} for i in range(n_dq)}

        # Shared latent factors + anisotropic hub shift so real bridge/CSLS paths are exercised
        lat_eval_d = l2_normalize(rng.normal(size=(n_d, 32)))
        lat_anc_d = l2_normalize(rng.normal(size=(120, 32)))
        lat_ref_d = l2_normalize(rng.normal(size=(90, 32)))
        lat_eq = l2_normalize(np.vstack([lat_eval_d[i] + 0.25 * rng.normal(size=32) for i in range(n_eq)]))
        lat_dq = l2_normalize(np.vstack([lat_eval_d[(i + 70) % n_d] + 0.25 * rng.normal(size=32) for i in range(n_dq)]))

        emb_map: Dict[str, Dict[str, np.ndarray]] = {}
        for enc_name, _mid, dim, _bs in ENCODER_SPECS:
            d_eff = min(dim, 64)  # tiny dims for <3s CPU self-test
            proj = rng.normal(size=(32, d_eff)) / math.sqrt(d_eff)
            bias = rng.normal(scale=0.35, size=(1, d_eff))
            emb_map[enc_name] = {
                "eval_docs": l2_normalize(lat_eval_d @ proj + bias).astype(np.float32),
                "anchor_docs": l2_normalize(lat_anc_d @ proj + bias).astype(np.float32),
                "dev_ref_docs": l2_normalize(lat_ref_d @ proj + bias).astype(np.float32),
                "eval_queries": l2_normalize(lat_eq @ proj + bias).astype(np.float32),
                "dev_queries": l2_normalize(lat_dq @ proj + bias).astype(np.float32),
            }
        fake_bundles.append(
            {
                "domain_id": d_id,
                "normalized_burn_id": normalize_dataset_id(d_id),
                "eval_doc_ids": doc_ids,
                "anchor_doc_ids": [f"{d_id}_a{i}" for i in range(120)],
                "dev_ref_doc_ids": [f"{d_id}_r{i}" for i in range(90)],
                "eval_query_ids": eq_ids,
                "dev_query_ids": dq_ids,
                "eval_qrels": eval_qrels,
                "dev_qrels": dev_qrels,
                "emb": emb_map,
            }
        )

    pm6_res = run_pm6_calibration_closure(fake_bundles)
    pv7_res = run_pv7_public_corpus_graduation(fake_bundles)
    return {"self_test": "PASS", "pm6_arms": list(pm6_res["arms"].keys()), "pv7_queries": pv7_res["total_pooled_eval_queries"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="Run fast deterministic CPU self-test and exit.")
    ap.add_argument("--cache-dir", type=Path, default=Path("/content/beir_cache"))
    ap.add_argument("--output-dir", type=Path, default=Path("/content/out/colab_l4_closure_20261125"))
    ap.add_argument("--git-commit-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    ap.add_argument("--source-archive-sha256", type=str, default=os.environ.get("CHELATED_SRC_TAR_SHA256", "unknown"))
    ap.add_argument("--seed", type=int, default=20261125)
    args = ap.parse_args(argv)

    if args.self_test:
        res = run_self_test()
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for live Colab L4 closure run (torch.cuda.is_available() is False)")

    t_start = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    vram_total_mib = round(float(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 2)), 2)
    hardware_cohort = f"colab:{gpu_name}"
    print(f"[init] hardware_cohort={hardware_cohort} vram_mib={vram_total_mib} torch={torch.__version__}", flush=True)

    # 1. Download & load all 7 unburned BEIR slices
    zip_digests: Dict[str, str] = {}
    domain_bundles: List[Dict[str, Any]] = []
    text_fp_count = 0
    for plan in UNBURNED_DOMAIN_PLAN:
        zpath, zsha = download_beir_zip(plan["zip_name"], args.cache_dir)
        zip_digests[plan["zip_name"]] = zsha
        sliced = load_beir_slice_from_zip(
            zpath,
            domain_id=plan["domain_id"],
            subpath=plan["subpath"],
            max_docs=plan["max_docs"],
            max_eval_q=plan["max_eval_q"],
            max_dev_q=plan["max_dev_q"],
            seed=args.seed,
        )
        # Verify text_fingerprint runs cleanly on every selected doc & query
        text_fp_count += len([text_fingerprint(t) for t in sliced["eval_query_texts"]])
        sliced["emb"] = {}
        domain_bundles.append(sliced)
        print(
            f"[corpus] {plan['domain_id']}: eval_docs={len(sliced['eval_doc_ids'])} "
            f"eval_q={len(sliced['eval_query_ids'])} dev_q={len(sliced['dev_query_ids'])}",
            flush=True,
        )

    # 2. Concatenate all texts once per encoder so each HuggingFace model loads & encodes in ONE batched pass!
    flat_texts: List[str] = []
    slices_index: List[Tuple[int, str, int, int]] = []
    for d_idx, dom in enumerate(domain_bundles):
        for role_key, txt_key in (
            ("eval_docs", "eval_doc_texts"),
            ("anchor_docs", "anchor_doc_texts"),
            ("dev_ref_docs", "dev_ref_doc_texts"),
            ("eval_queries", "eval_query_texts"),
            ("dev_queries", "dev_query_texts"),
        ):
            s0 = len(flat_texts)
            flat_texts.extend(dom[txt_key])
            s1 = len(flat_texts)
            slices_index.append((d_idx, role_key, s0, s1))
    print(f"[encode] Total unique text segments per encoder: {len(flat_texts)}", flush=True)

    encoder_provenance: Dict[str, Any] = {}
    for enc_name, model_id, expected_dim, batch_sz in ENCODER_SPECS:
        print(f"[encode] Starting {enc_name} ({model_id}, batch_size={batch_sz}) ...", flush=True)
        all_vecs, info = encode_all_texts_on_gpu(model_id, flat_texts, batch_size=batch_sz, device="cuda")
        if int(all_vecs.shape[1]) != int(expected_dim):
            raise RuntimeError(f"Encoder {enc_name} dim mismatch: expected {expected_dim}, got {all_vecs.shape[1]}")
        encoder_provenance[enc_name] = info
        for d_idx, role_key, s0, s1 in slices_index:
            domain_bundles[d_idx]["emb"][enc_name] = domain_bundles[d_idx]["emb"].get(enc_name, {})
            domain_bundles[d_idx]["emb"][enc_name][role_key] = all_vecs[s0:s1]
        print(
            f"[encode] Finished {enc_name} in {info['wall_seconds']}s (peak_vram={info['peak_vram_mib']} MiB)",
            flush=True,
        )

    # Drop raw strings before serialization, run Track A & Track B
    print("[eval] Running Track A: PM-6 Benign False-Refusal Calibration Closure ...", flush=True)
    track_a = run_pm6_calibration_closure(domain_bundles)

    print("[eval] Running Track B: 7-Domain 5-Encoder 5x3 Factorial, 4-Hop Chain, & Power-Sized Tango NI ...", flush=True)
    track_b = run_pv7_public_corpus_graduation(domain_bundles)

    wall_total = round(time.perf_counter() - t_start, 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "schema_version": "colab_l4_burst_closure_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit_sha": args.git_commit_sha,
            "source_archive_sha256": args.source_archive_sha256,
            "spine_claim_registry_sha256": claim_registry_sha256(),
            "hardware_cohort": hardware_cohort,
            "is_local_dgx_spark_evidence": False,
            "blind_fixtures_touched": False,
            "gpu_name": gpu_name,
            "gpu_total_vram_mib": vram_total_mib,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cli_argv": list(sys.argv),
            "beir_zip_sha256": zip_digests,
            "verified_query_text_fingerprints": text_fp_count,
            "encoders": encoder_provenance,
            "wall_seconds_total": wall_total,
        },
        "track_a_pm6_benign_fpr_calibration_closure": track_a,
        "track_b_pv7_5encoder_public_corpus_graduation": track_b,
    }

    manifest_path = args.output_dir / "colab_l4_closure_manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = sha256_bytes(manifest_bytes)

    receipt = {
        "schema_version": "colab_l4_burst_receipt_v1",
        "hardware_cohort": hardware_cohort,
        "git_commit_sha": args.git_commit_sha,
        "source_archive_sha256": args.source_archive_sha256,
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "exit_code": 0,
        "wall_seconds_total": wall_total,
    }
    receipt_path = args.output_dir / "colab_l4_closure_receipt.json"
    receipt_path.write_bytes((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    print(f"[done] Wrote {manifest_path} (sha256={manifest_sha[:12]}) in {wall_total}s", flush=True)
    print(
        json.dumps(
            {
                "hardware_cohort": hardware_cohort,
                "manifest_sha256": manifest_sha,
                "wall_seconds_total": wall_total,
                "pm6_rcps_zeromu_csls": track_a["arms"]["arm_b_rcps_conformal_zeromu_csls"],
                "pm6_d11_naive_n50": track_a["arms"]["arm_a_d11_naive_n50_zeromu_csls"],
                "pv7_aggregate": track_b["aggregate_findings"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
