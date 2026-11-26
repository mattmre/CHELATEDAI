#!/usr/bin/env python3
"""W22-06: migration information frontier on the 12 W22-04 domains (CPU only, cached embeddings).

Preregistration: ``artifacts/pv7_w23b_w22_06/PREREGISTRATION.md`` (committed alone before this file).

Reads MiniLM-384 / Qwen3-4096 vectors from the W22-04 scratch cache (read-only), never
encodes, and refuses to run unless ``CUDA_VISIBLE_DEVICES=""``, ``HF_HUB_OFFLINE=1`` and
``HF_DATASETS_OFFLINE=1`` are set. Sub-test (b) is DEFERRED_SCOPE (needs uncached encoders).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_utils import canonicalize_id  # noqa: E402
from chelated_pv7_spine.bridge_merge_harness import evaluate_score_matrix  # noqa: E402
from migration_frontier_w22_06 import (  # noqa: E402
    ALPHA,
    K_DEFAULT,
    bit_ledger,
    calibrate_margin,
    conformal_study,
    containment_score,
    decide_margin_claim,
    doc_tiebreak_keys,
    flip_pair_stats,
    heldout_fit_error,
    is_calibration_query,
    lodo_law,
    serve_scores,
    summarize_flips,
)
from scripts.run_pv7_w22_04_bridge_merge_factorial import (  # noqa: E402
    DATA_SEED,
    REMAINING5_ANCHORS,
    _bf20_mask,
    _group_texts,
    _load_group,
    _texts_digest,
)

OUT_DIR_DEFAULT = REPO_ROOT / "artifacts" / "pv7_w23b_w22_06"
W22_04_RESULTS = REPO_ROOT / "artifacts" / "pv7_w23a_w22_04" / "w22_04_results.json"
LEDGER_BRIDGES = ("UncenteredProcrustes", "CenteredProcrustes", "ZeroMuPolarBridge")
PRIMARY = "CenteredProcrustes"
CONTRAST = "UncenteredProcrustes"


def _require_cpu_offline() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError('CUDA_VISIBLE_DEVICES="" is required (CPU only)')
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1 are required")


def _load_cached(group: str, bundles: List[Dict[str, Any]], cache_dir: Path) -> Dict[str, Any]:
    """Slice cached vectors into bundles after verifying each cache's text digest."""
    base, base_slices, shift, _ = _group_texts(group, bundles)
    prov: Dict[str, Any] = {}
    for enc in ("MiniLM-384", "Qwen3-4096"):
        texts = base + shift if enc == "Qwen3-4096" else base
        digest = _texts_digest(texts)
        npy = cache_dir / f"emb_{group}_{enc}_{digest[:16]}.npy"
        meta = cache_dir / f"emb_{group}_{enc}_{digest[:16]}.json"
        if not npy.exists() or not meta.exists():
            raise FileNotFoundError(f"cache missing for {group}/{enc}: {npy}")
        info = json.loads(meta.read_text(encoding="utf-8"))
        if info.get("texts_sha256") != digest:
            raise ValueError(f"cache digest mismatch for {npy.name}")
        vecs = np.load(npy, mmap_mode="r")
        if vecs.shape[0] != len(texts):
            raise ValueError(f"cache row count {vecs.shape[0]} != {len(texts)} texts for {npy.name}")
        for d_idx, role, s0, s1 in base_slices:
            bundles[d_idx]["emb"].setdefault(enc, {})[role] = np.array(vecs[s0:s1], dtype=np.float32)
        prov[enc] = {"file": str(npy), "texts_sha256": digest, "rows": int(vecs.shape[0]), "dim": int(vecs.shape[1])}
    return prov


def _domain(group: str, dom: Dict[str, Any], recorded_cells: Dict[str, float], dump: List[Dict[str, Any]]) -> Dict[str, Any]:
    d_id = dom["domain_id"]
    emb = dom["emb"]
    src = emb["Qwen3-4096"]["anchor_docs"].astype(np.float64)
    tgt = emb["MiniLM-384"]["anchor_docs"].astype(np.float64)
    if group == "remaining5":
        n_fit = min(REMAINING5_ANCHORS, src.shape[0])
        src, tgt = src[:n_fit], tgt[:n_fit]
    q_eval = emb["Qwen3-4096"]["eval_queries"].astype(np.float64)
    q_dev = emb["Qwen3-4096"]["dev_queries"].astype(np.float64)
    docs_old = emb["MiniLM-384"]["eval_docs"].astype(np.float64)
    docs_new = emb["Qwen3-4096"]["eval_docs"].astype(np.float64)
    devref_src = emb["Qwen3-4096"]["dev_ref_docs"].astype(np.float64)
    devref_tgt = emb["MiniLM-384"]["dev_ref_docs"].astype(np.float64)
    doc_ids = dom["eval_doc_ids"]
    qids = dom["eval_query_ids"]
    mask = _bf20_mask(d_id, len(doc_ids))
    keys = doc_tiebreak_keys(doc_ids)

    served: Dict[str, Dict[str, Any]] = {}
    ndcg: Dict[str, float] = {}
    for b_name in (PRIMARY, CONTRAST):
        served[b_name] = serve_scores(
            b_name,
            src_anchors=src,
            tgt_anchors=tgt,
            queries_eval=q_eval,
            queries_dev=q_dev,
            docs_old=docs_old,
            docs_new=docs_new,
            backfill_mask=mask,
        )
        pq, _ = evaluate_score_matrix(served[b_name]["fused_eval"], qids, doc_ids, dom["eval_qrels"], k=K_DEFAULT)
        ndcg[b_name] = float(np.mean(list(pq.values())))
    p = served[PRIMARY]
    native_pq, _ = evaluate_score_matrix(p["native_eval"], qids, doc_ids, dom["eval_qrels"], k=K_DEFAULT)

    # (c) outcome + predictor
    e_unc = heldout_fit_error(served[CONTRAST]["bridge"], devref_src, devref_tgt)
    e_cen = heldout_fit_error(served[PRIMARY]["bridge"], devref_src, devref_tgt)

    # (a) margin + flips
    cal = calibrate_margin(p["fused_dev"], p["native_dev"], keys)
    calibrated = cal["a"] + cal["b"] * p["fused_eval"]
    flips: List[Dict[str, int]] = []
    cal_scores: List[int] = []
    test_scores: List[int] = []
    for i, qid in enumerate(qids):
        st = flip_pair_stats(calibrated[i], p["native_eval"][i], keys, k=K_DEFAULT, margin=cal["margin"])
        flips.append(st)
        # (d) containment score on the RAW fused row (affine calibration with b>0 preserves order;
        # the raw row is used so the score is exactly the served ranking).
        s = containment_score(p["fused_eval"][i], p["native_eval"][i], keys, K_DEFAULT)
        is_cal = is_calibration_query(d_id, qid)
        (cal_scores if is_cal else test_scores).append(s)
        dump.append({"domain": d_id, "qid": canonicalize_id(qid), "split": "cal" if is_cal else "test", "containment_score": s, **st})

    n_dev_ref = int(devref_src.shape[0])
    ledger = {
        b: bit_ledger(
            b,
            d_src=int(docs_new.shape[1]),
            d_tgt=int(docs_old.shape[1]),
            n_docs=len(doc_ids),
            n_backfill=int(mask.sum()),
            n_anchors=int(src.shape[0]),
            n_dev_queries=int(q_dev.shape[0]),
            n_dev_ref_docs=n_dev_ref,
        )
        for b in LEDGER_BRIDGES
    }
    rec_cen = recorded_cells.get(f"{PRIMARY}__CSLSMerge")
    rec_unc = recorded_cells.get(f"{CONTRAST}__CSLSMerge")
    return {
        "group": group,
        "n_eval_docs": len(doc_ids),
        "n_eval_queries": len(qids),
        "n_backfill": int(mask.sum()),
        "n_anchors_fit": int(src.shape[0]),
        "n_dev_queries": int(q_dev.shape[0]),
        "n_dev_ref_docs": n_dev_ref,
        "ndcg10": {
            "centered_csls": round(ndcg[PRIMARY], 6),
            "uncentered_csls": round(ndcg[CONTRAST], 6),
            "native_qwen_full_reindex": round(float(np.mean(list(native_pq.values()))), 6),
        },
        "reproduction_vs_w22_04": {
            "centered_csls_recorded": rec_cen,
            "uncentered_csls_recorded": rec_unc,
            "centered_abs_diff": None if rec_cen is None else round(abs(ndcg[PRIMARY] - rec_cen), 6),
            "uncentered_abs_diff": None if rec_unc is None else round(abs(ndcg[CONTRAST] - rec_unc), 6),
        },
        "centering": {
            "gain_g": ndcg[PRIMARY] - ndcg[CONTRAST],
            "heldout_fit_error_uncentered": e_unc,
            "heldout_fit_error_centered": e_cen,
            "predictor_x": e_unc - e_cen,
        },
        "margin_calibration": cal,
        "flips": summarize_flips(flips),
        "_flips_raw": flips,
        "conformal_input": {"cal": cal_scores, "test": test_scores, "n_docs": len(doc_ids)},
        "ledger": ledger,
    }


def run(out_dir: Path, cache_dir: Path, hf_cache: Path, groups: Sequence[str]) -> Dict[str, Any]:
    _require_cpu_offline()
    scratch = out_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    recorded = json.loads(W22_04_RESULTS.read_text(encoding="utf-8"))["per_domain"]
    per_domain: Dict[str, Any] = {}
    cache_prov: Dict[str, Any] = {}
    scrub: Dict[str, Any] = {}
    dump: List[Dict[str, Any]] = []
    for group in groups:
        bundles, _digests = _load_group(group, hf_cache)
        cache_prov[group] = _load_cached(group, bundles, cache_dir)
        if group == "prior7":
            from scripts.run_colab_l4_pm6_and_pv7_closure import scrub_embedding_row_collisions_globally
        else:
            from scripts.run_colab_l4_w22a_b54_remaining5_and_poisoning import scrub_embedding_row_collisions_globally
        scrub[group] = scrub_embedding_row_collisions_globally(bundles)
        for dom in bundles:
            d_id = dom["domain_id"]
            print(f"[w22-06] {group}/{d_id}", flush=True)
            per_domain[d_id] = _domain(group, dom, recorded.get(d_id, {}).get("cells_ndcg10", {}), dump)
        del bundles

    doms = sorted(per_domain)
    # (a)
    all_flips = [r for d in doms for r in per_domain[d].pop("_flips_raw")]
    pooled = summarize_flips(all_flips)
    claim_a = decide_margin_claim(
        {d: per_domain[d]["flips"]["frac_missing_margin_explainable"] for d in doms},
        pooled["frac_missing_margin_explainable"],
        pooled["frac_queries_with_flip"],
    )
    # (c)
    law = lodo_law(
        {d: per_domain[d]["centering"]["predictor_x"] for d in doms},
        {d: per_domain[d]["centering"]["gain_g"] for d in doms},
    )
    # (d)
    conf_in = {d: per_domain[d].pop("conformal_input") for d in doms}
    conf = conformal_study(conf_in, k=K_DEFAULT, alpha=ALPHA, groups={d: per_domain[d]["group"] for d in doms})

    dump_path = scratch / "w22_06_per_query.jsonl"
    with dump_path.open("w", encoding="utf-8") as fh:
        for row in dump:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "per_domain": per_domain,
        "subtest_a_margin_flips": {"pooled": pooled, "decision": claim_a},
        "subtest_b_chain_vs_direct": {
            "status": "DEFERRED_SCOPE",
            "reason": "needs bge-small-384, mpnet-768 and bge-m3-1024 embeddings; none cached; "
            "task rule: re-encoding parts are scoped out instead of using the GPU",
            "verdict": None,
        },
        "subtest_c_centering_law_lodo": law,
        "subtest_d_conformal_containment": conf,
        "subtest_e_bit_ledger_note": "per-domain ledgers under per_domain[*].ledger; descriptive, no kill rule",
        "embedding_cache_provenance": cache_prov,
        "row_collisions_scrubbed": scrub,
        "per_query_dump": {"path": str(dump_path), "rows": len(dump)},
        "hardware": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--cache-dir", type=Path, required=True, help="W22-04 scratch dir holding emb_*.npy (read-only)")
    ap.add_argument("--hf-cache", type=Path, default=Path.home() / ".cache" / "huggingface" / "hub")
    ap.add_argument("--groups", nargs="+", default=["prior7", "remaining5"], choices=["prior7", "remaining5"])
    ap.add_argument("--git-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    args = ap.parse_args(argv)
    live = run(args.out_dir, args.cache_dir, args.hf_cache, args.groups)
    result = {
        "item": "W22-06",
        "preregistration": "artifacts/pv7_w23b_w22_06/PREREGISTRATION.md",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_at_run": args.git_sha,
        "config": {"data_seed": DATA_SEED, "k": K_DEFAULT, "alpha": ALPHA, "primary_bridge": PRIMARY, "contrast_bridge": CONTRAST},
        **live,
    }
    out = args.out_dir / "w22_06_results.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=float), encoding="utf-8")
    print(f"[w22-06] wrote {out}", flush=True)
    print(json.dumps({
        "a": live["subtest_a_margin_flips"]["decision"],
        "c": {k: live["subtest_c_centering_law_lodo"][k] for k in ("r2_lodo", "sign_accuracy", "verdict")},
        "d": {k: live["subtest_d_conformal_containment"][k] for k in ("median_k_star_in_domain", "median_k_star_lodo", "mean_coverage_in_domain", "mean_coverage_lodo", "verdict")},
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
