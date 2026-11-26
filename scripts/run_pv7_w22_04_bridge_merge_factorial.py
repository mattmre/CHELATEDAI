#!/usr/bin/env python3
"""W22-04: real-data 5x4 bridge x merge factorial at equal total cost (12 domains).

Preregistration: ``artifacts/pv7_w23a_w22_04/PREREGISTRATION.md`` (frozen before any result).

Pipeline
--------
1. Load the 12 unburned domains with the two ORIGINAL loaders (prior7 from
   ``run_colab_l4_pm6_and_pv7_closure``; remaining5 from
   ``run_colab_l4_w22a_b54_remaining5_and_poisoning``) under seed 20261125 and
   verify dataset-file sha256 against the source manifests.
2. Encode MiniLM-384 and Qwen3-Embedding-8B (bf16) per group with the original
   ``encode_all_texts_on_gpu``; cache vectors under ``<out>/scratch/``.
3. Run ``run_factorial_5x3`` with bridges {Centered, Uncentered, RidgeDriftAdapter,
   Rank1QueryResidual, ZeroMu} x merges {Raw, ZScore, CSLS, RRF} (Mode B, 20% backfill).
4. Apply the frozen per-domain kill rule (paired bootstrap seed 1601) and the CSLS
   retention check (seed 1602); compare the 15 legacy cells to the source manifests.

All model/dataset access is offline (``HF_HUB_OFFLINE=1``, ``HF_DATASETS_OFFLINE=1``).
"""

from __future__ import annotations

import argparse
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

from chelated_pv7_spine.bridge_merge_harness import (  # noqa: E402
    paired_bootstrap_ci,
    run_factorial_5x3,
)

OUT_DIR_DEFAULT = REPO_ROOT / "artifacts" / "pv7_w23a_w22_04"
PRIOR7_MANIFEST = REPO_ROOT / "artifacts" / "pv7_colab_l4_closure_20261125" / "colab_l4_closure_manifest.json"
REMAINING5_MANIFEST = REPO_ROOT / "artifacts" / "pv7_w22a_b54_12dom_complete" / "w22a_b54_12dom_manifest.json"

DATA_SEED = 20261125
PRIMARY_BOOT_SEED = 1601
CSLS_BOOT_SEED = 1602
N_RESAMPLES = 2000
ALPHA = 0.05
EQUIV_MARGIN = 0.005
KILL_THRESHOLD = 10
N_DOMAINS_PLANNED = 12
FIDELITY_FLAG_TOL = 0.02
REMAINING5_ANCHORS = 500

QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
QWEN_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
QWEN_BATCH = 32
MINILM_BATCH = 256

PRIMARY_BRIDGES: Tuple[str, ...] = (
    "UncenteredProcrustes",
    "CenteredProcrustes",
    "RidgeDriftAdapter",
    "ZeroMuPolarBridge",
)
SECONDARY_BRIDGES: Tuple[str, ...] = ("Rank1QueryResidual",)
ALL_BRIDGES: Tuple[str, ...] = (
    "UncenteredProcrustes",
    "CenteredProcrustes",
    "RidgeDriftAdapter",
    "Rank1QueryResidual",
    "ZeroMuPolarBridge",
)
MERGES: Tuple[str, ...] = ("RawCosineMerge", "CalibratedZScoreMerge", "CSLSMerge", "RRFMerge")
LEGACY_MERGES: Tuple[str, ...] = ("RawCosineMerge", "CalibratedZScoreMerge", "CSLSMerge")
COMPARATORS: Tuple[str, ...] = ("CenteredProcrustes", "UncenteredProcrustes")
ZEROMU = "ZeroMuPolarBridge"
CSLS = "CSLSMerge"

COST_TABLE: Dict[str, Dict[str, str]] = {
    "UncenteredProcrustes": {"persisted": "W", "per_doc_extra": "none", "query_time_state": "none"},
    "CenteredProcrustes": {"persisted": "W, mu_x, mu_y", "per_doc_extra": "none", "query_time_state": "none"},
    "RidgeDriftAdapter": {"persisted": "W", "per_doc_extra": "none", "query_time_state": "none"},
    "Rank1QueryResidual": {"persisted": "W, rank-1 residual", "per_doc_extra": "none", "query_time_state": "none"},
    "ZeroMuPolarBridge": {
        "persisted": "Q, mu_x, mu_d (DEV docs)",
        "per_doc_extra": "float32 centered norm (4 B/doc)",
        "query_time_state": "causal query EMA mu_q",
    },
}


# ---------------------------------------------------------------------------
# Pure analysis (CPU, unit-tested)
# ---------------------------------------------------------------------------


def aligned_query_vectors(a: Mapping[str, float], b: Mapping[str, float]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Align two per-query maps; raises if their query sets differ (cells must share queries)."""
    if set(a.keys()) != set(b.keys()):
        raise ValueError("per-query maps cover different query sets")
    qids = sorted(a.keys())
    return qids, np.array([a[q] for q in qids], dtype=np.float64), np.array([b[q] for q in qids], dtype=np.float64)


def marginal_bootstrap_ci(values: Sequence[float], *, n_resamples: int, alpha: float, seed: int) -> Tuple[float, float, float]:
    """Percentile bootstrap CI of a single arm's mean (same resample scheme as the paired CI)."""
    return paired_bootstrap_ci(values, n_resamples=n_resamples, alpha=alpha, seed=seed)


def compare_arms(
    zeromu_pq: Mapping[str, float],
    comparator_pq: Mapping[str, float],
    *,
    margin: float = EQUIV_MARGIN,
    n_resamples: int = N_RESAMPLES,
    alpha: float = ALPHA,
    seed: int = PRIMARY_BOOT_SEED,
) -> Dict[str, Any]:
    """Frozen per-domain comparison of ZeroMu+CSLS vs comparator+CSLS (PREREGISTRATION §4-5)."""
    _, z, c = aligned_query_vectors(zeromu_pq, comparator_pq)
    delta, lcb, ucb = paired_bootstrap_ci((z - c).tolist(), n_resamples=n_resamples, alpha=alpha, seed=seed)
    z_m, z_l, z_u = marginal_bootstrap_ci(z.tolist(), n_resamples=n_resamples, alpha=alpha, seed=seed)
    c_m, c_l, c_u = marginal_bootstrap_ci(c.tolist(), n_resamples=n_resamples, alpha=alpha, seed=seed)
    within_margin = abs(delta) <= margin
    paired_ci_contains_zero = lcb <= 0.0 <= ucb
    marginal_overlap = not (z_u < c_l or c_u < z_l)
    if within_margin or paired_ci_contains_zero:
        direction = "within"
    elif delta > 0:
        direction = "zeromu_better"
    else:
        direction = "comparator_better"
    return {
        "n_queries": int(z.size),
        "delta_zeromu_minus_comparator": round(delta, 6),
        "paired_ci95": [round(lcb, 6), round(ucb, 6)],
        "within_margin": bool(within_margin),
        "paired_ci_contains_zero": bool(paired_ci_contains_zero),
        "within_primary": bool(within_margin or paired_ci_contains_zero),
        "zeromu_marginal_ci95": [round(z_l, 6), round(z_u, 6)],
        "comparator_marginal_ci95": [round(c_l, 6), round(c_u, 6)],
        "marginal_ci_overlap": bool(marginal_overlap),
        "within_secondary": bool(within_margin or marginal_overlap),
        "direction_primary": direction,
    }


def domain_decision(per_comparator: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """A domain is WITHIN if at least one comparator is within (PREREGISTRATION §5)."""
    within_p = [b for b, r in per_comparator.items() if r["within_primary"]]
    within_s = [b for b, r in per_comparator.items() if r["within_secondary"]]
    return {
        "within_primary": bool(within_p),
        "within_primary_via": within_p,
        "within_secondary": bool(within_s),
        "within_secondary_via": within_s,
    }


def kill_verdict(within_flags: Sequence[bool], *, n_planned: int = N_DOMAINS_PLANNED, threshold: int = KILL_THRESHOLD) -> Dict[str, Any]:
    """Frozen verdict: KILL iff >= threshold WITHIN; decided early only if missing domains cannot flip it."""
    n_run = len(within_flags)
    n_within = int(sum(bool(x) for x in within_flags))
    n_missing = max(0, n_planned - n_run)
    if n_within >= threshold:
        verdict = "KILL"
    elif n_within + n_missing < threshold:
        verdict = "NOT_KILLED"
    else:
        verdict = "UNDECIDED_DEFERRED_SCOPE"
    return {
        "n_domains_run": n_run,
        "n_domains_planned": n_planned,
        "n_within": n_within,
        "threshold": threshold,
        "verdict": verdict,
    }


def csls_retention(
    cells_pq: Mapping[str, Mapping[str, float]],
    *,
    n_resamples: int = N_RESAMPLES,
    alpha: float = ALPHA,
    seed: int = CSLS_BOOT_SEED,
) -> Dict[str, Any]:
    """CSLS vs the best non-CSLS merge (selected on point estimate) for one bridge/domain."""
    means = {m: float(np.mean(list(pq.values()))) for m, pq in cells_pq.items()}
    others = [m for m in cells_pq if m != CSLS]
    best_other = max(others, key=lambda m: (means[m], m))
    _, cs, bo = aligned_query_vectors(cells_pq[CSLS], cells_pq[best_other])
    d, lcb, ucb = paired_bootstrap_ci((cs - bo).tolist(), n_resamples=n_resamples, alpha=alpha, seed=seed)
    best_overall = max(means, key=lambda m: (means[m], m))
    return {
        "best_non_csls_merge": best_other,
        "csls_minus_best_other": round(d, 6),
        "paired_ci95": [round(lcb, 6), round(ucb, 6)],
        "csls_is_best_merge": best_overall == CSLS,
    }


def fidelity_compare(fresh: Mapping[str, float], recorded: Mapping[str, float]) -> Dict[str, Any]:
    """Compare fresh legacy cells with manifest-recorded values (keys `Bridge__Merge`)."""
    common = sorted(set(fresh) & set(recorded))
    diffs = {k: round(float(fresh[k]) - float(recorded[k]), 6) for k in common}
    max_abs = max((abs(v) for v in diffs.values()), default=float("nan"))
    return {
        "n_cells_compared": len(common),
        "missing_in_manifest": sorted(set(fresh) - set(recorded)),
        "max_abs_diff": round(max_abs, 6) if common else None,
        "diffs": diffs,
        "flagged": bool(common) and max_abs > FIDELITY_FLAG_TOL,
    }


def analyze_domain(
    d_id: str,
    fact: Mapping[Tuple[str, str], Any],
    *,
    recorded_cells: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Turn one domain's factorial result into the preregistered per-domain record."""
    cells = {f"{b}__{m}": round(float(r.mean_ndcg_at_10), 6) for (b, m), r in fact.items()}
    zm = fact[(ZEROMU, CSLS)].per_query_ndcg
    per_comp = {b: compare_arms(zm, fact[(b, CSLS)].per_query_ndcg) for b in COMPARATORS}
    retention = {
        b: csls_retention({m: fact[(b, m)].per_query_ndcg for m in MERGES})
        for b in sorted({b for (b, _) in fact.keys()})
    }
    legacy_fresh = {k: v for k, v in cells.items() if k.split("__")[1] in LEGACY_MERGES}
    out: Dict[str, Any] = {
        "domain": d_id,
        "n_eval_queries_with_positives": len(zm),
        "cells_ndcg10": cells,
        "kill_rule": {"per_comparator": per_comp, **domain_decision(per_comp)},
        "csls_retention": retention,
    }
    if recorded_cells is not None:
        out["fidelity"] = fidelity_compare(legacy_fresh, recorded_cells)
    return out


# ---------------------------------------------------------------------------
# Data / encoding stage (GPU)
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _texts_digest(texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(hashlib.sha256(t.encode("utf-8")).digest())
    return h.hexdigest()


def _load_group(group: str, hf_cache: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    if group == "prior7":
        from scripts import run_colab_l4_pm6_and_pv7_closure as mod

        plans = mod.UNBURNED_DOMAIN_PLAN
    else:
        from scripts import run_colab_l4_w22a_b54_remaining5_and_poisoning as mod

        plans = mod.REMAINING5_DOMAIN_PLAN
    bundles: List[Dict[str, Any]] = []
    digests: Dict[str, Dict[str, str]] = {}
    for plan in plans:
        kwargs = dict(
            domain_id=plan["domain_id"],
            hf_repo=plan["hf_repo"],
            cache_dir=hf_cache,
            max_docs=plan["max_docs"],
            max_eval_q=plan["max_eval_q"],
            max_dev_q=plan["max_dev_q"],
            seed=DATA_SEED,
        )
        if group == "remaining5":
            kwargs["anchor_pool"] = plan["anchor_pool"]
        sliced, dg = mod.load_beir_slice_from_hf(**kwargs)
        sliced["emb"] = {}
        sliced["hf_repo"] = plan["hf_repo"]
        bundles.append(sliced)
        digests[plan["hf_repo"]] = dg
        print(
            f"[w22-04] loaded {group}/{plan['domain_id']}: docs={len(sliced['eval_doc_ids'])} "
            f"anchors={len(sliced['anchor_doc_ids'])} eval_q={len(sliced['eval_query_ids'])} "
            f"dev_q={len(sliced['dev_query_ids'])}",
            flush=True,
        )
    return bundles, digests


def _group_texts(group: str, bundles: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[int, str, int, int]], List[str], List[Tuple[int, str, int, int]]]:
    base: List[str] = []
    base_slices: List[Tuple[int, str, int, int]] = []
    for d_idx, dom in enumerate(bundles):
        for role_key, txt_key in (
            ("eval_docs", "eval_doc_texts"),
            ("anchor_docs", "anchor_doc_texts"),
            ("dev_ref_docs", "dev_ref_doc_texts"),
            ("eval_queries", "eval_query_texts"),
            ("dev_queries", "dev_query_texts"),
        ):
            s0 = len(base)
            base.extend(dom[txt_key])
            base_slices.append((d_idx, role_key, s0, len(base)))
    shift: List[str] = []
    shift_slices: List[Tuple[int, str, int, int]] = []
    if group == "remaining5":
        for d_idx, dom in enumerate(bundles):
            for role_key, txt_key in (
                ("shifted_eval_queries", "shifted_eval_query_texts"),
                ("shifted_dev_queries", "shifted_dev_query_texts"),
            ):
                s0 = len(shift)
                shift.extend(dom[txt_key])
                shift_slices.append((d_idx, role_key, s0, len(shift)))
    return base, base_slices, shift, shift_slices


def _encode_group(group: str, bundles: List[Dict[str, Any]], scratch: Path) -> Dict[str, Any]:
    from scripts.run_colab_l4_pm6_and_pv7_closure import encode_all_texts_on_gpu

    base, base_slices, shift, shift_slices = _group_texts(group, bundles)
    prov: Dict[str, Any] = {}
    for enc_name, model_id, batch in (("MiniLM-384", MINILM_MODEL_ID, MINILM_BATCH), ("Qwen3-4096", QWEN_MODEL_ID, QWEN_BATCH)):
        texts = base + shift if enc_name == "Qwen3-4096" else base
        digest = _texts_digest(texts)
        cache = scratch / f"emb_{group}_{enc_name}_{digest[:16]}.npy"
        meta_path = scratch / f"emb_{group}_{enc_name}_{digest[:16]}.json"
        if cache.exists() and meta_path.exists():
            vecs = np.load(cache)
            info = json.loads(meta_path.read_text(encoding="utf-8"))
            info["loaded_from_cache"] = True
            print(f"[w22-04] cache hit {cache.name}", flush=True)
        else:
            print(f"[w22-04] encoding {group} {enc_name} N={len(texts)} bs={batch}", flush=True)
            vecs, info = encode_all_texts_on_gpu(model_id, texts, batch_size=batch, device="cuda")
            info["texts_sha256"] = digest
            info["batch_size"] = batch
            np.save(cache, vecs)
            meta_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
            info["loaded_from_cache"] = False
        prov[enc_name] = info
        for d_idx, role_key, s0, s1 in base_slices:
            bundles[d_idx]["emb"].setdefault(enc_name, {})[role_key] = vecs[s0:s1]
    return prov


def _bf20_mask(d_id: str, n_docs: int) -> np.ndarray:
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"bf20:{d_id}".encode("utf-8")).digest()[:4], "big"))
    mask = np.zeros(n_docs, dtype=bool)
    mask[rng.choice(n_docs, size=max(1, int(round(0.20 * n_docs))), replace=False)] = True
    return mask


def _run_domain_factorial(group: str, dom: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
    emb = dom["emb"]
    src = emb["Qwen3-4096"]["anchor_docs"].astype(np.float64)
    tgt = emb["MiniLM-384"]["anchor_docs"].astype(np.float64)
    if group == "remaining5":
        n_fit = min(REMAINING5_ANCHORS, src.shape[0])
        src, tgt = src[:n_fit], tgt[:n_fit]
    q_eval = emb["Qwen3-4096"]["eval_queries"].astype(np.float64)
    q_dev = emb["Qwen3-4096"]["dev_queries"].astype(np.float64)
    return run_factorial_5x3(
        domain=dom["domain_id"],
        src_anchors=src,
        tgt_anchors=tgt,
        eval_queries_src=q_eval,
        eval_queries_tgt=q_eval,
        docs_old=emb["MiniLM-384"]["eval_docs"].astype(np.float64),
        docs_new=emb["Qwen3-4096"]["eval_docs"].astype(np.float64),
        query_ids=dom["eval_query_ids"],
        doc_ids=dom["eval_doc_ids"],
        qrels=dom["eval_qrels"],
        dev_queries_src=q_dev,
        dev_queries_tgt=q_dev,
        dev_docs=emb["MiniLM-384"]["dev_ref_docs"].astype(np.float64),
        backfill_mask=_bf20_mask(dom["domain_id"], len(dom["eval_doc_ids"])),
        bridge_names=ALL_BRIDGES,
        merge_names=MERGES,
    )


def _recorded(group: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, str]]]:
    if group == "prior7":
        m = json.loads(PRIOR7_MANIFEST.read_text(encoding="utf-8"))
        per = m["track_b_pv7_5encoder_public_corpus_graduation"]["per_domain"]
    else:
        m = json.loads(REMAINING5_MANIFEST.read_text(encoding="utf-8"))
        per = m["results"]["per_domain_remaining5"]
    return per, m["provenance"]["hf_dataset_file_sha256"]


def run_live(out_dir: Path, hf_cache: Path, groups: Sequence[str]) -> Dict[str, Any]:
    import torch

    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1 are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    scratch = out_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    per_domain: Dict[str, Any] = {}
    dataset_checks: Dict[str, Any] = {}
    encoder_prov: Dict[str, Any] = {}
    scrub_counts: Dict[str, Any] = {}
    for group in groups:
        recorded_per, recorded_sha = _recorded(group)
        bundles, digests = _load_group(group, hf_cache)
        for repo, dg in digests.items():
            exp = recorded_sha.get(repo, {})
            dataset_checks[repo] = {
                "observed": dg,
                "recorded": exp,
                "match": bool(exp) and all(exp.get(f) == h for f, h in dg.items()),
            }
        encoder_prov[group] = _encode_group(group, bundles, scratch)
        if group == "prior7":
            from scripts.run_colab_l4_pm6_and_pv7_closure import scrub_embedding_row_collisions_globally
        else:
            from scripts.run_colab_l4_w22a_b54_remaining5_and_poisoning import scrub_embedding_row_collisions_globally
        scrub_counts[group] = scrub_embedding_row_collisions_globally(bundles)
        for dom in bundles:
            d_id = dom["domain_id"]
            print(f"[w22-04] factorial {d_id}", flush=True)
            fact = _run_domain_factorial(group, dom)
            rec = recorded_per.get(d_id, {})
            rec_cells = rec.get("factorial_mode_b_20pct_backfill_ndcg10")
            rec_out = analyze_domain(d_id, fact, recorded_cells=rec_cells)
            rec_out["group"] = group
            rec_out["n_eval_docs"] = len(dom["eval_doc_ids"])
            rec_out["n_eval_queries"] = len(dom["eval_query_ids"])
            rec_out["n_anchors_fit"] = int(
                min(REMAINING5_ANCHORS, len(dom["anchor_doc_ids"])) if group == "remaining5" else len(dom["anchor_doc_ids"])
            )
            rec_out["n_dev_queries"] = len(dom["dev_query_ids"])
            rec_out["n_dev_ref_docs"] = len(dom["dev_ref_doc_ids"])
            rec_out["recorded_counts"] = {
                "n_eval_docs": rec.get("n_eval_docs"),
                "n_eval_queries": rec.get("n_eval_queries"),
            }
            fid = rec_out.get("fidelity")
            if fid is not None:
                count_mismatch = (
                    rec.get("n_eval_docs") not in (None, rec_out["n_eval_docs"])
                    or rec.get("n_eval_queries") not in (None, rec_out["n_eval_queries"])
                )
                fid["count_mismatch"] = bool(count_mismatch)
                fid["flagged"] = bool(fid["flagged"] or count_mismatch)
            per_domain[d_id] = rec_out
        del bundles

    gpu = torch.cuda.get_device_name(0)
    return {
        "per_domain": per_domain,
        "dataset_checks": dataset_checks,
        "encoder_provenance": encoder_prov,
        "row_collisions_scrubbed": scrub_counts,
        "hardware": {
            "gpu_name": gpu,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }


def summarize(per_domain: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    doms = sorted(per_domain)
    prim = [per_domain[d]["kill_rule"]["within_primary"] for d in doms]
    sec = [per_domain[d]["kill_rule"]["within_secondary"] for d in doms]
    directions: Dict[str, Dict[str, str]] = {
        d: {b: per_domain[d]["kill_rule"]["per_comparator"][b]["direction_primary"] for b in COMPARATORS} for d in doms
    }
    retention_counts: Dict[str, Dict[str, int]] = {}
    bridges = sorted({b for d in doms for b in per_domain[d]["csls_retention"]})
    for b in bridges:
        rows = [per_domain[d]["csls_retention"][b] for d in doms if b in per_domain[d]["csls_retention"]]
        retention_counts[b] = {
            "csls_is_best_merge": sum(1 for r in rows if r["csls_is_best_merge"]),
            "csls_ci_above_best_other": sum(1 for r in rows if r["paired_ci95"][0] > 0.0),
            "csls_ci_below_best_other": sum(1 for r in rows if r["paired_ci95"][1] < 0.0),
            "n_domains": len(rows),
        }
    merge_means: Dict[str, float] = {}
    for d in doms:
        for k, v in per_domain[d]["cells_ndcg10"].items():
            merge_means.setdefault(k, 0.0)
            merge_means[k] += v / len(doms)
    return {
        "primary": kill_verdict(prim),
        "secondary_marginal_overlap": kill_verdict(sec),
        "within_primary_domains": [d for d, w in zip(doms, prim) if w],
        "not_within_primary_domains": [d for d, w in zip(doms, prim) if not w],
        "directions": directions,
        "csls_retention_counts": retention_counts,
        "cell_means_over_domains_ndcg10": {k: round(v, 6) for k, v in sorted(merge_means.items())},
        "fidelity_flagged_domains": [d for d in doms if per_domain[d].get("fidelity", {}).get("flagged")],
        "fidelity_max_abs_diff_by_domain": {d: per_domain[d].get("fidelity", {}).get("max_abs_diff") for d in doms},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--hf-cache", type=Path, default=Path.home() / ".cache" / "huggingface" / "hub")
    ap.add_argument("--groups", nargs="+", default=["prior7", "remaining5"], choices=["prior7", "remaining5"])
    ap.add_argument("--git-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    args = ap.parse_args(argv)

    live = run_live(args.out_dir, args.hf_cache, args.groups)
    result = {
        "item": "W22-04",
        "preregistration": "artifacts/pv7_w23a_w22_04/PREREGISTRATION.md",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_at_run": args.git_sha,
        "config": {
            "data_seed": DATA_SEED,
            "primary_bootstrap_seed": PRIMARY_BOOT_SEED,
            "csls_bootstrap_seed": CSLS_BOOT_SEED,
            "n_resamples": N_RESAMPLES,
            "alpha": ALPHA,
            "equivalence_margin": EQUIV_MARGIN,
            "kill_threshold": KILL_THRESHOLD,
            "primary_bridges": list(PRIMARY_BRIDGES),
            "secondary_bridges": list(SECONDARY_BRIDGES),
            "merges": list(MERGES),
            "comparators": list(COMPARATORS),
            "rrf_k": 60,
            "qwen_model": QWEN_MODEL_ID,
            "qwen_revision": QWEN_REVISION,
            "qwen_batch": QWEN_BATCH,
            "minilm_model": MINILM_MODEL_ID,
            "remaining5_anchors_fit": REMAINING5_ANCHORS,
        },
        "cost_table": COST_TABLE,
        "summary": summarize(live["per_domain"]),
        **live,
    }
    out = args.out_dir / "w22_04_results.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[w22-04] wrote {out}", flush=True)
    print(json.dumps(result["summary"]["primary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
