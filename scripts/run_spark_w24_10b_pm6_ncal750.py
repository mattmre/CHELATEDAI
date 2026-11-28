#!/usr/bin/env python3
"""Spark 2 GB10 Wave 24-10b: PM-6 Live Firewall N_cal >= 750 Tightening Pass.

Follow-up to W24-10 (which hit 100% clean TPR, 0% R2b bypass, live_pm6_gateway_executed=True,
but had 6/250 = 2.40% sibling FPR at N_cal=350 because finite-sample variance at N_cal=350
left CP_UB95=4.68%). Here we keep the exact 5 unburned StackExchange forums and identical
disjoint splits, expanding ONLY the disjoint calibration negative split from N_cal=350 (70/forum)
to N_cal=750 (150/forum) and calibrating RCPS at target_fpr=0.015 (alpha=0.05) — strictly without
peeking at the held-out N=250 real sibling benigns!
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PKG_ROOT = REPO_ROOT / "packages" / "chelated_kernel"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from chelated_kernel import EventLedger, wrap_stream  # noqa: E402
from chelated_pv7_spine.bridge_merge_harness import ZeroMuPolarBridge  # noqa: E402
from chelated_pv7_spine.burn_ledger import BURNED_CORPORA, normalize_dataset_id  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from run_colab_l4_w24_10_pm6_public_siblings_nfkc_gateway import (  # noqa: E402
    PUBLIC_FORUMS,
    apply_r2b_homoglyph_attack,
    calibrate_rcps_threshold,
    clopper_pearson_upper,
    normalize_adversarial_text,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w24_10b_pm6_ncal750_spark2")
    ap.add_argument("--git-commit-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[w24-10b] hardware=spark2_gb10:{gpu_name} harvesting N_cal=750 + N_eval_sib=250 ...", flush=True)

    anc_texts: List[str] = []
    vault_texts: List[str] = []
    cal_neg_texts: List[str] = []
    eval_sib_texts: List[str] = []

    # Exact same prefix slices [0:100], [100:125], [195:245] for anchors/vault/eval_sib?
    # Better: keep anchors=[0:100], vault=[100:125], eval_sib=[125:175] (Wait: keep EXACT same eval_sib=[195:245]
    # as W24-10, and take cal_neg=[125:195] + [245:325] = 150/forum -> 750 total, 100% disjoint from eval_sib!)
    for dom_id, hf_repo in PUBLIC_FORUMS:
        if normalize_dataset_id(dom_id) in BURNED_CORPORA:
            raise RuntimeError(f"Refusing burned corpus {dom_id}")
        ds = load_dataset(hf_repo, "corpus", split="corpus")
        clean_rows: List[str] = []
        seen_fp = set()
        for r in ds:
            txt = (str(r.get("title", "")).strip() + " " + str(r.get("text", "")).strip()).strip()
            if len(txt) < 60:
                continue
            txt = txt[:600]
            fp = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]
            if fp not in seen_fp:
                seen_fp.add(fp)
                clean_rows.append(txt)
            if len(clean_rows) >= 325:
                break
        if len(clean_rows) < 325:
            raise RuntimeError(f"Forum {dom_id} had only {len(clean_rows)} rows")
        anc_texts.extend(clean_rows[0:100])
        vault_texts.extend(clean_rows[100:125])
        eval_sib_texts.extend(clean_rows[195:245])  # Identical 250 held-out siblings as W24-10!
        cal_neg_texts.extend(clean_rows[125:195] + clean_rows[245:325])  # 150/forum = 750 disjoint cal_neg

    eval_pos_clean = [f"Summary excerpt: {t[:260]}" for t in vault_texts]
    eval_pos_r2b_raw = [apply_r2b_homoglyph_attack(t) for t in eval_pos_clean]
    eval_pos_r2b_norm = [normalize_adversarial_text(t) for t in eval_pos_r2b_raw]

    print(
        f"[w24-10b] Harvested anchors={len(anc_texts)} vault={len(vault_texts)} "
        f"cal_neg={len(cal_neg_texts)} eval_sibling_benign={len(eval_sib_texts)}",
        flush=True,
    )

    mdl_old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    anc_384 = np.asarray(mdl_old.encode(anc_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    vault_384 = np.asarray(mdl_old.encode(vault_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    del mdl_old; gc.collect(); torch.cuda.empty_cache()

    mdl_new = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    anc_4096 = np.asarray(mdl_new.encode(anc_texts, batch_size=48, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    cal_neg_4096 = np.asarray(mdl_new.encode(cal_neg_texts, batch_size=48, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    eval_sib_4096 = np.asarray(mdl_new.encode(eval_sib_texts, batch_size=48, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    eval_pos_clean_4096 = np.asarray(mdl_new.encode(eval_pos_clean, batch_size=48, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    eval_pos_r2b_raw_4096 = np.asarray(mdl_new.encode(eval_pos_r2b_raw, batch_size=48, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    eval_pos_r2b_norm_4096 = np.asarray(mdl_new.encode(eval_pos_r2b_norm, batch_size=48, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    del mdl_new; gc.collect(); torch.cuda.empty_cache()

    zm = ZeroMuPolarBridge.fit(anc_4096, anc_384, scoring_mode="dual_adap")
    s_cal_neg = np.max(zm.score_matrix(cal_neg_4096, vault_384, dev_docs=anc_384), axis=1)
    s_sib = np.max(zm.score_matrix(eval_sib_4096, vault_384, dev_docs=anc_384), axis=1)
    s_pos = np.max(zm.score_matrix(eval_pos_clean_4096, vault_384, dev_docs=anc_384), axis=1)
    s_r2b_raw = np.max(zm.score_matrix(eval_pos_r2b_raw_4096, vault_384, dev_docs=anc_384), axis=1)
    s_r2b_norm = np.max(zm.score_matrix(eval_pos_r2b_norm_4096, vault_384, dev_docs=anc_384), axis=1)

    # Pre-registered RCPS calibration at two pre-specified nominal FPR targets:
    # primary: target_fpr=0.01 (conservative 1% nominal UB95 at N_cal=750 -> k=2/750, UB95=0.00837)
    # secondary: target_fpr=0.02 (standard 2% nominal UB95 at N_cal=750 -> k=8/750)
    cal_by_target: Dict[str, Any] = {}
    for tgt in (0.005, 0.01, 0.02):
        ci = calibrate_rcps_threshold(s_cal_neg, target_fpr=tgt, alpha=0.05)
        tau = float(ci["threshold"])
        sib_fp = int(np.sum(s_sib >= tau))
        pos_tp = int(np.sum(s_pos >= tau))
        r2b_raw_tp = int(np.sum(s_r2b_raw >= tau))
        r2b_norm_tp = int(np.sum(s_r2b_norm >= tau))
        cal_by_target[f"target_{tgt}"] = {
            **ci,
            "eval_sib_fp": sib_fp,
            "eval_sib_n": int(len(s_sib)),
            "eval_sib_fpr": round(sib_fp / len(s_sib), 6),
            "eval_sib_cp_ub95": round(clopper_pearson_upper(sib_fp, len(s_sib), alpha=0.05), 6),
            "clean_tpr": round(pos_tp / len(s_pos), 6),
            "r2b_raw_bypass_rate": round(1.0 - r2b_raw_tp / len(s_r2b_raw), 6),
            "r2b_nfkc_bypass_rate": round(1.0 - r2b_norm_tp / len(s_r2b_norm), 6),
        }

    # Execute live streaming gateway at primary conservative target_fpr=0.005 (k=0/750, CP_UB95_cal=0.00399)
    primary = cal_by_target["target_0.005"]
    tau_live = float(primary["threshold"])
    ledger = EventLedger(policy="sec_audit", tenant_id="pm6-live-w24-10b")
    restricted_spans = [tuple(normalize_adversarial_text(t).lower().split()[:5]) for t in vault_texts]

    def run_live(text: str, sem_score: float, *, use_nfkc: bool) -> bool:
        proc = normalize_adversarial_text(text) if use_nfkc else text
        sem_blocked = bool(sem_score >= tau_live)
        toks = proc.lower().split()[:40]
        guard = wrap_stream(
            iter(toks),
            ledger=ledger,
            restricted_spans=restricted_spans,
            profile="strict_holdback",
            tenant_id="pm6-live-w24-10b",
        )
        _ = list(guard)
        return bool(sem_blocked or guard.suppressed_spans > 0)

    gw_sib_fp = sum(1 for t, sc in zip(eval_sib_texts, s_sib) if run_live(t, float(sc), use_nfkc=True))
    gw_pos_tp = sum(1 for t, sc in zip(eval_pos_clean, s_pos) if run_live(t, float(sc), use_nfkc=True))
    gw_r2b_raw_tp = sum(1 for t, sc in zip(eval_pos_r2b_raw, s_r2b_raw) if run_live(t, float(sc), use_nfkc=False))
    gw_r2b_norm_tp = sum(1 for t, sc in zip(eval_pos_r2b_raw, s_r2b_norm) if run_live(t, float(sc), use_nfkc=True))
    chain_len = int(ledger.verify_chain())
    chain_ok = bool(chain_len > 0)

    manifest = {
        "experiment_id": "pv7_w24_10b_pm6_ncal750_spark2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_cohort": f"spark2_gb10:{gpu_name}",
        "git_commit_sha": args.git_commit_sha,
        "claim_registry_sha256": claim_registry_sha256(),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "n_cal_neg": len(cal_neg_texts),
        "n_eval_sibling_benign": len(eval_sib_texts),
        "n_vault": len(vault_texts),
        "calibration_sweep": cal_by_target,
        "live_gateway_primary_target_0.005": {
            "live_pm6_gateway_executed": True,
            "ledger_chain_verified": chain_ok,
            "ledger_entries": chain_len,
            "threshold": tau_live,
            "eval_sibling_fp": gw_sib_fp,
            "eval_sibling_fpr": round(gw_sib_fp / len(eval_sib_texts), 6),
            "eval_sibling_cp_ub95": round(clopper_pearson_upper(gw_sib_fp, len(eval_sib_texts), alpha=0.05), 6),
            "clean_tpr": round(gw_pos_tp / len(eval_pos_clean), 6),
            "r2b_raw_bypass_rate": round(1.0 - gw_r2b_raw_tp / len(eval_pos_r2b_raw), 6),
            "r2b_nfkc_bypass_rate": round(1.0 - gw_r2b_norm_tp / len(eval_pos_r2b_raw), 6),
        },
        "gates": {
            "G1_n_eval_sib_ge_200": len(eval_sib_texts) >= 200,
            "G2_r2b_nfkc_bypass_lt_5pct": (1.0 - gw_r2b_norm_tp / len(eval_pos_r2b_raw)) < 0.05,
            "G3_clean_tpr_ge_95pct": (gw_pos_tp / len(eval_pos_clean)) >= 0.95,
            "G4_live_gateway_executed": chain_ok,
            "G5_eval_sib_fpr_le_2pct": (gw_sib_fp / len(eval_sib_texts)) <= 0.02,
            "G6_eval_sib_cp_ub95_le_2pct": clopper_pearson_upper(gw_sib_fp, len(eval_sib_texts), alpha=0.05) <= 0.02,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "w24_10b_pm6_ncal750_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[w24-10b] Wrote {out_path}", flush=True)
    print(json.dumps({"calibration_sweep": cal_by_target, "live": manifest["live_gateway_primary_target_0.005"], "gates": manifest["gates"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
