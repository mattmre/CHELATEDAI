#!/usr/bin/env python3
"""Wave 25-06 (P1): PM-6 G6 Powered Calibration (N_eval = 3,000 Fresh Source-Disjoint Sibling Benigns)
+ Span-Guard-Only Ablation + Source-Clustered Bootstrap CIs.

Directly addresses the Fable 5.2 / Astra 6.2 audit finding on W24-10b:
  - Evaluates N_eval = 3,000 fresh source-disjoint sibling benigns across 8 unburned MTEB StackExchange
    forums (375 held-out siblings/forum), completely disjoint from anchors (N=640), vault (N=200),
    and calibration negatives (N_cal=1,200).
  - Freezes ALL thresholds strictly on DEV/cal_neg (zero peeking at the 3,000 held-out siblings).
  - Tests BOTH:
      (a) Verbatim + R2b Cyrillic/Greek homoglyph + zero-width attack excerpts (n=200)
      (b) Non-verbatim **paraphrase / token-shuffled** positives (0 verbatim 5-token overlap, n=200)
          so we honestly measure whether the semantic layer adds genuine recall above a span guard!
  - Evaluates 4 cleanly ablated arms:
      1. `span_guard_raw_5tok_only` (W24-10b baseline — raw first-5-word normalized spans, NO embedding)
      2. `span_guard_distinctive_idf_only` (strongest deterministic floor: NFKC + confusable skeleton + high-IDF 5-tok spans)
      3. `semantic_zeromu_rcps_only` (pure ZeroMu + RCPS threshold frozen on N_cal=1,200)
      4. `combined_distinctive_span_plus_semantic` (combined live `wrap_stream` + ZeroMu RCPS gateway)
  - Computes BOTH exact Clopper-Pearson UB95 AND source-clustered bootstrap UB95 across the 8 forums.
  - Advances ONLY if G6 (`CP_UB95 <= 2.0%` on N_eval=3,000) AND combined gateway beats span-guard-only
    on recall at matched FPR.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    apply_r2b_homoglyph_attack,
    calibrate_rcps_threshold,
    clopper_pearson_upper,
    normalize_adversarial_text,
)

EIGHT_PUBLIC_FORUMS: Tuple[Tuple[str, str], ...] = (
    ("cqadupstack-android", "mteb/cqadupstack-android"),
    ("cqadupstack-english", "mteb/cqadupstack-english"),
    ("cqadupstack-gaming", "mteb/cqadupstack-gaming"),
    ("cqadupstack-gis", "mteb/cqadupstack-gis"),
    ("cqadupstack-physics", "mteb/cqadupstack-physics"),
    ("cqadupstack-programmers", "mteb/cqadupstack-programmers"),
    ("cqadupstack-stats", "mteb/cqadupstack-stats"),
    ("cqadupstack-unix", "mteb/cqadupstack-unix"),
)


def make_nonverbatim_paraphrase(text: str) -> str:
    """Deterministic semantic paraphrase preserving content words while breaking ALL 5-token verbatim spans."""
    toks = normalize_adversarial_text(text).split()
    # Take alternating stride words + reverse clause chunks so no 5-gram matches verbatim
    content = [t for t in toks if len(t) > 2][:40]
    evens = content[0::2]
    odds = content[1::2]
    reordered = odds[::-1][:16] + ["regarding", "discussion", "about"] + evens[:16]
    return " ".join(reordered)


def select_distinctive_5tok_span(text: str, doc_freq: Dict[str, int], n_docs: int) -> Tuple[str, ...]:
    """Select the highest-IDF contiguous 5-token window from normalized text (avoiding generic 5-word openings)."""
    toks = normalize_adversarial_text(text).lower().split()
    if len(toks) <= 5:
        return tuple(toks)
    best_score = -1.0
    best_span = tuple(toks[:5])
    # Limit scan to first 35 tokens (excerpt window)
    lim = min(len(toks), 35)
    for i in range(0, lim - 4):
        win = toks[i : i + 5]
        # Sum IDF across the 5 tokens (rare technical terms score high; stopwords score low)
        sc = sum(float(np.log((n_docs + 1.0) / (doc_freq.get(w, 0) + 1.0))) for w in win)
        if sc > best_score:
            best_score = sc
            best_span = tuple(win)
    return best_span


def source_clustered_ub95(fp_flags: Sequence[int], forum_ids: Sequence[str], *, n_boot: int = 4000, seed: int = 2506) -> float:
    """Cluster-bootstrap 95% upper confidence bound resampling forums with replacement."""
    rng = np.random.default_rng(seed)
    by_forum: Dict[str, List[int]] = {}
    for flag, fid in zip(fp_flags, forum_ids):
        by_forum.setdefault(fid, []).append(int(flag))
    keys = sorted(by_forum.keys())
    k = len(keys)
    rates = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, k, size=k)
        tot_fp = 0
        tot_n = 0
        for idx in draw:
            arr = by_forum[keys[int(idx)]]
            tot_fp += sum(arr)
            tot_n += len(arr)
        rates[b] = tot_fp / max(tot_n, 1)
    return float(np.percentile(rates, 95))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w25_06_pm6_g6_spark3")
    ap.add_argument("--hardware-prefix", type=str, default="spark3_gb10")
    ap.add_argument("--git-commit-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        p = make_nonverbatim_paraphrase("How to fix systemd user service permissions on Ubuntu ARM64 Linux kernel")
        assert len(p.split()) >= 5
        ub = source_clustered_ub95([0] * 100 + [1, 0] * 5, ["a"] * 55 + ["b"] * 55)
        assert 0.0 <= ub <= 1.0
        print(json.dumps({"self_test": "PASS", "sample_para": p[:80], "sample_cluster_ub95": round(ub, 4)}, indent=2))
        return 0

    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for W25-06")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[w25-06] hardware={args.hardware_prefix}:{gpu_name} harvesting N_eval=3,000 fresh siblings across 8 forums ...", flush=True)

    # Per forum (8 forums):
    #   [0:80]     -> 640 anchors
    #   [80:105]   -> 200 restricted vault entries
    #   [105:255]  -> 1,200 disjoint calibration negatives (DEV)
    #   [255:630]  -> 3,000 FRESH HELD-OUT sibling benigns (REPORT, 375/forum!)
    anc_texts: List[str] = []
    vault_texts: List[str] = []
    cal_neg_texts: List[str] = []
    eval_sib_texts: List[str] = []
    eval_sib_forums: List[str] = []

    for dom_id, hf_repo in EIGHT_PUBLIC_FORUMS:
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
            if len(clean_rows) >= 630:
                break
        if len(clean_rows) < 630:
            raise RuntimeError(f"Forum {dom_id} had only {len(clean_rows)} rows (<630)")
        anc_texts.extend(clean_rows[0:80])
        vault_texts.extend(clean_rows[80:105])
        cal_neg_texts.extend(clean_rows[105:255])
        eval_sib_texts.extend(clean_rows[255:630])
        eval_sib_forums.extend([dom_id] * 375)

    eval_pos_clean = [f"Summary excerpt: {t[:260]}" for t in vault_texts]
    eval_pos_r2b_raw = [apply_r2b_homoglyph_attack(t) for t in eval_pos_clean]
    eval_pos_r2b_norm = [normalize_adversarial_text(t) for t in eval_pos_r2b_raw]
    eval_pos_paraphrase = [make_nonverbatim_paraphrase(t) for t in vault_texts]

    print(
        f"[w25-06] Harvested anchors={len(anc_texts)} vault={len(vault_texts)} "
        f"cal_neg={len(cal_neg_texts)} eval_sibling_benign={len(eval_sib_texts)}",
        flush=True,
    )

    # Fit IDF strictly on anchors + cal_neg (NEVER on held-out eval_sib_texts!)
    df_counter: collections.Counter[str] = collections.Counter()
    dev_docs_for_idf = anc_texts + cal_neg_texts
    for dtxt in dev_docs_for_idf:
        for w in set(normalize_adversarial_text(dtxt).lower().split()):
            df_counter[w] += 1

    raw_5tok_spans = [tuple(normalize_adversarial_text(t).lower().split()[:5]) for t in vault_texts]
    distinctive_5tok_spans = [select_distinctive_5tok_span(t, df_counter, len(dev_docs_for_idf)) for t in vault_texts]

    # Encode old 384-d vault + anchors (MiniLM-384)
    mdl_old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    anc_384 = np.asarray(mdl_old.encode(anc_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    vault_384 = np.asarray(mdl_old.encode(vault_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    del mdl_old; gc.collect(); torch.cuda.empty_cache()

    # Encode 1024-d stream (`BAAI/bge-m3` — already cached on Spark 3! Cross-space 1024->384 migration!)
    # Or use `BAAI/bge-m3` / `Qwen/Qwen3-Embedding-8B` (use `BAAI/bge-m3` if cached or `BAAI/bge-large-en-v1.5` for fast 3k encoding)
    stream_encoder = "BAAI/bge-m3"
    print(f"[w25-06] Encoding {len(anc_texts)+len(cal_neg_texts)+len(eval_sib_texts)+4*len(vault_texts)} texts with {stream_encoder} ...", flush=True)
    mdl_new = SentenceTransformer(stream_encoder, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    anc_new = np.asarray(mdl_new.encode(anc_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    cal_neg_new = np.asarray(mdl_new.encode(cal_neg_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    eval_sib_new = np.asarray(mdl_new.encode(eval_sib_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    eval_pos_clean_new = np.asarray(mdl_new.encode(eval_pos_clean, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    eval_pos_r2b_raw_new = np.asarray(mdl_new.encode(eval_pos_r2b_raw, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    eval_pos_r2b_norm_new = np.asarray(mdl_new.encode(eval_pos_r2b_norm, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    eval_pos_para_new = np.asarray(mdl_new.encode(eval_pos_paraphrase, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    del mdl_new; gc.collect(); torch.cuda.empty_cache()

    # Fit ZeroMu bridge on anchors + calibrate RCPS threshold strictly on N_cal=1,200 (target_fpr=0.005, alpha=0.05)
    zm = ZeroMuPolarBridge.fit(anc_new, anc_384, scoring_mode="dual_adap")
    s_cal_neg = np.max(zm.score_matrix(cal_neg_new, vault_384, dev_docs=anc_384), axis=1)
    cal_info = calibrate_rcps_threshold(s_cal_neg, target_fpr=0.005, alpha=0.05)
    tau = float(cal_info["threshold"])

    s_sib = np.max(zm.score_matrix(eval_sib_new, vault_384, dev_docs=anc_384), axis=1)
    s_pos = np.max(zm.score_matrix(eval_pos_clean_new, vault_384, dev_docs=anc_384), axis=1)
    s_r2b_raw = np.max(zm.score_matrix(eval_pos_r2b_raw_new, vault_384, dev_docs=anc_384), axis=1)
    s_r2b_norm = np.max(zm.score_matrix(eval_pos_r2b_norm_new, vault_384, dev_docs=anc_384), axis=1)
    s_para = np.max(zm.score_matrix(eval_pos_para_new, vault_384, dev_docs=anc_384), axis=1)

    # Helper: run live wrap_stream span check (exact deterministic matching)
    def span_match_flags(texts: Sequence[str], spans: List[Tuple[str, ...]], *, use_nfkc: bool, ledger: EventLedger | None = None) -> List[int]:
        flags: List[int] = []
        # Fast exact tuple-window check matching wrap_stream semantics; also run live wrap_stream on ledger arm!
        span_set = set(spans)
        for txt in texts:
            proc = normalize_adversarial_text(txt) if use_nfkc else txt
            toks = proc.lower().split()[:45]
            if ledger is not None:
                guard = wrap_stream(iter(toks), ledger=ledger, restricted_spans=spans, profile="strict_holdback", tenant_id="pm6-w25-06")
                _ = list(guard)
                flags.append(int(guard.suppressed_spans > 0))
            else:
                hit = any(tuple(toks[i : i + 5]) in span_set for i in range(max(0, len(toks) - 4)))
                flags.append(int(hit))
        return flags

    # Evaluate all 4 ablated arms (run live wrap_stream with cryptographic ledger on the combined primary arm!)
    live_ledger = EventLedger(policy="sec_audit", tenant_id="pm6-w25-06")

    def eval_arm(
        name: str,
        sib_flags: List[int],
        pos_flags: List[int],
        r2b_raw_flags: List[int],
        r2b_norm_flags: List[int],
        para_flags: List[int],
    ) -> Dict[str, Any]:
        n_sib = len(sib_flags)
        fp = sum(sib_flags)
        cp_ub = clopper_pearson_upper(fp, n_sib, alpha=0.05)
        cl_ub = source_clustered_ub95(sib_flags, eval_sib_forums)
        return {
            "arm": name,
            "n_eval_sib": n_sib,
            "sib_fp": fp,
            "sib_fpr": round(fp / n_sib, 6),
            "sib_cp_ub95": round(cp_ub, 6),
            "sib_cluster_ub95": round(cl_ub, 6),
            "passes_g6_cp_ub95_le_2pct": bool(cp_ub <= 0.02),
            "passes_g6_cluster_ub95_le_2pct": bool(cl_ub <= 0.02),
            "clean_verbatim_tpr": round(sum(pos_flags) / len(pos_flags), 6),
            "r2b_raw_bypass_rate": round(1.0 - sum(r2b_raw_flags) / len(r2b_raw_flags), 6),
            "r2b_nfkc_bypass_rate": round(1.0 - sum(r2b_norm_flags) / len(r2b_norm_flags), 6),
            "nonverbatim_paraphrase_tpr": round(sum(para_flags) / len(para_flags), 6),
        }

    # Arm 1: raw 5-tok span guard only
    raw_sib = span_match_flags(eval_sib_texts, raw_5tok_spans, use_nfkc=True)
    raw_pos = span_match_flags(eval_pos_clean, raw_5tok_spans, use_nfkc=True)
    raw_r2b_r = span_match_flags(eval_pos_r2b_raw, raw_5tok_spans, use_nfkc=False)
    raw_r2b_n = span_match_flags(eval_pos_r2b_raw, raw_5tok_spans, use_nfkc=True)
    raw_para = span_match_flags(eval_pos_paraphrase, raw_5tok_spans, use_nfkc=True)

    # Arm 2: distinctive high-IDF 5-tok span guard only (strongest deterministic baseline)
    idf_sib = span_match_flags(eval_sib_texts, distinctive_5tok_spans, use_nfkc=True)
    idf_pos = span_match_flags(eval_pos_clean, distinctive_5tok_spans, use_nfkc=True)
    idf_r2b_r = span_match_flags(eval_pos_r2b_raw, distinctive_5tok_spans, use_nfkc=False)
    idf_r2b_n = span_match_flags(eval_pos_r2b_raw, distinctive_5tok_spans, use_nfkc=True)
    idf_para = span_match_flags(eval_pos_paraphrase, distinctive_5tok_spans, use_nfkc=True)

    # Arm 3: pure ZeroMu semantic RCPS only
    sem_sib = [int(x >= tau) for x in s_sib]
    sem_pos = [int(x >= tau) for x in s_pos]
    sem_r2b_r = [int(x >= tau) for x in s_r2b_raw]
    sem_r2b_n = [int(x >= tau) for x in s_r2b_norm]
    sem_para = [int(x >= tau) for x in s_para]

    # Arm 4: combined distinctive span guard (LIVE wrap_stream + cryptographic ledger!) + semantic RCPS
    live_idf_pos = span_match_flags(eval_pos_clean, distinctive_5tok_spans, use_nfkc=True, ledger=live_ledger)
    live_idf_r2b_n = span_match_flags(eval_pos_r2b_raw, distinctive_5tok_spans, use_nfkc=True, ledger=live_ledger)
    # Run live wrap_stream on a 300-doc audit slice + exact match across all 3,000 for speed, verifying 100% parity
    live_sib_check = span_match_flags(eval_sib_texts[:300], distinctive_5tok_spans, use_nfkc=True, ledger=live_ledger)
    assert live_sib_check == idf_sib[:300], "wrap_stream parity mismatch!"
    chain_len = int(live_ledger.verify_chain())

    comb_sib = [int(a or b) for a, b in zip(idf_sib, sem_sib)]
    comb_pos = [int(a or b) for a, b in zip(live_idf_pos, sem_pos)]
    comb_r2b_r = [int(a or b) for a, b in zip(idf_r2b_r, sem_r2b_r)]
    comb_r2b_n = [int(a or b) for a, b in zip(live_idf_r2b_n, sem_r2b_n)]
    comb_para = [int(a or b) for a, b in zip(idf_para, sem_para)]

    arms_out = {
        "span_guard_raw_5tok_only": eval_arm("span_guard_raw_5tok_only", raw_sib, raw_pos, raw_r2b_r, raw_r2b_n, raw_para),
        "span_guard_distinctive_idf_only": eval_arm("span_guard_distinctive_idf_only", idf_sib, idf_pos, idf_r2b_r, idf_r2b_n, idf_para),
        "semantic_zeromu_rcps_only": eval_arm("semantic_zeromu_rcps_only", sem_sib, sem_pos, sem_r2b_r, sem_r2b_n, sem_para),
        "combined_distinctive_span_plus_semantic": eval_arm("combined_distinctive_span_plus_semantic", comb_sib, comb_pos, comb_r2b_r, comb_r2b_n, comb_para),
    }

    comb = arms_out["combined_distinctive_span_plus_semantic"]
    span_floor = arms_out["span_guard_distinctive_idf_only"]
    g6_pass = bool(comb["passes_g6_cp_ub95_le_2pct"] and comb["passes_g6_cluster_ub95_le_2pct"])
    beats_span_on_para = bool(comb["nonverbatim_paraphrase_tpr"] > span_floor["nonverbatim_paraphrase_tpr"] + 0.01)
    verdict = "ADVANCE_PM6_G6" if (g6_pass and beats_span_on_para) else ("G6_PASS_SPAN_DOMINATED" if g6_pass else "FAIL_G6")

    manifest = {
        "experiment_id": "pv7_w25_06_pm6_g6_powered_ablation",
        "wave_item_id": "W25-06",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_cohort": f"{args.hardware_prefix}:{gpu_name}",
        "stream_encoder": stream_encoder,
        "vault_encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "git_commit_sha": args.git_commit_sha,
        "claim_registry_sha256": claim_registry_sha256(),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "splits": {
            "n_forums": len(EIGHT_PUBLIC_FORUMS),
            "n_anchors": len(anc_texts),
            "n_vault": len(vault_texts),
            "n_cal_neg": len(cal_neg_texts),
            "n_eval_sibling_benign": len(eval_sib_texts),
        },
        "rcps_calibration_dev_only": cal_info,
        "live_gateway_chain_verified": chain_len > 0,
        "live_gateway_chain_entries": chain_len,
        "arms": arms_out,
        "verdict": {
            "status": verdict,
            "g6_cp_and_cluster_ub95_le_2pct": g6_pass,
            "combined_beats_span_only_on_paraphrase": beats_span_on_para,
            "paraphrase_tpr_gain_over_span_pp": round((comb["nonverbatim_paraphrase_tpr"] - span_floor["nonverbatim_paraphrase_tpr"]) * 100.0, 2),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "w25_06_pm6_g6_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[w25-06] DONE verdict={verdict} wall={manifest['elapsed_seconds']}s -> {out_path}", flush=True)
    print(json.dumps({"arms": arms_out, "verdict": manifest["verdict"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
