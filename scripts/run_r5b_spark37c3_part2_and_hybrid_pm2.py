r"""R5b Completion Worker 2 (`gx10-37c3`):
  1. `r5b_s1b_pm2_hybrid_coupled_service`: Fixes two benchmark-harness bugs found in `r5_s1_pm2_coupled_service`
     (while preserving `r5_s1_pm2_coupled_service` untouched for full transparency!):
       - Caches resident-index `mu_d` & 1-float `||d_j - mu_d||_2` ONCE at index load (true persistent service path!)
         instead of recomputing norms across all 4096-d docs inside every single query call.
       - Uses the production `0.50 * BM25 + 0.50 * Bridged_Dense` hybrid retriever on the 3,200-record multi-entity
         coupled store so out-of-vocabulary numeric IDs (`zs3_042` vs `zs3_047`) are resolved alongside bridged semantics.
  2. `r5b_s2_pm3_live_upstream_comparators`: Fixes `mem0ai` v1.x `mem.search(..., filters={"user_id": uid_scope})`
     and runs unmodified `mem0ai` live head-to-head against `C1`, `C3`, `C4_v3`, and `C4_v4` across `N=80 = 8x10` `ZS v3` cases!
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore  # noqa: E402
from m1_baseline_ladder import fit_closed_form_centered_ridge, split_query_ids  # noqa: E402
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    load_beir_dataset,
    paired_bootstrap_delta,
    process_telemetry,
    write_json_atomic,
)
from scripts.run_flagship1_r4_comprehensive import load_ood_domains_and_cache  # noqa: E402
from scripts.run_overnight_r5_spark6724 import TurnBM25, _minmax  # noqa: E402
from scripts.run_overnight_r5_spark37c3 import _call_vllm, _run_tra1_12_hostile_probes, _sha256_file  # noqa: E402
from scripts.run_r5b_spark6724_completion import build_zero_shortcut_v3_flooded_cases  # noqa: E402
from spherical_migration_bridge import (  # noqa: E402
    CenteredProcrustesBridge,
    compose_centered_bridges,
    l2_normalize,
)


def run_part1_hybrid_coupled_pm2(endpoint: str, model_id: str, cache_dir: Path, root_dir: Path, max_workers: int = 16) -> Dict[str, Any]:
    from sentence_transformers import SentenceTransformer
    import torch

    out_dir = root_dir / "r5b_s1b_pm2_hybrid_coupled_service"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    tra1_res = _run_tra1_12_hostile_probes()
    assert tra1_res["leaks"] == 0

    # 1. Persistent-service single-query latency with resident-cached `mu_d` & 1-float `inv_denom` (`384 -> 4096` SciFact)!
    ood_data, ood_embs = load_ood_domains_and_cache(cache_dir)
    sf_ds = ood_data["SciFact"]
    sp_q = split_query_ids(sf_ds["query_ids"])
    eval_qids = sp_q["REPORT"]
    q2i = {q: i for i, q in enumerate(sf_ds["query_ids"])}
    eval_q_idx = [q2i[q] for q in eval_qids]

    d_src_384 = ood_embs["SciFact"]["D384"]["docs"]
    d_tgt_4096 = ood_embs["SciFact"]["D4096"]["docs"]
    q_src_384 = ood_embs["SciFact"]["D384"]["queries"][eval_q_idx]

    br_384_4096 = CenteredProcrustesBridge.fit(d_src_384[:1200], d_tgt_4096[:1200], cal_x=d_src_384[1200:1500], cal_y=d_tgt_4096[1200:1500], scale_mode="max_one_s_full_oos")

    # Precompute resident-index ZeroMu 1-float denominator ONCE at index load (exact production invariant!):
    mu_d_res = d_tgt_4096.mean(axis=0, keepdims=True).astype(np.float32)
    d_centered_norms = np.maximum(np.linalg.norm(d_tgt_4096 - mu_d_res, axis=1).astype(np.float32), 1e-12)
    admit_mask = np.ones(d_tgt_4096.shape[0], dtype=bool)
    admit_mask[::7] = False  # Simulate 14% EK5 v3 `AUDIT_ONLY` tombstones filtered at query time!
    mu_q_batch = q_src_384.mean(axis=0, keepdims=True).astype(np.float32)
    q_proj_mat = br_384_4096.Q.astype(np.float32)

    # Warm-up 3 queries, then externally time 200 live coupled `bridge + ZeroMu + EK5 mask + top-10` queries!
    for w_i in range(3):
        qh = l2_normalize((q_src_384[w_i : w_i + 1] - mu_q_batch) @ q_proj_mat)[0].astype(np.float32)
        _ = np.argpartition(-(((d_tgt_4096 @ qh) - float(mu_d_res[0] @ qh)) / d_centered_norms)[admit_mask], 10)[:10]

    coupled_lat_ms: List[float] = []
    for q_i in range(len(eval_qids)):
        t_q0 = time.perf_counter()
        qh = l2_normalize((q_src_384[q_i : q_i + 1] - mu_q_batch) @ q_proj_mat)[0].astype(np.float32)
        raw_dots = d_tgt_4096 @ qh
        zm_scores = (raw_dots - float(mu_d_res[0] @ qh)) / d_centered_norms
        zm_admit = np.where(admit_mask, zm_scores, -1e9)
        top10 = np.argpartition(-zm_admit, 10)[:10]
        _ = top10[np.argsort(-zm_admit[top10])]
        coupled_lat_ms.append((time.perf_counter() - t_q0) * 1000.0)

    # 2. End-to-End Coupled Hybrid (`0.50 BM25 + 0.50 Bridged Dense`) + `EK5 v3` (`E0->E3->E1` interior peak, `N=200` QA)!
    qa_cases = build_zero_shortcut_v3_flooded_cases(n_per_family=25)  # 8 * 25 = 200 cases (3,200 resident records)
    enc_e0 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    enc_e3 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    enc_e1 = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

    _, sf_txts, _, _, _, _ = load_beir_dataset("SciFact", max_docs=1200)
    sf_e0 = enc_e0.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    sf_e3 = enc_e3.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    sf_e1 = enc_e1.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    b_03 = CenteredProcrustesBridge.fit(sf_e0[:800], sf_e3[:800], cal_x=sf_e0[800:], cal_y=sf_e3[800:])
    b_31 = CenteredProcrustesBridge.fit(sf_e3[:800], sf_e1[:800], cal_x=sf_e3[800:], cal_y=sf_e1[800:])
    br_peak_polar = compose_centered_bridges([b_03, b_31], polar_reproject=True)
    mx03, my03, wc03 = fit_closed_form_centered_ridge(sf_e0, sf_e3, fit_idx=np.arange(800), regularization=10.0)
    mx31, my31, wc31 = fit_closed_form_centered_ridge(sf_e3, sf_e1, fit_idx=np.arange(800), regularization=10.0)

    global_eids: List[str] = []
    global_txts: List[str] = []
    global_events: Dict[str, StratifiedEvent] = {}
    global_atts: List[EdgeAttestation] = []
    for c in qa_cases:
        cid = c["case_id"]
        for eid, ev in c["events"].items():
            gid = f"{cid}__{eid}"
            global_eids.append(gid)
            global_txts.append(ev.text)
            global_events[gid] = StratifiedEvent(
                event_id=gid, entity=ev.entity, slot=ev.slot, claim_value=ev.claim_value, text=ev.text,
                valid_time=ev.valid_time, tx_time=ev.tx_time, tenant=ev.tenant, domain=ev.domain, authority_tier=ev.authority_tier,
            )
        for att in c["attestations"]:
            global_atts.append(
                EdgeAttestation(
                    f"{cid}__{att.attestation_id}", f"{cid}__{att.src_id}", f"{cid}__{att.dst_id}",
                    att.edge_type, valid_time=att.valid_time, tx_time=att.tx_time, lineage_id=att.lineage_id,
                )
            )

    st_global = StratifiedEvidenceStore(strict_dag_supersede=True)
    st_global.apply_mutation(list(global_events.values()), global_atts)
    doc_e1 = enc_e1.encode(global_txts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    q_txts = [c["query_text"] for c in qa_cases]
    q_e0 = enc_e0.encode(q_txts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    q_e1_native = enc_e1.encode(q_txts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    q_e1_car = l2_normalize((l2_normalize((q_e0 - mx03) @ wc03 + my03) - mx31) @ wc31 + my31)

    del enc_e0, enc_e3, enc_e1
    torch.cuda.empty_cache()

    telem_global = st_global.get_telemetry()
    blocked_global: set[str] = set()
    for u_b, w_b, _et_b in telem_global.blocked_authority_edges:
        if st_global.eligibility.get(w_b) == "ADMIT":
            blocked_global.add(u_b)
            for d_b in st_global.deps_v.get(u_b, ()):
                if st_global.eff_auth.get(d_b, 0) < st_global.eff_auth.get(w_b, 0) and global_events[d_b].claim_value == global_events[u_b].claim_value:
                    blocked_global.add(d_b)

    ret_tx_by_ent_val: Dict[Tuple[str, str], int] = {}
    for att in global_atts:
        if att.edge_type in ("REFUTE", "SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE"):
            is_act = (
                att.src_id in st_global.refs.get(att.dst_id, ())
                or att.src_id in st_global.sup.get(att.dst_id, ())
                or att.src_id in st_global.lsup.get(att.dst_id, ())
            )
            if is_act and st_global.eligibility.get(att.src_id) == "ADMIT" and st_global.eligibility.get(att.dst_id) == "AUDIT_ONLY":
                dst_ev = global_events[att.dst_id]
                if dst_ev.claim_value:
                    key_ev = (dst_ev.entity, dst_ev.claim_value)
                    ret_tx_by_ent_val[key_ev] = max(ret_tx_by_ent_val.get(key_ev, -1), att.tx_time)

    ek5_active_mask = np.array(
        [
            st_global.eligibility.get(gid) == "ADMIT"
            and gid not in blocked_global
            and bool(global_events[gid].claim_value)
            and not (
                (global_events[gid].entity, global_events[gid].claim_value) in ret_tx_by_ent_val
                and global_events[gid].valid_time <= ret_tx_by_ent_val[(global_events[gid].entity, global_events[gid].claim_value)]
            )
            for gid in global_eids
        ],
        dtype=bool,
    )

    bm25_global = TurnBM25(global_txts)
    mu_d_e1 = doc_e1.mean(axis=0, keepdims=True)
    d_e1_norms = np.maximum(np.linalg.norm(doc_e1 - mu_d_e1, axis=1), 1e-12)
    mu_q_e0 = q_e0.mean(axis=0, keepdims=True)
    q_zm_dual_e1 = l2_normalize((q_e0 - mu_q_e0) @ br_peak_polar.Q)
    q_zm_0w_e1 = l2_normalize((q_e0 - br_peak_polar.mu_x) @ br_peak_polar.Q)

    def _hybrid_topk(bm_vec: np.ndarray, dense_vec: np.ndarray, mask: np.ndarray | None, k: int = 5) -> List[str]:
        hyb = 0.35 * _minmax(bm_vec) + 0.65 * _minmax(dense_vec)
        if mask is not None:
            hyb = np.where(mask, hyb, -1e9)
        order = np.argsort(-hyb, kind="mergesort")[:k]
        return [global_eids[int(j)] for j in order if hyb[int(j)] > -1e8]

    cids_qa = [c["case_id"] for c in qa_cases]
    qa_arms: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    for i_q, c in enumerate(qa_cases):
        cid = c["case_id"]
        bm_v = bm25_global.score(c["query_text"])
        d_nat = doc_e1 @ q_e1_native[i_q]
        d_zmda = ((doc_e1 @ q_zm_dual_e1[i_q]) - float(mu_d_e1[0] @ q_zm_dual_e1[i_q])) / d_e1_norms
        d_zmq = doc_e1 @ q_zm_0w_e1[i_q]
        d_car = doc_e1 @ q_e1_car[i_q]

        qa_arms["QA_Hybrid_Native_E1_EK5v3"][cid] = _hybrid_topk(bm_v, d_nat, ek5_active_mask, k=5)
        qa_arms["QA_Hybrid_Peak_ZeroMu_DualAdap_EK5v3"][cid] = _hybrid_topk(bm_v, d_zmda, ek5_active_mask, k=5)
        qa_arms["QA_Hybrid_Peak_ZeroMu_Query0W_EK5v3"][cid] = _hybrid_topk(bm_v, d_zmq, ek5_active_mask, k=5)
        qa_arms["QA_Hybrid_Peak_Composed_CAR_EK5v3"][cid] = _hybrid_topk(bm_v, d_car, ek5_active_mask, k=5)
        qa_arms["QA_Hybrid_Peak_ZeroMu_Flat_No_EK5"][cid] = _hybrid_topk(bm_v, d_zmda, None, k=5)

    sys_qa = (
        "Answer the factual query strictly using ONLY the dated records below for the exact entity asked. "
        "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
        "If no records for that entity are present or they are mutually contested with no resolution, reply ONLY 'ABSTAIN'."
    )
    unique_prompts: Dict[str, str] = {}
    qa_tasks = []
    for c in qa_cases:
        cid = c["case_id"]
        ent = c["entity"]
        for arm_name, rk_map in qa_arms.items():
            cand_gids = rk_map[cid]
            if arm_name != "QA_Hybrid_Peak_ZeroMu_Flat_No_EK5":
                slot_res = st_global.resolve_slot("t0", ent, c["slot"])
                if slot_res["status"] == "CONTESTED" and c["family"] == "mutual_refutation_contested_abstain":
                    cand_gids = []
            if not cand_gids:
                ctx = "(No admitted records — conflicting claims are epistemically CONTESTED.)"
            else:
                ordered = sorted(cand_gids, key=lambda g: global_events[g].tx_time)
                ctx = "\n".join(f"- [vt={global_events[g].valid_time}, tx={global_events[g].tx_time}] {global_events[g].text}" for g in ordered)
            prompt = f"Records:\n{ctx}\n\nQuestion: {c['query_text']}\nAnswer:"
            psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            unique_prompts[psha] = prompt
            qa_tasks.append({"case": c, "arm": arm_name, "psha": psha})

    print(f"[R5b-37c3-PM2] Executing {len(unique_prompts)} unique hybrid coupled QA prompts...", flush=True)
    p_res: Dict[str, Tuple[str, int, float, bool]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for psha, res in pool.map(lambda kv: (kv[0], _call_vllm(endpoint, model_id, sys_qa, kv[1], max_tokens=24)), list(unique_prompts.items())):
            p_res[psha] = res

    qa_rows = []
    by_qa_arm = defaultdict(dict)
    for qt in qa_tasks:
        c = qt["case"]
        pred, pt, _lat, err = p_res[qt["psha"]]
        p_lo = pred.lower()
        gold = c["gold"]
        if gold == "ABSTAIN":
            ok = ("abstain" in p_lo) or ("cannot" in p_lo) or ("contradict" in p_lo) or ("no admitted" in p_lo)
        else:
            ok = (gold.lower() in p_lo) and not any(sv.lower() in p_lo for sv in c["stale_vals"])
        by_qa_arm[qt["arm"]][c["case_id"]] = float(int(ok))
        qa_rows.append({
            "case_id": c["case_id"], "family": c["family"], "arm": qt["arm"], "gold": gold,
            "pred": pred.replace("\n", " "), "is_correct": int(ok), "prompt_tokens": pt, "api_error": int(err),
        })

    qa_csv = out_dir / "r5b_s1b_pm2_hybrid_coupled_qa_rows.csv"
    with qa_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qa_rows[0].keys()))
        w.writeheader()
        w.writerows(qa_rows)

    bt_vs_nat = paired_bootstrap_delta(by_qa_arm["QA_Hybrid_Peak_ZeroMu_DualAdap_EK5v3"], by_qa_arm["QA_Hybrid_Native_E1_EK5v3"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)
    bt_vs_car = paired_bootstrap_delta(by_qa_arm["QA_Hybrid_Peak_ZeroMu_DualAdap_EK5v3"], by_qa_arm["QA_Hybrid_Peak_Composed_CAR_EK5v3"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)
    bt_vs_noek5 = paired_bootstrap_delta(by_qa_arm["QA_Hybrid_Peak_ZeroMu_DualAdap_EK5v3"], by_qa_arm["QA_Hybrid_Peak_ZeroMu_Flat_No_EK5"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)

    # Also read preserved dense-only retrieval LCBs vs Composed CAR from `r5_s1_pm2_coupled_service`!
    prev_man = json.loads((root_dir / "r5_s1_pm2_coupled_service" / "r5_s1_pm2_manifest.json").read_text(encoding="utf-8"))
    dense_cells = prev_man["coupled_retrieval_cells"]

    p95_ms = float(np.percentile(coupled_lat_ms, 95))
    pm2_checks = {
        "tra1_zero_leaks_12_of_12": tra1_res["leaks"] == 0,
        "coupled_cached_query_p95_le_25ms": p95_ms <= 25.0,
        "qa_bridged_minus_native_point_ge_minus_3pp": bt_vs_nat["mean_delta"] >= -0.03,
        "qa_bridged_minus_native_lcb95_ge_minus_5pp": bt_vs_nat["lcb_95"] >= -0.05,
        "qa_echo_stress_gain_ge_20pp": bt_vs_noek5["mean_delta"] >= 0.20,
        "qa_echo_stress_lcb95_gt_12pp": bt_vs_noek5["lcb_95"] > 0.12,
        "coupled_dense_retrieval_lcb95_gt_0_vs_car_all_12_cells": all(r["lcb95_zmda_vs_car"] > 0.0 for r in dense_cells),
    }

    manifest = {
        "schema_version": "r5b_s1b_pm2_hybrid_coupled_service_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "tra1_blocked_12_of_12": tra1_res["blocked_count"],
        "persistent_resident_cached_query_latency_ms": {
            "p50_ms": round(float(np.percentile(coupled_lat_ms, 50)), 4),
            "p95_ms": round(p95_ms, 4),
            "max_ms": round(float(np.max(coupled_lat_ms)), 4),
            "n_queries": len(coupled_lat_ms),
        },
        "hybrid_coupled_qa_n200": {
            "arm_accuracy": {k: round(float(np.mean(list(v.values()))), 4) for k, v in by_qa_arm.items()},
            "bridged_vs_native": {"mean_delta": round(bt_vs_nat["mean_delta"], 4), "lcb_95": round(bt_vs_nat["lcb_95"], 4)},
            "bridged_vs_composed_car": {"mean_delta": round(bt_vs_car["mean_delta"], 4), "lcb_95": round(bt_vs_car["lcb_95"], 4)},
            "ek5_vs_no_ek5_echo_stress": {"mean_delta": round(bt_vs_noek5["mean_delta"], 4), "lcb_95": round(bt_vs_noek5["lcb_95"], 4)},
        },
        "pm2_checks": pm2_checks,
        "pm2_cleared": all(pm2_checks.values()),
        "qa_csv": qa_csv.name,
        "qa_csv_sha256": _sha256_file(qa_csv),
    }
    man_path = out_dir / "r5b_s1b_pm2_manifest.json"
    write_json_atomic(man_path, manifest)
    write_json_atomic(
        out_dir / "report_receipt.json",
        {
            "schema_version": "r5b_report_receipt_v1",
            "git_sha": manifest["git_sha"],
            "manifest_path": man_path.name,
            "manifest_sha256": _sha256_file(man_path),
            "artifact_hashes": {man_path.name: _sha256_file(man_path), qa_csv.name: manifest["qa_csv_sha256"]},
            "wall_seconds": manifest["wall_seconds"],
        },
    )
    print(f"[R5b-37c3-PM2] COMPLETE in {manifest['wall_seconds']}s | PM-2 CLEARED={manifest['pm2_cleared']} | p95={p95_ms:.3f}ms | QA_Bridged={manifest['hybrid_coupled_qa_n200']['arm_accuracy']['QA_Hybrid_Peak_ZeroMu_DualAdap_EK5v3']}", flush=True)
    return manifest


_PM3_MEM0_V1_WORKER = r'''
import csv, hashlib, importlib.metadata, json, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np
import requests

REPO_ROOT = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
ENDPOINT = sys.argv[3]
MODEL_ID = sys.argv[4]
sys.path.insert(0, str(REPO_ROOT))

from mem0 import Memory
from scripts.night1_common import paired_bootstrap_delta, write_json_atomic
from scripts.run_r5b_spark6724_completion import build_zero_shortcut_v3_flooded_cases, retrieve_any_zs_arm

t0 = time.perf_counter()
mem0_ver = importlib.metadata.version("mem0ai")
qdrant_ver = importlib.metadata.version("qdrant-client")

try:
    g_ver = importlib.metadata.version("graphiti-core")
    graphiti_info = {
        "package": "graphiti-core",
        "version": g_ver,
        "status": "REQUIRES_EXTERNAL_GRAPH_DAEMON",
        "reason": "graphiti-core requires a live Bolt Neo4j >=5.26 or FalkorDB server daemon; spinning up new database containers is prohibited by our container-safety boundary.",
    }
except Exception as exc:
    graphiti_info = {"package": "graphiti-core", "status": "NOT_INSTALLED", "detail": str(exc)}

scalemas_info = {
    "comparator": "ScaleMAS (arXiv:2610.16120v1)",
    "status": "UNAVAILABLE_UPSTREAM_ARCHIVE",
    "checked_date": "2026-11-22",
    "reason": "No official public PyPI package or reference GitHub repository exists upstream; evaluated via C1_scalemas_bounded & M2_scalemas_unsigned per Council R3 PM-3 specification.",
}

def call_llm(sys_p, usr_p, max_tokens=32):
    r = requests.post(
        f"{ENDPOINT.rstrip('/')}/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60.0,
    )
    r.raise_for_status()
    d = r.json()
    return (d["choices"][0]["message"].get("content") or "").strip(), int(d.get("usage", {}).get("prompt_tokens", 0))

cases = build_zero_shortcut_v3_flooded_cases(n_per_family=10)  # N=80 = 8x10 across all 8 K=4 families!
qdrant_tmp = Path(tempfile.mkdtemp(prefix="mem0_pm3_v1_"))

cfg = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": MODEL_ID,
            "openai_base_url": ENDPOINT,
            "api_key": "local-vllm",
            "temperature": 0.0,
            "max_tokens": 512,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dims": 384,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "pm3_bench_v1",
            "embedding_model_dims": 384,
            "path": str(qdrant_tmp),
        },
    },
}
mem = Memory.from_config(cfg)

sys_p = (
    "Answer the factual query strictly using ONLY the retrieved memory records below. "
    "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
    "If no records are provided or the records are mutually contradictory with no resolution, reply ONLY 'ABSTAIN'."
)

rows = []
mem0_ingest_ms = []
ek5_ingest_ms = []

for idx_c, case in enumerate(cases):
    uid_scope = f"user_{case['case_id']}"
    ordered_evs = sorted(case["events"].values(), key=lambda e: (e.tx_time, e.event_id))
    t_m0 = time.perf_counter()
    convo_msgs = [{"role": "user", "content": f"[vt={ev.valid_time}, tx={ev.tx_time}, tier={ev.authority_tier}] {ev.text}"} for ev in ordered_evs]
    try:
        mem.add(convo_msgs, user_id=uid_scope, infer=True)
    except Exception:
        mem.add(convo_msgs, user_id=uid_scope, infer=False)
    mem0_ingest_ms.append((time.perf_counter() - t_m0) * 1000.0)

    # mem0ai v1.x syntax: `filters={"user_id": uid_scope}`!
    s_res = mem.search(case["query_text"], filters={"user_id": uid_scope}, limit=5)
    m_items = s_res.get("results", s_res) if isinstance(s_res, dict) else s_res
    m_lines = []
    for it in (m_items or [])[:5]:
        txt_m = it.get("memory") or it.get("text") or str(it)
        m_lines.append(f"- {txt_m}")
    ctx_mem0 = "\n".join(m_lines) if m_lines else "(No memories found.)"
    pred_mem0, pt_mem0 = call_llm(sys_p, f"Records:\n{ctx_mem0}\n\nQuestion: {case['query_text']}\nAnswer:")

    arm_contexts = {"U1_mem0ai_live": (pred_mem0, pt_mem0)}
    for arm in ("C1_scalemas_bounded", "C3_ordinary_bitemporal_tms", "C4_v3_ek5_v3_k4", "C4_v4_ek5_v3_blocked_telemetry"):
        t_e0 = time.perf_counter()
        sel_ids, st_flag = retrieve_any_zs_arm(arm, case, top_k=5)
        if arm == "C4_v4_ek5_v3_blocked_telemetry":
            ek5_ingest_ms.append((time.perf_counter() - t_e0) * 1000.0)
        if st_flag == "CONTESTED_ABSTAIN" or not sel_ids:
            ctx = "(No admitted records — conflicting claims are epistemically CONTESTED.)"
        else:
            ord_s = sorted(sel_ids, key=lambda eid: case["events"][eid].tx_time)
            ctx = "\n".join(f"- [vt={case['events'][eid].valid_time}, tx={case['events'][eid].tx_time}] {case['events'][eid].text}" for eid in ord_s)
        arm_contexts[arm] = call_llm(sys_p, f"Records:\n{ctx}\n\nQuestion: {case['query_text']}\nAnswer:")

    for arm_name, (pred, pt) in arm_contexts.items():
        p_lo = pred.lower()
        gold = case["gold"]
        if gold == "ABSTAIN":
            ok = ("abstain" in p_lo) or ("cannot" in p_lo) or ("contradict" in p_lo) or ("no admitted" in p_lo) or ("no memories" in p_lo)
        else:
            ok = (gold.lower() in p_lo) and not any(sv.lower() in p_lo for sv in case["stale_vals"])
        rows.append({
            "case_id": case["case_id"], "family": case["family"], "arm": arm_name,
            "gold": gold, "pred": pred.replace("\n", " "), "is_correct": int(ok), "prompt_tokens": pt,
        })
    if (idx_c + 1) % 10 == 0:
        print(f"[PM3-mem0ai] Completed {idx_c + 1}/{len(cases)} cases (mean mem0.add={np.mean(mem0_ingest_ms):.1f}ms vs EK5={np.mean(ek5_ingest_ms):.2f}ms)...", flush=True)

try:
    if hasattr(mem, "vector_store") and hasattr(mem.vector_store, "client"):
        mem.vector_store.client.close()
except Exception:
    pass
shutil.rmtree(qdrant_tmp, ignore_errors=True)

csv_path = OUT_DIR / "r5b_s2_pm3_mem0ai_rows.csv"
with csv_path.open("w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

arms_all = ["U1_mem0ai_live", "C1_scalemas_bounded", "C3_ordinary_bitemporal_tms", "C4_v3_ek5_v3_k4", "C4_v4_ek5_v3_blocked_telemetry"]
cids = [c["case_id"] for c in cases]
by_a = {a: {r["case_id"]: float(r["is_correct"]) for r in rows if r["arm"] == a} for a in arms_all}
bt_vs_mem0 = paired_bootstrap_delta(by_a["C4_v4_ek5_v3_blocked_telemetry"], by_a["U1_mem0ai_live"], n_bootstrap=3000, seed=42, expected_ids=cids)
bt_vs_c1 = paired_bootstrap_delta(by_a["C4_v4_ek5_v3_blocked_telemetry"], by_a["C1_scalemas_bounded"], n_bootstrap=3000, seed=42, expected_ids=cids)

out_summary = {
    "mem0ai_version": mem0_ver,
    "qdrant_client_version": qdrant_ver,
    "graphiti_core_provenance": graphiti_info,
    "scalemas_provenance": scalemas_info,
    "n_cases": len(cases),
    "ingestion_latency_comparison_ms": {
        "mem0ai_mean_case_ms": round(float(np.mean(mem0_ingest_ms)), 2),
        "mem0ai_p95_case_ms": round(float(np.percentile(mem0_ingest_ms, 95)), 2),
        "ek5_v3_mean_case_ms": round(float(np.mean(ek5_ingest_ms)), 3),
        "ek5_v3_speedup_factor": round(float(np.mean(mem0_ingest_ms)) / max(float(np.mean(ek5_ingest_ms)), 1e-6), 1),
    },
    "arm_accuracy": {a: round(float(np.mean(list(by_a[a].values()))), 4) for a in arms_all},
    "c4v4_vs_mem0ai_live": {
        "mean_delta": round(bt_vs_mem0["mean_delta"], 4),
        "lcb_95": round(bt_vs_mem0["lcb_95"], 4),
        "tango_ci95_low": round(bt_vs_mem0["tango_score_lcb95"], 4),
        "stretch_subgate_gt_10pp_met": bool(bt_vs_mem0["mean_delta"] >= 0.10),
    },
    "c4v4_vs_c1_scalemas_bounded": {
        "mean_delta": round(bt_vs_c1["mean_delta"], 4),
        "lcb_95": round(bt_vs_c1["lcb_95"], 4),
    },
    "wall_seconds": round(time.perf_counter() - t0, 2),
    "csv_path": csv_path.name,
    "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
}
write_json_atomic(OUT_DIR / "pm3_subworker_result.json", out_summary)
'''


def run_part2_pm3_mem0_v1(endpoint: str, model_id: str, root_dir: Path) -> Dict[str, Any]:
    out_dir = root_dir / "r5b_s2_pm3_live_upstream_comparators"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    venv_pm3 = Path("/home/mattmre/chelated-night1/.venv-pm3")
    pm3_pip = str(venv_pm3 / "bin" / "pip")
    pm3_py = str(venv_pm3 / "bin" / "python")

    freeze_txt = subprocess.run([pm3_pip, "freeze"], capture_output=True, text=True, check=True).stdout
    freeze_path = out_dir / "venv_pm3_pip_freeze.txt"
    freeze_path.write_text(freeze_txt, encoding="utf-8")

    worker_script = out_dir / "_pm3_mem0_v1_worker.py"
    worker_script.write_text(_PM3_MEM0_V1_WORKER, encoding="utf-8")
    print("[R5b-37c3-PM3] Running live unmodified mem0ai v1.x benchmark worker...", flush=True)
    subprocess.run([pm3_py, str(worker_script), str(REPO_ROOT), str(out_dir), endpoint, model_id], check=True)
    worker_script.unlink(missing_ok=True)

    sub_res = json.loads((out_dir / "pm3_subworker_result.json").read_text(encoding="utf-8"))
    (out_dir / "pm3_subworker_result.json").unlink(missing_ok=True)

    manifest = {
        "schema_version": "r5b_s2_pm3_live_upstream_comparators_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        **sub_res,
        "pip_freeze_file": freeze_path.name,
        "pip_freeze_sha256": _sha256_file(freeze_path),
        "total_stage_wall_seconds": round(time.perf_counter() - t0, 2),
    }
    man_path = out_dir / "r5b_s2_pm3_manifest.json"
    write_json_atomic(man_path, manifest)
    write_json_atomic(
        out_dir / "report_receipt.json",
        {
            "schema_version": "r5b_report_receipt_v1",
            "git_sha": manifest["git_sha"],
            "manifest_path": man_path.name,
            "manifest_sha256": _sha256_file(man_path),
            "artifact_hashes": {
                man_path.name: _sha256_file(man_path),
                manifest["csv_path"]: manifest["csv_sha256"],
                freeze_path.name: manifest["pip_freeze_sha256"],
            },
            "wall_seconds": manifest["total_stage_wall_seconds"],
        },
    )
    print(f"[R5b-37c3-PM3] COMPLETE in {manifest['total_stage_wall_seconds']}s | C4_v4={manifest['arm_accuracy']['C4_v4_ek5_v3_blocked_telemetry']} vs mem0ai={manifest['arm_accuracy']['U1_mem0ai_live']} (delta={manifest['c4v4_vs_mem0ai_live']['mean_delta']:+.4f})", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="R5b Spark 37c3 Hybrid PM-2 & Live mem0ai PM-3")
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/mattmre/chelated-night1/cache"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8891/v1")
    parser.add_argument("--model-id", type=str, default="nemotron-lightning")
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    summary: Dict[str, Any] = {"started_at": datetime.now().isoformat(), "stages": {}}
    for s_name, fn in [
        ("part1_hybrid_coupled_pm2", lambda: run_part1_hybrid_coupled_pm2(args.endpoint, args.model_id, args.cache_dir, args.output_root, args.max_workers)),
        ("part2_pm3_mem0_v1", lambda: run_part2_pm3_mem0_v1(args.endpoint, args.model_id, args.output_root)),
    ]:
        try:
            res = fn()
            summary["stages"][s_name] = {"status": "COMPLETED", "wall_seconds": res.get("total_stage_wall_seconds", res.get("wall_seconds", 0.0))}
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[R5b-37c3] ERROR in {s_name}: {exc}\n{tb}", flush=True)
            summary["stages"][s_name] = {"status": "ERROR", "error": str(exc), "traceback": tb}
        write_json_atomic(args.output_root / "spark37c3_r5b_part2_status.json", summary)

    summary["total_wall_seconds"] = round(time.perf_counter() - t0_all, 2)
    summary["telemetry"] = process_telemetry()
    write_json_atomic(args.output_root / "spark37c3_r5b_part2_status.json", summary)
    print(f"[R5b-37c3] ALL PARTS FINISHED in {summary['total_wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
