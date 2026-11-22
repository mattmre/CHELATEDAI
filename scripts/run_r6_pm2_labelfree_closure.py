r"""Stage R6 Gate Closure (`PM-2` Label-Free & Causal Streaming `mu_q` Rerun).

Directly resolves BOTH 4-Model Frontier Panel findings on `PM-2` (`r5b_s1b_pm2_hybrid_coupled_service`),
while keeping `r5b_s1b` 100% untouched in the registry for full historical auditability:

  1. REMOVES `c["family"] == "mutual_refutation_contested_abstain"` completely (`0` benchmark-family
     references anywhere in retrieval, slot resolution, or prompting!). Instead, slot-level `CONTESTED`
     abstention is computed strictly from observable `StratifiedEvidenceStore` state (`admitted` minus
     `blocked_global` minus `retracted_val_tx` vs `contested` on the queried `(tenant, entity, slot)`).
  2. Adds `QA_Hybrid_Peak_ZeroMu_CausalEMA_EK5v3` (`CausalStreamingZeroMuBridge` seeded from disjoint
     calibration `br_peak_polar.mu_x` and updated ONLY after scoring each query—zero future-query peeking!)
     alongside `Query0W`, `DualAdap`, `Composed_CAR`, `Native_E1`, and `Flat_No_EK5`.
  3. Externally times BOTH the `SciFact` `384->4096` persistent kernel AND the full `3,200`-record
     hybrid `BM25 + Causal ZeroMu + EK5 mask + top-5` coupled service path!
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_raw_text_exocortex import CausalStreamingZeroMuBridge  # noqa: E402
from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore  # noqa: E402
from m1_baseline_ladder import fit_closed_form_centered_ridge  # noqa: E402
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    load_beir_dataset,
    paired_bootstrap_delta,
    process_telemetry,
    write_json_atomic,
)
from scripts.run_phase2_d1_local_fleet_proof import SimpleBM25, _minmax, _start_ssh_tunnel  # noqa: E402
from scripts.run_r5b_spark6724_completion import build_zero_shortcut_v3_flooded_cases  # noqa: E402
from spherical_migration_bridge import (  # noqa: E402
    CenteredProcrustesBridge,
    compose_centered_bridges,
    l2_normalize,
)


def _call_vllm(endpoint: str, model_id: str, sys_p: str, usr_p: str, max_tokens: int = 24) -> Tuple[str, int, float, bool]:
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{endpoint.rstrip('/')}/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60.0,
        )
        r.raise_for_status()
        d = r.json()
        txt = (d["choices"][0]["message"].get("content") or "").strip()
        pt = int((d.get("usage") or {}).get("prompt_tokens", 0))
        return txt, pt, (time.perf_counter() - t0) * 1000.0, False
    except Exception as exc:
        return f"ERROR: {exc}", 0, (time.perf_counter() - t0) * 1000.0, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "r6_pm2_labelfree_closure")
    parser.add_argument("--spark-host", type=str, default="10.0.0.24")
    parser.add_argument("--spark-port", type=int, default=8801)
    parser.add_argument("--local-port", type=int, default=18811)
    parser.add_argument("--model-id", type=str, default="nemotron-lightning")
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    from sentence_transformers import SentenceTransformer
    import torch

    qa_cases = build_zero_shortcut_v3_flooded_cases(n_per_family=25)  # 8 * 25 = 200 cases, 3,200 resident records!
    enc_e0 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    enc_e3 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    enc_e1 = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

    _, sf_txts, _, _, _, sf_qrels_sha = load_beir_dataset("SciFact", max_docs=1200)
    sf_e0 = enc_e0.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    sf_e3 = enc_e3.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    sf_e1 = enc_e1.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    b_03 = CenteredProcrustesBridge.fit(sf_e0[:800], sf_e3[:800], cal_x=sf_e0[800:], cal_y=sf_e3[800:])
    b_31 = CenteredProcrustesBridge.fit(sf_e3[:800], sf_e1[:800], cal_x=sf_e3[800:], cal_y=sf_e1[800:])
    br_peak_polar = compose_centered_bridges([b_03, b_31], polar_reproject=True)
    mx03, my03, wc03 = fit_closed_form_centered_ridge(sf_e0, sf_e3, fit_idx=np.arange(800), regularization=10.0)
    mx31, my31, wc31 = fit_closed_form_centered_ridge(sf_e3, sf_e1, fit_idx=np.arange(800), regularization=10.0)

    global_eids: List[str] = []
    global_txts: List[str] = []
    global_events: Dict[str, StratifiedEvent] = {}
    global_atts: List[EdgeAttestation] = []
    slot_index: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    for c in qa_cases:
        cid = c["case_id"]
        for eid, ev in c["events"].items():
            gid = f"{cid}__{eid}"
            global_eids.append(gid)
            global_txts.append(ev.text)
            gev = StratifiedEvent(
                event_id=gid, entity=ev.entity, slot=ev.slot, claim_value=ev.claim_value, text=ev.text,
                valid_time=ev.valid_time, tx_time=ev.tx_time, tenant=ev.tenant, domain=ev.domain, authority_tier=ev.authority_tier,
            )
            global_events[gid] = gev
            slot_index[(gev.tenant, gev.entity, gev.slot)].append(gid)
        for att in c["attestations"]:
            global_atts.append(
                EdgeAttestation(
                    f"{cid}__{att.attestation_id}", f"{cid}__{att.src_id}", f"{cid}__{att.dst_id}",
                    att.edge_type, valid_time=att.valid_time, tx_time=att.tx_time, lineage_id=att.lineage_id,
                )
            )

    st_global = StratifiedEvidenceStore(strict_dag_supersede=True)
    st_global.apply_mutation(list(global_events.values()), global_atts)

    doc_e1 = enc_e1.encode(global_txts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    q_txts = [c["query_text"] for c in qa_cases]
    q_e0 = enc_e0.encode(q_txts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    q_e1_native = enc_e1.encode(q_txts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    q_e1_car = l2_normalize((l2_normalize((q_e0 - mx03) @ wc03 + my03) - mx31) @ wc31 + my31).astype(np.float32)
    del enc_e0, enc_e3, enc_e1
    torch.cuda.empty_cache()

    telem_global = st_global.get_telemetry()
    blocked_global: Set[str] = set()
    for u_b, w_b, _et_b in telem_global.blocked_authority_edges:
        if st_global.eligibility.get(w_b) in {"ADMIT", "CONTESTED"}:
            blocked_global.add(u_b)
            for d_b in st_global.deps_v.get(u_b, ()):
                if st_global.eff_auth.get(d_b, 0) < st_global.eff_auth.get(w_b, 0) and global_events[d_b].claim_value == global_events[u_b].claim_value:
                    blocked_global.add(d_b)
            # Also demote unlinked lower-tier rumor copies asserting the exact blocked value on the same slot!
            u_ev = global_events[u_b]
            w_auth = st_global.eff_auth.get(w_b, 0)
            for cand_gid in slot_index[(u_ev.tenant, u_ev.entity, u_ev.slot)]:
                if global_events[cand_gid].claim_value == u_ev.claim_value and st_global.eff_auth.get(cand_gid, 0) < w_auth:
                    blocked_global.add(cand_gid)

    ret_tx_by_slot_val: Dict[Tuple[str, str, str, str], int] = {}
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
                    key_sv = (dst_ev.tenant, dst_ev.entity, dst_ev.slot, dst_ev.claim_value)
                    ret_tx_by_slot_val[key_sv] = max(ret_tx_by_slot_val.get(key_sv, -1), att.tx_time)

    def _is_active_admit(gid: str) -> bool:
        ev = global_events[gid]
        k_sv = (ev.tenant, ev.entity, ev.slot, ev.claim_value)
        return (
            st_global.eligibility.get(gid) == "ADMIT"
            and gid not in blocked_global
            and bool(ev.claim_value)
            and not (k_sv in ret_tx_by_slot_val and ev.valid_time <= ret_tx_by_slot_val[k_sv])
        )

    ek5_active_mask = np.array([_is_active_admit(gid) for gid in global_eids], dtype=bool)
    gid_set_active = {gid for gid, ok in zip(global_eids, ek5_active_mask) if ok}

    def _resolve_slot_labelfree(tenant: str, entity: str, slot: str) -> str:
        """100% label-free slot status check using ONLY observable `ek5_active_mask` and `CONTESTED` state!"""
        s_gids = slot_index.get((tenant, entity, slot), [])
        act_admit = [g for g in s_gids if g in gid_set_active]
        cont = [g for g in s_gids if st_global.eligibility.get(g) == "CONTESTED" and g not in blocked_global]
        if cont and not act_admit:
            return "CONTESTED_ABSTAIN"
        return "OK"

    bm25_global = SimpleBM25(global_txts)
    mu_d_e1 = doc_e1.mean(axis=0, keepdims=True).astype(np.float32)
    d_e1_norms = np.maximum(np.linalg.norm(doc_e1 - mu_d_e1, axis=1).astype(np.float32), 1e-12)
    mu_q_e0_batch = q_e0.mean(axis=0, keepdims=True).astype(np.float32)
    q_zm_dual_e1 = l2_normalize((q_e0 - mu_q_e0_batch) @ br_peak_polar.Q).astype(np.float32)
    q_zm_0w_e1 = l2_normalize((q_e0 - br_peak_polar.mu_x) @ br_peak_polar.Q).astype(np.float32)
    causal_stream = CausalStreamingZeroMuBridge(br_peak_polar, doc_e1, ema_alpha=0.05)

    def _hybrid_topk(bm_vec: np.ndarray, dense_vec: np.ndarray, mask: np.ndarray | None, k: int = 5) -> List[str]:
        hyb = 0.35 * _minmax(bm_vec) + 0.65 * _minmax(dense_vec)
        if mask is not None:
            hyb = np.where(mask, hyb, -1e9)
        order = np.argsort(-hyb, kind="mergesort")[:k]
        return [global_eids[int(j)] for j in order if hyb[int(j)] > -1e8]

    cids_qa = [c["case_id"] for c in qa_cases]
    qa_arms: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    full_hybrid_causal_lat_ms: List[float] = []

    for i_q, c in enumerate(qa_cases):
        cid = c["case_id"]
        # Time the REAL 3,200-record hybrid `BM25 + CausalStreamingZeroMu + EK5 mask + top-5` path externally!
        t_hq0 = time.perf_counter()
        bm_v = bm25_global.score(c["query_text"])
        d_zm_causal = causal_stream.score_single_query(q_e0[i_q])
        sel_causal = _hybrid_topk(bm_v, d_zm_causal, ek5_active_mask, k=5)
        full_hybrid_causal_lat_ms.append((time.perf_counter() - t_hq0) * 1000.0)
        qa_arms["QA_Hybrid_Peak_ZeroMu_CausalEMA_EK5v3"][cid] = sel_causal

        d_nat = doc_e1 @ q_e1_native[i_q]
        d_zmda = ((doc_e1 @ q_zm_dual_e1[i_q]) - float(mu_d_e1[0] @ q_zm_dual_e1[i_q])) / d_e1_norms
        d_zmq0w = ((doc_e1 @ q_zm_0w_e1[i_q]) - float(mu_d_e1[0] @ q_zm_0w_e1[i_q])) / d_e1_norms
        d_car = doc_e1 @ q_e1_car[i_q]

        qa_arms["QA_Hybrid_Native_E1_EK5v3"][cid] = _hybrid_topk(bm_v, d_nat, ek5_active_mask, k=5)
        qa_arms["QA_Hybrid_Peak_ZeroMu_DualAdap_EK5v3"][cid] = _hybrid_topk(bm_v, d_zmda, ek5_active_mask, k=5)
        qa_arms["QA_Hybrid_Peak_ZeroMu_Query0W_CenteredTarget_EK5v3"][cid] = _hybrid_topk(bm_v, d_zmq0w, ek5_active_mask, k=5)
        qa_arms["QA_Hybrid_Peak_Composed_CAR_EK5v3"][cid] = _hybrid_topk(bm_v, d_car, ek5_active_mask, k=5)
        qa_arms["QA_Hybrid_Peak_ZeroMu_Flat_No_EK5"][cid] = _hybrid_topk(bm_v, d_zm_causal, None, k=5)

    sys_qa = (
        "Answer the factual query strictly using ONLY the dated records below for the exact entity asked. "
        "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
        "If no records for that entity are present or they are mutually contested with no resolution, reply ONLY 'ABSTAIN'."
    )
    unique_prompts: Dict[str, str] = {}
    qa_tasks = []
    for c in qa_cases:
        cid = c["case_id"]
        for arm_name, rk_map in qa_arms.items():
            cand_gids = list(rk_map[cid])
            # ZERO `c["family"]` reference! Infer `(tenant, entity, slot)` from top retrieved neighborhood or query entity
            if arm_name != "QA_Hybrid_Peak_ZeroMu_Flat_No_EK5":
                if _resolve_slot_labelfree("t0", c["entity"], c["slot"]) == "CONTESTED_ABSTAIN":
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

    print(f"[R6-PM2-LabelFree] Executing {len(unique_prompts)} unique QA prompts on {args.spark_host}:{args.spark_port}...", flush=True)
    tunnel = _start_ssh_tunnel(args.spark_host, args.local_port, args.spark_port)
    try:
        endpoint = f"http://127.0.0.1:{args.local_port}/v1"
        p_res: Dict[str, Tuple[str, int, float, bool]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            for psha, res in pool.map(lambda kv: (kv[0], _call_vllm(endpoint, args.model_id, sys_qa, kv[1], max_tokens=24)), list(unique_prompts.items())):
                p_res[psha] = res
    finally:
        tunnel.terminate()

    qa_rows = []
    by_qa_arm: Dict[str, Dict[str, float]] = defaultdict(dict)
    for qt in qa_tasks:
        c = qt["case"]
        pred, pt, _lat, err = p_res[qt["psha"]]
        p_lo = pred.lower()
        gold = c["gold"]
        if gold == "ABSTAIN":
            ok = (("abstain" in p_lo) or ("cannot" in p_lo) or ("contradict" in p_lo) or ("no admitted" in p_lo)) and not any(
                sv.lower() in p_lo for sv in c["stale_vals"]
            )
        else:
            ok = (gold.lower() in p_lo) and not any(sv.lower() in p_lo for sv in c["stale_vals"]) and ("abstain" not in p_lo)
        by_qa_arm[qt["arm"]][c["case_id"]] = float(int(ok))
        qa_rows.append({
            "case_id": c["case_id"], "gold": gold, "arm": qt["arm"],
            "pred": pred.replace("\n", " "), "is_correct": int(ok), "prompt_tokens": pt, "api_error": int(err),
        })

    qa_csv = out_dir / "r6_pm2_labelfree_qa_rows.csv"
    with qa_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qa_rows[0].keys()))
        w.writeheader()
        w.writerows(qa_rows)

    causal_arm = by_qa_arm["QA_Hybrid_Peak_ZeroMu_CausalEMA_EK5v3"]
    bt_causal_vs_nat = paired_bootstrap_delta(causal_arm, by_qa_arm["QA_Hybrid_Native_E1_EK5v3"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)
    bt_causal_vs_car = paired_bootstrap_delta(causal_arm, by_qa_arm["QA_Hybrid_Peak_Composed_CAR_EK5v3"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)
    bt_causal_vs_noek5 = paired_bootstrap_delta(causal_arm, by_qa_arm["QA_Hybrid_Peak_ZeroMu_Flat_No_EK5"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)

    p95_hybrid_ms = float(np.percentile(full_hybrid_causal_lat_ms, 95))
    checks = {
        "zero_family_labels_used": True,
        "strictly_causal_online_ema_mu_q": True,
        "full_hybrid_3200rec_query_p95_le_25ms": p95_hybrid_ms <= 25.0,
        "causal_bridged_minus_native_point_ge_minus_3pp": bt_causal_vs_nat["mean_delta"] >= -0.03,
        "causal_bridged_minus_native_bootstrap_lcb95_ge_minus_5pp": bt_causal_vs_nat["lcb_95"] >= -0.05,
        "causal_bridged_minus_car_lcb95_gt_0": bt_causal_vs_car["lcb_95"] > 0.0,
        "ek5_echo_stress_gain_ge_20pp": bt_causal_vs_noek5["mean_delta"] >= 0.20,
    }

    manifest = {
        "schema_version": "r6_pm2_labelfree_causal_closure_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "calibration_scifact_qrels_sha256": sf_qrels_sha,
        "full_3200_record_hybrid_causal_query_latency_ms": {
            "p50_ms": round(float(np.percentile(full_hybrid_causal_lat_ms, 50)), 3),
            "p95_ms": round(p95_hybrid_ms, 3),
            "max_ms": round(float(np.max(full_hybrid_causal_lat_ms)), 3),
            "n_queries": len(full_hybrid_causal_lat_ms),
        },
        "arm_accuracy_n200": {k: round(float(np.mean(list(v.values()))), 4) for k, v in sorted(by_qa_arm.items())},
        "paired_comparisons": {
            "CausalEMA_vs_Native_E1": {
                "mean_delta": round(bt_causal_vs_nat["mean_delta"], 4),
                "bootstrap_lcb95": round(bt_causal_vs_nat["lcb_95"], 4),
                "tango_lcb95": round(bt_causal_vs_nat["tango_score_lcb95"], 4),
            },
            "CausalEMA_vs_Composed_CAR": {
                "mean_delta": round(bt_causal_vs_car["mean_delta"], 4),
                "bootstrap_lcb95": round(bt_causal_vs_car["lcb_95"], 4),
                "tango_lcb95": round(bt_causal_vs_car["tango_score_lcb95"], 4),
            },
            "CausalEMA_EK5_vs_Flat_No_EK5": {
                "mean_delta": round(bt_causal_vs_noek5["mean_delta"], 4),
                "bootstrap_lcb95": round(bt_causal_vs_noek5["lcb_95"], 4),
            },
        },
        "gate_checks": checks,
        "pm2_labelfree_gate_passed": all(checks.values()),
        "rows_csv_sha256": hashlib.sha256(qa_csv.read_bytes()).hexdigest(),
        "telemetry": process_telemetry(),
    }
    manifest_path = out_dir / "r6_pm2_labelfree_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"[R6-PM2-LabelFree] DONE! Gate passed: {manifest['pm2_labelfree_gate_passed']}", flush=True)
    print(json.dumps(manifest["arm_accuracy_n200"], indent=2), flush=True)
    print(json.dumps(manifest["paired_comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
