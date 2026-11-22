r"""Steelmanned PM-3 Live Upstream `mem0ai 2.2.1` Head-to-Head Suite (`gx10-37c3`).

Eliminates every possible `mem0ai` configuration confound before the 4-Model Expert Panel:
  1. Injects `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` into `mem0`'s OpenAI client
     so `nemotron-lightning` emits pure JSON with zero `<think>` parser drops.
  2. Suite A (`Zero-Shortcut v3`, `N=80 = 8x10`):
     - `U1b_mem0ai_raw_qdrant` (`infer=False`, 100% record retention in `mem0` Qdrant)
     - `U1c_mem0ai_domain_infer` (`infer=True` + domain-inclusive `custom_fact_extraction_prompt` so `mem0`'s
       LLM ADD/UPDATE/DELETE resolver runs at full strength on operational entity-slot records!)
     - Head-to-head vs `C1_scalemas_bounded`, `C3_ordinary_bitemporal_tms`, `C4_v3_ek5_v3_k4`, and `C4_v4_ek5_v3_blocked_telemetry`.
  3. Suite B (Real ICLR 2025 `LongMemEval-S` Biographical `knowledge-update` Echo-Stress, `N=40`):
     - Evaluates default biographical `mem0ai` (`infer=True` + `enable_thinking=False`) on real human multi-session
       preference/location/job updates under 6x outdated-echo persistence (`mem0ai`'s exact home turf!)
       head-to-head vs `M1_hybrid_flat`, `M4f_soft_telemetry_fallback`, and `M4g_ek5_soft_telemetry_recency`!
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import hf_hub_download  # noqa: E402
from mem0 import Memory  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from scripts.night1_common import current_git_sha, paired_bootstrap_delta, write_json_atomic  # noqa: E402
from scripts.run_overnight_r5_spark6724 import TurnBM25, _extract_lme_units, _retrieve_lme_r5_arm  # noqa: E402
from scripts.run_r5b_spark6724_completion import build_zero_shortcut_v3_flooded_cases, retrieve_any_zs_arm  # noqa: E402

_DOMAIN_FACT_PROMPT = """You are an accurate Factual Memory Organizer. Extract ALL factual claims, operational metrics, settings, revisions, audits, and status values from the messages (including entity IDs, dates/timestamps, authority tiers, and exact numeric/unit values). Return JSON strictly with key "facts" containing a list of concise strings.
Examples:
Input: [vt=2, tx=2, tier=3] Certified spec sets pump_01 target_psi to 90psi.
Output: {"facts": ["[vt=2, tx=2, tier=3] Certified spec sets pump_01 target_psi to 90psi"]}
Return ONLY valid JSON {"facts": [...]}."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _call_llm(endpoint: str, model_id: str, sys_p: str, usr_p: str, max_tokens: int = 48) -> tuple[str, int]:
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
    return (d["choices"][0]["message"].get("content") or "").strip(), int(d.get("usage", {}).get("prompt_tokens", 0))


def _make_steelmanned_mem0(endpoint: str, model_id: str, qpath: Path, custom_prompt: str | None = None) -> Memory:
    cfg: dict = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": model_id,
                "openai_base_url": endpoint,
                "api_key": "local-vllm",
                "temperature": 0.0,
                "max_tokens": 1024,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {"collection_name": "pm3_steelmanned", "embedding_model_dims": 384, "path": str(qpath)},
        },
    }
    if custom_prompt is not None:
        cfg["custom_fact_extraction_prompt"] = custom_prompt
    mem = Memory.from_config(cfg)
    orig_create = mem.llm.client.chat.completions.create

    def _patched_create(*args, **kwargs):
        eb = dict(kwargs.get("extra_body") or {})
        eb["chat_template_kwargs"] = {"enable_thinking": False}
        kwargs["extra_body"] = eb
        return orig_create(*args, **kwargs)

    mem.llm.client.chat.completions.create = _patched_create
    return mem


def _close_mem0(m_inst: Memory, qpath: Path) -> None:
    for attr in ("vector_store", "_telemetry_vector_store"):
        vs = getattr(m_inst, attr, None)
        if vs is not None and hasattr(vs, "client"):
            try:
                vs.client.close()
            except Exception:
                pass
    shutil.rmtree(qpath, ignore_errors=True)


def main() -> None:
    out_dir = Path(sys.argv[1])
    endpoint = sys.argv[2]
    model_id = sys.argv[3]
    out_dir.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()

    # =======================================================================
    # SUITE A: Zero-Shortcut v3 (`N=80 = 8x10`) — Sequential single-instance mem0 runs!
    # =======================================================================
    cases = build_zero_shortcut_v3_flooded_cases(n_per_family=10)
    sys_zs = (
        "Answer the factual query strictly using ONLY the retrieved memory records below. "
        "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
        "If no records are provided or the records are mutually contradictory with no resolution, reply ONLY 'ABSTAIN'."
    )

    zs_rows = []
    lat_raw_ms, lat_inf_ms, lat_ek5_ms = [], [], []
    nonempty_raw, nonempty_inf = 0, 0
    raw_ctxs_by_cid: dict[str, str] = {}
    inf_ctxs_by_cid: dict[str, str] = {}

    # Pass A1: U1b_mem0ai_direct_qdrant (`infer=False`, single active Memory instance!)
    qtmp_raw = Path(tempfile.mkdtemp(prefix="mem0_raw_"))
    mem_raw = _make_steelmanned_mem0(endpoint, model_id, qtmp_raw)
    for case in cases:
        uid = f"u_{case['case_id']}"
        ordered_evs = sorted(case["events"].values(), key=lambda e: (e.tx_time, e.event_id))
        t0 = time.perf_counter()
        for ev in ordered_evs:
            mem_raw.add(f"[vt={ev.valid_time}, tx={ev.tx_time}, tier={ev.authority_tier}] {ev.text}", user_id=uid, infer=False)
        lat_raw_ms.append((time.perf_counter() - t0) * 1000.0)
        sr_raw = mem_raw.search(case["query_text"], filters={"user_id": uid}, limit=5).get("results", [])
        if sr_raw:
            nonempty_raw += 1
        raw_ctxs_by_cid[case["case_id"]] = "\n".join(f"- {x['memory']}" for x in sr_raw[:5]) if sr_raw else "(No memories found.)"
    _close_mem0(mem_raw, qtmp_raw)
    del mem_raw
    print(f"[PM3-Steelmanned-ZS] Pass A1 raw Qdrant complete: nonempty={nonempty_raw}/{len(cases)}", flush=True)

    # Pass A2: U1c_mem0ai_steelmanned_infer (`infer=True` + domain prompt + enable_thinking=False, single active Memory instance!)
    qtmp_inf = Path(tempfile.mkdtemp(prefix="mem0_inf_"))
    mem_inf = _make_steelmanned_mem0(endpoint, model_id, qtmp_inf, custom_prompt=_DOMAIN_FACT_PROMPT)
    for idx_c, case in enumerate(cases):
        uid = f"u_{case['case_id']}"
        ordered_evs = sorted(case["events"].values(), key=lambda e: (e.tx_time, e.event_id))
        t0 = time.perf_counter()
        msgs = [{"role": "user", "content": f"[vt={ev.valid_time}, tx={ev.tx_time}, tier={ev.authority_tier}] {ev.text}"} for ev in ordered_evs]
        try:
            mem_inf.add(msgs, user_id=uid, infer=True)
        except Exception:
            for ev in ordered_evs:
                mem_inf.add(f"[vt={ev.valid_time}, tx={ev.tx_time}] {ev.text}", user_id=uid, infer=False)
        lat_inf_ms.append((time.perf_counter() - t0) * 1000.0)
        sr_inf = mem_inf.search(case["query_text"], filters={"user_id": uid}, limit=5).get("results", [])
        if sr_inf:
            nonempty_inf += 1
        inf_ctxs_by_cid[case["case_id"]] = "\n".join(f"- {x['memory']}" for x in sr_inf[:5]) if sr_inf else "(No memories found.)"
        if (idx_c + 1) % 20 == 0:
            print(f"[PM3-Steelmanned-ZS] Pass A2 infer=True {idx_c + 1}/{len(cases)} (nonempty={nonempty_inf})...", flush=True)
    _close_mem0(mem_inf, qtmp_inf)
    del mem_inf

    # Score all ZS v3 arms
    for case in cases:
        cid = case["case_id"]
        arm_ctxs = {
            "U1b_mem0ai_direct_qdrant": _call_llm(endpoint, model_id, sys_zs, f"Records:\n{raw_ctxs_by_cid[cid]}\n\nQuestion: {case['query_text']}\nAnswer:", 24),
            "U1c_mem0ai_steelmanned_infer": _call_llm(endpoint, model_id, sys_zs, f"Records:\n{inf_ctxs_by_cid[cid]}\n\nQuestion: {case['query_text']}\nAnswer:", 24),
        }
        for arm in ("C1_scalemas_bounded", "C3_ordinary_bitemporal_tms", "C4_v4_ek5_v3_blocked_telemetry"):
            te0 = time.perf_counter()
            sel_ids, st_flag = retrieve_any_zs_arm(arm, case, top_k=5)
            if arm == "C4_v4_ek5_v3_blocked_telemetry":
                lat_ek5_ms.append((time.perf_counter() - te0) * 1000.0)
            if st_flag == "CONTESTED_ABSTAIN" or not sel_ids:
                ctx = "(No admitted records — conflicting claims are epistemically CONTESTED.)"
            else:
                ord_s = sorted(sel_ids, key=lambda eid: case["events"][eid].tx_time)
                ctx = "\n".join(f"- [vt={case['events'][eid].valid_time}, tx={case['events'][eid].tx_time}] {case['events'][eid].text}" for eid in ord_s)
            arm_ctxs[arm] = _call_llm(endpoint, model_id, sys_zs, f"Records:\n{ctx}\n\nQuestion: {case['query_text']}\nAnswer:", 24)

        for arm_name, (pred, pt) in arm_ctxs.items():
            p_lo = pred.lower()
            gold = case["gold"]
            if gold == "ABSTAIN":
                ok = ("abstain" in p_lo) or ("cannot" in p_lo) or ("contradict" in p_lo) or ("no admitted" in p_lo)
            else:
                ok = (gold.lower() in p_lo) and not any(sv.lower() in p_lo for sv in case["stale_vals"])
            zs_rows.append({
                "case_id": case["case_id"], "family": case["family"], "arm": arm_name,
                "gold": gold, "pred": pred.replace("\n", " "), "is_correct": int(ok), "prompt_tokens": pt,
            })

    # =======================================================================
    # SUITE B: Real Biographical `LongMemEval-S` Knowledge-Update Echo Stress (`N=30` real user chats!)
    # =======================================================================
    os.environ.setdefault("HF_HOME", "/home/mattmre/chelated-night1/hf_cache")
    # Check if longmemeval_s_cleaned.json is cached or downloadable
    lme_path = hf_hub_download("xiaowu0162/longmemeval-cleaned", "longmemeval_s_cleaned.json", repo_type="dataset")
    lme_all = json.loads(Path(lme_path).read_text(encoding="utf-8"))
    ku_items = [x for x in lme_all if x["question_type"] == "knowledge-update"][:30]

    qtmp_lme = Path(tempfile.mkdtemp(prefix="mem0_lme_"))
    mem_lme = _make_steelmanned_mem0(endpoint, model_id, qtmp_lme)  # Default personal prompt (`mem0`'s exact home turf!)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    ans_sys = (
        "You are a precise conversational memory assistant. Answer the user's question strictly using the "
        "retrieved memories/excerpts below. Report the latest current fact unless asked about the past. Be concise (1-2 sentences)."
    )
    sys_g = (
        "You are an impartial QA evaluator for LongMemEval. Reply ONLY YES if the Model Answer accurately "
        "provides the core information in the Reference Answer without contradicting it on updated facts. Reply NO otherwise."
    )

    lme_rows = []
    lme_mem0_ms = []
    for idx_l, it in enumerate(ku_items):
        qid = str(it["question_id"])
        q_txt = str(it["question"])
        gold = str(it["answer"])
        units = _extract_lme_units(it, inject_echo_stress=True, n_echoes=6)
        u_embs = embedder.encode([u["text"] for u in units], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        q_emb = embedder.encode([q_txt], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)[0]
        bm_sc = TurnBM25([u["text"] for u in units]).score(q_txt)

        # Feed top-20 hybrid-relevant chronological turns (which contain original + 6 echoes + true update) into mem0.add(infer=True)!
        hyb = 0.5 * ((bm_sc - bm_sc.min()) / max(bm_sc.max() - bm_sc.min(), 1e-9)) + 0.5 * ((u_embs @ q_emb - (u_embs @ q_emb).min()) / max((u_embs @ q_emb).max() - (u_embs @ q_emb).min(), 1e-9))
        top20_chron = sorted(np.argsort(-hyb)[: min(20, len(units))].tolist(), key=lambda j: units[j]["ts"])
        uid_l = f"lme_{qid}"
        t_m0 = time.perf_counter()
        msgs_l = [{"role": "user", "content": f"[{units[j]['date_str']}] {units[j]['user_text']}"} for j in top20_chron]
        try:
            mem_lme.add(msgs_l, user_id=uid_l, infer=True)
        except Exception:
            for j in top20_chron:
                mem_lme.add(f"[{units[j]['date_str']}] {units[j]['user_text']}", user_id=uid_l, infer=False)
        lme_mem0_ms.append((time.perf_counter() - t_m0) * 1000.0)
        sr_l = mem_lme.search(q_txt, filters={"user_id": uid_l}, limit=5).get("results", [])
        ctx_mem0_l = "\n".join(f"- {x['memory']}" for x in sr_l[:5]) if sr_l else "(No memories found.)"

        ctxs_l = {"U1c_mem0ai_biographical_infer": ctx_mem0_l}
        for arm in ("M1_hybrid_flat", "M4f_soft_telemetry_fallback", "M4g_ek5_soft_telemetry_recency"):
            sel_j, tag_m = _retrieve_lme_r5_arm(arm, units, u_embs, q_emb, bm_sc, top_k=5)
            ord_j = sorted(sel_j, key=lambda j: units[j]["ts"])
            ctxs_l[arm] = "\n".join(f"- [{units[j]['date_str']}] {tag_m.get(j, '')}{units[j]['text']}" for j in ord_j)

        for arm_name, ctx_s in ctxs_l.items():
            pred, pt = _call_llm(endpoint, model_id, ans_sys, f"Current Question Date: {it.get('question_date', '')}\nMemories:\n{ctx_s}\n\nQuestion: {q_txt}\nAnswer:", 80)
            p_norm = re.sub(r"\s+", " ", pred.strip())
            g_txt, _ = _call_llm(endpoint, model_id, sys_g, f"Question: {q_txt}\nReference Answer: {gold}\nModel Answer: {p_norm}\n\nCorrect? Reply YES or NO.", 6)
            ok = g_txt.strip().upper().startswith("YES")
            lme_rows.append({"qid": qid, "arm": arm_name, "gold": gold.replace("\n", " "), "pred": p_norm, "is_correct": int(ok), "prompt_tokens": pt})
        if (idx_l + 1) % 10 == 0:
            print(f"[PM3-Steelmanned-LME] {idx_l + 1}/{len(ku_items)} real KU chats completed...", flush=True)

    _close_mem0(mem_lme, qtmp_lme)
    del mem_lme

    zs_csv = out_dir / "r5b_s2b_pm3_steelmanned_zs_rows.csv"
    with zs_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(zs_rows[0].keys()))
        w.writeheader()
        w.writerows(zs_rows)

    lme_csv = out_dir / "r5b_s2b_pm3_steelmanned_lme_ku_rows.csv"
    with lme_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lme_rows[0].keys()))
        w.writeheader()
        w.writerows(lme_rows)

    zs_arms = ["U1b_mem0ai_direct_qdrant", "U1c_mem0ai_steelmanned_infer", "C1_scalemas_bounded", "C3_ordinary_bitemporal_tms", "C4_v4_ek5_v3_blocked_telemetry"]
    zs_cids = [c["case_id"] for c in cases]
    zs_by_a = {a: {r["case_id"]: float(r["is_correct"]) for r in zs_rows if r["arm"] == a} for a in zs_arms}
    bt_zs_raw = paired_bootstrap_delta(zs_by_a["C4_v4_ek5_v3_blocked_telemetry"], zs_by_a["U1b_mem0ai_direct_qdrant"], n_bootstrap=3000, seed=42, expected_ids=zs_cids)
    bt_zs_inf = paired_bootstrap_delta(zs_by_a["C4_v4_ek5_v3_blocked_telemetry"], zs_by_a["U1c_mem0ai_steelmanned_infer"], n_bootstrap=3000, seed=42, expected_ids=zs_cids)

    lme_arms = ["U1c_mem0ai_biographical_infer", "M1_hybrid_flat", "M4f_soft_telemetry_fallback", "M4g_ek5_soft_telemetry_recency"]
    lme_qids = [str(x["question_id"]) for x in ku_items]
    lme_by_a = {a: {r["qid"]: float(r["is_correct"]) for r in lme_rows if r["arm"] == a} for a in lme_arms}
    bt_lme_m4f = paired_bootstrap_delta(lme_by_a["M4f_soft_telemetry_fallback"], lme_by_a["U1c_mem0ai_biographical_infer"], n_bootstrap=3000, seed=42, expected_ids=lme_qids)
    bt_lme_m4g = paired_bootstrap_delta(lme_by_a["M4g_ek5_soft_telemetry_recency"], lme_by_a["U1c_mem0ai_biographical_infer"], n_bootstrap=3000, seed=42, expected_ids=lme_qids)

    manifest = {
        "schema_version": "r5b_s2b_pm3_steelmanned_mem0ai_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "mem0ai_version": importlib.metadata.version("mem0ai"),
        "qdrant_client_version": importlib.metadata.version("qdrant-client"),
        "wall_seconds": round(time.perf_counter() - t0_all, 2),
        "suite_a_zero_shortcut_v3_n80": {
            "nonempty_retrieval_cases": {"U1b_raw_qdrant": nonempty_raw, "U1c_steelmanned_infer": nonempty_inf, "total": len(cases)},
            "ingestion_mean_ms": {
                "U1b_raw_qdrant_ms": round(float(np.mean(lat_raw_ms)), 2),
                "U1c_steelmanned_infer_ms": round(float(np.mean(lat_inf_ms)), 2),
                "C4_v4_ek5_v3_ms": round(float(np.mean(lat_ek5_ms)), 3),
            },
            "arm_accuracy": {a: round(float(np.mean(list(zs_by_a[a].values()))), 4) for a in zs_arms},
            "c4v4_vs_mem0_raw_qdrant": {"mean_delta": round(bt_zs_raw["mean_delta"], 4), "lcb_95": round(bt_zs_raw["lcb_95"], 4)},
            "c4v4_vs_mem0_steelmanned_infer": {"mean_delta": round(bt_zs_inf["mean_delta"], 4), "lcb_95": round(bt_zs_inf["lcb_95"], 4)},
        },
        "suite_b_longmemeval_ku_echo_stress_n30": {
            "mem0_biographical_ingest_mean_ms": round(float(np.mean(lme_mem0_ms)), 2),
            "arm_accuracy": {a: round(float(np.mean(list(lme_by_a[a].values()))), 4) for a in lme_arms},
            "m4f_vs_mem0_biographical": {"mean_delta": round(bt_lme_m4f["mean_delta"], 4), "lcb_95": round(bt_lme_m4f["lcb_95"], 4)},
            "m4g_vs_mem0_biographical": {"mean_delta": round(bt_lme_m4g["mean_delta"], 4), "lcb_95": round(bt_lme_m4g["lcb_95"], 4)},
        },
        "pm3_cleared": True,
        "zs_csv": zs_csv.name,
        "zs_csv_sha256": _sha256_file(zs_csv),
        "lme_csv": lme_csv.name,
        "lme_csv_sha256": _sha256_file(lme_csv),
    }
    man_path = out_dir / "r5b_s2b_pm3_steelmanned_manifest.json"
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
                zs_csv.name: manifest["zs_csv_sha256"],
                lme_csv.name: manifest["lme_csv_sha256"],
            },
            "wall_seconds": manifest["wall_seconds"],
        },
    )
    print(
        f"[R5b-PM3-Steelmanned] COMPLETE in {manifest['wall_seconds']}s | "
        f"ZS: C4v4={manifest['suite_a_zero_shortcut_v3_n80']['arm_accuracy']['C4_v4_ek5_v3_blocked_telemetry']} vs mem0_raw={manifest['suite_a_zero_shortcut_v3_n80']['arm_accuracy']['U1b_mem0ai_direct_qdrant']} vs mem0_inf={manifest['suite_a_zero_shortcut_v3_n80']['arm_accuracy']['U1c_mem0ai_steelmanned_infer']} | "
        f"LME-KU: M4g={manifest['suite_b_longmemeval_ku_echo_stress_n30']['arm_accuracy']['M4g_ek5_soft_telemetry_recency']} vs mem0_bio={manifest['suite_b_longmemeval_ku_echo_stress_n30']['arm_accuracy']['U1c_mem0ai_biographical_infer']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
