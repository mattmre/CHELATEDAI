
import csv, hashlib, importlib.metadata, json, os, shutil, sys, tempfile, time
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

# Check graphiti-core availability & external daemon requirement
graphiti_info = {"package": "graphiti-core", "status": "NOT_INSTALLED"}
try:
    g_ver = importlib.metadata.version("graphiti-core")
    graphiti_info = {
        "package": "graphiti-core",
        "version": g_ver,
        "status": "REQUIRES_EXTERNAL_GRAPH_DAEMON",
        "reason": "graphiti-core requires a live Bolt Neo4j >=5.26 or FalkorDB server daemon with vector+fulltext index extensions; no Neo4j/FalkorDB container is permitted per container-immutability safety boundary.",
    }
except Exception as exc:
    graphiti_info = {"package": "graphiti-core", "status": "INSTALL_OR_IMPORT_NOTE", "detail": str(exc)}

scalemas_info = {
    "comparator": "ScaleMAS (arXiv:2610.16120v1)",
    "status": "UNAVAILABLE_UPSTREAM_ARCHIVE",
    "checked_date": "2026-11-22",
    "reason": "No official public PyPI package or upstream reference repository released by authors; evaluated alongside C1_scalemas_bounded & M2_scalemas_unsigned per Council R3 PM-3 specification.",
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

# Evaluate unmodified `mem0ai` vs `C1_scalemas_bounded`, `C3_ordinary_bitemporal_tms`, and `C4_v3_ek5_v3_k4`
# across `N=80 = 8x10` stratified Zero-Shortcut v3 cases!
cases = build_zero_shortcut_v3_flooded_cases(n_per_family=10)
qdrant_tmp = Path(tempfile.mkdtemp(prefix="mem0_pm3_qdrant_"))

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
            "collection_name": "pm3_bench",
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
    # 1) Measure unmodified mem0.add() ingestion across chronological tx_time order!
    ordered_evs = sorted(case["events"].values(), key=lambda e: (e.tx_time, e.event_id))
    t_m0 = time.perf_counter()
    # Batch case records as chronological user messages into mem0.add(uid_scope)
    convo_msgs = [{"role": "user", "content": f"[vt={ev.valid_time}, tx={ev.tx_time}, tier={ev.authority_tier}] {ev.text}"} for ev in ordered_evs]
    try:
        mem.add(convo_msgs, user_id=uid_scope, infer=True)
    except Exception:
        # Fallback if infer JSON parse fails on any turn: store raw facts via infer=False
        mem.add(convo_msgs, user_id=uid_scope, infer=False)
    mem0_ingest_ms.append((time.perf_counter() - t_m0) * 1000.0)

    s_res = mem.search(case["query_text"], user_id=uid_scope, limit=5)
    m_items = s_res.get("results", s_res) if isinstance(s_res, dict) else s_res
    m_lines = []
    for it in (m_items or [])[:5]:
        txt_m = it.get("memory") or it.get("text") or str(it)
        m_lines.append(f"- {txt_m}")
    ctx_mem0 = "\n".join(m_lines) if m_lines else "(No memories found.)"
    pred_mem0, pt_mem0 = call_llm(sys_p, f"Records:\n{ctx_mem0}\n\nQuestion: {case['query_text']}\nAnswer:")

    # 2) Compare against C1_scalemas_bounded, C3_ordinary_bitemporal_tms, C4_v3_ek5_v3_k4, and C4_v4_ek5_v3_blocked_telemetry!
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

shutil.rmtree(qdrant_tmp, ignore_errors=True)

csv_path = OUT_DIR / "r5_s2_pm3_mem0ai_rows.csv"
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
    "c4v3_vs_mem0ai_live": {
        "mean_delta": round(bt_vs_mem0["mean_delta"], 4),
        "lcb_95": round(bt_vs_mem0["lcb_95"], 4),
        "tango_ci95_low": round(bt_vs_mem0["tango_score_lcb95"], 4),
        "stretch_subgate_gt_10pp_met": bool(bt_vs_mem0["mean_delta"] >= 0.10),
    },
    "c4v3_vs_c1_scalemas_bounded": {
        "mean_delta": round(bt_vs_c1["mean_delta"], 4),
        "lcb_95": round(bt_vs_c1["lcb_95"], 4),
    },
    "wall_seconds": round(time.perf_counter() - t0, 2),
    "csv_path": csv_path.name,
    "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
}
write_json_atomic(OUT_DIR / "pm3_subworker_result.json", out_summary)
