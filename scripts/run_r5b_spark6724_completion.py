r"""R5b Completion Worker 1 (`gx10-6724`):
  1. `r5b_s1_zero_shortcut_v3_pm1a_closure`: Adds `C4_v4_ek5_v3_blocked_telemetry` (demoting attackers caught in `st.get_telemetry().blocked_authority_edges` + low-tier rumor sources) alongside `C0..C4_v3` on `Zero-Shortcut v3` (`N=240 = 8x30`, 16 events/case > `top_k=5`).
  2. `r5b_s4_pm4_cross_family_2x2`: Loads `Qwen/Qwen2.5-7B-Instruct` directly via `.to("cuda")` (`dtype=torch.bfloat16`, zero `accelerate` requirement!) to run the full `2x2` `{Nemotron-3.5-30B, Qwen2.5-7B} x {Nemotron Judge, Qwen2.5 Judge}` Cohen's kappa evaluation (`PM-4`).

Strictly ADDITIVE: writes ONLY to `--output-root` (`/home/mattmre/chelated-night1/out/20261122-r5b-completion/spark6724`), reading prior Stage 2/3 rows read-only from `--r5-prev-root`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ek5_v3_stratified import StratifiedEvent, StratifiedEvidenceStore  # noqa: E402
from iq04_temporal_evidence import EvidenceEvent, ScopedBM25  # noqa: E402
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    paired_bootstrap_delta,
    process_telemetry,
    write_json_atomic,
)
from scripts.run_overnight_r5_spark6724 import (  # noqa: E402
    _call_vllm,
    _cohens_kappa,
    _content_tokens,
    _jaccard,
    _sha256_file,
    build_zero_shortcut_v3_flooded_cases,
    retrieve_zs_v3_arm,
)


def retrieve_c4_v4_blocked_telemetry(case: Dict[str, Any], *, top_k: int = 5) -> Tuple[List[str], str]:
    """`EK5 v3` + `EM-3` Blocked-Authority Attacker Demotion (`C4_v4_ek5_v3_blocked_telemetry`).

    Uses ONLY observable `StratifiedEvidenceStore` state and `st.get_telemetry().blocked_authority_edges`:
    any node `u` whose outgoing `REFUTE`/`SUPERSEDE` against a higher-tier `ADMIT` target `w` was
    rejected by Stratum 1 (`eff_auth[u] < eff_auth[w]`), together with any low-tier dependency `d`
    in `deps_v[u]` asserting the same rejected value, is demoted from current `ADMIT` candidates.
    """
    import math

    events: Dict[str, StratifiedEvent] = case["events"]
    leg_events: Dict[str, EvidenceEvent] = case["legacy_events"]
    eids = sorted(events.keys())
    bm25 = ScopedBM25(leg_events)
    bscores = bm25.score(case["query_text"], eids)

    st = StratifiedEvidenceStore(strict_dag_supersede=True)
    st.apply_mutation(list(events.values()), case["attestations"])
    elig = st.eligibility
    telem = st.get_telemetry()

    blocked_attackers: Set[str] = set()
    for u, w, _et in telem.blocked_authority_edges:
        if elig.get(w) == "ADMIT":
            blocked_attackers.add(u)
            for d in st.deps_v.get(u, ()):
                if st.eff_auth.get(d, 0) < st.eff_auth.get(w, 0) and events[d].claim_value == events[u].claim_value:
                    blocked_attackers.add(d)

    admitted = [
        eid for eid in eids
        if elig.get(eid) == "ADMIT" and events[eid].claim_value and eid not in blocked_attackers
    ]
    contested = [eid for eid in eids if elig.get(eid) == "CONTESTED"]
    if not admitted and contested:
        return [], "CONTESTED_ABSTAIN"

    ret_tx: Dict[str, int] = {}
    for att in case["attestations"]:
        if att.edge_type in ("REFUTE", "SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE"):
            is_act = (
                att.src_id in st.refs.get(att.dst_id, ())
                or att.src_id in st.sup.get(att.dst_id, ())
                or att.src_id in st.lsup.get(att.dst_id, ())
            )
            if is_act and elig.get(att.src_id) == "ADMIT" and elig.get(att.dst_id) == "AUDIT_ONLY":
                rv = events[att.dst_id].claim_value
                if rv:
                    ret_tx[rv] = max(ret_tx.get(rv, -1), att.tx_time)

    active = [
        eid for eid in admitted
        if not (events[eid].claim_value in ret_tx and events[eid].valid_time <= ret_tx[events[eid].claim_value])
    ]
    if not active:
        return [], "CONTESTED_ABSTAIN"

    sup_cnt: Counter[str] = Counter()
    for att in case["attestations"]:
        if att.edge_type == "SUPPORT" and elig.get(att.src_id) == "ADMIT" and att.src_id not in blocked_attackers:
            sup_cnt[att.dst_id] += 1
    active_refs = {
        att.src_id for att in case["attestations"]
        if att.edge_type in ("REFUTE", "SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE")
        and elig.get(att.src_id) == "ADMIT"
        and (
            att.src_id in st.refs.get(att.dst_id, ())
            or att.src_id in st.sup.get(att.dst_id, ())
            or att.src_id in st.lsup.get(att.dst_id, ())
        )
    }
    val_cnt = Counter(events[eid].claim_value for eid in active)

    def _sc(eid: str) -> float:
        return (
            bscores[eid]
            + 2.0 * float(st.eff_auth.get(eid, 0))
            + (3.0 if eid in active_refs else 0.0)
            + 0.8 * math.log1p(sup_cnt[eid])
            + 0.4 * math.log1p(val_cnt[events[eid].claim_value])
        )

    tsets = {eid: _content_tokens(events[eid].text) for eid in active}
    sel: List[str] = []
    for eid in sorted(active, key=lambda x: (-_sc(x), x)):
        if any(_jaccard(tsets[eid], tsets[s]) >= 0.58 for s in sel):
            continue
        sel.append(eid)
        if len(sel) >= top_k:
            break
    return sel, "OK"


def retrieve_any_zs_arm(arm: str, case: Dict[str, Any], *, top_k: int = 5) -> Tuple[List[str], str]:
    if arm == "C4_v4_ek5_v3_blocked_telemetry":
        return retrieve_c4_v4_blocked_telemetry(case, top_k=top_k)
    return retrieve_zs_v3_arm(arm, case, top_k=top_k)


def run_part1_zs_v3_pm1a_closure(endpoint: str, model_id: str, root_dir: Path, max_workers: int = 16) -> Dict[str, Any]:
    out_dir = root_dir / "r5b_s1_zero_shortcut_v3_pm1a_closure"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    cases = build_zero_shortcut_v3_flooded_cases(n_per_family=30)
    arms = [
        "C0_bm25_flat",
        "C1_scalemas_bounded",
        "C2_latest_timestamp",
        "C2b_bm25_jaccard_dedup",
        "C2c_astra_4line_rule",
        "C3_ordinary_bitemporal_tms",
        "C4_v2_ek5_v2",
        "C4_v3_ek5_v3_k4",
        "C4_v4_ek5_v3_blocked_telemetry",
    ]
    sys_p = (
        "Answer the factual query strictly using ONLY the dated records below. "
        "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
        "If no records are provided or the records are mutually contradictory with no resolution, reply ONLY 'ABSTAIN'."
    )
    unique_prompts: Dict[str, str] = {}
    task_meta = []
    for case in cases:
        for arm in arms:
            sel_ids, status_flag = retrieve_any_zs_arm(arm, case, top_k=5)
            if status_flag == "CONTESTED_ABSTAIN" or not sel_ids:
                ctx = "(No admitted records — conflicting claims are epistemically CONTESTED.)"
            else:
                ordered = sorted(sel_ids, key=lambda eid: case["events"][eid].tx_time)
                ctx = "\n".join(
                    f"- [vt={case['events'][eid].valid_time}, tx={case['events'][eid].tx_time}] {case['events'][eid].text}"
                    for eid in ordered
                )
            prompt = f"Records:\n{ctx}\n\nQuestion: {case['query_text']}\nAnswer:"
            psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            unique_prompts[psha] = prompt
            task_meta.append({"case": case, "arm": arm, "psha": psha})

    print(f"[R5b-S1-PM1a] Executing {len(unique_prompts)} unique prompts across {len(arms)} arms...", flush=True)
    p_res: Dict[str, Tuple[str, int, float, bool]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for psha, res in pool.map(lambda kv: (kv[0], _call_vllm(endpoint, model_id, sys_p, kv[1], max_tokens=32)), list(unique_prompts.items())):
            p_res[psha] = res

    rows = []
    for tm in task_meta:
        case = tm["case"]
        pred, pt, lat, err = p_res[tm["psha"]]
        p_lo = pred.lower()
        gold = case["gold"]
        if gold == "ABSTAIN":
            ok = ("abstain" in p_lo) or ("cannot" in p_lo) or ("contradict" in p_lo) or ("no admitted" in p_lo)
        else:
            ok = (gold.lower() in p_lo) and not any(sv.lower() in p_lo for sv in case["stale_vals"])
        rows.append({
            "case_id": case["case_id"], "family": case["family"], "arm": tm["arm"],
            "gold": gold, "pred_raw": pred.replace("\n", " "), "is_correct": int(ok),
            "prompt_tokens": pt, "latency_ms": round(lat, 2), "api_error": int(err), "prompt_sha256": tm["psha"],
        })

    csv_path = out_dir / "r5b_s1_zero_shortcut_v3_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_arm = defaultdict(list)
    by_fam_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
        by_fam_arm[(r["family"], r["arm"])].append(r["is_correct"])
    fams = sorted({c["family"] for c in cases})
    arm_stats = {
        arm: {
            "overall_acc": round(float(np.mean([r["is_correct"] for r in by_arm[arm]])), 4),
            "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in by_arm[arm]])), 1),
            "api_errors": int(sum(r["api_error"] for r in by_arm[arm])),
            "by_family": {f: round(float(np.mean(by_fam_arm[(f, arm)])), 4) for f in fams},
        }
        for arm in arms
    }

    def _imap(a: str) -> Dict[str, float]:
        return {r["case_id"]: float(r["is_correct"]) for r in by_arm[a]}

    c4v4 = _imap("C4_v4_ek5_v3_blocked_telemetry")
    all_cids = [c["case_id"] for c in cases]
    boot = {}
    non_wfs_baselines = ["C0_bm25_flat", "C1_scalemas_bounded", "C2_latest_timestamp", "C2b_bm25_jaccard_dedup", "C2c_astra_4line_rule", "C3_ordinary_bitemporal_tms", "C4_v2_ek5_v2"]
    all_lcb_gt_15pp = True
    for ref in non_wfs_baselines + ["C4_v3_ek5_v3_k4"]:
        bt = paired_bootstrap_delta(c4v4, _imap(ref), n_bootstrap=3000, seed=42, expected_ids=all_cids)
        boot[f"C4v4_vs_{ref}"] = {
            "mean_delta": round(bt["mean_delta"], 4),
            "lcb_95": round(bt["lcb_95"], 4),
            "tango_ci95_low": round(bt["tango_score_lcb95"], 4),
            "clopper_pearson_ci95_low": round(bt["clopper_pearson_lcb95"], 4),
        }
        if ref in non_wfs_baselines and bt["lcb_95"] <= 0.15:
            all_lcb_gt_15pp = False

    pm1a_pass = bool(arm_stats["C4_v4_ek5_v3_blocked_telemetry"]["overall_acc"] >= 0.98 and all_lcb_gt_15pp)
    manifest = {
        "schema_version": "r5b_s1_zero_shortcut_v3_pm1a_closure_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "model_id": model_id,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "n_cases": len(cases),
        "mean_events_per_case": round(float(np.mean([len(c["events"]) for c in cases])), 2),
        "arms": arm_stats,
        "bootstrap": boot,
        "pm1a_cleared": pm1a_pass,
        "csv_path": csv_path.name,
        "csv_sha256": _sha256_file(csv_path),
    }
    man_path = out_dir / "r5b_s1_manifest.json"
    write_json_atomic(man_path, manifest)
    write_json_atomic(
        out_dir / "report_receipt.json",
        {
            "schema_version": "r5b_report_receipt_v1",
            "git_sha": manifest["git_sha"],
            "manifest_path": man_path.name,
            "manifest_sha256": _sha256_file(man_path),
            "artifact_hashes": {man_path.name: _sha256_file(man_path), csv_path.name: manifest["csv_sha256"]},
            "wall_seconds": manifest["wall_seconds"],
        },
    )
    print(f"[R5b-S1-PM1a] COMPLETE in {manifest['wall_seconds']}s | C4_v4={arm_stats['C4_v4_ek5_v3_blocked_telemetry']['overall_acc']} | PM-1a CLEARED={pm1a_pass}", flush=True)
    return manifest


def run_part2_pm4_cross_family_2x2(
    endpoint: str,
    nemotron_id: str,
    root_dir: Path,
    r5_prev_root: Path,
    second_hf_model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> Dict[str, Any]:
    out_dir = root_dir / "r5b_s4_pm4_cross_family_2x2"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    sealed_prev = r5_prev_root / "r5_s4_pm4_cross_family_2x2_and_audit" / "pm4_human_audit_100_sealed_key.json"
    audit_pool = json.loads(sealed_prev.read_text(encoding="utf-8"))["items"]

    print(f"[R5b-S4-PM4] Loading {second_hf_model} in bfloat16 via direct .to('cuda') (no accelerate required)...", flush=True)
    tok = AutoTokenizer.from_pretrained(second_hf_model)
    mdl_box = [AutoModelForCausalLM.from_pretrained(second_hf_model, dtype=torch.bfloat16).to("cuda").eval()]

    def _qwen_chat(sys_p: str, usr_p: str, max_new: int = 40) -> str:
        m_obj = mdl_box[0]
        msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
        txt_in = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok([txt_in], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = m_obj.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        gen_ids = out_ids[0][inp.input_ids.shape[1] :]
        return tok.decode(gen_ids, skip_special_tokens=True).strip()

    # 1) Cross-family Judge Agreement on the 100-item real-conversation audit pool (50 LoCoMo + 50 LongMemEval-S)
    sys_g = (
        "You are an impartial QA evaluator. Reply ONLY YES if the Candidate Answer conveys the same core fact, "
        "date, or entity as the Reference Answer without contradicting it. Reply NO otherwise."
    )
    nem_grades: List[int] = []
    qwen_grades: List[int] = []
    for idx_a, it in enumerate(audit_pool):
        usr_g = f"Reference Answer: {it['gold_answer']}\nCandidate Answer: {it['model_prediction']}\n\nCorrect? Reply YES or NO."
        t_nem, _, _, _ = _call_vllm(endpoint, nemotron_id, sys_g, usr_g, max_tokens=6)
        t_qw = _qwen_chat(sys_g, usr_g, max_new=6)
        g_n = int(t_nem.strip().upper().startswith("YES"))
        g_q = int(t_qw.strip().upper().startswith("YES"))
        nem_grades.append(g_n)
        qwen_grades.append(g_q)
        it["nemotron_fresh_judge"] = g_n
        it["qwen_judge"] = g_q
        if (idx_a + 1) % 25 == 0:
            print(f"[R5b-S4-PM4] Graded {idx_a + 1}/100 real-conv audit items...", flush=True)

    kappa_real_100 = _cohens_kappa(nem_grades, qwen_grades)
    raw_agree_100 = float(np.mean([int(a == b) for a, b in zip(nem_grades, qwen_grades)]))

    # Save enriched sealed key with BOTH Nemotron & Qwen2.5 judge grades!
    enriched_key_path = out_dir / "pm4_human_audit_100_sealed_key_with_qwen.json"
    write_json_atomic(enriched_key_path, {"n_items": len(audit_pool), "items": audit_pool})

    # 2) Full 2x2 Generator x Judge Matrix (`N=80 = 8x10` ZS v3 x 5 key arms = 400 arm-cases -> 800 answers -> 1,600 judgments!)
    zs_cases = build_zero_shortcut_v3_flooded_cases(n_per_family=10)
    zs_arms = [
        "C0_bm25_flat",
        "C2c_astra_4line_rule",
        "C3_ordinary_bitemporal_tms",
        "C4_v3_ek5_v3_k4",
        "C4_v4_ek5_v3_blocked_telemetry",
    ]
    zs_sys = (
        "Answer the factual query strictly using ONLY the dated records below. "
        "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
        "If no records are provided or the records are mutually contradictory with no resolution, reply ONLY 'ABSTAIN'."
    )
    zs_judge_sys = (
        "You are an impartial factual QA judge. Reply ONLY YES if the Candidate Answer matches the Gold Value "
        "and does NOT assert any Stale Retracted Value. For Gold='ABSTAIN', reply YES iff the Candidate abstains or notes unresolved contradiction. Otherwise reply NO."
    )

    matrix_rows = []
    j_nem_all: List[int] = []
    j_qwen_all: List[int] = []
    for idx_c, case in enumerate(zs_cases):
        for arm in zs_arms:
            sel_ids, st_flag = retrieve_any_zs_arm(arm, case, top_k=5)
            if st_flag == "CONTESTED_ABSTAIN" or not sel_ids:
                ctx = "(No admitted records — conflicting claims are epistemically CONTESTED.)"
            else:
                ordered = sorted(sel_ids, key=lambda eid: case["events"][eid].tx_time)
                ctx = "\n".join(f"- [vt={case['events'][eid].valid_time}, tx={case['events'][eid].tx_time}] {case['events'][eid].text}" for eid in ordered)
            prompt = f"Records:\n{ctx}\n\nQuestion: {case['query_text']}\nAnswer:"
            pred_nem, _, _, _ = _call_vllm(endpoint, nemotron_id, zs_sys, prompt, max_tokens=24)
            pred_qw = _qwen_chat(zs_sys, prompt, max_new=24)

            for gen_name, pred_txt in (("Nemotron_30B", pred_nem), ("Qwen2.5_7B", pred_qw)):
                jp = (
                    f"Question: {case['query_text']}\nGold Value: {case['gold']}\n"
                    f"Stale Retracted Values: {', '.join(case['stale_vals'])}\n"
                    f"Candidate Answer: {pred_txt}\n\nCorrect? Reply YES or NO."
                )
                jn_txt, _, _, _ = _call_vllm(endpoint, nemotron_id, zs_judge_sys, jp, max_tokens=6)
                jq_txt = _qwen_chat(zs_judge_sys, jp, max_new=6)
                jn = int(jn_txt.strip().upper().startswith("YES"))
                jq = int(jq_txt.strip().upper().startswith("YES"))
                j_nem_all.append(jn)
                j_qwen_all.append(jq)
                matrix_rows.append({
                    "case_id": case["case_id"], "family": case["family"], "arm": arm,
                    "generator": gen_name, "gold": case["gold"], "pred": pred_txt.replace("\n", " "),
                    "judge_nemotron": jn, "judge_qwen": jq,
                })
        if (idx_c + 1) % 20 == 0:
            print(f"[R5b-S4-PM4] Completed 2x2 Gen x Judge on {idx_c + 1}/{len(zs_cases)} ZS cases...", flush=True)

    kappa_2x2 = _cohens_kappa(j_nem_all, j_qwen_all)
    combined_nem = nem_grades + j_nem_all
    combined_qwen = qwen_grades + j_qwen_all
    kappa_combined = _cohens_kappa(combined_nem, combined_qwen)

    rng_np = np.random.default_rng(42)
    arr_n = np.asarray(combined_nem)
    arr_q = np.asarray(combined_qwen)
    k_boots = [
        _cohens_kappa(arr_n[b_idx].tolist(), arr_q[b_idx].tolist())
        for b_idx in (rng_np.integers(0, len(combined_nem), size=len(combined_nem)) for _ in range(2000))
    ]

    mat_csv = out_dir / "r5b_s4_pm4_2x2_matrix_rows.csv"
    with mat_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(matrix_rows[0].keys()))
        w.writeheader()
        w.writerows(matrix_rows)

    cell_2x2 = {}
    for gen_name in ("Nemotron_30B", "Qwen2.5_7B"):
        for judge_col, j_label in (("judge_nemotron", "Nemotron_Judge"), ("judge_qwen", "Qwen2.5_Judge")):
            cell_2x2[f"{gen_name}_x_{j_label}"] = {
                arm: round(float(np.mean([r[judge_col] for r in matrix_rows if r["generator"] == gen_name and r["arm"] == arm])), 4)
                for arm in zs_arms
            }

    mdl_box[0] = None
    torch.cuda.empty_cache()

    manifest = {
        "schema_version": "r5b_s4_pm4_cross_family_2x2_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "models_crossed": {"family_1": nemotron_id, "family_2": second_hf_model},
        "inter_judge_agreement": {
            "real_conv_100_cohens_kappa": round(kappa_real_100, 4),
            "real_conv_100_raw_agreement": round(raw_agree_100, 4),
            "zs_2x2_800_cohens_kappa": round(kappa_2x2, 4),
            "combined_900_cohens_kappa": round(kappa_combined, 4),
            "combined_kappa_ci95": [round(float(np.percentile(k_boots, 2.5)), 4), round(float(np.percentile(k_boots, 97.5)), 4)],
            "pm4a_kappa_ge_085_pass": bool(kappa_combined >= 0.85),
        },
        "matrix_2x2_accuracy_by_arm": cell_2x2,
        "enriched_sealed_key_json": enriched_key_path.name,
        "enriched_sealed_key_sha256": _sha256_file(enriched_key_path),
        "matrix_csv": mat_csv.name,
        "matrix_csv_sha256": _sha256_file(mat_csv),
    }
    man_path = out_dir / "r5b_s4_manifest.json"
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
                mat_csv.name: manifest["matrix_csv_sha256"],
                enriched_key_path.name: manifest["enriched_sealed_key_sha256"],
            },
            "wall_seconds": manifest["wall_seconds"],
        },
    )
    print(f"[R5b-S4-PM4] COMPLETE in {manifest['wall_seconds']}s | Combined Cohen's kappa={kappa_combined:.4f} (pass>=0.85: {manifest['inter_judge_agreement']['pm4a_kappa_ge_085_pass']})", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="R5b Completion Spark 6724")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--r5-prev-root", type=Path, default=Path("/home/mattmre/chelated-night1/out/20261122-overnight-r5/spark6724"))
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8890/v1")
    parser.add_argument("--model-id", type=str, default="nemotron-lightning")
    parser.add_argument("--second-hf-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    summary: Dict[str, Any] = {"started_at": datetime.now().isoformat(), "stages": {}}
    for s_name, fn in [
        ("part1_zs_v3_pm1a_closure", lambda: run_part1_zs_v3_pm1a_closure(args.endpoint, args.model_id, args.output_root, args.max_workers)),
        ("part2_pm4_cross_family_2x2", lambda: run_part2_pm4_cross_family_2x2(args.endpoint, args.model_id, args.output_root, args.r5_prev_root, args.second_hf_model)),
    ]:
        try:
            res = fn()
            summary["stages"][s_name] = {"status": "COMPLETED", "wall_seconds": res["wall_seconds"]}
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[R5b-6724] ERROR in {s_name}: {exc}\n{tb}", flush=True)
            summary["stages"][s_name] = {"status": "ERROR", "error": str(exc), "traceback": tb}
        write_json_atomic(args.output_root / "spark6724_r5b_status.json", summary)

    summary["total_wall_seconds"] = round(time.perf_counter() - t0_all, 2)
    summary["telemetry"] = process_telemetry()
    write_json_atomic(args.output_root / "spark6724_r5b_status.json", summary)
    print(f"[R5b-6724] ALL PARTS FINISHED in {summary['total_wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
