"""Flagship #2 Stage 2C — Hard Adversarial Live Nemotron 3.5 Stress Test (No Role-Tag Leakage).

Addresses the Stage 2B saturation finding where `C3_ordinary_bitemporal_tms` tied `C4_iq04_candidate`
at 100% on live Nemotron 3.5 because `top_k=5` still surfaced the `role=CORRECTION`-tagged message.

Hardening changes:
  1. Strip ALL synthetic `role=...` tags from the LLM prompt (plain natural text + timestamp only).
  2. Popularity flood exceeds `top_k=5`: 10-12 unlinked paraphrased copies + 1 correction, so any
     arm that fails to suppress unlinked copies evicts the true correction from the top-5 context!
  3. Hard Clean-Control False-Positive Trap: an independent later valid report shares vocabulary
     with an earlier retracted rumor (tests whether `C4_iq04_candidate` over-demotes valid updates!).
  4. Implicit downstream synthesis: an unlinked downstream calculation depends on a retracted value.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iq04_temporal_evidence import (  # noqa: E402
    EvidenceEdge,
    EvidenceEvent,
    LifecycleQuery,
    ScopedBM25,
    evaluate_arm_on_query,
)
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    paired_bootstrap_delta,
    process_telemetry,
    write_json_atomic,
)

HARD_FAMILIES = (
    "flood_12_unlinked_paraphrase_copies",
    "flood_10_linked_popular_error_corrected",
    "hard_clean_trap_lexical_overlap_valid_update",
    "unlinked_downstream_synthesis_from_retracted",
    "erroneous_retraction_then_reinstated_flood",
)

LIVE_ARMS = (
    "C0_bm25_flat",
    "C1_scalemas_unsigned",
    "C3_ordinary_bitemporal_tms",
    "C4_iq04_candidate",
)

_DOMAINS = (
    ("turbine_limit", "max_rpm", ("1800rpm", "2400rpm", "3100rpm", "3600rpm")),
    ("chemical_dose", "safe_ppm", ("15ppm", "35ppm", "60ppm", "95ppm")),
    ("bridge_cable", "tension_kn", ("410kn", "580kn", "720kn", "890kn")),
    ("battery_cell", "cutoff_mv", ("3200mv", "3650mv", "4050mv", "4200mv")),
    ("server_quota", "max_qps", ("250qps", "600qps", "1200qps", "2500qps")),
)


def _h_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16)


def build_hard_adversarial_corpus(n_per_family: int = 30, seed: int = 101) -> Tuple[
    Dict[str, EvidenceEvent], List[EvidenceEdge], List[LifecycleQuery]
]:
    events: Dict[str, EvidenceEvent] = {}
    edges: List[EvidenceEdge] = []
    queries: List[LifecycleQuery] = []
    idx = 0

    for fam in HARD_FAMILIES:
        for k in range(n_per_family):
            lc_id = f"hard_{idx:04d}"
            dom, slot, vals = _DOMAINS[(idx + seed) % len(_DOMAINS)]
            h = _h_int(f"{seed}:{lc_id}:{fam}")
            ent = f"{dom}_{idx:04d}"
            v_false = vals[h % len(vals)]
            v_true = vals[(h + 1) % len(vals)]

            def add_ev(suf: str, val: str, txt: str, vt: int, tt: int, role: str, h_true: bool) -> str:
                eid = f"{lc_id}_{suf}"
                events[eid] = EvidenceEvent(
                    event_id=eid, lifecycle_id=lc_id, entity=ent, slot=slot,
                    claim_value=val, text=txt, valid_time=vt, transaction_time=tt,
                    source_id=f"src_{lc_id}_{suf}", role=role, hidden_is_true_at_query=h_true,
                )
                return eid

            if fam == "flood_12_unlinked_paraphrase_copies":
                # Root false claim at t=1 + 12 UNLINKED paraphrased copies at t=2..9 (exceeds top_k=5!)
                # Plus 1 true correction at t=10 with brief text (lower BM25 than verbose copies!)
                r_bad = add_ev(
                    "root", v_false,
                    f"Field specification for {ent} {slot} reports {v_false} as the active operating parameter.",
                    1, 1, "CLAIM", False,
                )
                for c in range(12):
                    add_ev(
                        f"cp_{c}", v_false,
                        f"Regional bulletin {c} confirms {ent} {slot} active operating parameter is {v_false} for {ent} {slot}.",
                        2 + (c % 7), 2 + (c % 7), "COPY", False,
                    )
                corr = add_ev(
                    "corr", v_true,
                    f"Update {ent}: {slot} is {v_true} (prior {v_false} withdrawn).",
                    10, 10, "CORRECTION", True,
                )
                edges.append(EvidenceEdge(corr, r_bad, "REFUTE", 10))
                edges.append(EvidenceEdge(corr, r_bad, "SUPERSEDE", 10))
                queries.append(
                    LifecycleQuery(
                        f"q_{lc_id}", lc_id, fam, "REPORT", ent, slot,
                        f"What is the active operating parameter {slot} for {ent}?",
                        12, 12, v_true, True, False, (v_false,),
                    )
                )

            elif fam == "flood_10_linked_popular_error_corrected":
                # 10 LINKED popular copies citing root_bad + 1 correction -> tests ScaleMAS unsigned flood!
                r_bad = add_ev(
                    "root", v_false,
                    f"Primary release states {ent} {slot} operating parameter is {v_false}.",
                    1, 1, "CLAIM", False,
                )
                for c in range(10):
                    cid = add_ev(
                        f"sup_{c}", v_false,
                        f"Team {c} cites primary release confirming {ent} {slot} operating parameter {v_false}.",
                        2 + (c % 7), 2 + (c % 7), "DERIVED", False,
                    )
                    edges.append(EvidenceEdge(cid, r_bad, "SUPPORT", 2 + (c % 7)))
                    edges.append(EvidenceEdge(cid, r_bad, "DERIVES_FROM", 2 + (c % 7)))
                corr = add_ev(
                    "corr", v_true,
                    f"Revision: {ent} {slot} is {v_true}, replacing primary release.",
                    10, 10, "CORRECTION", True,
                )
                edges.append(EvidenceEdge(corr, r_bad, "REFUTE", 10))
                edges.append(EvidenceEdge(corr, r_bad, "SUPERSEDE", 10))
                queries.append(
                    LifecycleQuery(
                        f"q_{lc_id}", lc_id, fam, "REPORT", ent, slot,
                        f"What is the active operating parameter {slot} for {ent}?",
                        12, 12, v_true, True, False, (v_false,),
                    )
                )

            elif fam == "hard_clean_trap_lexical_overlap_valid_update":
                # HARD FALSE-POSITIVE TRAP:
                # Early draft at t=1 asserts v_true (VALID!), gets retracted erroneously at t=4 by a rumor,
                # AND at t=8 an independent fresh measurement re-confirms v_true WITHOUT an explicit
                # REINSTATEMENT role or edge! Does C4 wrongly kill the fresh t=8 confirmation?
                r_old = add_ev(
                    "draft", v_false,
                    f"Early calibration log for {ent} {slot} noted {v_false} as provisional value.",
                    1, 1, "CLAIM", False,
                )
                ret = add_ev(
                    "ret", v_false,
                    f"Early calibration log for {ent} {slot} ({v_false}) is voided.",
                    4, 4, "RETRACTION", True,
                )
                edges.append(EvidenceEdge(ret, r_old, "REFUTE", 4))
                # Independent valid report at t=8 using very similar phrasing but NEW valid value v_true!
                add_ev(
                    "fresh_valid", v_true,
                    f"Independent calibration log for {ent} {slot} confirms {v_true} as verified value.",
                    8, 8, "CLAIM", True,
                )
                queries.append(
                    LifecycleQuery(
                        f"q_{lc_id}", lc_id, fam, "REPORT", ent, slot,
                        f"What is the verified calibration value {slot} for {ent}?",
                        12, 12, v_true, False, True, (v_false,),
                    )
                )

            elif fam == "unlinked_downstream_synthesis_from_retracted":
                # Root false value retracted at t=9; 8 unlinked downstream syntheses written at t=3..8
                # echoing the retracted value with domain reasoning; 1 terse correction at t=9.
                r_bad = add_ev(
                    "root", v_false,
                    f"Base telemetry sets {ent} {slot} at {v_false}.",
                    1, 1, "CLAIM", False,
                )
                for c in range(8):
                    add_ev(
                        f"syn_{c}", v_false,
                        f"Downstream safety check {c} uses {ent} {slot} = {v_false} to certify {ent} {slot} clearance.",
                        3 + (c % 5), 3 + (c % 5), "DERIVED", False,
                    )
                corr = add_ev(
                    "corr", v_true,
                    f"Telemetry corrected: {ent} {slot} is {v_true}.",
                    9, 9, "CORRECTION", True,
                )
                edges.append(EvidenceEdge(corr, r_bad, "REFUTE", 9))
                edges.append(EvidenceEdge(corr, r_bad, "SUPERSEDE", 9))
                queries.append(
                    LifecycleQuery(
                        f"q_{lc_id}", lc_id, fam, "REPORT", ent, slot,
                        f"What is the current {slot} clearance setting for {ent}?",
                        12, 12, v_true, True, False, (v_false,),
                    )
                )

            else:  # erroneous_retraction_then_reinstated_flood
                # Valid root at t=1 -> erroneous retraction at t=5 + 8 unlinked rumors echoing v_false at t=5..8
                # -> reinstatement at t=10 refuting the erroneous retraction!
                r_good = add_ev(
                    "root_good", v_true,
                    f"Certified spec for {ent} {slot} is {v_true}.",
                    1, 1, "CLAIM", True,
                )
                err_ret = add_ev(
                    "err_ret", v_false,
                    f"Alert voids {ent} {slot} {v_true} and provisionally sets {v_false}.",
                    5, 5, "CLAIM", False,
                )
                edges.append(EvidenceEdge(err_ret, r_good, "REFUTE", 5))
                for c in range(8):
                    add_ev(
                        f"rum_{c}", v_false,
                        f"Dispatch {c} repeats alert setting {ent} {slot} to {v_false} for {ent} {slot}.",
                        6 + (c % 3), 6 + (c % 3), "COPY", False,
                    )
                reinst = add_ev(
                    "reinst", v_true,
                    f"Board cancels alert: {ent} {slot} restored to {v_true}.",
                    10, 10, "REINSTATEMENT", True,
                )
                edges.append(EvidenceEdge(reinst, err_ret, "REFUTE", 10))
                edges.append(EvidenceEdge(reinst, r_good, "SUPPORT", 10))
                queries.append(
                    LifecycleQuery(
                        f"q_{lc_id}", lc_id, fam, "REPORT", ent, slot,
                        f"What is the current certified {slot} for {ent}?",
                        12, 12, v_true, True, False, (v_false,),
                    )
                )
            idx += 1

    return events, edges, queries


def _query_nemotron_no_role_tags(
    *,
    api_base: str,
    model_id: str,
    query: LifecycleQuery,
    context_events: Sequence[EvidenceEvent],
    timeout_s: int = 90,
) -> Dict[str, Any]:
    """Plain-text prompt with ZERO synthetic role=... metadata tags."""
    if not context_events:
        return {"answer": "ABSTAIN", "raw": "NO_CONTEXT", "prompt_tokens": 0, "latency_ms": 0.0}
    ctx_lines = [
        f"[{i + 1}] (time={ev.transaction_time}) {ev.text}"
        for i, ev in enumerate(context_events)
    ]
    system_msg = (
        "Answer the factual query using ONLY the numbered snippets below. "
        "Output ONLY the single value token (e.g. 2400rpm, 60ppm, 580kn, 3650mv, 1200qps) "
        "or ABSTAIN if unknown. Do not explain."
    )
    user_msg = (
        f"Query: {query.query_text}\nEntity: {query.entity} | Attribute: {query.slot}\n\n"
        "Snippets:\n" + "\n".join(ctx_lines) + "\n\nAnswer:"
    )
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        "temperature": 0.0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{api_base.rstrip('/')}/chat/completions", json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        raw = (data["choices"][0]["message"].get("content") or "").strip()
        pt = int((data.get("usage") or {}).get("prompt_tokens", 0))
    except Exception as exc:
        return {"answer": "ERROR", "raw": repr(exc), "prompt_tokens": 0, "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)}

    low = raw.lower()
    if "abstain" in low:
        pred = "ABSTAIN"
    else:
        tokens = [t.strip(".,:;()[]{}\"'`") for t in low.split() if t.strip(".,:;()[]{}\"'`")]
        cands = [query.gold_answer.lower()] + [v.lower() for v in query.stale_false_values]
        matched = next((c for c in cands if c in tokens), None) or next((c for c in cands if c in low), None)
        pred = matched if matched is not None else (tokens[0] if tokens else "ABSTAIN")
    return {"answer": pred, "raw": raw[:120], "prompt_tokens": pt, "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)}


def run_hard_stress(
    *,
    output_dir: Path,
    api_base: str,
    model_id: str,
    n_per_family: int = 30,
    max_workers: int = 8,
    seed: int = 101,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    events, edges, queries = build_hard_adversarial_corpus(n_per_family=n_per_family, seed=seed)
    bm25 = ScopedBM25(events)
    lc_ev: Dict[str, List[str]] = defaultdict(list)
    for eid, ev in events.items():
        lc_ev[ev.lifecycle_id].append(eid)
    lc_ed: Dict[str, List[EvidenceEdge]] = defaultdict(list)
    for e in edges:
        lc_ed[events[e.src_id].lifecycle_id].append(e)

    # Retrieve top-5 per (arm, query)
    ctx_map: Dict[Tuple[str, str], List[str]] = {}
    det_err: Dict[str, Dict[str, float]] = {a: {} for a in LIVE_ARMS}
    for arm in LIVE_ARMS:
        for q in queries:
            res = evaluate_arm_on_query(arm, q, lc_ev[q.lifecycle_id], lc_ed[q.lifecycle_id], events, bm25, top_k=5)
            ctx_map[(arm, q.query_id)] = list(res["top_context_ids"])
            if q.is_post_correction:
                det_err[arm][q.query_id] = 1.0 if res["is_stale_or_false"] else 0.0

    tasks = [(arm, q) for arm in LIVE_ARMS for q in queries]
    print(f"[F2-HARD] Running {len(tasks)} hard adversarial live calls on {model_id} @ {api_base}...", flush=True)

    def _work(item: Tuple[str, LifecycleQuery]) -> Dict[str, Any]:
        arm, q = item
        evs = [events[eid] for eid in ctx_map[(arm, q.query_id)]]
        out = _query_nemotron_no_role_tags(api_base=api_base, model_id=model_id, query=q, context_events=evs)
        pred = out["answer"].lower()
        gold = q.gold_answer.lower()
        ok = (pred == gold)
        ab = (pred == "abstain")
        ae = (pred == "error")
        stale = (not ok) and (not ab) and (not ae)
        return {
            "query_id": q.query_id,
            "case_family": q.case_family,
            "is_post_correction": q.is_post_correction,
            "arm": arm,
            "gold": q.gold_answer,
            "pred": out["answer"],
            "is_correct": ok,
            "is_abstain": ab,
            "is_stale_or_false": stale,
            "api_error": ae,
            "prompt_tokens": out["prompt_tokens"],
            "latency_ms": out["latency_ms"],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(_work, tasks))

    with (output_dir / "f2_hard_live_rows.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    arm_summary: Dict[str, Any] = {}
    live_post_err: Dict[str, Dict[str, float]] = {}
    for arm in LIVE_ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        post = [r for r in sub if r["is_post_correction"]]
        clean = [r for r in sub if not r["is_post_correction"]]
        live_post_err[arm] = {r["query_id"]: (1.0 if r["is_stale_or_false"] else 0.0) for r in post}
        arm_summary[arm] = {
            "post_stale_or_false_rate": round(float(np.mean(list(live_post_err[arm].values()))), 6),
            "post_accuracy": round(float(np.mean([1.0 if r["is_correct"] else 0.0 for r in post])), 6),
            "hard_clean_trap_accuracy": round(float(np.mean([1.0 if r["is_correct"] else 0.0 for r in clean])), 6),
            "overall_accuracy": round(float(np.mean([1.0 if r["is_correct"] else 0.0 for r in sub])), 6),
            "api_errors": sum(1 for r in sub if r["api_error"]),
            "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in sub])), 1),
            "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in sub])), 1),
        }

    boot_c3 = paired_bootstrap_delta(
        live_post_err["C3_ordinary_bitemporal_tms"], live_post_err["C4_iq04_candidate"], n_bootstrap=10_000, seed=seed
    )
    boot_sm = paired_bootstrap_delta(
        live_post_err["C1_scalemas_unsigned"], live_post_err["C4_iq04_candidate"], n_bootstrap=10_000, seed=seed
    )
    c3_e = arm_summary["C3_ordinary_bitemporal_tms"]["post_stale_or_false_rate"]
    c4_e = arm_summary["C4_iq04_candidate"]["post_stale_or_false_rate"]
    rel_drop_c3 = ((c3_e - c4_e) / c3_e) if c3_e > 0 else 0.0

    md = [
        "# Flagship #2 Stage 2C — Hard Adversarial Live Nemotron 3.5 Stress Test (No Role Tags, Flood > Top-5)",
        "",
        f"- **Git SHA:** `{current_git_sha(REPO_ROOT)}`",
        f"- **Model:** `{model_id}` @ `{api_base}` (`{len(rows)}` live calls, `0` role-tag hints)",
        f"- **Live `C4` vs `C3_ordinary_bitemporal_tms`:** rel. stale-error reduction = **`{rel_drop_c3*100:.1f}%`** (`abs={boot_c3['mean_delta']:+.4f}`, `LCB95={boot_c3['lcb_95']:+.4f}`)",
        f"- **Live `C4` vs `C1_scalemas_unsigned`:** abs stale-error drop = **`{boot_sm['mean_delta']:+.4f}`** (`LCB95={boot_sm['lcb_95']:+.4f}`)",
        "",
        "## 1. Overall Live Adversarial Results",
        "",
        "| Arm | Post Stale/False Error ↓ | Post Accuracy ↑ | Hard Clean-Trap Acc ↑ | Overall Acc ↑ | Prompt Toks | Mean Lat (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in LIVE_ARMS:
        s = arm_summary[arm]
        md.append(
            f"| `{arm}` | **{s['post_stale_or_false_rate']:.4f}** | {s['post_accuracy']:.4f} "
            f"| {s['hard_clean_trap_accuracy']:.4f} | {s['overall_accuracy']:.4f} "
            f"| {s['mean_prompt_tokens']:.1f} | {s['mean_latency_ms']:.1f} |"
        )

    md += [
        "",
        "## 2. Per-Family Live Stale/False Error Rate (Lower Is Better)",
        "",
        "| Adversarial Family | C0 BM25 | C1 ScaleMAS Unsigned | C3 Ordinary Bitemp TMS | **C4 IQ-04 Candidate** |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam in HARD_FAMILIES:
        vals = []
        for arm in LIVE_ARMS:
            sub = [r for r in rows if r["case_family"] == fam and r["arm"] == arm]
            vals.append(f"{sum(1 for r in sub if r['is_stale_or_false'])/len(sub):.4f}")
        md.append(f"| `{fam}` | {vals[0]} | {vals[1]} | {vals[2]} | **{vals[3]}** |")

    (output_dir / "f2_hard_verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    manifest = {
        "job": "flagship2_hard_adversarial_live",
        "git_sha": current_git_sha(REPO_ROOT),
        "model_id": model_id,
        "arms": arm_summary,
        "live_c4_vs_c3_rel_drop": round(rel_drop_c3, 6),
        "live_c4_vs_c3_abs_drop": round(boot_c3["mean_delta"], 6),
        "live_c4_vs_c3_lcb95": round(boot_c3["lcb_95"], 6),
        "live_c4_vs_scalemas_abs_drop": round(boot_sm["mean_delta"], 6),
        "live_c4_vs_scalemas_lcb95": round(boot_sm["lcb_95"], 6),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "telemetry": process_telemetry(),
    }
    msha = write_json_atomic(output_dir / "f2_hard_manifest.json", manifest)
    write_json_atomic(output_dir / "report_receipt.json", {"job": "flagship2_hard_adversarial_live", "status": "COMPLETED", "manifest_sha256": msha})
    print(f"[F2-HARD] COMPLETE in {manifest['wall_seconds']}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Flagship #2 Hard Adversarial Live Nemotron Stress Test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8890/v1")
    parser.add_argument("--model-id", type=str, default="nemotron-lightning")
    parser.add_argument("--n-per-family", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    run_hard_stress(
        output_dir=args.output_dir,
        api_base=args.api_base,
        model_id=args.model_id,
        n_per_family=args.n_per_family,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
