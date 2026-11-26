"""W22-15 runner: selective-trigger 0.5B extraction on Suite A and the #62/D13 sham-controlled replay.

Preregistration: ``artifacts/pv7_w23b_w22_15/PREREGISTRATION.md`` (committed before this file).

Stages (run sequentially; each writes a small evidence JSON next to the preregistration, bulk goes to
``artifacts/pv7_w23b_w22_15/scratch/``):

* ``cpu-replay``  - #62 stage-1 recorded replay, full default grid, baseline vs selective backend (CPU).
* ``dev-recall``  - router recall/trigger rate on the D9 synthetic DEV split (CPU, non-gating).
* ``train``       - retrain the 0.5B LoRA claim extractor (GPU) into ``scratch/lora``.
* ``suite-a``     - FULL vs SELECTIVE arms on Suite A with per-case component latency (GPU).
* ``live62``      - local micro-batching OpenAI server + plain-live vs selective-live #62 arms (GPU).
* ``aggregate``   - apply the frozen decision rule and write ``w22_15_results.json``.

GPU stages refuse to start unless ``nvidia-smi`` reports memory.used < 4000 MiB and no other python process
on the GPU (``--skip-gpu-check`` is NOT provided on purpose).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pooled_agent_runtime import (  # noqa: E402
    JOB_SLA_DEADLINE_MS,
    TWIN_BRANCHES,
    LiveVLLMWorkerBackend,
    RecordedReplayWorkerBackend,
    run_pooled_agent_experiment,
)
from scripts.night1_common import current_git_sha, write_json_atomic  # noqa: E402
from selective_trigger_router import (  # noqa: E402
    AMENDMENT_CONFLICT_STEMS,
    AUTHORITY_TRIGGER_MIN,
    SelectiveTriggerLiveWorkerBackend,
    SelectiveTriggerReplayWorkerBackend,
    doc_trigger_reason,
    route_suite_a_case,
    rule_reader,
)

OUT_DIR = REPO_ROOT / "artifacts" / "pv7_w23b_w22_15"
SCRATCH = OUT_DIR / "scratch"
OSV_CACHE = SCRATCH / "osv_cve_seeds.json"
SUITE_A_SHA = "cb6bc0082b5850f4edfea668d1ecb2a73fe6f898b7476e34ed11036cb378765b"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GPU_FREE_MAX_MIB = 4000
MAX_NEW_TOKENS = 64

# Recorded comparison numbers (read-only, PREREGISTRATION.md D4/D5).
RECORDED = {
    "bare_70b_mean_query_ms_k5": 3257.82,
    "bare_70b_mean_query_ms_k17": 5361.28,
    "bare_70b_source": "artifacts/wave11_d10_5m_real_cutover_and_70b_scale/part_b_70b_sweep_summary.json",
    "d10b_chelated_e2e_ms_per_case": 26596.47,
    "d7_chelated_v4_e2e_ms_per_case": 10373.31,
    "d9a_deterministic_frozen_d1_accuracy": 0.9667,
    "d9a_extract_ms_per_case_mean": 2104.008,
    "d9a_merged_sha_prefix": "4cef79a7",
}

LIVE_REGIMES = ("no_change", "sham_subtask_reopen_interrupt", "withdrawal_only")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_path(name: str) -> Path:
    return OUT_DIR / f"stage_{name}.json"


def _summ(vals: Sequence[float]) -> Dict[str, float]:
    xs = sorted(float(v) for v in vals)
    if not xs:
        return {"n": 0}
    p95_idx = min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 3),
        "p50": round(statistics.median(xs), 3),
        "p95": round(xs[p95_idx], 3),
        "min": round(xs[0], 3),
        "max": round(xs[-1], 3),
    }


# ---------------------------------------------------------------------------
# GPU guard
# ---------------------------------------------------------------------------


def gpu_status() -> Dict[str, Any]:
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().splitlines()[0]
    name, mem, util = [x.strip() for x in q.split(",")]
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().splitlines()
    py = []
    for line in apps:
        parts = [x.strip() for x in line.split(",", 1)]
        if len(parts) == 2 and "python" in parts[1].lower() and int(parts[0]) != os.getpid():
            py.append({"pid": int(parts[0]), "process_name": parts[1]})
    return {"gpu_name": name, "memory_used_mib": int(mem), "utilization_pct": int(util), "other_python_on_gpu": py}


def require_gpu_free() -> Dict[str, Any]:
    st = gpu_status()
    if st["memory_used_mib"] >= GPU_FREE_MAX_MIB or st["other_python_on_gpu"]:
        raise SystemExit(f"GPU busy, refusing GPU stage: {json.dumps(st)}")
    return st


# ---------------------------------------------------------------------------
# #62 manifest summarisation
# ---------------------------------------------------------------------------


def _compact_cells(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    keep = (
        "pool_size",
        "schedule_id",
        "branch",
        "system_id",
        "event_regime",
        "is_full_restart_comparator",
        "eligible_jobs",
        "on_time_jobs",
        "late_or_timed_out_jobs",
        "stale_commits",
        "on_time_cited_correct_jobs",
        "mean_latency_ms",
        "p95_latency_ms",
        "mean_reopen_overhead_ms",
        "live_model_calls",
    )
    return [{k: c.get(k) for k in keep} for c in manifest.get("cells", [])]


def summarize_manifest(manifest: Dict[str, Any], router: Any = None) -> Dict[str, Any]:
    return {
        "status": manifest.get("status"),
        "mode": manifest.get("mode"),
        "worker_backend": manifest.get("worker_backend"),
        "live_model_calls": manifest.get("live_model_calls"),
        "total_cells_executed": manifest.get("total_cells_executed"),
        "total_eligible_jobs_executed": manifest.get("total_eligible_jobs_executed"),
        "zeromu_ek5_stale_commits_total": manifest.get("zeromu_ek5_stale_commits_total"),
        "gates": manifest.get("gates"),
        "sham_reopen_isolation": manifest.get("sham_reopen_isolation"),
        "full_restart_comparator": manifest.get("full_restart_comparator"),
        "router_stats": router.stats.to_dict() if router is not None else None,
        "cells": _compact_cells(manifest),
    }


def g3_from_cells(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sham = [
        c
        for c in cells
        if c.get("event_regime") == "sham_subtask_reopen_interrupt"
        and c.get("system_id") == "ZeroMu+EK5_v3"
        and not c.get("is_full_restart_comparator")
    ]
    otc = sum(int(c.get("on_time_cited_correct_jobs") or 0) for c in sham)
    stale = sum(int(c.get("stale_commits") or 0) for c in sham)
    return {
        "n_sham_cells": len(sham),
        "branches": sorted(str(c.get("branch")) for c in sham),
        "sham_on_time_cited_correct_jobs": otc,
        "sham_stale_commits": stale,
        "pass": bool(len(sham) == 2 and otc > 0 and stale == 0),
    }


def _run_62(backend: Any, **grid: Any) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"worker_backend": backend, "include_jobs": True}
    kw.update(grid)
    return run_pooled_agent_experiment(**kw)


# ---------------------------------------------------------------------------
# Stage: cpu-replay
# ---------------------------------------------------------------------------


def stage_cpu_replay() -> Dict[str, Any]:
    out: Dict[str, Any] = {"stage": "cpu-replay", "generated_utc": _now(), "grid": "full default grid"}
    for arm, backend in (
        ("baseline_recorded_replay", RecordedReplayWorkerBackend()),
        ("selective_recorded_replay", SelectiveTriggerReplayWorkerBackend()),
    ):
        t0 = time.perf_counter()
        man = _run_62(backend, mode="stage1-controlled-replay")
        wall = round(time.perf_counter() - t0, 3)
        sha = write_json_atomic(SCRATCH / f"cpu_replay_{arm}_manifest.json", man)
        s = summarize_manifest(man, getattr(backend, "router", None))
        s.pop("cells")
        s["wall_seconds"] = wall
        s["full_manifest_scratch_sha256"] = sha
        out[arm] = s
        print(f"[cpu-replay] {arm}: status={s['status']} wall={wall}s router={s['router_stats']}", flush=True)
    write_json_atomic(_stage_path("cpu_replay"), out)
    return out


# ---------------------------------------------------------------------------
# Stage: dev-recall (non-gating, CPU)
# ---------------------------------------------------------------------------


def stage_dev_recall() -> Dict[str, Any]:
    from chelated_fast_edge_extractor import build_disjoint_extractor_splits
    from chelated_raw_text_exocortex import DEFAULT_CHANNEL_AUTHORITY

    splits = build_disjoint_extractor_splits(n_train=600, n_dev=120, seed=20261123)
    dev = splits["dev"]
    by_stance: Dict[str, Dict[str, int]] = {}
    for ex in dev:
        auth = int(DEFAULT_CHANNEL_AUTHORITY.get(ex.doc.source_channel, 1))
        trig = doc_trigger_reason(ex.doc.text, auth) is not None
        b = by_stance.setdefault(str(ex.target_stance), {"n": 0, "triggered": 0})
        b["n"] += 1
        b["triggered"] += int(trig)
    non_assert_n = sum(v["n"] for k, v in by_stance.items() if k != "ASSERT")
    non_assert_t = sum(v["triggered"] for k, v in by_stance.items() if k != "ASSERT")
    assert_v = by_stance.get("ASSERT", {"n": 0, "triggered": 0})
    out = {
        "stage": "dev-recall",
        "generated_utc": _now(),
        "n_dev": len(dev),
        "by_target_stance": dict(sorted(by_stance.items())),
        "recall_non_assert": round(non_assert_t / non_assert_n, 4) if non_assert_n else None,
        "non_assert_n": non_assert_n,
        "non_assert_triggered": non_assert_t,
        "assert_trigger_rate": round(assert_v["triggered"] / assert_v["n"], 4) if assert_v["n"] else None,
        "overall_trigger_rate": round(sum(v["triggered"] for v in by_stance.values()) / max(1, len(dev)), 4),
    }
    write_json_atomic(_stage_path("dev_recall"), out)
    print(f"[dev-recall] {json.dumps(out)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Stage: train (GPU)
# ---------------------------------------------------------------------------


def stage_train() -> Dict[str, Any]:
    gpu = require_gpu_free()
    from chelated_fast_edge_extractor import (
        FastEdgeClaimExtractor,
        assert_train_dev_disjoint,
        build_disjoint_extractor_splits,
        train_fast_claim_extractor_lora,
    )

    splits = build_disjoint_extractor_splits(n_train=600, n_dev=120, seed=20261123)
    disjoint = assert_train_dev_disjoint(splits["train"], splits["dev"])
    t0 = time.perf_counter()
    summary = train_fast_claim_extractor_lora(
        base_model_id=BASE_MODEL,
        train_examples=splits["train"],
        output_dir=SCRATCH / "lora",
        probe_examples=splits["dev"][:4],
        epochs=3,
        seed=42,
        device="cuda",
    )
    train_wall = round(time.perf_counter() - t0, 3)
    with FastEdgeClaimExtractor(summary["merged_model_dir"], device="cuda", default_max_new_tokens=MAX_NEW_TOKENS) as ext:
        ext.warmup(splits["dev"][0].doc)
        dev_metrics = ext.evaluate_on_synthetic_dev(splits["dev"], batch_size=17, max_new_tokens=MAX_NEW_TOKENS)
    dev_metrics.pop("per_example", None)
    merged_sha = summary.get("merged_weights_sha256") or {}
    combined = hashlib.sha256(json.dumps(merged_sha, sort_keys=True).encode("utf-8")).hexdigest()
    out = {
        "stage": "train",
        "generated_utc": _now(),
        "gpu_at_start": gpu,
        "train_summary": {k: v for k, v in summary.items() if k not in ("mask_audit",)},
        "merged_weights_combined_sha256": combined,
        "recorded_d9a_merged_sha_prefix": RECORDED["d9a_merged_sha_prefix"],
        "label": "retrained, not byte-identical to D9-A",
        "train_dev_disjoint": disjoint,
        "dev_metrics": dev_metrics,
        "train_wall_seconds": train_wall,
    }
    write_json_atomic(_stage_path("train"), out)
    print(f"[train] done wall={train_wall}s dev={ {k: dev_metrics.get(k) for k in ('json_valid_rate', 'full_tuple_accuracy')} }", flush=True)
    return out


# ---------------------------------------------------------------------------
# Stage: suite-a (GPU)
# ---------------------------------------------------------------------------


def stage_suite_a() -> Dict[str, Any]:
    gpu = require_gpu_free()
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    from chelated_epistemic_proposer_v2 import AuthorityDominatedExocortexV2
    from chelated_fast_edge_extractor import FastEdgeClaimExtractor, build_disjoint_extractor_splits
    from prospective_realworld_drift_corpus import build_prospective_realworld_drift_suite
    from scripts.night1_common import FastBM25Index
    from scripts.run_wave6_d4_trained_epistemic_grpo import score_d1_prose_prediction
    from scripts.run_wave11_d9_fast_extractor_and_u_grpo import _rrf_combine
    from spherical_migration_bridge import l2_normalize

    train_ev = json.loads(_stage_path("train").read_text(encoding="utf-8"))
    merged_dir = train_ev["train_summary"]["merged_model_dir"]

    cases, all_docs, sha = build_prospective_realworld_drift_suite(OSV_CACHE)
    if sha != SUITE_A_SHA:
        raise SystemExit(f"Suite A sha mismatch: {sha}")
    by_id = {d.doc_id: d for d in all_docs}
    case_docs = [
        sorted([by_id[did] for did in c["doc_ids"]], key=lambda d: (d.ingested_ts, d.published_ts, d.doc_id))
        for c in cases
    ]

    bge = SentenceTransformer("BAAI/bge-m3", device="cuda")
    bge.max_seq_length = 384

    def enc(texts: Sequence[str], _bge: Any = bge) -> Any:
        return np.asarray(
            _bge.encode(list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )

    dev0 = build_disjoint_extractor_splits(n_train=600, n_dev=120, seed=20261123)["dev"][0].doc
    arms: Dict[str, Any] = {}
    with FastEdgeClaimExtractor(merged_dir, device="cuda", default_max_new_tokens=MAX_NEW_TOKENS) as ext:
        warm_ms = ext.warmup(dev0)
        enc(["warmup"])  # untimed encoder warmup
        for arm in ("FULL", "SELECTIVE"):
            rows: List[Dict[str, Any]] = []
            for case, docs in zip(cases, case_docs):
                if arm == "SELECTIVE":
                    dec = route_suite_a_case(docs)
                    keep = set(dec.triggered_ids)
                    idx_docs = [d for d in docs if d.doc_id in keep]
                    route_ms = dec.route_ms
                    reasons = dict(dec.reasons)
                else:
                    idx_docs = list(docs)
                    route_ms = 0.0
                    reasons = {}
                q = str(case["query_text"])
                gold = str(case["gold"])
                stale = list(case.get("stale_or_false_values") or [])
                if idx_docs:
                    claims, st = ext.extract_batch(idx_docs, batch_size=max(1, len(idx_docs)), max_new_tokens=MAX_NEW_TOKENS)
                    extract_ms = float(st["wall_ms"])
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    d_vecs = enc([d.text for d in idx_docs])
                    s_vecs = enc([f"{cl.entity} {cl.slot}" for cl in claims])
                    exo = AuthorityDominatedExocortexV2(idx_docs, claims, d_vecs, s_vecs)
                    bm = FastBM25Index([d.doc_id for d in idx_docs], [d.text for d in idx_docs])
                    torch.cuda.synchronize()
                    write_ms = (time.perf_counter() - t1) * 1000.0
                    t2 = time.perf_counter()
                    qv = enc([q])[0]
                    hyb = _rrf_combine(bm.score_query(q), l2_normalize(d_vecs) @ qv)
                    sel = exo.select_context_for_query(hyb, q, top_k=5)
                    torch.cuda.synchronize()
                    read_ms = (time.perf_counter() - t2) * 1000.0
                    t3 = time.perf_counter()
                    pred = rule_reader(sel, exo)
                    reader_ms = (time.perf_counter() - t3) * 1000.0
                    status = str(sel["status"])
                    json_valid = int(st["json_valid_count"])
                else:
                    extract_ms = write_ms = read_ms = reader_ms = 0.0
                    pred, status, json_valid = "", "NO_TRIGGERED_DOCS", 0
                sc = score_d1_prose_prediction(pred, gold, stale)
                e2e = route_ms + extract_ms + write_ms + read_ms + reader_ms
                rows.append(
                    {
                        "case_id": str(case["case_id"]),
                        "regime": str(case["regime"]),
                        "domain": str(case["domain"]),
                        "n_docs": len(docs),
                        "n_indexed": len(idx_docs),
                        "trigger_reasons": reasons,
                        "status": status,
                        "pred": pred,
                        "gold": gold,
                        "json_valid_count": json_valid,
                        "route_ms": round(route_ms, 4),
                        "extract_ms": round(extract_ms, 3),
                        "write_ack_ms": round(write_ms, 3),
                        "read_ms": round(read_ms, 3),
                        "reader_ms": round(reader_ms, 4),
                        "e2e_ms": round(e2e, 3),
                        "frozen_d1_match": int(sc["frozen_d1_match"]),
                        "strict_exact_match": int(sc["strict_exact_match"]),
                        "leaked_stale_value": bool(sc["leaked_stale_value"]),
                    }
                )
            n_correct = sum(r["frozen_d1_match"] for r in rows)
            arms[arm] = {
                "n_cases": len(rows),
                "frozen_d1_correct": n_correct,
                "frozen_d1_accuracy": round(n_correct / len(rows), 4),
                "strict_exact_correct": sum(r["strict_exact_match"] for r in rows),
                "leaked_stale": sum(int(r["leaked_stale_value"]) for r in rows),
                "docs_indexed_total": sum(r["n_indexed"] for r in rows),
                "docs_total": sum(r["n_docs"] for r in rows),
                "threads_triggered": sum(1 for r in rows if r["n_indexed"] > 0),
                "latency_ms": {
                    k: _summ([r[k] for r in rows])
                    for k in ("e2e_ms", "route_ms", "extract_ms", "write_ack_ms", "read_ms", "reader_ms")
                },
                "per_case": rows,
            }
            print(
                f"[suite-a] {arm}: acc={arms[arm]['frozen_d1_accuracy']} ({n_correct}/60) "
                f"e2e_mean={arms[arm]['latency_ms']['e2e_ms']['mean']}ms indexed={arms[arm]['docs_indexed_total']}",
                flush=True,
            )
    del enc, bge
    gc.collect()
    torch.cuda.empty_cache()
    out = {
        "stage": "suite-a",
        "generated_utc": _now(),
        "gpu_at_start": gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "suite_a_sha256": sha,
        "extractor_merged_dir_scratch": merged_dir,
        "extractor_warmup_ms": warm_ms,
        "router": {"stems": list(AMENDMENT_CONFLICT_STEMS), "authority_min": AUTHORITY_TRIGGER_MIN},
        "arms": arms,
    }
    write_json_atomic(_stage_path("suite_a"), out)
    return out


# ---------------------------------------------------------------------------
# Stage: live62 (GPU)
# ---------------------------------------------------------------------------


def stage_live62(port: int) -> Dict[str, Any]:
    gpu = require_gpu_free()
    import torch

    from scripts.run_colab_l4_w22b_b62_live_vllm_workers import MicroBatchedCausalServer, start_openai_server

    gpu_name = torch.cuda.get_device_name(0)
    cohort = f"local:{gpu_name}"
    engine = MicroBatchedCausalServer(BASE_MODEL, device="cuda", max_batch_size=8, batch_wait_ms=3.0)
    srv, _ = start_openai_server(engine, BASE_MODEL, port)
    url = f"http://127.0.0.1:{port}/v1"
    grid = {
        "mode": "run-live-workers",
        "pool_sizes": (4,),
        "schedules": ("S1_steady",),
        "branches": TWIN_BRANCHES,
        "event_regimes": LIVE_REGIMES,
    }
    out: Dict[str, Any] = {
        "stage": "live62",
        "generated_utc": _now(),
        "hardware_cohort": cohort,
        "gpu_at_start": gpu,
        "model": BASE_MODEL,
        "server": "MicroBatchedCausalServer(bf16, max_batch_size=8, batch_wait_ms=3.0)",
        "job_sla_deadline_ms": JOB_SLA_DEADLINE_MS,
        "grid": {k: list(v) if isinstance(v, tuple) else v for k, v in grid.items()},
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    try:
        # Untimed warmup: a few direct generations (batch 1 and batch 4 shapes).
        from concurrent.futures import ThreadPoolExecutor

        msgs = [
            {"role": "system", "content": 'Return ONLY JSON with keys "field_value" and "cited_source_id".'},
            {"role": "user", "content": "Visible: - [M01_src_memo]: delivery_date is May 30, 2026. Extract JSON."},
        ]
        for workers in (4, 1, 4, 1):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda _i: engine.generate_chat(msgs, max_tokens=64), range(workers)))
        for arm, backend in (
            ("plain_live", LiveVLLMWorkerBackend(vllm_url=url, model=BASE_MODEL, timeout_s=15.0)),
            ("selective_live", SelectiveTriggerLiveWorkerBackend(vllm_url=url, model=BASE_MODEL, timeout_s=15.0)),
        ):
            t0 = time.perf_counter()
            man = _run_62(backend, **grid)
            wall = round(time.perf_counter() - t0, 3)
            sha = write_json_atomic(SCRATCH / f"live62_{arm}_manifest.json", man)
            s = summarize_manifest(man, getattr(backend, "router", None))
            s["wall_seconds"] = wall
            s["full_manifest_scratch_sha256"] = sha
            s["g3"] = g3_from_cells(s["cells"])
            out[arm] = s
            print(f"[live62] {arm}: status={s['status']} calls={s['live_model_calls']} wall={wall}s g3={s['g3']}", flush=True)
            write_json_atomic(_stage_path("live62"), out)
    finally:
        srv.shutdown()
        engine.shutdown()
    write_json_atomic(_stage_path("live62"), out)
    return out


# ---------------------------------------------------------------------------
# Stage: aggregate
# ---------------------------------------------------------------------------


def _load(name: str) -> Optional[Dict[str, Any]]:
    p = _stage_path(name)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def decide(suite_a: Optional[Dict[str, Any]], live: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    gates: Dict[str, Any] = {}
    if suite_a is None:
        gates["G1"] = {"measured": False, "reason": "suite-a stage not run"}
        gates["G2"] = {"measured": False, "reason": "suite-a stage not run"}
    else:
        sel = suite_a["arms"]["SELECTIVE"]
        full = suite_a["arms"]["FULL"]
        mean_e2e = float(sel["latency_ms"]["e2e_ms"]["mean"])
        gates["G1"] = {
            "measured": True,
            "selective_mean_e2e_ms": mean_e2e,
            "threshold_ms": 1500.0,
            "stretch_lt_500ms": mean_e2e < 500.0,
            "below_recorded_bare70b_k5": mean_e2e < RECORDED["bare_70b_mean_query_ms_k5"],
            "pass": mean_e2e <= 1500.0,
        }
        sc, fc = int(sel["frozen_d1_correct"]), int(full["frozen_d1_correct"])
        gates["G2"] = {
            "measured": True,
            "selective_correct": sc,
            "full_correct": fc,
            "abs_floor": 57,
            "pass": bool(sc >= 57 and sc >= fc - 1),
        }
    if live is None or "selective_live" not in live:
        gates["G3"] = {"measured": False, "reason": "live62 stage not run (GPU not available)"}
    else:
        g3 = live["selective_live"]["g3"]
        gates["G3"] = {"measured": True, **g3}
    measured = all(g["measured"] for g in gates.values())
    if not measured:
        verdict = "INCONCLUSIVE"
        if any(g["measured"] and not g["pass"] for g in gates.values()):
            verdict = "KILL"
    else:
        verdict = "ADVANCE" if all(g["pass"] for g in gates.values()) else "KILL"
    return {"verdict": verdict, "gates": gates}


def stage_aggregate() -> Dict[str, Any]:
    cpu, dev, train, suite_a, live = (_load(n) for n in ("cpu_replay", "dev_recall", "train", "suite_a", "live62"))
    dec = decide(suite_a, live)
    verifier = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verifiers.d" / "62_verifier.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    try:
        v_json = json.loads(verifier.stdout)
        v_summary = {"passed": v_json.get("passed"), "status": v_json.get("status")}
    except json.JSONDecodeError:
        v_summary = {"stdout_tail": verifier.stdout[-500:], "stderr_tail": verifier.stderr[-500:]}
    suite_a_compact = None
    if suite_a is not None:
        suite_a_compact = {k: v for k, v in suite_a.items() if k != "arms"}
        suite_a_compact["arms"] = {a: {k: v for k, v in d.items() if k != "per_case"} for a, d in suite_a["arms"].items()}
    live_compact = None
    if live is not None:
        live_compact = {k: v for k, v in live.items() if k not in ("plain_live", "selective_live")}
        for arm in ("plain_live", "selective_live"):
            if arm in live:
                live_compact[arm] = {k: v for k, v in live[arm].items() if k != "cells"}
                live_compact[arm]["cells_sham_zeromu"] = [
                    c
                    for c in live[arm]["cells"]
                    if c["event_regime"] == "sham_subtask_reopen_interrupt" and c["system_id"] == "ZeroMu+EK5_v3"
                ]
    res = {
        "schema_version": "pv7_w23b_w22_15_results_v1",
        "item": "W22-15",
        "generated_utc": _now(),
        "git_sha": current_git_sha(REPO_ROOT),
        "preregistration": "artifacts/pv7_w23b_w22_15/PREREGISTRATION.md",
        "decision": dec,
        "recorded_comparators": RECORDED,
        "suite_a": suite_a_compact,
        "live62": live_compact,
        "cpu_replay": cpu,
        "dev_recall": dev,
        "train": train,
        "verifier_62": {
            "exit_code": verifier.returncode,
            "applicability": "verifies the committed W20 #62 bundle artifacts only; it does not read W22-15 outputs",
            **v_summary,
        },
        "stage_files": sorted(p.name for p in OUT_DIR.glob("stage_*.json")),
    }
    write_json_atomic(OUT_DIR / "w22_15_results.json", res)
    print(f"[aggregate] verdict={dec['verdict']} gates={json.dumps(dec['gates'])}", flush=True)
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        required=True,
        choices=("cpu-replay", "dev-recall", "train", "suite-a", "live62", "aggregate", "gpu-status"),
    )
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if args.stage == "gpu-status":
        print(json.dumps(gpu_status()))
    elif args.stage == "cpu-replay":
        stage_cpu_replay()
    elif args.stage == "dev-recall":
        stage_dev_recall()
    elif args.stage == "train":
        stage_train()
    elif args.stage == "suite-a":
        stage_suite_a()
    elif args.stage == "live62":
        stage_live62(args.port)
    else:
        stage_aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
