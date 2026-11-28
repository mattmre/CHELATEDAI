#!/usr/bin/env python3
"""Runner for W22-11 (preregistered in artifacts/pv7_w23c_w22_11/PREREGISTRATION.md).

Runs the preregistered tiers through a process pool in interleaved chunks and stops starting new chunks at
the hard wall deadline. It writes per-chunk JSON to ``<out>/scratch/`` (gitignored) and writes
``<out>/w22_11_results.json``. CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import w22_11_wfs_deletion_stress as w  # noqa: E402

PINNED = {
    "ek5_v3_stratified.py": "b54dadb51e1f54576f39e22808fa9c7cd9209fe3de86bf62139eded10b13dfd0",
    "formal/run_conformance.py": "49c56c3344d635e026284f9f44eb7e62d19a95339924ea6a3c84bd37d2ce271e",
    "chelated_pv7_spine/ek5_reason_canonicalizer.py": "b8440b62b8c20bf28d35a8c961ab671d3ea8ce90c45fd916c53e8b5a3d8dc432",
}
ANCHORS = {
    "A0": {"total": 153, "by_field": {"pos_cyc": 104, "reason": 49}, "checkpoints": 7722},
    "A1": {"total": 825, "by_field": {"pos_cyc": 548, "reason": 277}, "checkpoints": 13881},
}
TARGET_MUTATION_CHECKPOINTS = 1_000_000
DEADLINE_SECONDS = 40 * 60


def preregistered_tiers() -> List[w.TierConfig]:
    T = w.TierConfig
    return [
        T("A0", "baseline", 11, 6, 1000, 0),
        T("A1", "baseline", 11, 12, 1000, 0),
        T("T1", "baseline", 11, 24, 30000, 1),
        T("T2", "stress", 32, 24, 12000, 2, first_nodes=(8, 12), new_nodes_per_step=(0, 3), scc_size=(3, 16)),
        T("T3", "stress", 64, 24, 6000, 3, first_nodes=(16, 24), new_nodes_per_step=(1, 4),
          adds_first=(8, 16), adds_per_step=(2, 8), scc_size=(4, 32)),
        T("T4", "stress", 128, 32, 2500, 4, first_nodes=(32, 48), new_nodes_per_step=(2, 6),
          adds_first=(16, 32), adds_per_step=(4, 12), scc_size=(8, 64)),
        T("T5", "stress", 256, 40, 400, 5, first_nodes=(64, 96), new_nodes_per_step=(2, 8),
          adds_first=(32, 64), adds_per_step=(6, 16), scc_size=(16, 128)),
    ]


CHUNK = {"T4": 25, "T5": 10}


def build_chunks(tiers: List[w.TierConfig], scale: float) -> List[Tuple[Dict[str, Any], int, int]]:
    per_tier: List[List[Tuple[Dict[str, Any], int, int]]] = []
    for cfg in tiers:
        n = cfg.sequences if scale >= 1.0 else max(1, int(cfg.sequences * scale))
        size = CHUNK.get(cfg.name, 50)
        per_tier.append([(w.tier_dict(cfg), lo, min(n, lo + size)) for lo in range(0, n, size)])
    out: List[Tuple[Dict[str, Any], int, int]] = []
    # anchors first (small, known), then the rest interleaved round-robin
    for lst in per_tier[:2]:
        out.extend(lst)
    rest = per_tier[2:]
    i = 0
    while any(i < len(lst) for lst in rest):
        for lst in rest:
            if i < len(lst):
                out.append(lst[i])
        i += 1
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verdict(results: Mapping[str, Any]) -> Dict[str, Any]:
    tot = results["totals"]
    s = tot["summary"]
    live = tot.get("liveness", {})
    live_ok = (
        live.get("canonical_injected", 0) > 0
        and live.get("canonical_injected") == live.get("canonical_detected")
        and live.get("semantic_injected", 0) > 0
        and live.get("semantic_injected") == live.get("semantic_detected")
    )
    hashes_ok = all(v["ok"] for v in results["pinned_sources"].values())
    canonical_total = (
        s["canonical_inc_vs_fresh_total"]
        + s["canonical_fresh_vs_fresh_total"]
        + s["canonical_vs_fresh_raw_total"]
        + s["canonical_pos_cyc_vs_snapshot_scc"]
    )
    mut = tot.get("checkpoints_mutation", 0)
    if not live_ok or not hashes_ok:
        v = "INVALID_RUN"
    elif s["semantic_total"] > 0:
        v = "BLOCKING_BUG"
    elif canonical_total > 0:
        v = "CANONICALITY_NOT_ESTABLISHED"
    elif mut >= TARGET_MUTATION_CHECKPOINTS:
        v = "CLEAN_1M_CANONICAL_FROZEN"
    else:
        v = "CLEAN_PARTIAL_%d" % mut
    return {
        "verdict": v,
        "liveness_ok": live_ok,
        "pinned_hashes_ok": hashes_ok,
        "semantic_total": s["semantic_total"],
        "canonical_gate_total": canonical_total,
        "mutation_checkpoints": mut,
        "all_checkpoints": tot.get("checkpoints", 0),
        "target_mutation_checkpoints": TARGET_MUTATION_CHECKPOINTS,
        "raw_witness_inc_vs_fresh_total": s["raw_witness_inc_vs_fresh_total"],
    }


def anchor_checks(per_tier: Mapping[str, Any]) -> Dict[str, Any]:
    out = {}
    for name, exp in ANCHORS.items():
        got = per_tier.get(name)
        if not got:
            out[name] = {"ran": False}
            continue
        raw = got["raw_witness_inc_vs_fresh"]
        out[name] = {
            "ran": True,
            "expected": exp,
            "got": {"total": sum(raw.values()), "by_field": raw, "checkpoints": got["checkpoints"]},
            "reproduced": sum(raw.values()) == exp["total"] and raw == exp["by_field"] and got["checkpoints"] == exp["checkpoints"],
        }
    return out


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "pv7_w23c_w22_11"))
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--scale", type=float, default=1.0, help="<1.0 shrinks every tier (smoke only; not the prereg run)")
    ap.add_argument("--deadline-seconds", type=int, default=DEADLINE_SECONDS)
    ap.add_argument("--results-name", default="w22_11_results.json")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    scratch = out / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    tiers = preregistered_tiers()
    chunks = build_chunks(tiers, args.scale)
    t0 = time.time()
    per_tier: Dict[str, Dict[str, Any]] = {}
    completed = 0
    stopped_early = False
    with mp.Pool(args.workers) as pool:
        pending = []
        it = iter(chunks)
        submitted = 0
        # keep a bounded queue so the deadline actually stops new work
        def submit_next() -> bool:
            nonlocal submitted
            if time.time() - t0 > args.deadline_seconds:
                return False
            try:
                ch = next(it)
            except StopIteration:
                return False
            pending.append((ch, pool.apply_async(w.run_chunk, (ch,))))
            submitted += 1
            return True

        for _ in range(args.workers * 2):
            if not submit_next():
                break
        while pending:
            still = []
            for ch, res in pending:
                if res.ready():
                    part = res.get()
                    name = ch[0]["name"]
                    (scratch / ("%s_%06d_%06d.json" % (name, ch[1], ch[2]))).write_text(
                        json.dumps(part, sort_keys=True), encoding="utf-8"
                    )
                    w.merge_counts(per_tier.setdefault(name, {}), part)
                    completed += 1
                else:
                    still.append((ch, res))
            pending = still
            while len(pending) < args.workers * 2:
                if not submit_next():
                    break
            if pending:
                time.sleep(0.2)
        if submitted < len(chunks):
            stopped_early = True
    elapsed = time.time() - t0

    totals: Dict[str, Any] = {}
    for name in per_tier:
        w.merge_counts(totals, {k: v for k, v in per_tier[name].items() if k != "samples"})
    for name, c in per_tier.items():
        c["summary"] = w.summarize(c)
    totals["summary"] = w.summarize(totals)

    pinned = {}
    for rel, exp in PINNED.items():
        got = sha256_file(REPO_ROOT / rel)
        pinned[rel] = {"expected": exp, "got": got, "ok": got == exp}
    results: Dict[str, Any] = {
        "schema_version": 1,
        "item": "W22-11",
        "preregistration": "artifacts/pv7_w23c_w22_11/PREREGISTRATION.md",
        "scale": args.scale,
        "workers": args.workers,
        "host": {"cpu_count": os.cpu_count(), "python": platform.python_version(), "platform": platform.platform()},
        "elapsed_seconds": round(elapsed, 1),
        "deadline_seconds": args.deadline_seconds,
        "chunks_planned": len(chunks),
        "chunks_completed": completed,
        "stopped_early_at_deadline": stopped_early,
        "tiers": {cfg.name: w.tier_dict(cfg) for cfg in tiers},
        "pinned_sources": pinned,
        "per_tier": per_tier,
        "totals": totals,
        "anchors": anchor_checks(per_tier),
    }
    results["verdict"] = verdict(results)
    if results["verdict"]["verdict"] == "CLEAN_1M_CANONICAL_FROZEN":
        results["frozen_invariant"] = {
            "id": "CW-1",
            "definition": 'sha256(json.dumps(export_canonical_ek5_audit_trace(S).to_dict(), sort_keys=True, separators=(",",":")))'
            " is equal for the incremental store and every from-scratch rebuild of the same visible log under any insertion order",
            "canonicalizer": "chelated_pv7_spine/ek5_reason_canonicalizer.py",
            "canonicalizer_sha256": PINNED["chelated_pv7_spine/ek5_reason_canonicalizer.py"],
        }
    (out / args.results_name).write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": results["verdict"], "anchors": {k: v.get("reproduced") for k, v in results["anchors"].items()},
                      "elapsed_seconds": results["elapsed_seconds"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
