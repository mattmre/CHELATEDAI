#!/usr/bin/env python3
"""Five parallel worker gate for SHIM-CD-06 (measurable 5-way execution).

Each worker runs a disjoint unittest subset and writes JSON evidence under
artifacts/bhs_10min_loop/workers/.
"""

from __future__ import annotations

import json
import subprocess
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(
    os.environ.get(
        "CHELATED_SHIM_EVIDENCE_DIR",
        str((ROOT / "artifacts" / "bhs_10min_loop").resolve()),
    )
) / "workers"

WORKERS = {
    "A": ["tests.test_chelated_shim_research"],
    "B": ["tests.test_shim_vector_steerer_research"],
    "C": ["tests.test_shim_inference_evidence"],
    "D": ["tests.test_shim_promoted_probe", "tests.test_shim_promoted_sip_apply"],
    "E": ["tests.test_shim_harness_collector", "tests.test_shim_tts_intercept_evidence"],
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_worker(role: str, modules: list[str]) -> dict:
    cmd = [sys.executable, "-m", "unittest", "-v", *modules]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
    result = {
        "role": role,
        "timestamp": _utc(),
        "modules": modules,
        "exit_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-1500:],
        "stderr_tail": (proc.stderr or "")[-500:],
        "bhs_target": 100,
        "worker_mandate": "Iteratively improve code toward BHS 100/100; no doc-only output.",
        "env": {
            "CHELATED_SHIM_EVIDENCE_DIR": os.environ.get("CHELATED_SHIM_EVIDENCE_DIR"),
        },
    }
    path = OUT_DIR / f"worker_{role}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_run_worker, role, mods): role for role, mods in WORKERS.items()
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    summary = {
        "timestamp": _utc(),
        "worker_count": len(WORKERS),
        "all_passed": all(r["exit_code"] == 0 for r in results),
        "results": sorted(results, key=lambda x: x["role"]),
    }
    summary_path = OUT_DIR / "five_worker_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
