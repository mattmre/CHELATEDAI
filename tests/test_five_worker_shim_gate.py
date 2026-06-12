"""Five-worker shim gate should always emit a valid summary artifact."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_five_worker_shim_gate.py"
WORKER_SUMMARY = (
    ROOT / "artifacts" / "bhs_10min_loop" / "workers" / "five_worker_summary.json"
)


class TestFiveWorkerShimGate(unittest.TestCase):
    def test_five_worker_gate_runs_and_all_pass(self) -> None:
        with tempfile.TemporaryDirectory() as evidence_dir:
            worker_loop_dir = Path(evidence_dir) / "bhs_10min_loop" / "workers"
            env = os.environ.copy()
            env["CHELATED_SHIM_EVIDENCE_DIR"] = str(Path(evidence_dir) / "bhs_10min_loop")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            summary_path = worker_loop_dir / "five_worker_summary.json"
            self.assertTrue(summary_path.exists(), "five_worker_summary.json not written")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary.get("all_passed"))
            self.assertEqual(summary.get("worker_count"), 5)
            results = summary.get("results") or []
            self.assertEqual(len(results), 5)
            self.assertTrue(all(isinstance(r.get("exit_code"), int) for r in results))


if __name__ == "__main__":
    unittest.main()
