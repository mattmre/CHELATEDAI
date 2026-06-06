"""SHIM-CD-05: run_inference + enable_tts evidence script smoke."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "record_shim_inference_evidence.py"


class TestShimInferenceEvidence(unittest.TestCase):
    def test_record_inference_script_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as evidence_dir:
            env = os.environ.copy()
            env["CHELATED_SHIM_EVIDENCE_DIR"] = evidence_dir
            result = subprocess.run(
                [sys.executable, str(SCRIPT)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            matches = list(Path(evidence_dir).glob("bhs_shim_evidence_inference_*.json"))
            self.assertTrue(matches, "expected inference evidence json")
            data = json.loads(matches[-1].read_text(encoding="utf-8"))
            self.assertTrue(data.get("tts_enabled"))
            meta = data.get("research_shim_meta") or {}
            self.assertTrue(meta.get("research_shim_guard"))


if __name__ == "__main__":
    unittest.main()
