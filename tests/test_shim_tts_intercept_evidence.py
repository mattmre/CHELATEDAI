"""TTS intercept evidence script smoke (SHIM-CD-05)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "record_shim_tts_intercept_evidence.py"


class TestShimTtsInterceptEvidence(unittest.TestCase):
    def test_record_tts_intercept_script_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as evidence_dir:
            env = os.environ.copy()
            env["CHELATED_SHIM_EVIDENCE_DIR"] = evidence_dir
            result = subprocess.run(
                [sys.executable, str(SCRIPT)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            matches = list(Path(evidence_dir).glob("bhs_shim_evidence_tts_intercept_*.json"))
            self.assertTrue(matches, "expected tts intercept evidence json")
            data = json.loads(matches[-1].read_text(encoding="utf-8"))
            self.assertEqual(len(data["runs"]), 2)
            self.assertTrue(data["runs"][1]["collector"]["probe_hit"])


if __name__ == "__main__":
    unittest.main()
