"""SHIM-CD-01: AntigravityEngine.get_chelated_vector evidence script smoke."""

from __future__ import annotations

import json
import subprocess
import tempfile
import os
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "record_shim_engine_embed_evidence.py"


class TestShimEngineEmbedEvidence(unittest.TestCase):
    def test_record_engine_embed_script_writes_json(self) -> None:
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

            matches = sorted(Path(evidence_dir).glob("bhs_shim_evidence_engine_embed_*.json"))
            self.assertTrue(matches, "expected engine embed evidence json")
            data = json.loads(matches[-1].read_text(encoding="utf-8"))

            self.assertEqual(data.get("seam"), "AntigravityEngine.get_chelated_vector")
            self.assertIn(data.get("mode"), {"real", "stubbed", "stubbed_fallback"})
            self.assertTrue(data.get("promoted_sip_applied") is not None)
            self.assertIn("vector_norm", data)
            self.assertIsInstance(data.get("vector_norm"), float)
            self.assertIsInstance(data.get("skip_for_env_limits"), bool)


if __name__ == "__main__":
    unittest.main()
