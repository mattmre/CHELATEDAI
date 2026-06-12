"""Production-path shim evidence and script smoke."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chelated_shim_research import attach_research_meta, research_enabled
from tts_pipeline import VectorSteerer

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "record_shim_prod_evidence.py"


class TestShimProdEvidence(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)

    def test_attach_research_meta_adds_field(self) -> None:
        diag = attach_research_meta({"runtime": {}}, {"sip_seam": "test"})
        self.assertEqual(diag["research_shim"]["sip_seam"], "test")
        self.assertEqual(attach_research_meta({"a": 1}, None), {"a": 1})

    def test_record_script_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as evidence_dir:
            env = os.environ.copy()
            env["CHELATED_SHIM_EVIDENCE_DIR"] = evidence_dir
            result = subprocess.run(
                [sys.executable, str(SCRIPT)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            matches = list(Path(evidence_dir).glob("bhs_shim_evidence_prod_probe_*.json"))
            self.assertTrue(matches, "expected prod probe evidence json")
            data = json.loads(matches[-1].read_text(encoding="utf-8"))
            self.assertEqual(data["seam"], "VectorSteerer.steer")
            self.assertEqual(len(data["runs"]), 2)
            self.assertEqual(data["runs"][1]["stall_count"], 2)

    def test_research_disabled_by_default_in_process(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        self.assertFalse(research_enabled())
        steerer = VectorSteerer()
        _, meta = steerer.steer(np.ones(3))
        self.assertNotIn("research_shim_guard", meta)


if __name__ == "__main__":
    unittest.main()
