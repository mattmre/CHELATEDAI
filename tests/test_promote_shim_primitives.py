"""Validate SHIM-CD-02 promotion path from research artifact to runtime import."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "promote_shim_primitives.py"


class TestPromoteShimPrimitives(unittest.TestCase):
    def test_promote_shim_primitives_smoke(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        promoted = ROOT / "shim_node_promoted.py"
        self.assertTrue(promoted.exists())
        out = (proc.stdout or "") + (proc.stderr or "")
        self.assertIn("shim_node_promoted", out)
        self.assertIn("shim_collapse_benchmark_extension_promoted", out)
        self.assertTrue((ROOT / "shim_collapse_benchmark_extension_promoted.py").exists())


if __name__ == "__main__":
    unittest.main()
