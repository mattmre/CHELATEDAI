"""Tests for scripts/smoke_pipeline.py ceiling tier (CD-001 closure).

Mixed strategy:
  - Unit tests mock the heavy deps via sys.modules injection so they run fast
    on minimal envs (no torch / sentence-transformers required).
  - One non-mocked integration test actually constructs AntigravityEngine
    with the real local SentenceTransformer + in-memory Qdrant and asserts
    the ceiling tier returns 0. It is skipped only when the heavy deps are
    genuinely absent (which is the same honest-skip path the smoke pipeline
    itself uses).

unittest only — CI does not install pytest (see CLAUDE.md Test Conventions).
"""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

# Make repo root and scripts/ importable without polluting test order.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import smoke_pipeline  # noqa: E402  (after sys.path setup)


try:
    import torch  # noqa: F401
    import sentence_transformers  # noqa: F401
    import qdrant_client  # noqa: F401
    import numpy as np  # noqa: F401
    from antigravity_engine import AntigravityEngine  # noqa: F401

    _HEAVY_DEPS_AVAILABLE = True
except ImportError:
    _HEAVY_DEPS_AVAILABLE = False


class TestCeilingSmokeSkipSentinel(unittest.TestCase):
    """Verify the ceiling tier returns the SKIP sentinel when a heavy dep is missing.

    We simulate a missing dep by injecting an ImportError-raising stub into
    sys.modules. This proves the skip path is genuine (not silently swallowed)
    and that the SKIP message reaches stdout.
    """

    def test_skips_when_torch_missing(self):
        # builtins.__import__ swap is the cleanest way to simulate a missing
        # top-level import without breaking already-imported modules.
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                raise ImportError("simulated: torch not installed")
            return real_import(name, globals, locals, fromlist, level)

        buf = io.StringIO()
        with patch("builtins.__import__", side_effect=fake_import):
            with redirect_stdout(buf):
                status = smoke_pipeline.run_ceiling_smoke()

        self.assertEqual(status, smoke_pipeline.CEILING_SKIPPED)
        output = buf.getvalue()
        self.assertIn("SKIP", output)
        self.assertIn("torch", output)

    def test_skip_sentinel_is_not_pass_or_fail(self):
        # Sanity: the SKIP sentinel must be distinct from the success (0)
        # and generic-failure (1) codes so main() can disambiguate.
        self.assertNotEqual(smoke_pipeline.CEILING_SKIPPED, 0)
        self.assertNotEqual(smoke_pipeline.CEILING_SKIPPED, 1)

    def test_main_translates_skip_to_zero_exit(self):
        # When the ceiling tier honestly skips, main() must still exit 0 —
        # the skip is honest, not a failure.
        buf = io.StringIO()
        with patch.object(smoke_pipeline, "run_floor_smoke", return_value=0):
            with patch.object(
                smoke_pipeline,
                "run_ceiling_smoke",
                return_value=smoke_pipeline.CEILING_SKIPPED,
            ):
                with redirect_stdout(buf):
                    exit_code = smoke_pipeline.main()
        self.assertEqual(exit_code, 0)
        output = buf.getvalue()
        self.assertIn("ceiling tier honestly skipped", output)

    def test_main_propagates_ceiling_failure(self):
        # When the ceiling tier returns a real failure (1), main() MUST
        # propagate it as a non-zero exit. Anything else is L4.
        buf = io.StringIO()
        with patch.object(smoke_pipeline, "run_floor_smoke", return_value=0):
            with patch.object(smoke_pipeline, "run_ceiling_smoke", return_value=1):
                with redirect_stdout(buf):
                    exit_code = smoke_pipeline.main()
        self.assertEqual(exit_code, 1)


@unittest.skipUnless(
    _HEAVY_DEPS_AVAILABLE,
    "Integration test requires torch + sentence-transformers + qdrant_client "
    "(installed in CI per CLAUDE.md; honest skip otherwise).",
)
class TestCeilingSmokeIntegration(unittest.TestCase):
    """End-to-end integration: actually run the ceiling tier with real deps.

    Slow (~5-15s on cold cache because it downloads/loads all-MiniLM-L6-v2).
    Gated to environments that match the CI matrix.
    """

    def test_ceiling_smoke_passes_end_to_end(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            status = smoke_pipeline.run_ceiling_smoke()
        output = buf.getvalue()
        self.assertEqual(
            status,
            0,
            msg=f"Ceiling smoke did not return 0. Output:\n{output}",
        )
        # Sanity-check the OK markers landed.
        self.assertIn("engine built", output)
        self.assertIn("get_chelated_vector PASS", output)
        self.assertIn("embed batch PASS", output)

    def test_full_main_passes_end_to_end(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = smoke_pipeline.main()
        output = buf.getvalue()
        self.assertEqual(
            exit_code,
            0,
            msg=f"smoke_pipeline.main() exit={exit_code}. Output:\n{output}",
        )
        self.assertIn("floor + ceiling tiers", output)


class TestBhsValidatorCeilingDelegation(unittest.TestCase):
    """Verify scripts/bhs_validator.run_smoke_pipeline(CEILING) reaches our script.

    bhs_validator delegates to scripts/smoke_pipeline.py via subprocess; the two
    paths must not silently diverge. This test confirms the subprocess plumbing
    by patching subprocess.run and checking the right script is invoked.
    """

    def test_ceiling_invocation_targets_smoke_pipeline_py(self):
        from bhs_validator import HonestyTier, run_smoke_pipeline

        class _FakeResult:
            returncode = 0
            stdout = "SMOKE_PIPELINE PASS (mocked)"
            stderr = ""

        with patch("subprocess.run", return_value=_FakeResult()) as mock_run:
            ok = run_smoke_pipeline(HonestyTier.CEILING)

        self.assertTrue(ok)
        self.assertTrue(mock_run.called)
        call_args = mock_run.call_args[0][0]
        # First arg is sys.executable, second is the path to smoke_pipeline.py.
        self.assertEqual(len(call_args), 2)
        self.assertTrue(
            call_args[1].endswith(os.path.join("scripts", "smoke_pipeline.py"))
            or call_args[1].endswith("scripts/smoke_pipeline.py")
        )

    def test_ceiling_propagates_subprocess_failure(self):
        from bhs_validator import HonestyTier, run_smoke_pipeline

        class _FailResult:
            returncode = 1
            stdout = "SMOKE_PIPELINE FAIL: simulated"
            stderr = ""

        with patch("subprocess.run", return_value=_FailResult()):
            ok = run_smoke_pipeline(HonestyTier.CEILING)

        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
