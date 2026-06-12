"""BHS: promoted ShimRegistry probe at VectorSteerer seam (env-guarded)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from tts_pipeline import VectorSteerer


class TestShimPromotedProbe(unittest.TestCase):
    def test_promoted_probe_in_metadata_when_both_envs(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"CHELATED_SHIM_RESEARCH": "1", "CHELATED_SHIM_PROMOTED": "1"},
            clear=False,
        ):
            try:
                import shim_node_promoted  # noqa: F401
            except ImportError:
                self.skipTest("run scripts/promote_shim_primitives.py first")

            steerer = VectorSteerer()
            v = np.array([1.0, 0.0, 0.0], dtype=float)
            _, meta = steerer.steer(v)
            probe = meta.get("promoted_registry_probe")
            self.assertIsNotNone(probe)
            self.assertTrue(probe.get("promoted_shim_registry"))
            self.assertTrue(probe.get("probe_shim_registered"))

    def test_no_promoted_probe_without_promoted_env(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "CHELATED_SHIM_PROMOTED"}
        env["CHELATED_SHIM_RESEARCH"] = "1"
        with mock.patch.dict(os.environ, env, clear=True):
            steerer = VectorSteerer()
            v = np.array([1.0, 0.0], dtype=float)
            _, meta = steerer.steer(v)
            self.assertNotIn("promoted_registry_probe", meta)


if __name__ == "__main__":
    unittest.main()