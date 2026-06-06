"""BHS: promoted SIP apply_shim_cascade at VectorSteerer.steer."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from chelated_shim_research import promoted_sip_apply
from tts_pipeline import VectorSteerer


class TestPromotedSipApply(unittest.TestCase):
    def test_apply_changes_vector_when_both_envs(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"CHELATED_SHIM_RESEARCH": "1", "CHELATED_SHIM_PROMOTED": "1"},
            clear=False,
        ):
            try:
                import shim_node_promoted  # noqa: F401
            except ImportError:
                self.skipTest("run scripts/promote_shim_primitives.py first")

            v = np.zeros(8, dtype=float)
            out, meta = promoted_sip_apply(v)
            self.assertIsNotNone(meta)
            self.assertTrue(meta.get("promoted_sip_applied"))
            self.assertGreater(float(np.linalg.norm(out - v)), 0.0)
            self.assertLessEqual(float(meta.get("composite_delta_norm", 1.0)), 0.05)

    def test_steerer_surfaces_promoted_sip_meta(self) -> None:
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
            v = np.zeros(8, dtype=float)
            _, meta = steerer.steer(v)
            apply_meta = meta.get("promoted_sip_apply")
            self.assertIsNotNone(apply_meta)
            self.assertTrue(apply_meta.get("promoted_sip_applied"))


if __name__ == "__main__":
    unittest.main()