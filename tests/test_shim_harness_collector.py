"""SHIM-CD-04: collector for research probe metadata (prod helper + harness alias)."""

from __future__ import annotations

import os
import unittest

import numpy as np

from chelated_shim_research import collect_research_probe_from_tts_metadata
from tts_pipeline import VectorSteerer


class TestShimHarnessCollector(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)

    def test_collector_miss_when_meta_none(self) -> None:
        out = collect_research_probe_from_tts_metadata(None)
        self.assertFalse(out["probe_hit"])
        self.assertIn("reason", out)

    def test_collector_hits_guarded_steer_meta(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        _, meta = VectorSteerer().steer(np.ones(4))
        out = collect_research_probe_from_tts_metadata(meta)
        self.assertTrue(out["probe_hit"])
        self.assertEqual(out["seam"], "VectorSteerer.steer")
        self.assertGreaterEqual(out["probe_count"], 1)

    def test_collector_miss_without_guard_keys(self) -> None:
        steerer = VectorSteerer(enabled=False)
        _, meta = steerer.steer(np.array([1.0, 0.0]))
        out = collect_research_probe_from_tts_metadata(meta)
        self.assertFalse(out["probe_hit"])


if __name__ == "__main__":
    unittest.main()