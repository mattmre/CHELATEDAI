"""SHIM-CD-01: guarded VectorSteerer.steer research SIP (default-off)."""

from __future__ import annotations

import os
import unittest

import numpy as np

from tts_pipeline import VectorSteerer


class TestShimVectorSteererResearch(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)

    def test_default_path_unchanged_without_env(self) -> None:
        steerer = VectorSteerer(max_strength=0.3, enabled=True)
        v = np.ones(8, dtype=float)
        out, meta = steerer.steer(v)
        self.assertTrue(np.allclose(out, v))
        self.assertFalse(meta["was_steered"])
        self.assertEqual(meta["signals_applied"], 0)
        self.assertNotIn("research_shim_guard", meta)

    def test_research_guard_no_signals_returns_metadata(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        steerer = VectorSteerer()
        v = np.array([1.0, 0.0, 0.0])
        out, meta = steerer.steer(v)
        self.assertTrue(np.allclose(out, v))
        self.assertTrue(meta.get("research_shim_guard"))
        self.assertEqual(meta.get("sip_seam"), "VectorSteerer.steer")
        self.assertEqual(meta.get("research_stall_count"), 1)
        _, meta2 = steerer.steer(v)
        self.assertEqual(meta2.get("research_stall_count"), 2)

    def test_research_guard_resets_stall_when_signals_present(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        from tts_pipeline import SteeringSignal

        steerer = VectorSteerer(max_strength=0.5)
        v = np.zeros(4, dtype=float)
        steerer.steer(v)
        steerer.steer(v)
        self.assertEqual(steerer._research_stall_count, 2)
        steerer.add_signal(
            SteeringSignal(direction=np.array([1.0, 0.0, 0.0, 0.0]), strength=0.1, source="t")
        )
        _, meta = steerer.steer(v)
        self.assertTrue(meta["was_steered"])
        self.assertEqual(steerer._research_stall_count, 0)
        self.assertTrue(meta.get("research_shim_guard"))

    def test_research_activation_record_stored_on_probe(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        from tts_pipeline import SteeringSignal

        try:
            import shim_node_promoted  # noqa: F401
        except ImportError:
            self.skipTest("run scripts/promote_shim_primitives.py first")

        steerer = VectorSteerer()
        steerer.clear_signals()

        v = np.array([1.0, 0.0, 0.0], dtype=float)
        _, meta = steerer.steer(v)

        self.assertTrue(meta.get("research_shim_guard"))
        activation_record = meta.get("research_activation_record")
        self.assertIsInstance(activation_record, dict)
        self.assertEqual(activation_record.get("seam"), "VectorSteerer.steer")
        self.assertEqual(activation_record.get("signals_count"), 0)
        self.assertTrue(activation_record.get("probe_activated"))
        self.assertEqual(activation_record.get("probe_count"), 1)

        steerer.add_signal(
            SteeringSignal(
                direction=np.array([1.0, 0.0, 0.0], dtype=float),
                strength=0.1,
                source="vectorsteerer_activation_record",
            )
        )
        _, meta = steerer.steer(v)
        activation_record = meta.get("research_activation_record")
        self.assertIsInstance(activation_record, dict)
        self.assertEqual(activation_record.get("probe_count"), 2)

    def test_clear_signals_resets_research_stall_state(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        steerer = VectorSteerer()
        v = np.zeros(4, dtype=float)
        steerer.steer(v)
        steerer.steer(v)
        self.assertEqual(steerer._research_stall_count, 2)
        steerer.clear_signals()
        self.assertEqual(steerer._research_stall_count, 0)

    def test_research_guard_off_matches_disabled_empty_signals(self) -> None:
        steerer = VectorSteerer(enabled=False)
        v = np.array([2.0, 3.0])
        out_a, meta_a = steerer.steer(v)
        os.environ["CHELATED_SHIM_RESEARCH"] = "0"
        steerer2 = VectorSteerer(enabled=False)
        out_b, meta_b = steerer2.steer(v)
        self.assertTrue(np.allclose(out_a, out_b))
        self.assertEqual(meta_a["was_steered"], meta_b["was_steered"])


if __name__ == "__main__":
    unittest.main()
