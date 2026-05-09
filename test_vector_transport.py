"""Tests for VectorTransport (≥30 tests)."""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vector_transport import (
    TransportConfig,
    TransportMode,
    TransportResult,
    TransportTarget,
    VectorTransport,
)

_PATCH = "vector_transport.get_logger"


def _make_transport(**cfg_kwargs) -> VectorTransport:
    config = TransportConfig(**cfg_kwargs)
    with patch(_PATCH, return_value=MagicMock()):
        return VectorTransport(config)


def _unit(arr: np.ndarray) -> np.ndarray:
    return arr / np.linalg.norm(arr)


class TestTransportMode(unittest.TestCase):
    def test_values(self):
        self.assertEqual(TransportMode.LINEAR, "linear")
        self.assertEqual(TransportMode.COSINE, "cosine")
        self.assertEqual(TransportMode.ADAPTIVE, "adaptive")

    def test_is_str_enum(self):
        self.assertIsInstance(TransportMode.LINEAR, str)


class TestTransportConfig(unittest.TestCase):
    def test_defaults(self):
        c = TransportConfig()
        self.assertEqual(c.mode, TransportMode.LINEAR)
        self.assertAlmostEqual(c.default_weight, 0.2)
        self.assertAlmostEqual(c.max_weight, 0.5)
        self.assertAlmostEqual(c.min_similarity_for_transport, 0.85)
        self.assertTrue(c.enabled)

    def test_custom_values(self):
        c = TransportConfig(
            mode=TransportMode.COSINE,
            default_weight=0.4,
            max_weight=0.8,
            min_similarity_for_transport=0.5,
            enabled=False,
        )
        self.assertEqual(c.mode, TransportMode.COSINE)
        self.assertAlmostEqual(c.default_weight, 0.4)
        self.assertFalse(c.enabled)


class TestTransportTarget(unittest.TestCase):
    def test_construction_full(self):
        c = np.array([1.0, 0.0, 0.0])
        t = TransportTarget(target_id="t1", centroid=c, description="desc", support=7)
        self.assertEqual(t.target_id, "t1")
        self.assertEqual(t.description, "desc")
        self.assertEqual(t.support, 7)

    def test_defaults(self):
        t = TransportTarget(target_id="t2", centroid=np.ones(4))
        self.assertEqual(t.description, "")
        self.assertEqual(t.support, 0)


class TestVectorTransportNoTargets(unittest.TestCase):
    def setUp(self):
        self.dim = 8
        self.tp = _make_transport()

    def test_no_targets_passthrough(self):
        v = np.random.rand(self.dim)
        r = self.tp.transport(v)
        self.assertFalse(r.was_transported)

    def test_no_targets_result_is_original(self):
        v = np.random.rand(self.dim)
        r = self.tp.transport(v)
        np.testing.assert_array_almost_equal(r.transported, v)

    def test_no_targets_weight_zero(self):
        v = np.random.rand(self.dim)
        r = self.tp.transport(v)
        self.assertAlmostEqual(r.weight_used, 0.0)

    def test_result_is_transport_result_type(self):
        r = self.tp.transport(np.ones(self.dim))
        self.assertIsInstance(r, TransportResult)

    def test_unregistered_target_id_passthrough(self):
        self.tp.add_target(TransportTarget("registered", np.ones(self.dim)))
        r = self.tp.transport(np.ones(self.dim), target_id="unknown")
        self.assertFalse(r.was_transported)


class TestVectorTransportAdd(unittest.TestCase):
    def test_add_target_stores(self):
        tp = _make_transport()
        tp.add_target(TransportTarget("t1", np.ones(4)))
        self.assertIn("t1", tp._targets)

    def test_multiple_targets(self):
        tp = _make_transport()
        for i in range(5):
            c = np.zeros(8)
            c[i] = 1.0
            tp.add_target(TransportTarget(f"t{i}", c))
        self.assertEqual(len(tp._targets), 5)


class TestVectorTransportLinear(unittest.TestCase):
    def setUp(self):
        self.dim = 8
        self.v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.tc = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _tp(self, **kw) -> VectorTransport:
        return _make_transport(**kw)

    def test_linear_convex_combination(self):
        tp = self._tp(mode=TransportMode.LINEAR, default_weight=0.3, min_similarity_for_transport=1.0)
        tp.add_target(TransportTarget("t1", self.tc))
        r = tp.transport(self.v, target_id="t1")
        expected = 0.7 * self.v + 0.3 * self.tc
        np.testing.assert_array_almost_equal(r.transported, expected)

    def test_linear_was_transported_true(self):
        tp = self._tp(min_similarity_for_transport=1.0)
        tp.add_target(TransportTarget("t1", self.tc))
        r = tp.transport(self.v, target_id="t1")
        self.assertTrue(r.was_transported)

    def test_weight_clamped_to_max(self):
        tp = self._tp(max_weight=0.3, min_similarity_for_transport=1.0)
        tp.add_target(TransportTarget("t1", self.tc))
        r = tp.transport(self.v, target_id="t1", weight=0.9)
        self.assertLessEqual(r.weight_used, 0.3)

    def test_default_weight_used_when_no_weight_arg(self):
        tp = self._tp(default_weight=0.25, min_similarity_for_transport=1.0)
        tp.add_target(TransportTarget("t1", self.tc))
        r = tp.transport(self.v, target_id="t1")
        self.assertAlmostEqual(r.weight_used, 0.25, places=6)

    def test_transport_skipped_when_already_close(self):
        # sim(v, tc) > min_similarity_for_transport ⟹ skip
        tp = self._tp(min_similarity_for_transport=0.3)
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tc = _unit(np.array([0.99, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))  # sim ≈ 0.99
        tp.add_target(TransportTarget("t1", tc))
        r = tp.transport(v, target_id="t1")
        self.assertFalse(r.was_transported)

    def test_mode_used_in_result(self):
        tp = self._tp(mode=TransportMode.LINEAR)
        r = tp.transport(self.v)
        self.assertEqual(r.mode_used, "linear")

    def test_after_transport_closer_to_target(self):
        tp = self._tp(default_weight=0.3, min_similarity_for_transport=1.0)
        tp.add_target(TransportTarget("t1", self.tc))
        r = tp.transport(self.v, target_id="t1")
        self.assertGreater(r.cosine_similarity_after, r.cosine_similarity_before)


class TestVectorTransportCosine(unittest.TestCase):
    def setUp(self):
        self.dim = 8
        self.v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.tc = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_cosine_result_is_unit_vector(self):
        tp = _make_transport(mode=TransportMode.COSINE, min_similarity_for_transport=1.0)
        tp.add_target(TransportTarget("t1", self.tc))
        r = tp.transport(self.v, target_id="t1")
        self.assertAlmostEqual(float(np.linalg.norm(r.transported)), 1.0, places=4)

    def test_cosine_was_transported(self):
        tp = _make_transport(mode=TransportMode.COSINE, min_similarity_for_transport=1.0)
        tp.add_target(TransportTarget("t1", self.tc))
        r = tp.transport(self.v)
        self.assertTrue(r.was_transported)


class TestVectorTransportAdaptive(unittest.TestCase):
    def test_adaptive_weight_increases_with_distance(self):
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Opposite vector: max distance (sim=-1), so weight = 1.0, clamped to max_weight
        tc = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tp = _make_transport(mode=TransportMode.ADAPTIVE, max_weight=0.5, min_similarity_for_transport=1.0)
        tp.add_target(TransportTarget("t1", tc))
        r = tp.transport(v, target_id="t1")
        self.assertGreater(r.weight_used, 0.0)
        self.assertTrue(r.was_transported)


class TestVectorTransportSimilarity(unittest.TestCase):
    def setUp(self):
        self.dim = 8
        self.v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.tc = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_cosine_similarity_before_in_result(self):
        tp = _make_transport(min_similarity_for_transport=1.0)
        tp.add_target(TransportTarget("t1", self.tc))
        r = tp.transport(self.v, target_id="t1")
        expected = float(np.dot(self.v, self.tc))  # both unit, dot=0
        self.assertAlmostEqual(r.cosine_similarity_before, expected, places=5)

    def test_cosine_similarity_after_in_result(self):
        tp = _make_transport(default_weight=0.3, min_similarity_for_transport=1.0)
        tp.add_target(TransportTarget("t1", self.tc))
        r = tp.transport(self.v, target_id="t1")
        tv = r.transported
        expected_after = float(
            np.dot(tv, self.tc) / (np.linalg.norm(tv) * np.linalg.norm(self.tc))
        )
        self.assertAlmostEqual(r.cosine_similarity_after, expected_after, places=5)


class TestVectorTransportDisabled(unittest.TestCase):
    def test_disabled_passthrough(self):
        tp = _make_transport(enabled=False)
        v = np.random.rand(8)
        tp.add_target(TransportTarget("t1", np.ones(8)))
        r = tp.transport(v, target_id="t1")
        self.assertFalse(r.was_transported)
        np.testing.assert_array_almost_equal(r.transported, v)


class TestVectorTransportNearestTarget(unittest.TestCase):
    def test_nearest_selected_by_cosine_similarity(self):
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        t1 = TransportTarget("t1", _unit(np.array([0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
        t2 = TransportTarget("t2", np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        tp = _make_transport(min_similarity_for_transport=1.0)
        tp.add_target(t1)
        tp.add_target(t2)
        r = tp.transport(v)  # no target_id → find nearest
        self.assertEqual(r.target_id, "t1")


class TestVectorTransportSlerp(unittest.TestCase):
    def _tp(self) -> VectorTransport:
        return _make_transport()

    def test_slerp_unit_result(self):
        tp = self._tp()
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        vt = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        res = tp._slerp(v, vt, 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(res)), 1.0, places=4)

    def test_slerp_t0_returns_v(self):
        tp = self._tp()
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        vt = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        res = tp._slerp(v, vt, 0.0)
        np.testing.assert_array_almost_equal(res, v, decimal=4)

    def test_slerp_t1_returns_vt(self):
        tp = self._tp()
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        vt = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        res = tp._slerp(v, vt, 1.0)
        np.testing.assert_array_almost_equal(res, vt, decimal=4)


class TestVectorTransportSaveLoad(unittest.TestCase):
    _files: list[str] = []

    def tearDown(self):
        for f in self._files:
            if os.path.exists(f):
                os.remove(f)
        self._files.clear()

    def _tmp(self, name: str) -> str:
        self._files.append(name)
        return name

    def test_save_load_roundtrip_target(self):
        tp = _make_transport()
        centroid = _unit(np.ones(8))
        tp.add_target(TransportTarget("tt", centroid, description="desc", support=10))
        path = self._tmp("_tp_tgt.json")
        tp.save(path)
        with patch(_PATCH, return_value=MagicMock()):
            loaded = VectorTransport.load(path)
        self.assertIn("tt", loaded._targets)
        np.testing.assert_array_almost_equal(loaded._targets["tt"].centroid, centroid)

    def test_save_load_config_preserved(self):
        tp = _make_transport(mode=TransportMode.COSINE, default_weight=0.35)
        path = self._tmp("_tp_cfg.json")
        tp.save(path)
        with patch(_PATCH, return_value=MagicMock()):
            loaded = VectorTransport.load(path)
        self.assertEqual(loaded._config.mode, TransportMode.COSINE)
        self.assertAlmostEqual(loaded._config.default_weight, 0.35)


class TestVectorTransportRegistrationAPI(unittest.TestCase):
    """Tests for register_target, target_count, and clear_targets."""

    def setUp(self):
        self.tp = _make_transport()

    def test_register_target_stores_entry(self):
        centroid = _unit(np.array([1.0, 0.0, 0.0, 0.0]))
        self.tp.register_target("t1", centroid, "label-A")
        self.assertIn("t1", self.tp._targets)

    def test_register_target_sets_centroid(self):
        centroid = _unit(np.array([0.0, 1.0, 0.0, 0.0]))
        self.tp.register_target("t1", centroid)
        np.testing.assert_array_almost_equal(self.tp._targets["t1"].centroid, centroid)

    def test_register_target_sets_label_as_description(self):
        centroid = np.ones(4)
        self.tp.register_target("t1", centroid, label="my-label")
        self.assertEqual(self.tp._targets["t1"].description, "my-label")

    def test_register_target_default_label_empty(self):
        self.tp.register_target("t1", np.ones(4))
        self.assertEqual(self.tp._targets["t1"].description, "")

    def test_register_target_overwrite(self):
        c1 = _unit(np.array([1.0, 0.0, 0.0, 0.0]))
        c2 = _unit(np.array([0.0, 1.0, 0.0, 0.0]))
        self.tp.register_target("t1", c1, "first")
        self.tp.register_target("t1", c2, "second")
        self.assertEqual(self.tp.target_count(), 1)
        self.assertEqual(self.tp._targets["t1"].description, "second")

    def test_target_count_zero_initially(self):
        self.assertEqual(self.tp.target_count(), 0)

    def test_target_count_increments(self):
        self.tp.register_target("a", np.ones(4))
        self.assertEqual(self.tp.target_count(), 1)
        self.tp.register_target("b", np.zeros(4) + 0.1)
        self.assertEqual(self.tp.target_count(), 2)

    def test_clear_targets_removes_all(self):
        self.tp.register_target("a", np.ones(4))
        self.tp.register_target("b", np.zeros(4) + 0.1)
        self.tp.clear_targets()
        self.assertEqual(self.tp.target_count(), 0)

    def test_clear_targets_on_empty_is_safe(self):
        self.tp.clear_targets()  # should not raise
        self.assertEqual(self.tp.target_count(), 0)

    def test_register_target_converts_list_to_ndarray(self):
        self.tp.register_target("t1", [1.0, 0.0, 0.0, 0.0])
        self.assertIsInstance(self.tp._targets["t1"].centroid, np.ndarray)

    def test_register_then_transport_fires(self):
        # Vector orthogonal to target → sim=0 < 0.85 → transport fires
        target = _unit(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        query = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.tp.register_target("t1", target)
        r = self.tp.transport(query)
        self.assertTrue(r.was_transported)

    def test_register_target_id_matches(self):
        centroid = np.ones(4)
        self.tp.register_target("my-id", centroid)
        self.assertEqual(self.tp._targets["my-id"].target_id, "my-id")


class TestTransportConfigDefaultThreshold(unittest.TestCase):
    """Verify the new default threshold is 0.85."""

    def test_default_is_0_85(self):
        from vector_transport import TransportConfig

        c = TransportConfig()
        self.assertAlmostEqual(c.min_similarity_for_transport, 0.85)

    def test_transport_skips_when_sim_above_0_85(self):
        """With default threshold, vector already close (sim>0.85) is NOT transported."""
        tp = _make_transport()
        target = _unit(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        tp.register_target("t", target)
        # Vector very close to target (sim ≈ 0.999)
        query = _unit(target + np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        r = tp.transport(query)
        self.assertFalse(r.was_transported)

    def test_transport_fires_when_sim_below_0_85(self):
        """With default threshold, orthogonal vector IS transported."""
        tp = _make_transport()
        target = _unit(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        tp.register_target("t", target)
        query = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        r = tp.transport(query)
        self.assertTrue(r.was_transported)


if __name__ == "__main__":
    unittest.main()
