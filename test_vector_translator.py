"""Tests for VectorTranslator (≥30 tests)."""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vector_translator import (
    ClusterOffset,
    TranslationConfig,
    TranslationResult,
    VectorTranslator,
)

_PATCH = "vector_translator.get_logger"


def _make_translator(dim: int = 8, **cfg_kwargs) -> VectorTranslator:
    config = TranslationConfig(offset_dim=dim, **cfg_kwargs)
    with patch(_PATCH, return_value=MagicMock()):
        return VectorTranslator(config)


class TestTranslationConfig(unittest.TestCase):
    def test_defaults(self):
        c = TranslationConfig(offset_dim=384)
        self.assertEqual(c.offset_dim, 384)
        self.assertAlmostEqual(c.cluster_confidence_threshold, 0.7)
        self.assertAlmostEqual(c.max_offset_norm, 1.0)
        self.assertTrue(c.enabled)

    def test_custom_values(self):
        c = TranslationConfig(
            offset_dim=768,
            cluster_confidence_threshold=0.9,
            max_offset_norm=0.5,
            enabled=False,
        )
        self.assertEqual(c.offset_dim, 768)
        self.assertAlmostEqual(c.cluster_confidence_threshold, 0.9)
        self.assertAlmostEqual(c.max_offset_norm, 0.5)
        self.assertFalse(c.enabled)

    def test_offset_dim_required(self):
        with self.assertRaises(TypeError):
            TranslationConfig()  # type: ignore[call-arg]


class TestClusterOffset(unittest.TestCase):
    def test_construction(self):
        centroid = np.array([1.0, 0.0, 0.0])
        offset = np.array([0.1, 0.2, 0.3])
        co = ClusterOffset(cluster_id="c1", centroid=centroid, offset=offset, support=10)
        self.assertEqual(co.cluster_id, "c1")
        np.testing.assert_array_equal(co.centroid, centroid)
        np.testing.assert_array_equal(co.offset, offset)
        self.assertEqual(co.support, 10)

    def test_fields_accessible(self):
        co = ClusterOffset("x", np.zeros(4), np.ones(4), 5)
        self.assertEqual(co.support, 5)


class TestVectorTranslatorBasic(unittest.TestCase):
    def setUp(self):
        self.dim = 8
        self.t = _make_translator(self.dim)

    def test_initial_no_learned_offset(self):
        self.assertIsNone(self.t._learned_offset)

    def test_initial_cluster_offsets_empty(self):
        self.assertEqual(len(self.t._cluster_offsets), 0)

    def test_set_learned_offset_stores(self):
        offset = np.ones(self.dim) * 0.05
        self.t.set_learned_offset(offset)
        self.assertIsNotNone(self.t._learned_offset)
        np.testing.assert_array_almost_equal(self.t._learned_offset, offset)

    def test_set_learned_offset_clamps_large_norm(self):
        # norm of [2]*8 = sqrt(32) ≈ 5.66 > 1.0
        offset = np.ones(self.dim) * 2.0
        self.t.set_learned_offset(offset)
        stored_norm = float(np.linalg.norm(self.t._learned_offset))
        self.assertAlmostEqual(stored_norm, 1.0, places=5)

    def test_set_learned_offset_small_norm_not_clamped(self):
        offset = np.ones(self.dim) * 0.01  # norm ≈ 0.028
        self.t.set_learned_offset(offset)
        expected = float(np.linalg.norm(offset))
        stored = float(np.linalg.norm(self.t._learned_offset))
        self.assertAlmostEqual(stored, expected, places=6)

    def test_add_cluster_offset_stores(self):
        co = ClusterOffset("c99", np.ones(self.dim), np.zeros(self.dim), 3)
        self.t.add_cluster_offset(co)
        self.assertIn("c99", self.t._cluster_offsets)

    def test_multiple_cluster_offsets(self):
        for i in range(5):
            co = ClusterOffset(f"c{i}", np.ones(self.dim) * i, np.ones(self.dim) * 0.1 * i, i)
            self.t.add_cluster_offset(co)
        self.assertEqual(len(self.t._cluster_offsets), 5)


class TestVectorTranslatorTranslate(unittest.TestCase):
    def setUp(self):
        self.dim = 8
        self.t = _make_translator(self.dim)

    def test_no_offset_returns_passthrough(self):
        v = np.random.rand(self.dim)
        r = self.t.translate(v)
        self.assertEqual(r.mode, "passthrough")
        np.testing.assert_array_almost_equal(r.translated, v)

    def test_passthrough_offset_norm_zero(self):
        v = np.random.rand(self.dim)
        r = self.t.translate(v)
        self.assertAlmostEqual(r.offset_norm, 0.0)

    def test_passthrough_cluster_id_none(self):
        r = self.t.translate(np.ones(self.dim))
        self.assertIsNone(r.cluster_id)

    def test_translate_returns_translation_result(self):
        r = self.t.translate(np.zeros(self.dim))
        self.assertIsInstance(r, TranslationResult)

    def test_original_preserved_in_result(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.t.set_learned_offset(np.ones(self.dim) * 0.1)
        r = self.t.translate(v)
        np.testing.assert_array_equal(r.original, v)

    def test_learned_offset_mode(self):
        offset = np.ones(self.dim) * 0.05
        self.t.set_learned_offset(offset)
        v = np.zeros(self.dim)
        r = self.t.translate(v)
        self.assertEqual(r.mode, "learned")

    def test_learned_offset_applied(self):
        offset = np.ones(self.dim) * 0.05
        self.t.set_learned_offset(offset)
        v = np.zeros(self.dim)
        r = self.t.translate(v)
        np.testing.assert_array_almost_equal(r.translated, offset)

    def test_learned_offset_arbitrary_v(self):
        offset = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0])
        self.t.set_learned_offset(offset)
        v = np.ones(self.dim)
        r = self.t.translate(v)
        np.testing.assert_array_almost_equal(r.translated, v + offset)

    def test_offset_norm_in_result_matches_actual(self):
        offset = np.ones(self.dim) * 0.05
        self.t.set_learned_offset(offset)
        v = np.random.rand(self.dim)
        r = self.t.translate(v)
        self.assertAlmostEqual(r.offset_norm, float(np.linalg.norm(r.offset_used)), places=6)

    def test_cluster_override_high_confidence(self):
        v = np.zeros(self.dim)
        c_offset = np.ones(self.dim) * 0.3
        co = ClusterOffset("cx", np.ones(self.dim), c_offset, 5)
        self.t.add_cluster_offset(co)
        r = self.t.translate(v, cluster_id="cx", cluster_confidence=0.9)
        self.assertEqual(r.mode, "cluster_override")
        self.assertEqual(r.cluster_id, "cx")
        np.testing.assert_array_almost_equal(r.translated, v + c_offset)

    def test_cluster_override_at_exact_threshold(self):
        t = _make_translator(self.dim, cluster_confidence_threshold=0.7)
        co = ClusterOffset("cx", np.ones(self.dim), np.ones(self.dim) * 0.1, 1)
        t.add_cluster_offset(co)
        r = t.translate(np.zeros(self.dim), cluster_id="cx", cluster_confidence=0.7)
        self.assertEqual(r.mode, "cluster_override")

    def test_cluster_override_skipped_low_confidence(self):
        co = ClusterOffset("cx", np.ones(self.dim), np.ones(self.dim) * 0.5, 5)
        self.t.add_cluster_offset(co)
        r = self.t.translate(np.zeros(self.dim), cluster_id="cx", cluster_confidence=0.5)
        self.assertNotEqual(r.mode, "cluster_override")

    def test_cluster_override_skipped_unknown_id(self):
        r = self.t.translate(np.zeros(self.dim), cluster_id="no_such_id", cluster_confidence=0.99)
        self.assertNotEqual(r.mode, "cluster_override")

    def test_cluster_override_ignores_learned_offset(self):
        """When cluster override fires the cluster offset is used, not the learned one."""
        self.t.set_learned_offset(np.ones(self.dim) * 0.5)
        c_off = np.ones(self.dim) * 0.1
        co = ClusterOffset("cx", np.zeros(self.dim), c_off, 1)
        self.t.add_cluster_offset(co)
        v = np.zeros(self.dim)
        r = self.t.translate(v, cluster_id="cx", cluster_confidence=0.9)
        np.testing.assert_array_almost_equal(r.translated, v + c_off)

    def test_disabled_config_returns_passthrough(self):
        t = _make_translator(self.dim, enabled=False)
        t.set_learned_offset(np.ones(self.dim) * 0.5)
        r = t.translate(np.random.rand(self.dim))
        self.assertEqual(r.mode, "passthrough")


class TestVectorTranslatorSaveLoad(unittest.TestCase):
    _files: list[str] = []

    def tearDown(self):
        for f in self._files:
            if os.path.exists(f):
                os.remove(f)
        self._files.clear()

    def _tmp(self, name: str) -> str:
        self._files.append(name)
        return name

    def test_save_load_no_offset(self):
        t = _make_translator(8)
        path = self._tmp("_tl_no_offset.json")
        t.save(path)
        with patch(_PATCH, return_value=MagicMock()):
            loaded = VectorTranslator.load(path)
        self.assertIsNone(loaded._learned_offset)

    def test_save_load_learned_offset(self):
        t = _make_translator(8)
        offset = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        t.set_learned_offset(offset)
        path = self._tmp("_tl_offset.json")
        t.save(path)
        with patch(_PATCH, return_value=MagicMock()):
            loaded = VectorTranslator.load(path)
        np.testing.assert_array_almost_equal(loaded._learned_offset, t._learned_offset)

    def test_save_load_cluster_offset(self):
        t = _make_translator(8)
        co = ClusterOffset("clus", np.ones(8), np.ones(8) * 0.25, 42)
        t.add_cluster_offset(co)
        path = self._tmp("_tl_cluster.json")
        t.save(path)
        with patch(_PATCH, return_value=MagicMock()):
            loaded = VectorTranslator.load(path)
        self.assertIn("clus", loaded._cluster_offsets)
        self.assertEqual(loaded._cluster_offsets["clus"].support, 42)

    def test_save_load_config_preserved(self):
        t = _make_translator(8, cluster_confidence_threshold=0.85, max_offset_norm=0.5)
        path = self._tmp("_tl_cfg.json")
        t.save(path)
        with patch(_PATCH, return_value=MagicMock()):
            loaded = VectorTranslator.load(path)
        self.assertAlmostEqual(loaded._config.cluster_confidence_threshold, 0.85)
        self.assertAlmostEqual(loaded._config.max_offset_norm, 0.5)


class TestVectorTranslatorFromPhaseC(unittest.TestCase):
    _files: list[str] = []

    def tearDown(self):
        for f in self._files:
            if os.path.exists(f):
                os.remove(f)
        self._files.clear()

    def _tmp(self, name: str) -> str:
        self._files.append(name)
        return name

    def test_missing_file_returns_passthrough(self):
        config = TranslationConfig(offset_dim=8)
        with patch(_PATCH, return_value=MagicMock()):
            t = VectorTranslator.from_phase_c_results("__nonexistent__.json", config)
        self.assertIsNone(t._learned_offset)

    def test_invalid_json_returns_passthrough(self):
        path = self._tmp("_pc_bad.json")
        with open(path, "w") as fh:
            fh.write("{not valid json")
        config = TranslationConfig(offset_dim=8)
        with patch(_PATCH, return_value=MagicMock()):
            t = VectorTranslator.from_phase_c_results(path, config)
        self.assertIsNone(t._learned_offset)

    def test_empty_summaries_returns_passthrough(self):
        path = self._tmp("_pc_empty.json")
        with open(path, "w") as fh:
            json.dump({"summaries": []}, fh)
        config = TranslationConfig(offset_dim=8)
        with patch(_PATCH, return_value=MagicMock()):
            t = VectorTranslator.from_phase_c_results(path, config)
        self.assertIsNone(t._learned_offset)

    def test_with_data_offset_is_nonzero(self):
        path = self._tmp("_pc_data.json")
        with open(path, "w") as fh:
            json.dump(
                {
                    "summaries": [
                        {"baseline_ndcg": 0.50, "best_ndcg": 0.70},
                        {"baseline_ndcg": 0.40, "best_ndcg": 0.65},
                    ]
                },
                fh,
            )
        config = TranslationConfig(offset_dim=8)
        with patch(_PATCH, return_value=MagicMock()):
            t = VectorTranslator.from_phase_c_results(path, config)
        self.assertIsNotNone(t._learned_offset)
        self.assertGreater(float(np.linalg.norm(t._learned_offset)), 0.0)

    def test_with_dict_summaries(self):
        path = self._tmp("_pc_dict.json")
        with open(path, "w") as fh:
            json.dump(
                {
                    "summary": {
                        "ds1": {"baseline_ndcg": 0.4, "best_ndcg": 0.6},
                    }
                },
                fh,
            )
        config = TranslationConfig(offset_dim=8)
        with patch(_PATCH, return_value=MagicMock()):
            t = VectorTranslator.from_phase_c_results(path, config)
        self.assertIsNotNone(t._learned_offset)


class TestTranslationResultOffsetNorm(unittest.TestCase):
    def test_offset_norm_matches_l2_of_offset_used(self):
        dim = 16
        t = _make_translator(dim)
        offset = np.ones(dim) * 0.3
        t.set_learned_offset(offset)
        v = np.zeros(dim)
        r = t.translate(v)
        self.assertAlmostEqual(r.offset_norm, float(np.linalg.norm(r.offset_used)), places=6)


if __name__ == "__main__":
    unittest.main()
