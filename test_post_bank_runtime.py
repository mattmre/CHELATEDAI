"""Tests for post_bank_runtime.py — build-from-anchors + apply-to-store glue
(Phase II, H5b / S2b runtime). numpy-only, stub corrections, fake store, no GPU."""
from __future__ import annotations

import unittest

import numpy as np

from post_bank_runtime import apply_post_bank_to_store, build_post_bank_from_anchor_pairs
from steering_post_bank import SteeringPostBank


class _FakeQdrant:
    def __init__(self):
        self.collection = None
        self.upserted = None

    def upsert(self, collection_name, points):
        self.collection = collection_name
        self.upserted = list(points)


class _FakeEngine:
    def __init__(self):
        self.qdrant = _FakeQdrant()
        self.collection_name = "test_collection"


def _anchor_pairs():
    # 4 anchors in two well-separated 2D regions; doc + drifted-query vectors.
    return [
        {"doc_vector": [10.0, 0.0], "query_vector": [11.0, 0.0]},
        {"doc_vector": [10.1, 0.0], "query_vector": [11.1, 0.0]},
        {"doc_vector": [0.0, 10.0], "query_vector": [0.0, 11.0]},
        {"doc_vector": [0.0, 10.1], "query_vector": [0.0, 11.1]},
    ]


def _shift_by_centroid(d, q):
    centroid = d.mean(axis=0)
    return lambda v: v + centroid


def _bank():
    return build_post_bank_from_anchor_pairs(
        _anchor_pairs(), vector_size=2, k=2, make_post=_shift_by_centroid, seed=0
    )


class TestBuildFromAnchorPairs(unittest.TestCase):
    def test_builds_one_post_per_cluster(self):
        bank = _bank()
        self.assertIsInstance(bank, SteeringPostBank)
        self.assertEqual(len(bank), 2)
        self.assertEqual({p.key for p in bank.posts}, {"post:0", "post:1"})

    def test_validations(self):
        with self.assertRaises(ValueError):
            build_post_bank_from_anchor_pairs([], 2, 2, _shift_by_centroid, 0)  # empty
        with self.assertRaises(ValueError):
            # vector_size mismatch (anchors are 2D, claim 3)
            build_post_bank_from_anchor_pairs(_anchor_pairs(), 3, 2, _shift_by_centroid, 0)


class TestApplyToStore(unittest.TestCase):
    def _points(self):
        return [
            {"id": "d1", "vector": [10.0, 0.0], "payload": {"doc_id": "d1"}},
            {"id": "d2", "vector": [0.0, 10.0], "payload": None},
        ]

    def test_writes_region_correct_corrected_vectors_back(self):
        engine = _FakeEngine()
        result = apply_post_bank_to_store(engine, _bank(), self._points())
        self.assertEqual(result["updated"], 2)
        self.assertEqual(engine.qdrant.collection, "test_collection")
        upserted = {p.id: np.asarray(p.vector) for p in engine.qdrant.upserted}
        # d1 (region A ~[10,0]) shifted by region A's centroid ~[10.05,0] -> ~[20.05,0]
        np.testing.assert_allclose(upserted["d1"], [20.05, 0.0], atol=0.02)
        # d2 (region B ~[0,10]) shifted by region B's centroid ~[0,10.05] -> ~[0,20.05]
        np.testing.assert_allclose(upserted["d2"], [0.0, 20.05], atol=0.02)
        self.assertGreater(result["correction_norm_stats"]["mean"], 0.0)
        self.assertEqual(sum(result["per_post_hits"].values()), 2)

    def test_empty_points_is_noop(self):
        engine = _FakeEngine()
        result = apply_post_bank_to_store(engine, _bank(), [])
        self.assertEqual(result["updated"], 0)
        self.assertIsNone(engine.qdrant.upserted)  # never called upsert

    def test_apply_is_idempotent_on_fixed_snapshot(self):
        bank = _bank()
        e1, e2 = _FakeEngine(), _FakeEngine()
        apply_post_bank_to_store(e1, bank, self._points())
        apply_post_bank_to_store(e2, bank, self._points())
        v1 = {p.id: np.asarray(p.vector) for p in e1.qdrant.upserted}
        v2 = {p.id: np.asarray(p.vector) for p in e2.qdrant.upserted}
        for key in v1:
            np.testing.assert_allclose(v1[key], v2[key])

    def test_payload_preserved(self):
        engine = _FakeEngine()
        apply_post_bank_to_store(engine, _bank(), self._points())
        payloads = {p.id: p.payload for p in engine.qdrant.upserted}
        self.assertEqual(payloads["d1"], {"doc_id": "d1"})
        self.assertIsNone(payloads["d2"])


if __name__ == "__main__":
    unittest.main()
