"""Tests for post_bank_correction.py — clustering + bank-from-anchors assembly
(Phase II, H5b / S2a). numpy-only, stub corrections, no torch/GPU."""
from __future__ import annotations

import unittest

import numpy as np

from post_bank_correction import build_post_bank, cluster_vectors
from steering_post_bank import SteeringPostBank


def _two_groups():
    # Two well-separated groups in 2D.
    return np.array([[10.0, 0.0], [10.1, 0.0], [0.0, 10.0], [0.0, 10.1]])


def _shift_by_centroid(d, q):
    # Stub correction: shift a vector by this cluster's centroid (a region-unique,
    # data-derived marker), so we can verify the RIGHT post fired for a region.
    centroid = d.mean(axis=0)
    return lambda v: v + centroid


class TestClusterVectors(unittest.TestCase):
    def test_separates_two_groups(self):
        labels, centroids = cluster_vectors(_two_groups(), k=2, seed=0)
        self.assertEqual(centroids.shape, (2, 2))
        self.assertEqual(labels[0], labels[1])      # the two near (10,0) together
        self.assertEqual(labels[2], labels[3])      # the two near (0,10) together
        self.assertNotEqual(labels[0], labels[2])   # the groups are distinct

    def test_deterministic(self):
        a_labels, a_cent = cluster_vectors(_two_groups(), k=2, seed=7)
        b_labels, b_cent = cluster_vectors(_two_groups(), k=2, seed=7)
        np.testing.assert_array_equal(a_labels, b_labels)
        np.testing.assert_allclose(a_cent, b_cent)

    def test_k_clamped_to_n(self):
        labels, centroids = cluster_vectors(np.eye(3), k=10, seed=0)
        self.assertEqual(centroids.shape[0], 3)  # clamped to N=3
        self.assertTrue(set(labels.tolist()) <= {0, 1, 2})

    def test_no_empty_clusters(self):
        labels, centroids = cluster_vectors(_two_groups(), k=3, seed=1)
        # every cluster index in [0, k) is represented (empty clusters re-seeded)
        self.assertEqual(set(labels.tolist()), set(range(centroids.shape[0])))

    def test_validations(self):
        with self.assertRaises(ValueError):
            cluster_vectors(np.zeros((0, 2)), k=2, seed=0)   # empty
        with self.assertRaises(ValueError):
            cluster_vectors(_two_groups(), k=0, seed=0)      # k<=0
        with self.assertRaises(ValueError):
            cluster_vectors(np.zeros(5), k=2, seed=0)        # not 2D


class TestBuildPostBank(unittest.TestCase):
    def test_one_post_per_cluster(self):
        docs = _two_groups()
        queries = docs.copy()
        bank = build_post_bank(docs, queries, k=2, make_post=_shift_by_centroid, seed=0)
        self.assertIsInstance(bank, SteeringPostBank)
        self.assertEqual(len(bank), 2)
        self.assertEqual({p.key for p in bank.posts}, {"post:0", "post:1"})

    def test_routes_each_region_to_its_own_post_correction(self):
        docs = _two_groups()
        queries = docs.copy()
        bank = build_post_bank(docs, queries, k=2, make_post=_shift_by_centroid, seed=0)
        # A vector in group A (~[10,0]) must be shifted by group A's centroid (~[10,0]);
        # a vector in group B (~[0,10]) by group B's centroid (~[0,10]).
        out = bank.apply(np.array([[10.0, 0.0], [0.0, 10.0]]))
        self.assertAlmostEqual(out[0][0], 10.0 + 10.05, places=2)  # group A: +centroidA
        self.assertAlmostEqual(out[0][1], 0.0, places=2)
        self.assertAlmostEqual(out[1][0], 0.0, places=2)
        self.assertAlmostEqual(out[1][1], 10.0 + 10.05, places=2)  # group B: +centroidB

    def test_deterministic(self):
        docs = _two_groups()
        q = docs.copy()
        a = build_post_bank(docs, q, k=2, make_post=_shift_by_centroid, seed=3).to_dict()
        b = build_post_bank(docs, q, k=2, make_post=_shift_by_centroid, seed=3).to_dict()
        self.assertEqual(a["posts"], b["posts"])

    def test_validations(self):
        docs = _two_groups()
        with self.assertRaises(ValueError):
            build_post_bank(docs, docs[:3], k=2, make_post=_shift_by_centroid, seed=0)  # N mismatch
        with self.assertRaises(ValueError):
            build_post_bank(np.zeros(4), np.zeros(4), k=2, make_post=_shift_by_centroid, seed=0)  # not 2D


if __name__ == "__main__":
    unittest.main()
