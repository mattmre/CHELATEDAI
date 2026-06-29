"""Tests for teacher_supervised_correction.py — H3/C3b teacher distillation (H3a).
The pairing logic is stub-tested (no GPU); the training is exercised on a tiny CPU
adapter to PROVE the distillation learns (MSE drops)."""
from __future__ import annotations

import unittest

import numpy as np

from teacher_supervised_correction import build_teacher_pairs, train_distillation_adapter


def _zeros_teacher(dim=4):
    return lambda texts: np.zeros((len(texts), dim), dtype=np.float32)


def _unit(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


class TestBuildTeacherPairs(unittest.TestCase):
    def test_aligns_docs_and_teachers(self):
        docs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        d, t = build_teacher_pairs(docs, ["a", "bb"], _zeros_teacher())
        self.assertEqual(d.shape, (2, 4))
        self.assertEqual(t.shape, (2, 4))

    def test_validations(self):
        with self.assertRaises(ValueError):
            build_teacher_pairs([], [], _zeros_teacher())                       # empty
        with self.assertRaises(ValueError):
            build_teacher_pairs([[1.0, 0.0, 0.0, 0.0]], ["a", "b"], _zeros_teacher())  # N mismatch
        with self.assertRaises(ValueError):
            build_teacher_pairs(np.zeros(4), ["a"], _zeros_teacher())           # not 2D
        with self.assertRaises(ValueError):  # teacher dim != doc dim
            build_teacher_pairs([[1.0, 0.0, 0.0, 0.0]], ["a"], _zeros_teacher(dim=8))
        with self.assertRaises(ValueError):  # teacher row count mismatch
            build_teacher_pairs([[1.0, 0.0, 0.0, 0.0]], ["a"],
                                lambda xs: np.zeros((2, 4), dtype=np.float32))


class TestTrainDistillationAdapter(unittest.TestCase):
    def test_adapter_learns_toward_teacher(self):
        from chelation_adapter import create_adapter

        rng = np.random.RandomState(0)
        docs = _unit(rng.randn(8, 4).astype(np.float32))
        # The teacher = the new-encoder doc projection: a unit vector rotated from
        # the old doc (embeddings are L2-normalized, as the adapter output is).
        teachers = _unit(docs + np.array([0.2, -0.2, 0.2, -0.2], dtype=np.float32))
        adapter = create_adapter("mlp", input_dim=4)
        initial, final = train_distillation_adapter(
            adapter, docs, teachers, seed=0, steps=300, learning_rate=0.1)
        self.assertGreater(initial, 0.0)
        self.assertLess(final, initial)            # it LEARNED — the "proof it learns"
        self.assertLess(final, initial * 0.5)      # meaningfully, not marginally

    def test_bounded_adapter_does_not_worsen(self):
        from chelation_adapter import create_adapter

        rng = np.random.RandomState(1)
        docs = _unit(rng.randn(6, 4).astype(np.float32))
        teachers = _unit(docs + 0.1)  # small rotation within a bounded correction's reach
        adapter = create_adapter("mlp", input_dim=4, bounded=True)
        initial, final = train_distillation_adapter(adapter, docs, teachers, seed=1, steps=50)
        self.assertLessEqual(final, initial + 1e-6)  # bounded: never worsens


if __name__ == "__main__":
    unittest.main()
