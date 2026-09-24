"""Partial Qdrant upserts write the pre-cycle vectors back."""

import unittest

import numpy as np

from sedimentation_trainer import sync_vectors_to_qdrant


class _Point:
    def __init__(self, point_id, vector, payload):
        self.id = point_id
        self.vector = vector
        self.payload = payload


class _Client:
    def __init__(self):
        self.points = {}
        self.upserts = 0

    def upsert(self, collection_name, points):
        self.upserts += 1
        if any(point.id == "b" for point in points):
            raise RuntimeError("second chunk failed")
        for point in points:
            self.points[point.id] = (list(point.vector), dict(point.payload or {}))

    def retrieve(self, collection_name, ids, with_vectors=False):
        return []


class _Logger:
    def __init__(self):
        self.errors = []

    def log_error(self, kind, message, **kwargs):
        self.errors.append((kind, message))


class TestCorpusRollback(unittest.TestCase):
    def test_second_chunk_failure_restores_the_first_id(self):
        client = _Client()
        logger = _Logger()
        originals = np.array([[0.25], [0.5]], dtype=np.float32)
        adapted = np.array([[9.0], [9.0]], dtype=np.float32)
        total, failed = sync_vectors_to_qdrant(
            client,
            "docs",
            ["a", "b"],
            adapted,
            1,
            logger,
            {"a": {"text": "keep"}, "b": {"text": "later"}},
            original_vectors_np=originals,
        )
        self.assertEqual(failed, 1)
        self.assertEqual(client.points["a"][0], [0.25])
        self.assertEqual(client.points["a"][1], {"text": "keep"})
        self.assertNotIn("b", client.points)
        self.assertFalse(any("Rolling back" in message for _, message in logger.errors))
        self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main()
