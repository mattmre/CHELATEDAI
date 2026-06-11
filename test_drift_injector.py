from __future__ import annotations

import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from antigravity_engine import AntigravityEngine
from drift_injector import DriftInjector


class _IdentityAdapter(torch.nn.Module):
    def load(self, _path):
        return False

    def forward(self, tensor):
        return tensor


class _DeterministicBackend:
    def __init__(self, vectors):
        self.vectors = vectors
        self.vector_size = len(next(iter(vectors.values())))

    def embed_raw(self, texts):
        return np.vstack([self.vectors[text] for text in texts]).astype(np.float32)


def _read_vectors(engine):
    points, _next_offset = engine.qdrant.scroll(
        collection_name=engine.collection_name,
        limit=1000,
        with_vectors=True,
        with_payload=True,
    )
    return {point.id: np.asarray(point.vector, dtype=np.float32) for point in points}


def _ndcg_for_single_relevant(ranked_ids, relevant_id, k=10):
    gains = [1.0 if doc_id == relevant_id else 0.0 for doc_id in ranked_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    return dcg


def _mean_ndcg(engine, query_ids):
    scores = []
    for doc_id in query_ids:
        _std_top, final_top, _mask, _jaccard = engine.run_inference(f"query-{doc_id}")
        scores.append(_ndcg_for_single_relevant(final_top, doc_id))
    return float(np.mean(scores))


def _build_vectors(doc_count=50, vector_size=64):
    vectors = {}
    for doc_id in range(doc_count // 2):
        target = np.zeros(vector_size, dtype=np.float32)
        target[doc_id] = 1.0
        distractor = np.zeros(vector_size, dtype=np.float32)
        distractor[doc_id] = 0.94
        distractor[(doc_id + 31) % vector_size] = math.sqrt(1.0 - (0.94 * 0.94))
        vectors[f"doc-{doc_id}"] = target
        vectors[f"query-{doc_id}"] = target
        vectors[f"doc-{doc_id + (doc_count // 2)}"] = distractor
    return vectors


class TestDriftInjector(unittest.TestCase):
    def _make_engine(self, vectors):
        patchers = [
            patch("antigravity_engine.get_logger", return_value=MagicMock()),
            patch("antigravity_engine.create_adapter", return_value=_IdentityAdapter()),
            patch("antigravity_engine.create_embedding_backend", return_value=_DeterministicBackend(vectors)),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            store_full_text_payload=True,
        )
        return engine

    def test_rotation_drift_is_seed_deterministic(self):
        vectors = _build_vectors()
        corpus = [f"doc-{index}" for index in range(50)]

        engine_a = self._make_engine(vectors)
        engine_a.ingest(corpus, payloads=[{"doc_id": index} for index in range(50)])
        manifest_a = DriftInjector(engine_a, seed=17).inject_rotation_drift(0.5, 25.0)
        vectors_a = _read_vectors(engine_a)

        engine_b = self._make_engine(vectors)
        engine_b.ingest(corpus, payloads=[{"doc_id": index} for index in range(50)])
        manifest_b = DriftInjector(engine_b, seed=17).inject_rotation_drift(0.5, 25.0)
        vectors_b = _read_vectors(engine_b)

        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(vectors_a.keys(), vectors_b.keys())
        for point_id in vectors_a:
            np.testing.assert_array_equal(vectors_a[point_id], vectors_b[point_id])

    def test_fraction_affects_expected_count_and_leaves_rest_bit_identical(self):
        vectors = _build_vectors()
        engine = self._make_engine(vectors)
        engine.ingest([f"doc-{index}" for index in range(50)])
        before = _read_vectors(engine)

        manifest = DriftInjector(engine, seed=23).inject_noise_drift(0.3, 0.05)
        after = _read_vectors(engine)

        self.assertEqual(manifest["affected_count"], 15)
        affected_ids = set(manifest["affected_ids"])
        self.assertEqual(len(affected_ids), 15)
        for point_id, before_vector in before.items():
            if point_id not in affected_ids:
                np.testing.assert_array_equal(before_vector, after[point_id])
            else:
                self.assertFalse(np.array_equal(before_vector, after[point_id]))

    def test_rotation_drift_measurably_degrades_real_retrieval_ndcg(self):
        vectors = _build_vectors()
        engine = self._make_engine(vectors)
        engine.ingest([f"doc-{index}" for index in range(50)])

        query_ids = list(range(25))
        baseline_ndcg = _mean_ndcg(engine, query_ids)
        manifest = DriftInjector(engine, seed=7).inject_rotation_drift(0.5, 25.0)
        drifted_ndcg = _mean_ndcg(engine, query_ids)

        self.assertEqual(manifest["affected_count"], 25)
        self.assertGreaterEqual(baseline_ndcg, 0.99)
        self.assertLess(drifted_ndcg, baseline_ndcg - 0.05)

    def test_manifest_round_trips_through_json(self):
        vectors = _build_vectors()
        engine = self._make_engine(vectors)
        engine.ingest([f"doc-{index}" for index in range(50)])

        manifest = DriftInjector(engine, seed=31).inject_noise_drift(0.3, 0.05)
        decoded = json.loads(json.dumps(manifest, sort_keys=True))

        self.assertEqual(decoded, manifest)
        self.assertEqual(len(decoded["checksum_before"]), 64)
        self.assertEqual(len(decoded["checksum_after"]), 64)

    def test_single_named_vector_schema_is_preserved_on_upsert(self):
        class FakeQdrant:
            def __init__(self):
                self.upserted_points = None
                self._points = [
                    SimpleNamespace(id=1, vector={"dense": [1.0, 0.0, 0.0]}, payload={"text": "a"}),
                    SimpleNamespace(id=2, vector={"dense": [0.0, 1.0, 0.0]}, payload={"text": "b"}),
                ]

            def scroll(self, **_kwargs):
                return self._points, None

            def upsert(self, collection_name, points):
                self.collection_name = collection_name
                self.upserted_points = points

        fake_qdrant = FakeQdrant()
        engine = SimpleNamespace(qdrant=fake_qdrant, collection_name="named_vectors")

        manifest = DriftInjector(engine, seed=11).inject_noise_drift(0.5, 0.01)

        self.assertEqual(manifest["affected_count"], 1)
        self.assertEqual(fake_qdrant.collection_name, "named_vectors")
        self.assertEqual(len(fake_qdrant.upserted_points), 1)
        upserted_vector = fake_qdrant.upserted_points[0].vector
        self.assertIsInstance(upserted_vector, dict)
        self.assertEqual(set(upserted_vector.keys()), {"dense"})
        self.assertEqual(len(upserted_vector["dense"]), 3)


if __name__ == "__main__":
    unittest.main()
