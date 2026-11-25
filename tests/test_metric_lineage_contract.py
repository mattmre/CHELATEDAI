"""CI falsifier tests for the canonical complete-qrels metric contract (mlr-v1 / D-01)."""

from __future__ import annotations

import math
import random
import unittest

from benchmark_utils import (
    METRIC_CONTRACT_VERSION,
    build_metric_contract_manifest,
    mean_average_precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    validate_ranked_ids,
)
from drift_recovery_metrics import ndcg_at_k as drift_ndcg_at_k
from retrieval_fitness_evaluator import RetrievalFitnessEvaluator


def _reference_ndcg(ranked_ids, qrels_map, k, binary=False):
    pos = {str(d): (1.0 if binary else float(g)) for d, g in qrels_map.items() if float(g) > 0.0}
    if not pos:
        return 0.0
    dcg = 0.0
    for idx, doc_id in enumerate(list(ranked_ids)[:k]):
        key = str(doc_id)
        if key in pos:
            dcg += pos[key] / math.log2(idx + 2.0)
    ideal = sorted(pos.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(idx + 2.0) for idx, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


class TestMetricLineageContract(unittest.TestCase):
    def test_two_positive_one_retrieved_minimal_falsifier(self):
        expected = 1.0 / (1.0 + 1.0 / math.log2(3.0))
        self.assertAlmostEqual(ndcg_at_k(["a", "x"], {"a", "b"}, k=2), expected, places=12)
        self.assertAlmostEqual(drift_ndcg_at_k(["a", "x"], {"a", "b"}, k=2), expected, places=12)
        self.assertNotAlmostEqual(ndcg_at_k(["a", "x"], {"a", "b"}, k=2), 1.0, places=3)

    def test_graded_multi_positive_uses_all_qrels(self):
        # Retrieved a (gain 2) at rank 1, missed b (gain 3).
        expected = (2.0 / math.log2(2.0)) / (3.0 / math.log2(2.0) + 2.0 / math.log2(3.0))
        got = ndcg_at_k(["a", "x"], {"a": 2.0, "b": 3.0}, k=2)
        self.assertAlmostEqual(got, expected, places=12)

    def test_duplicate_ranked_ids_fail_closed_across_all_metrics(self):
        with self.assertRaises(ValueError):
            ndcg_at_k(["a", "a"], {"a"}, k=2)
        with self.assertRaises(ValueError):
            drift_ndcg_at_k(["a", "a"], {"a"}, k=2)
        with self.assertRaises(ValueError):
            mean_average_precision_at_k(["a", "a"], {"a"}, k=2)
        with self.assertRaises(ValueError):
            mean_reciprocal_rank(["a", "a"], {"a"})
        with self.assertRaises(ValueError):
            recall_at_k(["a", "a"], {"a"}, k=2)

    def test_legacy_retrieved_gain_signature_fails_closed(self):
        with self.assertRaises(TypeError):
            ndcg_at_k([1.0, 0.0], 2)  # type: ignore[arg-type]

    def test_edge_cases_and_invalid_k(self):
        for bad_k in (0, -1, True, False):
            with self.assertRaises(ValueError):
                ndcg_at_k(["a"], {"a"}, k=bad_k)  # type: ignore[arg-type]
        self.assertEqual(ndcg_at_k(["a", "b"], set(), k=2), 0.0)
        self.assertEqual(ndcg_at_k(["x", "y"], {"a", "b"}, k=2), 0.0)
        # More positives than k
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"a", "b", "c", "d"}, k=2), 1.0, places=12)
        self.assertEqual(validate_ranked_ids([1, 2, 3], k=2), ["1", "2"])

    def test_randomized_parity_with_independent_reference(self):
        rng = random.Random(20261121)
        universe = [f"d{i}" for i in range(20)]
        for _ in range(500):
            k = rng.randint(1, 10)
            ranked = rng.sample(universe, rng.randint(1, 12))
            pos_docs = rng.sample(universe, rng.randint(0, 8))
            qrels_map = {d: float(rng.randint(1, 3)) for d in pos_docs}
            for binary in (False, True):
                got = ndcg_at_k(ranked, qrels_map, k=k, binary=binary)
                ref = _reference_ndcg(ranked, qrels_map, k=k, binary=binary)
                self.assertAlmostEqual(got, ref, places=12)
                self.assertGreaterEqual(got, 0.0)
                self.assertLessEqual(got, 1.0)
            ap = mean_average_precision_at_k(ranked, set(pos_docs), k=k)
            self.assertGreaterEqual(ap, 0.0)
            self.assertLessEqual(ap, 1.0)

    def test_retrieval_fitness_evaluator_end_to_end_uses_complete_qrels(self):
        expected = 1.0 / (1.0 + 1.0 / math.log2(3.0))
        evaluator = RetrievalFitnessEvaluator(
            qrels={"q1": {"a": 1.0, "b": 1.0}},
            k=2,
        )
        res = evaluator.evaluate_rankings({"q1": ["a", "x"]})
        self.assertAlmostEqual(res.ndcg_at_k, expected, places=12)

    def test_metric_contract_manifest(self):
        manifest = build_metric_contract_manifest(qrels_sha256="0" * 64, k=10)
        self.assertEqual(manifest["contract_version"], METRIC_CONTRACT_VERSION)
        self.assertEqual(manifest["ideal_ranking_population"], "all_positive_qrels")
        self.assertEqual(len(manifest["implementation_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
