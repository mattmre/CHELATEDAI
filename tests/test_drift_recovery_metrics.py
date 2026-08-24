"""Regression tests for drift-recovery metric lineage."""

from __future__ import annotations

import math
import unittest

from drift_recovery_metrics import ndcg_at_k


class DriftRecoveryNDCGTests(unittest.TestCase):
    def test_idcg_uses_unretrieved_positive_qrels(self) -> None:
        expected = 1.0 / (1.0 + 1.0 / math.log2(3.0))
        self.assertAlmostEqual(
            ndcg_at_k(["a", "x"], {"a", "b"}, k=2),
            expected,
            places=15,
        )
        self.assertAlmostEqual(expected, 0.6131471927654584, places=15)

    def test_complete_ideal_ranking_is_one(self) -> None:
        self.assertEqual(ndcg_at_k(["a", "b"], {"a", "b"}, k=2), 1.0)

    def test_zero_hit_and_empty_positive_set_are_zero(self) -> None:
        self.assertEqual(ndcg_at_k(["x", "y"], {"a", "b"}, k=2), 0.0)
        self.assertEqual(ndcg_at_k(["x", "y"], set(), k=2), 0.0)

    def test_more_than_k_positives_uses_k_ideal_gains(self) -> None:
        self.assertEqual(
            ndcg_at_k(["a", "b"], {"a", "b", "c"}, k=2),
            1.0,
        )

    def test_invalid_cutoff_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be a positive integer"):
            ndcg_at_k(["a"], {"a"}, k=0)
        with self.assertRaisesRegex(ValueError, "k must be a positive integer"):
            ndcg_at_k(["a"], {"a"}, k=True)

    def test_duplicate_ranked_document_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate document IDs"):
            ndcg_at_k(["a", "a"], {"a"}, k=2)


if __name__ == "__main__":
    unittest.main()
