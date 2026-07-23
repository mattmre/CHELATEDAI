"""Contract tests for the pre-registered Obj A block-LOO analysis."""

from __future__ import annotations

import unittest

import numpy as np

from research.drift_recovery.artifacts import EmbeddingPack
from research.drift_recovery.estimator.features import leakage_safe_fit_indices
from research.drift_recovery.estimator.objA import (
    NULL_NAMES,
    PRIMARY_FEATURE_NAMES,
    block_loo_analysis,
    partial_spearman_controlling_gap,
)


def _records():
    rows = []
    for dataset_index, dataset in enumerate(("A", "B", "C")):
        for encoder_index, encoder in enumerate(("x", "y")):
            value = float(2 * dataset_index + encoder_index + 1)
            rows.append(
                {
                    "name": f"{dataset}_{encoder}",
                    "dataset": dataset,
                    "encoder_family": encoder,
                    "oracle_margin_mean": value,
                    "oracle_margin_q50": 1000.0 - value,
                    "oracle_margin_sign_rate": float(encoder_index),
                    "oracle_gap": 0.1 * value,
                    "R": 0.2 + 0.05 * value,
                }
            )
    return rows


class TestObjABlockLOO(unittest.TestCase):
    def test_block_loo_never_trains_on_held_out_block(self) -> None:
        records = _records()
        by_name = {row["name"]: row for row in records}
        for block_field in ("dataset", "encoder_family"):
            result = block_loo_analysis(records, block_field)
            for row in result["rows"]:
                held = row["held_out_block"]
                self.assertNotIn(held, row["training_blocks"])
                self.assertTrue(row["training_regimes"])
                self.assertTrue(
                    all(
                        by_name[name][block_field] != held
                        for name in row["training_regimes"]
                    )
                )

    def test_primary_is_univariate_oracle_margin_mean_only(self) -> None:
        self.assertEqual(PRIMARY_FEATURE_NAMES, ("oracle_margin_mean",))
        original = _records()
        mutated = [dict(row) for row in original]
        for index, row in enumerate(mutated):
            row["oracle_margin_q50"] = (-1.0) ** index * 1e12
            row["oracle_margin_sign_rate"] = float(index % 3) / 2.0
        before = block_loo_analysis(original, "dataset")
        after = block_loo_analysis(mutated, "dataset")
        before_predictions = {
            row["regime"]: row["predictions"]["oracle_margin_mean"]
            for row in before["rows"]
        }
        after_predictions = {
            row["regime"]: row["predictions"]["oracle_margin_mean"]
            for row in after["rows"]
        }
        self.assertEqual(before_predictions, after_predictions)

    def test_nulls_use_exactly_the_same_held_out_regimes(self) -> None:
        result = block_loo_analysis(_records(), "encoder_family")
        expected = {row["name"] for row in _records()}
        self.assertEqual({row["regime"] for row in result["rows"]}, expected)
        for row in result["rows"]:
            self.assertEqual(
                set(row["predictions"]),
                {"oracle_margin_mean", *NULL_NAMES},
            )
        self.assertTrue(
            all(metric["cell_n"] == len(expected) for metric in result["metrics"].values())
        )

    def test_leakage_guard_rejects_eval_positive_fit_document(self) -> None:
        pack = EmbeddingPack(
            Do=np.eye(3, dtype=np.float64),
            Dor=np.eye(3, dtype=np.float64),
            Qd=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
            doc_ids=np.asarray(["positive", "safe-a", "safe-b"]),
            query_ids=np.asarray(["q1"]),
            qrels={"q1": {"positive": 1.0}},
            fit_idx=np.asarray([1, 2], dtype=np.int64),
            per_query_scores={"floor": np.asarray([0.0])},
            extra_arrays={"leakage_safe_fit_idx": np.asarray([0, 1], dtype=np.int64)},
        )
        with self.assertRaisesRegex(ValueError, "not leakage-safe"):
            leakage_safe_fit_indices(pack)

    def test_partial_spearman_uses_pearson_of_rank_residuals(self) -> None:
        margin_rank = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        recovery_rank = np.asarray([1.0, 2.0, 4.0, 3.0, 6.0, 5.0])
        gap_rank = np.asarray([1.0, 3.0, 2.0, 6.0, 5.0, 4.0])
        records = [
            {
                "oracle_margin_mean": margin_rank[index],
                "R": recovery_rank[index],
                "oracle_gap": gap_rank[index],
            }
            for index in range(6)
        ]
        actual = partial_spearman_controlling_gap(records)[
            "partial_spearman_margin_R_controlling_oracle_gap"
        ]
        design = np.column_stack([np.ones(6), gap_rank])
        margin_residual = margin_rank - design @ np.linalg.lstsq(
            design, margin_rank, rcond=None
        )[0]
        recovery_residual = recovery_rank - design @ np.linalg.lstsq(
            design, recovery_rank, rcond=None
        )[0]
        margin_residual -= np.mean(margin_residual)
        recovery_residual -= np.mean(recovery_residual)
        denominator = np.sqrt(
            (margin_residual @ margin_residual)
            * (recovery_residual @ recovery_residual)
        )
        expected = (
            float(margin_residual @ recovery_residual / denominator)
            if denominator > 0.0
            else None
        )
        self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
