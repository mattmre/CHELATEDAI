from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from research.drift_recovery.artifacts import EmbeddingPack
from research.drift_recovery.d1.paper_stats import audit_fit_leakage, compute_leakage_sensitivity
from research.drift_recovery.harness_bridge import assert_harness_parity
from research.drift_recovery.stats.learning_curve import nested_prefixes
from research.drift_recovery.stats.multiple_testing import holm_adjust
from research.drift_recovery.stats.paired_bootstrap import (
    BootstrapDraws,
    paired_query_bootstrap,
    strip_draws,
)


class TestPairedBootstrap(unittest.TestCase):
    def test_recovery_uses_ratio_of_resampled_means(self) -> None:
        scores = {
            "floor": np.array([0.0, 0.4, 0.0, 0.4]),
            "oracle": np.array([1.0, 0.6, 1.0, 0.6]),
            "method": np.array([0.5, 0.5, 0.5, 0.5]),
        }
        result = paired_query_bootstrap(scores, ["method"], draws=1000, seed=9)
        self.assertAlmostEqual(result["methods"]["method"]["recovery"]["estimate"], 0.5)
        self.assertNotIn("_method_draws", strip_draws(result))

    def test_floor_oracle_and_methods_share_bootstrap_indices(self) -> None:
        scores = {
            "floor": np.array([0.0, 0.1, 0.2]),
            "oracle": np.array([0.7, 0.8, 0.9]),
            "ridge": np.array([0.4, 0.5, 0.6]),
            "mlp": np.array([0.3, 0.4, 0.5]),
        }
        indices = np.tile(np.array([[0, 1, 2], [2, 2, 0]]), (50, 1))
        draws = BootstrapDraws(indices=indices, seed=17)

        result = paired_query_bootstrap(
            scores,
            ["ridge", "mlp"],
            bootstrap_draws=draws,
        )

        np.testing.assert_array_equal(result["_bootstrap_indices"], indices)
        for name, values in scores.items():
            expected = values[indices].mean(axis=1)
            np.testing.assert_allclose(result["_resampled_means"][name], expected)

    def test_invalid_draw_gate_includes_epsilon_and_hard_fails_above_one_percent(self) -> None:
        scores = {
            "floor": np.array([0.0, 0.0]),
            "oracle": np.array([0.1, 1.0]),
            "method": np.array([0.05, 0.5]),
        }
        indices = np.tile(np.array([[0, 1]]), (100, 1))
        indices[0] = [0, 0]
        one_percent = BootstrapDraws(indices=indices, seed=23)

        result = paired_query_bootstrap(
            scores,
            ["method"],
            epsilon=0.1,
            bootstrap_draws=one_percent,
        )

        self.assertEqual(result["invalid_draws"], 1)
        self.assertAlmostEqual(result["invalid_fraction"], 0.01)

        indices[1] = [0, 0]
        above_limit = BootstrapDraws(indices=indices, seed=23)
        with self.assertRaisesRegex(RuntimeError, "exceeds the 1% protocol limit"):
            paired_query_bootstrap(
                scores,
                ["method"],
                epsilon=0.1,
                bootstrap_draws=above_limit,
            )

    def test_holm_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
        values = [adjusted[key]["holm_adjusted_p"] for key in ("a", "b", "c")]
        self.assertEqual(values, sorted(values))


class TestLearningCurve(unittest.TestCase):
    def test_prefixes_are_nested_and_seeded(self) -> None:
        first = nested_prefixes(32, counts=(4, 8, 16), seeds=(1, 2))
        second = nested_prefixes(32, counts=(4, 8, 16), seeds=(1, 2))
        for seed in (1, 2):
            self.assertTrue(set(first[seed][4]).issubset(set(first[seed][8])))
            np.testing.assert_array_equal(first[seed][16], second[seed][16])


class TestFrozenPack(unittest.TestCase):
    @staticmethod
    def _load_pack() -> EmbeddingPack:
        prefix = Path("research/drift_recovery/out/d1/scifact_evalsplit_pack")
        if not prefix.with_suffix(".npz").exists():
            raise unittest.SkipTest("D1 pack has not been built")
        return EmbeddingPack.load(prefix)

    def test_real_pack_reloads_and_matches_harness_aggregates(self) -> None:
        pack = self._load_pack()
        differences = assert_harness_parity(pack, atol=1e-12)
        self.assertTrue(differences)
        self.assertTrue(all(value == 0.0 for value in differences.values()))
        safe = pack.extra_arrays["leakage_safe_fit_idx"]
        self.assertEqual(len(safe), 600)
        self.assertEqual(
            pack.metadata["waypoint_audit"]["eval_positive_docs_in_leakage_safe_fit"],
            0,
        )

    def test_literal_and_safe_fit_leakage_counts_are_recomputed_from_pack(self) -> None:
        audit = audit_fit_leakage(self._load_pack())

        self.assertEqual(audit["eval_positive_docs_total"], 62)
        self.assertEqual(audit["eval_positive_docs_in_literal_fit"], 28)
        self.assertEqual(audit["eval_positive_docs_in_leakage_safe_fit"], 0)

    def test_leakage_safe_ridge_recovery_is_below_literal_fit(self) -> None:
        sensitivity = compute_leakage_sensitivity(
            self._load_pack(),
            methods=("ridge",),
            device="cpu",
        )
        ridge = sensitivity["methods"]["ridge"]

        self.assertLess(
            ridge["leakage_safe_full_600"]["recovery"],
            ridge["literal_waypoint"]["recovery"],
        )


if __name__ == "__main__":
    unittest.main()
