"""Focused D3 estimator tests; runnable with the repository's unittest gate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.drift_recovery.artifacts import EmbeddingPack
from research.drift_recovery.estimator.calibration import FEATURE_NAMES, fit_calibration
from research.drift_recovery.estimator.features import extract_regime_features
from research.drift_recovery.estimator.margin_bound import (
    order_is_preserved,
    order_preservation_slack,
    predict_inversion,
)
from research.drift_recovery.estimator.predictor import estimate_recovery
from research.drift_recovery.estimator.validation import leave_one_regime_out
from research.drift_recovery.run_estimator import (
    RegimeSpec,
    RegimeUnavailableError,
    frozen_ridge_ground_truth,
    run_d3,
)


def _tiny_pack() -> EmbeddingPack:
    # q=[1,0], oracle relevant=[1,0], and oracle competitor=[0.6,0.8],
    # hence the hand-checkable clean cosine margin is 1.0 - 0.6 = 0.4.
    oracle = np.asarray(
        [
            [1.0, 0.0],
            [0.6, 0.8],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float64,
    )
    old = np.asarray(
        [
            [0.8, 0.2],
            [0.4, 0.9],
            [-0.9, 0.1],
            [0.1, -0.8],
        ],
        dtype=np.float64,
    )
    return EmbeddingPack(
        Do=old,
        Dor=oracle,
        Qd=np.asarray([[1.0, 0.0]], dtype=np.float64),
        doc_ids=np.asarray(["relevant", "competitor", "fit-a", "fit-b"]),
        query_ids=np.asarray(["q1"]),
        qrels={"q1": {"relevant": 1.0}},
        # The legacy/literal index is deliberately contaminated.  D3 must use
        # the explicit safe index below and never pass qrels to ridge fitting.
        fit_idx=np.asarray([0, 1, 2], dtype=np.int64),
        per_query_scores={
            "floor": np.asarray([0.0]),
            "oracle": np.asarray([1.0]),
            "ridge": np.asarray([0.5]),
        },
        extra_arrays={
            "leakage_safe_fit_idx": np.asarray([1, 2, 3], dtype=np.int64),
            "method_documents__ridge": old.copy(),
        },
        metadata={
            "dataset": "synthetic",
            "seed": 7,
            "anchor_fraction": 0.4,
            "k": 10,
            "models": {"old": "synthetic-old", "swap": "synthetic-swap"},
            "harness_aggregate_ndcg": {"floor": 0.0, "oracle": 1.0, "ridge": 0.5},
        },
    )


def _calibration_features(value: float):
    return {"summary": {name: float(value) for name in FEATURE_NAMES}}


class TestMarginBound(unittest.TestCase):
    def test_hand_checkable_margin_and_tie_semantics(self) -> None:
        slack = order_preservation_slack(0.6, 1.0, 0.1, 0.2)
        self.assertAlmostEqual(float(slack), 0.3)
        self.assertTrue(bool(order_is_preserved(0.6, 1.0, 0.1, 0.2)))
        self.assertFalse(bool(predict_inversion(0.6, 1.0, 0.1, 0.2)))

        self.assertAlmostEqual(float(order_preservation_slack(0.3, 1.0, 0.1, 0.2)), 0.0)
        self.assertTrue(bool(predict_inversion(0.3, 1.0, 0.1, 0.2)))

    def test_negative_error_norm_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            predict_inversion(0.5, 1.0, -0.1, 0.2)


class TestEstimatorFeatures(unittest.TestCase):
    def test_synthetic_pack_margin_is_hand_checkable_and_deterministic(self) -> None:
        first = extract_regime_features(_tiny_pack())
        second = extract_regime_features(_tiny_pack())

        self.assertEqual(
            json.dumps(first, sort_keys=True, allow_nan=False),
            json.dumps(second, sort_keys=True, allow_nan=False),
        )
        self.assertAlmostEqual(first["per_query"]["oracle_margin"][0], 0.4)
        self.assertEqual(first["per_query"]["relevant_doc_ids"], ["relevant"])
        self.assertEqual(first["per_query"]["competitor_doc_ids"], ["competitor"])

    def test_synthetic_pack_bound_slack_and_inversion_are_end_to_end_consistent(self) -> None:
        # C2-F5: verify the order-preservation bound is composed correctly from its
        # extracted components, not just that the raw margin is 0.4.  For query 0:
        #   slack = margin - ||q|| * (||e_r|| + ||e_j||),  inversion <=> slack <= 0.
        per_query = extract_regime_features(_tiny_pack())["per_query"]
        margin = per_query["oracle_margin"][0]
        qn = per_query["query_norm"][0]
        er = per_query["relevant_error_norm"][0]
        ej = per_query["competitor_error_norm"][0]
        expected_slack = margin - qn * (er + ej)
        self.assertAlmostEqual(per_query["bound_slack"][0], expected_slack)
        self.assertAlmostEqual(per_query["bound_rhs"][0], qn * (er + ej))
        self.assertEqual(
            bool(per_query["predicted_inversion"][0]),
            bool(expected_slack <= 0.0),
        )
        # Independently recompute the slack via the public bound API and confirm
        # it matches the extracted per-query slack.
        self.assertAlmostEqual(
            float(order_preservation_slack(margin, qn, er, ej)),
            per_query["bound_slack"][0],
        )
        self.assertEqual(
            bool(predict_inversion(margin, qn, er, ej)),
            bool(per_query["predicted_inversion"][0]),
        )

    def test_ridge_fit_receives_only_safe_arrays_and_index_not_qrels(self) -> None:
        captured = {}

        def fake_fit(source, target, fit_idx, regularization=1.0):
            captured["source"] = np.asarray(source).copy()
            captured["target"] = np.asarray(target).copy()
            captured["fit_idx"] = np.asarray(fit_idx).copy()
            captured["regularization"] = regularization
            return np.eye(np.asarray(source).shape[1], dtype=np.float64)

        with patch(
            "research.drift_recovery.estimator.features.fit_ridge_map",
            side_effect=fake_fit,
        ) as mocked:
            features = extract_regime_features(_tiny_pack())

        np.testing.assert_array_equal(captured["fit_idx"], np.asarray([1, 2, 3]))
        np.testing.assert_array_equal(captured["source"], _tiny_pack().Do)
        np.testing.assert_array_equal(captured["target"], _tiny_pack().Dor)
        self.assertNotIn("qrels", mocked.call_args.kwargs)
        self.assertEqual(features["summary"]["eval_positive_docs_in_fit"], 0)
        self.assertEqual(
            features["summary"]["fit_index_source"],
            "extra_arrays.leakage_safe_fit_idx",
        )


class TestEstimatorCalibration(unittest.TestCase):
    def test_ground_truth_uses_frozen_pack_ridge_scores(self) -> None:
        ground_truth = frozen_ridge_ground_truth(_tiny_pack())

        self.assertEqual(ground_truth["ground_truth_source"], "frozen pack per_query_scores.ridge")
        self.assertAlmostEqual(ground_truth["ridge_recovery"], 0.5)
        self.assertIn("leakage_safe_feature_refit", ground_truth)

    def test_leave_one_regime_out_never_trains_on_held_out_target(self) -> None:
        regimes = [
            {"name": f"r{index}", "features": _calibration_features(value), "recovery": recovery}
            for index, (value, recovery) in enumerate(
                ((0.1, 0.15), (0.3, 0.35), (0.7, 0.72), (0.9, 0.88)), start=1
            )
        ]
        result = leave_one_regime_out(regimes)

        self.assertEqual(result["held_out_count"], 4)
        self.assertEqual(len(result["rows"]), 4)
        for row in result["rows"]:
            self.assertNotIn(row["regime"], row["training_regimes"])
            self.assertEqual(len(row["training_regimes"]), 3)
            self.assertIsInstance(row["R_hat"], float)

    def test_predictor_returns_calibrated_shape_without_ndcg_eval(self) -> None:
        records = [_calibration_features(0.2), _calibration_features(0.8)]
        calibration = fit_calibration(records, [0.25, 0.75], regime_names=["a", "b"])
        result = estimate_recovery(_tiny_pack(), calibration=calibration.to_dict())

        self.assertIn("R_hat", result)
        self.assertIn("lower_band", result)
        self.assertIn("features", result)
        self.assertLessEqual(result["lower_band"], result["R_hat"])

    def test_runner_does_not_mask_parity_failure_as_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "existing_pack"
            prefix.with_suffix(".npz").write_bytes(b"present")
            spec = RegimeSpec(
                name="parity_failure",
                dataset="synthetic",
                swap_model="synthetic",
                existing_pack=prefix,
            )
            with patch("research.drift_recovery.run_estimator.REGIMES", (spec,)), patch(
                "research.drift_recovery.run_estimator.EmbeddingPack.load",
                return_value=_tiny_pack(),
            ), patch(
                "research.drift_recovery.run_estimator.assert_harness_parity",
                side_effect=AssertionError("parity must fail loudly"),
            ):
                with self.assertRaisesRegex(AssertionError, "parity must fail loudly"):
                    run_d3(output_dir=root / "out", device="cpu")

    def test_runner_records_only_explicit_availability_failure_as_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = RegimeSpec(
                name="missing_cache",
                dataset="synthetic",
                swap_model="synthetic",
            )
            with patch("research.drift_recovery.run_estimator.REGIMES", (spec,)), patch(
                "research.drift_recovery.run_estimator.build_regime_pack",
                side_effect=RegimeUnavailableError("offline model cache unavailable"),
            ):
                result = run_d3(output_dir=root / "out", device="cpu")

            self.assertEqual(result["available_regimes"], [])
            manifest = json.loads((root / "out" / "regime_manifest.json").read_text())
            self.assertEqual(manifest["regimes"][0]["status"], "skipped")
            self.assertIn("RegimeUnavailableError", manifest["regimes"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
