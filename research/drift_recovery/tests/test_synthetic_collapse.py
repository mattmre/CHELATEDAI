from __future__ import annotations

import unittest
from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from research.drift_recovery.d2.crossover import (
    audit_anchor_indices,
    _prepare_output_dir,
    build_detector_assignments,
    build_leakage_safe_anchor_indices,
)
from research.drift_recovery.d2.detector_evaluation import (
    average_precision,
    cluster_counterfactual_harm,
    evaluate_detector,
)
from research.drift_recovery.harness_bridge import assert_score_transform_parity
from research.drift_recovery.methods import (
    AffineRidgeAdapter,
    DetectorGatedChelationAdapter,
    HubnessScoreScaling,
    SoftRoutedLocalAdapter,
)
from research.drift_recovery.regimes.synthetic_collapse import (
    deterministic_kmeans,
    inject_semantic_collapse,
)


class TestSyntheticCollapse(unittest.TestCase):
    @staticmethod
    def _documents() -> np.ndarray:
        rng = np.random.default_rng(20260710)
        return rng.normal(size=(256, 12))

    def test_exact_8_and_16_cluster_selection_is_deterministic(self) -> None:
        documents = self._documents()

        for collapsed_count in (8, 16):
            with self.subTest(collapsed_count=collapsed_count):
                first = inject_semantic_collapse(
                    documents,
                    collapse_cluster_count=collapsed_count,
                    beta=0.10,
                    seed=1337,
                    total_clusters=32,
                )
                second = inject_semantic_collapse(
                    documents,
                    collapse_cluster_count=collapsed_count,
                    beta=0.10,
                    seed=1337,
                    total_clusters=32,
                )

                self.assertEqual(len(first.collapsed_clusters), collapsed_count)
                self.assertEqual(len(np.unique(first.collapsed_clusters)), collapsed_count)
                np.testing.assert_array_equal(
                    first.collapsed_clusters, second.collapsed_clusters
                )
                np.testing.assert_array_equal(first.assignments, second.assignments)
                np.testing.assert_allclose(first.centroids, second.centroids, rtol=0.0, atol=0.0)
                np.testing.assert_allclose(first.corrupted, second.corrupted, rtol=0.0, atol=0.0)

    def test_beta_point_one_pulls_selected_rows_and_leaves_other_rows_identity(self) -> None:
        documents = self._documents()
        original = documents.copy()
        result = inject_semantic_collapse(
            documents,
            collapse_cluster_count=8,
            beta=0.10,
            seed=42,
            total_clusters=32,
        )
        selected = result.changed_mask
        expected_selected = result.clean[selected] + 0.10 * (
            result.centroids[result.assignments[selected]] - result.clean[selected]
        )

        np.testing.assert_array_equal(documents, original)
        np.testing.assert_array_equal(result.corrupted[~selected], result.clean[~selected])
        np.testing.assert_allclose(
            result.corrupted[selected], expected_selected, rtol=0.0, atol=1e-14
        )
        self.assertTrue(np.any(np.linalg.norm(result.corrupted - result.clean, axis=1) > 0.0))

        before = np.linalg.norm(
            result.clean[selected] - result.centroids[result.assignments[selected]], axis=1
        )
        after = np.linalg.norm(
            result.corrupted[selected] - result.centroids[result.assignments[selected]], axis=1
        )
        np.testing.assert_allclose(after, 0.90 * before, rtol=1e-12, atol=1e-12)

    def test_invalid_collapse_inputs_fail_closed(self) -> None:
        documents = self._documents()
        cases = (
            {"collapse_cluster_count": 0},
            {"collapse_cluster_count": 33},
            {"collapse_cluster_count": 8, "beta": 0.0},
            {"collapse_cluster_count": 8, "beta": 1.0},
            {"collapse_cluster_count": 8, "total_clusters": len(documents) + 1},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                inject_semantic_collapse(documents, **kwargs)

        contaminated = documents.copy()
        contaminated[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            inject_semantic_collapse(contaminated, collapse_cluster_count=8)
        with self.assertRaises(ValueError):
            deterministic_kmeans(documents, cluster_count=1)
        with self.assertRaises(ValueError):
            deterministic_kmeans(documents, iterations=0)


class TestHarmLabels(unittest.TestCase):
    def test_counterfactual_labels_measure_ndcg_loss_not_injection_membership(self) -> None:
        clean = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        corrupted = np.asarray(
            [
                [-1.0, 0.0],
                [0.0, 1.0],
                [-0.9, -0.1],
                [-0.1, -0.9],
            ]
        )
        assignments = np.asarray([0, 0, 1, 1])
        losses = cluster_counterfactual_harm(
            clean_documents=clean,
            corrupted_documents=corrupted,
            assignments=assignments,
            query_vectors=np.asarray([[1.0, 0.0]]),
            query_ids=["q0"],
            doc_ids=["d0", "d1", "d2", "d3"],
            qrels={"q0": {"d0": 1.0}},
            k=1,
            cluster_count=2,
        )

        # Both clusters contain changed (injected) rows, but only cluster zero
        # causes retrieval loss in its one-cluster counterfactual.
        self.assertGreater(losses[0], 0.0)
        self.assertEqual(losses[1], 0.0)
        self.assertNotEqual((losses > 1e-8).tolist(), [True, True])

    def test_detector_rejects_train_eval_query_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "query leakage"):
            evaluate_detector(
                features=np.arange(8, dtype=np.float64).reshape(4, 2),
                validation_harm_losses=np.asarray([0.0, 0.0, 1.0, 1.0]),
                eval_harm_losses=np.asarray([0.0, 0.0, 1.0, 1.0]),
                validation_query_ids=["anchor-0", "shared"],
                eval_query_ids=["shared", "eval-0"],
                gate_candidates=[0.4, 0.5, 0.6],
            )

    def test_auprc_is_wired_to_eval_harm_and_single_class_is_undefined(self) -> None:
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        validation_harm = np.asarray([0.0, 0.0, 1.0, 1.0])
        aligned = evaluate_detector(
            features=features,
            validation_harm_losses=validation_harm,
            eval_harm_losses=np.asarray([0.0, 0.0, 1.0, 1.0]),
            validation_query_ids=["anchor-0", "anchor-1"],
            eval_query_ids=["eval-0", "eval-1"],
            gate_candidates=[0.4, 0.5, 0.6],
        )
        reversed_labels = evaluate_detector(
            features=features,
            validation_harm_losses=validation_harm,
            eval_harm_losses=np.asarray([1.0, 1.0, 0.0, 0.0]),
            validation_query_ids=["anchor-0", "anchor-1"],
            eval_query_ids=["eval-0", "eval-1"],
            gate_candidates=[0.4, 0.5, 0.6],
        )

        self.assertEqual(aligned["auprc_status"], "ok")
        self.assertAlmostEqual(aligned["auprc"], 1.0)
        self.assertLess(reversed_labels["auprc"], aligned["auprc"])
        self.assertEqual(
            aligned["label_definition"], "counterfactual cluster NDCG drop > 1e-08"
        )

        score, status = average_precision(
            np.ones(4, dtype=np.float64), np.linspace(0.1, 0.9, 4)
        )
        self.assertIsNone(score)
        self.assertEqual(status, "undefined_single_class")

    def test_tied_score_average_precision_is_prevalence_and_permutation_invariant(self) -> None:
        harm = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        probabilities = np.zeros(6, dtype=np.float64)
        first, status = average_precision(harm, probabilities)
        permutation = np.asarray([5, 3, 1, 4, 0, 2])
        second, _ = average_precision(harm[permutation], probabilities[permutation])

        self.assertEqual(status, "ok")
        self.assertAlmostEqual(first, 2.0 / 6.0)
        self.assertAlmostEqual(second, first)


class TestD2MethodContracts(unittest.TestCase):
    def test_detector_partition_is_built_from_corrupted_vectors_only(self) -> None:
        corrupted = np.arange(24, dtype=np.float64).reshape(8, 3)
        expected_assignments = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        with patch(
            "research.drift_recovery.d2.crossover.deterministic_kmeans",
            return_value=(expected_assignments, np.zeros((2, 3))),
        ) as mocked:
            actual = build_detector_assignments(corrupted, cluster_count=2, seed=42)

        np.testing.assert_array_equal(actual, expected_assignments)
        np.testing.assert_array_equal(mocked.call_args.args[0], corrupted)
        self.assertNotIn("clean", signature(build_detector_assignments).parameters)

    def test_unpaired_chelation_target_invariance_alpha_cap_and_parameter_budget(self) -> None:
        rng = np.random.default_rng(19)
        source = rng.normal(size=(16, 4))
        target = rng.normal(size=(16, 4))
        assignments = np.repeat(np.arange(2), 8)
        probabilities = np.asarray([0.9, 0.8])
        kwargs = dict(
            dimension=4,
            global_rank=1,
            local_rank=1,
            cluster_count=2,
            alpha=0.05,
            radial_gain=0.2,
            detector_threshold=0.5,
        )
        without_target = DetectorGatedChelationAdapter(**kwargs).fit(
            source,
            cluster_assignments=assignments,
            detector_probabilities=probabilities,
        )
        with_target = DetectorGatedChelationAdapter(**kwargs).fit(
            source,
            target=target,
            cluster_assignments=assignments,
            detector_probabilities=probabilities,
        )
        first = without_target.transform(source)
        second = with_target.transform(source)
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
        relative = np.linalg.norm(first - source, axis=1) / np.linalg.norm(source, axis=1)
        self.assertLessEqual(float(relative.max()), 0.05 + 1e-12)

        local = SoftRoutedLocalAdapter(4, 1, 1, 2).fit(source, target)
        self.assertEqual(local.allocated_parameter_count, without_target.allocated_parameter_count)
        self.assertNotIn("qrels", signature(AffineRidgeAdapter.fit).parameters)
        self.assertNotIn("qrels", signature(SoftRoutedLocalAdapter.fit).parameters)
        self.assertNotIn("qrels", signature(DetectorGatedChelationAdapter.fit).parameters)

    def test_score_transform_ranking_uses_harness_parity_boundary(self) -> None:
        anchor_scores = np.asarray([[1.0, 0.0], [0.9, 0.1]])
        transform = HubnessScoreScaling(reference_k=1, strength=0.0).fit(anchor_scores)
        rankings = transform.rank(np.asarray([[1.0, 0.0]]), k=1)
        parity = assert_score_transform_parity(
            rankings,
            np.asarray([1.0]),
            1.0,
            ["q"],
            ["relevant", "other"],
            {"q": {"relevant": 1.0}},
            k=1,
        )
        self.assertEqual(parity["per_query_max_abs"], 0.0)
        self.assertEqual(parity["aggregate_abs"], 0.0)


class TestLeakageSafeAnchorPool(unittest.TestCase):
    def test_pool_excludes_validation_and_eval_positive_docs_deterministically(self) -> None:
        doc_ids = [f"d{index}" for index in range(10)]
        validation_qrels = {
            "v0": {"d1": 1.0, "d3": 2.0, "d8": 0.0},
        }
        eval_qrels = {
            "e0": {"d5": 1.0},
            "e1": {"d7": 1.0},
        }
        first = build_leakage_safe_anchor_indices(
            doc_ids, validation_qrels, eval_qrels, fraction=0.5, seed=101
        )
        second = build_leakage_safe_anchor_indices(
            doc_ids, validation_qrels, eval_qrels, fraction=0.5, seed=101
        )
        selected_ids = {doc_ids[index] for index in first}

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(selected_ids.isdisjoint({"d1", "d3", "d5", "d7"}))
        audit = audit_anchor_indices(first, doc_ids, validation_qrels, eval_qrels)
        self.assertEqual(audit["positive_documents_in_fit"], 0)
        self.assertEqual(audit["known_positive_document_count"], 4)

    def test_anchor_pool_invalid_inputs_fail_closed(self) -> None:
        doc_ids = ["d0", "d1", "d2"]
        for fraction in (0.0, -0.1, 1.1):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                build_leakage_safe_anchor_indices(
                    doc_ids, {}, {}, fraction=fraction, seed=42
                )
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            build_leakage_safe_anchor_indices(
                doc_ids,
                {"v": {"d0": 1.0}},
                {"e": {"d1": 1.0}},
                fraction=1.0,
                seed=42,
            )

    def test_force_output_rejects_paths_outside_drift_recovery_out(self) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "must be exactly"):
                _prepare_output_dir(Path(temporary) / "d2", force=True)


if __name__ == "__main__":
    unittest.main()
