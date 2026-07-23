"""Adversarial contract tests for the Obj B CPU preflight falsifier."""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from research.drift_recovery.d2b_preflight import preflight as preflight_module
from research.drift_recovery.d2b_preflight.preflight import (
    apply_local_ridges,
    fit_global_ridge,
    fit_local_ridges,
    infer_corrupted_clusters,
    select_ridge_hyperparameters,
)
from research.drift_recovery.regimes.sparse_local_nonaffine import (
    apply_cluster_warp,
    generate_sparse_local_regime,
)


class TestSparseLocalWarp(unittest.TestCase):
    def test_both_warp_families_are_sparse_nonaffine_and_nonradial(self) -> None:
        for family, canonical_family in (
            ("quadratic_form", "quadratic"),
            ("soft_fold", "soft_fold"),
        ):
            with self.subTest(family=family):
                geometry_regime = None
                for sparsity, expected_harmed in ((0.15, 3), (0.25, 5)):
                    regime = generate_sparse_local_regime(
                        seed=20260710,
                        n_clusters=20,
                        docs_per_cluster=30,
                        dimension=8,
                        sparsity=sparsity,
                        gamma=0.25,
                        family=family,
                        anchors_per_harmed_cluster=8,
                        queries_per_cluster=2,
                    )
                    harmed = set(int(value) for value in regime.harmed_clusters)
                    self.assertEqual(len(harmed), expected_harmed)
                    self.assertEqual(regime.warp_family, canonical_family)
                    clean_mask = ~np.isin(regime.clean_cluster_ids, list(harmed))
                    harmed_mask = ~clean_mask
                    np.testing.assert_array_equal(
                        regime.corrupted_docs[clean_mask], regime.clean_docs[clean_mask]
                    )
                    self.assertTrue(
                        np.any(
                            np.linalg.norm(
                                regime.corrupted_docs[harmed_mask]
                                - regime.clean_docs[harmed_mask],
                                axis=1,
                            )
                            > 0.0
                        )
                    )
                    self.assertGreater(regime.displacement_mean, 0.0)
                    self.assertGreaterEqual(
                        regime.displacement_max, regime.displacement_mean
                    )
                    if sparsity == 0.25:
                        geometry_regime = regime

                self.assertIsNotNone(geometry_regime)
                regime = geometry_regime
                harmed = set(int(value) for value in regime.harmed_clusters)
                cluster_id = min(harmed)
                cluster_mask = regime.clean_cluster_ids == cluster_id
                points = regime.clean_docs[cluster_mask]
                parameters = next(
                    value
                    for value in regime.warp_parameters
                    if value.cluster_id == cluster_id
                )
                warped = apply_cluster_warp(
                    points, parameters, gamma=0.35
                )
                displacement = warped - points

                # A genuine non-affine warp must leave error after the best
                # affine fit, rather than merely differing from identity.
                design = np.column_stack([points, np.ones(len(points))])
                affine = design @ np.linalg.lstsq(design, warped, rcond=None)[0]
                nonlinear_residual = np.linalg.norm(warped - affine)
                warp_size = np.linalg.norm(displacement)
                self.assertGreater(warp_size, 0.0)
                self.assertGreater(nonlinear_residual / warp_size, 0.02)

                # A radial warp has every displacement parallel to the point's
                # centered radius. Require a material orthogonal component.
                centered = points - np.mean(points, axis=0, keepdims=True)
                centered_norm_sq = np.sum(centered * centered, axis=1)
                projection_scale = np.sum(displacement * centered, axis=1) / np.maximum(
                    centered_norm_sq, 1e-15
                )
                radial_projection = projection_scale[:, None] * centered
                nonradial_fraction = np.linalg.norm(
                    displacement - radial_projection, axis=1
                ) / np.maximum(np.linalg.norm(displacement, axis=1), 1e-15)
                moved = np.linalg.norm(displacement, axis=1) > 1e-12
                self.assertGreater(float(np.median(nonradial_fraction[moved])), 0.10)


class TestPreflightLeakageContracts(unittest.TestCase):
    def test_corrupted_partition_and_local_fit_have_no_clean_id_input(self) -> None:
        partition_parameters = set(
            inspect.signature(infer_corrupted_clusters).parameters
        )
        self.assertIn("corrupted_docs", partition_parameters)
        for forbidden in (
            "clean_cluster_ids",
            "clean_assignments",
            "oracle_cluster_ids",
            "oracle_assignments",
            "true_cluster_ids",
        ):
            self.assertNotIn(forbidden, partition_parameters)

        fit_parameters = set(inspect.signature(fit_local_ridges).parameters)
        self.assertIn("routing_labels", fit_parameters)
        for forbidden in (
            "clean_cluster_ids",
            "clean_assignments",
            "oracle_cluster_ids",
            "oracle_assignments",
            "true_cluster_ids",
        ):
            self.assertNotIn(forbidden, fit_parameters)

        corrupted = np.vstack(
            [
                np.full((12, 4), -3.0),
                np.full((12, 4), 3.0),
            ]
        )
        first = infer_corrupted_clusters(corrupted, k=2, seed=7)
        second = infer_corrupted_clusters(corrupted, k=2, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 2)

    def test_run_cell_routes_local_fit_with_corrupted_space_assignments(self) -> None:
        captured = {}
        local_routing_labels = []
        real_generate = preflight_module.generate_sparse_local_regime
        real_infer = preflight_module.infer_corrupted_clusters
        real_fit_local = preflight_module.fit_local_ridges

        def capture_generate(**kwargs):
            regime = real_generate(**kwargs)
            captured["regime"] = regime
            return regime

        def capture_infer(corrupted_docs, k, seed=0):
            assignments = real_infer(corrupted_docs, k, seed=seed)
            captured["corrupted"] = np.asarray(corrupted_docs).copy()
            captured["assignments"] = assignments.copy()
            return assignments

        def capture_local(source, target, routing_labels, **kwargs):
            local_routing_labels.append(np.asarray(routing_labels).copy())
            return real_fit_local(source, target, routing_labels, **kwargs)

        with patch.object(
            preflight_module,
            "generate_sparse_local_regime",
            side_effect=capture_generate,
        ), patch.object(
            preflight_module,
            "infer_corrupted_clusters",
            side_effect=capture_infer,
        ), patch.object(
            preflight_module,
            "fit_local_ridges",
            side_effect=capture_local,
        ):
            result = preflight_module.run_preflight_cell(
                family="quadratic",
                gamma=0.20,
                anchors_per_harmed_cluster=12,
                sparse_fraction=0.25,
                seed=73,
                n_docs=160,
                dim=6,
                n_clusters=8,
                n_queries_per_cluster=2,
            )

        regime = captured["regime"]
        np.testing.assert_array_equal(captured["corrupted"], regime.corrupted_docs)
        expected_anchor_routing = captured["assignments"][regime.anchor_indices]
        self.assertEqual(len(local_routing_labels), 2)
        for routing_labels in local_routing_labels:
            np.testing.assert_array_equal(routing_labels, expected_anchor_routing)
        self.assertFalse(result["detector"]["clean_ids_used_by_fit"])

    def test_lambda_and_rank_selection_ignore_eval_values(self) -> None:
        rng = np.random.default_rng(41)
        source = rng.normal(size=(48, 6))
        target = source @ rng.normal(size=(6, 6)) + 0.02 * rng.normal(size=(48, 6))
        fit_indices = np.arange(0, 24, dtype=np.int64)
        dev_indices = np.arange(24, 36, dtype=np.int64)
        eval_indices = np.arange(36, 48, dtype=np.int64)

        poisoned_target = target.copy()
        poisoned_target[eval_indices] = rng.normal(
            loc=1e9, scale=1e8, size=(len(eval_indices), target.shape[1])
        )
        kwargs = dict(
            fit_indices=fit_indices,
            dev_indices=dev_indices,
            eval_indices=eval_indices,
            lambdas=(1e-6, 1e-3, 1.0, 100.0),
            ranks=(1, 2, 4, 6),
        )
        clean_audit = select_ridge_hyperparameters(source, target, **kwargs)
        poisoned_audit = select_ridge_hyperparameters(
            source, poisoned_target, **kwargs
        )

        self.assertEqual(clean_audit.selected_lambda, poisoned_audit.selected_lambda)
        self.assertEqual(clean_audit.selected_rank, poisoned_audit.selected_rank)
        np.testing.assert_array_equal(clean_audit.fit_indices, fit_indices)
        np.testing.assert_array_equal(clean_audit.dev_indices, dev_indices)
        np.testing.assert_array_equal(clean_audit.eval_indices, eval_indices)
        self.assertTrue(set(fit_indices).isdisjoint(dev_indices))
        self.assertTrue(set(fit_indices).isdisjoint(eval_indices))
        self.assertTrue(set(dev_indices).isdisjoint(eval_indices))

        routing_labels = np.tile(np.arange(4, dtype=np.int64), 12)
        local_kwargs = dict(
            routing_labels=routing_labels,
            train_indices=fit_indices,
            dev_indices=dev_indices,
            eval_indices=eval_indices,
            lambdas=(1e-6, 1e-3, 1.0, 100.0),
            candidate_ranks=(1, 2, 4, 6),
        )
        clean_local = fit_local_ridges(source, target, **local_kwargs)
        poisoned_local = fit_local_ridges(source, poisoned_target, **local_kwargs)
        self.assertEqual(clean_local.selected_lambdas, poisoned_local.selected_lambdas)
        self.assertEqual(clean_local.selected_rank, poisoned_local.selected_rank)
        self.assertEqual(clean_local.dev_mse, poisoned_local.dev_mse)
        self.assertEqual(
            clean_local.training_audit["selection_source"],
            "anchor_dev_mse_only",
        )

    def test_global_ridge_fails_while_oracle_id_local_fit_partially_recovers(self) -> None:
        for family in ("quadratic", "soft_fold"):
            with self.subTest(family=family):
                regime = generate_sparse_local_regime(
                    seed=818,
                    n_clusters=8,
                    docs_per_cluster=100,
                    dimension=8,
                    sparsity=0.25,
                    gamma=0.20,
                    family=family,
                    anchors_per_harmed_cluster=64,
                    queries_per_cluster=2,
                )
                anchor_indices = regime.anchor_indices
                train_rows = []
                dev_rows = []
                for cluster_id in regime.harmed_clusters:
                    cluster_anchors = anchor_indices[
                        regime.clean_cluster_ids[anchor_indices] == cluster_id
                    ]
                    train_rows.extend(cluster_anchors[:48])
                    dev_rows.extend(cluster_anchors[48:])
                train = np.asarray(train_rows, dtype=np.int64)
                dev = np.asarray(dev_rows, dtype=np.int64)
                fit_rows = np.concatenate((train, dev))
                eval_mask = np.isin(
                    regime.clean_cluster_ids, regime.harmed_clusters
                )
                eval_mask[anchor_indices] = False
                eval_indices = np.flatnonzero(eval_mask)

                global_audit = select_ridge_hyperparameters(
                    regime.corrupted_docs,
                    regime.clean_docs,
                    fit_indices=train,
                    dev_indices=dev,
                    eval_indices=tuple(int(value) for value in eval_indices),
                    lambdas=(1e-6, 1e-3, 1e-1, 1.0),
                    ranks=(None,),
                )
                global_model = fit_global_ridge(
                    regime.corrupted_docs[fit_rows],
                    regime.clean_docs[fit_rows],
                    global_audit.selected_lambda,
                )
                global_prediction = global_model.transform(
                    regime.corrupted_docs[eval_indices]
                )

                # Clean IDs are intentionally used only here as the named
                # diagnostic upper bound. Production routing is tested above
                # to accept corrupted-space inferred labels instead.
                local_model = fit_local_ridges(
                    regime.corrupted_docs,
                    regime.clean_docs,
                    regime.clean_cluster_ids,
                    lambdas=(1e-6, 1e-3, 1e-1, 1.0),
                    train_indices=train,
                    dev_indices=dev,
                    rank=None,
                    eval_indices=tuple(int(value) for value in eval_indices),
                )
                local_prediction = apply_local_ridges(
                    local_model,
                    regime.corrupted_docs[eval_indices],
                    regime.clean_cluster_ids[eval_indices],
                )
                target = regime.clean_docs[eval_indices]
                floor_mse = float(
                    np.mean((regime.corrupted_docs[eval_indices] - target) ** 2)
                )
                global_mse = float(np.mean((global_prediction - target) ** 2))
                local_mse = float(np.mean((local_prediction - target) ** 2))

                self.assertLess(local_mse, floor_mse)
                self.assertLess(local_mse, global_mse)
                self.assertGreater(local_mse, 1e-12)


if __name__ == "__main__":
    unittest.main()
