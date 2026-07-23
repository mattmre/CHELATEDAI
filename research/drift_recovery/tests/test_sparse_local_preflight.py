"""Offline CPU unit tests for Obj B sparse-local preflight (no pytest, no HF)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from research.drift_recovery.d2b_preflight.preflight import (
    apply_local_ridges,
    fit_global_ridge,
    fit_local_ridges,
    run_preflight_cell,
    score_ndcg,
    select_ridge_hyperparameters,
)
from research.drift_recovery.regimes.sparse_local_nonaffine import (
    generate_sparse_local_regime,
)


# Keep cells tiny so the module finishes well under ~60s on CPU.
_SMALL = dict(
    seed=7,
    docs_per_cluster=12,
    dimension=8,
    n_clusters=8,
    sparsity=0.25,
    gamma=0.60,
    queries_per_cluster=2,
    anchors_per_harmed_cluster=4,
)

# True affine warps leave residuals at ~machine noise (~1e-28 MSE).  A small but
# strictly positive floor catches non-affinity without depending on gamma scale.
_AFFINE_RESIDUAL_TOL = 1e-8
_RADIAL_COSINE_MAX = 0.98  # |cos| below this ⇒ not purely radial


def _best_affine_residual(source: np.ndarray, target: np.ndarray) -> float:
    """Least-squares residual of the best affine map source -> target (MSE)."""

    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    n, d = x.shape
    aug = np.column_stack((x, np.ones(n)))
    # Solve for each target column: aug @ W ≈ y
    weights, _, _, _ = np.linalg.lstsq(aug, y, rcond=None)
    pred = aug @ weights
    return float(np.mean((pred - y) ** 2))


def _max_abs_radial_cosine(clean: np.ndarray, corrupted: np.ndarray, center: np.ndarray) -> float:
    """Max |cos| between displacement and (x - center) over non-zero rows."""

    disp = corrupted - clean
    radial = clean - center[None, :]
    norms_d = np.linalg.norm(disp, axis=1)
    norms_r = np.linalg.norm(radial, axis=1)
    active = (norms_d > 1e-12) & (norms_r > 1e-12)
    if not np.any(active):
        return 0.0
    cos = np.sum(disp[active] * radial[active], axis=1) / (norms_d[active] * norms_r[active])
    return float(np.max(np.abs(cos)))


class TestWarpProperties(unittest.TestCase):
    def _assert_family_warp(self, family: str) -> None:
        data = generate_sparse_local_regime(family=family, **_SMALL)
        s_req = data.requested_sparsity
        self.assertAlmostEqual(data.actual_sparsity, s_req, places=6)

        clean = data.clean_docs
        corr = data.corrupted_docs
        labels = data.clean_cluster_ids
        harmed = set(int(c) for c in data.harmed_clusters)
        centers = data.clean_cluster_centers

        # Harmed-cluster fraction matches s; clean clusters are identity.
        for cluster_id in range(len(centers)):
            rows = labels == cluster_id
            disp = np.linalg.norm(corr[rows] - clean[rows], axis=1)
            if cluster_id in harmed:
                self.assertTrue(
                    np.any(disp > 1e-8),
                    msg=f"harmed cluster {cluster_id} should move under {family}",
                )
            else:
                np.testing.assert_allclose(
                    corr[rows],
                    clean[rows],
                    rtol=0.0,
                    atol=1e-12,
                    err_msg=f"clean cluster {cluster_id} must be unmoved",
                )

        # Within-cluster map is non-affine and non-radial on each harmed cluster.
        for cluster_id in harmed:
            rows = labels == cluster_id
            residual = _best_affine_residual(clean[rows], corr[rows])
            self.assertGreater(
                residual,
                _AFFINE_RESIDUAL_TOL,
                msg=(
                    f"{family} cluster {cluster_id}: affine residual {residual} "
                    f"not > {_AFFINE_RESIDUAL_TOL}"
                ),
            )
            rad_cos = _max_abs_radial_cosine(clean[rows], corr[rows], centers[cluster_id])
            self.assertLess(
                rad_cos,
                _RADIAL_COSINE_MAX,
                msg=(
                    f"{family} cluster {cluster_id}: max |radial cos| {rad_cos} "
                    f"not < {_RADIAL_COSINE_MAX} (displacement too radial)"
                ),
            )

    def test_quadratic_warp_properties(self) -> None:
        self._assert_family_warp("quadratic")

    def test_soft_fold_warp_properties(self) -> None:
        self._assert_family_warp("soft_fold")


# Distinctive routing labels outside the clean-ID space {0..n_clusters-1}.
# Routing with clean IDs would produce maps keyed in 0..k-1 and fail the sentinel test.
_SENTINEL_LABEL_BASE = 9000


class TestNoCleanIdLeakage(unittest.TestCase):
    def test_fit_local_ridges_audit_records_clean_ids_not_accepted(self) -> None:
        """Audit contract only: training_audit must record clean_cluster_ids_accepted=False.

        This does not prove routing source; see
        test_run_preflight_local_ridge_keys_follow_inferred_routing for the
        behavioral leak-catcher.
        """

        data = generate_sparse_local_regime(family="quadratic", **_SMALL)
        anchors = data.anchor_indices
        source = data.corrupted_docs[anchors]
        target = data.clean_docs[anchors]
        # Fake routing labels (corrupted-space style), not clean cluster IDs.
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 4, size=len(anchors))
        n = len(anchors)
        train = np.arange(0, max(2, n * 3 // 4))
        dev = np.arange(max(2, n * 3 // 4), n)
        if len(dev) == 0:
            dev = np.array([n - 1], dtype=np.int64)
            train = np.arange(0, n - 1)
        model = fit_local_ridges(
            source,
            target,
            labels,
            lambdas=(1e-2, 1.0),
            train_indices=train,
            dev_indices=dev,
            rank=None,
            eval_indices=(),
        )
        self.assertIn("clean_cluster_ids_accepted", model.training_audit)
        self.assertIs(model.training_audit["clean_cluster_ids_accepted"], False)

    def test_run_preflight_local_ridge_keys_follow_inferred_routing(self) -> None:
        """Behavioral leak-catcher: local ridge map keys must come from
        infer_corrupted_clusters, not regime.clean_cluster_ids.

        A load-bearing leak that routes with clean IDs while leaving the audit
        flag False would key maps in clean ID space and fail this test.
        """

        n_clusters = 8
        n_docs = 96

        def _sentinel_infer(corrupted_docs, k, seed=0):
            del seed
            n = len(corrupted_docs)
            # Fixed permutation into k labels, offset so keys cannot collide
            # with clean IDs {0..k-1}.
            return (
                _SENTINEL_LABEL_BASE + (np.arange(n, dtype=np.int64) % int(k))
            ).astype(np.int64)

        with patch(
            "research.drift_recovery.d2b_preflight.preflight.infer_corrupted_clusters",
            side_effect=_sentinel_infer,
        ):
            cell = run_preflight_cell(
                family="quadratic",
                gamma=0.60,
                anchors_per_harmed_cluster=4,
                sparse_fraction=0.25,
                seed=7,
                n_docs=n_docs,
                dim=8,
                n_clusters=n_clusters,
                n_queries_per_cluster=2,
            )

        sentinel_label_space = set(
            range(_SENTINEL_LABEL_BASE, _SENTINEL_LABEL_BASE + n_clusters)
        )
        clean_id_space = set(range(n_clusters))

        for path_name in ("local_dense", "local_low_rank"):
            per_cluster = cell["selection"][path_name]["selected_lambdas_per_cluster"]
            fitted_keys = {int(key) for key in per_cluster}
            self.assertTrue(
                fitted_keys,
                msg=f"{path_name}: expected at least one fitted cluster key",
            )
            self.assertTrue(
                fitted_keys.issubset(sentinel_label_space),
                msg=(
                    f"{path_name}: fitted keys {sorted(fitted_keys)} must be "
                    f"sentinel labels in {sorted(sentinel_label_space)}; "
                    f"clean-ID routing would produce keys in {sorted(clean_id_space)}"
                ),
            )
            self.assertFalse(
                fitted_keys & clean_id_space,
                msg=(
                    f"{path_name}: fitted keys intersect clean ID space "
                    f"{sorted(fitted_keys & clean_id_space)} — clean-ID leak"
                ),
            )

    def test_run_preflight_detector_clean_ids_from_audits(self) -> None:
        cell = run_preflight_cell(
            family="quadratic",
            gamma=0.60,
            anchors_per_harmed_cluster=4,
            sparse_fraction=0.25,
            seed=7,
            n_docs=96,  # 8 clusters * 12
            dim=8,
            n_clusters=8,
            n_queries_per_cluster=2,
        )
        self.assertIn("detector", cell)
        self.assertIs(cell["detector"]["clean_ids_used_by_fit"], False)
        self.assertIs(
            cell["selection"]["local_dense"]["clean_cluster_ids_accepted"], False
        )
        self.assertIs(
            cell["selection"]["local_low_rank"]["clean_cluster_ids_accepted"], False
        )

        # If either fit audit claims clean IDs, detector must flip True.
        original_fit = fit_local_ridges

        def _leaky_fit(*args, **kwargs):
            model = original_fit(*args, **kwargs)
            model.training_audit["clean_cluster_ids_accepted"] = True
            return model

        with patch(
            "research.drift_recovery.d2b_preflight.preflight.fit_local_ridges",
            side_effect=_leaky_fit,
        ):
            leaked = run_preflight_cell(
                family="quadratic",
                gamma=0.60,
                anchors_per_harmed_cluster=4,
                sparse_fraction=0.25,
                seed=7,
                n_docs=96,
                dim=8,
                n_clusters=8,
                n_queries_per_cluster=2,
            )
        self.assertIs(
            leaked["detector"]["clean_ids_used_by_fit"],
            True,
            msg=(
                "clean_ids_used_by_fit must derive from training_audit "
                "(expected True after monkeypatch)"
            ),
        )


class TestSanityLadder(unittest.TestCase):
    def test_oracle_local_beats_global_on_residual(self) -> None:
        """Diagnostic upper bound: dense local with true clean cluster IDs
        recovers strictly more NDCG than the best global ridge, with margin.

        Margin notes (brutally honest):
        - global residual >= 0.01 and local_ndcg - global_ndcg >= 0.01 are
          stable on this small quadratic cell at gamma=2.0 (typically >> 0.01).
        - Full ideal order floor <= global < local <= oracle is NOT asserted for
          floor <= global: on sparse-local non-affine cells a single global ridge
          applied to all docs routinely *hurts* NDCG below the corrupted floor
          (global residual grows while floor stays high). That inequality is not
          deterministically achievable on this tiny construction; we assert the
          largest stable order we can guarantee: global < local <= oracle, with
          floor recorded for diagnostics only.
        """

        # Stronger warp than the default _SMALL gamma so residual margins hold.
        cell_kw = {**_SMALL, "gamma": 2.0}
        data = generate_sparse_local_regime(family="quadratic", **cell_kw)
        clean = data.clean_docs
        corrupted = data.corrupted_docs
        queries = data.queries
        relevance = data.relevance
        anchors = data.anchor_indices
        # Oracle clean cluster IDs on anchor rows (diagnostic only).
        oracle_labels = data.clean_cluster_ids[anchors]
        source = corrupted[anchors]
        target = clean[anchors]

        n = len(anchors)
        # Simple 75/25 split on anchor rows; ensure non-empty sides.
        order = np.arange(n)
        rng = np.random.default_rng(11)
        rng.shuffle(order)
        split = max(2, int(0.75 * n))
        if split >= n:
            split = n - 1
        train, dev = order[:split], order[split:]

        selection = select_ridge_hyperparameters(
            source,
            target,
            train,
            dev,
            eval_indices=(),
            lambdas=(1e-4, 1e-2, 1.0),
            ranks=(None,),
        )
        global_map = fit_global_ridge(
            source[train],
            target[train],
            selection.selected_lambda,
            rank=None,
        )
        local = fit_local_ridges(
            source,
            target,
            oracle_labels,
            lambdas=(1e-4, 1e-2, 1.0),
            train_indices=train,
            dev_indices=dev,
            rank=None,
            eval_indices=(),
        )

        # Apply global ridge to all docs; local ridge routed by oracle clean IDs.
        global_docs = global_map.transform(corrupted)
        local_docs = apply_local_ridges(
            local, corrupted, data.clean_cluster_ids, gated_clusters=None
        )

        oracle_ndcg = score_ndcg(queries, clean, relevance, k=10)
        floor_ndcg = score_ndcg(queries, corrupted, relevance, k=10)
        global_ndcg = score_ndcg(queries, global_docs, relevance, k=10)
        local_ndcg = score_ndcg(queries, local_docs, relevance, k=10)

        global_residual = oracle_ndcg - global_ndcg
        local_gap = local_ndcg - global_ndcg

        self.assertGreaterEqual(
            global_residual,
            0.01,
            msg=(
                f"global residual too small: oracle={oracle_ndcg:.6f} "
                f"global={global_ndcg:.6f} floor={floor_ndcg:.6f} "
                f"residual={global_residual:.6f}"
            ),
        )
        self.assertGreaterEqual(
            local_gap,
            0.01,
            msg=(
                f"local-global NDCG margin too small: local={local_ndcg:.6f} "
                f"global={global_ndcg:.6f} gap={local_gap:.6f} "
                f"floor={floor_ndcg:.6f} oracle={oracle_ndcg:.6f}"
            ),
        )
        # Guaranteed order on this construction (see docstring for floor caveat).
        self.assertLess(
            global_ndcg,
            local_ndcg,
            msg=(
                f"expected global < local: global={global_ndcg:.6f} "
                f"local={local_ndcg:.6f}"
            ),
        )
        self.assertLessEqual(
            local_ndcg,
            oracle_ndcg + 1e-12,
            msg=(
                f"expected local <= oracle: local={local_ndcg:.6f} "
                f"oracle={oracle_ndcg:.6f}"
            ),
        )
        # Diagnostic only — not part of the hard ladder when global undercuts floor.
        self.assertIsInstance(floor_ndcg, float)


class TestHyperparameterHygiene(unittest.TestCase):
    def test_selection_source_and_eval_disjoint(self) -> None:
        data = generate_sparse_local_regime(family="soft_fold", **_SMALL)
        anchors = data.anchor_indices
        source = data.corrupted_docs[anchors]
        target = data.clean_docs[anchors]
        labels = np.arange(len(anchors)) % 3
        n = len(anchors)
        train = np.arange(0, max(2, n * 2 // 3))
        dev = np.arange(max(2, n * 2 // 3), max(3, n * 5 // 6))
        eval_idx = np.arange(max(3, n * 5 // 6), n)
        if len(dev) == 0:
            dev = np.array([n - 2], dtype=np.int64)
            train = np.arange(0, n - 2)
            eval_idx = np.array([n - 1], dtype=np.int64)
        if len(eval_idx) == 0:
            # Force a held-out eval row by shrinking train if needed.
            eval_idx = np.array([int(train[-1])], dtype=np.int64)
            train = train[:-1]

        model = fit_local_ridges(
            source,
            target,
            labels,
            lambdas=(1e-4, 1e-2, 1.0),
            train_indices=train,
            dev_indices=dev,
            rank=None,
            eval_indices=eval_idx,
        )
        audit = model.training_audit
        self.assertEqual(audit["selection_source"], "anchor_dev_mse_only")

        fit_set = set(int(i) for i in audit["fit_indices"])
        dev_set = set(int(i) for i in audit["dev_indices"])
        eval_set = set(int(i) for i in audit["eval_indices"])
        self.assertFalse(fit_set & dev_set, msg="fit/dev must be disjoint")
        self.assertFalse(
            eval_set & (fit_set | dev_set),
            msg="eval indices must be disjoint from fit+dev",
        )

        # Global selection path also records eval without using it.
        sel = select_ridge_hyperparameters(
            source,
            target,
            train,
            dev,
            eval_indices=eval_idx,
            lambdas=(1e-4, 1.0),
            ranks=(None,),
        )
        self.assertEqual(set(sel.eval_indices), eval_set)
        self.assertFalse(set(sel.fit_indices) & set(sel.dev_indices))
        self.assertFalse(set(sel.eval_indices) & (set(sel.fit_indices) | set(sel.dev_indices)))


class TestAuditFields(unittest.TestCase):
    def test_training_audit_hyperparameter_fields(self) -> None:
        data = generate_sparse_local_regime(family="quadratic", **_SMALL)
        anchors = data.anchor_indices
        source = data.corrupted_docs[anchors]
        target = data.clean_docs[anchors]
        labels = np.arange(len(anchors)) % 4
        n = len(anchors)
        train = np.arange(0, max(2, n * 3 // 4))
        dev = np.arange(max(2, n * 3 // 4), n)
        if len(dev) == 0:
            dev = np.array([n - 1], dtype=np.int64)
            train = np.arange(0, n - 1)

        dense = fit_local_ridges(
            source,
            target,
            labels,
            lambdas=(1e-4, 1e-2, 1.0),
            train_indices=train,
            dev_indices=dev,
            rank=None,
            eval_indices=(),
        )
        for key in (
            "selected_lambda",
            "selected_rank",
            "selected_lambdas_per_cluster",
        ):
            self.assertIn(key, dense.training_audit, msg=f"missing audit field {key}")
        self.assertIsInstance(dense.training_audit["selected_lambda"], float)
        # Dense path: selected_rank is None.
        self.assertIsNone(dense.training_audit["selected_rank"])
        self.assertIsInstance(dense.training_audit["selected_lambdas_per_cluster"], dict)

        low_rank = fit_local_ridges(
            source,
            target,
            labels,
            lambdas=(1e-4, 1.0),
            train_indices=train,
            dev_indices=dev,
            candidate_ranks=(1, 2),
            eval_indices=(),
        )
        self.assertIn("selected_lambda", low_rank.training_audit)
        self.assertIn("selected_rank", low_rank.training_audit)
        self.assertIn("selected_lambdas_per_cluster", low_rank.training_audit)
        rank = low_rank.training_audit["selected_rank"]
        self.assertTrue(rank is None or isinstance(rank, int))
        self.assertIsInstance(low_rank.training_audit["selected_lambdas_per_cluster"], dict)


if __name__ == "__main__":
    unittest.main()
