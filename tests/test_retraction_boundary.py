"""Unit tests for retraction_boundary (W22-09). unittest only."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import retraction_boundary as rb  # noqa: E402


class TestAffineState(unittest.TestCase):
    def setUp(self):
        self.As, self.bs, self.meta = rb.generate_regime("nonnormal-TV", 5, 37, 3)
        self.x0 = np.zeros(5)

    def test_regimes_deterministic(self):
        for regime in rb.REGIMES:
            a1, b1, _ = rb.generate_regime(regime, 4, 20, 1)
            a2, b2, _ = rb.generate_regime(regime, 4, 20, 1)
            np.testing.assert_array_equal(a1, a2)
            np.testing.assert_array_equal(b1, b2)

    def test_singular_regime_has_rank_deficient_steps(self):
        As, _bs, meta = rb.generate_regime("singular-TV", 6, 100, 0)
        self.assertEqual(len(meta["singular_steps"]), 5)
        for s in meta["singular_steps"]:
            self.assertEqual(np.linalg.matrix_rank(As[s]), 5)

    def test_segment_tree_root_equals_replay(self):
        tree = rb.AffineSegmentTree(self.As, self.bs)
        np.testing.assert_allclose(tree.apply_root(self.x0), rb.replay(self.As, self.bs, self.x0), rtol=1e-10, atol=1e-10)

    def test_segment_tree_point_update_matches_skip_and_input(self):
        tree = rb.AffineSegmentTree(self.As, self.bs)
        k = 11
        tree.update(k, np.eye(5), np.zeros(5))
        np.testing.assert_allclose(tree.apply_root(self.x0), rb.replay(self.As, self.bs, self.x0, skip=k), rtol=1e-9, atol=1e-9)
        tree.update(k, self.As[k], np.zeros(5))
        np.testing.assert_allclose(tree.apply_root(self.x0), rb.replay(self.As, self.bs, self.x0, zero_input=k), rtol=1e-9, atol=1e-9)
        self.assertEqual(tree.matmuls_last_update, int(np.log2(tree.size)))

    def test_geometric_positions(self):
        self.assertEqual(rb.geometric_checkpoint_positions(16), [0, 8, 12, 14, 15])
        self.assertEqual(rb.geometric_checkpoint_positions(10), [0, 2, 6, 8, 9])

    def test_states_and_checkpoint_replay(self):
        X = rb.states_trajectory(self.As, self.bs, self.x0)
        np.testing.assert_allclose(rb.replay(self.As, self.bs, X[20], start=20), X[-1], rtol=1e-12, atol=1e-12)

    def test_subtractive_tv_is_exact_for_input_deletion(self):
        X = rb.states_trajectory(self.As, self.bs, self.x0)
        Phi, c = rb.suffix_products(self.As, self.bs)
        for k in (0, 7, 36):
            ref = rb.replay(self.As, self.bs, self.x0, zero_input=k)
            self.assertLess(rb.rel_err(X[-1] - Phi[k] @ self.bs[k], ref), 1e-10)
            np.testing.assert_allclose(Phi[k] @ X[k + 1] + c[k], X[-1], rtol=1e-9, atol=1e-9)

    def test_subtractive_lti_binary_powers(self):
        As, bs, _ = rb.generate_regime("nonnormal-LTI", 4, 50, 0)
        X = rb.states_trajectory(As, bs, np.zeros(4))
        powers = rb.binary_powers(As[0], 50)
        k = 13
        y, mv = rb.apply_power(powers, 50 - 1 - k, bs[k])
        self.assertEqual(mv, bin(36).count("1"))
        ref = rb.replay(As, bs, np.zeros(4), zero_input=k)
        self.assertLess(rb.rel_err(X[-1] - y, ref), 1e-10)

    def test_inv_certified_exact_on_well_conditioned(self):
        As, bs, _ = rb.generate_regime("normal-TV", 4, 30, 0)
        X = rb.states_trajectory(As, bs, np.zeros(4))
        Phi, c = rb.suffix_products(As, bs)
        k = 9
        out, info = rb.inv_retract_skip(X[-1], Phi[k], c[k], As[k], bs[k], certified=True)
        self.assertFalse(info["refused"])
        self.assertLess(rb.rel_err(out, rb.replay(As, bs, np.zeros(4), skip=k)), 1e-9)

    def test_collision_pair_same_final_state_different_counterfactual(self):
        As, bs, meta = rb.generate_regime("singular-TV", 6, 80, 2)
        s = meta["singular_steps"][0]
        pair = rb.singular_collision_pair(As, bs, s)
        x0 = np.zeros(6)
        xT, xT2 = rb.replay(As, bs, x0), rb.replay(As, pair["bs_prime"], x0)
        self.assertLess(rb.rel_err(xT2, xT), 1e-12)
        t1 = rb.replay(As, bs, x0, skip=s)
        t2 = rb.replay(As, pair["bs_prime"], x0, skip=s)
        self.assertGreater(rb.rel_err(t2, t1), 1e-3)
        Phi, c = rb.suffix_products(As, bs)
        out, info = rb.inv_retract_skip(xT, Phi[s], c[s], As[s], bs[s], certified=True)
        self.assertIsNone(out)
        self.assertTrue(info["refused"])


class TestCrouzeix(unittest.TestCase):
    def test_numerical_range_of_normal_matrix(self):
        A = np.diag([1.0, -0.5, 0.25])
        pts = rb.numerical_range_boundary(A, 64)
        self.assertAlmostEqual(float(np.max(np.abs(pts))), 1.0, places=10)
        self.assertTrue(np.all(np.abs(pts.imag) < 1e-12))

    def test_jordan_block_numerical_radius(self):
        # W([[0,1],[0,0]]) is the disk of radius 1/2
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        pts = rb.numerical_range_boundary(A, 360)
        np.testing.assert_allclose(np.abs(pts), 0.5, atol=1e-12)

    def test_crouzeix_ratio_for_jordan_block_is_two(self):
        # ||A|| / max_W |z| = 1 / 0.5 = 2: the extremal case of the conjecture
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        pts = rb.numerical_range_boundary(A, 360)
        r = rb.crouzeix_ratio_power(A, 1, pts)
        self.assertAlmostEqual(r["ratio_W"], 2.0, places=9)

    def test_poly_matrix_horner(self):
        A = np.array([[1.0, 2.0], [0.0, 3.0]])
        P = rb.poly_matrix([1.0, 0.0, 2.0], A)
        np.testing.assert_allclose(P, np.eye(2) + 2 * A @ A)

    def test_check_holds_on_random_matrix(self):
        A = np.random.default_rng(0).standard_normal((5, 5))
        res = rb.crouzeix_check(A, max_power=8, n_random_polys=4)
        self.assertTrue(res["bound_1_plus_sqrt2_holds"])
        self.assertLessEqual(res["max_ratio_W"], rb.CROUZEIX_PALENCIA_CONSTANT)


class TestEK5Helpers(unittest.TestCase):
    def test_geo_ladder(self):
        self.assertEqual(rb.geo_ladder(16), [8, 12, 14, 15])

    def test_chain_withdrawal_matches_rebuild(self):
        from chelated_pv7_spine.ek5_reason_canonicalizer import reference_fresh_rebuild

        log = rb.chain_log(12)
        st = rb.build_incremental(log)
        st.apply_mutation(withdraw_attestation_ids=["att:e5:e4"])
        ref = reference_fresh_rebuild(st)
        self.assertEqual(rb.semantic_diffs(st, ref), 0)
        self.assertEqual(st.eligibility["e11"], "ADMIT")
        rebuilt = rb.build_batch(log, None, 0, "att:e5:e4")
        self.assertEqual(rb.semantic_diffs(rebuilt, ref), 0)
        ckpts, _total = rb.build_geo_checkpoints(log, 10 ** 9)
        geo, p = rb.geo_ckpt_delete(log, ckpts, log.attestation_turn()["att:e5:e4"], "att:e5:e4")
        self.assertLessEqual(p, 5)
        self.assertEqual(rb.semantic_diffs(geo, ref), 0)

    def test_asof_probe_shape(self):
        r = rb.asof_collision_probe()
        self.assertIn("collision", r)
        self.assertFalse(r["withdrawn_record_has_withdrawal_time"])


if __name__ == "__main__":
    unittest.main()
