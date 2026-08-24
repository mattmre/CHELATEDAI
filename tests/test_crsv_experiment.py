"""Adversarial tests for the CRSV/LIR/SRS METHOD_DEV primitives."""

from __future__ import annotations

import math
import unittest

import numpy as np

from crsv_experiment import (
    CRSVValidationError,
    ReplacementStep,
    SpaceContract,
    additive_operator_diagnostics,
    additivity_gap,
    classify_interaction,
    commutator_distance,
    crsv_interaction_energy,
    deterministic_ranking,
    home_correspondence_permutation_test,
    mobius_interaction,
    onion_path_metrics,
    order_spread,
    ordered_prefix_operator_diagnostics,
    spectral_diagnostics,
    subspace_attunement,
    useful_signal_atrophy,
    validate_declared_replacement_path,
    weighted_kendall_inversion_loss,
)


def _contract(lineage_id: str, compatibility_domain_id: str) -> SpaceContract:
    return SpaceContract(
        role="query",
        metric="cosine",
        dimension=2,
        normalization="l2",
        dtype="float32",
        lineage_id=lineage_id,
        compatibility_domain_id=compatibility_domain_id,
    )


def _step(
    step_id: str,
    layer_id: str,
    input_contract: SpaceContract,
    output_contract: SpaceContract,
    *,
    expires_at: str = "2030-01-01T00:00:00Z",
) -> ReplacementStep:
    return ReplacementStep(
        step_id=step_id,
        layer_id=layer_id,
        input_contract=input_contract,
        output_contract=output_contract,
        evidence_sha256=(step_id[0].lower() * 64),
        evidence_state="VALID",
        evidence_expires_at=expires_at,
    )


def _home_cube(size: int, groups: int = 3) -> np.ndarray:
    base = np.eye(size, dtype=np.float64)
    return np.stack([base + group * 0.01 for group in range(groups)], axis=0)


class ReplacementPathTests(unittest.TestCase):
    def test_valid_path_returns_every_exact_prefix(self) -> None:
        initial = _contract("old", "compat-old")
        middle = _contract("middle", "compat-middle")
        final = _contract("new", "compat-new")
        steps = (
            _step("a-step", "encoder", initial, middle),
            _step("b-step", "projection", middle, final),
        )

        prefixes = validate_declared_replacement_path(
            initial,
            steps,
            evaluated_at="2029-01-01T00:00:00+00:00",
        )

        self.assertEqual(prefixes, (initial, middle, final))

    def test_path_rejects_contract_mismatch_expiry_and_duplicate_layer(self) -> None:
        initial = _contract("old", "compat-old")
        middle = _contract("middle", "compat-middle")
        wrong = _contract("wrong", "compat-wrong")
        final = _contract("new", "compat-new")

        with self.assertRaisesRegex(CRSVValidationError, "does not match"):
            validate_declared_replacement_path(
                initial,
                (_step("a-step", "encoder", wrong, final),),
                evaluated_at="2029-01-01T00:00:00Z",
            )
        with self.assertRaisesRegex(CRSVValidationError, "expired"):
            validate_declared_replacement_path(
                initial,
                (
                    _step(
                        "a-step",
                        "encoder",
                        initial,
                        final,
                        expires_at="2028-01-01T00:00:00Z",
                    ),
                ),
                evaluated_at="2029-01-01T00:00:00Z",
            )
        with self.assertRaisesRegex(CRSVValidationError, "layer IDs"):
            validate_declared_replacement_path(
                initial,
                (
                    _step("a-step", "encoder", initial, middle),
                    _step("b-step", "encoder", middle, final),
                ),
                evaluated_at="2029-01-01T00:00:00Z",
            )

    def test_evidence_contract_fails_closed(self) -> None:
        initial = _contract("old", "compat-old")
        final = _contract("new", "compat-new")
        with self.assertRaisesRegex(CRSVValidationError, "lowercase SHA-256"):
            ReplacementStep(
                step_id="bad-sha",
                layer_id="encoder",
                input_contract=initial,
                output_contract=final,
                evidence_sha256="not-a-hash",
                evidence_state="VALID",
                evidence_expires_at="2030-01-01T00:00:00Z",
            )
        with self.assertRaisesRegex(CRSVValidationError, "must be VALID"):
            ReplacementStep(
                step_id="stale",
                layer_id="encoder",
                input_contract=initial,
                output_contract=final,
                evidence_sha256="a" * 64,
                evidence_state="STALE",
                evidence_expires_at="2030-01-01T00:00:00Z",
            )

    def test_step_rejects_untyped_terminal_contract(self) -> None:
        initial = _contract("old", "compat-old")
        with self.assertRaisesRegex(CRSVValidationError, "output_contract"):
            ReplacementStep(
                step_id="bad-terminal",
                layer_id="encoder",
                input_contract=initial,
                output_contract={"lineage_id": "forged"},
                evidence_sha256="a" * 64,
                evidence_state="VALID",
                evidence_expires_at="2030-01-01T00:00:00Z",
            )


class InversionAndOnionTests(unittest.TestCase):
    def test_ranking_ties_are_identifier_ascending(self) -> None:
        self.assertEqual(
            deterministic_ranking({"b": 1.0, "c": 2.0, "a": 1.0}),
            ("c", "a", "b"),
        )

    def test_weighted_kendall_exact_values(self) -> None:
        reference = ("a", "b", "c")
        self.assertEqual(
            weighted_kendall_inversion_loss(reference, ("c", "b", "a")),
            1.0,
        )
        self.assertAlmostEqual(
            weighted_kendall_inversion_loss(
                reference,
                ("b", "a", "c"),
                {
                    ("a", "b"): 3.0,
                    ("a", "c"): 1.0,
                    ("b", "c"): 1.0,
                },
            ),
            0.6,
        )

    def test_weighted_kendall_rejects_incomparable_universes(self) -> None:
        with self.assertRaisesRegex(CRSVValidationError, "identical IDs"):
            weighted_kendall_inversion_loss(("a", "b"), ("a", "c"))
        with self.assertRaisesRegex(CRSVValidationError, "positive mass"):
            weighted_kendall_inversion_loss(
                ("a", "b"),
                ("b", "a"),
                {("a", "b"): 0.0},
            )

    def test_weighted_kendall_scales_extreme_finite_weights(self) -> None:
        result = weighted_kendall_inversion_loss(
            ("a", "b", "c"),
            ("c", "b", "a"),
            {
                ("a", "b"): 1e308,
                ("a", "c"): 1e308,
                ("b", "c"): 1e308,
            },
        )
        self.assertEqual(result, 1.0)

    def test_onion_differentials_preserve_reversals(self) -> None:
        result = onion_path_metrics(
            (0.5, 0.3, 0.4, 0.2),
            shell_weights=(1.0, 2.0, 1.0),
        )

        np.testing.assert_allclose(
            result["marginal_differentials"],
            [0.2, -0.1, 0.2],
            atol=1e-12,
        )
        self.assertAlmostEqual(result["endpoint_benefit"], 0.3)
        self.assertAlmostEqual(result["reversal_mass"], 0.1)
        self.assertAlmostEqual(result["reversal_fraction"], 0.2)
        expected_burden = 2.0 * math.log((0.4 + 1e-12) / (0.3 + 1e-12))
        self.assertAlmostEqual(result["reversal_burden"], expected_burden)
        self.assertLess(result["telescoping_error"], 1e-12)
        self.assertAlmostEqual(
            result["endpoint_log_contraction"],
            math.log((0.5 + 1e-12) / (0.2 + 1e-12)),
        )

    def test_onion_rejects_out_of_range_losses(self) -> None:
        with self.assertRaisesRegex(CRSVValidationError, r"\[0, 1\]"):
            onion_path_metrics((0.2, 1.1))

    def test_additivity_and_mobius_keep_the_sign(self) -> None:
        subadditive = {
            frozenset(): 0.0,
            frozenset(("a",)): 0.2,
            frozenset(("b",)): 0.3,
            frozenset(("a", "b")): 0.4,
        }
        self.assertAlmostEqual(additivity_gap(subadditive, ("a", "b")), -0.1)
        self.assertAlmostEqual(mobius_interaction(subadditive, ("a", "b")), -0.1)
        self.assertEqual(classify_interaction(-0.1), "SUBADDITIVE")

        superadditive = dict(subadditive)
        superadditive[frozenset(("a", "b"))] = 0.6
        self.assertAlmostEqual(additivity_gap(superadditive, ("a", "b")), 0.1)
        self.assertEqual(classify_interaction(0.1), "SUPERADDITIVE")
        self.assertEqual(classify_interaction(1e-13), "ADDITIVE")

    def test_three_way_mobius_is_not_relabelled_pair_synergy(self) -> None:
        benefits = {
            frozenset(): 0.0,
            frozenset(("a",)): 0.1,
            frozenset(("b",)): 0.1,
            frozenset(("c",)): 0.1,
            frozenset(("a", "b")): 0.2,
            frozenset(("a", "c")): 0.2,
            frozenset(("b", "c")): 0.2,
            frozenset(("a", "b", "c")): 0.35,
        }
        self.assertAlmostEqual(mobius_interaction(benefits, ("a", "b", "c")), 0.05)

    def test_set_interactions_reject_nonfinite_true_results(self) -> None:
        with self.assertRaisesRegex(CRSVValidationError, "numerically finite"):
            mobius_interaction(
                {
                    frozenset(): 1e308,
                    frozenset(("a",)): -1e308,
                },
                ("a",),
            )
        with self.assertRaisesRegex(CRSVValidationError, "numerically finite"):
            additivity_gap(
                {
                    frozenset(): -1e308,
                    frozenset(("a",)): 1e308,
                    frozenset(("b",)): 1e308,
                    frozenset(("a", "b")): 1e308,
                },
                ("a", "b"),
            )

    def test_set_interactions_cancel_extreme_terms_stably(self) -> None:
        benefits = {
            frozenset(): 1e308,
            frozenset(("a",)): 1e308,
            frozenset(("b",)): 1e308,
            frozenset(("a", "b")): 1e308,
        }
        self.assertEqual(mobius_interaction(benefits, ("a", "b")), 0.0)
        self.assertEqual(additivity_gap(benefits, ("a", "b")), 0.0)

    def test_order_spread_and_commutator_detect_sequence_dependence(self) -> None:
        spread = order_spread({("a", "b"): 0.2, ("b", "a"): 0.35})
        self.assertEqual(spread["best_order"], ["a", "b"])
        self.assertEqual(spread["worst_order"], ["b", "a"])
        self.assertAlmostEqual(spread["spread"], 0.15)

        diagonal_a = [[1.0, 0.0], [0.0, 2.0]]
        diagonal_b = [[3.0, 0.0], [0.0, 4.0]]
        self.assertAlmostEqual(commutator_distance(diagonal_a, diagonal_b), 0.0)
        left = [[0.0, 1.0], [0.0, 0.0]]
        right = [[0.0, 0.0], [1.0, 0.0]]
        self.assertAlmostEqual(commutator_distance(left, right), 1.0)
        self.assertAlmostEqual(commutator_distance(left, right, [1.0, 0.0]), 1.0)


class SubspaceAndReverberationTests(unittest.TestCase):
    def test_principal_angle_attunement_separates_aligned_and_orthogonal(self) -> None:
        aligned = subspace_attunement([[1.0], [0.0]], [[2.0], [0.0]])
        orthogonal = subspace_attunement([[1.0], [0.0]], [[0.0], [1.0]])

        self.assertAlmostEqual(aligned["maximum_coherence"], 1.0)
        self.assertAlmostEqual(aligned["minimum_principal_angle_radians"], 0.0)
        self.assertAlmostEqual(orthogonal["maximum_coherence"], 0.0)
        self.assertAlmostEqual(
            orthogonal["minimum_principal_angle_radians"],
            math.pi / 2.0,
        )

    def test_spectral_radius_does_not_hide_nonnormal_transients(self) -> None:
        normal = spectral_diagnostics([[0.8, 0.0], [0.0, 0.5]], horizon=5)
        nonnormal = spectral_diagnostics([[0.8, 4.0], [0.0, 0.8]], horizon=5)
        boundary = spectral_diagnostics([[1.0 - 5e-9]], horizon=2)

        self.assertAlmostEqual(normal["spectral_radius"], 0.8)
        self.assertTrue(normal["asymptotically_contractive"])
        self.assertAlmostEqual(normal["peak_transient_gain"], 1.0)
        self.assertEqual(normal["peak_horizon"], 0)
        self.assertFalse(normal["strict_transient_amplification"])
        self.assertAlmostEqual(normal["departure_from_normality"], 0.0)
        self.assertAlmostEqual(nonnormal["spectral_radius"], 0.8)
        self.assertTrue(nonnormal["asymptotically_contractive"])
        self.assertGreater(nonnormal["peak_transient_gain"], 1.0)
        self.assertTrue(nonnormal["strict_transient_amplification"])
        self.assertGreater(nonnormal["departure_from_normality"], 0.0)
        self.assertTrue(boundary["near_unit_boundary"])
        self.assertFalse(boundary["asymptotically_contractive"])

    def test_additive_operator_diagnostics_resolve_reinforcement(self) -> None:
        result = additive_operator_diagnostics(
            [[0.0]],
            {"left": [[1.0]], "right": [[1.0]]},
            coordinate_space_id="shared-space-v1",
            horizon=3,
        )

        self.assertAlmostEqual(result["triangle_alignment_ratio"], 1.0)
        self.assertAlmostEqual(result["triangle_slack"], 0.0)
        self.assertAlmostEqual(result["net_destructive_interference_scalar"], 0.0)
        self.assertAlmostEqual(result["signed_interference_energy"], 2.0)
        self.assertAlmostEqual(result["normalized_signed_interference"], 1.0)
        self.assertAlmostEqual(
            result["pairwise"][0]["signed_frobenius_alignment"],
            1.0,
        )
        self.assertAlmostEqual(
            result["pairwise"][0]["dominant_subspace_coherence"],
            1.0,
        )
        self.assertAlmostEqual(result["pairwise"][0]["commutator_distance"], 0.0)
        self.assertAlmostEqual(
            result["aggregate_transition_spectral"]["spectral_radius"],
            2.0,
        )

    def test_additive_operator_diagnostics_resolve_cancellation(self) -> None:
        result = additive_operator_diagnostics(
            [[0.0]],
            {"left": [[1.0]], "right": [[-1.0]]},
            coordinate_space_id="shared-space-v1",
            horizon=3,
        )

        self.assertAlmostEqual(result["triangle_alignment_ratio"], 0.0)
        self.assertAlmostEqual(result["triangle_slack"], 1.0)
        self.assertAlmostEqual(result["net_destructive_interference_scalar"], 1.0)
        self.assertAlmostEqual(result["signed_interference_energy"], -2.0)
        self.assertAlmostEqual(result["normalized_signed_interference"], -1.0)
        self.assertAlmostEqual(
            result["pairwise"][0]["signed_frobenius_alignment"],
            -1.0,
        )
        self.assertAlmostEqual(
            result["aggregate_transition_spectral"]["peak_transient_gain"],
            1.0,
        )

    def test_triangle_slack_does_not_mislabel_orthogonality_as_cancellation(self) -> None:
        result = additive_operator_diagnostics(
            np.zeros((2, 2)),
            {
                "left": [[1.0, 0.0], [0.0, 0.0]],
                "right": [[0.0, 0.0], [0.0, 1.0]],
            },
            coordinate_space_id="shared-space-v1",
            horizon=2,
        )

        self.assertGreater(result["triangle_slack"], 0.0)
        self.assertAlmostEqual(result["signed_interference_energy"], 0.0)
        self.assertAlmostEqual(result["net_destructive_interference_scalar"], 0.0)

    def test_additive_operator_diagnostics_fail_on_undefined_subspace(self) -> None:
        with self.assertRaisesRegex(CRSVValidationError, "numerical rank"):
            additive_operator_diagnostics(
                np.zeros((2, 2)),
                {"left": [[1.0, 0.0], [0.0, 0.0]], "right": np.eye(2)},
                coordinate_space_id="shared-space-v1",
                horizon=2,
                subspace_rank=2,
            )
        with self.assertRaisesRegex(CRSVValidationError, "every operator"):
            additive_operator_diagnostics(
                [[0.0]],
                {"left": [[1.0]], "right": [[1.0]]},
                coordinate_space_id="shared-space-v1",
                weights={"left": 1.0},
                horizon=2,
            )

    def test_ordered_prefix_diagnostics_use_products_not_addition(self) -> None:
        result = ordered_prefix_operator_diagnostics(
            (("expand", [[2.0]]), ("contract", [[0.25]])),
            coordinate_space_id="shared-space-v1",
        )

        self.assertEqual(result["operator_order"], ["expand", "contract"])
        self.assertEqual(
            [row["cumulative_gain"] for row in result["prefixes"]],
            [1.0, 2.0, 0.5],
        )
        self.assertEqual(result["peak_prefix_length"], 1)

    def test_useful_signal_atrophy_is_basis_scoped(self) -> None:
        native = [[1.0, 2.0], [3.0, 4.0]]
        candidate = [[0.5, 100.0], [1.5, 200.0]]
        result = useful_signal_atrophy(native, candidate, [[1.0], [0.0]])

        self.assertAlmostEqual(result["native_signal_energy"], 10.0)
        self.assertAlmostEqual(result["candidate_signal_energy"], 2.5)
        self.assertAlmostEqual(result["signal_retention"], 0.25)
        self.assertAlmostEqual(result["signed_signal_change"], -7.5)
        self.assertAlmostEqual(result["signal_atrophy"], 0.75)

    def test_useful_signal_atrophy_rejects_unidentified_baseline(self) -> None:
        with self.assertRaisesRegex(CRSVValidationError, "identification floor"):
            useful_signal_atrophy(
                [[1e-8]],
                [[1e-8]],
                [[1.0]],
                minimum_native_signal_energy=1e-12,
            )

    def test_extreme_finite_inputs_fail_closed_or_remain_finite(self) -> None:
        onion = onion_path_metrics((0.0, 1.0), epsilon=5e-324)
        self.assertTrue(math.isfinite(onion["endpoint_log_contraction"]))
        with self.assertRaisesRegex(CRSVValidationError, "numerically finite"):
            spectral_diagnostics([[1e308]], horizon=2)


class ConditionalReplacementTests(unittest.TestCase):
    def test_interaction_energy_recovers_balanced_home_structure(self) -> None:
        result = crsv_interaction_energy(_home_cube(3))

        self.assertAlmostEqual(result["interaction_energy"], 2.0 / 9.0)
        self.assertAlmostEqual(
            result["unbiased_cross_group_interaction_energy"],
            2.0 / 9.0,
        )
        self.assertAlmostEqual(result["group_interaction_dispersion"], 0.0)

    def test_interaction_energy_is_zero_for_additive_main_effects(self) -> None:
        additive = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float64)
        cube = np.stack((additive, additive + 0.1, additive + 0.2), axis=0)
        result = crsv_interaction_energy(cube)

        self.assertLess(result["interaction_energy"], 1e-28)

    def test_exact_home_correspondence_permutation(self) -> None:
        labels = ("a", "b", "c", "d")
        result = home_correspondence_permutation_test(
            _home_cube(4),
            labels,
            labels,
            exchangeability_blocks=("designed",) * 4,
        )

        self.assertTrue(result["exact"])
        self.assertEqual(result["sampling_mode"], "EXACT")
        self.assertEqual(result["permutation_count"], 24)
        self.assertEqual(result["total_permutation_universe"], 24)
        self.assertIsNone(result["seed"])
        self.assertIsNone(result["monte_carlo_standard_error"])
        self.assertAlmostEqual(result["observed_home_effect"], 1.0)
        self.assertAlmostEqual(result["one_sided_p_value"], 1.0 / 24.0)

    def test_exchangeability_blocks_are_enforced(self) -> None:
        labels = ("a", "b", "c", "d")
        blocked = home_correspondence_permutation_test(
            _home_cube(4),
            labels,
            labels,
            exchangeability_blocks=("left", "left", "right", "right"),
        )
        self.assertEqual(blocked["permutation_count"], 4)
        self.assertEqual(blocked["exchangeability_block_count"], 2)

        with self.assertRaisesRegex(CRSVValidationError, "no non-identity"):
            home_correspondence_permutation_test(
                _home_cube(4),
                labels,
                labels,
                exchangeability_blocks=labels,
            )
        with self.assertRaisesRegex(CRSVValidationError, "seed"):
            home_correspondence_permutation_test(
                _home_cube(4),
                labels,
                labels,
                exchangeability_blocks=("designed",) * 4,
                seed=-1,
            )

    def test_monte_carlo_permutation_is_seeded_and_not_double_counted(self) -> None:
        labels = ("a", "b", "c", "d", "e")
        first = home_correspondence_permutation_test(
            _home_cube(5),
            labels,
            labels,
            exchangeability_blocks=("designed",) * 5,
            max_permutations=10,
            seed=99,
        )
        second = home_correspondence_permutation_test(
            _home_cube(5),
            labels,
            labels,
            exchangeability_blocks=("designed",) * 5,
            max_permutations=10,
            seed=99,
        )

        self.assertFalse(first["exact"])
        self.assertEqual(first["sampling_mode"], "MONTE_CARLO")
        self.assertEqual(first, second)
        self.assertEqual(first["permutation_count"], 10)
        self.assertEqual(first["total_permutation_universe"], 120)
        self.assertEqual(first["seed"], 99)
        self.assertGreater(first["monte_carlo_standard_error"], 0.0)
        self.assertGreaterEqual(first["one_sided_p_value"], 1.0 / 11.0)

    def test_exchangeability_must_be_explicit(self) -> None:
        labels = ("a", "b", "c", "d")
        with self.assertRaisesRegex(CRSVValidationError, "explicitly"):
            home_correspondence_permutation_test(
                _home_cube(4),
                labels,
                labels,
            )


if __name__ == "__main__":
    unittest.main()
