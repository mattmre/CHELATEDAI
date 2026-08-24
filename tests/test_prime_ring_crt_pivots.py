from __future__ import annotations

import math
import unittest

from prime_ring_crt_pivots import (
    AffineMap,
    CRTBudget,
    DEFAULT_MAX_ESTIMATED_BYTES,
    DEFAULT_MAX_WORK_UNITS,
    EVIDENCE_STATUS,
    HARD_MAX_AFFINE_WORD_LENGTH,
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_PRIME,
    PRWC1ResourceError,
    PRWC1ValidationError,
    TARGET_CRT_FACTORS,
    TARGET_ORDER,
    TARGET_PRIME,
    TARGET_PRIMITIVE_ROOT,
    crt_coordinates,
    crt_reconstruct,
    estimate_crt_pivot_resources,
    preflight_crt_pivot_screen,
    reduce_affine_word,
    run_prw_c1_redundancy_screen,
)


class _HostileInt(int):
    def __lt__(self, _other):
        return False

    def __le__(self, _other):
        return False

    def __gt__(self, _other):
        return False

    def __ge__(self, _other):
        return False


class _HostileFloat(float):
    def __lt__(self, _other):
        return False

    def __le__(self, _other):
        return False

    def __gt__(self, _other):
        return False

    def __ge__(self, _other):
        return False


class TestCRTPreflight(unittest.TestCase):
    def test_target_preflight_is_small_exact_and_non_benchmarking(self):
        estimate = estimate_crt_pivot_resources()
        self.assertTrue(estimate.allowed)
        self.assertLess(
            estimate.estimated_peak_bytes,
            DEFAULT_MAX_ESTIMATED_BYTES,
        )
        self.assertLess(
            estimate.estimated_work_units,
            DEFAULT_MAX_WORK_UNITS,
        )
        self.assertFalse(estimate.exact_prime_verified)
        self.assertFalse(estimate.primitive_root_verified)
        self.assertFalse(estimate.measured_process_peak)

        checked = preflight_crt_pivot_screen()
        self.assertEqual(checked.prime, TARGET_PRIME)
        self.assertEqual(checked.order, TARGET_ORDER)
        self.assertEqual(checked.primitive_root, TARGET_PRIMITIVE_ROOT)
        self.assertEqual(checked.crt_factors, TARGET_CRT_FACTORS)
        self.assertTrue(checked.exact_prime_verified)
        self.assertTrue(checked.primitive_root_verified)

    def test_caller_lowered_memory_and_work_limits_refuse_before_screen(self):
        low_memory = CRTBudget(max_estimated_bytes=1024)
        estimate = estimate_crt_pivot_resources(budget=low_memory)
        self.assertFalse(estimate.allowed)
        with self.assertRaises(PRWC1ResourceError):
            run_prw_c1_redundancy_screen(budget=low_memory)

        low_work = CRTBudget(max_work_units=1000)
        estimate = estimate_crt_pivot_resources(budget=low_work)
        self.assertFalse(estimate.allowed)
        with self.assertRaises(PRWC1ResourceError):
            preflight_crt_pivot_screen(budget=low_work)

    def test_hard_resource_ceilings_cannot_be_raised(self):
        with self.assertRaises(PRWC1ValidationError):
            CRTBudget(
                max_estimated_bytes=HARD_MAX_ESTIMATED_BYTES + 1
            )
        with self.assertRaises(PRWC1ResourceError):
            estimate_crt_pivot_resources(
                prime=HARD_MAX_PRIME + 2,
                crt_factors=(2, 5, 7, 67),
            )

    def test_hostile_numeric_subclasses_are_canonicalized_before_use(self):
        budget = CRTBudget(
            max_estimated_bytes=_HostileInt(1),
            max_work_units=_HostileInt(1),
            max_seconds=_HostileFloat(1.0),
            max_prime=_HostileInt(TARGET_PRIME),
            max_pivots=_HostileInt(8),
            max_layers=_HostileInt(8),
        )
        self.assertIs(type(budget.max_estimated_bytes), int)
        self.assertIs(type(budget.max_work_units), int)
        self.assertIs(type(budget.max_seconds), float)
        estimate = estimate_crt_pivot_resources(budget=budget)
        self.assertFalse(estimate.allowed)
        with self.assertRaises(PRWC1ResourceError):
            preflight_crt_pivot_screen(budget=budget)


class TestCRTCoordinates(unittest.TestCase):
    def test_target_coordinates_and_round_trip(self):
        expected = {
            0: (0, 0, 0, 0),
            1: (1, 1, 1, 1),
            67: (1, 2, 4, 0),
            2345: (1, 0, 0, 0),
            4689: (1, 4, 6, 66),
        }
        for exponent, coordinates in expected.items():
            with self.subTest(exponent=exponent):
                self.assertEqual(
                    crt_coordinates(exponent),
                    coordinates,
                )
                self.assertEqual(
                    crt_reconstruct(coordinates),
                    exponent,
                )

    def test_all_small_crt_tuples_are_one_flat_exponent_each(self):
        seen = set()
        for parity in range(2):
            for mod_three in range(3):
                exponent = crt_reconstruct(
                    (parity, mod_three),
                    order=6,
                    factors=(2, 3),
                )
                seen.add(exponent)
                self.assertEqual(
                    crt_coordinates(
                        exponent,
                        order=6,
                        factors=(2, 3),
                    ),
                    (parity, mod_three),
                )
        self.assertEqual(seen, set(range(6)))

    def test_malformed_factors_and_residues_fail_closed(self):
        invalid_factor_sets = (
            (2, 5, 7),
            (2, 5, 7, 67, 1),
            (2, 5, 7, 335),
        )
        for factors in invalid_factor_sets:
            with self.subTest(factors=factors):
                with self.assertRaises(PRWC1ValidationError):
                    crt_coordinates(0, factors=factors)
        with self.assertRaises(PRWC1ValidationError):
            crt_coordinates(0, factors=[2, 5, 7, 67])
        with self.assertRaises(PRWC1ValidationError):
            crt_reconstruct((0, 0, 0))
        with self.assertRaises(PRWC1ValidationError):
            crt_reconstruct((2, 0, 0, 0))


class TestTargetRedundancyReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_prw_c1_redundancy_screen()

    def test_exact_group_and_crt_results(self):
        report = self.report
        self.assertEqual(report.evidence_status, EVIDENCE_STATUS)
        self.assertTrue(report.theorem_applies_to_all_pivots)
        self.assertFalse(
            report.executable_witnesses_are_exhaustive_over_all_pivots
        )
        self.assertTrue(report.root_choice_invariance_theorem_applied)
        self.assertFalse(report.alternate_root_controls_exhaustive)
        self.assertTrue(report.power_coordinate_bijection_verified)
        self.assertTrue(report.crt_round_trip_verified)
        self.assertEqual(report.crt_tuple_count, TARGET_ORDER)
        self.assertFalse(
            report.component_factorization_creates_extra_states
        )
        self.assertEqual(report.shared_pivot_state_count, TARGET_ORDER)
        self.assertTrue(report.zero_excluded_from_exponent_coordinates)
        self.assertTrue(report.zero_fixed_point_verified)

    def test_every_witness_is_flat_and_alternate_root_control_passes(self):
        self.assertGreater(len(self.report.pivot_witnesses), 1)
        for witness in self.report.pivot_witnesses:
            with self.subTest(exponent=witness.exponent):
                self.assertEqual(witness.rows_checked, TARGET_ORDER)
                self.assertTrue(
                    witness.multiplication_equals_exponent_shift
                )
                self.assertTrue(
                    witness.crt_component_shift_equivalent
                )
                self.assertTrue(witness.flat_payload_gather_equivalent)
                self.assertTrue(witness.alternate_root_control_verified)
                self.assertTrue(witness.zero_fixed)
                self.assertEqual(
                    witness.nonzero_legendre_polarity_sign,
                    -1 if witness.exponent % 2 else 1,
                )
                self.assertTrue(
                    witness.nonzero_legendre_polarity_equivalent
                )
                self.assertTrue(
                    witness.shifted_legendre_center_transport_verified
                )
        self.assertTrue(
            self.report.shifted_legendre_center_transport_verified
        )

    def test_repo_zero_convention_is_not_mislabeled_global_polarity(self):
        even = next(
            witness
            for witness in self.report.pivot_witnesses
            if witness.exponent == 2
        )
        odd = next(
            witness
            for witness in self.report.pivot_witnesses
            if witness.exponent == 1
        )
        self.assertTrue(even.full_bipolar_global_polarity_equivalent)
        self.assertFalse(even.full_bipolar_fixed_zero_exception)
        self.assertFalse(odd.full_bipolar_global_polarity_equivalent)
        self.assertTrue(odd.full_bipolar_fixed_zero_exception)
        self.assertEqual(self.report.legendre_stabilizer_size, 2345)
        self.assertEqual(self.report.legendre_effective_orbit_size, 2)
        self.assertTrue(
            self.report.factor_two_coordinate_is_nonzero_polarity_bit
        )

    def test_independent_components_and_layers_are_counted_not_enumerated(self):
        report = self.report
        self.assertEqual(
            report.conditional_independent_layer_control_tuple_count,
            TARGET_ORDER**report.layer_count,
        )
        self.assertAlmostEqual(
            report.conditional_independent_layer_control_information_bits,
            report.layer_count * math.log2(TARGET_ORDER),
        )
        self.assertEqual(
            report.conditional_independent_layer_min_fixed_code_bits,
            math.ceil(report.layer_count * math.log2(TARGET_ORDER)),
        )
        self.assertTrue(
            report.independent_layer_addressability_assumed_not_verified
        )
        self.assertFalse(report.distinguishable_cross_layer_states_measured)
        self.assertFalse(report.independent_layers_enumerated)

    def test_random_control_is_a_relabeling_of_one_cyclic_family_only(self):
        control = self.report.random_control
        self.assertEqual(control.coordinate_count, TARGET_ORDER)
        self.assertEqual(control.family_state_count, TARGET_ORDER)
        self.assertTrue(control.relabeling_is_bijection)
        self.assertTrue(control.conjugated_generator_is_bijection)
        self.assertTrue(control.conjugated_generator_has_full_order)
        self.assertTrue(control.cyclic_composition_verified)
        self.assertFalse(control.native_exponent_shift)
        self.assertTrue(control.matched_as_permutation_only)
        self.assertFalse(control.implementation_cost_matched)
        self.assertFalse(control.utility_tested)

    def test_kills_are_scoped_and_unexecuted_claims_stay_false(self):
        criteria = {
            criterion.name: criterion.triggered
            for criterion in self.report.kill_criteria
        }
        self.assertTrue(criteria["flat_permutation_relabeling"])
        self.assertTrue(criteria["crt_components_add_no_states"])
        self.assertTrue(criteria["legendre_polarity_plus_fixed_zero"])
        self.assertTrue(criteria["mixed_rotation_pivot_layering"])
        self.assertFalse(
            criteria["random_or_additive_control_matches_utility"]
        )
        self.assertFalse(
            self.report.arbitrary_learned_payload_polarity_degeneracy_claimed
        )
        self.assertFalse(self.report.learned_payload_utility_tested)
        self.assertFalse(self.report.implementation_cost_tested)
        self.assertFalse(self.report.performance_benchmarked)
        self.assertFalse(self.report.novelty_claimed)
        self.assertTrue(self.report.affine_closure_theorem_applied)
        self.assertFalse(
            self.report
            .mixed_permutation_word_runtime_examples_exhaustive
        )


class TestSmallPrimeIndependentChecks(unittest.TestCase):
    def test_all_pivots_on_seven_are_exhaustively_witnessed(self):
        report = run_prw_c1_redundancy_screen(
            prime=7,
            primitive_root=3,
            crt_factors=(2, 3),
            pivot_exponents=(0, 1, 2, 3, 4, 5),
            layer_count=3,
            random_seed=17,
        )
        self.assertTrue(
            report.executable_witnesses_are_exhaustive_over_all_pivots
        )
        self.assertEqual(report.crt_tuple_count, 6)
        self.assertEqual(
            report.conditional_independent_layer_control_tuple_count,
            6**3,
        )
        self.assertEqual(report.legendre_stabilizer_size, 3)
        self.assertEqual(report.legendre_effective_orbit_size, 2)
        self.assertTrue(
            all(
                witness.flat_payload_gather_equivalent
                for witness in report.pivot_witnesses
            )
        )

    def test_all_pivots_are_exhaustive_on_two_other_small_primes(self):
        cases = (
            (11, 2, (2, 5)),
            (31, 3, (2, 3, 5)),
        )
        for prime, root, factors in cases:
            with self.subTest(prime=prime):
                report = run_prw_c1_redundancy_screen(
                    prime=prime,
                    primitive_root=root,
                    crt_factors=factors,
                    pivot_exponents=tuple(range(prime - 1)),
                    layer_count=2,
                    random_seed=prime,
                )
                self.assertTrue(
                    report.executable_witnesses_are_exhaustive_over_all_pivots
                )
                self.assertTrue(
                    report.shifted_legendre_center_transport_verified
                )
                self.assertTrue(
                    all(
                        criterion.triggered
                        for criterion in report.kill_criteria[:-1]
                    )
                )

    def test_full_bipolar_odd_pivot_is_minus_carrier_plus_zero_exception(self):
        prime = 7
        root = 3
        exponent = 1
        inverse_multiplier = pow(root, 5, prime)

        def carrier(coordinate):
            if coordinate == 0:
                return 1
            return (
                1
                if pow(coordinate, (prime - 1) // 2, prime) == 1
                else -1
            )

        observed = [
            carrier(inverse_multiplier * coordinate % prime)
            for coordinate in range(prime)
        ]
        expected = [
            -carrier(coordinate) + (2 if coordinate == 0 else 0)
            for coordinate in range(prime)
        ]
        literal_global_sign = [
            -carrier(coordinate) for coordinate in range(prime)
        ]
        self.assertEqual(observed, expected)
        self.assertNotEqual(observed, literal_global_sign)
        self.assertEqual(exponent % 2, 1)

    def test_random_conjugacy_is_deterministic_for_fixed_seed(self):
        arguments = dict(
            prime=7,
            primitive_root=3,
            crt_factors=(2, 3),
            pivot_exponents=(0, 1),
            random_seed=991,
        )
        left = run_prw_c1_redundancy_screen(**arguments).random_control
        right = run_prw_c1_redundancy_screen(**arguments).random_control
        self.assertEqual(left.relabeling_digest, right.relabeling_digest)
        self.assertEqual(left.generator_digest, right.generator_digest)


class TestAffineCollapse(unittest.TestCase):
    def test_direct_affine_construction_validates_and_canonicalizes(self):
        mapping = AffineMap(
            prime=_HostileInt(7),
            multiplier=_HostileInt(8),
            offset=_HostileInt(9),
        )
        self.assertIs(type(mapping.prime), int)
        self.assertIs(type(mapping.multiplier), int)
        self.assertIs(type(mapping.offset), int)
        self.assertEqual(mapping.multiplier, 1)
        self.assertEqual(mapping.offset, 2)
        invalid = (
            dict(prime=0, multiplier=1, offset=0),
            dict(prime=9, multiplier=1, offset=0),
            dict(prime=7, multiplier=0, offset=0),
            dict(prime=7, multiplier=7, offset=0),
            dict(prime=7, multiplier=1, offset=-1),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(PRWC1ValidationError):
                    AffineMap(**arguments)

    def test_multiplicative_conjugation_scales_rotation(self):
        prime = 7
        multiplier = 3
        inverse = 5
        offset = 2
        reduced = reduce_affine_word(
            prime,
            (("M", inverse), ("T", offset), ("M", multiplier)),
        )
        self.assertEqual(reduced.multiplier, 1)
        self.assertEqual(reduced.offset, multiplier * offset % prime)
        for coordinate in range(prime):
            explicit = (
                multiplier
                * ((inverse * coordinate % prime) + offset)
            ) % prime
            self.assertEqual(reduced.apply(coordinate), explicit)

    def test_arbitrary_mixed_word_collapses_to_one_affine_map(self):
        prime = 11
        operations = (
            ("T", 4),
            ("M", 2),
            ("T", 9),
            ("M", 7),
            ("T", 3),
        )
        reduced = reduce_affine_word(prime, operations)
        for coordinate in range(prime):
            explicit = coordinate
            for operation, value in operations:
                if operation == "T":
                    explicit = (explicit + value) % prime
                else:
                    explicit = value * explicit % prime
            self.assertEqual(reduced.apply(coordinate), explicit)

    def test_affine_validation_and_word_guard_fail_closed(self):
        invalid_words = (
            [("T", 1)],
            (("X", 1),),
            (("M", 0),),
            (("T", True),),
        )
        for word in invalid_words:
            with self.subTest(word=word):
                with self.assertRaises(PRWC1ValidationError):
                    reduce_affine_word(7, word)
        with self.assertRaises(PRWC1ResourceError):
            reduce_affine_word(
                7,
                tuple(
                    ("T", 1)
                    for _index in range(
                        HARD_MAX_AFFINE_WORD_LENGTH + 1
                    )
                ),
            )


class TestValidation(unittest.TestCase):
    def test_composite_bad_root_duplicate_pivots_and_booleans_fail(self):
        with self.assertRaises(PRWC1ValidationError):
            preflight_crt_pivot_screen(
                prime=9,
                primitive_root=2,
                crt_factors=(8,),
                pivot_exponents=(0,),
            )
        with self.assertRaises(PRWC1ValidationError):
            preflight_crt_pivot_screen(
                prime=7,
                primitive_root=2,
                crt_factors=(2, 3),
                pivot_exponents=(0,),
            )
        with self.assertRaises(PRWC1ValidationError):
            estimate_crt_pivot_resources(
                prime=7,
                primitive_root=3,
                crt_factors=(2, 3),
                pivot_exponents=(1, 7),
            )
        with self.assertRaises(PRWC1ValidationError):
            estimate_crt_pivot_resources(prime=True)
        with self.assertRaises(PRWC1ValidationError):
            run_prw_c1_redundancy_screen(random_seed=True)

    def test_arbitrary_bigints_fail_before_modulo_or_multiplication(self):
        huge = 1 << 65
        valid_map = AffineMap(prime=7, multiplier=1, offset=0)
        operations = (
            lambda: CRTBudget(max_estimated_bytes=huge),
            lambda: CRTBudget(max_seconds=huge),
            lambda: estimate_crt_pivot_resources(prime=huge),
            lambda: estimate_crt_pivot_resources(
                prime=7,
                primitive_root=3,
                crt_factors=(2, 3),
                pivot_exponents=(huge,),
            ),
            lambda: crt_coordinates(huge),
            lambda: crt_reconstruct((huge, 0, 0, 0)),
            lambda: AffineMap(prime=7, multiplier=huge, offset=0),
            lambda: AffineMap(prime=7, multiplier=1, offset=huge),
            lambda: valid_map.apply(huge),
            lambda: reduce_affine_word(7, (("T", huge),)),
            lambda: run_prw_c1_redundancy_screen(random_seed=huge),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(PRWC1ResourceError):
                    operation()


if __name__ == "__main__":
    unittest.main()
