import itertools
import unittest
from collections.abc import Sequence
from unittest.mock import patch

import numpy as np

from prime_ring_intersection import (
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_SECONDS,
    HARD_MAX_WORK_UNITS,
    EnumerationBudget,
    PRWT1ResourceError,
    PRWT1ValidationError,
    _maximum_spanning_tree_weight,
    analyze_legendre_mask_bank,
    binomial_upper_tail,
    brute_force_union_probability,
    build_legendre_mask_bank,
    bsc_pairwise_error,
    pair_event_intersection_probability,
)
from prime_ring_waypoint import policy_masks


class TestFiniteProbabilityKernels(unittest.TestCase):
    def test_binomial_tail_matches_direct_enumeration(self):
        probability = 0.2
        expected = sum(
            probability ** sum(bits)
            * (1.0 - probability) ** (5 - sum(bits))
            for bits in itertools.product((0, 1), repeat=5)
            if sum(bits) >= 3
        )
        self.assertAlmostEqual(
            binomial_upper_tail(5, probability, 3),
            expected,
            places=14,
        )
        self.assertAlmostEqual(
            bsc_pairwise_error(5, probability),
            expected,
            places=14,
        )

    def test_pair_intersection_matches_noise_pattern_enumeration(self):
        left = {0, 1, 2}
        right = {1, 2, 3, 4}
        probability = 0.25
        expected = 0.0
        for bits in itertools.product((0, 1), repeat=5):
            if (
                sum(bits[index] for index in left) >= 2
                and sum(bits[index] for index in right) >= 2
            ):
                expected += (
                    probability ** sum(bits)
                    * (1.0 - probability) ** (5 - sum(bits))
                )
        observed = pair_event_intersection_probability(
            len(left),
            len(right),
            len(left & right),
            probability,
        )
        self.assertAlmostEqual(observed, expected, places=14)

    def test_invalid_probability_and_overlap_fail_closed(self):
        with self.assertRaises(PRWT1ValidationError):
            bsc_pairwise_error(3, 1.1)
        with self.assertRaises(PRWT1ValidationError):
            bsc_pairwise_error(3, 0.5)
        with self.assertRaises(PRWT1ValidationError):
            pair_event_intersection_probability(2, 3, 3, 0.2)

    def test_even_distance_tie_counts_as_error(self):
        probability = 0.2
        self.assertAlmostEqual(
            bsc_pairwise_error(2, probability),
            1.0 - (1.0 - probability) ** 2,
            places=14,
        )

    def test_maximum_spanning_tree_matches_exhaustive_tree_search(self):
        disagreement_sets = (
            {0, 1, 2},
            {0, 1, 3},
            {0, 4, 5},
            {2, 3, 4, 5},
        )
        distances = np.asarray(
            [len(values) for values in disagreement_sets],
            dtype=np.int64,
        )
        intersections = np.asarray(
            [
                [len(left & right) for right in disagreement_sets]
                for left in disagreement_sets
            ],
            dtype=np.int64,
        )
        observed, edge_count = _maximum_spanning_tree_weight(
            distances,
            intersections,
            0.2,
        )
        edges = list(itertools.combinations(range(4), 2))
        maximum = -1.0
        for candidate in itertools.combinations(edges, 3):
            reached = {0}
            changed = True
            while changed:
                changed = False
                for left, right in candidate:
                    if left in reached and right not in reached:
                        reached.add(right)
                        changed = True
                    elif right in reached and left not in reached:
                        reached.add(left)
                        changed = True
            if len(reached) != 4:
                continue
            weight = sum(
                pair_event_intersection_probability(
                    int(distances[left]),
                    int(distances[right]),
                    int(intersections[left, right]),
                    0.2,
                )
                for left, right in candidate
            )
            maximum = max(maximum, weight)
        self.assertEqual(edge_count, 3)
        self.assertAlmostEqual(observed, maximum, places=14)


class TestLegendreIntersectionBank(unittest.TestCase):
    def setUp(self):
        self.budget = EnumerationBudget(
            max_estimated_bytes=16 * 1024 * 1024,
            max_hypotheses=64,
            max_competitor_pairs=128,
            max_coordinates=64,
            max_bruteforce_patterns=1024,
        )
        self.bank = build_legendre_mask_bank(
            3,
            ((0, 0), (0, 1)),
            ((1, 1),),
            budget=self.budget,
        )

    def test_bank_is_canonical_binary_and_has_expected_states(self):
        self.assertEqual(self.bank.templates.shape, (6, 6))
        self.assertEqual(self.bank.states[0].type_id, 0)
        self.assertEqual(self.bank.states[0].shift, 0)
        self.assertEqual(self.bank.states[-1].type_id, 1)
        self.assertEqual(self.bank.states[-1].shift, 2)
        self.assertEqual(
            set(np.unique(self.bank.templates).tolist()),
            {-1, 1},
        )
        self.assertFalse(self.bank.templates.flags.writeable)
        self.assertFalse(self.bank.prw_t1_structure_verified)

    def test_distance_spectrum_and_hunter_bound_cross_check(self):
        result = analyze_legendre_mask_bank(
            self.bank,
            planted_index=0,
            crossover=0.2,
            wrong_type_only=True,
            compute_bruteforce_union=True,
            require_prw_t1_structure=False,
            budget=self.budget,
        )
        self.assertEqual(result["distance_spectrum"], {"2": 2, "4": 1})
        self.assertEqual(result["minimum_distance"], 2)
        self.assertEqual(result["nearest_state_multiplicity"], 2)
        actual = result[
            "bruteforce_competitor_tie_or_better_union_probability"
        ]
        self.assertIsNotNone(actual)
        self.assertLessEqual(actual, result["hunter_upper_bound"] + 1e-14)
        self.assertLessEqual(
            result["hunter_upper_bound"],
            result["ordinary_union_bound"] + 1e-14,
        )
        self.assertEqual(
            result["hunter_spanning_tree_edge_count"],
            result["competitor_count"] - 1,
        )
        self.assertFalse(result["asymptotic_claim_established"])
        self.assertFalse(result["final_decoder_error_probability_computed"])
        self.assertGreater(
            len(result["intersection_signature_spectrum"]),
            0,
        )
        self.assertGreater(
            len(result["nearest_intersection_signature_spectrum"]),
            0,
        )
        self.assertEqual(
            sum(
                row["pair_count"]
                for row in result["intersection_signature_spectrum"]
            ),
            result["competitor_count"]
            * (result["competitor_count"] - 1)
            // 2,
        )
        self.assertEqual(
            sum(
                row["pair_count"]
                for row in result[
                    "nearest_intersection_signature_spectrum"
                ]
            ),
            result["nearest_state_multiplicity"]
            * (result["nearest_state_multiplicity"] - 1)
            // 2,
        )
        planted = self.bank.templates[0]
        wrong_type = np.asarray(
            [state.type_id != 0 for state in self.bank.states],
            dtype=bool,
        )
        disagreements = self.bank.templates[wrong_type] != planted
        distances = np.count_nonzero(disagreements, axis=1)
        nearest = np.flatnonzero(
            distances == np.min(distances)
        )
        direct_nearest_sum = sum(
            pair_event_intersection_probability(
                int(distances[left]),
                int(distances[right]),
                int(
                    np.count_nonzero(
                        disagreements[left] & disagreements[right]
                    )
                ),
                0.2,
            )
            for left, right in itertools.combinations(nearest, 2)
        )
        self.assertAlmostEqual(
            result["nearest_pair_intersection_sum"],
            direct_nearest_sum,
            places=14,
        )
        self.assertEqual(
            result["resource_guard"][
                "binomial_cache_actual_accounted_bytes"
            ],
            result["resource_guard"][
                "binomial_cache_estimated_bytes"
            ],
        )

    def test_bruteforce_kernel_matches_bank_result(self):
        planted = self.bank.templates[0]
        wrong_type = np.asarray(
            [
                state.type_id != 0
                for state in self.bank.states
            ],
            dtype=bool,
        )
        disagreements = self.bank.templates[wrong_type] != planted
        direct = brute_force_union_probability(
            disagreements,
            0.2,
            budget=self.budget,
        )
        analyzed = analyze_legendre_mask_bank(
            self.bank,
            planted_index=0,
            crossover=0.2,
            budget=self.budget,
            compute_bruteforce_union=True,
            require_prw_t1_structure=False,
        )
        self.assertAlmostEqual(
            direct,
            analyzed[
                "bruteforce_competitor_tie_or_better_union_probability"
            ],
            places=14,
        )

    def test_structure_gate_and_verified_small_prw_t1_bank(self):
        with self.assertRaises(PRWT1ValidationError):
            analyze_legendre_mask_bank(
                self.bank,
                planted_index=0,
                crossover=0.2,
                budget=self.budget,
            )
        structural_budget = EnumerationBudget(
            max_estimated_bytes=32 * 1024 * 1024,
            max_hypotheses=512,
            max_competitor_pairs=20_000,
            max_coordinates=128,
            max_bruteforce_patterns=1024,
        )
        signatures = np.mod(
            np.arange(2, dtype=np.int64)[:, np.newaxis]
            * np.arange(8, dtype=np.int64)[np.newaxis, :],
            11,
        )
        structural = build_legendre_mask_bank(
            11,
            signatures,
            policy_masks("typed16", 8),
            budget=structural_budget,
        )
        self.assertTrue(structural.overlap_one_verified)
        self.assertTrue(structural.rm_1_3_verified)
        self.assertTrue(structural.prime_3_mod_4_verified)
        self.assertTrue(structural.prw_t1_structure_verified)
        result = analyze_legendre_mask_bank(
            structural,
            planted_index=0,
            crossover=0.2,
            budget=structural_budget,
        )
        self.assertEqual(
            result["status"],
            "FINITE_PRW_T1_STRUCTURE_DIAGNOSTIC_ONLY",
        )
        self.assertTrue(
            result["structure_verification"]["direct_prw_t1_event_scope"]
        )
        self.assertFalse(result["transmitted_state_averaging_complete"])

    def test_resource_guards_refuse_before_large_analysis(self):
        tiny = EnumerationBudget(
            max_estimated_bytes=1024,
            max_hypotheses=8,
            max_competitor_pairs=2,
            max_coordinates=8,
            max_bruteforce_patterns=8,
        )
        with self.assertRaises(PRWT1ResourceError):
            build_legendre_mask_bank(
                7,
                ((0, 0), (0, 1)),
                ((1, 1),),
                budget=tiny,
            )
        with self.assertRaises(PRWT1ResourceError):
            brute_force_union_probability(
                np.ones((1, 4), dtype=bool),
                0.2,
                budget=tiny,
            )
        pair_limited = EnumerationBudget(
            max_estimated_bytes=16 * 1024 * 1024,
            max_hypotheses=64,
            max_competitor_pairs=2,
            max_coordinates=64,
            max_bruteforce_patterns=1024,
        )
        with self.assertRaises(PRWT1ResourceError):
            analyze_legendre_mask_bank(
                self.bank,
                planted_index=0,
                crossover=0.2,
                require_prw_t1_structure=False,
                budget=pair_limited,
            )

    def test_hard_caps_deadline_and_work_guard_fail_closed(self):
        with self.assertRaises(PRWT1ValidationError):
            EnumerationBudget(
                max_estimated_bytes=HARD_MAX_ESTIMATED_BYTES + 1
            )
        with self.assertRaises(PRWT1ValidationError):
            EnumerationBudget(max_seconds=HARD_MAX_SECONDS + 0.01)
        with self.assertRaises(PRWT1ValidationError):
            EnumerationBudget(max_work_units=HARD_MAX_WORK_UNITS + 1)
        work_limited = EnumerationBudget(
            max_estimated_bytes=16 * 1024 * 1024,
            max_work_units=1,
            max_hypotheses=64,
            max_competitor_pairs=128,
            max_coordinates=64,
            max_bruteforce_patterns=1024,
        )
        with self.assertRaises(PRWT1ResourceError):
            build_legendre_mask_bank(
                3,
                ((0, 0), (0, 1)),
                ((1, 1),),
                budget=work_limited,
            )
        with patch(
            "prime_ring_intersection.time.monotonic",
            side_effect=(0.0, HARD_MAX_SECONDS + 1.0),
        ):
            with self.assertRaises(PRWT1ResourceError):
                build_legendre_mask_bank(
                    3,
                    ((0, 0), (0, 1)),
                    ((1, 1),),
                    budget=self.budget,
                )

    def test_shape_preflight_refuses_before_cell_materialization(self):
        class ShapeOnlyRow(Sequence):
            def __len__(self):
                return 8

            def __getitem__(self, _index):
                raise AssertionError(
                    "cell materialization occurred before preflight refusal"
                )

        signatures = [ShapeOnlyRow()] * 1000
        masks = [ShapeOnlyRow()] * 1000
        with self.assertRaises(PRWT1ResourceError):
            build_legendre_mask_bank(
                7,
                signatures,
                masks,
                budget=self.budget,
            )
    def test_invalid_gauge_masks_and_duplicates_fail_closed(self):
        with self.assertRaises(PRWT1ValidationError):
            build_legendre_mask_bank(
                3,
                ((1, 0), (0, 1)),
                ((1, 1),),
                budget=self.budget,
            )
        with self.assertRaises(PRWT1ValidationError):
            build_legendre_mask_bank(
                3,
                ((0, 0), (0, 1)),
                ((1, 0),),
                budget=self.budget,
            )
        with self.assertRaises(PRWT1ValidationError):
            build_legendre_mask_bank(
                3,
                ((0, 0), (0, 0)),
                ((1, 1),),
                budget=self.budget,
            )


if __name__ == "__main__":
    unittest.main()
