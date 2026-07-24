import itertools
import math
import time
import unittest
from unittest.mock import patch

import numpy as np

from prime_ring_multifrequency import (
    FrequencyCondition,
    MultifrequencyBudget,
    _decode_multifrequency_with_deadline,
    build_legendre_phase_bank,
    decode_multifrequency,
)
from prime_ring_quantizer_null import (
    BOUNDARY_TOLERANCE_LEVEL_UNITS,
    DECLARED_ORIGINS,
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_EVALUATIONS,
    HARD_MAX_EXACT_PATTERNS,
    HARD_MAX_SECONDS,
    HARD_MAX_WORK_UNITS,
    MULTIPLIER_ORBITS,
    NONCONJUGATE_BINS,
    PRIME,
    PRWH1QResourceError,
    PRWH1QValidationError,
    QuantizerNullBudget,
    QuantizerNullSpec,
    TWO_BIN_SUBSETS,
    _build_phase_cache,
    _conditions,
    _decode_cached_phase,
    _near_half_step,
    _rank_groups,
    _summarize,
    preflight_p11_quantizer_null,
)


class TestP11QuantizerNullContract(unittest.TestCase):
    def test_all_pairs_split_into_two_exact_five_member_orbits(self):
        expected_pairs = tuple(itertools.combinations(range(1, 6), 2))
        self.assertEqual(TWO_BIN_SUBSETS, expected_pairs)
        self.assertEqual(len(TWO_BIN_SUBSETS), 10)
        self.assertEqual(set(MULTIPLIER_ORBITS), {"A", "B"})
        self.assertTrue(
            all(len(members) == 5 for members in MULTIPLIER_ORBITS.values())
        )
        self.assertEqual(
            set().union(*map(set, MULTIPLIER_ORBITS.values())),
            set(expected_pairs),
        )
        self.assertFalse(
            set(MULTIPLIER_ORBITS["A"])
            & set(MULTIPLIER_ORBITS["B"])
        )

    def test_scientific_grid_is_frozen_before_evaluation(self):
        spec = QuantizerNullSpec()
        self.assertEqual(spec.origins, DECLARED_ORIGINS)
        self.assertEqual(spec.phase_bits, 8)
        with self.assertRaises(PRWH1QValidationError):
            QuantizerNullSpec(origins=(0.0, 0.5, 0.75, 0.875))
        with self.assertRaises(PRWH1QValidationError):
            QuantizerNullSpec(origins=list(DECLARED_ORIGINS))
        with self.assertRaises(PRWH1QValidationError):
            QuantizerNullSpec(phase_bits=7)

    def test_local_and_upstream_hard_ceilings_cannot_be_raised(self):
        invalid = (
            {"max_estimated_bytes": HARD_MAX_ESTIMATED_BYTES + 1},
            {"max_seconds": HARD_MAX_SECONDS + 0.01},
            {"max_work_units": HARD_MAX_WORK_UNITS + 1},
            {"max_exact_patterns": HARD_MAX_EXACT_PATTERNS + 1},
            {"max_evaluations": HARD_MAX_EVALUATIONS + 1},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(PRWH1QValidationError):
                    QuantizerNullBudget(**values)

    def test_budget_subclasses_are_normalized_before_resource_use(self):
        class HostileInt(int):
            def __ge__(self, other):
                return True

            def __le__(self, other):
                return True

        class HostileFloat(float):
            def __radd__(self, other):
                return -1.0

        budget = QuantizerNullBudget(
            max_estimated_bytes=HostileInt(1),
            max_seconds=HostileFloat(1.0),
        )
        self.assertIs(type(budget.max_estimated_bytes), int)
        self.assertIs(type(budget.max_seconds), float)
        with self.assertRaises(PRWH1QResourceError):
            preflight_p11_quantizer_null(budget=budget)

    def test_complete_batch_is_preflighted_without_running_a_decoder(self):
        with patch(
            "prime_ring_quantizer_null._decode_cached_phase",
            side_effect=AssertionError("preflight must not decode"),
        ):
            plan = preflight_p11_quantizer_null()
        self.assertEqual(plan["pattern_count"], 2048)
        self.assertEqual(plan["evaluation_count"], 50)
        self.assertEqual(
            plan["cached_condition_evaluation_count"],
            102400,
        )
        self.assertEqual(plan["public_reference_decoder_invocation_count"], 0)
        self.assertEqual(
            plan["per_cached_condition_estimated_work_units"],
            164,
        )
        self.assertEqual(plan["query_fft_count"], 2048)
        self.assertEqual(plan["template_fft_count"], 1)
        self.assertEqual(plan["estimated_work_units"], 17951240)
        self.assertEqual(plan["estimated_peak_bytes"], 54432)
        self.assertTrue(plan["single_aggregate_deadline"])
        self.assertTrue(
            plan[
                "preflight_before_first_cached_condition_evaluation"
            ]
        )
        self.assertTrue(plan["patterns_streamed_once"])
        self.assertFalse(plan["pattern_matrix_allocated"])
        self.assertTrue(plan["template_fft_cached_across_entire_batch"])
        self.assertTrue(
            plan[
                "same_query_fft_shared_across_all_evaluations_per_pattern"
            ]
        )

    def test_lowered_pattern_guard_refuses_before_bank_or_decoder(self):
        budget = QuantizerNullBudget(max_exact_patterns=2047)
        with patch(
            "prime_ring_quantizer_null.build_legendre_phase_bank",
            side_effect=AssertionError("bank must not be built"),
        ), patch(
            "prime_ring_quantizer_null._decode_cached_phase",
            side_effect=AssertionError("decoder must not run"),
        ):
            with self.assertRaises(PRWH1QResourceError):
                preflight_p11_quantizer_null(budget=budget)

    def test_lowered_aggregate_work_guard_refuses_before_decoder(self):
        budget = QuantizerNullBudget(max_work_units=17_900_000)
        with patch(
            "prime_ring_quantizer_null._decode_cached_phase",
            side_effect=AssertionError("decoder must not run"),
        ):
            with self.assertRaisesRegex(
                PRWH1QResourceError,
                "aggregate work estimate",
            ):
                preflight_p11_quantizer_null(budget=budget)

    def test_stronger_symmetry_closed_grid_is_explicitly_refused(self):
        extension = preflight_p11_quantizer_null()[
            "symmetry_closed_per_bin_extension"
        ]
        self.assertFalse(extension["implemented"])
        self.assertEqual(extension["origin_vectors_per_pair"], 16)
        self.assertEqual(extension["evaluation_count"], 170)
        self.assertEqual(
            extension["cached_condition_evaluation_count"],
            348160,
        )
        self.assertEqual(
            extension["estimated_work_units_on_current_cached_path"],
            59734280,
        )
        self.assertTrue(extension["exceeds_local_hard_work_ceiling"])
        self.assertTrue(extension["exceeds_upstream_hard_work_ceiling"])
        self.assertFalse(
            extension["current_condition_model_supports_per_bin_origins"]
        )


class TestCachedPhaseScorerEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = QuantizerNullSpec()
        cls.bank = build_legendre_phase_bank(
            PRIME,
            np.asarray(((0,),), dtype=np.int64),
        )
        cls.conditions = _conditions(cls.spec)
        cls.cached, _ = _build_phase_cache(
            cls.bank,
            cls.conditions,
            deadline=time.monotonic() + 5.0,
        )
        cls.cached_by_key = {
            (item.pair, item.origin): item for item in cls.cached
        }
        cls.reference_budget = MultifrequencyBudget(
            max_hypotheses=PRIME,
            max_prime=PRIME,
            max_types=1,
            max_nodes=1,
            max_conditions=1,
        )

    def assert_equivalent(self, query, pair, origin):
        cached = self.cached_by_key[(pair, origin)]
        deadline = time.monotonic() + 5.0
        reference = _decode_multifrequency_with_deadline(
            self.bank,
            query,
            cached.condition,
            budget=self.reference_budget,
            deadline=deadline,
        )
        query_spectrum = np.fft.fft(query, axis=1)
        specialized = _decode_cached_phase(
            self.bank,
            query_spectrum,
            cached,
            deadline=deadline,
        )
        self.assertEqual(specialized.abstained, reference.abstained)
        self.assertEqual(
            specialized.abstention_reason,
            reference.abstention_reason,
        )
        self.assertEqual(
            specialized.unique_winner,
            reference.unique_winner,
        )
        self.assertEqual(specialized.top_states, reference.top_states)
        self.assertEqual(
            specialized.candidate_count,
            reference.candidate_count,
        )
        np.testing.assert_allclose(
            specialized.scores,
            reference.scores,
            rtol=0.0,
            atol=1e-14,
        )
        for name in ("best_score", "runner_up_score", "margin"):
            left = getattr(specialized, name)
            right = getattr(reference, name)
            if left is None or right is None:
                self.assertIsNone(left)
                self.assertIsNone(right)
            else:
                self.assertAlmostEqual(left, right, places=14)
        return specialized

    def test_clean_and_wrong_unique_states_match_in_both_orbits(self):
        template = self.bank.templates[0]
        clean = np.roll(template, 3, axis=1)
        wrong = np.roll(template, 4, axis=1)
        for pair in ((1, 2), (1, 3)):
            for origin in (None, *DECLARED_ORIGINS):
                with self.subTest(pair=pair, origin=origin, state="correct"):
                    correct = self.assert_equivalent(clean, pair, origin)
                    self.assertEqual(correct.top_states, ((0, 3),))
                with self.subTest(pair=pair, origin=origin, state="wrong"):
                    incorrect = self.assert_equivalent(wrong, pair, origin)
                    self.assertEqual(incorrect.top_states, ((0, 4),))

    def test_reference_tie_and_quantized_tie_breaks_match(self):
        template = self.bank.templates[0]
        query = np.roll(template, 3, axis=1) + np.roll(
            template,
            4,
            axis=1,
        )
        unquantized = self.assert_equivalent(query, (1, 2), None)
        self.assertFalse(unquantized.unique_winner)
        self.assertEqual(unquantized.top_states, ((0, 3), (0, 4)))
        self.assertEqual(unquantized.margin, 0.0)
        for origin in DECLARED_ORIGINS:
            with self.subTest(origin=origin):
                self.assert_equivalent(query, (1, 2), origin)

    def test_zero_spectrum_abstention_matches_for_every_variant(self):
        query = np.ones((1, PRIME), dtype=np.float64)
        for origin in (None, *DECLARED_ORIGINS):
            with self.subTest(origin=origin):
                result = self.assert_equivalent(query, (1, 2), origin)
                self.assertTrue(result.abstained)
                self.assertIn("zero magnitude", result.abstention_reason)

    def test_cached_arrays_are_read_only_and_corrections_are_shared(self):
        self.assertEqual(len(self.cached), 50)
        self.assertEqual(
            len({id(item.correction) for item in self.cached}),
            10,
        )
        for item in self.cached:
            self.assertFalse(item.template_unit.flags.writeable)
            self.assertFalse(item.correction.flags.writeable)
            self.assertTrue(item.accounting["cached_template_fft"])
            self.assertTrue(item.accounting["query_fft_shared_per_pattern"])


class TestP11BoundaryAndSummary(unittest.TestCase):
    def test_exact_half_step_is_audited_with_declared_tolerance(self):
        angle = (7.5 * 2.0 * math.pi) / (1 << 8)
        values = np.asarray((np.exp(1.0j * angle),), dtype=np.complex128)
        near, zero = _near_half_step(
            values,
            phase_bits=8,
            origin=0.0,
        )
        self.assertGreater(BOUNDARY_TOLERANCE_LEVEL_UNITS, 0.0)
        self.assertTrue(bool(near[0]))
        self.assertFalse(bool(zero[0]))

    def test_constant_query_zero_bins_are_not_misclassified_as_boundaries(self):
        bank = build_legendre_phase_bank(
            PRIME,
            np.asarray(((0,),), dtype=np.int64),
        )
        base = np.roll(bank.templates[0], 3, axis=1).copy()
        pattern = 0
        for coordinate, value in enumerate(base.reshape(-1)):
            if value < 0.0:
                pattern |= 1 << coordinate
        query = base.copy()
        for coordinate in range(PRIME):
            if pattern & (1 << coordinate):
                query.reshape(-1)[coordinate] *= -1.0
        np.testing.assert_array_equal(query, np.ones_like(query))
        selected = np.fft.fft(query, axis=1)[0][
            list(NONCONJUGATE_BINS)
        ]
        near, zero = _near_half_step(
            selected,
            phase_bits=8,
            origin=0.0,
        )
        self.assertFalse(bool(np.any(near)))
        self.assertTrue(bool(np.all(zero)))
        decoded = decode_multifrequency(
            bank,
            query,
            FrequencyCondition(
                "constant-query",
                "phase",
                (1, 2),
                phase_bits=8,
            ),
        )
        self.assertTrue(decoded.abstained)
        self.assertIn("zero magnitude", decoded.abstention_reason)

    def test_rank_groups_preserve_near_ties_instead_of_strict_ranking(self):
        values = {
            (1, 2): 0.5,
            (1, 3): 0.5 + 5e-13,
            (1, 4): 0.4,
        }
        groups = _rank_groups(values, 1e-12)
        self.assertEqual(groups[0], [[1, 2], [1, 3]])
        self.assertEqual(groups[1], [[1, 4]])

    def test_summary_reports_reversals_grid_average_and_orbit_caveat(self):
        spec = QuantizerNullSpec()

        def outcome(accuracy):
            return {
                "correct": accuracy,
                "wrong_unique": 1.0 - accuracy,
                "tie": 0.0,
                "abstain": 0.0,
            }

        outcomes = {}
        boundary = {}
        for pair in TWO_BIN_SUBSETS:
            outcomes[(pair, None)] = outcome(0.5)
            for origin_index, origin in enumerate(spec.origins):
                accuracy = 0.5
                if pair == (1, 2):
                    accuracy = 0.6 if origin_index % 2 == 0 else 0.4
                elif pair == (1, 3):
                    accuracy = 0.4 if origin_index % 2 == 0 else 0.6
                outcomes[(pair, origin)] = outcome(accuracy)
                boundary[(pair, origin)] = {
                    "template_count": 0.0,
                    "template_zero_count": 0.0,
                    "query_pattern_count": 0.0,
                    "query_probability_mass": 0.0,
                    "any_probability_mass": 0.0,
                    "query_zero_pattern_count": 0.0,
                    "query_zero_probability_mass": 0.0,
                    "any_zero_probability_mass": 0.0,
                }
        report = _summarize(
            spec,
            outcomes,
            boundary,
            probability_mass=1.0,
        )
        self.assertTrue(
            report["unquantized_diagnostic"][
                "within_each_multiplier_orbit_exchangeable_within_tolerance"
            ]
        )
        grid = report["declared_origin_grid_average"]
        self.assertTrue(grid["all_subsets_exchangeable_within_tolerance"])
        self.assertFalse(grid["stochastic_dither_implemented"])
        self.assertFalse(
            grid["continuous_uniform_origin_integration_implemented"]
        )
        self.assertGreater(
            report["rank_stability"]["rank_reversal_pair_count"],
            0,
        )
        cells = {tuple(cell["bins"]): cell for cell in report["cells"]}
        self.assertEqual(cells[(1, 2)]["multiplier_orbit"], "A")
        self.assertEqual(cells[(1, 3)]["multiplier_orbit"], "B")
        self.assertFalse(
            report["boundary_audit_contract"][
                "zero_magnitudes_classified_as_boundaries"
            ]
        )


if __name__ == "__main__":
    unittest.main()
