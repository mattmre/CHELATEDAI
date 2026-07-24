import itertools
import math
import unittest
from collections.abc import Sequence
from unittest.mock import patch

import numpy as np

from prime_ring_multifrequency import (
    DEFAULT_MAX_ESTIMATED_BYTES,
    EVIDENCE_STATUS,
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_EXACT_PATTERNS,
    HARD_MAX_HYPOTHESES,
    HARD_MAX_PRIME,
    HARD_MAX_SECONDS,
    HARD_MAX_WORK_UNITS,
    FrequencyCondition,
    MultifrequencyBank,
    MultifrequencyBudget,
    PRWH1ResourceError,
    PRWH1ValidationError,
    _quantized_unit_phase,
    _score_summary,
    build_legendre_phase_bank,
    compare_multifrequency_controls,
    decode_multifrequency,
    deterministic_control_bins,
    estimate_bank_peak_bytes,
    estimate_decode_peak_bytes,
    exact_bsc_comparison,
)
from prime_ring_waypoint import legendre_carrier


class TestMultifrequencyConstruction(unittest.TestCase):
    def setUp(self):
        self.signatures = np.asarray(((0, 0), (0, 1)), dtype=np.int64)
        self.bank = build_legendre_phase_bank(7, self.signatures)

    def test_legendre_bank_and_fourier_rotation_law(self):
        self.assertTrue(self.bank.overlap_one_verified)
        self.assertTrue(
            self.bank.legendre_equal_nonzero_magnitudes_verified
        )
        self.assertEqual(self.bank.hypothesis_count, 14)
        carrier = legendre_carrier(7).astype(np.float64)
        shift = 3
        spectrum = np.fft.fft(carrier)
        shifted = np.fft.fft(np.roll(carrier, shift))
        for frequency in (1, 2, 3):
            expected = spectrum[frequency] * np.exp(
                -2.0j * math.pi * frequency * shift / 7
            )
            self.assertAlmostEqual(
                abs(shifted[frequency] - expected), 0.0, places=12
            )

    def test_deterministic_control_bins_are_reproducible_and_distinct(self):
        first = deterministic_control_bins(31, 5, 42)
        second = deterministic_control_bins(31, 5, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(all(1 <= value <= 15 for value in first))
        with self.assertRaises(PRWH1ResourceError):
            deterministic_control_bins(HARD_MAX_PRIME + 2, 1, 42)

    def test_shape_and_resource_preflight_happen_before_carrier_build(self):
        tiny = MultifrequencyBudget(max_estimated_bytes=1)
        with patch(
            "prime_ring_multifrequency.legendre_carrier",
            side_effect=AssertionError("must not allocate carrier"),
        ):
            with self.assertRaises(PRWH1ResourceError):
                build_legendre_phase_bank(
                    7, self.signatures, budget=tiny
                )

    def test_signature_normalization_copy_is_preflighted(self):
        signatures = self.signatures.astype(np.int32)
        base_estimate = estimate_bank_peak_bytes(
            prime=7,
            type_count=signatures.shape[0],
            node_count=signatures.shape[1],
        )
        restricted = MultifrequencyBudget(
            max_estimated_bytes=base_estimate + signatures.nbytes - 1
        )
        with patch(
            "prime_ring_multifrequency.np.ascontiguousarray",
            side_effect=AssertionError("normalization must not allocate"),
        ):
            with self.assertRaises(PRWH1ResourceError):
                build_legendre_phase_bank(
                    7,
                    signatures,
                    budget=restricted,
                )

    def test_hard_resource_ceilings_cannot_be_raised(self):
        invalid = (
            {"max_estimated_bytes": HARD_MAX_ESTIMATED_BYTES + 1},
            {"max_seconds": HARD_MAX_SECONDS + 0.1},
            {"max_work_units": HARD_MAX_WORK_UNITS + 1},
            {"max_hypotheses": HARD_MAX_HYPOTHESES + 1},
            {"max_exact_patterns": HARD_MAX_EXACT_PATTERNS + 1},
            {"max_prime": HARD_MAX_PRIME + 1},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(PRWH1ValidationError):
                    MultifrequencyBudget(**values)
        self.assertLess(
            DEFAULT_MAX_ESTIMATED_BYTES, HARD_MAX_ESTIMATED_BYTES
        )

    def test_invalid_signatures_and_conjugate_bins_fail_closed(self):
        with self.assertRaises(PRWH1ValidationError):
            build_legendre_phase_bank(
                7, np.asarray(((1, 0),), dtype=np.int64)
            )
        query = np.roll(self.bank.templates[0], 1, axis=1)
        with self.assertRaises(PRWH1ValidationError):
            decode_multifrequency(
                self.bank,
                query,
                FrequencyCondition("bad", "phase", (4,)),
            )

    def test_quantizer_origin_requires_phase_quantization(self):
        with self.assertRaises(PRWH1ValidationError):
            FrequencyCondition(
                "bad-origin",
                "phase",
                (1,),
                quantizer_origin_fraction=0.25,
            )
        for origin in (-0.01, 1.0, math.inf, math.nan):
            with self.subTest(origin=origin):
                with self.assertRaises(PRWH1ValidationError):
                    FrequencyCondition(
                        "bad-origin",
                        "phase",
                        (1,),
                        phase_bits=8,
                        quantizer_origin_fraction=origin,
                    )
        for mode, bins in (("magnitude", (1,)), ("time_domain", ())):
            with self.subTest(mode=mode):
                with self.assertRaises(PRWH1ValidationError):
                    FrequencyCondition(
                        "inapplicable-quantizer",
                        mode,
                        bins,
                        phase_bits=8,
                    )

    def test_condition_normalizes_used_integer_fields_and_rejects_dead_seed(self):
        condition = FrequencyCondition(
            "normalized",
            "randomized_phase",
            (np.int64(1),),
            seed=np.int64(7),
            phase_bits=np.int64(8),
            quantizer_origin_fraction=np.float64(0.25),
        )
        self.assertIs(type(condition.seed), int)
        self.assertIs(type(condition.phase_bits), int)
        self.assertIs(type(condition.quantizer_origin_fraction), float)
        with self.assertRaises(PRWH1ValidationError):
            FrequencyCondition(
                "dead-seed",
                "phase",
                (1,),
                seed=7,
            )

    def test_shifted_quantizer_boundary_has_deterministic_upper_tie(self):
        boundary = np.asarray(
            (np.exp(1.0j * 3.0 * math.pi / 8.0),),
            dtype=np.complex128,
        )
        quantized = _quantized_unit_phase(
            boundary,
            phase_bits=2,
            quantizer_origin_fraction=0.25,
        )
        expected = np.exp(1.0j * 5.0 * math.pi / 8.0)
        self.assertAlmostEqual(abs(quantized[0] - expected), 0.0, places=14)


class TestMultifrequencyDecode(unittest.TestCase):
    def setUp(self):
        signatures = np.asarray(((0, 0), (0, 1)), dtype=np.int64)
        self.bank = build_legendre_phase_bank(7, signatures)
        self.truth = (1, 3)
        self.query = np.roll(
            self.bank.templates[self.truth[0]], self.truth[1], axis=1
        )
        self.comparison = compare_multifrequency_controls(
            self.bank,
            self.query,
            (1, 2),
            planted_state=self.truth,
            control_seed=42,
        )

    def test_noiseless_selected_bins_recover_one_shared_state(self):
        proposed = self.comparison["results"]["proposed_bins"]["decode"]
        self.assertTrue(proposed.unique_winner)
        self.assertEqual(proposed.top_states, (self.truth,))
        self.assertEqual(proposed.candidate_count, 14)
        self.assertAlmostEqual(proposed.best_score, 1.0, places=12)
        self.assertEqual(self.comparison["status"], EVIDENCE_STATUS)
        self.assertFalse(
            self.comparison["independent_shift_per_bin_permitted"]
        )
        self.assertFalse(self.comparison["novelty_claim"])
        self.assertFalse(self.comparison["promotion_evidence"])
        self.assertFalse(
            self.comparison["frequency_diversity_established"]
        )

    def test_all_conditions_keep_the_same_hypothesis_count(self):
        for name, result in self.comparison["results"].items():
            with self.subTest(name=name):
                self.assertEqual(
                    result["decode"].candidate_count,
                    self.bank.hypothesis_count,
                )
                self.assertTrue(
                    result["hypothesis_budget_matches_proposed"]
                )

    def test_phase_features_are_unit_energy_and_matched_controls_reported(self):
        proposed = self.comparison["results"]["proposed_bins"]
        random_bins = self.comparison["results"]["random_same_count_bins"]
        repeated = self.comparison["results"]["repeated_single_bin"]
        for result in (proposed, random_bins, repeated):
            self.assertEqual(
                result["decode"].accounting["feature_norm"], 1.0
            )
            self.assertTrue(
                result["decode"].accounting["unit_energy_normalization"]
            )
            self.assertTrue(
                result["coefficient_budget_matches_proposed"]
            )
            self.assertTrue(
                result["arithmetic_slot_budget_matches_proposed"]
            )
            self.assertTrue(
                result["stored_scalar_budget_matches_proposed"]
            )
        time_domain = self.comparison["results"]["time_domain"]
        self.assertTrue(time_domain["energy_budget_matches_proposed"])
        self.assertTrue(
            time_domain["decode"].accounting["unit_energy_normalization"]
        )
        self.assertFalse(
            self.comparison["random_control_collides_with_proposed"]
        )
        self.assertTrue(
            self.comparison["selected_frequency_comparison_eligible"]
        )
        self.assertFalse(
            self.comparison["all_bins_control_collides_with_proposed"]
        )
        self.assertEqual(
            self.comparison["aggregate_resource_guard"]["deadline_scope"],
            "all_conditions",
        )
        active_peak = max(
            estimate_decode_peak_bytes(
                self.bank,
                result["decode"].condition,
                self.query,
            )
            for result in self.comparison["results"].values()
        )
        retained_scores = (
            (len(self.comparison["results"]) - 1)
            * self.bank.type_count
            * self.bank.prime
            * 8
        )
        resource = self.comparison["aggregate_resource_guard"]
        self.assertEqual(
            resource["estimated_peak_bytes"],
            active_peak + retained_scores,
        )
        self.assertTrue(resource["retained_prior_score_arrays_included"])
        self.assertEqual(
            resource["byte_scope"],
            "model_arrays_and_declared_temporaries_not_process_rss",
        )

    def test_repeated_single_bin_scores_equal_single_bin_scores(self):
        repeated = self.comparison["results"]["repeated_single_bin"][
            "decode"
        ]
        single = self.comparison["results"]["single_bin"]["decode"]
        np.testing.assert_allclose(
            repeated.scores, single.scores, rtol=0.0, atol=1e-14
        )

    def test_magnitude_only_is_shift_invariant_and_tied(self):
        magnitude = self.comparison["results"]["magnitude_only"]["decode"]
        self.assertFalse(magnitude.unique_winner)
        self.assertEqual(
            len(magnitude.top_states), self.bank.hypothesis_count
        )
        for row in magnitude.scores:
            np.testing.assert_allclose(
                row, np.full_like(row, row[0]), rtol=0.0, atol=1e-14
            )
        magnitude_contract = self.comparison["results"]["magnitude_only"]
        self.assertFalse(
            magnitude_contract["stored_scalar_budget_matches_proposed"]
        )
        self.assertFalse(
            magnitude_contract["arithmetic_slot_budget_matches_proposed"]
        )

    def test_eight_bit_phase_quantization_retains_clean_recovery(self):
        quantized = compare_multifrequency_controls(
            self.bank,
            self.query,
            (1, 2),
            planted_state=self.truth,
            control_seed=42,
            phase_bits=8,
        )
        proposed = quantized["results"]["proposed_bins"]["decode"]
        self.assertTrue(proposed.unique_winner)
        self.assertEqual(proposed.top_states, (self.truth,))
        contract = quantized["quantization_contract"]
        self.assertTrue(contract["unit_phase_angles_quantized"])
        self.assertEqual(
            contract["fft_and_score_arithmetic"],
            "float64_and_complex128",
        )
        self.assertFalse(contract["packed_representation_implemented"])
        self.assertFalse(
            contract["packed_storage_or_speed_claim_eligible"]
        )

    def test_quantizer_origins_are_echoed_and_magnitude_is_inapplicable(self):
        for origin in (0.0, 0.25, 0.5, 0.75):
            with self.subTest(origin=origin):
                quantized = compare_multifrequency_controls(
                    self.bank,
                    self.query,
                    (1, 2),
                    planted_state=self.truth,
                    control_seed=42,
                    phase_bits=8,
                    quantizer_origin_fraction=origin,
                )
                contract = quantized["quantization_contract"]
                self.assertEqual(
                    contract["quantizer_origin_fraction"], origin
                )
                self.assertEqual(
                    contract["applies_to_modes"],
                    ["phase", "randomized_phase"],
                )
                magnitude = quantized["results"]["magnitude_only"]["decode"]
                self.assertIsNone(magnitude.condition.phase_bits)
                self.assertEqual(
                    magnitude.condition.quantizer_origin_fraction, 0.0
                )
                self.assertFalse(
                    contract["magnitude_only_quantization_applied"]
                )

    def test_unquantized_zero_origin_is_behaviorally_unchanged(self):
        repeated = compare_multifrequency_controls(
            self.bank,
            self.query,
            (1, 2),
            planted_state=self.truth,
            control_seed=42,
            phase_bits=None,
            quantizer_origin_fraction=0.0,
        )
        for name in self.comparison["results"]:
            with self.subTest(name=name):
                np.testing.assert_array_equal(
                    repeated["results"][name]["decode"].scores,
                    self.comparison["results"][name]["decode"].scores,
                )

    def test_public_compare_rejects_origin_without_quantization(self):
        with self.assertRaises(PRWH1ValidationError):
            compare_multifrequency_controls(
                self.bank,
                self.query,
                (1, 2),
                phase_bits=None,
                quantizer_origin_fraction=0.25,
            )

    def test_zero_selected_magnitude_causes_fail_closed_abstention(self):
        templates = np.ones((1, 1, 7), dtype=np.float64)
        signatures = np.zeros((1, 1), dtype=np.int64)
        templates.setflags(write=False)
        signatures.setflags(write=False)
        constant_bank = MultifrequencyBank(
            prime=7,
            templates=templates,
            phase_signatures=signatures,
            overlap_one_verified=True,
            legendre_equal_nonzero_magnitudes_verified=False,
            build_estimated_peak_bytes=templates.nbytes,
            build_estimated_work_units=7,
        )
        decoded = decode_multifrequency(
            constant_bank,
            np.ones((1, 7), dtype=np.float64),
            FrequencyCondition("zero-bin", "phase", (1,)),
        )
        self.assertTrue(decoded.abstained)
        self.assertFalse(decoded.unique_winner)
        self.assertIn("zero magnitude", decoded.abstention_reason)

    def test_unrelated_validation_error_with_zero_text_propagates(self):
        condition = FrequencyCondition("phase", "phase", (1,))
        with patch(
            "prime_ring_multifrequency._decode_fourier",
            side_effect=PRWH1ValidationError(
                "unrelated zero-valued metadata is invalid"
            ),
        ):
            with self.assertRaisesRegex(
                PRWH1ValidationError,
                "unrelated zero-valued metadata",
            ):
                decode_multifrequency(
                    self.bank,
                    self.query,
                    condition,
                )

    def test_scores_are_read_only(self):
        proposed = self.comparison["results"]["proposed_bins"]["decode"]
        self.assertFalse(proposed.scores.flags.writeable)
        with self.assertRaises(ValueError):
            proposed.scores[0, 0] = 0.0

    def test_tied_winners_have_zero_margin(self):
        winners, best, runner_up, margin = _score_summary(
            np.asarray(((1.0, 1.0), (0.0, 0.0)))
        )
        self.assertEqual(winners, ((0, 0), (0, 1)))
        self.assertEqual(best, 1.0)
        self.assertEqual(runner_up, 1.0)
        self.assertEqual(margin, 0.0)

    def test_compare_preflights_bank_before_control_construction(self):
        templates = np.ones((1, 1, 32), dtype=np.float64)
        signatures = np.zeros((1, 1), dtype=np.int64)
        malformed = MultifrequencyBank(
            prime=32,
            templates=templates,
            phase_signatures=signatures,
            overlap_one_verified=True,
            legendre_equal_nonzero_magnitudes_verified=False,
            build_estimated_peak_bytes=templates.nbytes,
            build_estimated_work_units=1,
        )
        with patch(
            "prime_ring_multifrequency.deterministic_control_bins",
            side_effect=AssertionError("controls must not be constructed"),
        ):
            with self.assertRaises(PRWH1ResourceError):
                compare_multifrequency_controls(
                    malformed,
                    np.ones((1, 32), dtype=np.float64),
                    (1,),
                )

    def test_compare_rejects_frozen_condition_count_drift(self):
        short_conditions = tuple(
            FrequencyCondition(f"short-{index}", "phase", (1,))
            for index in range(7)
        )
        with patch(
            "prime_ring_multifrequency._conditions",
            return_value=short_conditions,
        ):
            with self.assertRaisesRegex(
                PRWH1ValidationError,
                "control condition count changed",
            ):
                compare_multifrequency_controls(
                    self.bank,
                    self.query,
                    (1, 2),
                )

    def test_bins_refuse_by_length_before_cell_access(self):
        class BombBins(Sequence):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                raise AssertionError("bin cell must not be accessed")

        with self.assertRaises(PRWH1ResourceError):
            compare_multifrequency_controls(
                self.bank,
                self.query,
                BombBins(),
            )

    def test_unsized_bin_iterator_is_rejected_without_iteration(self):
        def bomb():
            raise AssertionError("iterator must not be consumed")
            yield 1

        with self.assertRaises(PRWH1ValidationError):
            compare_multifrequency_controls(
                self.bank,
                self.query,
                bomb(),
            )

    def test_query_normalization_copy_is_preflighted(self):
        condition = FrequencyCondition("phase", "phase", (1,))
        query = self.query.astype(np.float32)
        without_copy = estimate_decode_peak_bytes(
            self.bank, condition
        )
        with_copy = estimate_decode_peak_bytes(
            self.bank, condition, query
        )
        self.assertGreater(with_copy, without_copy)
        restricted = MultifrequencyBudget(
            max_estimated_bytes=with_copy - 1
        )
        with patch(
            "prime_ring_multifrequency.np.ascontiguousarray",
            side_effect=AssertionError("normalization must not allocate"),
        ):
            with self.assertRaises(PRWH1ResourceError):
                decode_multifrequency(
                    self.bank,
                    query,
                    condition,
                    budget=restricted,
                )


class TestExactBSCComparison(unittest.TestCase):
    def setUp(self):
        signatures = np.asarray(((0, 0), (0, 1)), dtype=np.int64)
        self.bank = build_legendre_phase_bank(3, signatures)
        self.truth = (1, 2)
        self.crossover = 0.2
        self.result = exact_bsc_comparison(
            self.bank,
            planted_type=self.truth[0],
            planted_shift=self.truth[1],
            crossover=self.crossover,
            bins=(1,),
            control_seed=7,
        )

    def test_streamed_enumeration_has_complete_probability_mass(self):
        self.assertEqual(self.result["bit_count"], 6)
        self.assertEqual(self.result["pattern_count"], 64)
        self.assertAlmostEqual(
            self.result["probability_mass"], 1.0, places=14
        )
        self.assertEqual(
            self.result["enumeration_storage"],
            "streamed_without_pattern_matrix",
        )
        self.assertFalse(self.result["novelty_claim"])
        self.assertFalse(self.result["promotion_evidence"])
        self.assertFalse(self.result["false_unlock_rate_computed"])
        self.assertEqual(
            self.result["condition_contracts"]["proposed_bins"][
                "candidate_count"
            ],
            self.bank.hypothesis_count,
        )
        self.assertTrue(
            self.result["random_control_collides_with_proposed"]
        )
        self.assertFalse(
            self.result["selected_frequency_comparison_eligible"]
        )
        for name in self.result["correct_probability"]:
            categorized = sum(
                self.result[field][name]
                for field in (
                    "correct_probability",
                    "wrong_unique_probability",
                    "tie_probability",
                    "abstain_probability",
                )
            )
            self.assertAlmostEqual(
                categorized,
                self.result["probability_mass"],
                places=13,
            )

    def test_exact_accuracy_matches_independent_pattern_loop(self):
        base = np.roll(
            self.bank.templates[self.truth[0]], self.truth[1], axis=1
        )
        manual = 0.0
        for bits in itertools.product((0, 1), repeat=6):
            query = base.copy()
            for index, flipped in enumerate(bits):
                if flipped:
                    query.reshape(-1)[index] *= -1.0
            weight = sum(bits)
            mass = self.crossover**weight * (
                1.0 - self.crossover
            ) ** (6 - weight)
            decoded = decode_multifrequency(
                self.bank,
                query,
                FrequencyCondition("proposed", "phase", (1,)),
            )
            if (
                decoded.unique_winner
                and decoded.top_states[0] == self.truth
            ):
                manual += mass
        self.assertAlmostEqual(
            self.result["correct_probability"]["proposed_bins"],
            manual,
            places=14,
        )

    def test_paired_effect_identity_is_exact(self):
        proposed = self.result["correct_probability"]["proposed_bins"]
        for name, effect in self.result[
            "paired_effects_vs_proposed"
        ].items():
            control = self.result["correct_probability"][name]
            self.assertAlmostEqual(
                effect["paired_accuracy_difference"],
                proposed - control,
                places=14,
            )

    def test_pattern_guard_refuses_before_enumeration(self):
        restricted = MultifrequencyBudget(max_exact_patterns=32)
        with self.assertRaises(PRWH1ResourceError):
            exact_bsc_comparison(
                self.bank,
                planted_type=self.truth[0],
                planted_shift=self.truth[1],
                crossover=self.crossover,
                bins=(1,),
                budget=restricted,
            )

    def test_exact_preflights_bytes_before_content_scan(self):
        restricted = MultifrequencyBudget(max_estimated_bytes=1)
        with patch(
            "prime_ring_multifrequency._bipolar_templates",
            side_effect=AssertionError("content must not be scanned"),
        ):
            with self.assertRaises(PRWH1ResourceError):
                exact_bsc_comparison(
                    self.bank,
                    planted_type=self.truth[0],
                    planted_shift=self.truth[1],
                    crossover=self.crossover,
                    bins=(1,),
                    budget=restricted,
                )

    def test_exact_honors_caller_lowered_condition_cap(self):
        restricted = MultifrequencyBudget(max_conditions=1)
        with self.assertRaises(PRWH1ResourceError):
            exact_bsc_comparison(
                self.bank,
                planted_type=self.truth[0],
                planted_shift=self.truth[1],
                crossover=self.crossover,
                bins=(1,),
                budget=restricted,
            )

    def test_exact_rejects_frozen_condition_count_drift(self):
        short_conditions = tuple(
            FrequencyCondition(f"short-{index}", "phase", (1,))
            for index in range(7)
        )
        with patch(
            "prime_ring_multifrequency._conditions",
            return_value=short_conditions,
        ):
            with self.assertRaisesRegex(
                PRWH1ValidationError,
                "control condition count changed",
            ):
                exact_bsc_comparison(
                    self.bank,
                    planted_type=self.truth[0],
                    planted_shift=self.truth[1],
                    crossover=self.crossover,
                    bins=(1,),
                )

    def test_exact_quantizer_origins_complete_and_echo_contract(self):
        for origin in (0.0, 0.25, 0.5, 0.75):
            with self.subTest(origin=origin):
                result = exact_bsc_comparison(
                    self.bank,
                    planted_type=self.truth[0],
                    planted_shift=self.truth[1],
                    crossover=self.crossover,
                    bins=(1,),
                    phase_bits=8,
                    quantizer_origin_fraction=origin,
                )
                self.assertAlmostEqual(
                    result["probability_mass"], 1.0, places=14
                )
                self.assertEqual(
                    result["quantization_contract"][
                        "quantizer_origin_fraction"
                    ],
                    origin,
                )
                magnitude = result["condition_contracts"][
                    "magnitude_only"
                ]
                self.assertIsNone(magnitude["phase_bits"])
                self.assertEqual(
                    magnitude["quantizer_origin_fraction"], 0.0
                )
                self.assertFalse(magnitude["quantization_applied"])

    def test_public_exact_rejects_origin_without_quantization(self):
        with self.assertRaises(PRWH1ValidationError):
            exact_bsc_comparison(
                self.bank,
                planted_type=self.truth[0],
                planted_shift=self.truth[1],
                crossover=self.crossover,
                bins=(1,),
                phase_bits=None,
                quantizer_origin_fraction=0.25,
            )

    def test_p7_bin_ranking_changes_with_quantizer_grid_origin(self):
        bank = build_legendre_phase_bank(
            7,
            np.asarray(((0,),), dtype=np.int64),
        )
        common = {
            "planted_type": 0,
            "planted_shift": 3,
            "crossover": 0.45,
            "control_seed": 42,
        }
        unquantized = []
        for bins in ((1, 2), (1, 3), (2, 3)):
            result = exact_bsc_comparison(bank, bins=bins, **common)
            unquantized.append(
                result["correct_probability"]["proposed_bins"]
            )
        self.assertAlmostEqual(max(unquantized), min(unquantized), places=14)

        def accuracy(bins, origin):
            result = exact_bsc_comparison(
                bank,
                bins=bins,
                phase_bits=8,
                quantizer_origin_fraction=origin,
                **common,
            )
            self.assertFalse(result["frequency_diversity_established"])
            return result["correct_probability"]["proposed_bins"]

        self.assertGreater(
            accuracy((1, 2), 0.5),
            accuracy((2, 3), 0.5),
        )
        self.assertGreater(
            accuracy((2, 3), 0.75),
            accuracy((1, 2), 0.75),
        )


if __name__ == "__main__":
    unittest.main()
