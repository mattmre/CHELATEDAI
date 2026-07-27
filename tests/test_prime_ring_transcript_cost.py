from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import prime_ring_transcript_cost as transcript_cost
from prime_ring_transcript_cost import (
    CANDIDATE_STATUS,
    DEFAULT_MAX_ESTIMATED_BYTES,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_WORK_UNITS,
    GLOBAL_HYPOTHESIS_STATUS,
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_SECONDS,
    HARD_MAX_WORK_UNITS,
    LEARNER_NAMES,
    NO_SIGNAL_STATUS,
    NULL_PASS_STATUS,
    TranscriptCostBudget,
    TranscriptCostConfig,
    TranscriptCostResourceError,
    TranscriptCostValidationError,
    TranscriptTeacher,
    analyze_transcript_cost,
    generate_transcript_dataset,
    metric_by_name,
    preflight_transcript_cost,
    result_as_dict,
    validate_result_integrity,
)


class TranscriptCostScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze_transcript_cost()

    def test_default_campaign_covers_two_and_three_bit_families(self):
        signatures = {(cell.branch_bits, cell.family, cell.shift_mode) for cell in self.result.cells}
        self.assertEqual(
            signatures,
            {
                (2, "separable", "no_shift"),
                (2, "separable", "nuisance_remapping"),
                (2, "xor", "spurious_reversal"),
                (3, "separable", "no_shift"),
                (3, "separable", "nuisance_remapping"),
                (3, "parity", "spurious_reversal"),
            },
        )
        self.assertEqual(self.result.learner_names, LEARNER_NAMES)

    def test_generation_is_deterministic_and_identifier_disjoint(self):
        teacher = TranscriptTeacher(
            "parity",
            3,
            "spurious_reversal",
            104729,
        )
        first = generate_transcript_dataset(teacher)
        second = generate_transcript_dataset(teacher)

        self.assertEqual(first, second)
        self.assertTrue(first.identifiers_disjoint)
        self.assertNotEqual(first.train_digest, first.shifted_digest)
        self.assertFalse({row.example_id for row in first.train} & {row.example_id for row in first.shifted})

    def test_spurious_reversal_is_a_real_distribution_shift_witness(self):
        for cell in self.result.cells:
            if cell.shift_mode != "spurious_reversal":
                continue
            dataset = generate_transcript_dataset(
                TranscriptTeacher(
                    cell.family,
                    cell.branch_bits,
                    cell.shift_mode,
                    cell.seed,
                )
            )
            self.assertEqual(dataset.train_payload_alignment, 1.0)
            self.assertEqual(dataset.shifted_payload_alignment, 0.0)
            self.assertTrue(dataset.raw_rows_disjoint)
            self.assertTrue(dataset.distribution_shift_witness)
            self.assertTrue(cell.distribution_shift_witness)
        self.assertTrue(self.result.all_shift_cells_have_distribution_shift_witness)

    def test_flat_and_factorized_have_identical_padded_inference_envelopes(self):
        for cell in self.result.cells:
            for point in cell.learning_curve:
                factorized = metric_by_name(point.metrics, "factorized")
                flat = metric_by_name(point.metrics, "matched_flat")
                self.assertEqual(
                    factorized.charge.matching_signature,
                    flat.charge.matching_signature,
                )
                self.assertEqual(
                    {metric.charge.matching_signature for metric in point.metrics},
                    {factorized.charge.matching_signature},
                )
                self.assertTrue(point.all_common_envelopes_matched)
                self.assertFalse(factorized.charge.measured_resources)
                self.assertFalse(flat.charge.measured_resources)
        self.assertTrue(self.result.all_inference_envelopes_matched)

    def test_paid_transcript_is_perfect_and_explicitly_charged_side_information(self):
        for cell in self.result.cells:
            for point in cell.learning_curve:
                paid = metric_by_name(point.metrics, "paid_transcript")
                self.assertEqual(paid.route_accuracy, 1.0)
                self.assertEqual(paid.bit_accuracy, 1.0)
                self.assertEqual(
                    paid.charge.side_information_bits_per_example,
                    cell.branch_bits,
                )
                self.assertEqual(paid.charge.actual_parameter_bits, 0)
                self.assertTrue(paid.charge.padding_to_common_envelope)

    def test_null_cells_make_flat_and_factorized_tie_at_full_coverage(self):
        null_cells = [cell for cell in self.result.cells if cell.family == "separable"]
        self.assertEqual(len(null_cells), 4)
        for cell in null_cells:
            final = cell.learning_curve[-1]
            factorized = metric_by_name(final.metrics, "factorized")
            flat = metric_by_name(final.metrics, "matched_flat")
            causal_only = metric_by_name(final.metrics, "causal_only_flat")
            self.assertEqual(factorized.route_accuracy, 1.0)
            self.assertEqual(flat.route_accuracy, 1.0)
            self.assertEqual(causal_only.route_accuracy, 1.0)
            self.assertEqual(
                cell.final_factorized_minus_flat_route_accuracy,
                0.0,
            )
            self.assertEqual(
                cell.final_factorized_minus_causal_only_route_accuracy,
                0.0,
            )
            self.assertTrue(cell.null_sanity_pass)
            self.assertEqual(cell.status, NULL_PASS_STATUS)

    def test_true_no_shift_and_nuisance_remapping_are_distinct_controls(self):
        no_shift_cells = [cell for cell in self.result.cells if cell.shift_mode == "no_shift"]
        remapping_cells = [cell for cell in self.result.cells if cell.shift_mode == "nuisance_remapping"]
        self.assertEqual(len(no_shift_cells), 2)
        self.assertEqual(len(remapping_cells), 2)
        for cell in no_shift_cells:
            dataset = generate_transcript_dataset(
                TranscriptTeacher(
                    cell.family,
                    cell.branch_bits,
                    cell.shift_mode,
                    cell.seed,
                )
            )
            self.assertTrue(dataset.equal_environment_witness)
            self.assertFalse(dataset.nuisance_remapping_witness)
            self.assertEqual(dataset.train_content_digest, dataset.shifted_content_digest)
            self.assertEqual(dataset.payload_remapped_fraction, 0.0)
        for cell in remapping_cells:
            dataset = generate_transcript_dataset(
                TranscriptTeacher(
                    cell.family,
                    cell.branch_bits,
                    cell.shift_mode,
                    cell.seed,
                )
            )
            self.assertFalse(dataset.equal_environment_witness)
            self.assertTrue(dataset.nuisance_remapping_witness)
            self.assertNotEqual(dataset.train_content_digest, dataset.shifted_content_digest)
            self.assertGreater(dataset.payload_remapped_fraction, 0.0)
        self.assertTrue(self.result.all_no_shift_cells_have_equal_environment_witness)
        self.assertTrue(self.result.all_nuisance_remapping_cells_have_remapping_witness)

    def test_causal_only_flat_counterfactual_exposes_shortcut_selection(self):
        nonlinear = [cell for cell in self.result.cells if cell.family in ("xor", "parity")]
        self.assertEqual(len(nonlinear), 2)
        for cell in nonlinear:
            final = cell.learning_curve[-1]
            factorized = metric_by_name(final.metrics, "factorized")
            unrestricted_flat = metric_by_name(final.metrics, "matched_flat")
            causal_only_flat = metric_by_name(
                final.metrics,
                "causal_only_flat",
            )

            self.assertEqual(factorized.route_accuracy, 1.0)
            self.assertEqual(causal_only_flat.route_accuracy, 1.0)
            self.assertEqual(unrestricted_flat.route_accuracy, 0.0)
            self.assertEqual(
                causal_only_flat.selected_feature_subsets,
                factorized.selected_feature_subsets,
            )
            self.assertTrue(
                all(
                    all(index >= cell.branch_bits for index in subset)
                    for subset in causal_only_flat.selected_feature_subsets
                )
            )
            self.assertTrue(
                all(
                    all(index < cell.branch_bits for index in subset)
                    for subset in unrestricted_flat.selected_feature_subsets
                )
            )
            self.assertGreater(
                causal_only_flat.charge.training_search_work_units,
                factorized.charge.training_search_work_units,
            )
            self.assertEqual(
                cell.final_factorized_minus_causal_only_route_accuracy,
                0.0,
            )
            self.assertGreater(
                cell.final_causal_only_to_factorized_training_work_ratio,
                1.0,
            )
            self.assertTrue(
                cell.causal_only_counterfactual_matches_factorized,
            )
            self.assertTrue(
                cell.unrestricted_flat_shortcut_selection_witness,
            )
            wrong_group = metric_by_name(
                final.metrics,
                "wrong_group_factorized",
            )
            self.assertEqual(
                wrong_group.charge.actual_parameter_bits,
                factorized.charge.actual_parameter_bits,
            )
            self.assertEqual(
                wrong_group.charge.actual_inference_operations,
                factorized.charge.actual_inference_operations,
            )
            self.assertLess(wrong_group.route_accuracy, factorized.route_accuracy)
            self.assertTrue(cell.wrong_group_equal_complexity)
            self.assertTrue(cell.wrong_group_control_underperforms)
        self.assertTrue(self.result.all_candidate_cells_have_wrong_group_control)

    def test_nonlinear_shift_cells_show_only_a_nonconfirmatory_candidate(self):
        nonlinear = [cell for cell in self.result.cells if cell.family in ("xor", "parity")]
        self.assertEqual(len(nonlinear), 2)
        for cell in nonlinear:
            final = cell.learning_curve[-1]
            factorized = metric_by_name(final.metrics, "factorized")
            flat = metric_by_name(final.metrics, "matched_flat")
            payload = metric_by_name(final.metrics, "payload_only")
            random_control = metric_by_name(
                final.metrics,
                "deterministic_random",
            )
            self.assertEqual(factorized.route_accuracy, 1.0)
            self.assertEqual(flat.route_accuracy, 0.0)
            self.assertEqual(payload.route_accuracy, 0.0)
            self.assertGreater(
                factorized.route_accuracy,
                random_control.route_accuracy,
            )
            self.assertGreater(
                flat.charge.training_search_work_units,
                factorized.charge.training_search_work_units,
            )
            self.assertGreater(
                cell.final_flat_to_factorized_training_work_ratio,
                1.0,
            )
            self.assertGreater(
                cell.final_factorized_minus_flat_actual_parameter_bits,
                0,
            )
            self.assertFalse(cell.actual_compression_advantage)
            self.assertFalse(cell.actual_inference_operation_advantage)
            self.assertTrue(
                cell.factorized_reaches_target_before_unrestricted_flat,
            )
            self.assertTrue(cell.candidate_signal)
            self.assertFalse(cell.frozen_screen_kill)
            self.assertEqual(cell.status, CANDIDATE_STATUS)

        self.assertTrue(self.result.candidate_signal)
        self.assertTrue(self.result.all_nonlinear_candidate_cells_pass)
        self.assertTrue(self.result.all_separable_null_cells_pass)

    def test_result_explicitly_refuses_capacity_utility_novelty_and_promotion(self):
        self.assertEqual(
            self.result.hypothesis_status,
            GLOBAL_HYPOTHESIS_STATUS,
        )
        self.assertFalse(self.result.representational_capacity_claim)
        self.assertFalse(self.result.compression_advantage_claim)
        self.assertFalse(self.result.real_utility_claim)
        self.assertFalse(self.result.novelty_claim)
        self.assertFalse(self.result.promotion_eligible)
        self.assertGreaterEqual(len(self.result.limitations), 6)
        self.assertEqual(
            CANDIDATE_STATUS,
            "CANDIDATE_STRUCTURAL_PRIOR_TRAINING_SEARCH_SIGNAL_NON_CONFIRMATORY",
        )
        self.assertNotIn("INFERENCE_COST", CANDIDATE_STATUS)
        serialized = json.dumps(result_as_dict(self.result), sort_keys=True)
        self.assertIn("NON_CONFIRMATORY", serialized)
        self.assertIn("paid-transcript", serialized)
        self.assertIn("shortcut-selection", serialized)
        self.assertIn("run_contract_digest", serialized)
        self.assertTrue(validate_result_integrity(self.result))

        forged = replace(self.result, run_contract_digest="0" * 64)
        with self.assertRaises(TranscriptCostValidationError):
            validate_result_integrity(forged)

    def test_resource_preflight_is_componentized_and_under_default_caps(self):
        guard = self.result.resource_guard
        self.assertEqual(
            guard.estimated_peak_bytes,
            sum(value for _name, value in guard.component_bytes),
        )
        self.assertEqual(
            guard.estimated_work_units,
            sum(value for _name, value in guard.component_work),
        )
        self.assertLessEqual(
            guard.estimated_peak_bytes,
            DEFAULT_MAX_ESTIMATED_BYTES,
        )
        self.assertLessEqual(
            guard.estimated_work_units,
            DEFAULT_MAX_WORK_UNITS,
        )
        self.assertEqual(guard.max_seconds, DEFAULT_MAX_SECONDS)
        self.assertFalse(guard.measured_process_peak)

    def test_preflight_refuses_tiny_budgets_before_analysis(self):
        config = TranscriptCostConfig()
        normal = preflight_transcript_cost(config)
        with self.assertRaises(TranscriptCostResourceError):
            preflight_transcript_cost(
                config,
                TranscriptCostBudget(
                    max_estimated_bytes=normal.estimated_peak_bytes - 1,
                    max_work_units=DEFAULT_MAX_WORK_UNITS,
                    max_seconds=DEFAULT_MAX_SECONDS,
                ),
            )
        with self.assertRaises(TranscriptCostResourceError):
            preflight_transcript_cost(
                config,
                TranscriptCostBudget(
                    max_estimated_bytes=DEFAULT_MAX_ESTIMATED_BYTES,
                    max_work_units=normal.estimated_work_units - 1,
                    max_seconds=DEFAULT_MAX_SECONDS,
                ),
            )

    def test_hard_resource_ceilings_cannot_be_raised(self):
        with self.assertRaises(TranscriptCostValidationError):
            TranscriptCostBudget(max_estimated_bytes=HARD_MAX_ESTIMATED_BYTES + 1)
        with self.assertRaises(TranscriptCostValidationError):
            TranscriptCostBudget(max_work_units=HARD_MAX_WORK_UNITS + 1)
        with self.assertRaises(TranscriptCostValidationError):
            TranscriptCostBudget(max_seconds=HARD_MAX_SECONDS + 0.001)

    def test_validation_rejects_malformed_or_oversized_teachers(self):
        with self.assertRaises(TranscriptCostValidationError):
            TranscriptTeacher("unknown", 2, "no_shift", 1)
        with self.assertRaises(TranscriptCostValidationError):
            TranscriptTeacher("xor", 4, "no_shift", 1)
        with self.assertRaises(TranscriptCostValidationError):
            TranscriptTeacher("xor", 2, "unknown", 1)
        with self.assertRaises(TranscriptCostValidationError):
            TranscriptCostConfig(train_sizes=(8, 4))

    def test_expired_deadline_fails_closed(self):
        with self.assertRaises(TranscriptCostResourceError):
            # A valid but microscopic deadline must not silently publish a
            # partial result.  Monotonic time has already advanced by call time.
            analyze_transcript_cost(budget=TranscriptCostBudget(max_seconds=1e-12))

    def test_repeated_analysis_is_exactly_deterministic(self):
        repeated = analyze_transcript_cost()
        self.assertEqual(self.result, repeated)
        self.assertTrue(self.result.all_identifiers_disjoint)

    def test_candidate_status_is_gated_by_integrity_flags(self):
        original = transcript_cost._analyze_cell

        def with_identifier_overlap(*args, **kwargs):
            return replace(
                original(*args, **kwargs),
                identifiers_disjoint=False,
            )

        with patch.object(
            transcript_cost,
            "_analyze_cell",
            side_effect=with_identifier_overlap,
        ):
            result = analyze_transcript_cost()

        self.assertFalse(result.all_identifiers_disjoint)
        self.assertFalse(result.candidate_signal)
        self.assertEqual(result.frozen_screen_status, NO_SIGNAL_STATUS)


if __name__ == "__main__":
    unittest.main()
