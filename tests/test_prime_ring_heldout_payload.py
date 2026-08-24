from __future__ import annotations

import unittest
from unittest.mock import patch

from prime_ring_heldout_payload import (
    CANDIDATE_COUNT,
    DEFAULT_MAX_ESTIMATED_BYTES,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_WORK_UNITS,
    FROZEN_SEEDS,
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_PREDICATE_BITS,
    HARD_MAX_SECONDS,
    HARD_MAX_WORK_UNITS,
    HELDOUT_COUNT,
    HELDOUT_FALSE_UNLOCK_COUNT,
    HELDOUT_TRUE_UNLOCK_COUNT,
    KILLED_STATUS,
    LEAF_COUNT,
    MODULI,
    PREDICATE_BITS,
    ROUTER_NAMES,
    TRAIN_COUNT,
    TRAIN_FALSE_UNLOCK_COUNT,
    TRAIN_TRUE_UNLOCK_COUNT,
    TYPE_COUNT,
    WAYPOINT_COUNT,
    HeldoutPayloadBudget,
    HeldoutPayloadResourceError,
    HeldoutPayloadValidationError,
    analyze_heldout_payload,
    fit_router,
    generate_dataset,
    metric_by_name,
    predict_conditional,
    predict_paid_transcript_flat,
    xor_teacher_non_linearly_separable,
)


class HeldoutNonseparablePayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze_heldout_payload()

    def test_frozen_dataset_sizes_balance_and_identifier_separation(self):
        split = generate_dataset(7, FROZEN_SEEDS[0])
        train_ids = {example.example_id for example in split.train}
        heldout_ids = {example.example_id for example in split.heldout}

        self.assertEqual(len(split.train), TRAIN_COUNT)
        self.assertEqual(len(split.heldout), HELDOUT_COUNT)
        self.assertFalse(train_ids & heldout_ids)
        self.assertEqual(
            sum(example.should_unlock for example in split.train),
            TRAIN_TRUE_UNLOCK_COUNT,
        )
        self.assertEqual(
            sum(not example.should_unlock for example in split.train),
            TRAIN_FALSE_UNLOCK_COUNT,
        )
        self.assertEqual(
            sum(example.should_unlock for example in split.heldout),
            HELDOUT_TRUE_UNLOCK_COUNT,
        )
        self.assertEqual(
            sum(not example.should_unlock for example in split.heldout),
            HELDOUT_FALSE_UNLOCK_COUNT,
        )

    def test_every_type_waypoint_is_present_in_both_positive_splits(self):
        split = generate_dataset(11, FROZEN_SEEDS[1])
        expected = set(range(CANDIDATE_COUNT))
        train_targets = {
            example.target_candidate
            for example in split.train
            if example.should_unlock
        }
        heldout_targets = {
            example.target_candidate
            for example in split.heldout
            if example.should_unlock
        }

        self.assertEqual(TYPE_COUNT, 4)
        self.assertEqual(WAYPOINT_COUNT, 8)
        self.assertEqual(train_targets, expected)
        self.assertEqual(heldout_targets, expected)

    def test_generation_and_analysis_are_deterministic(self):
        first = generate_dataset(7, FROZEN_SEEDS[2])
        second = generate_dataset(7, FROZEN_SEEDS[2])
        repeated = analyze_heldout_payload()

        self.assertEqual(first, second)
        self.assertEqual(self.result, repeated)

    def test_teacher_is_xor_and_not_additively_linearly_separable(self):
        self.assertTrue(xor_teacher_non_linearly_separable())
        for run in self.result.runs:
            self.assertTrue(run.xor_teacher_non_linearly_separable)
            self.assertEqual(run.gate_training_errors, 0)
            self.assertGreater(run.linear_training_errors, 0)

    def test_fit_records_training_ids_only(self):
        split = generate_dataset(7, FROZEN_SEEDS[0])
        model = fit_router(split.train, split.modulus, split.seed)

        self.assertEqual(
            set(model.fit_example_ids),
            {example.example_id for example in split.train},
        )
        self.assertFalse(
            set(model.fit_example_ids)
            & {example.example_id for example in split.heldout}
        )
        self.assertEqual(len(model.candidates), CANDIDATE_COUNT)
        self.assertEqual(len(model.flat_leaves), LEAF_COUNT)
        self.assertEqual(len(model.gate_tables), PREDICATE_BITS)

    def test_conditional_and_paid_transcript_flat_are_exact_heldout(self):
        for modulus in MODULI:
            for seed in FROZEN_SEEDS:
                split = generate_dataset(modulus, seed)
                model = fit_router(
                    split.train,
                    split.modulus,
                    split.seed,
                )
                for example in split.heldout:
                    conditional = predict_conditional(example, model)
                    flat = predict_paid_transcript_flat(
                        example,
                        model,
                        conditional.branch_transcript,
                    )
                    self.assertEqual(conditional, flat)

        self.assertEqual(
            self.result.total_conditional_flat_exact_cases,
            len(MODULI) * len(FROZEN_SEEDS) * HELDOUT_COUNT,
        )
        self.assertTrue(
            self.result.conditional_flat_exact_on_all_heldout
        )
        self.assertTrue(self.result.extra_capacity_kill_criterion_met)
        self.assertEqual(self.result.hypothesis_status, KILLED_STATUS)

    def test_all_controls_have_identical_preregistered_charges(self):
        for run in self.result.runs:
            signatures = {
                charge.matching_signature
                for charge in run.control_charges
            }
            self.assertEqual(len(signatures), 1)
            self.assertEqual(
                tuple(charge.name for charge in run.control_charges),
                ROUTER_NAMES,
            )
            for charge in run.control_charges:
                self.assertEqual(charge.candidate_count, 32)
                self.assertEqual(charge.candidate_scores_per_query, 32)
                self.assertEqual(charge.branch_bits, 2)
                self.assertEqual(charge.leaf_count, 4)
                self.assertGreater(charge.charged_storage_bytes, 0)
                self.assertGreater(
                    charge.charged_operations_per_query,
                    0,
                )
                self.assertEqual(
                    charge.false_unlock_opportunities,
                    HELDOUT_FALSE_UNLOCK_COUNT,
                )
                self.assertEqual(charge.unlocks_allowed_per_query, 1)
                self.assertFalse(charge.measured_resources)
                self.assertTrue(charge.padding_to_common_envelope)
        self.assertTrue(self.result.all_controls_matched)

    def test_conditional_recovers_payload_only_misses_without_extra_capacity(self):
        conditional = metric_by_name(
            self.result.aggregate_metrics,
            "conditional",
        )
        flat = metric_by_name(
            self.result.aggregate_metrics,
            "paid_transcript_flat",
        )
        payload = metric_by_name(
            self.result.aggregate_metrics,
            "payload_only",
        )
        direct = metric_by_name(
            self.result.aggregate_metrics,
            "direct_metadata",
        )

        self.assertEqual(conditional, flat.__class__(
            router_name="conditional",
            true_unlock_opportunities=flat.true_unlock_opportunities,
            correct_true_unlocks=flat.correct_true_unlocks,
            recall=flat.recall,
            payload_only_misses=flat.payload_only_misses,
            corrections_of_payload_only=flat.corrections_of_payload_only,
            correction_rate=flat.correction_rate,
            false_unlock_opportunities=flat.false_unlock_opportunities,
            false_unlock_events=flat.false_unlock_events,
            false_unlock_rate=flat.false_unlock_rate,
            false_locks=flat.false_locks,
        ))
        self.assertEqual(conditional.recall, 1.0)
        self.assertEqual(conditional.correction_rate, 1.0)
        self.assertEqual(payload.recall, 0.25)
        self.assertLess(direct.recall, conditional.recall)
        self.assertGreater(direct.recall, payload.recall)

    def test_false_unlock_opportunities_are_real_heldout_negatives(self):
        for metric in self.result.aggregate_metrics:
            self.assertEqual(
                metric.false_unlock_opportunities,
                len(MODULI)
                * len(FROZEN_SEEDS)
                * HELDOUT_FALSE_UNLOCK_COUNT,
            )
            self.assertEqual(metric.false_unlock_events, 0)
            self.assertEqual(metric.false_unlock_rate, 0.0)
            self.assertEqual(metric.false_locks, 0)

    def test_every_frozen_run_has_requested_counts_and_clean_fit_threshold(self):
        self.assertEqual(
            {(run.modulus, run.seed) for run in self.result.runs},
            {
                (modulus, seed)
                for modulus in MODULI
                for seed in FROZEN_SEEDS
            },
        )
        for run in self.result.runs:
            self.assertEqual(run.train_count, 64)
            self.assertEqual(run.heldout_count, 128)
            self.assertEqual(run.train_true_unlock_count, 32)
            self.assertEqual(run.train_false_unlock_count, 32)
            self.assertEqual(run.heldout_true_unlock_count, 96)
            self.assertEqual(run.heldout_false_unlock_count, 32)
            self.assertTrue(run.identifiers_disjoint)
            self.assertEqual(run.threshold_training_false_unlocks, 0)
            self.assertEqual(run.threshold_training_false_locks, 0)
            self.assertEqual(run.conditional_flat_exact_cases, 128)
        self.assertTrue(self.result.all_split_identifiers_disjoint)
        self.assertFalse(self.result.distribution_shift_claim)
        self.assertFalse(self.result.class_separation_claim)

    def test_resource_preflight_is_below_every_hard_cap(self):
        resource = self.result.resource_guard

        self.assertEqual(resource.run_count, 6)
        self.assertEqual(resource.total_train_examples, 384)
        self.assertEqual(resource.total_heldout_examples, 768)
        self.assertEqual(resource.candidate_count, 32)
        self.assertEqual(resource.predicate_bits, 2)
        self.assertEqual(resource.leaf_count, 4)
        self.assertEqual(
            resource.estimated_peak_bytes,
            sum(value for _name, value in resource.component_bytes),
        )
        self.assertEqual(
            resource.estimated_work_units,
            sum(value for _name, value in resource.component_work),
        )
        self.assertLessEqual(
            resource.estimated_peak_bytes,
            DEFAULT_MAX_ESTIMATED_BYTES,
        )
        self.assertLessEqual(
            resource.estimated_peak_bytes,
            HARD_MAX_ESTIMATED_BYTES,
        )
        self.assertLessEqual(
            resource.estimated_work_units,
            DEFAULT_MAX_WORK_UNITS,
        )
        self.assertLessEqual(
            resource.estimated_work_units,
            HARD_MAX_WORK_UNITS,
        )
        self.assertEqual(resource.max_seconds, DEFAULT_MAX_SECONDS)
        self.assertTrue(resource.one_run_materialized_at_a_time)
        self.assertFalse(resource.measured_process_peak)

    def test_resource_refusal_precedes_generation_and_fit(self):
        for budget in (
            HeldoutPayloadBudget(max_estimated_bytes=1),
            HeldoutPayloadBudget(max_work_units=1),
            HeldoutPayloadBudget(max_predicate_bits=1),
        ):
            with self.subTest(budget=budget):
                with (
                    patch(
                        "prime_ring_heldout_payload.generate_dataset"
                    ) as generator,
                    patch(
                        "prime_ring_heldout_payload.fit_router"
                    ) as fitter,
                    self.assertRaises(HeldoutPayloadResourceError),
                ):
                    analyze_heldout_payload(budget=budget)
                generator.assert_not_called()
                fitter.assert_not_called()

    def test_deadline_refusal_precedes_generation_and_fit(self):
        with (
            patch(
                "prime_ring_heldout_payload.time.monotonic",
                side_effect=(0.0, DEFAULT_MAX_SECONDS + 1.0),
            ),
            patch(
                "prime_ring_heldout_payload.generate_dataset"
            ) as generator,
            patch("prime_ring_heldout_payload.fit_router") as fitter,
            self.assertRaises(HeldoutPayloadResourceError),
        ):
            analyze_heldout_payload()
        generator.assert_not_called()
        fitter.assert_not_called()

    def test_budget_numeric_subclasses_are_canonicalized(self):
        class HostileInt(int):
            def __gt__(self, _other):
                return False

            def __mul__(self, _other):
                return 0

        class HostileFloat(float):
            def __gt__(self, _other):
                return False

            def __add__(self, _other):
                return 0.0

        budget = HeldoutPayloadBudget(
            max_estimated_bytes=HostileInt(
                DEFAULT_MAX_ESTIMATED_BYTES
            ),
            max_work_units=HostileInt(DEFAULT_MAX_WORK_UNITS),
            max_seconds=HostileFloat(DEFAULT_MAX_SECONDS),
            max_predicate_bits=HostileInt(HARD_MAX_PREDICATE_BITS),
        )

        self.assertIs(type(budget.max_estimated_bytes), int)
        self.assertIs(type(budget.max_work_units), int)
        self.assertIs(type(budget.max_seconds), float)
        self.assertIs(type(budget.max_predicate_bits), int)
        self.assertTrue(
            analyze_heldout_payload(
                budget=budget
            ).extra_capacity_kill_criterion_met
        )

    def test_frozen_inputs_and_hard_caps_fail_closed(self):
        with self.assertRaises(HeldoutPayloadValidationError):
            generate_dataset(13, FROZEN_SEEDS[0])
        with self.assertRaises(HeldoutPayloadValidationError):
            generate_dataset(7, 1)
        with self.assertRaises(HeldoutPayloadValidationError):
            predict_paid_transcript_flat(
                generate_dataset(7, FROZEN_SEEDS[0]).heldout[0],
                fit_router(
                    generate_dataset(7, FROZEN_SEEDS[0]).train,
                    7,
                    FROZEN_SEEDS[0],
                ),
                4,
            )

        for field, value in (
            ("max_estimated_bytes", HARD_MAX_ESTIMATED_BYTES + 1),
            ("max_work_units", HARD_MAX_WORK_UNITS + 1),
            ("max_seconds", HARD_MAX_SECONDS + 0.01),
            ("max_predicate_bits", HARD_MAX_PREDICATE_BITS + 1),
        ):
            with self.subTest(field=field):
                with self.assertRaises(
                    HeldoutPayloadValidationError
                ):
                    HeldoutPayloadBudget(**{field: value})

    def test_result_is_non_promotional_and_scope_bounded(self):
        self.assertEqual(
            self.result.evidence_mode,
            "DETERMINISTIC_HELDOUT_SYNTHETIC_FALSIFICATION",
        )
        self.assertFalse(self.result.promotion_eligible)
        self.assertFalse(self.result.novelty_claim)
        self.assertFalse(self.result.production_claim)
        self.assertFalse(self.result.training_savings_claim)
        self.assertFalse(self.result.runtime_benefit_claim)
        self.assertTrue(
            any("p in {7,11}" in item for item in self.result.limitations)
        )
        self.assertTrue(
            any(
                "payload-vector length" in item
                for item in self.result.limitations
            )
        )
        self.assertTrue(
            any(
                "granted the conditional router's exact" in item
                for item in self.result.limitations
            )
        )
        self.assertTrue(
            any(
                "modeled rather than measured" in item
                for item in self.result.limitations
            )
        )
        self.assertTrue(
            any(
                "No result establishes novelty" in item
                for item in self.result.limitations
            )
        )


if __name__ == "__main__":
    unittest.main()
