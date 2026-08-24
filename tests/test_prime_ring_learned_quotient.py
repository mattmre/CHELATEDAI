import math
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import prime_ring_learned_quotient as learned_quotient
from prime_ring_learned_quotient import (
    ANCHORED_RAW_COEFFICIENTS,
    AUGMENTATION_OFFSETS,
    DEFAULT_MAX_ESTIMATED_BYTES,
    DEFAULT_MAX_WORK_UNITS,
    EVIDENCE_MODE,
    FROZEN_SEEDS,
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_SECONDS,
    HARD_MAX_SEED_COUNT,
    HARD_MAX_WORK_UNITS,
    HELDOUT_CLASS_COUNT,
    INVARIANT_BIAS,
    INVARIANT_QUOTIENT_COEFFICIENTS,
    INVARIANT_RAW_COEFFICIENTS,
    LIMITATIONS,
    MODEL_ANCHORED_CONTROL,
    MODEL_AUGMENTED_RAW,
    MODEL_MINIMAL_RANK_WITNESS,
    MODEL_QUOTIENT_ORACLE,
    MODEL_UNCONSTRAINED_RAW,
    MODULUS,
    MINIMAL_FULL_RANK_ROWS,
    QUOTIENT_CLASS_COUNT,
    SPLIT_UNSEEN_CLASS,
    SPLIT_UNSEEN_GAUGE,
    SUPPORTED_STATUS,
    TRAIN_CLASS_COUNT,
    UNSEEN_GAUGE_OFFSETS,
    UNSUPPORTED_STATUS,
    LearnedQuotientBudget,
    LearnedQuotientResourceError,
    LearnedQuotientValidationError,
    analyze_learned_quotient,
    apply_global_gauge,
    estimate_learned_quotient_resources,
    invariant_teacher,
    quotient_key,
)


class LearnedQuotientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze_learned_quotient()

    def test_partition_is_exact_and_resource_bounded(self):
        resource = self.result.resource_guard

        self.assertEqual(QUOTIENT_CLASS_COUNT, 343)
        self.assertEqual(TRAIN_CLASS_COUNT, 274)
        self.assertEqual(HELDOUT_CLASS_COUNT, 69)
        self.assertEqual(resource.seed_count, len(FROZEN_SEEDS))
        self.assertEqual(resource.unique_states_per_seed, MODULUS**4)
        self.assertEqual(
            resource.raw_training_rows_per_seed,
            TRAIN_CLASS_COUNT,
        )
        self.assertEqual(
            resource.augmented_training_rows_per_seed,
            TRAIN_CLASS_COUNT * len(AUGMENTATION_OFFSETS),
        )
        self.assertEqual(
            resource.unseen_gauge_rows_per_seed,
            TRAIN_CLASS_COUNT * len(UNSEEN_GAUGE_OFFSETS),
        )
        self.assertEqual(
            resource.unseen_class_rows_per_seed,
            HELDOUT_CLASS_COUNT * MODULUS,
        )
        self.assertLessEqual(
            resource.estimated_peak_bytes,
            DEFAULT_MAX_ESTIMATED_BYTES,
        )
        self.assertLessEqual(
            resource.estimated_work_units,
            DEFAULT_MAX_WORK_UNITS,
        )
        self.assertTrue(resource.one_seed_materialized_at_a_time)
        self.assertFalse(resource.measured_process_rss)

    def test_identifier_state_class_and_gauge_separation(self):
        for run in self.result.runs:
            self.assertTrue(run.train_evaluation_ids_disjoint)
            self.assertTrue(run.train_evaluation_states_disjoint)
            self.assertTrue(run.train_heldout_classes_disjoint)
            self.assertTrue(run.unseen_gauge_classes_match_train_classes)
            self.assertEqual(run.train_class_count, TRAIN_CLASS_COUNT)
            self.assertEqual(run.heldout_class_count, HELDOUT_CLASS_COUNT)
        self.assertTrue(self.result.all_identifier_and_class_separation_checks_pass)

    def test_unconstrained_raw_learner_recovers_invariant_rule(self):
        for run in self.result.runs:
            rule = run.unconstrained_raw_rule
            self.assertEqual(rule.coefficients, INVARIANT_RAW_COEFFICIENTS)
            self.assertEqual(rule.design_rank, 5)
            self.assertEqual(rule.gauge_coefficient_sum, 0)
            for split in (SPLIT_UNSEEN_GAUGE, SPLIT_UNSEEN_CLASS):
                metric = run.metric(MODEL_UNCONSTRAINED_RAW, split)
                self.assertEqual(metric.task_accuracy, 1.0)
                self.assertEqual(metric.orbit_consistency_rate, 1.0)
                self.assertEqual(metric.orbit_disagreement_rate, 0.0)
        self.assertTrue(self.result.unconstrained_raw_invariant_all_seeds)

    def test_augmented_and_quotient_controls_are_exact(self):
        for run in self.result.runs:
            self.assertEqual(
                run.augmented_raw_rule.coefficients,
                INVARIANT_RAW_COEFFICIENTS,
            )
            self.assertEqual(
                run.quotient_oracle_rule.coefficients,
                INVARIANT_QUOTIENT_COEFFICIENTS,
            )
            for model in (MODEL_AUGMENTED_RAW, MODEL_QUOTIENT_ORACLE):
                for split in (SPLIT_UNSEEN_GAUGE, SPLIT_UNSEEN_CLASS):
                    metric = run.metric(model, split)
                    self.assertEqual(metric.task_accuracy, 1.0)
                    self.assertEqual(metric.orbit_disagreement_rate, 0.0)
        self.assertTrue(self.result.augmented_raw_invariant_all_seeds)
        self.assertTrue(self.result.quotient_oracle_invariant_all_seeds)

    def test_anchored_learned_control_detects_every_gauge_pair(self):
        for run in self.result.runs:
            self.assertEqual(
                run.anchored_control_rule.coefficients,
                ANCHORED_RAW_COEFFICIENTS,
            )
            self.assertEqual(
                run.anchored_control_rule.gauge_coefficient_sum,
                1,
            )
            self.assertEqual(
                run.anchored_control_rule.training_rows,
                run.raw_training_rows,
            )
            self.assertNotEqual(
                run.anchored_control_rule.training_rows,
                run.augmented_training_rows,
            )
            for split in (SPLIT_UNSEEN_GAUGE, SPLIT_UNSEEN_CLASS):
                metric = run.metric(MODEL_ANCHORED_CONTROL, split)
                self.assertEqual(metric.task_accuracy, 1.0)
                self.assertEqual(metric.orbit_consistency_rate, 0.0)
                self.assertEqual(metric.orbit_disagreement_rate, 1.0)
        self.assertTrue(self.result.anchored_control_sensitive_all_seeds)

    def test_five_rows_are_the_exact_minimal_full_rank_solver_witness(self):
        self.assertEqual(MINIMAL_FULL_RANK_ROWS, 5)
        for run in self.result.runs:
            self.assertEqual(
                run.pre_witness_design_rank,
                MINIMAL_FULL_RANK_ROWS - 1,
            )
            rule = run.minimal_rank_witness_rule
            self.assertEqual(rule.coefficients, INVARIANT_RAW_COEFFICIENTS)
            self.assertEqual(rule.bias, INVARIANT_BIAS)
            self.assertEqual(rule.training_rows, MINIMAL_FULL_RANK_ROWS)
            self.assertEqual(rule.design_rank, MINIMAL_FULL_RANK_ROWS)
            self.assertEqual(rule.gauge_coefficient_sum, 0)
            for split in (SPLIT_UNSEEN_GAUGE, SPLIT_UNSEEN_CLASS):
                metric = run.metric(MODEL_MINIMAL_RANK_WITNESS, split)
                self.assertEqual(metric.task_accuracy, 1.0)
                self.assertEqual(metric.orbit_consistency_rate, 1.0)
                self.assertEqual(metric.orbit_disagreement_rate, 0.0)
        self.assertTrue(
            self.result.minimal_rank_solver_witness_all_seeds,
        )

    def test_global_gauge_preserves_invariant_teacher(self):
        state = (6, 1, 4, 0)
        expected_key = quotient_key(state)
        expected_target = invariant_teacher(state)

        for shift in range(MODULUS):
            shifted = apply_global_gauge(state, shift)
            self.assertEqual(quotient_key(shifted), expected_key)
            self.assertEqual(invariant_teacher(shifted), expected_target)

    def test_public_state_helpers_fail_closed_on_malformed_states(self):
        for state in ((0, 1, 2), (0, 1, 2, 7), (0, 1, 2, -1), "0123"):
            with self.subTest(state=state):
                with self.assertRaises(LearnedQuotientValidationError):
                    quotient_key(state)
                with self.assertRaises(LearnedQuotientValidationError):
                    invariant_teacher(state)

    def test_result_is_deterministic(self):
        first = analyze_learned_quotient(seeds=(7,))
        second = analyze_learned_quotient(seeds=(7,))

        self.assertEqual(first, second)

    def test_labels_are_explicitly_non_confirmatory(self):
        result = self.result

        self.assertEqual(
            EVIDENCE_MODE,
            "DETERMINISTIC_SYNTHETIC_EXACT_AFFINE_SOLVER_CONTROL",
        )
        self.assertEqual(result.hypothesis_status, SUPPORTED_STATUS)
        self.assertEqual(
            SUPPORTED_STATUS,
            "NARROW_EXACT_AFFINE_SOLVER_RECOVERY_NON_CONFIRMATORY",
        )
        self.assertTrue(result.all_publication_conditions_pass)
        self.assertTrue(result.non_confirmatory)
        self.assertFalse(result.promotion_eligible)
        self.assertFalse(result.novelty_claim)
        self.assertFalse(result.empirical_emergence_claim)
        self.assertFalse(result.arbitrary_observer_claim)
        self.assertFalse(result.intrinsic_dimension_claim)
        self.assertFalse(result.real_world_utility_claim)
        self.assertEqual(result.limitations, LIMITATIONS)
        self.assertTrue(any("modular-affine" in limitation for limitation in LIMITATIONS))
        self.assertTrue(any("Five full-rank raw rows" in limitation for limitation in LIMITATIONS))
        self.assertTrue(any("empirical emergence" in limitation for limitation in LIMITATIONS))
        self.assertTrue(any("not measured process RSS" in limitation for limitation in LIMITATIONS))

    def test_publication_gate_rejects_hostile_control_or_integrity_drift(self):
        baseline = self.result.runs[0]
        corrupted_metrics = tuple(
            (
                replace(metric, task_accuracy=0.0)
                if metric.model_name == MODEL_UNCONSTRAINED_RAW and metric.split == SPLIT_UNSEEN_GAUGE
                else metric
            )
            for metric in baseline.metrics
        )
        variants = (
            (
                "unconstrained_rule",
                replace(
                    baseline,
                    unconstrained_raw_rule=replace(
                        baseline.unconstrained_raw_rule,
                        bias=(baseline.unconstrained_raw_rule.bias + 1) % MODULUS,
                    ),
                ),
            ),
            (
                "augmented_rule",
                replace(
                    baseline,
                    augmented_raw_rule=replace(
                        baseline.augmented_raw_rule,
                        bias=(baseline.augmented_raw_rule.bias + 1) % MODULUS,
                    ),
                ),
            ),
            (
                "quotient_oracle",
                replace(
                    baseline,
                    quotient_oracle_rule=replace(
                        baseline.quotient_oracle_rule,
                        bias=(baseline.quotient_oracle_rule.bias + 1) % MODULUS,
                    ),
                ),
            ),
            (
                "anchored_negative",
                replace(
                    baseline,
                    anchored_control_rule=replace(
                        baseline.anchored_control_rule,
                        bias=(baseline.anchored_control_rule.bias + 1) % MODULUS,
                    ),
                ),
            ),
            (
                "minimal_rank_witness",
                replace(
                    baseline,
                    minimal_rank_witness_rule=replace(
                        baseline.minimal_rank_witness_rule,
                        design_rank=MINIMAL_FULL_RANK_ROWS - 1,
                    ),
                ),
            ),
            (
                "metric_drift",
                replace(
                    baseline,
                    metrics=corrupted_metrics,
                ),
            ),
            (
                "identifier_separation",
                replace(
                    baseline,
                    train_evaluation_ids_disjoint=False,
                ),
            ),
            (
                "state_separation",
                replace(
                    baseline,
                    train_evaluation_states_disjoint=False,
                ),
            ),
            (
                "class_separation",
                replace(
                    baseline,
                    train_heldout_classes_disjoint=False,
                ),
            ),
            (
                "gauge_class_coverage",
                replace(
                    baseline,
                    unseen_gauge_classes_match_train_classes=False,
                ),
            ),
        )

        for name, hostile_run in variants:
            with self.subTest(name=name):
                with patch.object(
                    learned_quotient,
                    "_run_seed",
                    return_value=hostile_run,
                ):
                    result = analyze_learned_quotient(seeds=(7,))
                self.assertFalse(result.all_publication_conditions_pass)
                self.assertEqual(
                    result.hypothesis_status,
                    UNSUPPORTED_STATUS,
                )

    def test_preflight_refuses_bytes_and_work_before_execution(self):
        with self.assertRaises(LearnedQuotientResourceError):
            estimate_learned_quotient_resources(
                seeds=(7,),
                budget=LearnedQuotientBudget(max_estimated_bytes=1),
            )
        with self.assertRaises(LearnedQuotientResourceError):
            estimate_learned_quotient_resources(
                seeds=(7,),
                budget=LearnedQuotientBudget(max_work_units=1),
            )

    def test_deadline_is_enforced(self):
        budget = LearnedQuotientBudget(max_seconds=1.0)
        with patch(
            "prime_ring_learned_quotient.time.monotonic",
            side_effect=(0.0, 2.0),
        ):
            with self.assertRaises(LearnedQuotientResourceError):
                analyze_learned_quotient(seeds=(7,), budget=budget)

    def test_budget_hard_caps_and_types_are_immutable(self):
        with self.assertRaises(LearnedQuotientValidationError):
            LearnedQuotientBudget(max_estimated_bytes=HARD_MAX_ESTIMATED_BYTES + 1)
        with self.assertRaises(LearnedQuotientValidationError):
            LearnedQuotientBudget(max_work_units=HARD_MAX_WORK_UNITS + 1)
        with self.assertRaises(LearnedQuotientValidationError):
            LearnedQuotientBudget(max_seconds=HARD_MAX_SECONDS + 0.01)
        with self.assertRaises(LearnedQuotientValidationError):
            LearnedQuotientBudget(max_work_units=True)
        with self.assertRaises(FrozenInstanceError):
            self.result.resource_guard.max_work_units = 1

    def test_seed_shape_and_membership_are_validated(self):
        with self.assertRaises(LearnedQuotientValidationError):
            analyze_learned_quotient(seeds="7")
        with self.assertRaises(LearnedQuotientValidationError):
            analyze_learned_quotient(seeds=())
        with self.assertRaises(LearnedQuotientValidationError):
            analyze_learned_quotient(seeds=(7, 7))
        with self.assertRaises(LearnedQuotientValidationError):
            analyze_learned_quotient(seeds=(99,))
        with self.assertRaises(LearnedQuotientValidationError):
            analyze_learned_quotient(seeds=tuple(range(HARD_MAX_SEED_COUNT + 1)))

    def test_non_finite_deadlines_are_rejected(self):
        for value in (0.0, -1.0, math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(LearnedQuotientValidationError):
                    LearnedQuotientBudget(max_seconds=value)


if __name__ == "__main__":
    unittest.main()
