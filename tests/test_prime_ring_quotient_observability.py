import math
import time
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from prime_ring_quotient_observability import (
    DEFAULT_MAX_ESTIMATED_BYTES,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_WORK_UNITS,
    GAUGE_ORBIT_SIZE,
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_SECONDS,
    HARD_MAX_WORK_UNITS,
    HYPOTHESIS_STATUS,
    LIMITATIONS,
    MODULUS,
    QUOTIENT_CLASS_COUNT,
    QUOTIENT_DIMENSION,
    STATE_COUNT,
    QuotientObservation,
    QuotientObservabilityBudget,
    QuotientObservabilityResourceError,
    QuotientObservabilityValidationError,
    analyze_quotient_observability,
    apply_global_gauge,
    canonicalize_state,
    estimate_quotient_resources,
    finite_difference_matrix,
    observe_anchored,
    observe_quotient,
    quotient_difference,
    quotient_key,
)


class QuotientObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze_quotient_observability()

    def test_exact_state_and_quotient_class_counts(self):
        result = self.result

        self.assertEqual(STATE_COUNT, 2_401)
        self.assertEqual(QUOTIENT_CLASS_COUNT, 343)
        self.assertEqual(result.total_states, 2_401)
        self.assertEqual(result.quotient_class_count, 343)
        self.assertEqual(result.expected_quotient_class_count, 343)
        self.assertEqual(result.class_size_min, 7)
        self.assertEqual(result.class_size_max, 7)
        self.assertEqual(result.expected_class_size, GAUGE_ORBIT_SIZE)
        self.assertTrue(result.all_classes_have_exact_gauge_orbit_size)

    def test_canonicalization_is_exact_idempotent_and_gauge_invariant(self):
        state = (6, 1, 4, 0)
        expected = (0, 2, 5, 1)

        self.assertEqual(quotient_key(state), expected[1:])
        self.assertEqual(canonicalize_state(state), expected)
        self.assertEqual(canonicalize_state(expected), expected)
        for shift in range(MODULUS):
            shifted = apply_global_gauge(state, shift)
            self.assertEqual(quotient_key(shifted), expected[1:])
            self.assertEqual(canonicalize_state(shifted), expected)

    def test_declared_observer_exposes_all_required_behavior(self):
        observation = observe_quotient((6, 1, 4, 0))

        self.assertEqual(observation.routes, (2, 5, 1))
        self.assertEqual(observation.margins, (1, 1, 1))
        self.assertEqual(observation.tie_counts, (1, 1, 1))
        self.assertEqual(observation.tied_routes, ((2,), (5,), (1,)))
        self.assertEqual(observation.payload_reads, (1_002, 1_012, 1_015))
        self.assertEqual(len(observation.scores), QUOTIENT_DIMENSION)
        self.assertTrue(all(len(row) == MODULUS for row in observation.scores))
        for stage, route in enumerate(observation.routes):
            self.assertEqual(observation.scores[stage][route], 0)
            self.assertEqual(max(observation.scores[stage]), 0)

    def test_all_declared_behavior_is_constant_on_each_gauge_orbit(self):
        for key in ((0, 0, 0), (1, 3, 6), (6, 6, 2)):
            canonical = (0,) + key
            expected = observe_quotient(canonical)
            for shift in range(MODULUS):
                self.assertEqual(
                    observe_quotient(apply_global_gauge(canonical, shift)),
                    expected,
                )
        self.assertTrue(
            self.result.invariant_observer_constant_within_every_class
        )
        self.assertEqual(
            self.result.distinct_invariant_observations,
            QUOTIENT_CLASS_COUNT,
        )
        self.assertTrue(
            self.result.invariant_observer_separates_all_quotient_classes
        )

    def test_anchored_negative_control_detects_global_gauge(self):
        canonical = (0, 1, 3, 6)

        self.assertEqual(
            observe_anchored(canonical),
            observe_quotient(canonical),
        )
        for shift in range(1, MODULUS):
            shifted = apply_global_gauge(canonical, shift)
            self.assertEqual(
                observe_quotient(shifted),
                observe_quotient(canonical),
            )
            self.assertNotEqual(
                observe_anchored(shifted),
                observe_anchored(canonical),
            )
        self.assertEqual(
            self.result.anchored_control_classes_checked,
            QUOTIENT_CLASS_COUNT,
        )
        self.assertEqual(
            self.result.anchored_control_gauge_changes_checked,
            QUOTIENT_CLASS_COUNT * (MODULUS - 1),
        )
        self.assertEqual(
            self.result.anchored_control_gauge_changes_detected,
            QUOTIENT_CLASS_COUNT * (MODULUS - 1),
        )
        self.assertTrue(
            self.result.anchored_control_sensitive_in_every_class
        )

    def test_every_unit_quotient_perturbation_changes_visible_behavior(self):
        result = self.result

        self.assertEqual(
            result.quotient_unit_perturbations_checked,
            QUOTIENT_CLASS_COUNT * QUOTIENT_DIMENSION,
        )
        self.assertTrue(result.quotient_unit_perturbations_all_observable)

    def test_finite_difference_gauge_null_and_rank_are_exact(self):
        audit = self.result.finite_difference
        expected = (
            (6, 1, 0, 0),
            (6, 0, 1, 0),
            (6, 0, 0, 1),
        )

        self.assertEqual(audit.jacobian_rows, expected)
        self.assertEqual(audit.rank, 3)
        self.assertEqual(audit.kernel_dimension, 1)
        self.assertEqual(audit.null_direction_count, 7)
        self.assertEqual(audit.gauge_direction, (1, 1, 1, 1))
        self.assertEqual(audit.gauge_difference, (0, 0, 0))
        self.assertTrue(audit.gauge_direction_is_null)
        self.assertTrue(audit.kernel_equals_global_gauge_span)
        self.assertTrue(audit.matrix_constant_over_all_states)
        self.assertEqual(audit.states_checked, STATE_COUNT)

    def test_finite_difference_wraparound_remains_modularly_linear(self):
        expected = (
            (6, 1, 0, 0),
            (6, 0, 1, 0),
            (6, 0, 0, 1),
        )

        self.assertEqual(finite_difference_matrix((6, 6, 6, 6)), expected)
        self.assertEqual(
            quotient_difference((6, 0, 1, 2), (1, 1, 1, 1)),
            (0, 0, 0),
        )
        self.assertEqual(
            quotient_difference((6, 0, 1, 2), (0, 1, 0, 0)),
            (1, 0, 0),
        )

    def test_resource_preflight_is_conservative_streaming_and_bounded(self):
        resource = estimate_quotient_resources()

        self.assertEqual(resource.states, STATE_COUNT)
        self.assertEqual(resource.quotient_classes, QUOTIENT_CLASS_COUNT)
        self.assertEqual(resource.expected_orbit_size, GAUGE_ORBIT_SIZE)
        self.assertEqual(
            resource.estimated_peak_bytes,
            sum(value for _name, value in resource.component_bytes),
        )
        self.assertEqual(resource.estimated_peak_bytes, 1_310_720)
        self.assertEqual(resource.estimated_work_units, 509_355)
        self.assertLessEqual(
            resource.estimated_peak_bytes,
            DEFAULT_MAX_ESTIMATED_BYTES,
        )
        self.assertLessEqual(
            resource.estimated_work_units,
            DEFAULT_MAX_WORK_UNITS,
        )
        self.assertEqual(resource.max_seconds, DEFAULT_MAX_SECONDS)
        self.assertTrue(resource.states_streamed_not_materialized)
        self.assertTrue(resource.class_representatives_materialized)
        self.assertFalse(resource.caller_owned_inputs_included)
        self.assertFalse(resource.measured_process_peak)

    def test_preflight_refuses_bytes_and_work_before_state_enumeration(self):
        with (
            patch(
                "prime_ring_quotient_observability._iter_states"
            ) as iterator,
            self.assertRaises(QuotientObservabilityResourceError),
        ):
            analyze_quotient_observability(
                budget=QuotientObservabilityBudget(max_estimated_bytes=1)
            )
        iterator.assert_not_called()

        with (
            patch(
                "prime_ring_quotient_observability._iter_states"
            ) as iterator,
            self.assertRaises(QuotientObservabilityResourceError),
        ):
            analyze_quotient_observability(
                budget=QuotientObservabilityBudget(max_work_units=1)
            )
        iterator.assert_not_called()

    def test_deadline_fails_before_enumeration(self):
        with (
            patch(
                "prime_ring_quotient_observability.time.monotonic",
                side_effect=(0.0, DEFAULT_MAX_SECONDS + 1.0),
            ),
            patch(
                "prime_ring_quotient_observability._iter_states"
            ) as iterator,
            self.assertRaises(QuotientObservabilityResourceError),
        ):
            analyze_quotient_observability()
        iterator.assert_not_called()

    def test_deadline_covers_enumeration(self):
        original = range(MODULUS)

        class SlowOnceIterable:
            def __iter__(self):
                time.sleep(0.01)
                return iter(original)

        with (
            patch(
                "prime_ring_quotient_observability._iter_states",
                return_value=(
                    (a, b, c, d)
                    for a in SlowOnceIterable()
                    for b in original
                    for c in original
                    for d in original
                ),
            ),
            self.assertRaises(QuotientObservabilityResourceError),
        ):
            analyze_quotient_observability(
                budget=QuotientObservabilityBudget(max_seconds=0.001)
            )

    def test_budget_numeric_subclasses_are_canonicalized(self):
        class HostileInt(int):
            def __gt__(self, other):
                return False

            def __lt__(self, other):
                return False

        class HostileFloat(float):
            def __gt__(self, other):
                return False

        budget = QuotientObservabilityBudget(
            max_estimated_bytes=HostileInt(DEFAULT_MAX_ESTIMATED_BYTES),
            max_work_units=HostileInt(DEFAULT_MAX_WORK_UNITS),
            max_seconds=HostileFloat(DEFAULT_MAX_SECONDS),
        )

        self.assertIs(type(budget.max_estimated_bytes), int)
        self.assertIs(type(budget.max_work_units), int)
        self.assertIs(type(budget.max_seconds), float)
        self.assertEqual(
            estimate_quotient_resources(budget).max_work_units,
            DEFAULT_MAX_WORK_UNITS,
        )

    def test_hard_caps_and_nonfinite_limits_fail_closed(self):
        for field, value in (
            ("max_estimated_bytes", HARD_MAX_ESTIMATED_BYTES + 1),
            ("max_work_units", HARD_MAX_WORK_UNITS + 1),
            ("max_seconds", HARD_MAX_SECONDS + 0.01),
            ("max_seconds", math.nan),
            ("max_seconds", math.inf),
            ("max_work_units", True),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(QuotientObservabilityValidationError):
                    QuotientObservabilityBudget(**{field: value})

    def test_malformed_states_directions_and_budget_fail_closed(self):
        malformed = (
            None,
            "0123",
            (0, 1, 2),
            (0, 1, 2, 7),
            (0, 1, 2, -1),
            (0, 1, 2, True),
            (0, 1, 2, 1.0),
        )
        for value in malformed:
            with self.subTest(value=value):
                with self.assertRaises(
                    QuotientObservabilityValidationError
                ):
                    quotient_key(value)

        with self.assertRaises(QuotientObservabilityValidationError):
            apply_global_gauge((0, 1, 2, 3), 7)
        with self.assertRaises(QuotientObservabilityValidationError):
            quotient_difference((0, 1, 2, 3), (0, 0, 0))
        with self.assertRaises(QuotientObservabilityValidationError):
            analyze_quotient_observability(budget=object())

    def test_observation_direct_construction_validates_derived_fields(self):
        valid = observe_quotient((0, 1, 2, 3))
        with self.assertRaises(QuotientObservabilityValidationError):
            QuotientObservation(
                scores=valid.scores,
                margins=(2, 1, 1),
                tie_counts=valid.tie_counts,
                tied_routes=valid.tied_routes,
                routes=valid.routes,
                payload_reads=valid.payload_reads,
            )
        with self.assertRaises(QuotientObservabilityValidationError):
            QuotientObservation(
                scores=valid.scores,
                margins=valid.margins,
                tie_counts=valid.tie_counts,
                tied_routes=((2,),) + valid.tied_routes[1:],
                routes=valid.routes,
                payload_reads=valid.payload_reads,
            )
        with self.assertRaises(QuotientObservabilityValidationError):
            QuotientObservation(
                scores=None,
                margins=valid.margins,
                tie_counts=valid.tie_counts,
                tied_routes=valid.tied_routes,
                routes=valid.routes,
                payload_reads=valid.payload_reads,
            )
        with self.assertRaises(QuotientObservabilityValidationError):
            QuotientObservation(
                scores=valid.scores,
                margins=valid.margins,
                tie_counts=valid.tie_counts,
                tied_routes=valid.tied_routes,
                routes=valid.routes,
                payload_reads=(0, 0, 0),
            )

    def test_nonpromotional_boundaries_are_explicit_and_immutable(self):
        result = self.result

        self.assertEqual(
            result.hypothesis_status,
            HYPOTHESIS_STATUS,
        )
        self.assertFalse(result.promotion_eligible)
        self.assertFalse(result.universal_dimension_claim)
        self.assertFalse(result.physical_claim)
        self.assertFalse(result.novelty_claim)
        self.assertFalse(result.ml_utility_claim)
        self.assertEqual(result.limitations, LIMITATIONS)
        self.assertTrue(
            any("intentionally constructed" in item for item in LIMITATIONS)
        )
        self.assertTrue(
            any("not an estimate" in item for item in LIMITATIONS)
        )
        with self.assertRaises(FrozenInstanceError):
            result.novelty_claim = True


if __name__ == "__main__":
    unittest.main()
