from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import prime_ring_conditional_replacements as replacements
from prime_ring_conditional_replacements import (
    AffineMap,
    ConditionalReplacementBudget,
    ConditionalReplacementResourceError,
    ConditionalReplacementValidationError,
    DEFAULT_MAX_ESTIMATED_BYTES,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_WORK_UNITS,
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_SECONDS,
    HARD_MAX_WORK_UNITS,
    MAX_OPERATIONS,
    OverwriteRule,
    PAYLOAD_COUNT,
    affine_normal_form,
    analyze_conditional_replacements,
    apply_affine,
    apply_normal_form,
    apply_overwrite,
    apply_program,
    canonicalize_program,
    compose_overwrites,
    pivot,
    relabel_overwrite,
    relabel_payload,
    rotation,
    sector_overwrite,
)


def _overlapping_program():
    return (
        rotation(2),
        sector_overwrite(1, (1, 0, 1)),
        pivot(3),
        sector_overwrite(5, (0, 1, 1)),
    )


def _disjoint_program():
    return (
        rotation(1),
        sector_overwrite(0, (1, 0)),
        pivot(5),
        sector_overwrite(3, (0, 1)),
    )


class ConditionalReplacementNormalFormTests(unittest.TestCase):
    def test_rotations_and_pivots_collapse_to_one_affine_map(self):
        operations = (rotation(2), pivot(3), rotation(4), pivot(5))
        normal = affine_normal_form(operations)

        self.assertEqual(normal, AffineMap(1, 1))
        for address in range(7):
            direct = address
            for operation in operations:
                direct = apply_affine(direct, operation.affine)
            self.assertEqual(direct, apply_affine(address, normal))

    def test_mixed_program_matches_affine_then_lww_normal_form_exhaustively(self):
        operations = _overlapping_program()
        normal = canonicalize_program(operations)

        self.assertEqual(normal.affine, AffineMap(3, 6))
        for payload in range(PAYLOAD_COUNT):
            self.assertEqual(
                apply_program(payload, operations),
                apply_normal_form(payload, normal),
            )

        result = analyze_conditional_replacements(operations)
        self.assertTrue(result.address_normal_form_holds)
        self.assertTrue(result.program_normal_form_holds)
        self.assertTrue(result.overlapping_last_write_wins_applicable)
        self.assertTrue(result.last_write_wins_holds)
        self.assertFalse(result.pair_is_disjoint)
        self.assertIsNone(result.disjoint_commutation_holds)

    def test_disjoint_overwrites_commute_on_every_binary_payload(self):
        result = analyze_conditional_replacements(_disjoint_program())

        self.assertTrue(result.pair_is_disjoint)
        self.assertTrue(result.disjoint_commutation_applicable)
        self.assertTrue(result.disjoint_commutation_holds)
        self.assertTrue(result.pair_commutes_on_all_payloads)
        self.assertFalse(result.overlapping_last_write_wins_applicable)
        self.assertTrue(result.last_write_wins_holds)

    def test_overlapping_overwrites_use_declared_last_write_wins_order(self):
        left = OverwriteRule(0b0000011, 0b0000011)
        right = OverwriteRule(0b0000110, 0b0000100)
        forward = compose_overwrites((left, right))
        reverse = compose_overwrites((right, left))

        self.assertEqual(forward, OverwriteRule(0b0000111, 0b0000101))
        self.assertEqual(reverse, OverwriteRule(0b0000111, 0b0000111))
        self.assertNotEqual(
            apply_overwrite(0, forward),
            apply_overwrite(0, reverse),
        )

    def test_sector_replacement_is_conjugate_under_shifts(self):
        rule = sector_overwrite(5, (1, 0, 1))
        shift = AffineMap(1, 3)
        inverse_shift = AffineMap(1, 4)
        shifted = relabel_overwrite(rule, shift)

        for payload in range(PAYLOAD_COUNT):
            conjugated = relabel_payload(
                apply_overwrite(
                    relabel_payload(payload, inverse_shift),
                    rule,
                ),
                shift,
            )
            self.assertEqual(conjugated, apply_overwrite(payload, shifted))

    def test_controls_match_sector_size_overlap_and_address_invariants(self):
        result = analyze_conditional_replacements(
            _overlapping_program(),
            control_shift=3,
            random_seed=4691,
        )

        for control in (
            result.shift_control,
            result.random_matched_control,
        ):
            with self.subTest(control=control.name):
                self.assertTrue(control.profile_matches_original)
                self.assertEqual(control.profile, result.pair_profile)
                self.assertTrue(control.overwrite_sequence_conjugacy_holds)
                self.assertTrue(control.last_write_wins_holds)
        self.assertTrue(result.shift_control.full_program_invariance_checked)
        self.assertTrue(result.shift_control.full_program_invariance_holds)
        self.assertFalse(
            result.random_matched_control.full_program_invariance_checked
        )
        self.assertIsNone(
            result.random_matched_control.full_program_invariance_holds
        )

        self.assertEqual(
            tuple(item.root for item in result.primitive_root_controls),
            (3, 5),
        )
        for item in result.primitive_root_controls:
            with self.subTest(root=item.root):
                self.assertTrue(item.is_primitive_root)
                self.assertTrue(item.profile_matches_original)
                self.assertTrue(item.overwrite_sequence_conjugacy_holds)
                self.assertTrue(item.last_write_wins_holds)
                self.assertTrue(item.full_program_invariance_holds)

    def test_random_matched_control_is_deterministic_and_seeded(self):
        first = analyze_conditional_replacements(
            _overlapping_program(),
            random_seed=67,
        )
        repeated = analyze_conditional_replacements(
            _overlapping_program(),
            random_seed=67,
        )
        changed = analyze_conditional_replacements(
            _overlapping_program(),
            random_seed=68,
        )

        self.assertEqual(
            first.random_matched_control,
            repeated.random_matched_control,
        )
        self.assertNotEqual(
            first.random_matched_control.permutation,
            changed.random_matched_control.permutation,
        )

    def test_resource_estimate_is_tiny_streamed_and_exactly_accounted(self):
        result = analyze_conditional_replacements(_overlapping_program())
        resource = result.resource_guard

        self.assertEqual(resource.operation_count, 4)
        self.assertEqual(resource.address_operation_count, 2)
        self.assertEqual(resource.overwrite_operation_count, 2)
        self.assertEqual(resource.exhaustive_payloads_per_pass, 128)
        self.assertEqual(resource.exhaustive_payload_passes_upper_bound, 16)
        self.assertEqual(resource.exhaustive_payload_cases_upper_bound, 2_048)
        self.assertEqual(resource.primitive_root_controls, 2)
        self.assertEqual(
            resource.estimated_peak_bytes,
            sum(value for _name, value in resource.component_bytes),
        )
        self.assertEqual(resource.estimated_peak_bytes, 99_072)
        self.assertEqual(resource.estimated_work_units, 29_808)
        self.assertLessEqual(
            resource.estimated_peak_bytes,
            DEFAULT_MAX_ESTIMATED_BYTES,
        )
        self.assertLessEqual(
            resource.estimated_work_units,
            DEFAULT_MAX_WORK_UNITS,
        )
        self.assertEqual(resource.max_seconds, DEFAULT_MAX_SECONDS)
        self.assertEqual(resource.max_operations, MAX_OPERATIONS)
        self.assertTrue(resource.payloads_streamed_not_materialized)
        self.assertFalse(resource.caller_owned_inputs_included)
        self.assertFalse(resource.measured_process_peak)

    def test_payload_pass_preflight_upper_bound_matches_success_path(self):
        with patch.object(
            replacements,
            "_iter_payloads",
            wraps=replacements._iter_payloads,
        ) as payload_pass:
            result = analyze_conditional_replacements(_overlapping_program())

        self.assertEqual(
            payload_pass.call_count,
            result.resource_guard.exhaustive_payload_passes_upper_bound,
        )

    def test_byte_and_work_budgets_fail_before_payload_or_control_enumeration(self):
        operations = _overlapping_program()
        for budget in (
            ConditionalReplacementBudget(max_estimated_bytes=1),
            ConditionalReplacementBudget(max_work_units=1),
        ):
            with self.subTest(budget=budget):
                with (
                    patch(
                        "prime_ring_conditional_replacements._iter_payloads"
                    ) as payloads,
                    patch(
                        "prime_ring_conditional_replacements._random_permutation"
                    ) as random_control,
                    self.assertRaises(ConditionalReplacementResourceError),
                ):
                    analyze_conditional_replacements(
                        operations,
                        budget=budget,
                    )
                payloads.assert_not_called()
                random_control.assert_not_called()

    def test_deadline_fails_before_payload_or_control_enumeration(self):
        with (
            patch(
                "prime_ring_conditional_replacements.time.monotonic",
                side_effect=(0.0, DEFAULT_MAX_SECONDS + 1.0),
            ),
            patch(
                "prime_ring_conditional_replacements._iter_payloads"
            ) as payloads,
            patch(
                "prime_ring_conditional_replacements._random_permutation"
            ) as random_control,
            self.assertRaises(ConditionalReplacementResourceError),
        ):
            analyze_conditional_replacements(_overlapping_program())
        payloads.assert_not_called()
        random_control.assert_not_called()

    def test_deadline_covers_accepted_sequence_inspection(self):
        class SlowOnceList(list):
            delayed = False

            def __len__(self):
                if not self.delayed:
                    self.delayed = True
                    time.sleep(0.02)
                return super().__len__()

        with self.assertRaises(ConditionalReplacementResourceError):
            analyze_conditional_replacements(
                SlowOnceList(_overlapping_program()),
                budget=ConditionalReplacementBudget(max_seconds=0.001),
            )

    def test_four_operation_cap_and_input_contract_fail_closed(self):
        with self.assertRaises(ConditionalReplacementResourceError):
            analyze_conditional_replacements(
                _overlapping_program() + (rotation(1),)
            )
        with self.assertRaises(ConditionalReplacementValidationError):
            analyze_conditional_replacements(
                (rotation(1), sector_overwrite(0, (1,)))
            )
        with self.assertRaises(ConditionalReplacementValidationError):
            sector_overwrite(0, (1, True))
        with self.assertRaises(ConditionalReplacementValidationError):
            OverwriteRule(0, 0)
        with self.assertRaises(ConditionalReplacementValidationError):
            OverwriteRule(1, 2)
        with self.assertRaises(ConditionalReplacementValidationError):
            pivot(0)
        with self.assertRaises(ConditionalReplacementValidationError):
            analyze_conditional_replacements(
                _overlapping_program(),
                control_shift=0,
            )
        with self.assertRaises(ConditionalReplacementValidationError):
            analyze_conditional_replacements(
                _overlapping_program(),
                random_seed=2**32,
            )

        for field, value in (
            ("max_estimated_bytes", HARD_MAX_ESTIMATED_BYTES + 1),
            ("max_work_units", HARD_MAX_WORK_UNITS + 1),
            ("max_seconds", HARD_MAX_SECONDS + 0.01),
            ("max_operations", MAX_OPERATIONS + 1),
        ):
            with self.subTest(field=field):
                with self.assertRaises(
                    ConditionalReplacementValidationError
                ):
                    ConditionalReplacementBudget(**{field: value})

    def test_result_cannot_be_misread_as_novelty_or_runtime_evidence(self):
        result = analyze_conditional_replacements(_overlapping_program())

        self.assertEqual(
            result.evidence_mode,
            "EXACT_BOUNDED_ALGEBRAIC_FALSIFICATION",
        )
        self.assertEqual(
            result.hypothesis_status,
            "NON_PROMOTIONAL_NORMAL_FORM_SCREEN",
        )
        self.assertFalse(result.promotion_eligible)
        self.assertFalse(result.novelty_claim)
        self.assertFalse(result.ml_utility_claim)
        self.assertFalse(result.runtime_benefit_claim)
        self.assertTrue(any("p=7" in item for item in result.limitations))
        self.assertTrue(
            any(
                "payload-, query-, or layer-state-dependent" in item
                for item in result.limitations
            )
        )
        self.assertTrue(
            any("not measured" in item for item in result.limitations)
        )
        self.assertTrue(
            any("do not establish novelty" in item for item in result.limitations)
        )


if __name__ == "__main__":
    unittest.main()
