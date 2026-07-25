from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from prime_ring_conditional_replacements import (
    OverwriteRule,
    pivot,
    rotation,
    sector_overwrite,
)
from prime_ring_runtime_gates import (
    DEFAULT_MAX_ESTIMATED_BYTES,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_WORK_UNITS,
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_SECONDS,
    HARD_MAX_WORK_UNITS,
    KILLED_STATUS,
    MAX_GATE_BITS,
    MAX_OPERATIONS,
    PAYLOAD_COUNT,
    QUERY_COUNT,
    RuntimeGateBudget,
    RuntimeGateResourceError,
    RuntimeGateValidationError,
    RuntimeOverwrite,
    RuntimePredicate,
    affine_parity_predicate,
    analyze_runtime_gates,
    apply_flat_decision_table,
    compile_flat_decision_table,
    execute_runtime_program,
    execute_static_branch,
    runtime_overwrite,
)


def _nested_runtime_program():
    return (
        rotation(1),
        runtime_overwrite(
            affine_parity_predicate(
                query_mask=0b01,
                state_mask=0b0000001,
            ),
            sector_overwrite(1, (1, 0, 1)),
        ),
        pivot(3),
        runtime_overwrite(
            affine_parity_predicate(
                query_mask=0b10,
                state_mask=0b0000100,
                constant=1,
            ),
            sector_overwrite(5, (0, 1, 1)),
        ),
    )


class RuntimeGateExactScreenTests(unittest.TestCase):
    def test_predicate_is_an_explicit_query_state_truth_table(self):
        predicate = affine_parity_predicate(
            query_mask=0b01,
            state_mask=0b0000101,
            constant=1,
        )

        for query in range(QUERY_COUNT):
            for state in range(PAYLOAD_COUNT):
                expected = bool(
                    1
                    ^ ((query & 0b01).bit_count() & 1)
                    ^ ((state & 0b0000101).bit_count() & 1)
                )
                self.assertEqual(predicate.evaluate(query, state), expected)

    def test_non_linear_explicit_predicate_is_not_reduced_to_polarity(self):
        truth_bits = 0
        for query in range(QUERY_COUNT):
            for state in range(PAYLOAD_COUNT):
                if query == 3 and state.bit_count() == 2:
                    truth_bits |= 1 << (query * PAYLOAD_COUNT + state)
        predicate = RuntimePredicate(truth_bits)
        program = (
            runtime_overwrite(
                predicate,
                OverwriteRule(0b0000011, 0b0000001),
            ),
        )

        self.assertFalse(predicate.evaluate(3, 0b0000001))
        self.assertTrue(predicate.evaluate(3, 0b0000011))
        self.assertFalse(predicate.evaluate(2, 0b0000011))
        self.assertTrue(analyze_runtime_gates(program).kill_criteria_met)

    def test_second_gate_observes_state_changed_by_first_gate(self):
        first = runtime_overwrite(
            affine_parity_predicate(query_mask=1),
            OverwriteRule(0b0000001, 0b0000001),
        )
        second = runtime_overwrite(
            affine_parity_predicate(state_mask=1),
            OverwriteRule(0b0000010, 0b0000010),
        )
        program = (first, second)

        false_path = execute_runtime_program(0, 0, program)
        true_path = execute_runtime_program(0, 1, program)

        self.assertEqual(false_path.decisions, (False, False))
        self.assertEqual(false_path.output_payload, 0)
        self.assertEqual(true_path.decisions, (True, True))
        self.assertEqual(true_path.branch_transcript, 0b11)
        self.assertEqual(true_path.output_payload, 0b11)

    def test_compiler_emits_at_most_two_to_the_paid_branch_bits(self):
        table = compile_flat_decision_table(_nested_runtime_program())

        self.assertEqual(table.gate_bits, 2)
        self.assertEqual(len(table.leaves), 4)
        self.assertEqual(
            tuple(leaf.transcript for leaf in table.leaves),
            (0, 1, 2, 3),
        )
        self.assertEqual(table.leaves[0].active_gate_indices, ())
        self.assertEqual(table.leaves[3].active_gate_indices, (0, 1))

    def test_every_static_leaf_normal_form_matches_its_forced_branch(self):
        program = _nested_runtime_program()
        table = compile_flat_decision_table(program)

        for transcript in range(len(table.leaves)):
            for payload in range(PAYLOAD_COUNT):
                with self.subTest(
                    transcript=transcript,
                    payload=payload,
                ):
                    self.assertEqual(
                        execute_static_branch(
                            payload,
                            program,
                            transcript,
                        ),
                        apply_flat_decision_table(
                            payload,
                            transcript,
                            table,
                        ),
                    )

    def test_all_payload_query_executions_equal_paid_transcript_table(self):
        program = _nested_runtime_program()
        table = compile_flat_decision_table(program)

        reachable = set()
        for payload in range(PAYLOAD_COUNT):
            for query in range(QUERY_COUNT):
                direct = execute_runtime_program(payload, query, program)
                reachable.add(direct.branch_transcript)
                self.assertEqual(
                    direct.output_payload,
                    apply_flat_decision_table(
                        payload,
                        direct.branch_transcript,
                        table,
                    ),
                )
        self.assertEqual(reachable, {0, 1, 2, 3})

    def test_analysis_exhausts_fixed_domain_and_meets_kill_criteria(self):
        result = analyze_runtime_gates(_nested_runtime_program())

        self.assertEqual(result.exhaustive_runtime_cases, 512)
        self.assertEqual(result.exhaustive_static_leaf_cases, 512)
        self.assertTrue(result.all_four_queries_tested)
        self.assertEqual(result.reachable_transcripts, (0, 1, 2, 3))
        self.assertTrue(result.flat_equivalence_holds)
        self.assertTrue(result.static_leaf_normal_forms_hold)
        self.assertTrue(result.leaf_bound_holds)
        self.assertTrue(result.controls_matched)
        self.assertTrue(result.kill_criteria_met)
        self.assertFalse(result.behavioral_state_gain_supported)
        self.assertEqual(result.hypothesis_status, KILLED_STATUS)

    def test_control_is_matched_on_every_preregistered_dimension(self):
        result = analyze_runtime_gates(_nested_runtime_program())
        structured = result.structured_control
        flat = result.flat_control

        self.assertNotEqual(structured.name, flat.name)
        self.assertEqual(
            structured.matching_signature,
            flat.matching_signature,
        )
        self.assertEqual(structured.branch_bits, 2)
        self.assertEqual(structured.leaf_capacity, 4)
        self.assertEqual(structured.compiled_leaves, 4)
        self.assertEqual(structured.replacement_templates, 2)
        self.assertEqual(structured.maximum_replacement_applications, 2)
        self.assertEqual(structured.replacement_mask_sizes, (3, 3))
        self.assertEqual(structured.replacement_pair_overlap, 1)
        self.assertEqual(structured.leaf_searches_per_case, 1)
        self.assertGreater(structured.charged_storage_bytes, 0)
        self.assertGreater(structured.charged_work_units_per_case, 0)
        self.assertFalse(structured.measured_storage_or_work)
        self.assertTrue(structured.common_envelope_padding_allowed)
        self.assertTrue(result.branch_transcript_granted_to_flat_control)

    def test_resource_estimate_is_small_streamed_and_exactly_accounted(self):
        resource = analyze_runtime_gates(
            _nested_runtime_program()
        ).resource_guard

        self.assertEqual(resource.operation_count, 4)
        self.assertEqual(resource.gate_bits, 2)
        self.assertEqual(resource.leaf_capacity, 4)
        self.assertEqual(resource.payload_count, 128)
        self.assertEqual(resource.query_count, 4)
        self.assertEqual(resource.runtime_case_count, 512)
        self.assertEqual(resource.static_leaf_case_count, 512)
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
            resource.estimated_work_units,
            DEFAULT_MAX_WORK_UNITS,
        )
        self.assertEqual(resource.max_seconds, DEFAULT_MAX_SECONDS)
        self.assertEqual(resource.max_operations, MAX_OPERATIONS)
        self.assertEqual(resource.max_gate_bits, MAX_GATE_BITS)
        self.assertTrue(resource.cases_streamed_not_materialized)
        self.assertFalse(resource.measured_process_peak)

    def test_byte_and_work_budgets_fail_before_compile_or_enumeration(self):
        for budget in (
            RuntimeGateBudget(max_estimated_bytes=1),
            RuntimeGateBudget(max_work_units=1),
        ):
            with self.subTest(budget=budget):
                with (
                    patch(
                        "prime_ring_runtime_gates.compile_flat_decision_table"
                    ) as compiler,
                    patch(
                        "prime_ring_runtime_gates._iter_runtime_cases"
                    ) as cases,
                    self.assertRaises(RuntimeGateResourceError),
                ):
                    analyze_runtime_gates(
                        _nested_runtime_program(),
                        budget=budget,
                    )
                compiler.assert_not_called()
                cases.assert_not_called()

    def test_deadline_fails_before_compile_or_enumeration(self):
        with (
            patch(
                "prime_ring_runtime_gates.time.monotonic",
                side_effect=(0.0, DEFAULT_MAX_SECONDS + 1.0),
            ),
            patch(
                "prime_ring_runtime_gates.compile_flat_decision_table"
            ) as compiler,
            patch(
                "prime_ring_runtime_gates._iter_runtime_cases"
            ) as cases,
            self.assertRaises(RuntimeGateResourceError),
        ):
            analyze_runtime_gates(_nested_runtime_program())
        compiler.assert_not_called()
        cases.assert_not_called()

    def test_deadline_covers_accepted_sequence_inspection(self):
        class SlowOnceList(list):
            delayed = False

            def __len__(self):
                if not self.delayed:
                    self.delayed = True
                    time.sleep(0.02)
                return super().__len__()

        with self.assertRaises(RuntimeGateResourceError):
            analyze_runtime_gates(
                SlowOnceList(_nested_runtime_program()),
                budget=RuntimeGateBudget(max_seconds=0.001),
            )

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

        budget = RuntimeGateBudget(
            max_estimated_bytes=HostileInt(DEFAULT_MAX_ESTIMATED_BYTES),
            max_work_units=HostileInt(DEFAULT_MAX_WORK_UNITS),
            max_seconds=HostileFloat(DEFAULT_MAX_SECONDS),
            max_operations=HostileInt(MAX_OPERATIONS),
            max_gate_bits=HostileInt(MAX_GATE_BITS),
        )

        self.assertIs(type(budget.max_estimated_bytes), int)
        self.assertIs(type(budget.max_work_units), int)
        self.assertIs(type(budget.max_seconds), float)
        self.assertIs(type(budget.max_operations), int)
        self.assertIs(type(budget.max_gate_bits), int)
        self.assertTrue(
            analyze_runtime_gates(
                _nested_runtime_program(),
                budget=budget,
            ).kill_criteria_met
        )

    def test_operation_gate_and_transcript_limits_fail_closed(self):
        predicate = affine_parity_predicate(query_mask=1)
        overwrite = OverwriteRule(1, 1)
        gate = runtime_overwrite(predicate, overwrite)

        with self.assertRaises(RuntimeGateValidationError):
            analyze_runtime_gates(())
        with self.assertRaises(RuntimeGateValidationError):
            analyze_runtime_gates((rotation(1),))
        with self.assertRaises(RuntimeGateResourceError):
            analyze_runtime_gates((gate, gate, gate))
        with self.assertRaises(RuntimeGateResourceError):
            analyze_runtime_gates(
                _nested_runtime_program() + (rotation(1),)
            )
        with self.assertRaises(RuntimeGateValidationError):
            execute_runtime_program(128, 0, (gate,))
        with self.assertRaises(RuntimeGateValidationError):
            execute_runtime_program(0, 4, (gate,))
        with self.assertRaises(RuntimeGateValidationError):
            execute_static_branch(0, (gate,), 2)
        with self.assertRaises(RuntimeGateValidationError):
            apply_flat_decision_table(
                0,
                2,
                compile_flat_decision_table((gate,)),
            )

    def test_predicate_and_wrapper_inputs_fail_closed(self):
        with self.assertRaises(RuntimeGateValidationError):
            RuntimePredicate(True)
        with self.assertRaises(RuntimeGateValidationError):
            RuntimePredicate(1 << 512)
        with self.assertRaises(RuntimeGateValidationError):
            affine_parity_predicate()
        with self.assertRaises(RuntimeGateValidationError):
            affine_parity_predicate(query_mask=4)
        with self.assertRaises(RuntimeGateValidationError):
            affine_parity_predicate(state_mask=128)
        with self.assertRaises(RuntimeGateValidationError):
            affine_parity_predicate(query_mask=1, constant=2)
        with self.assertRaises(RuntimeGateValidationError):
            RuntimeOverwrite(object(), OverwriteRule(1, 1))
        with self.assertRaises(RuntimeGateValidationError):
            RuntimeOverwrite(RuntimePredicate(0), object())

        for field, value in (
            ("max_estimated_bytes", HARD_MAX_ESTIMATED_BYTES + 1),
            ("max_work_units", HARD_MAX_WORK_UNITS + 1),
            ("max_seconds", HARD_MAX_SECONDS + 0.01),
            ("max_operations", MAX_OPERATIONS + 1),
            ("max_gate_bits", MAX_GATE_BITS + 1),
        ):
            with self.subTest(field=field):
                with self.assertRaises(RuntimeGateValidationError):
                    RuntimeGateBudget(**{field: value})

    def test_result_scope_cannot_be_misread_as_novelty_or_runtime_evidence(self):
        result = analyze_runtime_gates(_nested_runtime_program())

        self.assertEqual(
            result.evidence_mode,
            "EXACT_BOUNDED_BEHAVIORAL_EQUIVALENCE",
        )
        self.assertFalse(result.promotion_eligible)
        self.assertFalse(result.novelty_claim)
        self.assertFalse(result.ml_utility_claim)
        self.assertFalse(result.runtime_benefit_claim)
        self.assertTrue(
            any("p=7" in limitation for limitation in result.limitations)
        )
        self.assertTrue(
            any(
                "replacement values are fixed" in limitation
                for limitation in result.limitations
            )
        )
        self.assertTrue(
            any(
                "granted the exact runtime branch transcript" in limitation
                for limitation in result.limitations
            )
        )
        self.assertTrue(
            any(
                "not measured" in limitation
                for limitation in result.limitations
            )
        )
        self.assertEqual(len(result.kill_criteria), 4)


if __name__ == "__main__":
    unittest.main()
