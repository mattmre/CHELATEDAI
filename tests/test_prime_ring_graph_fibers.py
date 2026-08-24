from __future__ import annotations

from fractions import Fraction
import time
import unittest
from unittest.mock import patch

from prime_ring_graph_fibers import (
    ALLOWED_MODULI,
    DEFAULT_MAX_ESTIMATED_BYTES,
    DEFAULT_MAX_LABEL_PERMUTATIONS,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_WORK_UNITS,
    EVIDENCE_MODE,
    GraphFiberBudget,
    GraphFiberResourceError,
    GraphFiberValidationError,
    HARD_MAX_EDGES,
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_HYPOTHESES,
    HARD_MAX_LABEL_PERMUTATIONS,
    HARD_MAX_NODES,
    HARD_MAX_SECONDS,
    HARD_MAX_WORK_UNITS,
    KILL_CRITERIA,
    PROTOCOL_ID,
    analyze_graph_fibers,
)


def _helpful_instance():
    modulus = 7
    planted = (1, 3, 6)
    unary = []
    for true_phase in planted:
        row = [0] * modulus
        row[0] = 2
        row[true_phase] = 1
        unary.append(row)
    edges = (
        (0, 1, 2),
        (0, 2, 5),
        (1, 2, 3),
    )
    return unary, edges, planted


class TestPrimeRingGraphFibers(unittest.TestCase):
    def test_graph_coupling_beats_both_controls_on_same_exact_search(self):
        unary, edges, planted = _helpful_instance()
        result = analyze_graph_fibers(
            unary,
            edges,
            planted,
            modulus=7,
            coupling=2,
        )

        self.assertEqual(result.protocol_id, PROTOCOL_ID)
        self.assertEqual(result.evidence_mode, EVIDENCE_MODE)
        self.assertEqual(result.independent.assignment, (0, 0, 0))
        self.assertFalse(
            result.independent.selected_assignment_matches_planted
        )
        self.assertEqual(result.graph_coupled.assignment, planted)
        self.assertTrue(
            result.graph_coupled.selected_assignment_matches_planted
        )
        self.assertTrue(result.graph_coupled.unique_exact_recovery)
        self.assertFalse(
            result.shuffled_graph.selected_assignment_matches_planted
        )
        self.assertEqual(result.graph_coupled.planted_margin, Fraction(1))
        self.assertEqual(result.independent.planted_margin, Fraction(-3))
        self.assertEqual(result.shuffled_graph.planted_margin, Fraction(-4))
        self.assertEqual(
            result.paired_effects[
                "graph_minus_independent_unique_exact_recovery"
            ],
            1,
        )
        self.assertEqual(
            result.paired_effects[
                "graph_minus_shuffled_unique_exact_recovery"
            ],
            1,
        )
        self.assertEqual(
            result.paired_effects["graph_minus_shuffled_planted_margin"],
            Fraction(5),
        )
        self.assertEqual(
            result.paired_effects[
                "label_specific_coupling_interaction_on_unique_exact_recovery"
            ],
            1,
        )
        self.assertEqual(
            result.paired_effects[
                "label_specific_coupling_interaction_on_planted_margin"
            ],
            Fraction(5),
        )
        self.assertEqual(
            result.paired_effects["selected_graph_edge_violations"],
            0,
        )
        self.assertEqual(result.global_phase_offsets, (0, 2, 5))
        self.assertEqual(
            result.global_phase_comparator_status,
            "COMPLETED_EXACT_ONE_GLOBAL_PHASE_PLUS_FIXED_OFFSETS",
        )
        self.assertTrue(
            result.zero_violation_manifold_equals_global_phase_comparator
        )
        self.assertEqual(
            result.global_phase_comparator,
            result.graph_coupled,
        )
        self.assertEqual(result.global_phase_hypotheses_evaluated, 7)
        self.assertEqual(
            result.paired_effects[
                "graph_minus_global_phase_unique_exact_recovery"
            ],
            0,
        )
        self.assertEqual(
            result.paired_effects[
                "graph_minus_global_phase_planted_margin"
            ],
            Fraction(0),
        )
        self.assertEqual(
            result.paired_effects["graph_minus_global_phase_tie_count"],
            0,
        )
        self.assertTrue(
            result.paired_effects[
                "soft_graph_outcome_equals_global_phase_comparator"
            ]
        )

        contract = result.comparison_contract
        self.assertTrue(contract["same_unary_observation"])
        self.assertTrue(contract["same_assignment_order"])
        self.assertTrue(
            contract["full_space_decoders_same_hypothesis_budget"]
        )
        self.assertEqual(
            contract["full_space_hypotheses_per_decoder"],
            7**3,
        )
        self.assertTrue(
            contract["global_phase_comparator_uses_same_assignment_stream"]
        )
        self.assertEqual(
            contract["global_phase_comparator_hypotheses_evaluated"],
            7,
        )
        self.assertEqual(
            contract["global_phase_comparator_expected_hypotheses"],
            7,
        )
        self.assertTrue(
            contract["global_phase_comparator_search_accounting_matches"]
        )
        self.assertFalse(
            contract["graph_and_global_phase_same_hypothesis_count"]
        )
        self.assertEqual(contract["graph_rank_denominator"], 7**3)
        self.assertEqual(contract["global_phase_rank_denominator"], 7)
        self.assertTrue(
            contract[
                "zero_violation_manifold_equals_global_phase_comparator"
            ]
        )
        self.assertTrue(contract["graph_and_control_same_endpoints"])
        self.assertTrue(contract["graph_and_control_same_degree_sequence"])
        self.assertTrue(contract["graph_and_control_same_edge_count"])
        self.assertTrue(contract["graph_and_control_same_label_multiset"])
        self.assertTrue(contract["graph_and_control_same_cycle_consistency"])
        self.assertTrue(
            contract[
                "graph_and_control_same_satisfying_assignment_count"
            ]
        )
        self.assertTrue(contract["negative_control_changes_only_label_placement"])
        self.assertFalse(contract["matched_stored_bytes"])
        self.assertFalse(contract["matched_channel_uses"])
        self.assertFalse(contract["latency_measured"])
        self.assertFalse(contract["false_unlock_rate_measured"])

    def test_consistent_connected_graph_exposes_global_phase_collapse_kill(self):
        unary, edges, planted = _helpful_instance()
        result = analyze_graph_fibers(
            unary,
            edges,
            planted,
            modulus=7,
            coupling=2,
        )
        structure = result.graph_structure

        self.assertEqual(structure.degrees, (2, 2, 2))
        self.assertEqual(structure.component_count, 1)
        self.assertEqual(structure.cycle_rank, 1)
        self.assertTrue(structure.cycle_consistent)
        self.assertEqual(structure.satisfying_assignment_count, 7)
        self.assertTrue(structure.connected)
        self.assertTrue(structure.collapses_to_one_global_phase)
        collapse = (
            "connected_consistent_graph_is_one_global_phase_up_to_fixed_offsets"
        )
        self.assertIn(collapse, KILL_CRITERIA)
        self.assertIn(collapse, result.kill_criteria_triggered)

    def test_global_phase_collapse_is_structural_even_when_optimum_violates_edges(self):
        _unary, edges, planted = _helpful_instance()
        unary = [[0] * 7 for _ in range(3)]
        for row in unary:
            row[0] = 100
        result = analyze_graph_fibers(
            unary,
            edges,
            planted,
            modulus=7,
            coupling=1,
        )

        self.assertEqual(result.graph_coupled.assignment, (0, 0, 0))
        self.assertEqual(
            result.paired_effects["selected_graph_edge_violations"],
            3,
        )
        self.assertTrue(
            result.zero_violation_manifold_equals_global_phase_comparator
        )
        self.assertIsNotNone(result.global_phase_comparator)
        self.assertFalse(
            result.paired_effects[
                "selected_graph_within_global_phase_family"
            ]
        )
        self.assertFalse(
            result.paired_effects[
                "soft_graph_outcome_equals_global_phase_comparator"
            ]
        )
        self.assertIn(
            "connected_consistent_graph_is_one_global_phase_up_to_fixed_offsets",
            result.kill_criteria_triggered,
        )

    def test_label_shuffle_is_deterministic_degree_matched_and_order_invariant(self):
        unary, edges, planted = _helpful_instance()
        canonical = analyze_graph_fibers(
            unary,
            edges,
            planted,
            modulus=7,
            coupling=2,
        )
        reversed_input = (
            (2, 1, 4),
            (2, 0, 2),
            (1, 0, 5),
        )
        repeated = analyze_graph_fibers(
            unary,
            reversed_input,
            planted,
            modulus=7,
            coupling=2,
        )

        self.assertEqual(canonical, repeated)
        self.assertNotEqual(canonical.graph_edges, canonical.shuffled_edges)
        self.assertEqual(
            tuple((edge.source, edge.target) for edge in canonical.graph_edges),
            tuple(
                (edge.source, edge.target)
                for edge in canonical.shuffled_edges
            ),
        )
        self.assertEqual(
            sorted(edge.phase for edge in canonical.graph_edges),
            sorted(edge.phase for edge in canonical.shuffled_edges),
        )
        self.assertEqual(
            canonical.graph_structure.degrees,
            canonical.shuffled_structure.degrees,
        )
        self.assertEqual(
            canonical.shuffled_control_status,
            "COMPLETED_EXACT_STRUCTURE_MATCHED_LABEL_PERMUTATION",
        )
        self.assertEqual(
            canonical.graph_structure.cycle_consistent,
            canonical.shuffled_structure.cycle_consistent,
        )
        self.assertEqual(
            canonical.graph_structure.satisfying_assignment_count,
            canonical.shuffled_structure.satisfying_assignment_count,
        )

    def test_identical_labels_fail_closed_and_ties_are_explicit(self):
        unary = [[0] * 7 for _ in range(3)]
        edges = ((0, 1, 0), (0, 2, 0), (1, 2, 0))
        planted = (0, 0, 0)
        result = analyze_graph_fibers(
            unary,
            edges,
            planted,
            modulus=7,
        )

        self.assertEqual(result.graph_edges, result.shuffled_edges)
        self.assertEqual(
            result.shuffled_control_status,
            "STRUCTURALLY_UNAVAILABLE_NO_DISTINCT_STRUCTURE_MATCHED_LABEL_PERMUTATION",
        )
        self.assertFalse(
            result.comparison_contract[
                "negative_control_changes_only_label_placement"
            ]
        )
        self.assertIn(
            "label_shuffled_control_not_distinct",
            result.kill_criteria_triggered,
        )
        self.assertEqual(result.independent.assignment, planted)
        self.assertEqual(result.independent.tie_count, 7**3)
        self.assertFalse(result.independent.unique_exact_recovery)
        self.assertEqual(result.independent.planted_rank_min, 1)
        self.assertEqual(result.independent.planted_rank_max, 7**3)
        self.assertEqual(result.graph_coupled.tie_count, 7)
        self.assertEqual(result.graph_coupled.planted_rank_max, 7)
        self.assertEqual(result.graph_coupled.planted_margin, Fraction(0))
        self.assertEqual(
            result.global_phase_comparator,
            result.graph_coupled,
        )
        self.assertEqual(result.global_phase_comparator.tie_count, 7)
        self.assertEqual(
            result.global_phase_comparator.planted_rank_min,
            1,
        )
        self.assertEqual(
            result.global_phase_comparator.planted_rank_max,
            7,
        )
        self.assertEqual(
            result.paired_effects[
                "graph_minus_independent_unique_exact_recovery"
            ],
            0,
        )
        self.assertIn(
            "graph_coupling_not_better_than_independent_recovery",
            result.kill_criteria_triggered,
        )

    def test_inconsistent_cycle_and_invalid_plant_are_diagnostic_not_hidden(self):
        unary = [[0] * 7 for _ in range(3)]
        edges = ((0, 1, 1), (0, 2, 3), (1, 2, 1))
        planted = (0, 1, 2)
        result = analyze_graph_fibers(
            unary,
            edges,
            planted,
            modulus=7,
        )

        self.assertFalse(result.graph_structure.cycle_consistent)
        self.assertEqual(result.graph_structure.cycle_rank, 1)
        self.assertEqual(result.graph_structure.satisfying_assignment_count, 0)
        self.assertFalse(
            result.graph_structure.collapses_to_one_global_phase
        )
        self.assertIsNone(result.global_phase_offsets)
        self.assertEqual(
            result.global_phase_comparator_status,
            "STRUCTURALLY_UNAVAILABLE_CONTRADICTORY_CYCLE",
        )
        self.assertFalse(
            result.zero_violation_manifold_equals_global_phase_comparator
        )
        self.assertIsNone(result.global_phase_comparator)
        self.assertEqual(result.global_phase_hypotheses_evaluated, 0)
        self.assertIsNone(
            result.paired_effects[
                "graph_minus_global_phase_planted_margin"
            ]
        )
        self.assertTrue(
            result.comparison_contract[
                "global_phase_comparator_search_accounting_matches"
            ]
        )
        self.assertNotIn(
            "connected_consistent_graph_is_one_global_phase_up_to_fixed_offsets",
            result.kill_criteria_triggered,
        )
        self.assertEqual(
            result.paired_effects["original_planted_edge_violations"],
            1,
        )
        self.assertIn(
            "planted_assignment_violates_declared_edges",
            result.kill_criteria_triggered,
        )

    def test_global_phase_comparison_refuses_plant_outside_its_domain(self):
        _unary, edges, _planted = _helpful_instance()
        result = analyze_graph_fibers(
            [[0] * 7 for _ in range(3)],
            edges,
            (0, 0, 0),
            modulus=7,
        )

        self.assertEqual(result.global_phase_offsets, (0, 2, 5))
        self.assertTrue(
            result.zero_violation_manifold_equals_global_phase_comparator
        )
        self.assertEqual(
            result.global_phase_comparator_status,
            "REFUSED_PLANTED_ASSIGNMENT_OUTSIDE_ZERO_VIOLATION_MANIFOLD",
        )
        self.assertIsNone(result.global_phase_comparator)
        self.assertEqual(result.global_phase_hypotheses_evaluated, 0)
        self.assertIn(
            "planted_assignment_violates_declared_edges",
            result.kill_criteria_triggered,
        )

    def test_disconnected_graph_refuses_one_global_phase_comparator(self):
        result = analyze_graph_fibers(
            [[0] * 7 for _ in range(3)],
            ((0, 1, 2),),
            (0, 2, 4),
            modulus=7,
        )

        self.assertFalse(result.graph_structure.connected)
        self.assertTrue(result.graph_structure.cycle_consistent)
        self.assertEqual(
            result.graph_structure.satisfying_assignment_count,
            7**2,
        )
        self.assertIsNone(result.global_phase_offsets)
        self.assertEqual(
            result.global_phase_comparator_status,
            "STRUCTURALLY_UNAVAILABLE_REQUIRES_CONNECTED_GRAPH",
        )
        self.assertFalse(
            result.zero_violation_manifold_equals_global_phase_comparator
        )
        self.assertIsNone(result.global_phase_comparator)

    def test_exact_rational_scores_and_nonpromotional_boundaries(self):
        unary, edges, planted = _helpful_instance()
        unary[0][planted[0]] = 0.1
        result = analyze_graph_fibers(
            unary,
            edges,
            planted,
            modulus=7,
            coupling=Fraction(3, 2),
        )

        self.assertIsInstance(result.graph_coupled.score, Fraction)
        self.assertEqual(result.coupling, Fraction(3, 2))
        self.assertEqual(
            result.independent.planted_score,
            Fraction(21, 10),
        )
        self.assertEqual(
            result.hypothesis_status,
            "NON_CONFIRMATORY_SINGLE_INSTANCE",
        )
        self.assertFalse(result.promotion_eligible)
        self.assertFalse(result.physical_claim)
        self.assertFalse(result.novelty_claim)
        self.assertTrue(
            any("does not construct" in item for item in result.limitations)
        )
        self.assertTrue(
            any("No physical 3D" in item for item in result.limitations)
        )
        with self.assertRaises(TypeError):
            result.paired_effects["post_hoc_claim"] = True
        with self.assertRaises(TypeError):
            result.comparison_contract["same_unary_observation"] = False

    def test_resource_estimate_is_streaming_conservative_and_bounded(self):
        unary, edges, planted = _helpful_instance()
        result = analyze_graph_fibers(
            unary,
            edges,
            planted,
            modulus=7,
            coupling=2,
        )
        resource = result.resource_guard

        self.assertEqual(resource.candidate_assignments, 343)
        self.assertEqual(
            resource.full_space_hypotheses_evaluated_per_decoder,
            343,
        )
        self.assertEqual(
            resource.full_space_decoder_hypothesis_evaluations,
            3 * 343,
        )
        self.assertEqual(
            resource.global_phase_comparator_hypotheses_upper_bound,
            7,
        )
        self.assertEqual(
            resource.total_decoder_hypothesis_evaluations,
            3 * 343 + 7,
        )
        self.assertTrue(
            resource.total_decoder_hypothesis_evaluations_is_upper_bound
        )
        self.assertEqual(
            resource.label_permutation_candidates_upper_bound,
            6,
        )
        self.assertEqual(
            resource.estimated_peak_bytes,
            sum(value for _name, value in resource.component_bytes),
        )
        self.assertEqual(resource.estimated_peak_bytes, 65_600)
        self.assertEqual(resource.estimated_work_units, 36_913)
        self.assertLessEqual(
            resource.estimated_peak_bytes,
            DEFAULT_MAX_ESTIMATED_BYTES,
        )
        self.assertLessEqual(
            resource.estimated_work_units,
            DEFAULT_MAX_WORK_UNITS,
        )
        self.assertEqual(resource.max_seconds, DEFAULT_MAX_SECONDS)
        self.assertEqual(
            resource.max_label_permutations,
            DEFAULT_MAX_LABEL_PERMUTATIONS,
        )
        self.assertTrue(resource.streaming_assignments_not_materialized)
        self.assertFalse(resource.caller_owned_inputs_included)
        self.assertFalse(resource.measured_process_peak)

    def test_preflight_refuses_oversized_hypothesis_bytes_and_work(self):
        oversized_unary = [[0] * 7 for _ in range(4)]
        with patch("prime_ring_graph_fibers.product") as enumerator:
            with self.assertRaises(GraphFiberResourceError):
                analyze_graph_fibers(
                    oversized_unary,
                    ((0, 1, 1),),
                    (0, 1, 2, 3),
                    modulus=7,
                )
        enumerator.assert_not_called()

        unary, edges, planted = _helpful_instance()
        with (
            patch(
                "prime_ring_graph_fibers._normalize_unary_scores"
            ) as normalizer,
            self.assertRaises(GraphFiberResourceError),
        ):
            analyze_graph_fibers(
                unary,
                edges,
                planted,
                modulus=7,
                budget=GraphFiberBudget(max_estimated_bytes=1),
            )
        normalizer.assert_not_called()

        with (
            patch(
                "prime_ring_graph_fibers._normalize_unary_scores"
            ) as normalizer,
            self.assertRaises(GraphFiberResourceError),
        ):
            analyze_graph_fibers(
                unary,
                edges,
                planted,
                modulus=7,
                budget=GraphFiberBudget(max_work_units=1),
            )
        normalizer.assert_not_called()

        with (
            patch(
                "prime_ring_graph_fibers._normalize_unary_scores"
            ) as normalizer,
            self.assertRaises(GraphFiberResourceError),
        ):
            analyze_graph_fibers(
                unary,
                edges,
                planted,
                modulus=7,
                budget=GraphFiberBudget(max_label_permutations=1),
            )
        normalizer.assert_not_called()

    def test_deadline_fails_before_enumeration(self):
        unary, edges, planted = _helpful_instance()
        with (
            patch(
                "prime_ring_graph_fibers.time.monotonic",
                side_effect=(0.0, DEFAULT_MAX_SECONDS + 1.0),
            ),
            patch("prime_ring_graph_fibers.product") as enumerator,
            self.assertRaises(GraphFiberResourceError),
        ):
            analyze_graph_fibers(
                unary,
                edges,
                planted,
                modulus=7,
            )
        enumerator.assert_not_called()

    def test_deadline_covers_accepted_sequence_inspection(self):
        unary, edges, planted = _helpful_instance()

        class SlowOnceList(list):
            delayed = False

            def __len__(self):
                if not self.delayed:
                    self.delayed = True
                    time.sleep(0.02)
                return super().__len__()

        with self.assertRaises(GraphFiberResourceError):
            analyze_graph_fibers(
                SlowOnceList(unary),
                edges,
                planted,
                modulus=7,
                budget=GraphFiberBudget(max_seconds=0.001),
            )

    def test_hard_caps_and_malformed_inputs_fail_closed(self):
        self.assertEqual(ALLOWED_MODULI, (7, 11, 31))
        for field, value in (
            ("max_estimated_bytes", HARD_MAX_ESTIMATED_BYTES + 1),
            ("max_seconds", HARD_MAX_SECONDS + 0.01),
            ("max_work_units", HARD_MAX_WORK_UNITS + 1),
            ("max_hypotheses", HARD_MAX_HYPOTHESES + 1),
            ("max_nodes", HARD_MAX_NODES + 1),
            ("max_edges", HARD_MAX_EDGES + 1),
            (
                "max_label_permutations",
                HARD_MAX_LABEL_PERMUTATIONS + 1,
            ),
        ):
            with self.subTest(field=field):
                with self.assertRaises(GraphFiberValidationError):
                    GraphFiberBudget(**{field: value})

        unary, edges, planted = _helpful_instance()
        malformed = (
            ({"modulus": 4096}, edges, planted),
            ({"modulus": 7, "coupling": 0}, edges, planted),
            ({"modulus": 7}, (), planted),
            ({"modulus": 7}, ((0, 0, 1),), planted),
            ({"modulus": 7}, ((0, 1, 1), (1, 0, 6)), planted),
            ({"modulus": 7}, ((0, 1, 7),), planted),
            ({"modulus": 7}, edges, (0, 1)),
        )
        for kwargs, bad_edges, bad_planted in malformed:
            with self.subTest(
                kwargs=kwargs,
                edges=bad_edges,
                planted=bad_planted,
            ):
                with self.assertRaises(GraphFiberValidationError):
                    analyze_graph_fibers(
                        unary,
                        bad_edges,
                        bad_planted,
                        **kwargs,
                    )

        invalid_unary = [list(row) for row in unary]
        invalid_unary[0][0] = True
        with self.assertRaises(GraphFiberValidationError):
            analyze_graph_fibers(
                invalid_unary,
                edges,
                planted,
                modulus=7,
            )

    def test_largest_allowed_tiny_modulus_case_stays_under_hypothesis_cap(self):
        unary = [[0] * 31 for _ in range(2)]
        result = analyze_graph_fibers(
            unary,
            ((0, 1, 0),),
            (0, 0),
            modulus=31,
        )
        self.assertEqual(result.resource_guard.candidate_assignments, 31**2)
        self.assertLessEqual(
            result.resource_guard.candidate_assignments,
            HARD_MAX_HYPOTHESES,
        )
        self.assertIn(
            "label_shuffled_control_not_distinct",
            result.kill_criteria_triggered,
        )


if __name__ == "__main__":
    unittest.main()
