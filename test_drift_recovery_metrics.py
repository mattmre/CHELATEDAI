from __future__ import annotations

import json
import math
import unittest

from drift_recovery_metrics import RecoveryTracker, ndcg_at_k


class TestNdcgAtK(unittest.TestCase):
    def test_ndcg_at_k_matches_hand_computed_single_relevant_rank_three(self):
        score = ndcg_at_k(["d0", "d1", "d2"], {"d2"}, k=3)
        self.assertAlmostEqual(score, 1.0 / math.log2(4))

    def test_ndcg_at_k_matches_hand_computed_two_relevant_docs(self):
        ranked = ["a", "x", "b", "c"]
        relevant = {"a", "b"}
        dcg = 1.0 + (1.0 / math.log2(4))
        ideal = 1.0 + (1.0 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(ranked, relevant, k=4), dcg / ideal)

    def test_ndcg_at_k_empty_relevant_set_is_zero(self):
        self.assertEqual(ndcg_at_k(["a", "b"], set(), k=10), 0.0)

    def test_ndcg_at_k_rejects_invalid_k(self):
        with self.assertRaises(ValueError):
            ndcg_at_k(["a"], {"a"}, k=0)


class TestRecoveryTracker(unittest.TestCase):
    def test_recovery_cycle_requires_two_consecutive_cycles(self):
        tracker = RecoveryTracker(baseline_ndcg=1.0, recovery_threshold=0.95)
        tracker.record_cycle(1, 0.90, {"phase": "drifted"})
        tracker.record_cycle(2, 0.96, {"phase": "candidate"})
        tracker.record_cycle(3, 0.94, {"phase": "dip"})
        tracker.record_cycle(4, 0.95, {"phase": "recover"})
        tracker.record_cycle(5, 0.97, {"phase": "sustain"})

        self.assertEqual(tracker.recovery_cycle(), 4)

    def test_recovery_cycle_none_when_never_sustained(self):
        tracker = RecoveryTracker(baseline_ndcg=0.8, recovery_threshold=0.95)
        tracker.record_cycle(1, 0.76, {})
        tracker.record_cycle(2, 0.70, {})
        tracker.record_cycle(3, 0.77, {})

        self.assertIsNone(tracker.recovery_cycle())
        self.assertIsNone(tracker.post_recovery_stability())

    def test_recovery_then_dip_still_reports_first_sustained_pair_and_stability(self):
        tracker = RecoveryTracker(baseline_ndcg=1.0, recovery_threshold=0.9)
        tracker.record_cycle(1, 0.91, {})
        tracker.record_cycle(2, 0.92, {})
        tracker.record_cycle(3, 0.60, {})

        self.assertEqual(tracker.recovery_cycle(), 1)
        self.assertAlmostEqual(tracker.post_recovery_stability(), math.sqrt(((0.10 ** 2) + (0.11 ** 2) + (0.21 ** 2)) / 3))

    def test_non_consecutive_cycles_do_not_count_as_sustained(self):
        tracker = RecoveryTracker(baseline_ndcg=1.0, recovery_threshold=0.9)
        tracker.record_cycle(1, 0.91, {})
        tracker.record_cycle(3, 0.92, {})
        tracker.record_cycle(4, 0.93, {})

        self.assertEqual(tracker.recovery_cycle(), 3)

    def test_trajectory_returns_copy_and_json_round_trips(self):
        tracker = RecoveryTracker(baseline_ndcg=0.5)
        metadata = {"condition": "C3"}
        tracker.record_cycle(1, 0.4, metadata)
        metadata["condition"] = "mutated"

        trajectory = tracker.trajectory()
        self.assertEqual(trajectory, [(1, 0.4, {"condition": "C3"})])
        trajectory[0][2]["condition"] = "external"
        self.assertEqual(tracker.trajectory(), [(1, 0.4, {"condition": "C3"})])
        self.assertEqual(json.loads(json.dumps(tracker.to_json()))["trajectory"][0]["metadata"]["condition"], "C3")

    def test_invalid_constructor_values_raise(self):
        with self.assertRaises(ValueError):
            RecoveryTracker(baseline_ndcg=-0.1)
        with self.assertRaises(ValueError):
            RecoveryTracker(baseline_ndcg=1.0, recovery_threshold=0.0)


if __name__ == "__main__":
    unittest.main()
