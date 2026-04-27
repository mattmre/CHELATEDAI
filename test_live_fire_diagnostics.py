"""Live-fire integration tests for the adaptive diagnostics harness."""

import json
import unittest

from run_live_fire_diagnostics import KNOWN_GOOD_THRESHOLDS, run_live_fire_diagnostics


class TestLiveFireDiagnostics(unittest.TestCase):
    def test_live_fire_harness_exercises_controls_and_reporting(self):
        report = run_live_fire_diagnostics()
        json.dumps(report)

        self.assertIn(report["overall"], {"pass", "warning"})
        self.assertEqual(report["failures"], [])
        self.assertGreaterEqual(
            report["initial_chelated_values"]["fitness"],
            KNOWN_GOOD_THRESHOLDS["retrieval_fitness_min"],
        )
        self.assertIn("integrated_diagnostics", report["live_fire"])
        self.assertIn("adaptive_gate", report["live_fire"])
        self.assertIn("norm_drift", report["live_fire"]["stability_report"])
        self.assertGreater(report["summary"]["dashboard_summary"]["runtime_diagnostics_count"], 0)
        self.assertGreater(report["summary"]["dashboard_summary"]["adapter_route_breakdown"]["adaptive"], 0)
        self.assertIn("query_summary", report["live_fire"]["integrated_diagnostics"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
