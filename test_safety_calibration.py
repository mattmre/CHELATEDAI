"""Stage 4 iterative calibration-loop tests."""

import json
import unittest

from run_safety_testbed import CALIBRATION_PROFILES, run_calibration_matrix, run_profile_calibration


class TestStage4CalibrationLoops(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = run_calibration_matrix()

    def test_calibration_schema_contains_required_sections(self):
        self.assertEqual(
            self.matrix["schema"]["fields"],
            ["profile", "controls", "baseline", "candidate", "gates", "diagnostics", "decision"],
        )
        json.dumps(self.matrix, sort_keys=True)

    def test_each_profile_loop_records_controls_metrics_gates_and_rollback(self):
        profiles = {profile["profile"]: profile for profile in self.matrix["profiles"]}

        self.assertEqual(set(profiles), set(CALIBRATION_PROFILES))
        for name, profile in profiles.items():
            self.assertEqual(profile["controls"], CALIBRATION_PROFILES[name])
            self.assertIn("fitness", profile["baseline"])
            self.assertIn("fitness_delta", profile["candidate"])
            self.assertIn("quantization", profile["gates"])
            self.assertIn("adaptive", profile["gates"])
            self.assertFalse(profile["decision"]["default_change_allowed"])
            self.assertIn("Rollback profile", profile["decision"]["rollback_condition"])

    def test_conservative_profile_is_near_baseline_or_rejected_not_promoted_by_default(self):
        profile = run_profile_calibration("conservative")

        self.assertIn(profile["decision"]["status"], {"hold", "reject", "profile_candidate"})
        self.assertFalse(profile["decision"]["default_change_allowed"])
        self.assertEqual(profile["controls"]["chelation_threshold"], 999.0)

    def test_aggressive_and_experimental_profiles_emit_reformulation_diagnostics(self):
        profiles = {
            profile["profile"]: profile
            for profile in self.matrix["profiles"]
        }

        for name in ["aggressive", "experimental"]:
            diagnostics = profiles[name]["diagnostics"]
            self.assertTrue(any(
                diagnostic.get("runtime", {}).get("action") == "REFORMULATE"
                for diagnostic in diagnostics
            ))

    def test_experimental_profile_records_routing_when_enabled(self):
        experimental = run_profile_calibration("experimental")

        self.assertTrue(experimental["controls"]["adapter_routing"])
        self.assertTrue(any(
            diagnostic.get("query_reformulation", {}).get("variant_count", 0) >= 1
            for diagnostic in experimental["diagnostics"]
        ))

    def test_decision_report_blocks_default_promotion_even_with_candidates(self):
        report = self.matrix["decision_report"]

        self.assertFalse(report["default_change_allowed"])
        self.assertIn("Keep defaults unchanged", report["recommendation"])
        for candidate in report["profile_candidates"]:
            self.assertIn(candidate, CALIBRATION_PROFILES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
