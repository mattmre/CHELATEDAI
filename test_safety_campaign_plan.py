"""Stage 6 road-course campaign planning and documentation tests."""

import unittest

from run_safety_testbed import build_road_course_campaign_plan, render_campaign_documentation


class TestStage6RoadCourseCampaignPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_road_course_campaign_plan()
        cls.markdown = render_campaign_documentation(cls.plan)

    def test_campaign_plan_includes_beir_transfer_repeatability_and_quantization(self):
        campaigns = self.plan["campaigns"]

        for name in ["beir_small_tier", "multitask_transfer", "repeatability_matrix", "quantization_survival"]:
            self.assertIn(name, campaigns)
            self.assertTrue(campaigns[name]["commands"])
            self.assertTrue(campaigns[name]["required_evidence"])

    def test_default_promotion_gate_is_blocked_and_evidence_based(self):
        gate = self.plan["promotion_gate"]

        self.assertFalse(gate["default_change_allowed"])
        self.assertEqual(gate["approved_guardrail_changes"]["chelation_threshold"], 0.01)
        self.assertGreaterEqual(gate["required_conditions"]["retrieval_lift_min"], 0.01)
        self.assertGreaterEqual(gate["required_conditions"]["quantized_retained_gain_min"], 0.8)
        self.assertIn("norm ratio exits the hard band", "; ".join(gate["reject_conditions"]))

    def test_documentation_refresh_schema_names_required_outputs(self):
        refresh = self.plan["documentation_refresh"]

        self.assertEqual(refresh["summary_doc"], "docs/safety-testbed-road-course-plan.md")
        self.assertIn("quantization retention", refresh["required_sections"])
        self.assertIn("warnings/failures", refresh["required_sections"])

    def test_markdown_renderer_mentions_campaigns_and_gates(self):
        self.assertIn("# Safety Testbed Road-Course Campaign Plan", self.markdown)
        self.assertIn("beir_small_tier", self.markdown)
        self.assertIn("multitask_transfer", self.markdown)
        self.assertIn("Default change allowed now: `False`", self.markdown)
        self.assertIn("Quantized retained gain minimum", self.markdown)


if __name__ == "__main__":
    unittest.main(verbosity=2)
