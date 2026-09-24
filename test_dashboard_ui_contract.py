"""Markup and evidence-index contracts for the dashboard control page."""

import os
import tempfile
import unittest

import dashboard_server


class TestDashboardMarkup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(__file__), "dashboard", "index.html")
        with open(path, encoding="utf-8") as handle:
            cls.html = handle.read()

    def test_sweep_status_does_not_claim_a_live_grid(self):
        self.assertNotIn("Running (", self.html)
        self.assertNotIn("7350", self.html)
        self.assertIn("Loaded (", self.html)

    def test_page_does_not_name_a_missing_report_generator(self):
        self.assertNotIn("generate_report_json.py", self.html)
        self.assertNotIn("server_gen4", self.html)
        self.assertNotIn("laptop_gen4", self.html)
        self.assertIn("consumer_gen4", self.html)
        self.assertIn("dual_nvme_workstation", self.html)

    def test_chain_cards_count_every_parsed_file(self):
        self.assertIn("Chains passed", self.html)
        self.assertIn("Chains failed", self.html)
        self.assertNotIn("Passed on this page", self.html)
        script_open = self.html.rindex("<script>")
        script_close = self.html.index("</script>", script_open)
        script = self.html[script_open:script_close]
        self.assertIn("Showing ", script)
        self.assertIn("readable chains", script)
        self.assertIn("unreadable_reports", script)

    def test_evidence_status_mentions_missing_files(self):
        start = self.html.index("async function loadEvidenceIndex")
        end = self.html.index("async function loadEvidenceChainHistory")
        slice_ = self.html[start:end]
        self.assertIn("evidence_chain_files_present", slice_)
        self.assertIn("index only", slice_)

    def test_event_timestamps_do_not_multiply_iso_strings(self):
        start = self.html.index("async function loadEvents")
        end = self.html.index("async function loadCampaignHistory")
        slice_ = self.html[start:end]
        self.assertNotIn("e.timestamp * 1000).toLocaleString()", slice_)
        self.assertIn("* 1000", slice_)
        self.assertIn("Z", slice_)


class TestEvidenceIndexFileCounts(unittest.TestCase):
    def test_missing_referenced_files_are_counted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "evidence_index.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    '{"summary": {"latest_chain_passed": true, "latest_review_allowed": false},'
                    '"artifacts": {"evidence_chain_summaries": [{"path": "missing\\\\chain.json"}],'
                    '"default_promotion_preflights": [{"path": "missing/preflight.json"}]}}'
                )
            result = dashboard_server.load_evidence_index(path)
        summary = result["summary"]
        self.assertTrue(summary["latest_chain_passed"])
        self.assertEqual(summary["evidence_chain_files_present"], 0)
        self.assertEqual(summary["evidence_chain_files_missing"], 1)
        self.assertEqual(summary["preflight_files_present"], 0)
        self.assertEqual(summary["preflight_files_missing"], 1)

    def test_existing_chain_file_is_counted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = os.path.join(tmpdir, "evidence_chain_summary.json")
            with open(chain, "w", encoding="utf-8") as handle:
                handle.write("{}")
            path = os.path.join(tmpdir, "evidence_index.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    '{"summary": {"latest_chain_passed": true},'
                    '"artifacts": {"evidence_chain_summaries": [{"path": "%s"}]}}'
                    % chain.replace("\\", "/")
                )
            result = dashboard_server.load_evidence_index(path)
        self.assertEqual(result["summary"]["evidence_chain_files_present"], 1)
        self.assertEqual(result["summary"]["evidence_chain_files_missing"], 0)


if __name__ == "__main__":
    unittest.main()
