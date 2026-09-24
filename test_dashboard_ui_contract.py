"""Markup and evidence-index contracts for the dashboard control page."""

import json
import os
import shutil
import subprocess
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


def _format_timestamp_source(text):
    start = text.index("function formatTimestamp")
    end = text.index("function formatNumber")
    return text[start:end]


_NODE_DRIVER = r"""
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const formatTimestamp = new Function(`${input.fn}
return formatTimestamp;`)();

function run(value) {
  try {
    return {ok: true, value: formatTimestamp(value)};
  } catch (err) {
    return {ok: false, error: String(err && err.stack ? err.stack : err)};
  }
}

const doubled = 1700000000 * 1000 * 1000;
const payload = {
  zero: run(0),
  seconds: run(1700000000),
  naive: run('2026-09-24T00:00:00'),
  zulu: run('2026-09-24T00:00:00Z'),
  offset: run('2026-09-24T00:00:00-05:00'),
  bad: run('not-a-time'),
  meta: {
    zeroLocale: new Date(0).toLocaleString(),
    zeroUtcYear: new Date(0).getUTCFullYear(),
    secondsScaledLocale: new Date(1700000000 * 1000).toLocaleString(),
    secondsScaledUtcYear: new Date(1700000000 * 1000).getUTCFullYear(),
    secondsRawLocale: new Date(1700000000).toLocaleString(),
    secondsRawUtcYear: new Date(1700000000).getUTCFullYear(),
    secondsDoubledLocale: new Date(doubled).toLocaleString(),
    secondsDoubledUtcYear: new Date(doubled).getUTCFullYear(),
    naiveUtcLocale: new Date('2026-09-24T00:00:00Z').toLocaleString(),
    naiveLocalLocale: new Date('2026-09-24T00:00:00').toLocaleString(),
    zuluLocale: new Date('2026-09-24T00:00:00Z').toLocaleString(),
    offsetLocale: new Date('2026-09-24T00:00:00-05:00').toLocaleString(),
    offsetMinutes: new Date('2026-09-24T00:00:00Z').getTimezoneOffset(),
  },
};
process.stdout.write(JSON.stringify(payload));
"""


class TestInlineDashboardTimestamp(unittest.TestCase):
    def test_format_timestamp_scales_numbers_and_keeps_regex_backslash_d(self):
        source_path = os.path.splitext(dashboard_server.__file__)[0] + ".py"
        with open(source_path, encoding="utf-8") as handle:
            source_fn = _format_timestamp_source(handle.read())
        html_fn = _format_timestamp_source(dashboard_server.get_inline_dashboard_html())

        # Python source writes \\d; the runtime HTML string contains \d, not \\d.
        self.assertEqual(source_fn.count("\\\\d"), 2)
        self.assertIn("/(?:Z|[+-]\\\\d{2}:?\\\\d{2})$/", source_fn)
        self.assertEqual(html_fn.count("\\\\d"), 0)
        self.assertEqual(html_fn.count("\\d"), 2)
        self.assertIn("/(?:Z|[+-]\\d{2}:?\\d{2})$/", html_fn)

        number_at = html_fn.index("typeof timestamp === 'number'")
        string_at = html_fn.index("typeof timestamp === 'string'")
        else_at = html_fn.index("} else {", string_at)
        number_branch = html_fn[number_at:string_at]
        string_branch = html_fn[string_at:else_at]
        self.assertIn("timestamp * 1000", number_branch)
        self.assertNotIn("timestamp * 1000", string_branch)
        self.assertNotIn("* 1000", string_branch)
        self.assertIn("+ 'Z'", string_branch)
        self.assertEqual(html_fn.count("timestamp * 1000"), 1)

        node = shutil.which("node")
        if node is None:
            print("node was missing")
            return

        utc = self._run_node(node, html_fn, "UTC")
        self._assert_common(utc)
        self.assertIn("1970", utc["zero"]["value"])
        self.assertEqual(utc["zero"]["value"], utc["meta"]["zeroLocale"])
        self.assertEqual(utc["naive"]["value"], utc["meta"]["naiveUtcLocale"])

        zoned = self._run_node(node, html_fn, "America/New_York")
        self._assert_common(zoned)
        self.assertNotEqual(zoned["meta"]["offsetMinutes"], 0)
        self.assertEqual(zoned["naive"]["value"], zoned["meta"]["naiveUtcLocale"])
        self.assertNotEqual(zoned["naive"]["value"], zoned["meta"]["naiveLocalLocale"])
        self.assertNotIn("2001", zoned["zero"]["value"])

    def _assert_common(self, payload):
        zero = payload["zero"]
        seconds = payload["seconds"]
        naive = payload["naive"]
        bad = payload["bad"]
        meta = payload["meta"]
        self.assertTrue(zero["ok"])
        self.assertTrue(seconds["ok"])
        self.assertTrue(naive["ok"])
        self.assertTrue(payload["zulu"]["ok"])
        self.assertTrue(payload["offset"]["ok"])
        self.assertTrue(bad["ok"])
        self.assertEqual(bad["value"], "-")
        self.assertEqual(meta["zeroUtcYear"], 1970)
        self.assertNotIn("2001", zero["value"])
        self.assertEqual(seconds["value"], meta["secondsScaledLocale"])
        self.assertNotEqual(seconds["value"], meta["secondsRawLocale"])
        self.assertNotEqual(seconds["value"], meta["secondsDoubledLocale"])
        self.assertEqual(meta["secondsScaledUtcYear"], 2023)
        self.assertEqual(meta["secondsRawUtcYear"], 1970)
        self.assertGreater(meta["secondsDoubledUtcYear"], 50000)
        self.assertIn("2023", seconds["value"])
        self.assertNotIn(str(meta["secondsDoubledUtcYear"]), seconds["value"])
        self.assertNotEqual(naive["value"], "-")
        self.assertEqual(payload["zulu"]["value"], meta["zuluLocale"])
        self.assertNotEqual(payload["zulu"]["value"], "-")
        self.assertEqual(payload["offset"]["value"], meta["offsetLocale"])
        self.assertNotEqual(payload["offset"]["value"], "-")

    def _run_node(self, node, fn_source, tz_name):
        env = os.environ.copy()
        env["TZ"] = tz_name
        env["LANG"] = "en_US.UTF-8"
        env["LC_ALL"] = "en_US.UTF-8"
        completed = subprocess.run(
            [node, "-e", _NODE_DRIVER],
            input=json.dumps({"fn": fn_source}),
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)


if __name__ == "__main__":
    unittest.main()
