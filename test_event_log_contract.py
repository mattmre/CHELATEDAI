"""Default event-log path and mixed timestamp sorting."""

import unittest
from pathlib import Path

import dashboard_server
from chelation_logger import ChelationLogger


class TestEventLogContract(unittest.TestCase):
    def test_default_log_path_matches_dashboard_events_file(self):
        logger = ChelationLogger(console_level="ERROR")
        self.assertEqual(logger.log_path, Path("chelation_events.jsonl"))
        self.assertEqual(dashboard_server.LOG_FILE_PATH, "chelation_events.jsonl")
        explicit = ChelationLogger(log_path=Path("custom.jsonl"), console_level="ERROR")
        self.assertEqual(explicit.log_path, Path("custom.jsonl"))

    def test_filter_iso_timestamp_does_not_crash_when_timestamp_missing(self):
        rows = [
            {"timestamp": "2026-09-24T00:00:00", "query_snippet": "a"},
            {"message": "no ts"},
        ]
        result = dashboard_server.filter_events(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["timestamp"], "2026-09-24T00:00:00")
        self.assertEqual(result[0]["query_snippet"], "a")
        self.assertEqual(result[1]["message"], "no ts")


if __name__ == "__main__":
    unittest.main()
