import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from capture_hardware_evidence import (  # noqa: E402
    capture_hardware_evidence,
    render_markdown_report,
    write_evidence_report,
)
from payload_contract import SECTOR_SIZE, TRIGGER_SECTOR_LBA, build_trigger_sector_payload  # noqa: E402
from usb_host_inference import resolve_drive_path  # noqa: E402


class TestHardwareEvidenceCapture(unittest.TestCase):
    def test_resolve_drive_path_preserves_windows_device_path(self):
        with patch("usb_host_inference.os.name", "nt"), patch(
            "usb_host_inference.os.path.exists",
            return_value=False,
        ):
            self.assertEqual(resolve_drive_path(r"\\.\PhysicalDrive2"), r"\\.\PhysicalDrive2")
            self.assertEqual(resolve_drive_path("2"), r"\\.\PhysicalDrive2")

    def test_capture_from_file_path_matches_expected_payload(self):
        payload = build_trigger_sector_payload()
        disk_size = (TRIGGER_SECTOR_LBA + 1) * SECTOR_SIZE

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"\x00" * disk_size)
            temp_file.seek(TRIGGER_SECTOR_LBA * SECTOR_SIZE)
            temp_file.write(payload)
            temp_path = temp_file.name

        try:
            evidence = capture_hardware_evidence(
                temp_path,
                notes="file-backed validation",
                metadata_fn=lambda target: {"target": target},
            )
        finally:
            os.remove(temp_path)

        self.assertTrue(evidence["matches_expected_payload"])
        self.assertEqual(evidence["target_kind"], "path")
        self.assertEqual(evidence["device_metadata"], {"target": temp_path})

    def test_capture_reports_mismatches_for_wrong_payload(self):
        observed = {
            "blocks_processed": 2,
            "input": [1.0, -2.0, 3.0, 0.5],
            "logits": [9.375, -3.6875],
            "predicted_class": 1,
            "sector_lba": TRIGGER_SECTOR_LBA,
        }
        evidence = capture_hardware_evidence(
            "mock-device",
            read_fn=lambda *args, **kwargs: json.dumps(observed),
            metadata_fn=lambda target: None,
        )

        self.assertFalse(evidence["matches_expected_payload"])
        self.assertEqual(len(evidence["mismatches"]), 1)
        self.assertEqual(evidence["mismatches"][0]["field"], "predicted_class")

        report = render_markdown_report(evidence)
        self.assertIn("FAIL", report)
        self.assertIn("predicted_class", report)

    def test_write_evidence_report_outputs_json_and_markdown(self):
        evidence = capture_hardware_evidence(
            "mock-device",
            read_fn=lambda *args, **kwargs: json.dumps(
                {
                    "blocks_processed": 2,
                    "input": [1.0, -2.0, 3.0, 0.5],
                    "logits": [9.375, -3.6875],
                    "predicted_class": 0,
                    "sector_lba": TRIGGER_SECTOR_LBA,
                }
            ),
            metadata_fn=lambda target: {"kind": "mock"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, md_path = write_evidence_report(
                evidence,
                output_dir=Path(temp_dir),
                stem="sample-evidence",
            )

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertIn("sample-evidence", json_path.name)
            self.assertIn("sample-evidence", md_path.name)
            self.assertTrue(json.loads(json_path.read_text(encoding="utf-8"))["matches_expected_payload"])


if __name__ == "__main__":
    unittest.main()
