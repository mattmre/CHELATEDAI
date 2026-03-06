import json
import os
import sys
import tempfile
import unittest

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from payload_contract import (  # noqa: E402
    SECTOR_SIZE,
    TRIGGER_SECTOR_LBA,
    build_trigger_sector_payload,
    compute_payload_result,
    overlaps_trigger_sector,
    read_virtual_disk,
)
from usb_host_inference import decode_inference_bytes, read_inference_from_drive  # noqa: E402


class TestPayloadContract(unittest.TestCase):
    def test_payload_result_matches_expected_block_graph_values(self):
        result = compute_payload_result()

        self.assertEqual(result["blocks_processed"], 2)
        self.assertEqual(result["predicted_class"], 0)
        self.assertEqual(result["sector_lba"], TRIGGER_SECTOR_LBA)
        self.assertEqual(result["logits"], [9.375, -3.6875])

    def test_trigger_sector_payload_is_utf8_json(self):
        payload = build_trigger_sector_payload()
        decoded = json.loads(decode_inference_bytes(payload))

        self.assertEqual(decoded, compute_payload_result())
        self.assertEqual(len(payload), SECTOR_SIZE)

    def test_virtual_disk_only_populates_the_trigger_sector(self):
        payload = build_trigger_sector_payload()
        disk_size = (TRIGGER_SECTOR_LBA + 2) * SECTOR_SIZE

        self.assertFalse(overlaps_trigger_sector(0, SECTOR_SIZE))
        self.assertTrue(overlaps_trigger_sector(TRIGGER_SECTOR_LBA * SECTOR_SIZE, SECTOR_SIZE))

        empty_read = read_virtual_disk(
            size=SECTOR_SIZE,
            offset=0,
            total_size=disk_size,
            trigger_sector_payload=payload,
        )
        trigger_read = read_virtual_disk(
            size=SECTOR_SIZE,
            offset=TRIGGER_SECTOR_LBA * SECTOR_SIZE,
            total_size=disk_size,
            trigger_sector_payload=payload,
        )

        self.assertEqual(empty_read, b"\x00" * SECTOR_SIZE)
        self.assertEqual(trigger_read, payload)

    def test_host_reader_decodes_trigger_sector_from_file_path(self):
        payload = build_trigger_sector_payload()
        disk_size = (TRIGGER_SECTOR_LBA + 1) * SECTOR_SIZE

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"\x00" * disk_size)
            temp_file.seek(TRIGGER_SECTOR_LBA * SECTOR_SIZE)
            temp_file.write(payload)
            temp_path = temp_file.name

        try:
            result = json.loads(read_inference_from_drive(temp_path))
        finally:
            os.remove(temp_path)

        self.assertEqual(result, compute_payload_result())


if __name__ == "__main__":
    unittest.main()
