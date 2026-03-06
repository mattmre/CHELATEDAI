import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
EMULATION_DIR = os.path.join(POC_DIR, "emulation")

for candidate in (POC_DIR, EMULATION_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from payload_contract import (  # noqa: E402
    SECTOR_SIZE,
    TRIGGER_SECTOR_LBA,
    build_trigger_sector_payload,
    compute_payload_result,
)
from usb_host_inference import read_inference_from_drive  # noqa: E402
from validate_emulation_path import validate_emulation_path  # noqa: E402
from virtual_controller import (  # noqa: E402
    FLASH_IMAGE_NAME,
    FLASH_IMAGE_PATH,
    VirtualComputationalStorageCore,
)


class TestComputationalStorageEmulation(unittest.TestCase):
    def test_virtual_controller_reads_trigger_sector_only(self):
        controller = VirtualComputationalStorageCore(size=(TRIGGER_SECTOR_LBA + 2) * SECTOR_SIZE)

        self.assertIn(FLASH_IMAGE_NAME, controller.readdir_entries("/"))
        empty_read = controller.read(FLASH_IMAGE_PATH, SECTOR_SIZE, 0)
        trigger_read = controller.read(
            FLASH_IMAGE_PATH,
            SECTOR_SIZE,
            TRIGGER_SECTOR_LBA * SECTOR_SIZE,
        )

        self.assertEqual(empty_read, b"\x00" * SECTOR_SIZE)
        self.assertEqual(trigger_read, build_trigger_sector_payload())

    def test_materialized_image_is_host_readable(self):
        controller = VirtualComputationalStorageCore(size=(TRIGGER_SECTOR_LBA + 1) * SECTOR_SIZE)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "flash.img"
            controller.materialize_file_image(image_path, total_size=(TRIGGER_SECTOR_LBA + 1) * SECTOR_SIZE)
            observed = json.loads(read_inference_from_drive(str(image_path), sector_start=TRIGGER_SECTOR_LBA))

        self.assertEqual(observed, compute_payload_result())

    def test_validate_emulation_path_matches_expected_payload(self):
        result = validate_emulation_path()
        self.assertEqual(result["observed"], compute_payload_result())


if __name__ == "__main__":
    unittest.main()
