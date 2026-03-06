from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EMULATION_DIR = os.path.dirname(os.path.abspath(__file__))
if EMULATION_DIR not in sys.path:
    sys.path.insert(0, EMULATION_DIR)

from payload_contract import SECTOR_SIZE, TRIGGER_SECTOR_LBA, compute_payload_result  # noqa: E402
from usb_host_inference import read_inference_from_drive  # noqa: E402
from virtual_controller import VirtualComputationalStorageCore  # noqa: E402


def validate_emulation_path() -> dict:
    image_size = (TRIGGER_SECTOR_LBA + 1) * SECTOR_SIZE
    controller = VirtualComputationalStorageCore(size=image_size)
    expected = compute_payload_result()

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        image_path = Path(temp_file.name)

    try:
        controller.materialize_file_image(image_path, total_size=image_size)
        observed = json.loads(read_inference_from_drive(str(image_path), sector_start=TRIGGER_SECTOR_LBA))
    finally:
        image_path.unlink(missing_ok=True)

    if observed != expected:
        raise AssertionError(
            "Emulation path diverged from the deterministic payload contract: "
            f"observed={observed}, expected={expected}"
        )

    return {
        "image_size": image_size,
        "observed": observed,
        "expected": expected,
    }


def main() -> int:
    result = validate_emulation_path()
    print("Emulation path validation passed.")
    print(json.dumps(result["observed"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
