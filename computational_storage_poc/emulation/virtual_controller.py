from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Callable

from payload_contract import (
    SECTOR_SIZE,
    TRIGGER_SECTOR_LBA,
    build_trigger_sector_payload,
    format_payload_result,
    overlaps_trigger_sector,
    read_virtual_disk,
)

DEFAULT_FLASH_SIZE = 1024 * 1024 * 100
FLASH_IMAGE_NAME = "flash.img"
FLASH_IMAGE_PATH = f"/{FLASH_IMAGE_NAME}"


class VirtualComputationalStorageCore:
    def __init__(
        self,
        size: int = DEFAULT_FLASH_SIZE,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.size = size
        self.sector_size = SECTOR_SIZE
        self.trigger_sector_payload = build_trigger_sector_payload()
        self.result_summary = format_payload_result()
        self._logger = logger

    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger(message)

    def getattr_entry(self, path: str) -> dict[str, int]:
        if path == "/":
            return {"st_mode": stat.S_IFDIR | 0o755, "st_nlink": 2}
        if path == FLASH_IMAGE_PATH:
            return {"st_mode": stat.S_IFREG | 0o666, "st_nlink": 1, "st_size": self.size}
        raise FileNotFoundError(path)

    def readdir_entries(self, path: str) -> tuple[str, str, str]:
        if path != "/":
            raise FileNotFoundError(path)
        return ".", "..", FLASH_IMAGE_NAME

    def read(self, path: str, size: int, offset: int) -> bytes:
        if path != FLASH_IMAGE_PATH:
            raise FileNotFoundError(path)

        if overlaps_trigger_sector(offset, size, self.sector_size, TRIGGER_SECTOR_LBA):
            self._log("\n[FUSE/OS] Intercepted OS read command at sector 100")
            self._log("[SILICON] Executing deterministic 4->3->2 block graph")
            self._log(f"[SILICON] Returning {self.result_summary}\n")

        return read_virtual_disk(
            size=size,
            offset=offset,
            total_size=self.size,
            trigger_sector_payload=self.trigger_sector_payload,
            sector_size=self.sector_size,
            trigger_sector_lba=TRIGGER_SECTOR_LBA,
        )

    def write(self, path: str, data: bytes, offset: int) -> int:
        if path != FLASH_IMAGE_PATH:
            raise FileNotFoundError(path)
        return len(data)

    def materialize_file_image(self, output_path: str | os.PathLike[str], *, total_size: int | None = None) -> Path:
        effective_size = total_size or self.size
        minimum_size = (TRIGGER_SECTOR_LBA + 1) * self.sector_size
        if effective_size < minimum_size:
            raise ValueError(
                f"File image size must be at least {minimum_size} bytes to include trigger sector"
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            handle.truncate(effective_size)
            handle.seek(TRIGGER_SECTOR_LBA * self.sector_size)
            handle.write(self.trigger_sector_payload)
        return output
