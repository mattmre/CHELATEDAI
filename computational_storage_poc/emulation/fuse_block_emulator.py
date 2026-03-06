#!/usr/bin/env python3
import os
import sys
import errno
import stat

try:
    from fuse import FUSE, Operations, FuseOSError
except ImportError:
    print("Error: fusepy is not installed. Run 'pip install fusepy'")
    sys.exit(1)

# Ensure flush is immediate on print for Docker logging
import functools
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from payload_contract import (  # noqa: E402
    SECTOR_SIZE,
    TRIGGER_SECTOR_LBA,
    build_trigger_sector_payload,
    format_payload_result,
    overlaps_trigger_sector,
    read_virtual_disk,
)

class VirtualComputationalStorage(Operations):
    """
    Creates a FUSE (Filesystem in Userspace) virtual file that 
    intercepts raw OS block reads.
    
    This purely emulates the TinyUSB hardware interception without needing
    actual hardware or risky kernel modifications.
    """
    def __init__(self, size=1024*1024*100): 
        # Present as a 100MB dummy flash drive
        self.size = size
        self.sector_size = SECTOR_SIZE
        self.trigger_sector_payload = build_trigger_sector_payload()
        self.result_summary = format_payload_result()
        print(f"[INIT] Standing up Virtual FUSE Microcontroller ({self.size // (1024*1024)} MB)")
        print(f"[INIT] Trigger sector {TRIGGER_SECTOR_LBA} returns {self.result_summary}")
        
    def getattr(self, path, fh=None):
        if path == '/':
            return {'st_mode': (stat.S_IFDIR | 0o755), 'st_nlink': 2}
        elif path == '/flash.img':
            # Present as a large binary file
            return {'st_mode': (stat.S_IFREG | 0o666), 'st_nlink': 1, 'st_size': self.size}
        else:
            raise FuseOSError(errno.ENOENT)

    def readdir(self, path, fh):
        if path == '/':
            yield '.'
            yield '..'
            yield 'flash.img'

    def read(self, path, size, offset, fh):
        if path == '/flash.img':
            if overlaps_trigger_sector(offset, size, self.sector_size, TRIGGER_SECTOR_LBA):
                print("\n[FUSE/OS] 🚨 INTERCEPTED OS READ COMMAND AT SECTOR 100!")
                print("[SILICON] 🧠 Executing deterministic 4->3->2 block graph inside the virtual controller...")
                print(f"[SILICON] 📤 Returning {self.result_summary}\n")

            return read_virtual_disk(
                size=size,
                offset=offset,
                total_size=self.size,
                trigger_sector_payload=self.trigger_sector_payload,
                sector_size=self.sector_size,
                trigger_sector_lba=TRIGGER_SECTOR_LBA,
            )
            
        raise FuseOSError(errno.ENOENT)
        
    # We fake answering OS writes so mounting scripts don't crash
    def write(self, path, data, offset, fh):
        return len(data)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fuse_block_emulator.py <mountpoint>")
        sys.exit(1)
        
    mountpoint = sys.argv[1]
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint, exist_ok=True)
        
    print("\n=======================================================")
    print(" 🔌 VIRTUAL COMPUTATIONAL STORAGE CONTROLLER ACTIVE ")
    print("=======================================================")
    print(f"Mounted virtual silicon block file to: {mountpoint}/flash.img")
    print("Waiting for OS SCSI Read Interceptions against the validated trigger sector...\n")
    
    # Run the FUSE filesystem synchronously in the foreground
    FUSE(VirtualComputationalStorage(), mountpoint, nothreads=True, foreground=True)
