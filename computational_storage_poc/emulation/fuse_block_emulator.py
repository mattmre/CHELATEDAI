#!/usr/bin/env python3
import os
import sys
import errno

try:
    from fuse import FUSE, Operations, FuseOSError
except ImportError:
    FUSE = None

    class Operations:  # type: ignore[no-redef]
        pass

    class FuseOSError(OSError):  # type: ignore[no-redef]
        pass

# Ensure flush is immediate on print for Docker logging
import functools
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EMULATION_DIR = os.path.dirname(os.path.abspath(__file__))
if EMULATION_DIR not in sys.path:
    sys.path.insert(0, EMULATION_DIR)

from virtual_controller import (  # noqa: E402
    DEFAULT_FLASH_SIZE,
    FLASH_IMAGE_PATH,
    VirtualComputationalStorageCore,
)

class VirtualComputationalStorage(Operations):
    """
    Creates a FUSE (Filesystem in Userspace) virtual file that 
    intercepts raw OS block reads.
    
    This purely emulates the TinyUSB hardware interception without needing
    actual hardware or risky kernel modifications.
    """
    def __init__(self, size=DEFAULT_FLASH_SIZE):
        self.core = VirtualComputationalStorageCore(size=size, logger=print)
        self.size = size
        print(f"[INIT] Standing up Virtual FUSE Microcontroller ({self.size // (1024*1024)} MB)")
        print(f"[INIT] Trigger sector 100 returns {self.core.result_summary}")
        
    def getattr(self, path, fh=None):
        try:
            return self.core.getattr_entry(path)
        except FileNotFoundError:
            raise FuseOSError(errno.ENOENT)

    def readdir(self, path, fh):
        try:
            for entry in self.core.readdir_entries(path):
                yield entry
        except FileNotFoundError:
            raise FuseOSError(errno.ENOENT) from None

    def read(self, path, size, offset, fh):
        try:
            return self.core.read(path, size, offset)
        except FileNotFoundError:
            raise FuseOSError(errno.ENOENT) from None
        
    # We fake answering OS writes so mounting scripts don't crash
    def write(self, path, data, offset, fh):
        try:
            return self.core.write(path, data, offset)
        except FileNotFoundError:
            raise FuseOSError(errno.ENOENT) from None

if __name__ == '__main__':
    if FUSE is None:
        print("Error: fusepy is not installed. Run 'pip install fusepy'")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python fuse_block_emulator.py <mountpoint>")
        sys.exit(1)
        
    mountpoint = sys.argv[1]
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint, exist_ok=True)
        
    print("\n=======================================================")
    print(" 🔌 VIRTUAL COMPUTATIONAL STORAGE CONTROLLER ACTIVE ")
    print("=======================================================")
    print(f"Mounted virtual silicon block file to: {mountpoint}{FLASH_IMAGE_PATH}")
    print("Waiting for OS SCSI Read Interceptions against the validated trigger sector...\n")
    
    # Run the FUSE filesystem synchronously in the foreground
    FUSE(VirtualComputationalStorage(), mountpoint, nothreads=True, foreground=True)
