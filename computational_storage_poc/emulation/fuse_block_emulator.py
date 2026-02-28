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
        self.sector_size = 512
        print(f"[INIT] Standing up Virtual FUSE Microcontroller ({self.size // (1024*1024)} MB)")
        
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
            sector = offset // self.sector_size
            
            # The Magic Sector 100 Interception!
            if sector == 100:
                print(f"\n[FUSE/OS] 🚨 INTERCEPTED OS READ COMMAND AT SECTOR 100!")
                print(f"[SILICON] 🧠 Computing Neural Network Graph localized on simulated 'Flash'...")
                
                # Mock result showing what the physical RP2040 firmware produces
                inference_res = b"CHELATEDAI HARDWARE INFERENCE: GRAPH EXECUTED ON VIRTUAL BLOCK DEVICE OVER FUSE"
                
                # Pad out to requested OS block size (typically 512 or 4096 bytes)
                if len(inference_res) < size:
                    padded = inference_res + (b'\x00' * (size - len(inference_res)))
                else:
                    padded = inference_res[:size]
                    
                print("[SILICON] 📤 Returning inference data directly to OS caller...\n")
                return padded
                
            # For all other reads (like the OS trying to find a partition table), return null bytes
            return b'\x00' * size
            
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
        
    print(f"\n=======================================================")
    print(f" 🔌 VIRTUAL COMPUTATIONAL STORAGE CONTROLLER ACTIVE ")
    print(f"=======================================================")
    print(f"Mounted virtual silicon block file to: {mountpoint}/flash.img")
    print(f"Waiting for OS SCSI Read Interceptions...\n")
    
    # Run the FUSE filesystem synchronously in the foreground
    FUSE(VirtualComputationalStorage(), mountpoint, nothreads=True, foreground=True)
