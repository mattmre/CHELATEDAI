import os
import sys

def read_inference_from_drive(drive_id, sector_start=100, num_sectors=1):
    """
    Reads a raw sector from a physical drive on Windows.
    For the RP2040 Computational Storage Firmware, reading sector 100 
    triggers the onboard neural network inference.
    """
    # Path branching for OS agnostic raw drive reading
    if os.name == 'nt':
        # Windows physical drive path format
        drive_path = f"\\\\.\\PhysicalDrive{drive_id}"
    else:
        # Linux / Docker path formats
        if str(drive_id).isdigit():
            drive_path = f"/dev/loop{drive_id}"
        else:
            drive_path = str(drive_id) # Direct path like /mnt/virtual_usb/flash.img

    print(f"Attempting to open {drive_path} in raw mode...")
    
    try:
        # Requires Administrator privileges on Windows
        with open(drive_path, 'rb') as f:
            # Storage Sectors are universally 512 bytes. 
            # We seek to the byte offset of sector 100
            f.seek(sector_start * 512)
            
            # The host OS issues a SCSI READ(10) block request here!
            # The Pico's custom firmware intercepts this and calculates the graph.
            print(">> Fetching data from silicon...")
            data = f.read(512 * num_sectors)
            
            # Parse result (the firmware writes a UTF-8 string padded with NULL bytes)
            try:
                result = data.decode('utf-8').rstrip('\x00')
                if not result.strip():
                    return "[No ASCII data returned. Is this the right drive?]"
                return result
            except UnicodeDecodeError:
                return f"[Raw Hex Dump]: {data[:64].hex()}..."
                
    except PermissionError:
        print("\n[ERROR] Administrator privileges are required to read raw physical drives on Windows.")
        print("Please restart your terminal/IDE as Administrator and try again.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n[ERROR] Could not find '{drive_path}'.")
        print("Ensure the Pico is plugged in and you selected the correct Drive ID.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to read drive: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("============================================================")
    print("  CHELATEDAI Computational Storage - USB Host Verifier")
    print("============================================================")
    print("This script reads raw sectors directly from a USB Mass Storage device")
    print("to trigger the onboard AI inference on the RP2040 firmware.\n")
    
    try:
        if len(sys.argv) > 1:
            drive_id = sys.argv[1]
        else:
            print("To find your USB Drive ID, you can run this command in PowerShell:")
            print("  Get-Disk | Select-Object Number, FriendlyName")
            print("On Linux/Docker, you can pass a path like: /mnt/virtual_usb/flash.img\n")
            
            drive_input = input("Enter the Drive Number OR Path: ")
            drive_id = drive_input.strip()
            
        print(f"\nSending read request to Sector 100 on {drive_id}...")
        result = read_inference_from_drive(drive_id)
        
        print("\n" + "="*50)
        print(" 💡 INFERENCE RESULT FROM STORAGE CONTROLLER:")
        print("="*50)
        print(f"  {result}")
        print("="*50 + "\n")
        print("Notice how no standard math was run on your host CPU!")
        
    except ValueError:
        print("Please enter a valid integer for the drive number.")
        sys.exit(1)
