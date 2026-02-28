#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "bsp/board.h"
#include "tusb.h"

/*
 * Computational Storage Proof of Concept (Firmware)
 * 
 * This firmware uses the TinyUSB stack on an RP2040 to mount as a USB Mass Storage Class (MSC) device.
 * It demonstrates how to intercept standard SCSI reads/writes and redirect them to run localized 
 * "Look Up Table" graph traversal, bypassing the host CPU.
 */

#define TRIGGER_SECTOR_LBA 100
#define SECTOR_SIZE 512
#define DISK_BLOCK_COUNT 200

// In a real implementation, this would point directly to the RP2040's attached QSPI Flash.
// For this POC, we use a tiny block of RAM to simulate the formatting.
uint8_t msc_disk[DISK_BLOCK_COUNT][SECTOR_SIZE];

// Mock Inference implementation representing the internal hardware calculation
void internal_graph_traversal(void* output_buffer) {
    // In reality, this would read the 64-bit guide nodes and traverse the Matrix BLOCKS
    // stored in NAND flash without ever sending data over the USB/PCIe bus.
    const char* mock_result = "COMPUTATIONAL_STORAGE_RESULT: [0.12, 0.98, 0.04, 0.00] (Calculated internally)";
    memset(output_buffer, 0, SECTOR_SIZE);
    memcpy(output_buffer, mock_result, strlen(mock_result));
}

int main(void) {
    board_init();
    tusb_init();

    // Initialize the simulated format
    memset(msc_disk, 0, sizeof(msc_disk));
    strcpy((char*)msc_disk[0], "FAT16_OR_SIMILAR_HEADER");

    while (1) {
        tud_task(); // Yield to TinyUSB device task
    }
    return 0;
}

//--------------------------------------------------------------------+
// Mass Storage Class (MSC) Callbacks
//--------------------------------------------------------------------+

// Invoked when received SCSI READ10 command
// - lun: Logical unit number
// - lba: Logical block address
// - offset: Offset in bytes from LBA
// - buffer: Point to buffer for host to read
// - bufsize: Buffer size in bytes
int32_t tud_msc_read10_cb(uint8_t lun, uint32_t lba, uint32_t offset, void* buffer, uint32_t bufsize) {
    (void) lun;

    // ----- [THE MAGIC HAPPENS HERE] -----
    if (lba == TRIGGER_SECTOR_LBA) {
        // 1. Host attempted to read Sector 100.
        // 2. We intercept the read.
        // 3. We trigger the internal Vector-Table hardware inference.
        internal_graph_traversal(buffer);
        
        // 4. We return the inference result as if it were a standard file sector.
        return bufsize;
    }
    // ------------------------------------

    // Normal disk read (just read from mocked flash array)
    memcpy(buffer, msc_disk[lba] + offset, bufsize);
    return bufsize;
}

// Invoked when received SCSI WRITE10 command
int32_t tud_msc_write10_cb(uint8_t lun, uint32_t lba, uint32_t offset, uint8_t* buffer, uint32_t bufsize) {
    (void) lun;
    
    // Similarly, we could intercept WRITEs here to "load" the initial prompt/activations
    // before triggering the inference read.
    
    memcpy(msc_disk[lba] + offset, buffer, bufsize);
    return bufsize;
}

// (Other required MSC callbacks like Inquiry, Test Unit Ready, and Capacity are omitted for brevity in POC)
