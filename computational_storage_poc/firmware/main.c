#include <stdbool.h>
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
#define TOY_INPUT_DIM 4
#define TOY_HIDDEN_DIM 3
#define TOY_OUTPUT_DIM 2

static const float kToyInput[TOY_INPUT_DIM] = {1.0f, -2.0f, 3.0f, 0.5f};
static const float kToyLayer1[TOY_INPUT_DIM][TOY_HIDDEN_DIM] = {
    {1.0f, 0.0f, 0.5f},
    {-1.0f, 2.0f, 0.0f},
    {0.5f, 0.5f, 1.0f},
    {0.0f, 1.0f, -0.5f},
};
static const float kToyLayer2[TOY_HIDDEN_DIM][TOY_OUTPUT_DIM] = {
    {1.0f, -1.0f},
    {0.5f, 2.0f},
    {1.5f, 0.25f},
};

// In a real implementation, this would point directly to the RP2040's attached QSPI Flash.
// For this POC, we use a tiny block of RAM to simulate the formatting.
uint8_t msc_disk[DISK_BLOCK_COUNT][SECTOR_SIZE];

static void compute_toy_graph(float output[TOY_OUTPUT_DIM]) {
    float hidden[TOY_HIDDEN_DIM] = {0};

    for (uint8_t hidden_idx = 0; hidden_idx < TOY_HIDDEN_DIM; hidden_idx++) {
        for (uint8_t input_idx = 0; input_idx < TOY_INPUT_DIM; input_idx++) {
            hidden[hidden_idx] += kToyInput[input_idx] * kToyLayer1[input_idx][hidden_idx];
        }
        if (hidden[hidden_idx] < 0.0f) {
            hidden[hidden_idx] = 0.0f;
        }
    }

    for (uint8_t output_idx = 0; output_idx < TOY_OUTPUT_DIM; output_idx++) {
        output[output_idx] = 0.0f;
        for (uint8_t hidden_idx = 0; hidden_idx < TOY_HIDDEN_DIM; hidden_idx++) {
            output[output_idx] += hidden[hidden_idx] * kToyLayer2[hidden_idx][output_idx];
        }
    }
}

static void internal_graph_traversal(uint8_t sector_buffer[SECTOR_SIZE]) {
    float logits[TOY_OUTPUT_DIM] = {0};
    uint8_t predicted_class = 0;

    compute_toy_graph(logits);
    if (logits[1] > logits[0]) {
        predicted_class = 1;
    }

    memset(sector_buffer, 0, SECTOR_SIZE);
    snprintf(
        (char*) sector_buffer,
        SECTOR_SIZE,
        "{\"blocks_processed\":2,\"input\":[%.1f,%.1f,%.1f,%.1f],\"logits\":[%.4f,%.4f],\"predicted_class\":%u,\"sector_lba\":%u}",
        kToyInput[0],
        kToyInput[1],
        kToyInput[2],
        kToyInput[3],
        logits[0],
        logits[1],
        predicted_class,
        TRIGGER_SECTOR_LBA
    );
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
    if (lba >= DISK_BLOCK_COUNT || offset + bufsize > SECTOR_SIZE) {
        return -1;
    }

    // ----- [THE MAGIC HAPPENS HERE] -----
    if (lba == TRIGGER_SECTOR_LBA) {
        uint8_t trigger_sector[SECTOR_SIZE];

        // 1. Host attempted to read Sector 100.
        // 2. We intercept the read.
        // 3. We trigger the internal Vector-Table hardware inference.
        internal_graph_traversal(trigger_sector);
        
        // 4. We return the inference result as if it were a standard file sector.
        memcpy(buffer, trigger_sector + offset, bufsize);
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
    if (lba >= DISK_BLOCK_COUNT || offset + bufsize > SECTOR_SIZE) {
        return -1;
    }
    
    // Similarly, we could intercept WRITEs here to "load" the initial prompt/activations
    // before triggering the inference read.
    
    memcpy(msc_disk[lba] + offset, buffer, bufsize);
    return bufsize;
}

void tud_msc_inquiry_cb(uint8_t lun, uint8_t vendor_id[8], uint8_t product_id[16], uint8_t product_rev[4]) {
    (void) lun;
    memcpy(vendor_id, "CHELATE ", 8);
    memcpy(product_id, "CompStoragePOC ", 16);
    memcpy(product_rev, "0001", 4);
}

bool tud_msc_test_unit_ready_cb(uint8_t lun) {
    (void) lun;
    return true;
}

void tud_msc_capacity_cb(uint8_t lun, uint32_t* block_count, uint16_t* block_size) {
    (void) lun;
    *block_count = DISK_BLOCK_COUNT;
    *block_size = SECTOR_SIZE;
}

bool tud_msc_start_stop_cb(uint8_t lun, uint8_t power_condition, bool start, bool load_eject) {
    (void) lun;
    (void) power_condition;
    (void) start;
    (void) load_eject;
    return true;
}

bool tud_msc_is_writable_cb(uint8_t lun) {
    (void) lun;
    return true;
}

int32_t tud_msc_scsi_cb(uint8_t lun, uint8_t const scsi_cmd[16], void* buffer, uint16_t bufsize) {
    (void) scsi_cmd;
    (void) buffer;
    (void) bufsize;
    tud_msc_set_sense(lun, SCSI_SENSE_ILLEGAL_REQUEST, 0x20, 0x00);
    return -1;
}
