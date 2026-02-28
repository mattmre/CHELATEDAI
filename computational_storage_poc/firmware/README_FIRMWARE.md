# Computational Storage Firmware (Phase 3)

This directory contains the prototype C code for the **Raspberry Pi Pico (RP2040)** running the **TinyUSB** stack.

## Theory of Operation
Traditional SSDs run proprietary controllers that prevent custom execution. By using an RP2040 configured as a USB Mass Storage device, we can act as the "SSD Controller". 

1. The Host OS mounts the USB Drive.
2. The Host writes the "input activations" (the context prompt) to a specific sector.
3. The Host requests to read the "output" sector.
4. The `tud_msc_read10_cb` callback in `main.c` intercepts this specific SCSI READ request.
5. Instead of returning raw flash memory, it traverses the `model.bin` graph stored in the Pico's memory, running the entire AI model locally.
6. It returns the final text token as the payload of the Sector Read.

To compile this, you'll need the [Pico C/C++ SDK](https://github.com/raspberrypi/pico-sdk) and TinyUSB installed on your build machine.
