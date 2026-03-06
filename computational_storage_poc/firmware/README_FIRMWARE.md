# Computational Storage Firmware (Phase 3)

This directory contains the prototype C code for the **Raspberry Pi Pico (RP2040)** running the **TinyUSB** stack.

## Theory of Operation
Traditional SSDs run proprietary controllers that prevent custom execution. By using an RP2040 configured as a USB Mass Storage device, we can act as the "SSD Controller". 

1. The Host OS mounts the USB Drive.
2. The Host writes the "input activations" (the context prompt) to a specific sector.
3. The Host requests to read the "output" sector.
4. The `tud_msc_read10_cb` callback in `main.c` intercepts this specific SCSI READ request.
5. Instead of returning a static demo string, the firmware computes a deterministic `4 -> 3 -> 2` block-graph result that mirrors the validated software transport contract.
6. It returns a JSON payload from sector `100`, including the toy input vector, computed logits, predicted class, and block count.

To compile this, you'll need the [Pico C/C++ SDK](https://github.com/raspberrypi/pico-sdk) and TinyUSB installed on your build machine.

## Current Scope

This firmware now validates three concrete things:

- TinyUSB descriptors and MSC callbacks compile cleanly in CI,
- sector `100` returns a computed payload rather than a hard-coded placeholder, and
- the USB transport path stays aligned with the emulation contract used by `usb_host_inference.py`.

The full digits-model execution path remains software-validated in the foundation branch. The RP2040 firmware is still an experimental transport/control-plane step, not a claim that the full trained digits model already fits and executes on-device.

The scope lock for that statement is recorded in [docs/computational-storage-transport-scope-decision.md](../../docs/computational-storage-transport-scope-decision.md). Until the promotion gates there are satisfied, describe this path as a deterministic transport proof rather than on-device digits inference.
