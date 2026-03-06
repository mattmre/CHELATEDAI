# Computational Storage AI Proof of Concept

This project implements a testbed for the **Computational Storage (SSD Array) AI Inference** architecture.
The premise replaces matrix multiplication entirely with direct memory routing using memory tables, mapping it directly onto NAND flash memory.

## What Is Verified

- Exact block-graph traversal from offset `0x0`, including the first block.
- Hidden-layer `ReLU` application on non-terminal blocks so the storage path matches the trained MLP.
- Deterministic host-vs-storage output parity and theoretical latency comparisons.
- Real-data round-trip validation on the handwritten digits dataset with explicit accuracy floors.
- Deterministic trigger-sector payload generation for the USB/emulation transport path.
- Host-reader decoding and virtual-disk interception tests for sector `100`.

## Binary Format

The `compiler.py` packs neural network matrices into continuous graph nodes with the following rigid format:

| Offset | Length (Bytes) | Description |
|--------|----------------|-------------|
| 0x0 | 524,288 | **Matrix Payload** (512x512 matrix stored in FP16 format). Unused dimensions are zero-padded. |
| 0x80000 | 8 | **64-bit Guide Node**: Little-endian unsigned integer representing the byte offset of the next node/block in the neural network graph. A value of 0 indicates the end of inference. |

**Total Block Size**: 524,296 bytes (0x80008).

### Usage
Generate a sample binary payload:
```bash
python compiler.py
```
This will produce a `model.bin` file containing the connected graph.

Run the full proof-of-concept validation:

```bash
python run_all_tests.py
```

The validation suite now fails if:

- host and storage outputs diverge,
- storage accuracy falls below the configured floor, or
- the speculative execution demo regresses against its sequential baseline.

## USB / Emulation Payload Contract

The USB firmware and FUSE emulator now share a deterministic transport contract:

- sector `100` returns a JSON payload rather than a hard-coded demo string,
- the payload includes the deterministic toy input vector plus the computed `4 -> 3 -> 2` block-graph result, and
- the host-side reader is tested against both the raw sector bytes and a virtual-disk file path.

Run the transport-layer tests with:

```bash
python -m unittest test_computational_storage_payload.py -v
```

The RP2040 path is still an experimental payload track. The full digits model is validated in software today; the firmware currently uses the deterministic toy graph above to prove the USB interception path and descriptor plumbing without claiming full on-device digits inference yet.
