# Computational Storage AI Proof of Concept

This project implements a testbed for the **Computational Storage (SSD Array) AI Inference** architecture.
The premise replaces matrix multiplication entirely with direct memory routing using memory tables, mapping it directly onto NAND flash memory.

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
