import json

import numpy as np

from block_graph import BLOCK_SIZE, build_graph_payload, run_block_graph

SECTOR_SIZE = 512
TRIGGER_SECTOR_LBA = 100

TOY_INPUT_VECTOR = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float32)
TOY_LAYER_1 = np.array(
    [
        [1.0, 0.0, 0.5],
        [-1.0, 2.0, 0.0],
        [0.5, 0.5, 1.0],
        [0.0, 1.0, -0.5],
    ],
    dtype=np.float32,
)
TOY_LAYER_2 = np.array(
    [
        [1.0, -1.0],
        [0.5, 2.0],
        [1.5, 0.25],
    ],
    dtype=np.float32,
)


def _build_input_activations() -> np.ndarray:
    activations = np.zeros((1, BLOCK_SIZE), dtype=np.float32)
    activations[0, : len(TOY_INPUT_VECTOR)] = TOY_INPUT_VECTOR
    return activations


def compute_payload_result() -> dict:
    flash_memory = build_graph_payload([TOY_LAYER_1, TOY_LAYER_2])
    logits, blocks_processed = run_block_graph(
        flash_memory,
        _build_input_activations(),
        hidden_activation="relu",
    )
    trimmed_logits = logits[0, : TOY_LAYER_2.shape[1]]
    return {
        "blocks_processed": int(blocks_processed),
        "input": [round(float(value), 5) for value in TOY_INPUT_VECTOR.tolist()],
        "logits": [round(float(value), 5) for value in trimmed_logits.tolist()],
        "predicted_class": int(np.argmax(trimmed_logits)),
        "sector_lba": TRIGGER_SECTOR_LBA,
    }


def format_payload_result() -> str:
    return json.dumps(compute_payload_result(), sort_keys=True)


def build_trigger_sector_payload() -> bytes:
    encoded = format_payload_result().encode("utf-8")
    if len(encoded) > SECTOR_SIZE:
        raise ValueError(f"Trigger-sector payload exceeds {SECTOR_SIZE} bytes")
    return encoded.ljust(SECTOR_SIZE, b"\x00")


def overlaps_trigger_sector(
    offset: int,
    size: int,
    sector_size: int = SECTOR_SIZE,
    trigger_sector_lba: int = TRIGGER_SECTOR_LBA,
) -> bool:
    if size <= 0:
        return False
    read_start = offset
    read_end = offset + size
    sector_start = trigger_sector_lba * sector_size
    sector_end = sector_start + sector_size
    return read_start < sector_end and read_end > sector_start


def read_virtual_disk(
    *,
    size: int,
    offset: int,
    total_size: int,
    trigger_sector_payload: bytes,
    sector_size: int = SECTOR_SIZE,
    trigger_sector_lba: int = TRIGGER_SECTOR_LBA,
) -> bytes:
    if size < 0:
        raise ValueError("Read size must be non-negative")
    if offset < 0:
        raise ValueError("Read offset must be non-negative")
    if offset >= total_size or size == 0:
        return b""

    clamped_size = min(size, total_size - offset)
    response = bytearray(clamped_size)

    if not overlaps_trigger_sector(offset, clamped_size, sector_size, trigger_sector_lba):
        return bytes(response)

    sector_start = trigger_sector_lba * sector_size
    overlap_start = max(offset, sector_start)
    overlap_end = min(offset + clamped_size, sector_start + sector_size)
    source_start = overlap_start - sector_start
    dest_start = overlap_start - offset
    overlap_length = overlap_end - overlap_start
    response[dest_start : dest_start + overlap_length] = trigger_sector_payload[
        source_start : source_start + overlap_length
    ]
    return bytes(response)
