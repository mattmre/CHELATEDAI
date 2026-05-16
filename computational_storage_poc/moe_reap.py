from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from block_graph import apply_hidden_activation
from cpu_backends import CPUInferenceBackend, NumpyInt8DynamicBackend
from packed_graph import INT8_STORAGE_DTYPE, _quantize_matrix_to_int8
from _experimental import mark_experimental

# Research-stage / POC module. No production code path in this repo consumes it.
# EXPERIMENTAL is read by _experimental.mark_experimental — flipping it to
# False suppresses the import-time warning; a non-bool raises TypeError.
EXPERIMENTAL = True
mark_experimental(__name__, EXPERIMENTAL)

MOE_ARTIFACT_MAGIC = b"CSMOE1"
HEADER_LENGTH_BYTES = 4


@dataclass(frozen=True)
class ExpertBlockMetadata:
    rows: int
    cols: int
    matrix_offset: int
    matrix_nbytes: int
    storage_dtype: str
    scale: float


@dataclass(frozen=True)
class ExpertLayout:
    expert_id: int
    active: bool
    score: float
    blocks: list[ExpertBlockMetadata]


@dataclass(frozen=True)
class RoutedExpertResult:
    expert_id: int
    router_weight: float
    output: np.ndarray
    bytes_read: int


@dataclass(frozen=True)
class MoEInferenceResult:
    output: np.ndarray
    active_expert_ids: list[int]
    bytes_read: int
    experts_evaluated: int


def _pack_matrix(matrix: np.ndarray) -> tuple[bytes, float]:
    quantized, scale = _quantize_matrix_to_int8(matrix.astype(np.float32, copy=False))
    return quantized.tobytes(order="C"), scale


def _build_expert_blob(expert_matrices: list[np.ndarray]) -> tuple[list[dict[str, float | int | str]], bytes]:
    blocks: list[dict[str, float | int | str]] = []
    blob_parts: list[bytes] = []
    current_offset = 0

    for matrix in expert_matrices:
        rows, cols = matrix.shape
        packed, scale = _pack_matrix(matrix)
        blocks.append(
            {
                "rows": int(rows),
                "cols": int(cols),
                "matrix_offset": current_offset,
                "matrix_nbytes": len(packed),
                "storage_dtype": INT8_STORAGE_DTYPE,
                "scale": scale,
            }
        )
        blob_parts.append(packed)
        current_offset += len(packed)

    return blocks, b"".join(blob_parts)


def build_moe_reap_artifact(
    router_weights: np.ndarray,
    expert_matrices: list[list[np.ndarray]],
    *,
    expert_scores: list[float] | None = None,
    active_expert_ids: list[int] | None = None,
) -> bytes:
    router_packed, router_scale = _pack_matrix(np.asarray(router_weights, dtype=np.float32))
    active_experts = set(range(len(expert_matrices))) if active_expert_ids is None else set(active_expert_ids)
    scores = expert_scores or [float(sum(np.linalg.norm(matrix) for matrix in expert)) for expert in expert_matrices]

    experts_manifest: list[dict[str, object]] = []
    expert_blob_parts: list[bytes] = []
    current_offset = len(router_packed)

    for expert_id, matrices in enumerate(expert_matrices):
        if expert_id not in active_experts:
            continue
        blocks, expert_blob = _build_expert_blob([np.asarray(matrix, dtype=np.float32) for matrix in matrices])
        for block in blocks:
            block["matrix_offset"] = int(block["matrix_offset"]) + current_offset
        experts_manifest.append(
            {
                "expert_id": expert_id,
                "active": True,
                "score": float(scores[expert_id]),
                "blocks": blocks,
            }
        )
        expert_blob_parts.append(expert_blob)
        current_offset += len(expert_blob)

    manifest = {
        "version": 1,
        "router": {
            "rows": int(router_weights.shape[0]),
            "cols": int(router_weights.shape[1]),
            "matrix_offset": 0,
            "matrix_nbytes": len(router_packed),
            "storage_dtype": INT8_STORAGE_DTYPE,
            "scale": router_scale,
        },
        "experts": experts_manifest,
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return MOE_ARTIFACT_MAGIC + struct.pack("<I", len(manifest_bytes)) + manifest_bytes + router_packed + b"".join(expert_blob_parts)


def write_moe_reap_artifact(
    path: str | Path,
    router_weights: np.ndarray,
    expert_matrices: list[list[np.ndarray]],
    *,
    expert_scores: list[float] | None = None,
    active_expert_ids: list[int] | None = None,
) -> dict[str, int]:
    payload = build_moe_reap_artifact(
        router_weights,
        expert_matrices,
        expert_scores=expert_scores,
        active_expert_ids=active_expert_ids,
    )
    Path(path).write_bytes(payload)
    return {
        "artifact_bytes": len(payload),
        "expert_count": len(expert_matrices),
        "active_expert_count": len(expert_matrices) if active_expert_ids is None else len(active_expert_ids),
    }


def reap_prune_experts(
    expert_matrices: list[list[np.ndarray]],
    *,
    keep_fraction: float = 0.5,
) -> tuple[list[int], list[float]]:
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be within (0, 1]")

    scores = [float(sum(np.linalg.norm(matrix) for matrix in expert)) for expert in expert_matrices]
    keep_count = max(1, int(np.ceil(len(expert_matrices) * keep_fraction)))
    ranked = sorted(range(len(expert_matrices)), key=lambda index: scores[index], reverse=True)
    return sorted(ranked[:keep_count]), scores


class DiskBackedMoEArtifact:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file: BinaryIO = self.path.open("rb")
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        header = self._mapping[: len(MOE_ARTIFACT_MAGIC)]
        if header != MOE_ARTIFACT_MAGIC:
            self.close()
            raise ValueError("Not a MoE artifact")

        manifest_size_start = len(MOE_ARTIFACT_MAGIC)
        manifest_size_end = manifest_size_start + HEADER_LENGTH_BYTES
        manifest_size = struct.unpack("<I", self._mapping[manifest_size_start:manifest_size_end])[0]
        manifest_start = manifest_size_end
        manifest_end = manifest_start + manifest_size
        self.manifest = json.loads(self._mapping[manifest_start:manifest_end].decode("utf-8"))
        self.data_start = manifest_end
        router = self.manifest["router"]
        self.router_metadata = ExpertBlockMetadata(
            rows=int(router["rows"]),
            cols=int(router["cols"]),
            matrix_offset=int(router["matrix_offset"]),
            matrix_nbytes=int(router["matrix_nbytes"]),
            storage_dtype=str(router["storage_dtype"]),
            scale=float(router["scale"]),
        )
        self.experts = [
            ExpertLayout(
                expert_id=int(expert["expert_id"]),
                active=bool(expert["active"]),
                score=float(expert["score"]),
                blocks=[
                    ExpertBlockMetadata(
                        rows=int(block["rows"]),
                        cols=int(block["cols"]),
                        matrix_offset=int(block["matrix_offset"]),
                        matrix_nbytes=int(block["matrix_nbytes"]),
                        storage_dtype=str(block["storage_dtype"]),
                        scale=float(block["scale"]),
                    )
                    for block in expert["blocks"]
                ],
            )
            for expert in self.manifest["experts"]
        ]
        self._expert_by_id = {expert.expert_id: expert for expert in self.experts}

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        file_obj = getattr(self, "_file", None)
        if file_obj is not None:
            file_obj.close()
            self._file = None

    def __enter__(self) -> "DiskBackedMoEArtifact":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _read_quantized_matrix(self, metadata: ExpertBlockMetadata) -> np.ndarray:
        start = self.data_start + metadata.matrix_offset
        end = start + metadata.matrix_nbytes
        return np.frombuffer(self._mapping[start:end], dtype=np.int8).reshape((metadata.rows, metadata.cols))

    def router_weights(self) -> tuple[np.ndarray, float]:
        quantized = self._read_quantized_matrix(self.router_metadata)
        return quantized, self.router_metadata.scale

    def expert_block(self, expert_id: int, block_index: int) -> tuple[np.ndarray, float, int]:
        expert = self._expert_by_id[expert_id]
        metadata = expert.blocks[block_index]
        return self._read_quantized_matrix(metadata), metadata.scale, metadata.matrix_nbytes

    @property
    def artifact_size_bytes(self) -> int:
        return len(self._mapping)


def run_moe_artifact(
    artifact: DiskBackedMoEArtifact,
    input_activations: np.ndarray,
    *,
    backend: CPUInferenceBackend | None = None,
    top_k: int = 2,
    hidden_activation: str | None = "relu",
) -> MoEInferenceResult:
    inference_backend = backend or NumpyInt8DynamicBackend()
    router_quantized, router_scale = artifact.router_weights()
    router_result = inference_backend.matmul_quantized_weights(
        input_activations[:, : artifact.router_metadata.rows],
        router_quantized,
        router_scale,
    )
    router_logits = router_result.output[0]

    active_experts = [expert for expert in artifact.experts if expert.active]
    ranked_active = sorted(active_experts, key=lambda expert: router_logits[expert.expert_id], reverse=True)[:top_k]

    routed_outputs: list[RoutedExpertResult] = []
    total_bytes_read = artifact.router_metadata.matrix_nbytes

    for expert in ranked_active:
        current = input_activations[:, : expert.blocks[0].rows].astype(np.float32)
        expert_bytes = 0
        for block_index, block in enumerate(expert.blocks):
            quantized, scale, bytes_read = artifact.expert_block(expert.expert_id, block_index)
            result = inference_backend.matmul_quantized_weights(current, quantized, scale)
            current = result.output
            expert_bytes += bytes_read
            if block_index != len(expert.blocks) - 1:
                current = apply_hidden_activation(current, hidden_activation)

        total_bytes_read += expert_bytes
        routed_outputs.append(
            RoutedExpertResult(
                expert_id=expert.expert_id,
                router_weight=float(router_logits[expert.expert_id]),
                output=current,
                bytes_read=expert_bytes,
            )
        )

    if not routed_outputs:
        output = np.zeros((input_activations.shape[0], artifact.experts[0].blocks[-1].cols), dtype=np.float32)
        active_ids: list[int] = []
    else:
        weights = np.array([result.router_weight for result in routed_outputs], dtype=np.float32)
        weights = np.exp(weights - np.max(weights))
        weights /= np.sum(weights)
        output = sum(weight * result.output for weight, result in zip(weights, routed_outputs))
        active_ids = [result.expert_id for result in routed_outputs]

    return MoEInferenceResult(
        output=output,
        active_expert_ids=active_ids,
        bytes_read=total_bytes_read,
        experts_evaluated=len(routed_outputs),
    )
