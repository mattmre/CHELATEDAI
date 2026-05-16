from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# Research-stage / POC module. No production code path in this repo consumes it.
EXPERIMENTAL = True


@dataclass(frozen=True)
class BackendResult:
    output: np.ndarray
    backend_name: str


class CPUInferenceBackend(ABC):
    name: str

    @abstractmethod
    def matmul(self, activations: np.ndarray, weights: np.ndarray) -> BackendResult:
        raise NotImplementedError

    def matmul_quantized_weights(
        self,
        activations: np.ndarray,
        quantized_weights: np.ndarray,
        weight_scale: float,
    ) -> BackendResult:
        dequantized_weights = quantized_weights.astype(np.float32) * weight_scale
        return self.matmul(activations, dequantized_weights)


class NumpyFloat32Backend(CPUInferenceBackend):
    name = "numpy_float32"

    def matmul(self, activations: np.ndarray, weights: np.ndarray) -> BackendResult:
        output = activations.astype(np.float32, copy=False) @ weights.astype(np.float32, copy=False)
        return BackendResult(output=output, backend_name=self.name)


def _max_abs_scale(values: np.ndarray) -> float:
    max_abs = float(np.max(np.abs(values)))
    return 1.0 if max_abs == 0.0 else max_abs / 127.0


class NumpyInt8DynamicBackend(CPUInferenceBackend):
    name = "numpy_int8_dynamic"

    def matmul(self, activations: np.ndarray, weights: np.ndarray) -> BackendResult:
        activations_f32 = activations.astype(np.float32, copy=False)
        weights_f32 = weights.astype(np.float32, copy=False)

        activation_scale = _max_abs_scale(activations_f32)
        weight_scale = _max_abs_scale(weights_f32)

        quantized_activations = np.clip(np.rint(activations_f32 / activation_scale), -127, 127).astype(np.int8)
        quantized_weights = np.clip(np.rint(weights_f32 / weight_scale), -127, 127).astype(np.int8)

        int32_output = quantized_activations.astype(np.int32) @ quantized_weights.astype(np.int32)
        output = int32_output.astype(np.float32) * (activation_scale * weight_scale)
        return BackendResult(output=output, backend_name=self.name)

    def matmul_quantized_weights(
        self,
        activations: np.ndarray,
        quantized_weights: np.ndarray,
        weight_scale: float,
    ) -> BackendResult:
        activations_f32 = activations.astype(np.float32, copy=False)
        activation_scale = _max_abs_scale(activations_f32)
        quantized_activations = np.clip(np.rint(activations_f32 / activation_scale), -127, 127).astype(np.int8)
        int32_output = quantized_activations.astype(np.int32) @ quantized_weights.astype(np.int32)
        output = int32_output.astype(np.float32) * (activation_scale * weight_scale)
        return BackendResult(output=output, backend_name=f"{self.name}_prequantized_weights")
