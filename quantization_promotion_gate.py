"""Promotion gate for quantization-surviving candidate gains."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

from chelation_logger import get_logger


@dataclass
class QuantizationGateResult:
    """Decision record for FP32-vs-quantized candidate fitness."""

    passed: bool
    fp32_fitness: float
    quantized_fitness: float
    baseline_fitness: float
    fp32_gain: float
    quantized_gain: float
    retained_gain_ratio: float
    threshold: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "fp32_fitness": self.fp32_fitness,
            "quantized_fitness": self.quantized_fitness,
            "baseline_fitness": self.baseline_fitness,
            "fp32_gain": self.fp32_gain,
            "quantized_gain": self.quantized_gain,
            "retained_gain_ratio": self.retained_gain_ratio,
            "threshold": self.threshold,
            "reasons": self.reasons,
        }


class QuantizationPromotionGate:
    """Reject candidates whose improvements disappear after quantization."""

    def __init__(self, retained_gain_threshold: float = 0.8, minimum_fp32_gain: float = 0.0, logger=None):
        retained_threshold = float(retained_gain_threshold)
        minimum_gain = float(minimum_fp32_gain)
        if not math.isfinite(retained_threshold) or retained_threshold < 0:
            raise ValueError("retained_gain_threshold must be finite and non-negative")
        if not math.isfinite(minimum_gain) or minimum_gain < 0:
            raise ValueError("minimum_fp32_gain must be finite and non-negative")
        self.retained_gain_threshold = retained_threshold
        self.minimum_fp32_gain = minimum_gain
        self.logger = logger or get_logger()

    def evaluate(
        self,
        fp32_fitness: float,
        quantized_fitness: float,
        baseline_fitness: float = 0.0,
    ) -> QuantizationGateResult:
        fp32_value = float(fp32_fitness)
        quantized_value = float(quantized_fitness)
        baseline_value = float(baseline_fitness)
        if not all(
            math.isfinite(value)
            for value in (fp32_value, quantized_value, baseline_value)
        ):
            raise ValueError("quantization promotion fitness values must be finite")
        fp32_gain = fp32_value - baseline_value
        quantized_gain = quantized_value - baseline_value
        if fp32_gain <= self.minimum_fp32_gain:
            retained_ratio = 1.0 if quantized_gain >= fp32_gain else 0.0
            passed = False
        else:
            retained_ratio = quantized_gain / fp32_gain
            passed = retained_ratio >= self.retained_gain_threshold
        reasons = []
        if fp32_gain <= self.minimum_fp32_gain:
            reasons.append("fp32_gain_below_minimum")
        if retained_ratio < self.retained_gain_threshold:
            reasons.append("retained_gain_below_threshold")
        if quantized_gain < 0:
            reasons.append("quantized_fitness_below_baseline")

        result = QuantizationGateResult(
            passed=bool(passed),
            fp32_fitness=fp32_value,
            quantized_fitness=quantized_value,
            baseline_fitness=baseline_value,
            fp32_gain=float(fp32_gain),
            quantized_gain=float(quantized_gain),
            retained_gain_ratio=float(retained_ratio),
            threshold=self.retained_gain_threshold,
            reasons=reasons,
        )
        self.logger.log_event(
            "quantization_gate_evaluated",
            "Evaluated quantization promotion gate",
            passed=result.passed,
            fp32_fitness=result.fp32_fitness,
            quantized_fitness=result.quantized_fitness,
            baseline_fitness=result.baseline_fitness,
            retained_gain_ratio=result.retained_gain_ratio,
            threshold=result.threshold,
            reasons=result.reasons,
        )
        return result
