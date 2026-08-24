"""Reproducible, bounded RADER-1 timing and correctness harness.

The default protocol compares steady-state Rader and NumPy implementations for
both DFT and cyclic correlation at prime lengths 4091 and 4691.  Every matched
comparison uses the same complex128 inputs, length, and tolerance.  A separate
NumPy-only length-4096 context is reported without treating it as a matched
prime-length comparator.

Timing observations from this harness are not performance conclusions.  Plan
construction is reported separately, input generation is excluded from method
timings, and neither smooth ``p - 1`` factorization nor a favorable block sign
count establishes speed, utility, or novelty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from prime_ring_rader import (
    DEFAULT_MAX_TEMPORARY_BYTES,
    RaderResourceError,
    RaderValidationError,
    _RaderPlan,
    _build_plan,
    _rader_dft_with_plan,
    estimate_rader_temporary_bytes,
)


PROTOCOL_ID = "RADER-1"
SCHEMA_VERSION = "rader-1.v1"
DTYPE_NAME = "complex128"
COMPLEX_BYTES = np.dtype(np.complex128).itemsize

HARD_MAX_ESTIMATED_PEAK_BYTES = DEFAULT_MAX_TEMPORARY_BYTES
HARD_MAX_MODELED_WORK_UNITS = 500_000_000
HARD_MAX_SECONDS = 120.0
HARD_MAX_PRIME_COUNT = 4
HARD_MAX_SEED_COUNT = 8
HARD_MAX_BLOCKS = 12
HARD_MAX_VECTORS_PER_BLOCK = 4
HARD_MAX_WARMUPS = 2
HARD_MAX_TRANSFORM_LENGTH = 8_192
HARD_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
HARD_MAX_SEED = (1 << 63) - 1

DEFAULT_MAX_MODELED_WORK_UNITS = 400_000_000
DEFAULT_MAX_SECONDS = 60.0
DEFAULT_SEEDS = (10_007, 20_011, 30_013)
DEFAULT_PRIMES = (4_091, 4_691)

MATCHED_COMPARISON_LABEL = (
    "MATCHED_SAME_PRIME_LENGTH_DTYPE_INPUT_AND_TOLERANCE"
)
GENERIC_CONTEXT_LABEL = (
    "GENERIC_NUMPY_CONTEXT_NOT_A_MATCHED_PRIME_COMPARATOR"
)
TIMING_STATUS = "BOUNDED_TIMING_OBSERVATIONS_ONLY"
PERFORMANCE_DISCLAIMER = (
    "Block timings and sign counts are local observations, not a calibrated "
    "performance, speedup, cost, utility, hardware, or novelty claim."
)


class RaderHarnessValidationError(ValueError):
    """Raised when the frozen RADER-1 protocol is malformed."""


class RaderHarnessResourceError(RuntimeError):
    """Raised before or during work when a declared boundary is crossed."""


class RaderHarnessCorrectnessError(RuntimeError):
    """Raised when matched Rader and NumPy results violate the tolerance."""


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise RaderHarnessValidationError(
            "{} must be an integer".format(name)
        )
    result = int(value)
    if result < minimum:
        raise RaderHarnessValidationError(
            "{} must be >= {}".format(name, minimum)
        )
    return result


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise RaderHarnessValidationError(
            "{} must be a finite positive number".format(name)
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RaderHarnessValidationError(
            "{} must be a finite positive number".format(name)
        )
    return result


def _integer_tuple(
    values: object,
    name: str,
    *,
    minimum: int,
    maximum_item: Optional[int] = None,
) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RaderHarnessValidationError(
            "{} must be a non-empty sequence of integers".format(name)
        )
    if not values:
        raise RaderHarnessValidationError(
            "{} must not be empty".format(name)
        )
    normalized = []
    for index, value in enumerate(values):
        item = _plain_int(
            value,
            "{}[{}]".format(name, index),
            minimum,
        )
        if maximum_item is not None and item > maximum_item:
            raise RaderHarnessValidationError(
                "{}[{}] cannot exceed {}".format(
                    name,
                    index,
                    maximum_item,
                )
            )
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise RaderHarnessValidationError(
            "{} must contain unique values".format(name)
        )
    return tuple(normalized)


@dataclass(frozen=True)
class RaderHarnessConfig:
    """Frozen deterministic RADER-1 experiment configuration."""

    primes: Tuple[int, ...] = DEFAULT_PRIMES
    generic_numpy_length: int = 4_096
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    blocks: int = 6
    vectors_per_block: int = 2
    warmup_repetitions: int = 1
    rtol: float = 1e-10
    atol: float = 1e-9
    dtype: str = DTYPE_NAME

    def __post_init__(self) -> None:
        primes = _integer_tuple(self.primes, "primes", minimum=3)
        seeds = _integer_tuple(
            self.seeds,
            "seeds",
            minimum=0,
            maximum_item=HARD_MAX_SEED,
        )
        generic_length = _plain_int(
            self.generic_numpy_length,
            "generic_numpy_length",
            2,
        )
        blocks = _plain_int(self.blocks, "blocks", 2)
        vectors = _plain_int(
            self.vectors_per_block,
            "vectors_per_block",
            1,
        )
        warmups = _plain_int(
            self.warmup_repetitions,
            "warmup_repetitions",
            1,
        )
        if blocks % 2:
            raise RaderHarnessValidationError(
                "blocks must be even so method-first order is balanced"
            )
        rtol = _positive_real(self.rtol, "rtol")
        atol = _positive_real(self.atol, "atol")
        dtype = str(self.dtype)
        if dtype != DTYPE_NAME:
            raise RaderHarnessValidationError(
                "dtype is frozen to complex128 for RADER-1"
            )
        object.__setattr__(self, "primes", primes)
        object.__setattr__(self, "generic_numpy_length", generic_length)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "vectors_per_block", vectors)
        object.__setattr__(self, "warmup_repetitions", warmups)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "dtype", dtype)


@dataclass(frozen=True)
class RaderHarnessBudget:
    """Hostile-safe immutable boundaries for one harness execution."""

    max_estimated_peak_bytes: int = HARD_MAX_ESTIMATED_PEAK_BYTES
    max_modeled_work_units: int = DEFAULT_MAX_MODELED_WORK_UNITS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_prime_count: int = HARD_MAX_PRIME_COUNT
    max_seed_count: int = HARD_MAX_SEED_COUNT
    max_blocks: int = HARD_MAX_BLOCKS
    max_vectors_per_block: int = HARD_MAX_VECTORS_PER_BLOCK
    max_warmups: int = HARD_MAX_WARMUPS
    max_transform_length: int = HARD_MAX_TRANSFORM_LENGTH
    max_output_bytes: int = HARD_MAX_OUTPUT_BYTES

    def __post_init__(self) -> None:
        integer_caps = (
            (
                "max_estimated_peak_bytes",
                self.max_estimated_peak_bytes,
                HARD_MAX_ESTIMATED_PEAK_BYTES,
            ),
            (
                "max_modeled_work_units",
                self.max_modeled_work_units,
                HARD_MAX_MODELED_WORK_UNITS,
            ),
            ("max_prime_count", self.max_prime_count, HARD_MAX_PRIME_COUNT),
            ("max_seed_count", self.max_seed_count, HARD_MAX_SEED_COUNT),
            ("max_blocks", self.max_blocks, HARD_MAX_BLOCKS),
            (
                "max_vectors_per_block",
                self.max_vectors_per_block,
                HARD_MAX_VECTORS_PER_BLOCK,
            ),
            ("max_warmups", self.max_warmups, HARD_MAX_WARMUPS),
            (
                "max_transform_length",
                self.max_transform_length,
                HARD_MAX_TRANSFORM_LENGTH,
            ),
            (
                "max_output_bytes",
                self.max_output_bytes,
                HARD_MAX_OUTPUT_BYTES,
            ),
        )
        for name, value, hard_cap in integer_caps:
            normalized = _plain_int(value, name, 1)
            if normalized > hard_cap:
                raise RaderHarnessValidationError(
                    "{} cannot exceed immutable hard cap {}".format(
                        name,
                        hard_cap,
                    )
                )
            object.__setattr__(self, name, normalized)
        seconds = _positive_real(self.max_seconds, "max_seconds")
        if seconds > HARD_MAX_SECONDS:
            raise RaderHarnessValidationError(
                "max_seconds cannot exceed immutable hard cap {:g}".format(
                    HARD_MAX_SECONDS
                )
            )
        object.__setattr__(self, "max_seconds", seconds)


@dataclass(frozen=True)
class PrimeResourcePreflight:
    """Per-prime Rader temporary-allocation preflight."""

    p: int
    convolution_length: int
    factorization: Tuple[int, ...]
    rader_correlation_temporary_bytes: int
    input_generation_peak_bytes: int
    output_block_bytes: int
    estimated_live_peak_bytes: int


@dataclass(frozen=True)
class RaderHarnessPreflight:
    """Complete allocation/work estimate before plan or vector allocation."""

    protocol_id: str
    prime_preflights: Tuple[PrimeResourcePreflight, ...]
    generic_numpy_length: int
    generic_input_generation_peak_bytes: int
    generic_output_block_bytes: int
    result_metadata_allowance_bytes: int
    estimated_peak_bytes: int
    max_estimated_peak_bytes: int
    modeled_work_units: int
    max_modeled_work_units: int
    work_components: Tuple[Tuple[str, int], ...]
    total_timed_vectors_per_operation: int
    total_warmup_vectors_per_operation: int
    max_seconds: float
    input_batches_materialized_one_seed_at_a_time: bool
    measured_process_peak: bool
    work_model_is_time_calibration: bool


def balanced_order_schedule(
    blocks: int,
    *,
    rader_first: bool = True,
) -> Tuple[Tuple[str, str], ...]:
    """Return an alternating, exactly balanced method-first schedule."""

    count = _plain_int(blocks, "blocks", 2)
    if not isinstance(rader_first, bool):
        raise RaderHarnessValidationError(
            "rader_first must be boolean"
        )
    if count % 2:
        raise RaderHarnessValidationError(
            "blocks must be even for an exactly balanced schedule"
        )
    first = ("rader", "numpy") if rader_first else ("numpy", "rader")
    second = tuple(reversed(first))
    return tuple(first if block % 2 == 0 else second for block in range(count))


def _fft_work_units(length: int) -> int:
    return int(math.ceil(5.0 * length * math.log2(max(2, length))))


def _rader_dft_work_units(p: int) -> int:
    return int(2 * _fft_work_units(p - 1) + 8 * p)


def _input_generation_peak_bytes(
    length: int,
    blocks: int,
    vectors_per_block: int,
) -> int:
    elements = blocks * vectors_per_block * length
    # Three retained complex batches (DFT, correlation left/right) plus the
    # real and imaginary float64 scratch arrays used to construct the newest.
    return int(elements * (3 * COMPLEX_BYTES + 2 * 8))


def _output_block_bytes(length: int, vectors_per_block: int) -> int:
    # Both methods' output vectors are held only until their matched check.
    # One complex difference and one float64 magnitude scratch vector are also
    # included for ``allclose`` and maximum-error evaluation.
    retained_outputs = 2 * vectors_per_block * length * COMPLEX_BYTES
    comparison_scratch = length * (COMPLEX_BYTES + 8)
    return int(retained_outputs + comparison_scratch)


def preflight_rader_harness(
    config: RaderHarnessConfig = RaderHarnessConfig(),
    budget: RaderHarnessBudget = RaderHarnessBudget(),
) -> RaderHarnessPreflight:
    """Validate and estimate the complete harness without building a plan."""

    if not isinstance(config, RaderHarnessConfig):
        raise RaderHarnessValidationError(
            "config must be RaderHarnessConfig"
        )
    if not isinstance(budget, RaderHarnessBudget):
        raise RaderHarnessValidationError(
            "budget must be RaderHarnessBudget"
        )
    limit_checks = (
        ("prime count", len(config.primes), budget.max_prime_count),
        ("seed count", len(config.seeds), budget.max_seed_count),
        ("blocks", config.blocks, budget.max_blocks),
        (
            "vectors_per_block",
            config.vectors_per_block,
            budget.max_vectors_per_block,
        ),
        (
            "warmup_repetitions",
            config.warmup_repetitions,
            budget.max_warmups,
        ),
        (
            "generic_numpy_length",
            config.generic_numpy_length,
            budget.max_transform_length,
        ),
    )
    for name, observed, maximum in limit_checks:
        if observed > maximum:
            raise RaderHarnessResourceError(
                "{} exceeds budget: {} > {}".format(
                    name,
                    observed,
                    maximum,
                )
            )
    for p in config.primes:
        if p > budget.max_transform_length:
            raise RaderHarnessResourceError(
                "prime length exceeds budget: {} > {}".format(
                    p,
                    budget.max_transform_length,
                )
            )

    prime_preflights = []
    live_peaks = []
    for p in config.primes:
        try:
            rader = estimate_rader_temporary_bytes(
                p,
                operation="correlation",
                max_temporary_bytes=budget.max_estimated_peak_bytes,
            )
        except (RaderValidationError, RaderResourceError) as exc:
            raise RaderHarnessValidationError(str(exc)) from exc
        if not rader.allowed:
            raise RaderHarnessResourceError(
                "Rader temporary estimate exceeds harness byte budget for "
                "p={}: {} > {}".format(
                    p,
                    rader.estimated_peak_temporary_bytes,
                    budget.max_estimated_peak_bytes,
                )
            )
        input_peak = _input_generation_peak_bytes(
            p,
            config.blocks,
            config.vectors_per_block,
        )
        output_bytes = _output_block_bytes(
            p,
            config.vectors_per_block,
        )
        live_peak = (
            rader.estimated_peak_temporary_bytes
            + input_peak
            + output_bytes
        )
        prime_preflights.append(
            PrimeResourcePreflight(
                p=p,
                convolution_length=p - 1,
                factorization=rader.primitive_length_factorization,
                rader_correlation_temporary_bytes=(
                    rader.estimated_peak_temporary_bytes
                ),
                input_generation_peak_bytes=input_peak,
                output_block_bytes=output_bytes,
                estimated_live_peak_bytes=live_peak,
            )
        )
        live_peaks.append(live_peak)

    generic_input_peak = _input_generation_peak_bytes(
        config.generic_numpy_length,
        config.blocks,
        config.vectors_per_block,
    )
    generic_output_bytes = _output_block_bytes(
        config.generic_numpy_length,
        config.vectors_per_block,
    )
    generic_live_peak = generic_input_peak + generic_output_bytes
    result_allowance = int(
        65_536
        + 512
        * len(config.primes)
        * len(config.seeds)
        * config.blocks
    )
    estimated_peak = max(live_peaks + [generic_live_peak]) + result_allowance
    if estimated_peak > budget.max_estimated_peak_bytes:
        raise RaderHarnessResourceError(
            "estimated harness peak exceeds byte budget: {} > {}".format(
                estimated_peak,
                budget.max_estimated_peak_bytes,
            )
        )

    calls_per_method_operation = len(config.seeds) * (
        config.warmup_repetitions
        + config.blocks * config.vectors_per_block
    )
    work_components = []
    for p in config.primes:
        plan_work = 4 * _fft_work_units(p - 1) + 10 * p
        # Per input: one DFT plus one three-transform correlation, for both
        # Rader and NumPy.
        timed_and_warm_work = calls_per_method_operation * 4 * (
            _rader_dft_work_units(p) + _fft_work_units(p)
        )
        correctness_work = (
            2
            * calls_per_method_operation
            * p
        )
        work_components.append(
            ("p{}_plan_build".format(p), int(plan_work))
        )
        work_components.append(
            (
                "p{}_timed_warm_and_correctness".format(p),
                int(timed_and_warm_work + correctness_work),
            )
        )
    generic_work = (
        calls_per_method_operation
        * 4
        * _fft_work_units(config.generic_numpy_length)
    )
    work_components.append(("generic_numpy_context", int(generic_work)))
    modeled_work = int(sum(value for _name, value in work_components))
    if modeled_work > budget.max_modeled_work_units:
        raise RaderHarnessResourceError(
            "modeled harness work exceeds budget: {} > {}".format(
                modeled_work,
                budget.max_modeled_work_units,
            )
        )

    return RaderHarnessPreflight(
        protocol_id=PROTOCOL_ID,
        prime_preflights=tuple(prime_preflights),
        generic_numpy_length=config.generic_numpy_length,
        generic_input_generation_peak_bytes=generic_input_peak,
        generic_output_block_bytes=generic_output_bytes,
        result_metadata_allowance_bytes=result_allowance,
        estimated_peak_bytes=estimated_peak,
        max_estimated_peak_bytes=budget.max_estimated_peak_bytes,
        modeled_work_units=modeled_work,
        max_modeled_work_units=budget.max_modeled_work_units,
        work_components=tuple(work_components),
        total_timed_vectors_per_operation=(
            len(config.primes)
            * len(config.seeds)
            * config.blocks
            * config.vectors_per_block
        ),
        total_warmup_vectors_per_operation=(
            len(config.primes)
            * len(config.seeds)
            * config.warmup_repetitions
        ),
        max_seconds=budget.max_seconds,
        input_batches_materialized_one_seed_at_a_time=True,
        measured_process_peak=False,
        work_model_is_time_calibration=False,
    )


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() > deadline:
        raise RaderHarnessResourceError(
            "RADER-1 exceeded its total wall-clock deadline"
        )


def _random_complex_batch(
    rng: np.random.Generator,
    shape: Tuple[int, int, int],
) -> np.ndarray:
    real = rng.standard_normal(shape)
    imaginary = rng.standard_normal(shape)
    result = np.empty(shape, dtype=np.complex128)
    result.real = real
    result.imag = imaginary
    return np.ascontiguousarray(result)


def _generate_inputs(
    *,
    seed: int,
    length: int,
    blocks: int,
    vectors_per_block: int,
    domain: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_sequence = np.random.SeedSequence([seed, length, domain])
    rng = np.random.default_rng(seed_sequence)
    shape = (blocks, vectors_per_block, length)
    return (
        _random_complex_batch(rng, shape),
        _random_complex_batch(rng, shape),
        _random_complex_batch(rng, shape),
    )


def _array_digest(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _rader_correlation_with_plan(
    left: np.ndarray,
    right: np.ndarray,
    plan: _RaderPlan,
) -> np.ndarray:
    left_spectrum = _rader_dft_with_plan(left, plan)
    right_spectrum = _rader_dft_with_plan(right, plan)
    spectral_product = left_spectrum * np.conjugate(right_spectrum)
    inverse_transform = _rader_dft_with_plan(
        np.conjugate(spectral_product),
        plan,
    )
    return np.conjugate(inverse_transform) / plan.p


def _numpy_correlation(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return np.fft.ifft(
        np.fft.fft(left) * np.conjugate(np.fft.fft(right))
    )


def _assert_matched(
    rader_outputs: Sequence[np.ndarray],
    numpy_outputs: Sequence[np.ndarray],
    *,
    rtol: float,
    atol: float,
    operation: str,
    p: int,
) -> float:
    if len(rader_outputs) != len(numpy_outputs):
        raise RaderHarnessCorrectnessError(
            "matched output counts differ"
        )
    maximum_error = 0.0
    for index, (rader, numpy_result) in enumerate(
        zip(rader_outputs, numpy_outputs)
    ):
        if (
            rader.shape != numpy_result.shape
            or rader.dtype != np.dtype(np.complex128)
            or numpy_result.dtype != np.dtype(np.complex128)
        ):
            raise RaderHarnessCorrectnessError(
                "matched output shape or dtype differs"
            )
        error = float(np.max(np.abs(rader - numpy_result)))
        maximum_error = max(maximum_error, error)
        if not np.allclose(
            rader,
            numpy_result,
            rtol=rtol,
            atol=atol,
        ):
            raise RaderHarnessCorrectnessError(
                "{} correctness failed at p={} vector={}: max error {}".format(
                    operation,
                    p,
                    index,
                    error,
                )
            )
    return maximum_error


def _time_batch(
    function: Any,
    arguments: Sequence[Tuple[np.ndarray, ...]],
    *,
    deadline: float,
) -> Tuple[float, Tuple[np.ndarray, ...]]:
    _check_deadline(deadline)
    started = time.perf_counter()
    outputs = tuple(function(*values) for values in arguments)
    elapsed = time.perf_counter() - started
    _check_deadline(deadline)
    return float(elapsed), outputs


def _benchmark_matched_seed(
    *,
    p: int,
    plan: _RaderPlan,
    operation: str,
    inputs: Tuple[np.ndarray, ...],
    schedule: Tuple[Tuple[str, str], ...],
    config: RaderHarnessConfig,
    deadline: float,
) -> Dict[str, Any]:
    if operation == "dft":
        def rader_function(vector: np.ndarray) -> np.ndarray:
            return _rader_dft_with_plan(vector, plan)

        numpy_function = np.fft.fft
    elif operation == "cyclic_correlation":
        def rader_function(
            left: np.ndarray,
            right: np.ndarray,
        ) -> np.ndarray:
            return _rader_correlation_with_plan(left, right, plan)

        numpy_function = _numpy_correlation
    else:
        raise RaderHarnessValidationError("unsupported operation")

    warmup_arguments = tuple(array[0, 0] for array in inputs)
    maximum_error = 0.0
    for _ in range(config.warmup_repetitions):
        _check_deadline(deadline)
        rader_warm = rader_function(*warmup_arguments)
        numpy_warm = numpy_function(*warmup_arguments)
        maximum_error = max(
            maximum_error,
            _assert_matched(
                (rader_warm,),
                (numpy_warm,),
                rtol=config.rtol,
                atol=config.atol,
                operation=operation,
                p=p,
            ),
        )

    rader_seconds = []
    numpy_seconds = []
    for block, order in enumerate(schedule):
        arguments = tuple(
            tuple(array[block, vector] for array in inputs)
            for vector in range(config.vectors_per_block)
        )
        outputs: Dict[str, Tuple[np.ndarray, ...]] = {}
        elapsed: Dict[str, float] = {}
        for method in order:
            function = (
                rader_function if method == "rader" else numpy_function
            )
            duration, method_outputs = _time_batch(
                function,
                arguments,
                deadline=deadline,
            )
            elapsed[method] = duration
            outputs[method] = method_outputs
        maximum_error = max(
            maximum_error,
            _assert_matched(
                outputs["rader"],
                outputs["numpy"],
                rtol=config.rtol,
                atol=config.atol,
                operation=operation,
                p=p,
            ),
        )
        rader_seconds.append(elapsed["rader"])
        numpy_seconds.append(elapsed["numpy"])
    return {
        "rader_block_seconds": rader_seconds,
        "numpy_block_seconds": numpy_seconds,
        "max_abs_error": maximum_error,
        "correctness_check_passed": True,
        "order_schedule": schedule,
    }


def _new_comparison_result(
    operation: str,
    config: RaderHarnessConfig,
) -> Dict[str, Any]:
    return {
        "operation": operation,
        "comparison_label": MATCHED_COMPARISON_LABEL,
        "same_length": True,
        "same_dtype": True,
        "same_input": True,
        "same_tolerance": True,
        "dtype": DTYPE_NAME,
        "rtol": config.rtol,
        "atol": config.atol,
        "warmup_performed": True,
        "warmup_repetitions_per_method": config.warmup_repetitions,
        "plan_build_included_in_method_timing": False,
        "input_generation_included_in_method_timing": False,
        "independent_input_per_block": True,
        "rader_block_seconds": [],
        "numpy_block_seconds": [],
        "order_schedule_by_seed": [],
        "input_sha256_by_seed": [],
        "max_abs_error": 0.0,
        "correctness_check_passed": True,
    }


def _finalize_comparison(result: Dict[str, Any]) -> Dict[str, Any]:
    rader_seconds = result["rader_block_seconds"]
    numpy_seconds = result["numpy_block_seconds"]
    result["rader_median_block_seconds"] = float(
        statistics.median(rader_seconds)
    )
    result["numpy_median_block_seconds"] = float(
        statistics.median(numpy_seconds)
    )
    result["rader_faster_block_count"] = sum(
        left < right
        for left, right in zip(rader_seconds, numpy_seconds)
    )
    result["numpy_faster_block_count"] = sum(
        right < left
        for left, right in zip(rader_seconds, numpy_seconds)
    )
    result["exact_tie_block_count"] = sum(
        left == right
        for left, right in zip(rader_seconds, numpy_seconds)
    )
    result["timed_block_count"] = len(rader_seconds)
    result["performance_claim"] = False
    result["performance_disclaimer"] = PERFORMANCE_DISCLAIMER
    return result


def _time_numpy_context_seed(
    *,
    operation: str,
    inputs: Tuple[np.ndarray, ...],
    config: RaderHarnessConfig,
    deadline: float,
) -> Tuple[float, ...]:
    function = np.fft.fft if operation == "dft" else _numpy_correlation
    warmup_arguments = tuple(array[0, 0] for array in inputs)
    for _ in range(config.warmup_repetitions):
        _check_deadline(deadline)
        output = function(*warmup_arguments)
        if (
            output.dtype != np.dtype(np.complex128)
            or not np.all(np.isfinite(output))
        ):
            raise RaderHarnessCorrectnessError(
                "generic NumPy context produced invalid output"
            )
    seconds = []
    for block in range(config.blocks):
        arguments = tuple(
            tuple(array[block, vector] for array in inputs)
            for vector in range(config.vectors_per_block)
        )
        duration, outputs = _time_batch(
            function,
            arguments,
            deadline=deadline,
        )
        if any(
            output.dtype != np.dtype(np.complex128)
            or not np.all(np.isfinite(output))
            for output in outputs
        ):
            raise RaderHarnessCorrectnessError(
                "generic NumPy context produced invalid output"
            )
        seconds.append(duration)
    return tuple(seconds)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


_TIMING_DERIVED_KEYS = {
    "rader_faster_block_count",
    "numpy_faster_block_count",
    "exact_tie_block_count",
    "artifact_write",
}


def non_timing_projection(value: Any) -> Any:
    """Remove timing-derived fields for deterministic-result comparison."""

    normalized = _jsonable(value)
    if isinstance(normalized, dict):
        result = {}
        for key, item in normalized.items():
            if (
                key in _TIMING_DERIVED_KEYS
                or "seconds" in key
            ):
                continue
            result[key] = non_timing_projection(item)
        return result
    if isinstance(normalized, list):
        return [non_timing_projection(item) for item in normalized]
    return normalized


def _atomic_write_json(
    path: Path,
    payload: Dict[str, Any],
    *,
    max_output_bytes: int,
    deadline: Optional[float] = None,
) -> None:
    if deadline is not None:
        _check_deadline(deadline)
    encoded = (
        json.dumps(
            _jsonable(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > max_output_bytes:
        raise RaderHarnessResourceError(
            "JSON artifact exceeds output byte budget: {} > {}".format(
                len(encoded),
                max_output_bytes,
            )
        )
    if deadline is not None:
        _check_deadline(deadline)
    target = Path(path)
    if target.exists() and target.is_dir():
        raise RaderHarnessValidationError(
            "output path must name a file"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(target.name),
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if deadline is not None:
            _check_deadline(deadline)
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run_rader_harness(
    config: RaderHarnessConfig = RaderHarnessConfig(),
    budget: RaderHarnessBudget = RaderHarnessBudget(),
    *,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the bounded RADER-1 protocol and optionally atomically emit JSON."""

    preflight = preflight_rader_harness(config, budget)
    run_started = time.perf_counter()
    deadline = run_started + budget.max_seconds
    _check_deadline(deadline)
    prime_preflight_by_p = {
        item.p: item for item in preflight.prime_preflights
    }
    prime_results = []
    all_correct = True

    for prime_index, p in enumerate(config.primes):
        _check_deadline(deadline)
        plan_started = time.perf_counter()
        plan = _build_plan(p, None)
        plan_build_seconds = time.perf_counter() - plan_started
        _check_deadline(deadline)
        comparisons = {
            "dft": _new_comparison_result("dft", config),
            "cyclic_correlation": _new_comparison_result(
                "cyclic_correlation",
                config,
            ),
        }
        for seed_index, seed in enumerate(config.seeds):
            _check_deadline(deadline)
            dft_inputs, correlation_left, correlation_right = (
                _generate_inputs(
                    seed=seed,
                    length=p,
                    blocks=config.blocks,
                    vectors_per_block=config.vectors_per_block,
                    domain=1,
                )
            )
            operation_inputs = {
                "dft": (dft_inputs,),
                "cyclic_correlation": (
                    correlation_left,
                    correlation_right,
                ),
            }
            for operation_index, operation in enumerate(
                ("dft", "cyclic_correlation")
            ):
                schedule = balanced_order_schedule(
                    config.blocks,
                    rader_first=(
                        (prime_index + seed_index + operation_index) % 2
                        == 0
                    ),
                )
                seed_result = _benchmark_matched_seed(
                    p=p,
                    plan=plan,
                    operation=operation,
                    inputs=operation_inputs[operation],
                    schedule=schedule,
                    config=config,
                    deadline=deadline,
                )
                comparison = comparisons[operation]
                comparison["rader_block_seconds"].extend(
                    seed_result["rader_block_seconds"]
                )
                comparison["numpy_block_seconds"].extend(
                    seed_result["numpy_block_seconds"]
                )
                comparison["order_schedule_by_seed"].append(
                    seed_result["order_schedule"]
                )
                comparison["input_sha256_by_seed"].append(
                    _array_digest(operation_inputs[operation])
                )
                comparison["max_abs_error"] = max(
                    comparison["max_abs_error"],
                    seed_result["max_abs_error"],
                )
                comparison["correctness_check_passed"] = (
                    comparison["correctness_check_passed"]
                    and seed_result["correctness_check_passed"]
                )
            del dft_inputs, correlation_left, correlation_right
        finalized = {
            operation: _finalize_comparison(comparison)
            for operation, comparison in comparisons.items()
        }
        prime_correct = all(
            comparison["correctness_check_passed"]
            for comparison in finalized.values()
        )
        all_correct = all_correct and prime_correct
        prime_resource = prime_preflight_by_p[p]
        prime_results.append(
            {
                "p": p,
                "primitive_root": plan.primitive_root,
                "convolution_length": p - 1,
                "convolution_length_factorization": (
                    prime_resource.factorization
                ),
                "plan_build_seconds": float(plan_build_seconds),
                "plan_build_reported_separately": True,
                "plan_reused_for_all_steady_state_blocks": True,
                "correctness_checks_passed": prime_correct,
                "comparisons": finalized,
            }
        )
        del plan

    context_dft_seconds = []
    context_correlation_seconds = []
    context_dft_digests = []
    context_correlation_digests = []
    for seed in config.seeds:
        _check_deadline(deadline)
        dft_inputs, correlation_left, correlation_right = _generate_inputs(
            seed=seed,
            length=config.generic_numpy_length,
            blocks=config.blocks,
            vectors_per_block=config.vectors_per_block,
            domain=2,
        )
        context_dft_digests.append(_array_digest((dft_inputs,)))
        context_correlation_digests.append(
            _array_digest((correlation_left, correlation_right))
        )
        context_dft_seconds.extend(
            _time_numpy_context_seed(
                operation="dft",
                inputs=(dft_inputs,),
                config=config,
                deadline=deadline,
            )
        )
        context_correlation_seconds.extend(
            _time_numpy_context_seed(
                operation="cyclic_correlation",
                inputs=(correlation_left, correlation_right),
                config=config,
                deadline=deadline,
            )
        )
        del dft_inputs, correlation_left, correlation_right

    total_elapsed = time.perf_counter() - run_started
    _check_deadline(deadline)
    artifact_requested = output_path is not None
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": TIMING_STATUS,
        "config": _jsonable(config),
        "resource_guard": _jsonable(preflight),
        "method_contract": {
            "matched_prime_comparison_label": MATCHED_COMPARISON_LABEL,
            "same_length_dtype_input_and_tolerance_required": True,
            "balanced_interleaved_method_first_order": True,
            "independent_input_per_block": True,
            "plan_build_reported_separately": True,
            "plan_build_included_in_method_timing": False,
            "input_generation_included_in_method_timing": False,
            "warmup_performed": True,
            "dtype": DTYPE_NAME,
            "generic_context_is_matched_prime_comparator": False,
            "timer": "time.perf_counter",
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "numpy_fft_backend": (
                "numpy.fft_backend_not_independently_verified"
            ),
        },
        "prime_results": prime_results,
        "generic_numpy_context": {
            "label": GENERIC_CONTEXT_LABEL,
            "length": config.generic_numpy_length,
            "dtype": DTYPE_NAME,
            "eligible_for_matched_prime_comparison": False,
            "dft_block_seconds": context_dft_seconds,
            "cyclic_correlation_block_seconds": (
                context_correlation_seconds
            ),
            "dft_median_block_seconds": float(
                statistics.median(context_dft_seconds)
            ),
            "cyclic_correlation_median_block_seconds": float(
                statistics.median(context_correlation_seconds)
            ),
            "dft_input_sha256_by_seed": context_dft_digests,
            "cyclic_correlation_input_sha256_by_seed": (
                context_correlation_digests
            ),
            "input_generation_included_in_method_timing": False,
            "warmup_performed": True,
            "correctness_check_scope": (
                "finite_shape_dtype_only_no_cross_method_comparator"
            ),
        },
        "correctness_checks_passed": all_correct,
        "total_elapsed_seconds": float(total_elapsed),
        "total_deadline_seconds": budget.max_seconds,
        "artifact_write_included_in_total_elapsed": False,
        "total_elapsed_is_not_a_benchmark_calibration": True,
        "performance_claim": False,
        "novelty_claim": False,
        "cost_claim": False,
        "hardware_generalization_claim": False,
        "performance_disclaimer": PERFORMANCE_DISCLAIMER,
        "limitations": [
            (
                "CPU affinity, frequency scaling, thermal state, background "
                "load, and NumPy backend threads are not controlled."
            ),
            (
                "The generic NumPy length is context only and is not paired "
                "with either prime-length input."
            ),
            (
                "Correctness is numerical at the declared complex128 "
                "tolerance, not symbolic or bitwise equality."
            ),
            (
                "Modeled work and allocation bytes are not time calibration, "
                "process RSS, allocator peaks, or hardware utilization."
            ),
        ],
        "artifact_write": {
            "requested": artifact_requested,
            "atomic_replace": artifact_requested,
            "path_embedded_in_payload": False,
        },
    }
    if output_path is not None:
        _atomic_write_json(
            Path(output_path),
            result,
            max_output_bytes=budget.max_output_bytes,
            deadline=deadline,
        )
    return _jsonable(result)


def _comma_separated_ints(value: str) -> Tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded RADER-1 matched prime-length timing observations."
        )
    )
    parser.add_argument(
        "--primes",
        type=_comma_separated_ints,
        default=DEFAULT_PRIMES,
    )
    parser.add_argument(
        "--seeds",
        type=_comma_separated_ints,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--vectors-per-block", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--generic-length", type=int, default=4_096)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=HARD_MAX_ESTIMATED_PEAK_BYTES,
    )
    parser.add_argument(
        "--max-work-units",
        type=int,
        default=DEFAULT_MAX_MODELED_WORK_UNITS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON artifact path; no file is written when omitted.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config = RaderHarnessConfig(
        primes=args.primes,
        generic_numpy_length=args.generic_length,
        seeds=args.seeds,
        blocks=args.blocks,
        vectors_per_block=args.vectors_per_block,
        warmup_repetitions=args.warmups,
    )
    budget = RaderHarnessBudget(
        max_estimated_peak_bytes=args.max_bytes,
        max_modeled_work_units=args.max_work_units,
        max_seconds=args.max_seconds,
    )
    result = run_rader_harness(
        config,
        budget,
        output_path=args.output,
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
