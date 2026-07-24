"""Correctness-first Rader transforms for bounded prime-ring experiments.

Rader's construction rewrites a prime-length DFT as a cyclic convolution of
length ``p - 1``.  This module uses NumPy FFTs for that convolution and keeps
the indexing explicit so it can be checked against a direct DFT.

A smooth factorization of ``p - 1`` only describes the convolution length.  It
is not, by itself, evidence that Rader is faster, cheaper, or preferable to a
prime-length FFT.  Any performance statement requires a separately controlled
benchmark on the target hardware.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from prime_ring_waypoint import (
    factor_integer_exact,
    is_prime_exact,
    is_primitive_root_exact,
)


DEFAULT_MAX_TEMPORARY_BYTES = 512 * 1024 * 1024
MAX_BENCHMARK_SECONDS = 120.0
MAX_BENCHMARK_REPETITIONS = 120
RADER_PERFORMANCE_DISCLAIMER = (
    "Smooth p-1 only describes the Rader convolution length; it is not a "
    "performance claim."
)


class RaderValidationError(ValueError):
    """Raised when a Rader input or configuration violates the contract."""


class RaderResourceError(RuntimeError):
    """Raised before work when a configured resource boundary is exceeded."""


@dataclass(frozen=True)
class RaderPreflight:
    """Conservative allocation estimate for one guarded operation."""

    p: int
    convolution_length: int
    operation: str
    primitive_length_factorization: Tuple[int, ...]
    estimated_peak_temporary_bytes: int
    max_temporary_bytes: int
    allowed: bool
    component_bytes: Tuple[Tuple[str, int], ...]
    measured_process_peak: bool = False
    smooth_length_performance_claim: bool = False
    performance_disclaimer: str = RADER_PERFORMANCE_DISCLAIMER


@dataclass(frozen=True)
class RaderBenchmarkResult:
    """Bounded timing observations; never a general performance conclusion."""

    p: int
    primitive_root: int
    repetitions: int
    elapsed_seconds: float
    rader_elapsed_seconds: float
    numpy_elapsed_seconds: float
    rader_seconds_per_transform: float
    numpy_seconds_per_transform: float
    max_abs_error: float
    estimated_peak_temporary_bytes: int
    max_temporary_bytes: int
    correctness_check_passed: bool
    plan_build_included_in_timing: bool = False
    input_generation_included_in_timing: bool = False
    warmup_performed: bool = False
    same_input_reused: bool = True
    performance_claim: bool = False
    performance_disclaimer: str = RADER_PERFORMANCE_DISCLAIMER


@dataclass(frozen=True)
class _RaderPlan:
    p: int
    primitive_root: int
    powers: np.ndarray
    reverse_indices: np.ndarray
    kernel_spectrum: np.ndarray


def _plain_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise RaderValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise RaderValidationError(f"{name} must be >= {minimum}")
    return result


def _finite_positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise RaderValidationError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RaderValidationError(f"{name} must be a finite positive number")
    return result


def _raw_numeric_vector(values: Any, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise RaderValidationError(f"{name} must be a one-dimensional numeric array")
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise RaderValidationError(
            f"{name} must be a rectangular one-dimensional numeric array"
        ) from exc
    if array.ndim != 1:
        raise RaderValidationError(f"{name} must be one-dimensional")
    if array.size < 3:
        raise RaderValidationError(f"{name} length must be an odd prime >= 3")
    if array.dtype.kind not in "iufc" or array.dtype.kind == "b":
        raise RaderValidationError(f"{name} must contain non-boolean numeric values")
    return array


def _complex_vector(array: np.ndarray, name: str) -> np.ndarray:
    converted = np.ascontiguousarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(converted.real)) or not np.all(
        np.isfinite(converted.imag)
    ):
        raise RaderValidationError(f"{name} must contain only finite values")
    return converted


def _validated_memory_limit(value: object) -> int:
    limit = _plain_int(value, "max_temporary_bytes", 1)
    if limit > DEFAULT_MAX_TEMPORARY_BYTES:
        raise RaderValidationError(
            "max_temporary_bytes cannot exceed the hard 512 MiB policy"
        )
    return limit


def _memory_components(p: int, operation: str) -> Tuple[Tuple[str, int], ...]:
    convolution_length = p - 1
    complex_bytes = np.dtype(np.complex128).itemsize
    index_bytes = np.dtype(np.int64).itemsize
    components = [
        ("input_complex", p * complex_bytes),
        ("output_complex", p * complex_bytes),
        ("power_permutation", convolution_length * index_bytes),
        ("reverse_permutation", convolution_length * index_bytes),
        ("power_permutation_float64", convolution_length * 8),
        ("rader_kernel_during_plan_build", convolution_length * complex_bytes),
        ("kernel_spectrum", convolution_length * complex_bytes),
        ("reordered_input", convolution_length * complex_bytes),
        ("reversed_input", convolution_length * complex_bytes),
        ("reordered_input_spectrum", convolution_length * complex_bytes),
        ("frequency_product", convolution_length * complex_bytes),
        ("cyclic_convolution", convolution_length * complex_bytes),
        (
            "fft_workspace_allowance",
            4 * convolution_length * complex_bytes,
        ),
    ]
    if operation == "correlation":
        components.append(
            ("retained_correlation_spectra_and_conjugates", 5 * p * complex_bytes)
        )
    elif operation == "benchmark":
        components.append(("numpy_reference_and_comparison", 2 * p * complex_bytes))
    return tuple((name, int(size)) for name, size in components)


def estimate_rader_temporary_bytes(
    p: int,
    *,
    operation: str = "dft",
    max_temporary_bytes: int = DEFAULT_MAX_TEMPORARY_BYTES,
) -> RaderPreflight:
    """Estimate and classify a Rader operation without allocating its arrays.

    The estimate deliberately sums a conservative set of arrays, including an
    FFT-workspace allowance.  It is shape accounting, not measured process RSS.
    Inputs above the memory boundary are refused before an exact primality test,
    preventing an expensive validation attempt for an already-disallowed job.
    """

    prime_candidate = _plain_int(p, "p", 3)
    if operation not in ("dft", "correlation", "benchmark"):
        raise RaderValidationError(
            "operation must be 'dft', 'correlation', or 'benchmark'"
        )
    limit = _validated_memory_limit(max_temporary_bytes)
    components = _memory_components(prime_candidate, operation)
    estimate = int(sum(size for _name, size in components))
    allowed = estimate <= limit
    if allowed and not is_prime_exact(prime_candidate):
        raise RaderValidationError("p must be an odd prime")
    factorization = (
        factor_integer_exact(prime_candidate - 1) if allowed else ()
    )
    return RaderPreflight(
        p=prime_candidate,
        convolution_length=prime_candidate - 1,
        operation=operation,
        primitive_length_factorization=factorization,
        estimated_peak_temporary_bytes=estimate,
        max_temporary_bytes=limit,
        allowed=allowed,
        component_bytes=components,
    )


def preflight_rader(
    p: int,
    *,
    operation: str = "dft",
    max_temporary_bytes: int = DEFAULT_MAX_TEMPORARY_BYTES,
) -> RaderPreflight:
    """Return an allowed preflight or raise before allocating transform arrays."""

    result = estimate_rader_temporary_bytes(
        p,
        operation=operation,
        max_temporary_bytes=max_temporary_bytes,
    )
    if not result.allowed:
        raise RaderResourceError(
            "Rader {} estimate {} bytes exceeds limit {} bytes".format(
                operation,
                result.estimated_peak_temporary_bytes,
                result.max_temporary_bytes,
            )
        )
    return result


def _primitive_root(p: int, requested: Optional[int]) -> int:
    if requested is not None:
        root = _plain_int(requested, "primitive_root", 2)
        if root >= p or not is_primitive_root_exact(root, p):
            raise RaderValidationError(
                "primitive_root must generate the nonzero residues modulo p"
            )
        return root
    for candidate in range(2, p):
        if is_primitive_root_exact(candidate, p):
            return candidate
    raise RaderValidationError("failed to find a primitive root for p")


def _build_plan(p: int, primitive_root: Optional[int]) -> _RaderPlan:
    root = _primitive_root(p, primitive_root)
    convolution_length = p - 1
    powers = np.fromiter(
        (pow(root, exponent, p) for exponent in range(convolution_length)),
        dtype=np.int64,
        count=convolution_length,
    )
    # `_primitive_root` has already proved the generator order exactly.
    # Rechecking with `unique` or a Python set would create a large, unbudgeted
    # allocation that defeats the preflight guard.
    reverse_indices = (-np.arange(convolution_length, dtype=np.int64)) % (
        convolution_length
    )
    kernel = np.exp(-2j * np.pi * powers.astype(np.float64) / p)
    kernel_spectrum = np.fft.fft(kernel)
    for array in (powers, reverse_indices, kernel_spectrum):
        array.setflags(write=False)
    return _RaderPlan(
        p=p,
        primitive_root=root,
        powers=powers,
        reverse_indices=reverse_indices,
        kernel_spectrum=kernel_spectrum,
    )


def _rader_dft_with_plan(values: np.ndarray, plan: _RaderPlan) -> np.ndarray:
    # With n_j = g**j and b_j = exp(-2*pi*i*g**j/p),
    # X[g**m] = x[0] + sum_j x[n_j] b[(m+j) mod (p-1)].
    # Reversing the reordered input converts that plus-index correlation into
    # the ordinary cyclic convolution evaluated below.
    reordered = values[plan.powers]
    reversed_input = reordered[plan.reverse_indices]
    reordered_spectrum = np.fft.fft(reversed_input)
    frequency_product = reordered_spectrum * plan.kernel_spectrum
    convolution = np.fft.ifft(frequency_product)
    result = np.empty(plan.p, dtype=np.complex128)
    result[0] = np.sum(values, dtype=np.complex128)
    result[plan.powers] = values[0] + convolution
    return result


def rader_dft(
    values: Any,
    *,
    primitive_root: Optional[int] = None,
    max_temporary_bytes: int = DEFAULT_MAX_TEMPORARY_BYTES,
) -> np.ndarray:
    """Return the NumPy-convention forward DFT of an odd-prime-length vector."""

    raw = _raw_numeric_vector(values, "values")
    p = int(raw.size)
    preflight_rader(
        p,
        operation="dft",
        max_temporary_bytes=max_temporary_bytes,
    )
    vector = _complex_vector(raw, "values")
    plan = _build_plan(p, primitive_root)
    return _rader_dft_with_plan(vector, plan)


def rader_cyclic_correlation(
    left: Any,
    right: Any,
    *,
    primitive_root: Optional[int] = None,
    max_temporary_bytes: int = DEFAULT_MAX_TEMPORARY_BYTES,
) -> np.ndarray:
    """Return ``ifft(fft(left) * conj(fft(right)))`` using Rader transforms.

    Consequently, output shift ``s`` is
    ``sum_n left[n] * conj(right[(n - s) mod p])``.  A positive ``np.roll`` of
    ``right`` therefore peaks at that same positive shift.
    """

    left_raw = _raw_numeric_vector(left, "left")
    right_raw = _raw_numeric_vector(right, "right")
    if left_raw.shape != right_raw.shape:
        raise RaderValidationError("left and right must have identical shapes")
    p = int(left_raw.size)
    preflight_rader(
        p,
        operation="correlation",
        max_temporary_bytes=max_temporary_bytes,
    )
    left_vector = _complex_vector(left_raw, "left")
    right_vector = _complex_vector(right_raw, "right")
    plan = _build_plan(p, primitive_root)
    left_spectrum = _rader_dft_with_plan(left_vector, plan)
    right_spectrum = _rader_dft_with_plan(right_vector, plan)
    spectral_product = left_spectrum * np.conjugate(right_spectrum)
    inverse_input = np.conjugate(spectral_product)
    inverse_transform = _rader_dft_with_plan(inverse_input, plan)
    return np.conjugate(inverse_transform) / p


def run_rader_microbenchmark(
    p: int,
    *,
    repetitions: int = 3,
    duration_limit_seconds: float = 5.0,
    seed: int = 0,
    primitive_root: Optional[int] = None,
    max_temporary_bytes: int = DEFAULT_MAX_TEMPORARY_BYTES,
) -> RaderBenchmarkResult:
    """Run a small guarded DFT comparison against ``numpy.fft.fft``.

    This helper refuses requests over 120 repetitions or 120 seconds and checks
    the conservative memory estimate before allocating benchmark vectors.  Its
    output is an observation for the current process, not a speed claim.
    """

    count = _plain_int(repetitions, "repetitions", 1)
    if count > MAX_BENCHMARK_REPETITIONS:
        raise RaderValidationError(
            f"repetitions cannot exceed {MAX_BENCHMARK_REPETITIONS}"
        )
    duration = _finite_positive_real(
        duration_limit_seconds, "duration_limit_seconds"
    )
    if duration > MAX_BENCHMARK_SECONDS:
        raise RaderValidationError(
            f"duration_limit_seconds cannot exceed {MAX_BENCHMARK_SECONDS:g}"
        )
    random_seed = _plain_int(seed, "seed", 0)
    preflight = preflight_rader(
        p,
        operation="benchmark",
        max_temporary_bytes=max_temporary_bytes,
    )
    plan = _build_plan(preflight.p, primitive_root)
    rng = np.random.default_rng(random_seed)
    vector = np.ascontiguousarray(
        rng.normal(size=preflight.p) + 1j * rng.normal(size=preflight.p),
        dtype=np.complex128,
    )
    reference = np.fft.fft(vector)
    started = time.perf_counter()
    deadline = started + duration
    rader_elapsed = 0.0
    numpy_elapsed = 0.0
    max_abs_error = 0.0
    for _index in range(count):
        if time.perf_counter() >= deadline:
            raise RaderResourceError(
                "microbenchmark duration limit reached before all repetitions"
            )
        rader_started = time.perf_counter()
        observed = _rader_dft_with_plan(vector, plan)
        rader_elapsed += time.perf_counter() - rader_started
        numpy_started = time.perf_counter()
        numpy_observed = np.fft.fft(vector)
        numpy_elapsed += time.perf_counter() - numpy_started
        error = float(np.max(np.abs(observed - numpy_observed)))
        max_abs_error = max(max_abs_error, error)
        if not np.allclose(observed, reference, rtol=1e-10, atol=1e-10):
            raise RuntimeError("Rader correctness check failed during benchmark")
        if time.perf_counter() > deadline:
            raise RaderResourceError(
                "microbenchmark duration limit exceeded during a repetition"
            )
    elapsed = time.perf_counter() - started
    return RaderBenchmarkResult(
        p=preflight.p,
        primitive_root=plan.primitive_root,
        repetitions=count,
        elapsed_seconds=float(elapsed),
        rader_elapsed_seconds=float(rader_elapsed),
        numpy_elapsed_seconds=float(numpy_elapsed),
        rader_seconds_per_transform=float(rader_elapsed / count),
        numpy_seconds_per_transform=float(numpy_elapsed / count),
        max_abs_error=max_abs_error,
        estimated_peak_temporary_bytes=preflight.estimated_peak_temporary_bytes,
        max_temporary_bytes=preflight.max_temporary_bytes,
        correctness_check_passed=True,
    )


__all__ = [
    "DEFAULT_MAX_TEMPORARY_BYTES",
    "MAX_BENCHMARK_REPETITIONS",
    "MAX_BENCHMARK_SECONDS",
    "RADER_PERFORMANCE_DISCLAIMER",
    "RaderBenchmarkResult",
    "RaderPreflight",
    "RaderResourceError",
    "RaderValidationError",
    "estimate_rader_temporary_bytes",
    "preflight_rader",
    "rader_cyclic_correlation",
    "rader_dft",
    "run_rader_microbenchmark",
]
