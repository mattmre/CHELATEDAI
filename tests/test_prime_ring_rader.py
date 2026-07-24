from __future__ import annotations

import unittest

import numpy as np

from prime_ring_rader import (
    DEFAULT_MAX_TEMPORARY_BYTES,
    MAX_BENCHMARK_REPETITIONS,
    MAX_BENCHMARK_SECONDS,
    RADER_PERFORMANCE_DISCLAIMER,
    RaderResourceError,
    RaderValidationError,
    estimate_rader_temporary_bytes,
    preflight_rader,
    rader_cyclic_correlation,
    rader_dft,
    run_rader_microbenchmark,
)


def _direct_dft(values: np.ndarray) -> np.ndarray:
    p = values.size
    coordinates = np.arange(p)
    matrix = np.exp(
        -2j
        * np.pi
        * coordinates[:, np.newaxis]
        * coordinates[np.newaxis, :]
        / p
    )
    return matrix @ values


def _direct_cyclic_correlation(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    p = left.size
    return np.asarray(
        [
            sum(
                left[index] * np.conjugate(right[(index - shift) % p])
                for index in range(p)
            )
            for shift in range(p)
        ],
        dtype=np.complex128,
    )


class TestRaderDFT(unittest.TestCase):
    def test_matches_direct_dft_and_numpy_fft_on_small_primes(self):
        for p in (7, 11, 31):
            with self.subTest(p=p):
                rng = np.random.default_rng(20260724 + p)
                values = rng.normal(size=p) + 1j * rng.normal(size=p)
                observed = rader_dft(values)
                np.testing.assert_allclose(
                    observed,
                    _direct_dft(values),
                    rtol=1e-10,
                    atol=1e-10,
                )
                np.testing.assert_allclose(
                    observed,
                    np.fft.fft(values),
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_real_input_and_explicit_primitive_root(self):
        values = np.arange(11, dtype=np.float64) - 3.0
        observed = rader_dft(values, primitive_root=2)
        np.testing.assert_allclose(
            observed,
            np.fft.fft(values),
            rtol=1e-11,
            atol=1e-11,
        )
        self.assertEqual(observed.dtype, np.dtype(np.complex128))

    def test_impulse_has_unit_spectrum_and_dc_is_sum(self):
        values = np.zeros(31)
        values[0] = 1.0
        observed = rader_dft(values)
        np.testing.assert_allclose(observed, np.ones(31), atol=1e-12)

        shifted = np.arange(7, dtype=np.float64)
        transformed = rader_dft(shifted)
        self.assertAlmostEqual(transformed[0].real, float(np.sum(shifted)), places=12)
        self.assertAlmostEqual(transformed[0].imag, 0.0, places=12)


class TestRaderCyclicCorrelation(unittest.TestCase):
    def test_matches_direct_correlation_and_numpy_fft(self):
        for p in (7, 11, 31):
            with self.subTest(p=p):
                rng = np.random.default_rng(9173 + p)
                left = rng.normal(size=p) + 1j * rng.normal(size=p)
                right = rng.normal(size=p) + 1j * rng.normal(size=p)
                observed = rader_cyclic_correlation(left, right)
                direct = _direct_cyclic_correlation(left, right)
                numpy_result = np.fft.ifft(
                    np.fft.fft(left) * np.conjugate(np.fft.fft(right))
                )
                np.testing.assert_allclose(
                    observed, direct, rtol=1e-10, atol=1e-10
                )
                np.testing.assert_allclose(
                    observed, numpy_result, rtol=1e-10, atol=1e-10
                )

    def test_positive_roll_peaks_at_positive_shift(self):
        rng = np.random.default_rng(7711)
        reference = rng.normal(size=31)
        shift = 9
        query = np.roll(reference, shift)
        correlation = rader_cyclic_correlation(query, reference)
        self.assertEqual(int(np.argmax(correlation.real)), shift)
        np.testing.assert_allclose(
            correlation,
            np.fft.ifft(
                np.fft.fft(query) * np.conjugate(np.fft.fft(reference))
            ),
            rtol=1e-10,
            atol=1e-10,
        )


class TestRaderPreflight(unittest.TestCase):
    def test_small_prime_preflight_is_conservative_and_nonpromotional(self):
        dft = estimate_rader_temporary_bytes(31, operation="dft")
        correlation = estimate_rader_temporary_bytes(
            31, operation="correlation"
        )
        self.assertTrue(dft.allowed)
        self.assertTrue(correlation.allowed)
        self.assertGreater(
            correlation.estimated_peak_temporary_bytes,
            dft.estimated_peak_temporary_bytes,
        )
        self.assertEqual(dft.primitive_length_factorization, (2, 3, 5))
        self.assertFalse(dft.measured_process_peak)
        self.assertFalse(dft.smooth_length_performance_claim)
        self.assertEqual(dft.performance_disclaimer, RADER_PERFORMANCE_DISCLAIMER)
        self.assertLess(
            dft.estimated_peak_temporary_bytes,
            DEFAULT_MAX_TEMPORARY_BYTES,
        )

    def test_low_memory_limit_refuses_before_transform(self):
        estimate = estimate_rader_temporary_bytes(
            31,
            max_temporary_bytes=1024,
        )
        self.assertFalse(estimate.allowed)
        with self.assertRaises(RaderResourceError):
            preflight_rader(31, max_temporary_bytes=1024)
        with self.assertRaises(RaderResourceError):
            rader_dft(np.ones(31), max_temporary_bytes=1024)
        with self.assertRaises(RaderResourceError):
            rader_cyclic_correlation(
                np.ones(31),
                np.ones(31),
                max_temporary_bytes=1024,
            )

    def test_hard_memory_policy_cannot_be_raised(self):
        with self.assertRaises(RaderValidationError):
            preflight_rader(
                31,
                max_temporary_bytes=DEFAULT_MAX_TEMPORARY_BYTES + 1,
            )

    def test_estimate_over_512_mib_refuses_without_transform_allocation(self):
        # 2,500,009 is prime, but this path performs shape arithmetic only:
        # the estimate is rejected before plan or transform arrays are built.
        estimate = estimate_rader_temporary_bytes(2_500_009)
        self.assertFalse(estimate.allowed)
        self.assertGreater(
            estimate.estimated_peak_temporary_bytes,
            DEFAULT_MAX_TEMPORARY_BYTES,
        )
        with self.assertRaises(RaderResourceError):
            preflight_rader(2_500_009)


class TestRaderMicrobenchmark(unittest.TestCase):
    def test_small_benchmark_is_bounded_and_not_a_performance_claim(self):
        result = run_rader_microbenchmark(
            7,
            repetitions=2,
            duration_limit_seconds=2.0,
            seed=7,
        )
        self.assertEqual(result.p, 7)
        self.assertEqual(result.repetitions, 2)
        self.assertTrue(result.correctness_check_passed)
        self.assertFalse(result.performance_claim)
        self.assertFalse(result.plan_build_included_in_timing)
        self.assertFalse(result.input_generation_included_in_timing)
        self.assertFalse(result.warmup_performed)
        self.assertTrue(result.same_input_reused)
        self.assertEqual(
            result.performance_disclaimer,
            RADER_PERFORMANCE_DISCLAIMER,
        )
        self.assertLessEqual(result.elapsed_seconds, 2.0)
        self.assertLess(result.max_abs_error, 1e-10)

    def test_duration_repetition_and_memory_guards_refuse(self):
        with self.assertRaises(RaderValidationError):
            run_rader_microbenchmark(
                7,
                repetitions=MAX_BENCHMARK_REPETITIONS + 1,
            )
        with self.assertRaises(RaderValidationError):
            run_rader_microbenchmark(
                7,
                duration_limit_seconds=MAX_BENCHMARK_SECONDS + 0.01,
            )
        with self.assertRaises(RaderResourceError):
            run_rader_microbenchmark(
                31,
                max_temporary_bytes=1024,
            )


class TestRaderValidation(unittest.TestCase):
    def test_composite_short_boolean_and_multidimensional_inputs_fail(self):
        for values in (
            np.ones(9),
            np.ones(2),
            np.ones(7, dtype=np.bool_),
            np.ones((1, 7)),
        ):
            with self.subTest(shape=values.shape, dtype=str(values.dtype)):
                with self.assertRaises(RaderValidationError):
                    rader_dft(values)
        with self.assertRaises(RaderValidationError):
            preflight_rader(True)

    def test_nonfinite_ragged_and_mismatched_inputs_fail(self):
        nonfinite = np.ones(7, dtype=np.complex128)
        nonfinite[3] = np.nan + 1j
        with self.assertRaises(RaderValidationError):
            rader_dft(nonfinite)
        with self.assertRaises(RaderValidationError):
            rader_dft([[1.0, 2.0], [3.0]])
        with self.assertRaises(RaderValidationError):
            rader_cyclic_correlation(np.ones(7), np.ones(11))

    def test_invalid_primitive_root_operation_and_scalar_limits_fail(self):
        with self.assertRaises(RaderValidationError):
            rader_dft(np.ones(7), primitive_root=2)
        with self.assertRaises(RaderValidationError):
            estimate_rader_temporary_bytes(7, operation="inverse-ish")
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(limit=invalid):
                with self.assertRaises(RaderValidationError):
                    preflight_rader(7, max_temporary_bytes=invalid)
        with self.assertRaises(RaderValidationError):
            run_rader_microbenchmark(7, duration_limit_seconds=True)


if __name__ == "__main__":
    unittest.main()
