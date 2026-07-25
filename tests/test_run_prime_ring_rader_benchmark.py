import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from run_prime_ring_rader_benchmark import (
    DTYPE_NAME,
    GENERIC_CONTEXT_LABEL,
    HARD_MAX_ESTIMATED_PEAK_BYTES,
    HARD_MAX_MODELED_WORK_UNITS,
    HARD_MAX_SECONDS,
    MATCHED_COMPARISON_LABEL,
    PERFORMANCE_DISCLAIMER,
    RaderHarnessBudget,
    RaderHarnessConfig,
    RaderHarnessResourceError,
    RaderHarnessValidationError,
    balanced_order_schedule,
    non_timing_projection,
    preflight_rader_harness,
    run_rader_harness,
)


def _tiny_config():
    return RaderHarnessConfig(
        primes=(7, 11),
        generic_numpy_length=16,
        seeds=(3, 5),
        blocks=2,
        vectors_per_block=1,
        warmup_repetitions=1,
        rtol=1e-10,
        atol=1e-9,
    )


def _tiny_budget(**overrides):
    values = {
        "max_estimated_peak_bytes": 8 * 1024 * 1024,
        "max_modeled_work_units": 2_000_000,
        "max_seconds": 10.0,
        "max_prime_count": 2,
        "max_seed_count": 2,
        "max_blocks": 2,
        "max_vectors_per_block": 1,
        "max_warmups": 1,
        "max_transform_length": 32,
        "max_output_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return RaderHarnessBudget(**values)


class RaderHarnessTests(unittest.TestCase):
    def test_balanced_order_is_interleaved_and_reproducible(self):
        rader_first = balanced_order_schedule(6, rader_first=True)
        numpy_first = balanced_order_schedule(6, rader_first=False)

        self.assertEqual(rader_first, balanced_order_schedule(6))
        self.assertEqual(rader_first[0], ("rader", "numpy"))
        self.assertEqual(rader_first[1], ("numpy", "rader"))
        self.assertEqual(numpy_first[0], ("numpy", "rader"))
        self.assertEqual(
            sum(order[0] == "rader" for order in rader_first),
            3,
        )
        self.assertEqual(
            sum(order[0] == "numpy" for order in rader_first),
            3,
        )
        with self.assertRaises(RaderHarnessValidationError):
            balanced_order_schedule(3)

    def test_tiny_prime_dft_and_correlation_are_correct(self):
        result = run_rader_harness(_tiny_config(), _tiny_budget())

        self.assertTrue(result["correctness_checks_passed"])
        self.assertEqual(len(result["prime_results"]), 2)
        for prime_result in result["prime_results"]:
            self.assertTrue(prime_result["correctness_checks_passed"])
            self.assertTrue(
                prime_result["plan_build_reported_separately"]
            )
            for operation in ("dft", "cyclic_correlation"):
                comparison = prime_result["comparisons"][operation]
                self.assertTrue(comparison["correctness_check_passed"])
                self.assertEqual(
                    comparison["comparison_label"],
                    MATCHED_COMPARISON_LABEL,
                )
                self.assertTrue(comparison["same_length"])
                self.assertTrue(comparison["same_dtype"])
                self.assertTrue(comparison["same_input"])
                self.assertTrue(comparison["same_tolerance"])
                self.assertEqual(comparison["dtype"], DTYPE_NAME)
                self.assertLess(comparison["max_abs_error"], 1e-9)
                self.assertEqual(comparison["timed_block_count"], 4)
                for schedule in comparison["order_schedule_by_seed"]:
                    self.assertEqual(
                        sum(order[0] == "rader" for order in schedule),
                        1,
                    )
                    self.assertEqual(
                        sum(order[0] == "numpy" for order in schedule),
                        1,
                    )

    def test_non_timing_fields_are_reproducible(self):
        first = run_rader_harness(_tiny_config(), _tiny_budget())
        second = run_rader_harness(_tiny_config(), _tiny_budget())

        self.assertEqual(
            non_timing_projection(first),
            non_timing_projection(second),
        )
        for prime_result in first["prime_results"]:
            for comparison in prime_result["comparisons"].values():
                self.assertEqual(
                    len(set(comparison["input_sha256_by_seed"])),
                    2,
                )

    def test_preflight_reports_batch_vector_temp_bytes_and_work(self):
        preflight = preflight_rader_harness(
            _tiny_config(),
            _tiny_budget(),
        )

        self.assertEqual(len(preflight.prime_preflights), 2)
        self.assertGreater(preflight.estimated_peak_bytes, 0)
        self.assertLessEqual(
            preflight.estimated_peak_bytes,
            preflight.max_estimated_peak_bytes,
        )
        self.assertGreater(preflight.modeled_work_units, 0)
        self.assertLessEqual(
            preflight.modeled_work_units,
            preflight.max_modeled_work_units,
        )
        for prime in preflight.prime_preflights:
            self.assertGreater(prime.input_generation_peak_bytes, 0)
            self.assertGreater(prime.output_block_bytes, 0)
            self.assertGreater(
                prime.rader_correlation_temporary_bytes,
                0,
            )
            self.assertEqual(
                prime.estimated_live_peak_bytes,
                prime.rader_correlation_temporary_bytes
                + prime.input_generation_peak_bytes
                + prime.output_block_bytes,
            )
        self.assertTrue(
            preflight.input_batches_materialized_one_seed_at_a_time
        )
        self.assertFalse(preflight.measured_process_peak)
        self.assertFalse(preflight.work_model_is_time_calibration)

    def test_default_preflight_targets_two_primes_and_separate_4096_context(self):
        preflight = preflight_rader_harness()

        self.assertEqual(
            tuple(item.p for item in preflight.prime_preflights),
            (4_091, 4_691),
        )
        self.assertEqual(
            preflight.prime_preflights[0].factorization,
            (2, 5, 409),
        )
        self.assertEqual(
            preflight.prime_preflights[1].factorization,
            (2, 5, 7, 67),
        )
        self.assertEqual(preflight.generic_numpy_length, 4_096)
        self.assertEqual(preflight.estimated_peak_bytes, 5_562_856)
        self.assertEqual(preflight.modeled_work_units, 300_884_380)

    def test_resource_refusal_precedes_plan_or_vector_allocation(self):
        config = _tiny_config()
        with (
            patch(
                "run_prime_ring_rader_benchmark._build_plan"
            ) as plan_builder,
            patch(
                "run_prime_ring_rader_benchmark._generate_inputs"
            ) as input_builder,
            self.assertRaises(RaderHarnessResourceError),
        ):
            run_rader_harness(
                config,
                _tiny_budget(max_estimated_peak_bytes=1),
            )
        plan_builder.assert_not_called()
        input_builder.assert_not_called()

        with self.assertRaises(RaderHarnessResourceError):
            preflight_rader_harness(
                config,
                _tiny_budget(max_modeled_work_units=1),
            )

    def test_total_deadline_fails_before_plan_build(self):
        with (
            patch(
                "run_prime_ring_rader_benchmark.time.perf_counter",
                side_effect=(0.0, 11.0),
            ),
            patch(
                "run_prime_ring_rader_benchmark._build_plan"
            ) as plan_builder,
            self.assertRaises(RaderHarnessResourceError),
        ):
            run_rader_harness(_tiny_config(), _tiny_budget())
        plan_builder.assert_not_called()

    def test_atomic_output_is_written_only_when_requested(self):
        config = RaderHarnessConfig(
            primes=(7,),
            generic_numpy_length=8,
            seeds=(3,),
            blocks=2,
            vectors_per_block=1,
            warmup_repetitions=1,
        )
        budget = RaderHarnessBudget(
            max_estimated_peak_bytes=4 * 1024 * 1024,
            max_modeled_work_units=500_000,
            max_seconds=10.0,
            max_prime_count=1,
            max_seed_count=1,
            max_blocks=2,
            max_vectors_per_block=1,
            max_warmups=1,
            max_transform_length=16,
            max_output_bytes=1024 * 1024,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "nested" / "rader-1.json"
            no_output = run_rader_harness(config, budget)
            self.assertFalse(target.exists())
            self.assertFalse(no_output["artifact_write"]["requested"])

            written = run_rader_harness(
                config,
                budget,
                output_path=target,
            )
            self.assertTrue(target.exists())
            self.assertTrue(written["artifact_write"]["requested"])
            self.assertTrue(written["artifact_write"]["atomic_replace"])
            observed = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(observed, written)
            self.assertEqual(
                list(target.parent.glob("*.tmp")),
                [],
            )

    def test_budget_fields_are_hostile_safe_and_hard_capped(self):
        class HostileInt(int):
            def __gt__(self, _other):
                return False

            def __lt__(self, _other):
                return False

        class HostileFloat(float):
            def __gt__(self, _other):
                return False

            def __lt__(self, _other):
                return False

        budget = RaderHarnessBudget(
            max_estimated_peak_bytes=HostileInt(8 * 1024 * 1024),
            max_modeled_work_units=HostileInt(2_000_000),
            max_seconds=HostileFloat(10.0),
            max_prime_count=HostileInt(2),
            max_seed_count=HostileInt(2),
            max_blocks=HostileInt(2),
            max_vectors_per_block=HostileInt(1),
            max_warmups=HostileInt(1),
            max_transform_length=HostileInt(32),
            max_output_bytes=HostileInt(1024 * 1024),
        )
        for field in (
            "max_estimated_peak_bytes",
            "max_modeled_work_units",
            "max_prime_count",
            "max_seed_count",
            "max_blocks",
            "max_vectors_per_block",
            "max_warmups",
            "max_transform_length",
            "max_output_bytes",
        ):
            self.assertIs(type(getattr(budget, field)), int)
        self.assertIs(type(budget.max_seconds), float)

        for field, value in (
            (
                "max_estimated_peak_bytes",
                HostileInt(HARD_MAX_ESTIMATED_PEAK_BYTES + 1),
            ),
            (
                "max_modeled_work_units",
                HostileInt(HARD_MAX_MODELED_WORK_UNITS + 1),
            ),
            (
                "max_seconds",
                HostileFloat(HARD_MAX_SECONDS + 1.0),
            ),
        ):
            with self.subTest(field=field):
                with self.assertRaises(RaderHarnessValidationError):
                    RaderHarnessBudget(**{field: value})

    def test_labels_and_claim_boundaries_are_honest(self):
        result = run_rader_harness(_tiny_config(), _tiny_budget())

        self.assertEqual(
            result["status"],
            "BOUNDED_TIMING_OBSERVATIONS_ONLY",
        )
        self.assertEqual(
            result["generic_numpy_context"]["label"],
            GENERIC_CONTEXT_LABEL,
        )
        self.assertFalse(
            result["generic_numpy_context"][
                "eligible_for_matched_prime_comparison"
            ]
        )
        self.assertFalse(result["performance_claim"])
        self.assertFalse(result["novelty_claim"])
        self.assertFalse(result["cost_claim"])
        self.assertFalse(result["hardware_generalization_claim"])
        self.assertEqual(
            result["performance_disclaimer"],
            PERFORMANCE_DISCLAIMER,
        )
        self.assertEqual(
            result["environment"]["numpy_fft_backend"],
            "numpy.fft_backend_not_independently_verified",
        )
        self.assertTrue(
            any("CPU affinity" in item for item in result["limitations"])
        )
        self.assertTrue(
            any(
                "not time calibration" in item
                for item in result["limitations"]
            )
        )
        contract = result["method_contract"]
        self.assertTrue(
            contract["same_length_dtype_input_and_tolerance_required"]
        )
        self.assertFalse(
            contract["generic_context_is_matched_prime_comparator"]
        )
        self.assertFalse(
            contract["input_generation_included_in_method_timing"]
        )
        self.assertFalse(
            contract["plan_build_included_in_method_timing"]
        )

    def test_invalid_configuration_fails_closed(self):
        with self.assertRaises(RaderHarnessValidationError):
            RaderHarnessConfig(blocks=3)
        with self.assertRaises(RaderHarnessValidationError):
            RaderHarnessConfig(dtype="complex64")
        with self.assertRaises(RaderHarnessValidationError):
            RaderHarnessConfig(primes=(7, 7))
        with self.assertRaises(RaderHarnessValidationError):
            RaderHarnessConfig(seeds=(True,))
        with self.assertRaises(RaderHarnessValidationError):
            RaderHarnessConfig(rtol=np.nan)


if __name__ == "__main__":
    unittest.main()
