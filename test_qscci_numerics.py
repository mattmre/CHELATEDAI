import hashlib
import inspect
import math
import unittest
from decimal import Decimal

import torch

from qscci_experiment import (
    COSINE_TOLERANCE,
    QSCCIError,
    apply_bf16_update,
    canonical_delta_norms,
    canonicalize_direction_cosine,
    deterministic_topk,
    feature_digest,
    q9,
    reference_delta_components,
)


class TestQSCCIQuantization(unittest.TestCase):
    def test_q9_returns_frozen_nine_place_decimal(self):
        self.assertEqual(q9(0.05), Decimal("0.050000000"))
        self.assertEqual(q9(-0.0), Decimal("-0E-9"))

    def test_q9_uses_decimal_from_float_and_half_even(self):
        values = (1.2345678904, 1.2345678905, 1.2345678906, -1.2345678905)
        expected = tuple(
            Decimal.from_float(float(value)).quantize(Decimal("0.000000001"), rounding="ROUND_HALF_EVEN")
            for value in values
        )
        self.assertEqual(tuple(q9(value) for value in values), expected)

    def test_q9_rejects_nonfinite_and_non_numeric_values(self):
        for value in (math.nan, math.inf, -math.inf, None, "0.5", True):
            with self.subTest(value=value), self.assertRaises((QSCCIError, TypeError, ValueError)):
                q9(value)


class TestQSCCICosineCanonicalization(unittest.TestCase):
    def test_frozen_tolerance_is_exact(self):
        self.assertEqual(COSINE_TOLERANCE, 1e-12)

    def test_in_range_values_are_preserved_exactly(self):
        for value in (-1.0, -0.25, -0.0, 0.0, 0.25, 1.0):
            with self.subTest(value=value):
                result = canonicalize_direction_cosine(value)
                self.assertEqual(result, value)
                self.assertEqual(math.copysign(1.0, result), math.copysign(1.0, value))

    def test_one_ulp_overflow_is_canonicalized_symmetrically(self):
        self.assertEqual(canonicalize_direction_cosine(math.nextafter(1.0, math.inf)), 1.0)
        self.assertEqual(canonicalize_direction_cosine(math.nextafter(-1.0, -math.inf)), -1.0)

    def test_inclusive_envelope_boundary_is_canonicalized(self):
        self.assertEqual(canonicalize_direction_cosine(1.0 + COSINE_TOLERANCE), 1.0)
        self.assertEqual(canonicalize_direction_cosine(-1.0 - COSINE_TOLERANCE), -1.0)

    def test_first_representable_value_beyond_envelope_is_rejected_symmetrically(self):
        positive = math.nextafter(1.0 + COSINE_TOLERANCE, math.inf)
        negative = math.nextafter(-1.0 - COSINE_TOLERANCE, -math.inf)
        for value in (positive, negative):
            with self.subTest(value=value), self.assertRaisesRegex(QSCCIError, "frozen numerical envelope"):
                canonicalize_direction_cosine(value)

    def test_nonfinite_bool_and_non_numeric_values_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, True, False, None, "1.0"):
            with self.subTest(value=value), self.assertRaises(QSCCIError):
                canonicalize_direction_cosine(value)


class TestQSCCIDeterministicTopK(unittest.TestCase):
    def test_topk_uses_descending_fp32_value_then_lower_feature_id(self):
        preactivation = [3.0, -2.0, 3.0, 1.0, 3.0]
        self.assertEqual(
            deterministic_topk(preactivation, k=4),
            ((0, 3.0), (2, 3.0), (4, 3.0), (3, 1.0)),
        )

    def test_topk_preserves_signed_negative_values_without_relu(self):
        preactivation = [-4.0, -1.0, -3.0, -2.0]
        self.assertEqual(deterministic_topk(preactivation, k=2), ((1, -1.0), (3, -2.0)))

    def test_topk_is_repeatable_and_does_not_mutate_input(self):
        preactivation = [0.0, 1.0, 1.0, -1.0]
        before = list(preactivation)
        first = deterministic_topk(preactivation, k=3)
        second = deterministic_topk(preactivation, k=3)
        self.assertEqual(first, second)
        self.assertEqual(preactivation, before)

    def test_topk_rejects_nonfinite_invalid_shape_dtype_and_k(self):
        cases = (
            ([0.0, math.nan], 1),
            ([0.0, math.inf], 1),
            ([[0.0, 1.0]], 1),
            ([0.0, 1.0], 0),
            ([0.0, 1.0], 3),
        )
        for preactivation, k in cases:
            with self.subTest(preactivation=preactivation, k=k):
                with self.assertRaises((QSCCIError, TypeError, ValueError)):
                    deterministic_topk(preactivation, k=k)


class TestQSCCIRandomDigest(unittest.TestCase):
    def test_feature_digest_matches_frozen_ascii_sha256_bytes(self):
        expected = hashlib.sha256(b"uniform_random|1701|4|100").digest()
        actual = feature_digest("uniform_random", 1701, 4, 100)
        self.assertIsInstance(actual, bytes)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.hex(), "33c211d2ff9e425627a2e8d3bb5b9c75d03a5a492445be75571ccb00cdf442b9")

    def test_feature_digest_ranking_has_frozen_library_independent_order(self):
        eligible = (2, 5, 9, 12, 23, 44, 100, 101, 32767)
        ordered = sorted(eligible, key=lambda feature_id: (feature_digest("uniform_random", 1701, 4, feature_id), feature_id))
        self.assertEqual(ordered[:4], [100, 101, 23, 32767])

    def test_feature_digest_rejects_noncanonical_identifiers(self):
        cases = (
            ("Uniform_Random", 1701, 4, 100),
            ("uniform_random", -1, 4, 100),
            ("uniform_random", 1701, 0, 100),
            ("uniform_random", 1701, 4, -1),
            ("uniform_random", True, 4, 100),
            ("uniform_random", 1701.0, 4, 100),
        )
        for args in cases:
            with self.subTest(args=args), self.assertRaises((QSCCIError, TypeError, ValueError)):
                feature_digest(*args)


class TestQSCCIBF16HookUpdate(unittest.TestCase):
    def test_update_changes_only_each_last_nonpadding_token_and_stays_bf16(self):
        hidden = torch.ones((2, 4, 4), dtype=torch.bfloat16)
        direction = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        modified, evidence = apply_bf16_update(
            hidden,
            direction,
            labels=torch.tensor([1.0, -1.0], dtype=torch.float32),
            alpha=2.0,
            scale=1.0,
            last_token_indices=torch.tensor([3, 1]),
        )
        self.assertEqual(modified.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(modified[0, :3], hidden[0, :3]))
        self.assertTrue(torch.equal(modified[1, 0], hidden[1, 0]))
        self.assertTrue(torch.equal(modified[1, 2:], hidden[1, 2:]))
        self.assertGreater(float(modified[0, 3, 0]), float(hidden[0, 3, 0]))
        self.assertLess(float(modified[1, 1, 0]), float(hidden[1, 1, 0]))
        self.assertGreater(evidence["shared_delta_norm"], 0.0)
        self.assertEqual(len(evidence["cast_delta_norms"]), 2)
        self.assertEqual(len(evidence["realized_update_norms"]), 2)
        for value in evidence["cast_delta_norms"]:
            self.assertGreater(value, 0.0)
        for value in evidence["realized_update_norms"]:
            self.assertGreater(value, 0.0)

    def test_update_rejects_vanished_bf16_prompt_update(self):
        hidden = torch.full((1, 1, 2), 4096.0, dtype=torch.bfloat16)
        direction = torch.tensor([1.0, 0.0], dtype=torch.float32)
        with self.assertRaises(QSCCIError):
            apply_bf16_update(
                hidden,
                direction,
                labels=torch.tensor([1.0], dtype=torch.float32),
                alpha=0.5,
                scale=1e-6,
                last_token_indices=torch.tensor([0]),
            )

    def test_update_rejects_nonfinite_direction_scale_and_invalid_indices(self):
        hidden = torch.ones((1, 2, 2), dtype=torch.bfloat16)
        base = {
            "hidden": hidden,
            "direction": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "labels": torch.tensor([1.0], dtype=torch.float32),
            "alpha": 1.0,
            "scale": 1.0,
            "last_token_indices": torch.tensor([1]),
        }
        mutations = (
            {"direction": torch.tensor([math.nan, 0.0], dtype=torch.float32)},
            {"scale": math.inf},
            {"alpha": math.nan},
            {"last_token_indices": torch.tensor([2])},
            {"labels": torch.tensor([0.0], dtype=torch.float32)},
        )
        for mutation in mutations:
            with self.subTest(mutation=tuple(mutation)), self.assertRaises((QSCCIError, TypeError, ValueError)):
                apply_bf16_update(**(base | mutation))


class TestQSCCIRetainedDeltaNormContract(unittest.TestCase):
    @staticmethod
    def _direction():
        generator = torch.Generator(device="cpu").manual_seed(1721)
        raw = torch.randn(2048, dtype=torch.float32, generator=generator)
        return raw / torch.linalg.vector_norm(raw)

    @staticmethod
    def _frozen_expected(direction, alpha, scale):
        delta = (
            torch.tensor(float(alpha), dtype=torch.float32)
            * torch.tensor(float(scale), dtype=torch.float32)
            * direction
        )
        return (
            float(torch.linalg.vector_norm(delta)),
            float(torch.linalg.vector_norm(delta.to(torch.bfloat16).float())),
        )

    def test_cpu_retained_norms_exactly_reconstruct_from_alpha_scale_and_unit_direction(self):
        direction = self._direction()
        alpha = 0.5
        scale = float(torch.tensor(0.137, dtype=torch.float32))
        hidden = torch.ones((2, 1, 2048), dtype=torch.bfloat16)
        _, evidence = apply_bf16_update(hidden, direction, [1, -1], alpha, scale, [0, 0])
        expected_shared, expected_cast = self._frozen_expected(direction, alpha, scale)
        self.assertEqual(evidence["shared_delta_norm"], expected_shared)
        self.assertEqual(evidence["cast_delta_norms"], [expected_cast, expected_cast])

    def test_hook_preserves_frozen_device_formula_and_only_canonicalizes_reduction(self):
        source = inspect.getsource(apply_bf16_update)
        self.assertIn(
            "delta_fp32 = labels[:, None] * alpha_value * scale_value * direction_device[None, :]",
            source,
        )
        self.assertIn("delta_bf16 = delta_fp32.to(dtype=hidden.dtype)", source)
        self.assertIn("canonical_delta_norms(actual_fp32[index], actual_bf16[index])", source)

    def test_reference_components_preserve_frozen_signed_prompt_formula_bitwise(self):
        direction = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float32)
        reference_fp32, reference_bf16 = reference_delta_components(direction, 0.5, 0.25)
        labels = torch.tensor([1.0, -1.0], dtype=torch.float32)[:, None]
        expected_fp32 = labels * reference_fp32[None, :]
        expected_bf16 = expected_fp32.to(torch.bfloat16)
        hidden = torch.zeros((2, 1, 4), dtype=torch.bfloat16)
        modified, _ = apply_bf16_update(hidden, direction, [1, -1], 0.5, 0.25, [0, 0])
        actual_bf16 = modified[:, 0, :]
        self.assertTrue(torch.equal(actual_bf16.view(torch.int16), expected_bf16.view(torch.int16)))

        signed_zero_direction = torch.tensor([0.0, -0.0], dtype=torch.float32)
        zero_reference, _ = reference_delta_components(signed_zero_direction, 0.5, 0.25)
        actual_zero_formula = labels * 0.5 * 0.25 * signed_zero_direction[None, :]
        expected_zero_formula = labels * zero_reference[None, :]
        self.assertTrue(
            torch.equal(actual_zero_formula.view(torch.int32), expected_zero_formula.view(torch.int32))
        )

    def test_shared_reducer_rejects_wrong_shape_dtype_nonfinite_and_zero(self):
        good_fp32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        good_bf16 = good_fp32.to(torch.bfloat16)
        cases = (
            (good_fp32.double(), good_bf16),
            (good_fp32, good_bf16.float()),
            (good_fp32[None, :], good_bf16),
            (good_fp32, good_bf16[:1]),
            (torch.tensor([math.nan, 1.0]), good_bf16),
            (torch.tensor([math.inf, 1.0]), good_bf16),
            (torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.bfloat16)),
        )
        for fp32, bf16 in cases:
            with self.subTest(fp32=fp32, bf16=bf16), self.assertRaises(QSCCIError):
                canonical_delta_norms(fp32, bf16)

    def test_reference_builder_rejects_wrong_shape_dtype_nonfinite_zero_and_bool_scalars(self):
        good = torch.tensor([1.0, 0.0], dtype=torch.float32)
        cases = (
            (good.double(), 1.0, 1.0),
            (good[None, :], 1.0, 1.0),
            (torch.tensor([math.nan, 0.0]), 1.0, 1.0),
            (torch.tensor([math.inf, 0.0]), 1.0, 1.0),
            (good, 0.0, 1.0),
            (good, 1.0, 0.0),
            (good, True, 1.0),
        )
        for direction, alpha, scale in cases:
            with self.subTest(alpha=alpha, scale=scale), self.assertRaises((QSCCIError, TypeError)):
                reference_delta_components(direction, alpha, scale)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA reduction portability regression")
    def test_cuda_retained_norms_exactly_match_frozen_cpu_verifier_reconstruction(self):
        direction = self._direction()
        alpha = 0.5
        scale = float(torch.tensor(0.137, dtype=torch.float32))
        hidden = torch.ones((2, 1, 2048), dtype=torch.bfloat16, device="cuda")
        _, evidence = apply_bf16_update(hidden, direction, [1, -1], alpha, scale, [0, 0])
        expected_shared, expected_cast = self._frozen_expected(direction, alpha, scale)
        self.assertEqual(evidence["shared_delta_norm"], expected_shared)
        self.assertEqual(evidence["cast_delta_norms"], [expected_cast, expected_cast])

    def test_retained_norms_scale_with_each_frozen_alpha(self):
        direction = self._direction()
        scale = 0.5
        for alpha in (0.5, 1.0, 2.0, 4.0):
            with self.subTest(alpha=alpha):
                hidden = torch.zeros((1, 1, 2048), dtype=torch.bfloat16)
                _, evidence = apply_bf16_update(hidden, direction, [1], alpha, scale, [0])
                expected_shared, expected_cast = self._frozen_expected(direction, alpha, scale)
                self.assertEqual(evidence["shared_delta_norm"], expected_shared)
                self.assertEqual(evidence["cast_delta_norms"], [expected_cast])

    def test_retained_norms_follow_direction_magnitude_and_ignore_direction_sign(self):
        direction = self._direction()
        hidden = torch.zeros((1, 1, 2048), dtype=torch.bfloat16)
        evidence_by_direction = []
        for candidate in (direction, -direction, 2.0 * direction):
            _, evidence = apply_bf16_update(hidden, candidate, [1], 0.5, 0.5, [0])
            evidence_by_direction.append(evidence)
            expected_shared, expected_cast = self._frozen_expected(candidate, 0.5, 0.5)
            self.assertEqual(evidence["shared_delta_norm"], expected_shared)
            self.assertEqual(evidence["cast_delta_norms"], [expected_cast])
        self.assertEqual(
            evidence_by_direction[0]["shared_delta_norm"],
            evidence_by_direction[1]["shared_delta_norm"],
        )
        self.assertEqual(
            evidence_by_direction[2]["shared_delta_norm"],
            2.0 * evidence_by_direction[0]["shared_delta_norm"],
        )

    def test_realized_update_norm_is_actual_bf16_addition_not_shared_formula(self):
        direction = self._direction()
        hidden = torch.stack(
            (
                torch.zeros((1, 2048), dtype=torch.bfloat16),
                torch.full((1, 2048), 0.125, dtype=torch.bfloat16),
            )
        )
        _, evidence = apply_bf16_update(hidden, direction, [1, 1], 0.5, 0.5, [0, 0])
        self.assertGreater(evidence["realized_update_norms"][0], 0.0)
        self.assertGreater(evidence["realized_update_norms"][1], 0.0)
        self.assertNotEqual(evidence["realized_update_norms"][0], evidence["shared_delta_norm"])
        self.assertNotEqual(evidence["realized_update_norms"][0], evidence["realized_update_norms"][1])


if __name__ == "__main__":
    unittest.main()
