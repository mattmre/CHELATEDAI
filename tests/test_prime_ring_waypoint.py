"""Unit tests for the isolated PRW METHOD_DEV mechanism."""

from __future__ import annotations

import unittest

import numpy as np

from prime_ring_waypoint import (
    ObservedRing,
    PRWValidationError,
    RingTemplate,
    crt_encode_integer,
    crt_reconstruct_integer,
    exact_periodic_autocorrelation,
    factor_integer_exact,
    has_ideal_legendre_autocorrelation,
    is_prime_exact,
    is_primitive_root_exact,
    layer_circular_correlations,
    legendre_carrier,
    make_observation,
    make_template,
    observe_template,
    packed_byte_accounting,
    policy_masks,
    relative_phase_signature,
    rademacher_carrier,
    score_independent_layers,
    score_shared_shift,
    score_waypoint,
)


def _payload(layers: int, p: int) -> np.ndarray:
    base = legendre_carrier(p).astype(np.float64)
    return np.vstack([np.roll(base, layer) for layer in range(layers)])


def _rademacher_payload(layers: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice((-1.0, 1.0), size=(layers, p))


class TestPRWArithmetic(unittest.TestCase):
    def test_small_ideal_primes_have_exact_legendre_autocorrelation(self):
        for p in (7, 11, 31):
            with self.subTest(p=p):
                carrier = legendre_carrier(p)
                self.assertEqual(carrier.shape, (p,))
                self.assertEqual(set(carrier.tolist()), {-1, 1})
                self.assertEqual(
                    exact_periodic_autocorrelation(carrier),
                    (p,) + (-1,) * (p - 1),
                )
                self.assertTrue(has_ideal_legendre_autocorrelation(p))

    def test_exact_primality_distinguishes_controls(self):
        for p in (7, 11, 31, 4091, 4691):
            with self.subTest(p=p):
                self.assertTrue(is_prime_exact(p))
        for composite in (0, 1, 9, 21, 4095, 4690):
            with self.subTest(composite=composite):
                self.assertFalse(is_prime_exact(composite))

    def test_4691_multiplicative_exponent_crt_round_trip(self):
        moduli = (2, 5, 7, 67)
        self.assertEqual(factor_integer_exact(4690), moduli)
        self.assertTrue(is_primitive_root_exact(2, 4691))
        for exponent in (0, 1, 2, 66, 67, 2345, 4689, 4691, -1):
            with self.subTest(exponent=exponent):
                residues = crt_encode_integer(exponent, moduli)
                self.assertEqual(
                    crt_reconstruct_integer(residues, moduli),
                    exponent % 4690,
                )

    def test_4690_crt_uses_the_declared_exact_reconstruction_coefficients(self):
        moduli = (2, 5, 7, 67)
        residues = (1, 2, 3, 4)
        declared = (
            2345 * residues[0]
            + 1876 * residues[1]
            + 2010 * residues[2]
            + 3150 * residues[3]
        ) % 4690
        self.assertEqual(crt_reconstruct_integer(residues, moduli), declared)

    def test_direct_and_fft_correlation_agree(self):
        reference = _payload(3, 11)
        query = np.roll(reference, 4, axis=1)
        direct = layer_circular_correlations(query, reference, method="direct")
        fft = layer_circular_correlations(query, reference, method="fft")
        np.testing.assert_allclose(fft, direct, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(np.argmax(fft, axis=1), np.array([4, 4, 4]))


class TestPRWPhaseGeometry(unittest.TestCase):
    def test_signature_is_invariant_to_one_global_rotation(self):
        for p, phases, offset in (
            (7, (0, 1, 4), 3),
            (11, (2, 8, 1, 9), 7),
            (31, (30, 0, 17), 29),
        ):
            with self.subTest(p=p):
                shifted = tuple((phase + offset) % p for phase in phases)
                self.assertEqual(
                    relative_phase_signature(phases, p),
                    relative_phase_signature(shifted, p),
                )
                self.assertEqual(relative_phase_signature(phases, p)[0], 0)

    def test_observation_folds_phase_origin_into_global_shift(self):
        payload = _payload(3, 11)
        first = make_observation(
            "PRW-WP-QUOTIENT-A",
            payload,
            (2, 5, 9),
            global_shift=4,
        )
        second = make_observation(
            "PRW-WP-QUOTIENT-B",
            payload,
            (0, 3, 7),
            global_shift=6,
        )
        self.assertEqual(first.relative_phase, (0, 3, 7))
        self.assertEqual(first.global_shift, 6)
        np.testing.assert_array_equal(first.carrier, second.carrier)
        np.testing.assert_array_equal(first.payload, second.payload)

    def test_joint_shift_preserves_signature_while_independent_control_erases_it(self):
        p = 11
        payload = _payload(3, p)
        reference = make_observation("PRW-WP-SIGNATURE-A", payload, (0, 2, 7))
        wrong_signature = make_observation("PRW-WP-SIGNATURE-B", payload, (0, 3, 7))
        query = make_observation(
            "PRW-WP-SIGNATURE-QUERY",
            payload,
            (0, 2, 7),
            global_shift=4,
        )

        planted = score_shared_shift(query.carrier, reference.carrier)
        wrong_joint = score_shared_shift(query.carrier, wrong_signature.carrier)
        degenerate = score_independent_layers(
            query.carrier,
            wrong_signature.carrier,
            allow_polarity=False,
        )

        self.assertAlmostEqual(planted.score, 1.0, places=12)
        self.assertEqual(planted.shared_shift, 4)
        self.assertLess(wrong_joint.score, 1.0)
        self.assertAlmostEqual(degenerate.score, 1.0, places=12)
        self.assertIsNone(degenerate.shared_shift)
        self.assertTrue(degenerate.negative_control)
        self.assertTrue(degenerate.mechanism_id.startswith("PRW-NEG-"))

    def test_global_shift_recovery_is_quotient_invariant(self):
        p = 31
        payload = _payload(4, p)
        reference = make_observation("PRW-WP-GLOBAL-REFERENCE", payload, (8, 11, 17, 30))
        query = make_observation(
            "PRW-WP-GLOBAL-QUERY",
            payload,
            (1, 4, 10, 23),
            global_shift=12,
        )
        # The phase vectors differ by one global offset; the query's effective
        # shift relative to the reference is (12 + 1 - 8) mod 31.
        result = score_shared_shift(query.carrier, reference.carrier)
        self.assertAlmostEqual(result.score, 1.0, places=12)
        self.assertEqual(result.shared_shift, 5)
        self.assertEqual(len(set(result.layer_shifts)), 1)


class TestPRWCarrierPayloadSeparation(unittest.TestCase):
    def test_carrier_only_is_degenerate_for_waypoints_with_same_signature(self):
        p = 31
        payload_a = _payload(3, p)
        payload_b = np.roll(payload_a, 1, axis=1)
        phases = (0, 7, 19)
        candidate_a = make_observation("PRW-WP-CARRIER-A", payload_a, phases)
        candidate_b = make_observation("PRW-WP-CARRIER-B", payload_b, phases)
        query = make_observation(
            "PRW-WP-CARRIER-QUERY",
            payload_a,
            phases,
            global_shift=9,
        )

        carrier_a = score_shared_shift(query.carrier, candidate_a.carrier)
        carrier_b = score_shared_shift(query.carrier, candidate_b.carrier)
        self.assertAlmostEqual(carrier_a.score, 1.0, places=12)
        self.assertAlmostEqual(carrier_b.score, 1.0, places=12)
        self.assertEqual(carrier_a.shared_shift, carrier_b.shared_shift)

    def test_payload_disambiguates_only_after_carrier_freezes_shift(self):
        p = 31
        payload_a = _payload(3, p)
        payload_b = np.roll(payload_a, 1, axis=1)
        phases = (0, 7, 19)
        candidate_a = make_observation("PRW-WP-PAYLOAD-A", payload_a, phases)
        candidate_b = make_observation("PRW-WP-PAYLOAD-B", payload_b, phases)
        query = make_observation(
            "PRW-WP-PAYLOAD-QUERY",
            payload_a,
            phases,
            global_shift=9,
        )

        result_a = score_waypoint(query, candidate_a)
        result_b = score_waypoint(query, candidate_b)
        self.assertAlmostEqual(result_a.carrier_score, result_b.carrier_score, places=12)
        self.assertAlmostEqual(result_a.payload_score, 1.0, places=12)
        self.assertAlmostEqual(result_b.payload_score, -1.0 / p, places=12)
        self.assertGreater(result_a.payload_score, result_b.payload_score)
        self.assertTrue(result_a.mechanism_id.startswith("PRW-"))

    def test_orientation_mask_modifies_carrier_but_not_identity_payload(self):
        payload = _payload(8, 7)
        phases = tuple(range(7)) + (0,)
        unmasked = make_observation("PRW-WP-UNMASKED", payload, phases, global_shift=3)
        masked = make_observation(
            "PRW-WP-MASKED",
            payload,
            phases,
            global_shift=3,
            orientation_mask=(-1, 1, -1, 1, -1, 1, -1, 1),
        )
        self.assertFalse(np.array_equal(unmasked.carrier, masked.carrier))
        np.testing.assert_array_equal(unmasked.payload, masked.payload)

    def test_independent_rademacher_payload_is_not_carrier_identity(self):
        p = 31
        phases = (0, 7, 19)
        payload_a = _rademacher_payload(3, p, 20260724)
        payload_b = _rademacher_payload(3, p, 20260725)
        base_carrier = legendre_carrier(p)
        self.assertFalse(
            any(
                np.array_equal(payload_a[0], np.roll(base_carrier, shift))
                for shift in range(p)
            )
        )

        candidate_a = make_template("PRW-WP-RANDOM-A", payload_a, phases)
        candidate_b = make_template("PRW-WP-RANDOM-B", payload_b, phases)
        query = observe_template(
            candidate_a,
            waypoint_id="PRW-WP-RANDOM-QUERY",
            global_shift=9,
        )

        result_a = score_waypoint(query, candidate_a)
        result_b = score_waypoint(query, candidate_b)
        self.assertAlmostEqual(result_a.carrier_score, result_b.carrier_score, places=12)
        self.assertAlmostEqual(result_a.payload_score, 1.0, places=12)
        self.assertLess(result_b.payload_score, result_a.payload_score)

    def test_payload_score_is_declared_macro_average_not_flattened_cosine(self):
        p = 7
        query_payload = np.zeros((2, p))
        query_payload[0, 0] = 100.0
        query_payload[1, 0] = 1.0
        candidate_a_payload = query_payload.copy()
        candidate_a_payload[1] *= -1.0
        candidate_b_payload = np.zeros((2, p))
        candidate_b_payload[0, 1] = 100.0
        candidate_b_payload[1, 0] = 1.0

        query_template = make_template(
            "PRW-WP-MACRO-QUERY-TEMPLATE",
            query_payload,
            (0, 0),
        )
        candidate_a = make_template(
            "PRW-WP-MACRO-A",
            candidate_a_payload,
            (0, 0),
        )
        candidate_b = make_template(
            "PRW-WP-MACRO-B",
            candidate_b_payload,
            (0, 0),
        )
        query = observe_template(
            query_template,
            waypoint_id="PRW-WP-MACRO-QUERY",
        )

        score_a = score_waypoint(query, candidate_a).payload_score
        score_b = score_waypoint(query, candidate_b).payload_score
        self.assertAlmostEqual(score_a, 0.0, places=12)
        self.assertAlmostEqual(score_b, 0.5, places=12)
        self.assertGreater(score_b, score_a)

        flat_a = float(
            np.vdot(query_payload.ravel(), candidate_a_payload.ravel())
            / (
                np.linalg.norm(query_payload)
                * np.linalg.norm(candidate_a_payload)
            )
        )
        flat_b = float(
            np.vdot(query_payload.ravel(), candidate_b_payload.ravel())
            / (
                np.linalg.norm(query_payload)
                * np.linalg.norm(candidate_b_payload)
            )
        )
        self.assertGreater(flat_a, flat_b)


class TestPRWMaskPolicies(unittest.TestCase):
    def test_typed16_is_hadamard_code_with_minimum_distance_four(self):
        masks = policy_masks("typed16", 8)
        self.assertEqual(masks.shape, (16, 8))
        self.assertEqual(len({tuple(row) for row in masks}), 16)
        distances = []
        for left in range(16):
            for right in range(left + 1, 16):
                distances.append(int(np.count_nonzero(masks[left] != masks[right])))
        self.assertEqual(min(distances), 4)
        self.assertIn(8, distances)

    def test_typed16_recovers_an_admissible_mask(self):
        payload = _payload(8, 7)
        phases = tuple(range(7)) + (0,)
        typed_mask = tuple(int(value) for value in policy_masks("typed16", 8)[5])
        candidate = make_observation("PRW-WP-TYPED-CANDIDATE", payload, phases)
        query = make_observation(
            "PRW-WP-TYPED-QUERY",
            payload,
            phases,
            global_shift=2,
            orientation_mask=typed_mask,
        )

        constrained = score_waypoint(query, candidate, policy="typed16")
        unmasked = score_waypoint(query, candidate, policy="none")
        self.assertAlmostEqual(constrained.carrier_score, 1.0, places=12)
        self.assertAlmostEqual(constrained.payload_score, 1.0, places=12)
        self.assertEqual(constrained.orientation_mask, typed_mask)
        self.assertLess(unmasked.carrier_score, constrained.carrier_score)
        self.assertFalse(constrained.negative_control)

    def test_shared_policy_recovers_only_one_common_polarity(self):
        payload = _payload(8, 11)
        phases = tuple(range(8))
        candidate = make_observation("PRW-WP-SHARED-CANDIDATE", payload, phases)
        query = make_observation(
            "PRW-WP-SHARED-QUERY",
            payload,
            phases,
            global_shift=6,
            orientation_mask=(-1,) * 8,
        )
        shared = score_waypoint(query, candidate, policy="shared")
        none = score_waypoint(query, candidate, policy="none")
        self.assertAlmostEqual(shared.carrier_score, 1.0, places=12)
        self.assertAlmostEqual(shared.payload_score, 1.0, places=12)
        self.assertEqual(shared.orientation_mask, (-1,) * 8)
        self.assertLess(none.payload_score, shared.payload_score)

    def test_free256_factorizes_and_exposes_false_unlock_boundary(self):
        p = 7
        payload = _payload(8, p)
        phases = tuple(range(7)) + (0,)
        inadmissible_mask = (-1, 1, 1, 1, 1, 1, 1, 1)
        candidate = make_observation("PRW-WP-FREE-CANDIDATE", payload, phases)
        query = make_observation(
            "PRW-WP-FREE-QUERY",
            payload,
            phases,
            global_shift=3,
            orientation_mask=inadmissible_mask,
        )

        correlations = layer_circular_correlations(query.carrier, candidate.carrier)
        factorized_expected = float(np.max(np.sum(np.abs(correlations), axis=0) / 8))
        free = score_shared_shift(query.carrier, candidate.carrier, policy="free256")
        typed = score_shared_shift(query.carrier, candidate.carrier, policy="typed16")

        self.assertAlmostEqual(free.score, factorized_expected, places=12)
        self.assertAlmostEqual(free.score, 1.0, places=12)
        self.assertEqual(free.mask, inadmissible_mask)
        self.assertEqual(free.candidate_count, p * 256)
        self.assertTrue(free.negative_control)
        self.assertTrue(free.mechanism_id.startswith("PRW-NEG-"))
        self.assertLess(typed.score, free.score)
        self.assertAlmostEqual(typed.score, 0.75, places=12)

    def test_direct_and_fft_search_have_identical_shift_and_mask(self):
        rng = np.random.default_rng(20260723)
        reference = rng.normal(size=(8, 31))
        orientation = tuple(int(value) for value in policy_masks("typed16", 8)[7])
        query = np.roll(reference, 13, axis=1) * np.asarray(orientation)[:, np.newaxis]

        direct = score_shared_shift(query, reference, policy="typed16", method="direct")
        fft = score_shared_shift(query, reference, policy="typed16", method="fft")
        self.assertEqual(direct.shared_shift, 13)
        self.assertEqual(fft.shared_shift, direct.shared_shift)
        self.assertEqual(fft.mask, direct.mask)
        self.assertEqual(fft.layer_shifts, direct.layer_shifts)
        self.assertAlmostEqual(fft.score, direct.score, places=12)
        np.testing.assert_allclose(fft.per_layer_scores, direct.per_layer_scores, atol=1e-12)

        independent_direct = score_independent_layers(
            query,
            reference,
            allow_polarity=True,
            method="direct",
        )
        independent_fft = score_independent_layers(
            query,
            reference,
            allow_polarity=True,
            method="fft",
        )
        self.assertEqual(independent_fft.layer_shifts, independent_direct.layer_shifts)
        self.assertEqual(independent_fft.mask, independent_direct.mask)
        self.assertAlmostEqual(independent_fft.score, independent_direct.score, places=12)

    def test_low_level_fft_scoring_accepts_4096_random_control(self):
        rng = np.random.default_rng(4096)
        reference = rng.normal(size=(2, 4096))
        query = np.roll(reference, 137, axis=1)
        result = score_shared_shift(query, reference, method="fft")
        self.assertEqual(result.shared_shift, 137)
        self.assertAlmostEqual(result.score, 1.0, places=12)


class TestPRWGenericCyclicControls(unittest.TestCase):
    def test_4096_rademacher_control_runs_end_to_end(self):
        p = 4096
        payload = _rademacher_payload(8, p, 20264096)
        base_carrier = rademacher_carrier(p, 4096)
        orientation = tuple(
            int(value) for value in policy_masks("typed16", 8)[7]
        )
        template = make_template(
            "PRW-WP-RANDOM-4096-TEMPLATE",
            payload,
            tuple(range(8)),
            carrier_family="rademacher",
            base_carrier=base_carrier,
        )
        query = observe_template(
            template,
            waypoint_id="PRW-WP-RANDOM-4096-QUERY",
            global_shift=137,
            orientation_mask=orientation,
        )

        result = score_waypoint(
            query,
            template,
            policy="typed16",
        )
        self.assertIsInstance(template, RingTemplate)
        self.assertIsInstance(query, ObservedRing)
        self.assertEqual(result.carrier_shift, 137)
        self.assertEqual(result.orientation_mask, orientation)
        self.assertAlmostEqual(result.carrier_score, 1.0, places=12)
        self.assertAlmostEqual(result.payload_score, 1.0, places=12)

    def test_rademacher_carrier_is_reproducible_and_seed_specific(self):
        first = rademacher_carrier(4096, 7)
        second = rademacher_carrier(4096, 7)
        other = rademacher_carrier(4096, 42)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertFalse(first.flags.writeable)


class TestPRWPackedAccounting(unittest.TestCase):
    def test_eight_4691_float64_layers_are_accounted_at_actual_width(self):
        accounting = packed_byte_accounting(4691, 8, policy="typed16")
        self.assertEqual(accounting.payload_encoding, "float64")
        self.assertEqual(accounting.payload_bits_per_value, 64)
        self.assertEqual(accounting.layer_plane_bytes, 300224)
        self.assertEqual(accounting.payload_bytes, 300224)
        self.assertEqual(accounting.canonical_carrier_bytes, 587)
        self.assertEqual(accounting.bits_per_residue, 13)
        self.assertEqual(accounting.relative_phase_bytes, 12)
        self.assertEqual(accounting.orientation_bytes, 1)
        self.assertEqual(accounting.total_bytes, 300824)
        self.assertEqual(accounting.theoretical_packed_layer_plane_bytes, 4691)
        self.assertEqual(accounting.theoretical_packed_payload_bytes, 4691)
        self.assertEqual(accounting.theoretical_packed_total_bytes, 5291)
        self.assertEqual(accounting.mechanism_id, "PRW-PACKED-BYTES-001")

    def test_explicit_bitpacked_encoding_preserves_binary_plane_accounting(self):
        accounting = packed_byte_accounting(
            7,
            8,
            payload_planes=2,
            policy="free256",
            payload_encoding="bitpacked",
        )
        self.assertEqual(accounting.payload_encoding, "bitpacked")
        self.assertEqual(accounting.payload_bits_per_value, 1)
        self.assertEqual(accounting.canonical_carrier_bytes, 1)
        self.assertEqual(accounting.layer_plane_bytes, 7)
        self.assertEqual(accounting.payload_bytes, 14)
        self.assertEqual(accounting.relative_phase_bytes, 3)
        self.assertEqual(accounting.orientation_bytes, 1)
        self.assertEqual(accounting.total_bytes, 19)
        self.assertEqual(accounting.theoretical_packed_total_bytes, 19)

    def test_composite_control_accounting_is_available(self):
        accounting = packed_byte_accounting(4096, 8, policy="typed16")
        self.assertEqual(accounting.p, 4096)
        self.assertEqual(accounting.payload_bytes, 262144)
        self.assertEqual(accounting.theoretical_packed_payload_bytes, 4096)
        self.assertEqual(accounting.total_bytes, 262668)


class TestPRWValidation(unittest.TestCase):
    def test_invalid_ring_lengths_fail_closed(self):
        for p in (2, 5, 9, 13, 21):
            with self.subTest(p=p):
                with self.assertRaises(PRWValidationError):
                    legendre_carrier(p)
        with self.assertRaises(PRWValidationError):
            is_prime_exact(True)

    def test_invalid_ids_phases_masks_and_policies_fail_closed(self):
        payload = _payload(8, 7)
        phases = tuple(range(7)) + (0,)
        with self.assertRaises(PRWValidationError):
            make_observation("waypoint", payload, phases)
        with self.assertRaises(PRWValidationError):
            make_observation("PRW-WP-BAD-PHASE", payload, phases[:-1])
        with self.assertRaises(PRWValidationError):
            make_observation("PRW-WP-BAD-PHASE", payload, (0.0,) * 8)
        with self.assertRaises(PRWValidationError):
            make_observation("PRW-WP-BAD-MASK", payload, phases, orientation_mask=(1,) * 7 + (0,))
        with self.assertRaises(PRWValidationError):
            policy_masks("typed16", 7)
        with self.assertRaises(PRWValidationError):
            policy_masks("free256", 9)
        with self.assertRaises(PRWValidationError):
            policy_masks("unknown", 8)

    def test_nonfinite_zero_and_shape_mismatches_fail_closed(self):
        payload = _payload(3, 7)
        nonfinite = payload.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaises(PRWValidationError):
            make_observation("PRW-WP-NAN", nonfinite, (0, 1, 2))
        zero_layer = payload.copy()
        zero_layer[1] = 0.0
        with self.assertRaises(PRWValidationError):
            make_observation("PRW-WP-ZERO", zero_layer, (0, 1, 2))
        with self.assertRaises(PRWValidationError):
            layer_circular_correlations(payload, np.ones((2, 7)))
        infinite = payload.copy()
        infinite[0, 0] = np.inf
        with self.assertRaises(PRWValidationError):
            layer_circular_correlations(payload, infinite)
        with self.assertRaises(PRWValidationError):
            layer_circular_correlations(payload, payload, method="rader")
        with self.assertRaises(PRWValidationError):
            make_observation(
                "PRW-WP-RAGGED",
                [[1.0, 2.0, 3.0], [1.0, 2.0]],
                (0, 1),
            )

    def test_exact_autocorrelation_rejects_non_integer_values(self):
        with self.assertRaises(PRWValidationError):
            exact_periodic_autocorrelation(np.array([1.0, -1.0, 1.0]))
        with self.assertRaises(PRWValidationError):
            exact_periodic_autocorrelation([])

    def test_multiplicative_crt_validation_fails_closed(self):
        with self.assertRaises(PRWValidationError):
            factor_integer_exact(0)
        with self.assertRaises(PRWValidationError):
            is_primitive_root_exact(2, 21)
        with self.assertRaises(PRWValidationError):
            crt_encode_integer(1, (2, 4))
        with self.assertRaises(PRWValidationError):
            crt_reconstruct_integer((0,), (2, 5))
        with self.assertRaises(PRWValidationError):
            crt_reconstruct_integer((0, 5), (2, 5))

    def test_waypoint_score_requires_matching_observation_contracts(self):
        candidate = make_observation("PRW-WP-CONTRACT-A", _payload(3, 7), (0, 1, 2))
        query = make_observation("PRW-WP-CONTRACT-B", _payload(3, 11), (0, 1, 2))
        with self.assertRaises(PRWValidationError):
            score_waypoint(query, candidate)
        with self.assertRaises(PRWValidationError):
            score_waypoint(query, object())

    def test_clean_template_rejects_metadata_carrier_inconsistency(self):
        template = make_template(
            "PRW-WP-STRICT-TEMPLATE",
            _rademacher_payload(3, 7, 77),
            (0, 2, 5),
        )
        inconsistent = np.array(template.carrier, copy=True)
        inconsistent[0] *= -1.0
        with self.assertRaises(PRWValidationError):
            RingTemplate(
                waypoint_id=template.waypoint_id,
                p=template.p,
                base_carrier=template.base_carrier,
                carrier=inconsistent,
                payload=template.payload,
                relative_phase=template.relative_phase,
                orientation_mask=template.orientation_mask,
                carrier_family=template.carrier_family,
            )

    def test_corrupted_observation_cannot_be_used_as_clean_candidate(self):
        template = make_template(
            "PRW-WP-CLEAN-CANDIDATE",
            _rademacher_payload(3, 7, 88),
            (0, 2, 5),
        )
        query = observe_template(
            template,
            waypoint_id="PRW-WP-CORRUPTED-QUERY",
            global_shift=2,
        )
        corrupted_carrier = np.array(query.carrier, copy=True)
        corrupted_carrier[0, 0] = 0.25
        corrupted = ObservedRing(
            waypoint_id="PRW-WP-CORRUPTED-OBS",
            p=7,
            carrier=corrupted_carrier,
            payload=query.payload,
        )
        score_waypoint(corrupted, template)
        with self.assertRaises(PRWValidationError):
            score_waypoint(query, corrupted)
        legacy_candidate = make_observation(
            "PRW-WP-LEGACY-CANDIDATE",
            _rademacher_payload(3, 7, 89),
            (0, 2, 5),
        )
        with self.assertRaises(PRWValidationError):
            score_waypoint(query, legacy_candidate)

    def test_invalid_payload_encoding_fails_closed(self):
        with self.assertRaises(PRWValidationError):
            packed_byte_accounting(
                4691,
                8,
                payload_encoding="implicit-magic-compression",
            )


if __name__ == "__main__":
    unittest.main()
