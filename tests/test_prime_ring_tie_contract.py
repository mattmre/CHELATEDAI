import math
import unittest

import numpy as np

from prime_ring_tie_contract_audit import (
    AUDIT_PRIMES,
    _bank,
    _contract_fft,
    _direct,
    _exact_hamming,
    _flip,
    _states,
    run_audit,
)
from run_prime_ring_waypoint_experiment import (
    FFT_INTEGER_ROUNDING_CEILING,
    CampaignValidationError,
    _correlation_bank,
    _decode_dense_bank,
    _decode_independent_layers,
    _exact_bipolar_correlation_counts,
    _top_two_integer_semantics,
    fft_integer_rounding_bound,
)


class TestLeadingMidpointTieParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = run_audit(AUDIT_PRIMES)

    def test_all_336_midpoint_ties_agree_across_three_paths(self):
        totals = self.audit["totals"]
        self.assertEqual(totals["cases"], 336)
        self.assertEqual(totals["exact_hamming_ties"], 336)
        self.assertEqual(totals["direct_ties"], 336)
        self.assertEqual(totals["contract_fft_ties"], 336)
        self.assertEqual(totals["midpoint_three_path_agreement"], 336)
        for row in self.audit["rows"]:
            self.assertEqual(row["contract_fft_max_tie_margin"], 0.0)
        self.assertEqual(self.audit["contract_fft_boundary_parity"], "PASSED")

    def test_canonical_winner_matches_exact_hamming(self):
        self.assertEqual(
            self.audit["totals"]["canonical_winner_matches_exact_hamming"], 336
        )

    def test_adjacent_unequal_cases_rank_strictly(self):
        totals = self.audit["totals"]
        self.assertEqual(totals["adjacent_cases"], 672)
        self.assertEqual(totals["adjacent_three_path_strict_agreement"], 672)

    def test_legacy_float_scorer_reproduces_recorded_failure(self):
        # findings.md RB-11 table: per-cell bit-exact float-FFT tie counts.
        recorded = {
            (11, 0): (27, 30, 26),
            (11, 1): (32, 51, 5),
            (19, 0): (17, 19, 37),
            (19, 1): (7, 8, 48),
            (31, 0): (15, 22, 34),
            (31, 1): (19, 56, 0),
        }
        for row in self.audit["rows"]:
            self.assertEqual(
                (
                    row["legacy_float_fft_ties"],
                    row["legacy_float_fft_winner_type_0"],
                    row["legacy_float_fft_winner_type_1"],
                ),
                recorded[(row["p"], row["truth_type"])],
            )
        self.assertEqual(self.audit["totals"]["legacy_float_fft_ties"], 117)


class TestNumericalTieContract(unittest.TestCase):
    def test_minimum_count_gap_is_never_a_false_tie(self):
        layers, length = 8, 11
        base = np.full((2, layers, length), -length, dtype=np.int64)
        base[0, :, 3] = 1
        base[1, :, 3] = 1
        base[1, 0, 3] = 3  # type one wins by two counts (minimum parity gap)
        decoded = _decode_dense_bank(base / float(length), "none")
        self.assertEqual(decoded["predicted_type"], 1)
        self.assertEqual(decoded["tie_count"], 1)
        self.assertEqual(decoded["margin"], 2 / float(layers * length))
        winner, _score, margin, tie_count = _top_two_integer_semantics(
            np.asarray([41, 40], dtype=np.int64), 88
        )
        self.assertEqual((winner, tie_count), (0, 1))
        self.assertEqual(margin, 1 / 88.0)
        winner, _score, margin, tie_count = _top_two_integer_semantics(
            np.asarray([40, 40], dtype=np.int64), 88
        )
        self.assertEqual((winner, margin, tie_count), (0, 0.0, 2))

    def test_fft_counts_match_direct_integer_correlation(self):
        templates, _masks = _bank(31)
        rng = np.random.default_rng(11)
        reference_ffts = np.fft.rfft(templates, axis=2)
        reference_norms = np.linalg.norm(templates, axis=2)
        integer_templates = templates.astype(np.int64)
        for _ in range(5):
            query = rng.choice(np.array([-1.0, 1.0]), size=(8, 31))
            counts = _exact_bipolar_correlation_counts(
                query, reference_ffts, reference_norms, 31
            )
            expected = np.empty_like(counts)
            for shift in range(31):
                expected[:, :, shift] = np.sum(
                    query.astype(np.int64)[np.newaxis]
                    * np.roll(integer_templates, shift, axis=2),
                    axis=2,
                )
            np.testing.assert_array_equal(counts, expected)

    def test_rounding_bound_is_declared_and_tight_enough(self):
        u = 2.0 ** -53
        self.assertAlmostEqual(
            fft_integer_rounding_bound(11) / (64.0 * 5 * u * 11 ** 1.5), 1.0, places=12
        )
        self.assertLess(fft_integer_rounding_bound(4691), 1e-7)
        self.assertLess(fft_integer_rounding_bound(4691), FFT_INTEGER_ROUNDING_CEILING)
        self.assertTrue(math.isfinite(fft_integer_rounding_bound(4096)))

    def test_non_bipolar_inputs_fail_closed(self):
        templates, _masks = _bank(11)
        reference_ffts = np.fft.rfft(templates, axis=2)
        reference_norms = np.linalg.norm(templates, axis=2)
        query = np.ones((8, 11))
        query[0, 0] = 0.5
        with self.assertRaises(CampaignValidationError):
            _correlation_bank(query, reference_ffts, reference_norms, 11)
        with self.assertRaises(CampaignValidationError):
            _correlation_bank(np.ones((8, 11)), reference_ffts, reference_norms * 1.1, 11)

    def test_off_lattice_correlations_fail_closed(self):
        correlations = np.full((2, 8, 11), 1.0 / 11.0)
        correlations[0, 0, 0] += 1e-12
        with self.assertRaises(CampaignValidationError):
            _decode_dense_bank(correlations, "typed16")
        with self.assertRaises(CampaignValidationError):
            _decode_independent_layers(correlations)

    def test_residual_above_bound_fails_closed(self):
        templates, _masks = _bank(11)
        reference_ffts = np.fft.rfft(templates, axis=2).copy()
        reference_norms = np.linalg.norm(templates, axis=2)
        reference_ffts[0, 0, 0] += 3.3  # shifts every count by 0.3
        with self.assertRaisesRegex(CampaignValidationError, "residual"):
            _correlation_bank(np.ones((8, 11)), reference_ffts, reference_norms, 11)

    def test_free256_and_independent_paths_rank_exact_integers(self):
        templates, masks = _bank(19)
        integer_templates = templates.astype(np.int64)
        states = [_states(integer_templates[t], masks) for t in range(2)]
        truth = states[0][0]
        wrong = states[1]
        leading = (7 * 19 - 1) // 2
        competitor = wrong[np.sum(wrong != truth, axis=(1, 2)) == leading][0]
        disagreements = np.flatnonzero((competitor != truth).ravel())
        midpoint = _flip(truth, disagreements[: leading // 2])
        self.assertEqual(_exact_hamming(midpoint, states)["tie_count"], 2)
        self.assertEqual(_direct(midpoint, states)["tie_count"], 2)
        self.assertEqual(_contract_fft(midpoint, templates)["tie_count"], 2)
        correlations = _correlation_bank(
            midpoint.astype(np.float64),
            np.fft.rfft(templates, axis=2),
            np.linalg.norm(templates, axis=2),
            19,
        )
        counts = np.empty((2, 8, 19), dtype=np.int64)
        for shift in range(19):
            counts[:, :, shift] = np.sum(
                midpoint[np.newaxis] * np.roll(integer_templates, shift, axis=2),
                axis=2,
            )
        free_expected = np.max(np.sum(np.abs(counts), axis=1), axis=1)
        independent_expected = np.sum(np.max(np.abs(counts), axis=2), axis=1)
        for decoded, expected in (
            (_decode_dense_bank(correlations, "free256"), free_expected),
            (_decode_independent_layers(correlations), independent_expected),
        ):
            winner = int(np.flatnonzero(expected == expected.max())[0])
            self.assertEqual(decoded["predicted_type"], winner)
            self.assertEqual(
                decoded["tie_count"], int(np.sum(expected == expected.max()))
            )
            self.assertEqual(decoded["score"], int(expected.max()) / (8.0 * 19))

if __name__ == "__main__":
    unittest.main()
