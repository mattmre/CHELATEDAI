import itertools
import math
import unittest

import numpy as np

from prime_ring_waypoint import legendre_carrier
from run_prime_ring_waypoint_experiment import _binomial_upper_tail


class TestKnownCodeReductions(unittest.TestCase):
    def test_dense_legendre_scores_are_affine_oppw_overlap(self):
        p = 7
        phases_left = (0, 1, 2, 3)
        phases_right = (0, 4, 2, 6)
        layers = len(phases_left)
        carrier = legendre_carrier(p).astype(np.int64)

        dense_inner_products = np.asarray(
            [
                np.dot(
                    np.roll(carrier, left),
                    np.roll(carrier, right),
                )
                for left, right in zip(phases_left, phases_right)
            ],
            dtype=np.int64,
        )
        overlap_count = sum(
            left == right
            for left, right in zip(phases_left, phases_right)
        )
        oppw_overlap = overlap_count / layers

        dense_none = float(np.sum(dense_inner_products) / (layers * p))
        dense_free = float(
            np.sum(np.abs(dense_inner_products)) / (layers * p)
        )

        self.assertAlmostEqual(
            dense_none,
            ((p + 1) / p) * oppw_overlap - 1 / p,
        )
        self.assertAlmostEqual(
            dense_free,
            ((p - 1) / p) * oppw_overlap + 1 / p,
        )

    def test_dense_distance_gain_is_exact_channel_use_expansion(self):
        p = 7
        layers = 4
        overlap_count = 2
        carrier = legendre_carrier(p)
        phases_left = (0, 1, 2, 3)
        phases_right = (0, 4, 2, 6)

        dense_left = np.concatenate(
            [np.roll(carrier, phase) for phase in phases_left]
        )
        dense_right = np.concatenate(
            [np.roll(carrier, phase) for phase in phases_right]
        )
        dense_distance = int(np.count_nonzero(dense_left != dense_right))

        oppw_left = np.zeros((layers, p), dtype=np.int8)
        oppw_right = np.zeros((layers, p), dtype=np.int8)
        oppw_left[np.arange(layers), phases_left] = 1
        oppw_right[np.arange(layers), phases_right] = 1
        oppw_distance = int(np.count_nonzero(oppw_left != oppw_right))

        self.assertEqual(
            dense_distance,
            (layers - overlap_count) * (p + 1) // 2,
        )
        self.assertEqual(oppw_distance, 2 * (layers - overlap_count))
        self.assertEqual(
            dense_distance / oppw_distance,
            (p + 1) / 4,
        )

    def test_pairwise_bsc_error_is_the_binomial_majority_tail(self):
        distance = 7
        crossover = 0.2
        enumerated = 0.0
        for flips in itertools.product((0, 1), repeat=distance):
            flip_count = sum(flips)
            if flip_count >= math.ceil(distance / 2):
                enumerated += (
                    crossover**flip_count
                    * (1 - crossover) ** (distance - flip_count)
                )
        self.assertAlmostEqual(
            enumerated,
            _binomial_upper_tail(distance, crossover),
            places=14,
        )

    def test_4691_full_size_falsifier_distances_and_bound(self):
        p = 4691
        layers = 8
        overlap_cap = 1
        none_distance = (layers - overlap_cap) * (p + 1) // 2
        typed16_lower_bound = none_distance - 4
        free256_distance = (layers - overlap_cap) * (p - 1) // 2

        self.assertEqual(none_distance, 16422)
        self.assertEqual(typed16_lower_bound, 16418)
        self.assertEqual(free256_distance, 16415)

        wrong_state_count = (16 - 1) * p * 16
        union_bound = wrong_state_count * _binomial_upper_tail(
            typed16_lower_bound,
            0.45,
        )
        self.assertLess(union_bound, 2.0e-30)


if __name__ == "__main__":
    unittest.main()
