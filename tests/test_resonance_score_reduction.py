"""Constructed checks of arXiv:2509.09691v1, equations (2), (3), and (5).

These independently derived fixtures test representation equivalence, not
semantic quality, RHPC acceptance, or execution of the upstream implementation.
"""

import cmath
import math
import random
import unittest


def _paper_score(left, right):
    e1 = math.fsum(abs(z) ** 2 for z in left)
    e2 = math.fsum(abs(z) ** 2 for z in right)
    if e1 + e2 == 0:
        return 0.0
    interference = math.fsum(abs(a + b) ** 2 for a, b in zip(left, right))
    return 0.5 * interference / (e1 + e2) * 2 * math.sqrt(e1 * e2) / (e1 + e2)


def _unfold(pattern):
    return [z.real for z in pattern] + [z.imag for z in pattern]


def _real_score(left, right):
    u, v = _unfold(left), _unfold(right)
    e1 = math.fsum(x * x for x in u)
    e2 = math.fsum(x * x for x in v)
    if e1 + e2 == 0:
        return 0.0
    dot = math.fsum(x * y for x, y in zip(u, v))
    return (e1 + e2 + 2 * dot) * math.sqrt(e1 * e2) / (e1 + e2) ** 2


def _cosine(left, right):
    dot = math.fsum(x * y for x, y in zip(left, right))
    norm_product = math.sqrt(math.fsum(x * x for x in left) * math.fsum(y * y for y in right))
    return dot / norm_product


def _normalize(pattern):
    norm = math.sqrt(math.fsum(abs(z) ** 2 for z in pattern))
    return [z / norm for z in pattern]


def _random_pattern(rng, dimension=16):
    scale = 10 ** rng.uniform(-2, 2)
    return [cmath.rect(scale * rng.random(), rng.uniform(-math.pi, math.pi)) for _ in range(dimension)]


class TestResonanceScoreReduction(unittest.TestCase):
    def test_arbitrary_phases_and_unequal_energies_reduce_to_real_dot(self):
        rng = random.Random(20261008)
        for pair in range(512):
            a, b = _random_pattern(rng), _random_pattern(rng)
            with self.subTest(pair=pair):
                self.assertAlmostEqual(_paper_score(a, b), _real_score(a, b), delta=2e-14)

    def test_equal_energy_complex_patterns_have_cosine_scores_and_rankings(self):
        rng = random.Random(20261009)
        candidates = [_normalize(_random_pattern(rng)) for _ in range(48)]
        for query in range(8):
            q = _normalize(_random_pattern(rng))
            resonance = [_paper_score(q, c) for c in candidates]
            cosine = [_cosine(_unfold(q), _unfold(c)) for c in candidates]
            for s, c in zip(resonance, cosine):
                self.assertAlmostEqual(s, (1 + c) / 2, delta=2e-14)
            with self.subTest(query=query):
                self.assertEqual(
                    sorted(range(48), key=lambda i: (-resonance[i], i)),
                    sorted(range(48), key=lambda i: (-cosine[i], i)),
                )

    def test_sign_phase_mapping_preserves_signed_real_coordinates(self):
        real = [-3.0, 0.0, 2.0, -0.25]
        mapped = [cmath.rect(abs(x), 0 if x >= 0 else math.pi) for x in real]
        for x, z in zip(real, mapped):
            self.assertAlmostEqual(z.real, x, delta=1e-14)
            self.assertAlmostEqual(z.imag, 0, delta=1e-14)
        other = [1.0, -2.0, 0.0, 4.0]
        self.assertAlmostEqual(
            _paper_score(_normalize(mapped), _normalize(other)),
            (1 + _cosine(real, other)) / 2,
            delta=2e-14,
        )

    def test_unequal_energies_need_calibration_beyond_vanilla_cosine(self):
        q, large, balanced = [1 + 0j], [100 + 0j], [0.8 + 0.6j]
        self.assertGreater(_cosine(_unfold(q), _unfold(large)), _cosine(_unfold(q), _unfold(balanced)))
        self.assertLess(_paper_score(q, large), _paper_score(q, balanced))
        for candidate in (large, balanced):
            self.assertAlmostEqual(_paper_score(q, candidate), _real_score(q, candidate), delta=2e-14)

    def test_zero_energy_self_match_and_antiphase(self):
        q = [1 + 2j, -3 + 1j]
        zero = [0j, 0j]
        for a, b, expected in ((zero, zero, 0), (q, zero, 0), (q, q, 1), (q, [-z for z in q], 0)):
            self.assertAlmostEqual(_paper_score(a, b), expected, delta=2e-14)
            self.assertAlmostEqual(_real_score(a, b), expected, delta=2e-14)

    def test_common_phase_rotation_preserves_score(self):
        rng = random.Random(20261010)
        a, b = _random_pattern(rng), _random_pattern(rng)
        rotation = cmath.rect(1, 0.731)
        rotated_a, rotated_b = [rotation * z for z in a], [rotation * z for z in b]
        self.assertAlmostEqual(_paper_score(a, b), _paper_score(rotated_a, rotated_b), delta=2e-14)
        self.assertAlmostEqual(_real_score(a, b), _real_score(rotated_a, rotated_b), delta=2e-14)

    def test_erasing_phase_removes_information_from_the_baseline(self):
        q, aligned, opposite = [1 + 0j], [1 + 0j], [-1 + 0j]
        self.assertEqual([abs(z) for z in aligned], [abs(z) for z in opposite])
        self.assertAlmostEqual(_paper_score(q, aligned), 1)
        self.assertAlmostEqual(_paper_score(q, opposite), 0)
        self.assertAlmostEqual(_real_score(q, aligned), 1)
        self.assertAlmostEqual(_real_score(q, opposite), 0)


if __name__ == "__main__":
    unittest.main()
