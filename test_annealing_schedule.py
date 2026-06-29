"""Tests for annealing_schedule.py — the post-bank lifecycle temperature schedule
(Phase II, H6, rung 11). stdlib-only, no torch/GPU."""
from __future__ import annotations

import json
import unittest

from annealing_schedule import (
    SCHEDULES,
    AnnealingSchedule,
    create_annealing_schedule,
)


class TestDeterministicSchedules(unittest.TestCase):
    def test_constant_is_flat_at_t_start(self):
        s = AnnealingSchedule("constant", n_cycles=5, t_start=0.7, t_end=0.0)
        self.assertEqual([s.temperature(c) for c in range(5)], [0.7] * 5)

    def test_linear_endpoints_and_monotonic(self):
        s = AnnealingSchedule("linear", n_cycles=5, t_start=1.0, t_end=0.0)
        temps = s.schedule_over_cycles()
        self.assertAlmostEqual(temps[0], 1.0)   # explore at start
        self.assertAlmostEqual(temps[-1], 0.0)  # stabilize at end
        self.assertTrue(all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1)))
        self.assertAlmostEqual(temps[2], 0.5)   # midpoint of 5 cycles

    def test_cosine_endpoints_and_monotonic_decreasing(self):
        s = AnnealingSchedule("cosine", n_cycles=7, t_start=1.0, t_end=0.0)
        temps = s.schedule_over_cycles()
        self.assertAlmostEqual(temps[0], 1.0)
        self.assertAlmostEqual(temps[-1], 0.0)
        self.assertTrue(all(temps[i] >= temps[i + 1] - 1e-12 for i in range(len(temps) - 1)))

    def test_step_switches_at_fraction(self):
        s = AnnealingSchedule("step", n_cycles=10, t_start=1.0, t_end=0.0, step_fraction=0.5)
        temps = s.schedule_over_cycles()
        # progress < 0.5 -> explore (1.0); progress >= 0.5 -> stabilize (0.0)
        self.assertEqual(temps[0], 1.0)
        self.assertEqual(temps[-1], 0.0)
        # exactly one transition, from 1.0 to 0.0
        transitions = [i for i in range(1, len(temps)) if temps[i] != temps[i - 1]]
        self.assertEqual(len(transitions), 1)

    def test_n_cycles_one_collapses_to_t_end_for_decaying_schedules(self):
        # progress is 1.0 when n_cycles<=1 -> linear/cosine return t_end.
        self.assertAlmostEqual(AnnealingSchedule("linear", n_cycles=1, t_start=1.0, t_end=0.2).temperature(0), 0.2)
        self.assertAlmostEqual(AnnealingSchedule("cosine", n_cycles=1, t_start=1.0, t_end=0.2).temperature(0), 0.2)
        self.assertAlmostEqual(AnnealingSchedule("constant", n_cycles=1, t_start=0.9).temperature(0), 0.9)

    def test_temperatures_clamped_to_unit_interval(self):
        # Even with out-of-range t_start/t_end (clamped at init), outputs stay [0,1].
        s = AnnealingSchedule("linear", n_cycles=4, t_start=5.0, t_end=-3.0)
        for c in range(4):
            t = s.temperature(c)
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, 1.0)

    def test_cycle_index_is_clamped(self):
        s = AnnealingSchedule("linear", n_cycles=5, t_start=1.0, t_end=0.0)
        self.assertEqual(s.temperature(-10), s.temperature(0))
        self.assertEqual(s.temperature(99), s.temperature(4))


class TestAdaptiveSchedule(unittest.TestCase):
    def test_adaptive_maps_drift_to_temperature(self):
        s = AnnealingSchedule("adaptive", n_cycles=12, t_start=1.0, t_end=0.0, drift_reference=0.5)
        self.assertAlmostEqual(s.temperature(0, drift_magnitude=0.0), 0.0)   # no drift -> stabilize
        self.assertAlmostEqual(s.temperature(0, drift_magnitude=0.5), 1.0)   # ref drift -> explore
        self.assertAlmostEqual(s.temperature(0, drift_magnitude=10.0), 1.0)  # saturates
        self.assertAlmostEqual(s.temperature(0, drift_magnitude=0.25), 0.5)  # halfway

    def test_adaptive_requires_drift_magnitude(self):
        s = AnnealingSchedule("adaptive", n_cycles=12)
        with self.assertRaises(ValueError):
            s.temperature(0)

    def test_schedule_over_cycles_invalid_for_adaptive(self):
        with self.assertRaises(ValueError):
            AnnealingSchedule("adaptive", n_cycles=5).schedule_over_cycles()


class TestContractAndSerialization(unittest.TestCase):
    def test_validations(self):
        with self.assertRaises(ValueError):
            AnnealingSchedule("nope", n_cycles=5)
        with self.assertRaises(ValueError):
            AnnealingSchedule("linear", n_cycles=0)
        with self.assertRaises(ValueError):
            AnnealingSchedule("step", n_cycles=5, step_fraction=1.5)
        with self.assertRaises(ValueError):
            AnnealingSchedule("adaptive", n_cycles=5, drift_reference=0.0)

    def test_factory_builds_equivalent_schedule(self):
        s = create_annealing_schedule("cosine", n_cycles=8, t_start=0.8)
        self.assertIsInstance(s, AnnealingSchedule)
        self.assertEqual(s.schedule, "cosine")
        self.assertEqual(s.n_cycles, 8)
        self.assertAlmostEqual(s.t_start, 0.8)

    def test_to_dict_is_json_safe_with_temperatures(self):
        d = AnnealingSchedule("linear", n_cycles=4).to_dict()
        json.dumps(d)
        self.assertEqual(d["record_type"], "annealing_schedule")
        self.assertEqual(len(d["temperatures"]), 4)

    def test_to_dict_adaptive_has_no_temperatures(self):
        d = AnnealingSchedule("adaptive", n_cycles=4).to_dict()
        json.dumps(d)
        self.assertNotIn("temperatures", d)

    def test_schedules_constant_exposes_all_five(self):
        self.assertEqual(set(SCHEDULES), {"constant", "linear", "cosine", "step", "adaptive"})


if __name__ == "__main__":
    unittest.main()
