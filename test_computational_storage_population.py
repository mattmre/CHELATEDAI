import unittest

from computational_storage_poc.mock_array import ArraySimulation


class TestStorageShardedPopulationEvaluation(unittest.TestCase):
    def test_best_candidate_and_latency_are_reported(self):
        simulation = ArraySimulation(num_drives=4)
        result = simulation.sharded_population_evaluation([
            {"candidate_id": "p0", "fitness": 0.1},
            {"candidate_id": "p1", "fitness": 0.4},
            {"candidate_id": "p2", "fitness": 0.2},
            {"candidate_id": "p3", "fitness": 0.3},
            {"candidate_id": "p4", "fitness": 0.5},
        ])

        self.assertEqual(result["best_candidate_id"], "p4")
        self.assertAlmostEqual(result["best_fitness"], 0.5)
        self.assertEqual(result["drive_depths"][0], 2)
        self.assertGreater(result["storage_latency_ms"], 0.0)

    def test_empty_population_returns_noop_summary(self):
        simulation = ArraySimulation(num_drives=4)
        result = simulation.sharded_population_evaluation([])

        self.assertIsNone(result["best_candidate_id"])
        self.assertIsNone(result["best_fitness"])
        self.assertEqual(result["storage_latency_ms"], 0.0)

    def test_malformed_candidate_is_rejected(self):
        simulation = ArraySimulation(num_drives=4)

        with self.assertRaises(ValueError):
            simulation.sharded_population_evaluation([{"candidate_id": "missing-fitness"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
