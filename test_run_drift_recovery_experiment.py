from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from chelation_adapter import BoundedAdapter
from run_drift_recovery_experiment import DriftRecoveryConfig, CONDITIONS, run_experiment


class TinyEmbeddingBackend:
    vector_size = 4

    def __init__(self):
        self.vectors = {
            "alpha topic": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "beta topic": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "gamma topic": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "delta topic": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }

    def embed_raw(self, texts):
        rows = []
        for text in texts:
            vector = self.vectors.get(str(text), np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32))
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype=np.float32)


class TestRunDriftRecoveryExperiment(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.corpus = {
            "d-alpha": "alpha topic",
            "d-beta": "beta topic",
            "d-gamma": "gamma topic",
            "d-delta": "delta topic",
        }
        self.queries = {
            "q-alpha": "alpha topic",
            "q-beta": "beta topic",
        }
        self.qrels = {
            "q-alpha": {"d-alpha": 1.0},
            "q-beta": {"d-beta": 1.0},
        }

    def test_all_conditions_write_well_formed_json_through_engine_path(self):
        with self._patched_backend():
            for condition in CONDITIONS:
                output = Path(self.tempdir.name) / f"{condition}.json"
                config = self._config(condition, output)
                result = run_experiment(config, self.corpus, self.queries, self.qrels)

                self.assertEqual(result["record_type"], "drift_recovery_experiment")
                self.assertEqual(result["config"]["condition"], condition)
                self.assertEqual(result["config"]["seed"], 123)
                self.assertEqual(result["config"]["injection_index"], 0)
                self.assertEqual(result["drift_manifest"]["injection_index"], 0)
                self.assertEqual(len(result["recovery"]["trajectory"]), 2)
                self.assertIn("baseline_ndcg", result["recovery"])
                self.assertTrue(output.exists())

                loaded = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(loaded["config"]["condition"], condition)
                self.assertEqual(loaded["drift_manifest"]["seed"], 123)
                self.assertEqual(loaded["cycle_errors"], [])
                if condition == "C2":
                    first_meta = loaded["recovery"]["trajectory"][0]["metadata"]
                    self.assertEqual(first_meta["action"], "maintenance_reindex_affected")
                    self.assertEqual(first_meta["refresh"]["updated"], loaded["drift_manifest"]["affected_count"])
                if condition in {"C3", "C4"}:
                    self.assertGreater(loaded["correction_norm_stats"]["count"], 0)
                    first_meta = loaded["recovery"]["trajectory"][0]["metadata"]
                    self.assertEqual(
                        first_meta["detector_source"],
                        "AntigravityEngine._compute_annealing_drift_magnitude",
                    )
                    self.assertIn("drift_magnitude", first_meta["annealing_observation"])
                    self.assertIn("should_correct", first_meta)
                    self.assertIn("sedimentation_attempted", first_meta)
                    self.assertIn("correction_applied", first_meta)
                    self.assertNotIn("corrected", first_meta)

    def test_c3_uses_bounded_adapter_and_c4_uses_unbounded_adapter(self):
        observed_adapter_types = {}

        def capture_cycle(engine, condition, queries, drift_manifest):
            observed_adapter_types[condition] = isinstance(engine.adapter, BoundedAdapter)
            return {"action": "captured"}

        with self._patched_backend(), patch(
            "run_drift_recovery_experiment._run_condition_cycle",
            side_effect=capture_cycle,
        ):
            run_experiment(self._config("C3", Path(self.tempdir.name) / "c3.json"), self.corpus, self.queries, self.qrels)
            run_experiment(self._config("C4", Path(self.tempdir.name) / "c4.json"), self.corpus, self.queries, self.qrels)

        self.assertTrue(observed_adapter_types["C3"])
        self.assertFalse(observed_adapter_types["C4"])

    def test_same_seed_twice_produces_identical_trajectory(self):
        with self._patched_backend():
            first = run_experiment(
                self._config("C3", Path(self.tempdir.name) / "first.json"),
                self.corpus,
                self.queries,
                self.qrels,
            )
            second = run_experiment(
                self._config("C3", Path(self.tempdir.name) / "second.json"),
                self.corpus,
                self.queries,
                self.qrels,
            )

        self.assertEqual(first["drift_manifest"], second["drift_manifest"])
        self.assertEqual(first["recovery"]["trajectory"], second["recovery"]["trajectory"])

    def _config(self, condition, output):
        return DriftRecoveryConfig(
            task="Tiny",
            condition=condition,
            drift="rotation",
            fraction=0.5,
            angle=25.0,
            sigma=0.05,
            cycles=2,
            seed=123,
            max_queries=2,
            sample_docs=4,
            output=str(output),
            model="tiny-local",
            device="cpu",
        )

    def _patched_backend(self):
        @contextmanager
        def manager():
            with ExitStack() as stack:
                stack.enter_context(
                    patch("antigravity_engine.create_embedding_backend", return_value=TinyEmbeddingBackend())
                )
                stack.enter_context(patch("antigravity_engine.get_logger", return_value=MagicMock()))
                yield

        return manager()


if __name__ == "__main__":
    unittest.main()
