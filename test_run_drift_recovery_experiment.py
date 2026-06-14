from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

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
        # Order-independence isolation against torch GLOBAL state a seed does not
        # control. run_experiment() reseeds random/numpy/torch internally, so the
        # seed is not the issue; but the trajectory floats come from a torch
        # matmul whose value/executability depends on torch globals an earlier
        # test in a full-suite run can leak:
        #   - default dtype: a leaked torch.float64 makes adapter init float64 and
        #     CRASHES the matmul (Float vs Double) — pinned to float32.
        #   - default device: a leaked 'cuda' pushes the matmul onto the GPU and
        #     crashes the .numpy() readback (and would diverge low bits) — pinned
        #     to 'cpu' (guarded; the API exists on torch>=2.0).
        #   - intra-op thread count: defensively pinned to 1 for a deterministic
        #     single reduction order.
        # The golden comparison itself is also tolerance-based (see
        # _assert_trajectory_close) so residual low-bit FP differences from any
        # un-enumerated global cannot flake it. State is restored via addCleanup
        # so this test neither depends on nor pollutes sibling tests.
        self._prev_num_threads = torch.get_num_threads()
        self._prev_default_dtype = torch.get_default_dtype()
        self._prev_default_device = None
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        if hasattr(torch, "set_default_device"):
            getter = getattr(torch, "get_default_device", None)
            self._prev_default_device = getter() if getter is not None else torch.device("cpu")
            torch.set_default_device("cpu")
        np.random.seed(0)
        torch.manual_seed(0)
        self.addCleanup(self._restore_torch_state)

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

        def capture_cycle(engine, condition, queries, drift_manifest, run_config=None):
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

    def test_explicit_default_knobs_preserve_default_trajectory(self):
        with self._patched_backend():
            implicit = run_experiment(
                self._config("C3", Path(self.tempdir.name) / "implicit.json"),
                self.corpus,
                self.queries,
                self.qrels,
            )
            explicit_config = self._config("C3", Path(self.tempdir.name) / "explicit.json")
            explicit_config = DriftRecoveryConfig(
                **{
                    **explicit_config.__dict__,
                    "bound_epsilon": 0.01,
                    "trigger_threshold": 0.0,
                    "max_temperature": 1.0,
                    "epochs_scale": 1.0,
                }
            )
            explicit = run_experiment(explicit_config, self.corpus, self.queries, self.qrels)

        # Exact: explicit-default knobs must reproduce the implicit-default
        # trajectory bit-for-bit (same process/environment -> robust to globals).
        self.assertEqual(implicit["recovery"]["trajectory"], explicit["recovery"]["trajectory"])
        # Tolerance on float leaves only: guards against trajectory regressions
        # without flaking on ~1e-16 FP noise from un-enumerated torch globals.
        self._assert_trajectory_close(
            implicit["recovery"]["trajectory"], self._pre_change_c3_default_golden_trajectory()
        )
        first_meta = implicit["recovery"]["trajectory"][0]["metadata"]
        self.assertNotIn("knobs", first_meta)
        self.assertNotIn("epochs_scale", first_meta.get("annealing_settings", {}))

    def test_c3_records_non_default_knobs_and_applies_epoch_scale(self):
        config = self._config("C3", Path(self.tempdir.name) / "knobs.json")
        config = DriftRecoveryConfig(
            **{
                **config.__dict__,
                "bound_epsilon": 0.05,
                "trigger_threshold": 0.0,
                "max_temperature": 0.5,
                "epochs_scale": 2.0,
            }
        )

        with self._patched_backend():
            result = run_experiment(config, self.corpus, self.queries, self.qrels)

        first_meta = result["recovery"]["trajectory"][0]["metadata"]
        self.assertEqual(
            first_meta["knobs"],
            {
                "bound_epsilon": 0.05,
                "trigger_threshold": 0.0,
                "max_temperature": 0.5,
                "epochs_scale": 2.0,
            },
        )
        self.assertEqual(first_meta["annealing_settings"]["epochs_scale"], 2.0)
        self.assertGreaterEqual(first_meta["annealing_settings"]["effective_epochs"], 2)
        self.assertGreater(result["correction_norm_stats"]["mean"], 0.03)

    def test_invalid_knob_values_raise(self):
        bad_config = self._config("C3", Path(self.tempdir.name) / "bad.json")
        bad_config = DriftRecoveryConfig(**{**bad_config.__dict__, "bound_epsilon": 0.0})

        with self.assertRaisesRegex(ValueError, "bound_epsilon"):
            run_experiment(bad_config, self.corpus, self.queries, self.qrels)

    def _restore_torch_state(self):
        # Restore (via addCleanup) the global torch knobs we pinned in setUp so
        # this test does not leak single-threaded / float32 / cpu state onto
        # later tests.
        torch.set_num_threads(self._prev_num_threads)
        torch.set_default_dtype(self._prev_default_dtype)
        if self._prev_default_device is not None and hasattr(torch, "set_default_device"):
            torch.set_default_device(self._prev_default_device)

    def _assert_trajectory_close(self, actual, expected, path="trajectory"):
        """Compare two trajectory structures: float leaves within a tight
        tolerance (robust to ~1e-16 FP-environment noise), everything else
        (ints, bools, strings, list lengths, dict keys) exact. Keeps the test a
        real regression guard while immune to un-enumerated FP-affecting globals.
        """
        if isinstance(expected, bool) or isinstance(actual, bool):
            self.assertEqual(actual, expected, msg=path)
        elif isinstance(expected, float) or isinstance(actual, float):
            self.assertIsInstance(actual, (int, float), msg=path)
            self.assertTrue(
                np.isclose(float(actual), float(expected), rtol=1e-9, atol=1e-12),
                msg=f"{path}: {actual!r} != {expected!r} (beyond tolerance)",
            )
        elif isinstance(expected, dict):
            self.assertIsInstance(actual, dict, msg=path)
            self.assertEqual(set(actual.keys()), set(expected.keys()), msg=path)
            for key in expected:
                self._assert_trajectory_close(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected), msg=path)
            for i, (a, e) in enumerate(zip(actual, expected)):
                self._assert_trajectory_close(a, e, f"{path}[{i}]")
        else:
            self.assertEqual(actual, expected, msg=path)

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

    def _pre_change_c3_default_golden_trajectory(self):
        base_metadata = {
            "action": "detection_triggered_sedimentation",
            "annealing_observation": {
                "drift_magnitude": 0.2132925720177655,
                "should_correct": True,
                "temperature": 0.2132925720177655,
            },
            "annealing_settings": {
                "effective_epochs": 1,
                "effective_learning_rate": 0.00029196331481598893,
                "learning_rate_scale": 0.2919633148159889,
                "online_intensity": 0.2132925720177655,
                "original_epochs": 1,
                "original_learning_rate": 0.001,
                "temperature": 0.2132925720177655,
            },
            "bounded": True,
            "condition": "C3",
            "correction_applied": False,
            "correction_norm_stats": {
                "count": 4,
                "max": 0.009997744113206863,
                "mean": 0.007490877062082291,
                "sample_norms": [
                    0.009968805126845837,
                    0.0,
                    0.009997744113206863,
                    0.009996959008276463,
                ],
            },
            "detector_source": "AntigravityEngine._compute_annealing_drift_magnitude",
            "evaluated_queries": 2,
            "query_ndcg": [
                {
                    "ndcg": 1.0,
                    "query_id": "q-alpha",
                    "ranked_ids": ["d-alpha", "d-delta", "d-gamma", "d-beta"],
                    "relevant_ids": ["d-alpha"],
                },
                {
                    "ndcg": 1.0,
                    "query_id": "q-beta",
                    "ranked_ids": ["d-beta", "d-gamma", "d-delta", "d-alpha"],
                    "relevant_ids": ["d-beta"],
                },
            ],
            "sedimentation_attempted": True,
            "should_correct": True,
        }
        return [
            {"cycle_index": 1, "metadata": base_metadata, "ndcg": 1.0},
            {"cycle_index": 2, "metadata": dict(base_metadata), "ndcg": 1.0},
        ]


if __name__ == "__main__":
    unittest.main()
