from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_scope_runtime import ActivationEvent, LocalModelRuntime, create_model_scope_runtime


class TestModelScopeRuntime(unittest.TestCase):
    def _fake_runtime_with_events(self, events):
        runtime = LocalModelRuntime(
            model_name="test-model",
            layer_indices=[0, "model.layers.1", "2"],
            artifact_dir=tempfile.mkdtemp(prefix="ms-runtime-"),
            max_input_tokens=8,
            summary_top_dimensions=4,
        )

        runtime.load = lambda: setattr(runtime, "_model", object())
        runtime.run_inference = lambda input_ids, run_id=None: self._install_events(runtime, events, run_id)
        return runtime

    def _install_events(self, runtime, events, run_id):
        resolved_run_id = run_id if run_id is not None else "run-noid"
        for event in events:
            event.run_id = resolved_run_id
        runtime.clear_events()
        runtime._events = list(events)
        return list(events)

    def test_create_model_scope_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = create_model_scope_runtime(
                "Qwen/Qwen3.5-2B",
                layer_indices=[0, "5", "model.layers.9"],
                artifact_dir=td,
                max_input_tokens=12,
                summary_top_dimensions=5,
                eager_load=False,
            )
            self.assertEqual(runtime.model_name, "Qwen/Qwen3.5-2B")
            self.assertEqual(runtime._max_input_tokens, 12)
            self.assertEqual(runtime._summary_top_dimensions, 5)
            self.assertTrue(set(["model.layers.0", "model.layers.5", "model.layers.9"]).issubset(runtime._hook_layers))

    def test_observe_text_writes_artifact(self):
        sample_event = ActivationEvent(
            schema_version="1.0",
            model_id="test-model",
            layer_id="model.layers.0",
            token_count=2,
            shape=(1, 2, 3),
            mean_activation=0.0,
            norm_activation=1.0,
            captured_at="2026-01-01T00:00:00+00:00",
            run_id="run-1",
        )

        runtime = self._fake_runtime_with_events([sample_event])
        artifact = runtime.observe_text("hello world", metadata={"source": "unit-test"})

        self.assertEqual(artifact["runtime"]["model_name"], "test-model")
        self.assertEqual(artifact["capture"]["captured_layer_count"], 1)
        self.assertEqual(artifact["capture"]["captured_layer_ids"], ["model.layers.0"])
        self.assertEqual(artifact["capture"]["metadata"]["source"], "unit-test")

        artifact_path = Path(artifact["output_path"])
        self.assertTrue(artifact_path.exists())
        self.assertEqual(artifact["steering"], None)
        self.assertEqual(artifact["memory"], None)

    def test_observe_text_works_with_no_events(self):
        runtime = self._fake_runtime_with_events([])
        artifact = runtime.observe_text("hello world", metadata={"query_id": "abc"})

        self.assertEqual(artifact["capture"]["captured_layer_count"], 0)
        self.assertEqual(artifact["capture"]["captured_layer_ids"], [])
        self.assertEqual(artifact["capture"]["metadata"]["query_id"], "abc")
        self.assertEqual(artifact["expectation_comparison"]["captured"], False)


if __name__ == "__main__":
    unittest.main()
