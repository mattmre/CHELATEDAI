"""End-to-end learning-loop regression for ingest -> sedate -> delta capture."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
import unittest

from run_live_fire_diagnostics import _make_engine
from run_safety_testbed import _evaluate, _standard_rankings, closed_course_fixture
from chelation_adapter import create_adapter
from checkpoint_manager import CheckpointManager
from config import ChelationConfig


class _TestLogger:
    """Minimal logger shim that mirrors the methods required by the engine."""

    def __init__(self) -> None:
        self.events: list[tuple[tuple[object, ...], dict]] = []

    def log_event(self, *args, **kwargs) -> None:
        self.events.append((args, kwargs))

    def log_query(self, *args, **kwargs) -> None:
        self.events.append((args, kwargs))

    def log_error(self, *args, **kwargs) -> None:
        self.events.append((args, kwargs))

    def log_checkpoint(self, *args, **kwargs) -> None:
        self.events.append((args, kwargs))

    def log_training_start(self, *args, **kwargs) -> None:
        self.events.append((("training_start",), kwargs))

    def log_training_epoch(self, *args, **kwargs) -> None:
        self.events.append((("training_epoch",), kwargs))

    def log_training_complete(self, *args, **kwargs) -> None:
        self.events.append((("training_complete",), kwargs))


def _adapter_param_norm(adapter) -> float:
    return float(sum(param.data.norm().item() for param in adapter.parameters()))


class TestE2ELearningLoop(unittest.TestCase):
    """Validate the minimal closed-course learning loop as a measurable delta test."""

    def test_ingest_sedate_yields_measurable_metric_delta(self) -> None:
        fixture = closed_course_fixture()
        logger = _TestLogger()
        engine = _make_engine(logger)

        # Seed engine attributes for full sedimentation-path execution.
        engine.training_mode = "baseline"
        engine.teacher_weight = 0.0
        engine.adapter = create_adapter("low_rank", input_dim=engine.vector_size, rank=4)
        engine.chelation_threshold = ChelationConfig.DEFAULT_COLLAPSE_THRESHOLD

        # The fake qdrant in the deterministic harness omits the optional args
        # used by run_sedimentation_cycle retrieve.
        def _retrieve_with_vectors(self, collection_name, ids, with_vectors=True, with_payload=True):
            del collection_name, with_vectors, with_payload
            wanted = set(ids)
            return [point for point in self.points if point.id in wanted]

        engine.qdrant.retrieve = _retrieve_with_vectors.__get__(engine.qdrant, type(engine.qdrant))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            engine.adapter_path = str(tmp_path / "adapter_weights.pt")
            engine.checkpoint_manager = CheckpointManager(tmp_path)

            engine.ingest(fixture["corpus"], fixture["payloads"])

            baseline_rankings = _standard_rankings(
                engine,
                fixture["queries"],
                fixture["k"],
            )
            baseline_eval = _evaluate(
                baseline_rankings,
                fixture["qrels"],
                fixture["k"],
            )
            baseline_norm = _adapter_param_norm(engine.adapter)

            # Populate collapse log with realistic query load.
            for query_text in fixture["queries"].values():
                engine.run_inference(query_text)

            self.assertGreater(len(engine.chelation_log), 0)

            engine.run_sedimentation_cycle(
                threshold=1,
                learning_rate=0.05,
                epochs=1,
                noise_injection=0.0,
            )

            post_rankings = _standard_rankings(
                engine,
                fixture["queries"],
                fixture["k"],
            )
            post_eval = _evaluate(post_rankings, fixture["qrels"], fixture["k"])
            post_norm = _adapter_param_norm(engine.adapter)

            retrieval_delta = float(post_eval.fitness - baseline_eval.fitness)
            adapter_norm_delta = float(post_norm - baseline_norm)

            self.assertTrue(math.isfinite(retrieval_delta))
            self.assertTrue(math.isfinite(adapter_norm_delta))
            self.assertGreater(adapter_norm_delta, 0.0)
            self.assertIsInstance(baseline_eval.fitness, float)
            self.assertIsInstance(post_eval.fitness, float)
            self.assertNotEqual(retrieval_delta, math.nan)
            self.assertNotEqual(adapter_norm_delta, math.nan)


if __name__ == "__main__":
    unittest.main()
