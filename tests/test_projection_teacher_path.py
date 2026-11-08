"""Engine-path projection training.

The embedding backend is a dim-8 fixture. The teacher embeddings are a dim-16
stand-in. Neither is MiniLM or a BEIR encoder. The projection and the loss
are the real modules.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from antigravity_engine import AntigravityEngine  # noqa: E402
from benchmark_utils import isolated_adapter_state  # noqa: E402
from teacher_distillation import DimensionProjection  # noqa: E402

_DOCUMENTS = [f"doc-{index}" for index in range(4)]


class _FloorDim8Backend:
    vector_size = 8

    def embed_raw(self, texts):
        rows = []
        for index, _text in enumerate(texts):
            vector = np.zeros(self.vector_size, dtype=np.float32)
            vector[index % 8] = 1.0
            vector[0] = 0.25
            rows.append(vector)
        return np.vstack(rows)


def _teacher_rows(texts):
    rows = np.zeros((len(texts), 16), dtype=np.float32)
    for index in range(len(texts)):
        rows[index, index % 16] = 1.0
        rows[index, 1] = 0.4
    return rows


def _arm_teacher(engine):
    helper = engine.teacher_helper
    helper.teacher_dim = 16

    def _load():
        helper.teacher_dim = 16

    helper.load_teacher_model = _load
    helper.get_teacher_embeddings = _teacher_rows
    return helper


def _snapshot(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _moved(before, module):
    return any(
        not torch.equal(previous, parameter)
        for previous, parameter in zip(before, module.parameters())
    )


def _grad_reached_projection(module):
    reached = False
    for parameter in module.parameters():
        if parameter.grad is None:
            return False
        if float(parameter.grad.detach().abs().sum()) > 0:
            reached = True
    return reached


@contextmanager
def _engine(training_mode):
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="projection-teacher-") as temp_dir:
        os.chdir(temp_dir)
        try:
            with isolated_adapter_state():
                with patch("antigravity_engine.get_logger", return_value=MagicMock()), patch(
                    "teacher_distillation.get_logger", return_value=MagicMock()
                ), patch(
                    "antigravity_engine.create_embedding_backend",
                    return_value=_FloorDim8Backend(),
                ):
                    torch.manual_seed(0)
                    np.random.seed(0)
                    engine = AntigravityEngine(
                        qdrant_location=":memory:",
                        use_centering=True,
                        training_mode=training_mode,
                        store_full_text_payload=True,
                        model_name="floor-dim8-fixture",
                    )
                    yield engine
        finally:
            os.chdir(cwd)


@contextmanager
def _capture_projection_init():
    """Record projection weights at construction, before the training step."""
    captured = []
    original = DimensionProjection.__init__

    def _init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        captured.append(_snapshot(self))

    DimensionProjection.__init__ = _init
    try:
        yield captured
    finally:
        DimensionProjection.__init__ = original


class TestProjectionTeacherPath(unittest.TestCase):
    def _prepare(self, engine, fill_log):
        _arm_teacher(engine)
        engine.ingest(_DOCUMENTS)
        if fill_log:
            engine.run_inference("query-seed")
            if not any(len(events) >= 1 for events in engine.chelation_log.values()):
                for doc_id in range(len(_DOCUMENTS)):
                    engine.chelation_log[doc_id].append(np.zeros(8, dtype=np.float32))

    def test_sedimentation_moves_projection_without_detaching_targets(self):
        with _engine("offline") as engine:
            self._prepare(engine, fill_log=True)
            with _capture_projection_init() as captured:
                engine.run_sedimentation_cycle(threshold=1, learning_rate=0.5, epochs=1)
            projection = engine.teacher_helper._projection
            self.assertIsNotNone(projection)
            self.assertTrue(captured)
            self.assertTrue(_moved(captured[0], projection))
            self.assertTrue(_grad_reached_projection(projection))

    def test_offline_distillation_moves_projection_without_detaching_targets(self):
        with _engine("offline") as engine:
            self._prepare(engine, fill_log=False)
            engine.run_offline_distillation(batch_size=2, learning_rate=0.5, epochs=1)
            projection = engine.teacher_helper._projection
            self.assertIsNotNone(projection)
            before = _snapshot(projection)
            engine.run_offline_distillation(batch_size=2, learning_rate=0.5, epochs=1)
            self.assertTrue(_moved(before, projection))
            self.assertTrue(_grad_reached_projection(projection))

    def test_sedimentation_es_branch_moves_projection(self):
        with _engine("offline") as engine:
            self._prepare(engine, fill_log=True)
            engine._sedimentation_optimizer_type = "eggroll_es"
            engine._es_optimizer_kwargs = {
                "population_size": 4,
                "generations": 1,
                "sigma": 0.2,
                "learning_rate": 0.2,
                "seed": 3,
            }
            with _capture_projection_init() as captured:
                engine.run_sedimentation_cycle(threshold=1, learning_rate=0.2, epochs=1)
            projection = engine.teacher_helper._projection
            self.assertIsNotNone(projection)
            self.assertTrue(captured)
            self.assertTrue(_moved(captured[0], projection))

    def test_offline_es_branch_moves_projection(self):
        with _engine("offline") as engine:
            self._prepare(engine, fill_log=False)
            engine._sedimentation_optimizer_type = "eggroll_es"
            engine._es_optimizer_kwargs = {
                "population_size": 4,
                "generations": 1,
                "sigma": 0.2,
                "learning_rate": 0.2,
                "seed": 5,
            }
            engine.run_offline_distillation(batch_size=2, learning_rate=0.2, epochs=1)
            projection = engine.teacher_helper._projection
            self.assertIsNotNone(projection)
            before = _snapshot(projection)
            engine.run_offline_distillation(batch_size=2, learning_rate=0.2, epochs=1)
            self.assertTrue(_moved(before, projection))


if __name__ == "__main__":
    unittest.main()
