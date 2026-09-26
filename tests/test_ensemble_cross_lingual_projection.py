"""Ensemble and cross-lingual projections train on a live tensor.

The single-teacher engine path is pull 338. These callers were still
project_numpy. The embedding backend in the engine case is a dim-8 fixture.
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
from cross_lingual_distillation import (  # noqa: E402
    CrossLingualTeacherRouter,
    LanguageTeacherMapping,
)
from teacher_distillation import EnsembleTeacherHelper, TeacherDistillationHelper  # noqa: E402


def _teacher(dim):
    teacher = MagicMock(spec=TeacherDistillationHelper)
    teacher.teacher_dim = dim

    def _embed(texts, *_args, **_kwargs):
        rows = np.zeros((len(texts), dim), dtype=np.float32)
        for index in range(len(texts)):
            rows[index, index % dim] = 1.0
            rows[index, 0] = 0.2
        return rows

    teacher.get_teacher_embeddings.side_effect = _embed
    return teacher


def _moved_and_graded(modules):
    before = {
        id(module): [parameter.detach().clone() for parameter in module.parameters()]
        for module in modules
    }
    return before


def _assert_trained(test, before, modules):
    trained = False
    for module in modules:
        previous = before[id(module)]
        for old, parameter in zip(previous, module.parameters()):
            if parameter.grad is None or float(parameter.grad.detach().abs().sum()) == 0:
                continue
            if not torch.equal(old, parameter):
                trained = True
    test.assertTrue(trained)


class _FloorDim8Backend:
    vector_size = 8

    def embed_raw(self, texts):
        rows = []
        for index, _text in enumerate(texts):
            vector = np.zeros(self.vector_size, dtype=np.float32)
            vector[index % 8] = 1.0
            rows.append(vector)
        return np.vstack(rows)


@contextmanager
def _offline_engine():
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="ensemble-proj-") as temp_dir:
        os.chdir(temp_dir)
        try:
            with isolated_adapter_state():
                with patch("antigravity_engine.get_logger", return_value=MagicMock()), patch(
                    "teacher_distillation.get_logger", return_value=MagicMock()
                ), patch(
                    "cross_lingual_distillation.get_logger", return_value=MagicMock()
                ), patch(
                    "antigravity_engine.create_embedding_backend",
                    return_value=_FloorDim8Backend(),
                ):
                    torch.manual_seed(0)
                    engine = AntigravityEngine(
                        qdrant_location=":memory:",
                        training_mode="offline",
                        store_full_text_payload=True,
                        model_name="floor-dim8-fixture",
                    )
                    yield engine
        finally:
            os.chdir(cwd)


class TestEnsembleAndCrossLingualProjection(unittest.TestCase):
    def test_ensemble_mismatch_keeps_a_live_tensor(self):
        ensemble = EnsembleTeacherHelper(
            [_teacher(16), _teacher(8)],
            weights=[0.5, 0.5],
            logger=MagicMock(),
            parallel_encoding=False,
        )
        student = np.random.randn(3, 8).astype(np.float32)
        numpy_targets = ensemble.generate_distillation_targets(
            ["a", "b", "c"], student, teacher_weight=1.0
        )
        live = ensemble.take_live_distillation_targets()
        self.assertIsInstance(numpy_targets, np.ndarray)
        self.assertTrue(live.requires_grad)
        self.assertIsNotNone(live.grad_fn)
        modules = list(ensemble._projections.values())
        self.assertEqual(len(modules), 1)
        before = _moved_and_graded(modules)
        loss = torch.nn.MSELoss()(torch.zeros_like(live), live)
        loss.backward()
        for module in modules:
            for parameter in module.parameters():
                if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0:
                    parameter.data.add_(-0.5 * parameter.grad)
        _assert_trained(self, before, modules)

    def test_cross_lingual_mismatch_keeps_a_live_tensor(self):
        mapping = LanguageTeacherMapping({"en": "wide-teacher"}, default_teacher="wide-teacher")
        router = CrossLingualTeacherRouter(
            mapping, detector=MagicMock(), projection_enabled=True, teacher_weight=1.0
        )
        router.detector.detect_batch.return_value = ["en", "en", "en"]
        wide = _teacher(16)
        router._teachers["wide-teacher"] = wide
        student = np.random.randn(3, 8).astype(np.float32)
        numpy_targets = router.generate_distillation_targets(
            ["a", "b", "c"], student, teacher_weight=1.0
        )
        live = router.take_live_distillation_targets()
        self.assertIsInstance(numpy_targets, np.ndarray)
        self.assertTrue(live.requires_grad)
        modules = list(router._projections.values())
        before = _moved_and_graded(modules)
        loss = torch.nn.MSELoss()(torch.zeros_like(live), live)
        loss.backward()
        for module in modules:
            for parameter in module.parameters():
                if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0:
                    parameter.data.add_(-0.5 * parameter.grad)
        _assert_trained(self, before, modules)

    def _run_engine(self, helper):
        with _offline_engine() as engine:
            helper.load_teacher_model = lambda: None
            engine.teacher_helper = helper
            engine.ingest(["one", "two", "three", "four"])
            for doc_id in range(4):
                engine.chelation_log[doc_id].append(np.zeros(8, dtype=np.float32))
            engine.run_sedimentation_cycle(threshold=1, learning_rate=0.5, epochs=1)
            modules = list(helper._projections.values())
            self.assertTrue(modules)
            self.assertTrue(
                any(
                    parameter.grad is not None and float(parameter.grad.detach().abs().sum()) > 0
                    for module in modules
                    for parameter in module.parameters()
                )
            )

    def test_engine_es_moves_ensemble_projection(self):
        ensemble = EnsembleTeacherHelper(
            [_teacher(16)],
            weights=[1.0],
            logger=MagicMock(),
            parallel_encoding=False,
        )
        student = np.random.randn(2, 8).astype(np.float32)
        ensemble.generate_distillation_targets(["a", "b"], student, teacher_weight=1.0)
        module = ensemble._projections[0]
        before = [parameter.detach().clone() for parameter in module.parameters()]
        with _offline_engine() as engine:
            engine.teacher_helper = ensemble
            engine._sedimentation_optimizer_type = "eggroll_es"
            engine._es_optimizer_kwargs = {
                "population_size": 4,
                "generations": 1,
                "sigma": 0.2,
                "learning_rate": 0.2,
                "seed": 2,
            }
            engine.ingest(["one", "two", "three", "four"])
            for doc_id in range(4):
                engine.chelation_log[doc_id].append(np.zeros(8, dtype=np.float32))
            engine.run_sedimentation_cycle(threshold=1, learning_rate=0.2, epochs=1)
        self.assertTrue(
            any(
                not torch.equal(old, parameter)
                for old, parameter in zip(before, module.parameters())
            )
        )

    def test_engine_sedimentation_trains_ensemble_projection(self):
        ensemble = EnsembleTeacherHelper(
            [_teacher(16), _teacher(16)],
            weights=[0.4, 0.6],
            logger=MagicMock(),
            parallel_encoding=False,
        )
        self._run_engine(ensemble)

    def test_engine_sedimentation_trains_cross_lingual_projection(self):
        mapping = LanguageTeacherMapping({"en": "wide-teacher"}, default_teacher="wide-teacher")
        router = CrossLingualTeacherRouter(
            mapping, detector=MagicMock(), projection_enabled=True, teacher_weight=1.0
        )
        router.detector.detect_batch.return_value = ["en"] * 4
        router._teachers["wide-teacher"] = _teacher(16)
        self._run_engine(router)


if __name__ == "__main__":
    unittest.main()
