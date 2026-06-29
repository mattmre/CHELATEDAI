"""Regression tests for the CUDA telemetry guard in
``AntigravityEngine.get_runtime_telemetry``.

A GPU-hidden / CPU-only environment (e.g. ``CUDA_VISIBLE_DEVICES=""`` or ``"-1"``,
or CI without a GPU) can have ``torch.cuda.is_available()`` return True while
``device_count() == 0``. The telemetry path then called ``get_device_name(0)``
and crashed with ``AssertionError: Invalid device id``. These tests pin the
``device_count() > 0`` guard so telemetry never raises off-GPU.

Stub-only: builds the engine via ``object.__new__`` and patches ``torch.cuda`` —
no model load, no GPU required.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from antigravity_engine import AntigravityEngine


def _stub_engine() -> AntigravityEngine:
    eng = object.__new__(AntigravityEngine)
    eng._runtime_telemetry = {}
    eng.mode = "local"
    eng.model_name = "stub-model"
    eng.vector_size = 4
    eng._model_scope_runtime = None
    return eng


class TestTelemetryCudaGuard(unittest.TestCase):
    def test_no_crash_when_cuda_available_but_zero_visible_devices(self):
        eng = _stub_engine()
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda, "device_count", return_value=0
        ):
            tel = eng.get_runtime_telemetry()
        self.assertTrue(tel["torch_cuda_available"])
        self.assertIsNone(tel["cuda_device_name"])
        self.assertEqual(tel["cuda_memory_allocated_mb"], 0.0)

    def test_happy_path_reports_device_when_visible(self):
        eng = _stub_engine()
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda, "device_count", return_value=1
        ), patch.object(
            torch.cuda, "get_device_name", return_value="FakeGPU"
        ), patch.object(
            torch.cuda, "memory_allocated", return_value=0
        ), patch.object(
            torch.cuda, "memory_reserved", return_value=0
        ), patch.object(
            torch.cuda, "max_memory_allocated", return_value=0
        ):
            tel = eng.get_runtime_telemetry()
        self.assertEqual(tel["cuda_device_name"], "FakeGPU")

    def test_cpu_only_reports_none(self):
        eng = _stub_engine()
        with patch.object(torch.cuda, "is_available", return_value=False):
            tel = eng.get_runtime_telemetry()
        self.assertFalse(tel["torch_cuda_available"])
        self.assertIsNone(tel["cuda_device_name"])


if __name__ == "__main__":
    unittest.main()
