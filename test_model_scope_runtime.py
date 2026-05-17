"""Tests for model_scope_runtime.py — LocalModelRuntime and ActivationEvent.

Integration tests that exercise the real transformers code path are gated behind
the ``CHELATED_INTEGRATION_MODEL`` environment variable.  These tests are skipped
in CI because the weights are too large to download on hosted runners.  To run
them locally::

    CHELATED_INTEGRATION_MODEL=sshleifer/tiny-gpt2 python test_model_scope_runtime.py

``sshleifer/tiny-gpt2`` is ~5 MB and exercises the exact same
``AutoModelForCausalLM.from_pretrained`` → hook registration → ``run_inference``
code path that the Qwen3.5-9B pilot would use.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from model_scope_runtime import ActivationEvent, LocalModelRuntime


def _make_activation_event(**kwargs) -> ActivationEvent:
    defaults = {
        "schema_version": "1.0",
        "model_id": "Qwen/Qwen3.5-2B",
        "layer_id": "model.layers.15",
        "token_count": 12,
        "shape": (1, 12, 2048),
        "mean_activation": 0.03714,
        "norm_activation": 42.618,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(uuid.uuid4()),
    }
    defaults.update(kwargs)
    return ActivationEvent(**defaults)


def _mock_model_loader(layer_ids=None):
    """Return (loader_callable, mock_model, layer_mocks_dict)."""
    layer_ids = layer_ids or ["model.layers.0", "model.layers.15"]
    model = MagicMock()

    layer_mocks = {}
    for lid in layer_ids:
        parts = lid.split(".")
        obj = model
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                lm = MagicMock()
                lm.register_forward_hook = MagicMock(return_value=MagicMock())
                setattr(obj, part, lm)
                layer_mocks[lid] = lm
            else:
                child = MagicMock()
                setattr(obj, part, child)
                obj = child

    def loader(name):
        return model

    return loader, model, layer_mocks


class TestActivationEventConstruction(unittest.TestCase):
    def test_basic_construction(self):
        run_id = str(uuid.uuid4())
        evt = ActivationEvent(
            schema_version="1.0",
            model_id="Qwen/Qwen3.5-2B",
            layer_id="model.layers.10",
            token_count=8,
            shape=(1, 8, 2048),
            mean_activation=0.0127,
            norm_activation=18.44,
            captured_at="2025-01-01T00:00:00+00:00",
            run_id=run_id,
        )
        self.assertEqual(evt.schema_version, "1.0")
        self.assertEqual(evt.model_id, "Qwen/Qwen3.5-2B")
        self.assertEqual(evt.layer_id, "model.layers.10")
        self.assertEqual(evt.token_count, 8)
        self.assertEqual(evt.shape, (1, 8, 2048))
        self.assertAlmostEqual(evt.mean_activation, 0.0127)
        self.assertAlmostEqual(evt.norm_activation, 18.44)
        self.assertEqual(evt.run_id, run_id)

    def test_raw_tensor_shape_defaults_to_shape(self):
        """raw_tensor_shape documents the true tensor dimensions at capture time."""
        evt = ActivationEvent(
            schema_version="1.0",
            model_id="Qwen/Qwen3.5-2B",
            layer_id="model.layers.10",
            token_count=8,
            shape=(1, 8, 2048),
            mean_activation=0.0127,
            norm_activation=18.44,
            captured_at="2025-01-01T00:00:00+00:00",
            run_id=str(uuid.uuid4()),
        )
        # raw_tensor_shape must match shape — it records the actual captured
        # tensor dimensions (the full tensor data is NOT stored).
        self.assertEqual(evt.raw_tensor_shape, (1, 8, 2048))

    def test_raw_tensor_shape_explicit_override(self):
        """raw_tensor_shape can be set explicitly when shape differs from capture."""
        evt = ActivationEvent(
            schema_version="1.0",
            model_id="m",
            layer_id="l",
            token_count=4,
            shape=(1, 4, 512),
            mean_activation=0.0,
            norm_activation=1.0,
            captured_at="2025-01-01T00:00:00+00:00",
            run_id=str(uuid.uuid4()),
            raw_tensor_shape=(1, 4, 512),
        )
        self.assertEqual(evt.raw_tensor_shape, (1, 4, 512))

    def test_raw_tensor_shape_is_tuple(self):
        evt = _make_activation_event()
        self.assertIsInstance(evt.raw_tensor_shape, tuple)

    def test_to_dict_serialization(self):
        evt = _make_activation_event()
        d = evt.to_dict()
        self.assertEqual(d["schema_version"], "1.0")
        self.assertIsInstance(d["shape"], list)
        self.assertEqual(d["shape"], [1, 12, 2048])
        self.assertIn("run_id", d)
        self.assertIn("captured_at", d)
        # raw_tensor_shape must be serialised as a list (JSON-compatible)
        self.assertIn("raw_tensor_shape", d)
        self.assertIsInstance(d["raw_tensor_shape"], list)
        self.assertEqual(d["raw_tensor_shape"], [1, 12, 2048])

    def test_from_dict_roundtrip(self):
        evt = _make_activation_event()
        d = evt.to_dict()
        evt2 = ActivationEvent.from_dict(d)
        self.assertEqual(evt2.model_id, evt.model_id)
        self.assertEqual(evt2.layer_id, evt.layer_id)
        self.assertEqual(evt2.shape, (1, 12, 2048))
        self.assertEqual(evt2.token_count, evt.token_count)
        self.assertAlmostEqual(evt2.mean_activation, evt.mean_activation)
        self.assertAlmostEqual(evt2.norm_activation, evt.norm_activation)
        # raw_tensor_shape must survive the roundtrip as a tuple
        self.assertIsInstance(evt2.raw_tensor_shape, tuple)
        self.assertEqual(evt2.raw_tensor_shape, (1, 12, 2048))

    def test_json_serializable(self):
        evt = _make_activation_event()
        raw = json.dumps(evt.to_dict())
        parsed = json.loads(raw)
        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertIsInstance(parsed["shape"], list)

    def test_shape_roundtrip_is_tuple(self):
        evt = _make_activation_event(shape=(2, 16, 4096))
        d = evt.to_dict()
        evt2 = ActivationEvent.from_dict(d)
        self.assertIsInstance(evt2.shape, tuple)
        self.assertEqual(evt2.shape, (2, 16, 4096))


class TestLocalModelRuntimeInit(unittest.TestCase):
    def test_init_stores_model_name(self):
        rt = LocalModelRuntime("Qwen/Qwen3.5-2B")
        self.assertEqual(rt.model_name, "Qwen/Qwen3.5-2B")

    def test_init_default_device(self):
        rt = LocalModelRuntime("mistral-7b")
        self.assertEqual(rt.device, "cpu")

    def test_init_custom_device(self):
        rt = LocalModelRuntime("mistral-7b", device="cuda")
        self.assertEqual(rt.device, "cuda")

    def test_init_hook_layers(self):
        rt = LocalModelRuntime("gpt-neo", hook_layers=["transformer.h.0", "transformer.h.11"])
        self.assertIn("transformer.h.0", rt._hook_layers)
        self.assertIn("transformer.h.11", rt._hook_layers)

    def test_not_loaded_before_load_call(self):
        rt = LocalModelRuntime("Qwen/Qwen3.5-2B")
        self.assertFalse(rt.is_loaded())

    def test_init_does_not_import_transformers(self):
        with patch.dict("sys.modules", {"transformers": None}):
            rt = LocalModelRuntime("some-model")
            self.assertFalse(rt.is_loaded())


class TestLocalModelRuntimeLoad(unittest.TestCase):
    def test_load_with_model_loader(self):
        mock_model = MagicMock()
        rt = LocalModelRuntime("Qwen/Qwen3.5-2B")
        rt.load(model_loader=lambda name: mock_model)
        self.assertTrue(rt.is_loaded())
        self.assertIs(rt._model, mock_model)

    def test_load_calls_loader_with_model_name(self):
        received = []

        def loader(name):
            received.append(name)
            return MagicMock()

        rt = LocalModelRuntime("Qwen/Qwen3.5-2B")
        rt.load(model_loader=loader)
        self.assertEqual(received, ["Qwen/Qwen3.5-2B"])

    def test_is_loaded_true_after_load(self):
        rt = LocalModelRuntime("model-x")
        rt.load(model_loader=lambda _: MagicMock())
        self.assertTrue(rt.is_loaded())


class TestRegisterHookLayers(unittest.TestCase):
    def test_register_single_layer(self):
        rt = LocalModelRuntime("model-x")
        rt.register_hook_layers(["model.layers.0"])
        self.assertIn("model.layers.0", rt._hook_layers)

    def test_register_multiple_layers(self):
        rt = LocalModelRuntime("model-x")
        rt.register_hook_layers(["layer.0", "layer.5", "layer.11"])
        self.assertEqual(len(rt._hook_layers), 3)

    def test_no_duplicates(self):
        rt = LocalModelRuntime("model-x", hook_layers=["layer.0"])
        rt.register_hook_layers(["layer.0", "layer.1"])
        self.assertEqual(rt._hook_layers.count("layer.0"), 1)
        self.assertIn("layer.1", rt._hook_layers)


class TestRunInference(unittest.TestCase):
    def _build_runtime_with_hooks(self, layer_ids=None):
        layer_ids = layer_ids or ["model.layers.0", "model.layers.15"]
        loader, model, layer_mocks = _mock_model_loader(layer_ids)

        rt = LocalModelRuntime("Qwen/Qwen3.5-2B", hook_layers=layer_ids)
        rt.load(model_loader=loader)

        def fake_call(input_ids):
            for lm in layer_mocks.values():
                if lm.register_forward_hook.called:
                    hook_fn = lm.register_forward_hook.call_args[0][0]
                    tensor = MagicMock()
                    tensor.shape = (1, 5, 2048)
                    tensor.mean.return_value = 0.0215
                    tensor.norm.return_value = 31.5
                    tensor.float.return_value = tensor
                    hook_fn(lm, None, tensor)

        model.side_effect = fake_call
        return rt, model, layer_mocks

    def test_run_inference_returns_list(self):
        rt, model, _ = self._build_runtime_with_hooks()
        events = rt.run_inference(MagicMock())
        self.assertIsInstance(events, list)

    def test_run_inference_creates_events(self):
        rt, model, _ = self._build_runtime_with_hooks()
        events = rt.run_inference(MagicMock())
        self.assertGreater(len(events), 0)

    def test_run_inference_event_fields(self):
        rt, model, _ = self._build_runtime_with_hooks()
        events = rt.run_inference(MagicMock())
        if events:
            evt = events[0]
            self.assertIsInstance(evt, ActivationEvent)
            self.assertEqual(evt.model_id, "Qwen/Qwen3.5-2B")
            self.assertIsInstance(evt.mean_activation, float)
            self.assertIsInstance(evt.norm_activation, float)
            self.assertIsInstance(evt.shape, tuple)

    def test_run_inference_raw_tensor_shape_populated(self):
        """raw_tensor_shape must be set by the hook and match the captured tensor shape."""
        rt, model, _ = self._build_runtime_with_hooks()
        events = rt.run_inference(MagicMock())
        if events:
            evt = events[0]
            # The mock tensor has shape (1, 5, 2048) — raw_tensor_shape must
            # reflect that exact shape (the full tensor is not stored).
            self.assertIsNotNone(evt.raw_tensor_shape)
            self.assertIsInstance(evt.raw_tensor_shape, tuple)
            self.assertEqual(evt.raw_tensor_shape, evt.shape)

    def test_run_inference_auto_run_id_is_uuid(self):
        rt, model, _ = self._build_runtime_with_hooks()
        events = rt.run_inference(MagicMock())
        if events:
            try:
                uuid.UUID(events[0].run_id)
            except ValueError:
                self.fail("run_id is not a valid UUID")

    def test_run_inference_explicit_run_id(self):
        rt, model, _ = self._build_runtime_with_hooks()
        custom_id = "aaaabbbb-cccc-dddd-eeee-ffffffffffff"
        events = rt.run_inference(MagicMock(), run_id=custom_id)
        for evt in events:
            self.assertEqual(evt.run_id, custom_id)

    def test_run_inference_same_run_id_all_events(self):
        rt, model, _ = self._build_runtime_with_hooks()
        events = rt.run_inference(MagicMock())
        run_ids = {e.run_id for e in events}
        self.assertLessEqual(len(run_ids), 1)

    def test_run_inference_accumulates_in_get_events(self):
        rt, model, _ = self._build_runtime_with_hooks()
        rt.run_inference(MagicMock())
        rt.run_inference(MagicMock())
        # Two successive run_inference calls; accumulation is tested in
        # TestGetAndClearEvents.  This just verifies no exception is raised.
        self.assertIsInstance(rt.get_events(), list)

    def test_run_inference_without_load_raises(self):
        rt = LocalModelRuntime("no-load-model")
        with self.assertRaises(RuntimeError):
            rt.run_inference(MagicMock())


class TestGetAndClearEvents(unittest.TestCase):
    def test_get_events_empty_initially(self):
        rt = LocalModelRuntime("m")
        self.assertEqual(rt.get_events(), [])

    def test_clear_events_resets(self):
        rt = LocalModelRuntime("m")
        rt._events.append(_make_activation_event())
        rt._events.append(_make_activation_event())
        rt.clear_events()
        self.assertEqual(rt.get_events(), [])

    def test_get_events_returns_copy(self):
        rt = LocalModelRuntime("m")
        rt._events.append(_make_activation_event())
        first_call = rt.get_events()
        rt._events.append(_make_activation_event())
        second_call = rt.get_events()
        self.assertEqual(len(first_call), 1)
        self.assertEqual(len(second_call), 2)


class TestSaveLoadEvents(unittest.TestCase):
    def _temp_path(self):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, dir=".", prefix="test_events_"
        )
        tmp.close()
        return tmp.name

    def tearDown(self):
        for f in Path(".").glob("test_events_*.jsonl"):
            try:
                f.unlink()
            except OSError:
                pass

    def test_save_events_creates_file(self):
        rt = LocalModelRuntime("m")
        rt._events.append(_make_activation_event())
        path = self._temp_path()
        rt.save_events(path)
        self.assertTrue(Path(path).exists())

    def test_save_and_load_roundtrip(self):
        rt = LocalModelRuntime("m")
        run_id = str(uuid.uuid4())
        evt1 = _make_activation_event(
            model_id="Qwen/Qwen3.5-2B",
            layer_id="model.layers.10",
            token_count=7,
            shape=(1, 7, 2048),
            mean_activation=0.01234,
            norm_activation=15.678,
            run_id=run_id,
        )
        evt2 = _make_activation_event(
            model_id="Qwen/Qwen3.5-2B",
            layer_id="model.layers.20",
            token_count=7,
            shape=(1, 7, 2048),
            mean_activation=-0.00512,
            norm_activation=14.22,
            run_id=run_id,
        )
        rt._events.extend([evt1, evt2])
        path = self._temp_path()
        rt.save_events(path)
        loaded = LocalModelRuntime.load_events(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].layer_id, "model.layers.10")
        self.assertEqual(loaded[1].layer_id, "model.layers.20")
        self.assertAlmostEqual(loaded[0].mean_activation, 0.01234, places=5)

    def test_load_events_preserves_shape_tuple(self):
        rt = LocalModelRuntime("m")
        rt._events.append(_make_activation_event(shape=(2, 16, 4096)))
        path = self._temp_path()
        rt.save_events(path)
        loaded = LocalModelRuntime.load_events(path)
        self.assertIsInstance(loaded[0].shape, tuple)
        self.assertEqual(loaded[0].shape, (2, 16, 4096))

    def test_load_events_preserves_run_id(self):
        rt = LocalModelRuntime("m")
        rid = str(uuid.uuid4())
        rt._events.append(_make_activation_event(run_id=rid))
        path = self._temp_path()
        rt.save_events(path)
        loaded = LocalModelRuntime.load_events(path)
        self.assertEqual(loaded[0].run_id, rid)

    def test_empty_save_load(self):
        rt = LocalModelRuntime("m")
        path = self._temp_path()
        rt.save_events(path)
        loaded = LocalModelRuntime.load_events(path)
        self.assertEqual(loaded, [])


class TestRunInferenceTryBranchDiscrimination(unittest.TestCase):
    """Verify the try-branch (torch.linalg.norm) is exercised, not the except fallback.

    Previous tests used a real torch.Tensor and asserted ``norm_activation > 0``
    or ``norm_activation ≈ 3.74``.  Both branches produce the same value for a
    real tensor so those tests pass even when the except-branch runs — they are
    non-discriminating.

    This suite uses a **sentinel value approach**: ``torch.linalg.norm`` is patched
    to return an object whose ``.item()`` returns a known sentinel (42.0).  If the
    try-branch runs, ``norm_activation`` will equal 42.0.  If the except-branch
    runs instead (calling ``tensor.norm()``), it returns the real Frobenius norm
    (~3.74) and the sentinel assertion fails — proving branch discrimination.

    WHY THIS WORKS
    ---------------
    Inside ``run_inference._make_hook``, torch is imported locally:
    ``import torch``.  Python resolves that to ``sys.modules['torch']``, which is
    the same object patched by ``unittest.mock.patch('torch.linalg.norm')``.
    Patching the attribute on the live module object is therefore visible to the
    hook regardless of where it imports torch from.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # type: ignore[import]
            import torch.nn as nn  # type: ignore[import]

            cls.torch = torch
            cls.nn = nn
            cls.torch_available = True
        except ImportError:
            cls.torch_available = False

    def _skip_if_no_torch(self):
        if not self.torch_available:
            self.skipTest("torch not installed — sentinel discrimination test skipped")

    def _build_tiny_runtime(self):
        """Return a LocalModelRuntime with a tiny identity nn.Module on layer 'fc'."""
        torch = self.torch
        nn = self.nn

        class TinyModel(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.fc = nn.Linear(3, 3, bias=False)
                with torch.no_grad():
                    self_inner.fc.weight.copy_(torch.eye(3))

            def forward(self_inner, x):
                return (self_inner.fc(x),)

        rt = LocalModelRuntime("test-sentinel", hook_layers=["fc"])
        rt.load(model_loader=lambda _: TinyModel())
        return rt

    def test_try_branch_uses_torch_linalg_norm_not_fallback(self):
        """Patch torch.linalg.norm to a sentinel; assert norm_activation == sentinel.

        If the try-branch runs: ``norm_val = float(torch.linalg.norm(t_float).item())``
        resolves to ``float(sentinel_obj.item())`` == 42.0.

        If the except-branch runs instead: ``norm_val = float(tensor.norm())``
        which returns the real Frobenius norm (~3.74) — NOT 42.0.  The assertion
        therefore fails, proving the try-branch is the one that actually executed.
        """
        self._skip_if_no_torch()
        import torch  # type: ignore[import]

        SENTINEL = 42.0

        # Build a minimal return object whose .item() yields the sentinel.
        class _SentinelResult:
            def item(self):
                return SENTINEL

        rt = self._build_tiny_runtime()

        with patch("torch.linalg.norm", return_value=_SentinelResult()):
            events = rt.run_inference(torch.tensor([[1.0, 2.0, 3.0]]))

        self.assertTrue(events, "No events captured — hook did not fire")
        evt = events[-1]
        self.assertAlmostEqual(
            evt.norm_activation,
            SENTINEL,
            places=5,
            msg=(
                f"norm_activation should equal sentinel {SENTINEL} "
                f"(got {evt.norm_activation!r}) — if this fails, "
                "torch.linalg.norm was NOT called in the try-branch; "
                "the except fallback ran instead."
            ),
        )

    def test_try_branch_mean_uses_float_tensor_mean(self):
        """mean_activation equals the true tensor mean — proves t_float.mean() ran.

        The try-branch computes ``float(t_float.mean().item())``.  With identity
        weights and input [[1, 2, 3]], output is also [[1, 2, 3]] so mean = 2.0.
        The except-branch calls ``float(tensor.mean())``, which on a real tensor
        also equals 2.0 — so this is *not* a discriminating assertion by itself.
        It is kept here as a sanity check alongside the sentinel norm test.
        """
        self._skip_if_no_torch()
        import torch  # type: ignore[import]

        rt = self._build_tiny_runtime()
        events = rt.run_inference(torch.tensor([[1.0, 2.0, 3.0]]))
        self.assertEqual(len(events), 1, "Expected exactly one event from the fc hook")
        evt = events[0]
        self.assertIsInstance(evt.mean_activation, float,
                              "mean_activation must be a real Python float")
        # identity fc: output == input [[1, 2, 3]] → mean = 2.0
        self.assertAlmostEqual(evt.mean_activation, 2.0, places=4,
                               msg="mean_activation must equal (1+2+3)/3 = 2.0")

    def test_raw_tensor_shape_set_by_try_branch(self):
        """raw_tensor_shape must equal the actual tensor shape from the fc hook."""
        self._skip_if_no_torch()
        import torch  # type: ignore[import]

        rt = self._build_tiny_runtime()
        events = rt.run_inference(torch.tensor([[1.0, 2.0, 3.0]]))
        self.assertEqual(len(events), 1)
        evt = events[0]
        self.assertIsNotNone(evt.raw_tensor_shape)
        self.assertIsInstance(evt.raw_tensor_shape, tuple)
        # Input (1, 3) through Linear(3→3, bias=False) → output (1, 3)
        self.assertEqual(evt.raw_tensor_shape, (1, 3),
                         "raw_tensor_shape must equal the captured tensor shape (1, 3)")
        self.assertEqual(evt.shape, evt.raw_tensor_shape)


@unittest.skipUnless(
    os.getenv("CHELATED_INTEGRATION_MODEL"),
    "Integration test requires a real model download.  "
    "Set CHELATED_INTEGRATION_MODEL=sshleifer/tiny-gpt2 (or any causal-LM on HF Hub) "
    "to run.  Skipped in CI because weight download is too large for hosted runners.",
)
class TestLocalModelRuntimeIntegration(unittest.TestCase):
    """Exercise the real ``AutoModelForCausalLM.from_pretrained`` code path.

    Uses the model named by ``CHELATED_INTEGRATION_MODEL``.  The recommended
    value is ``sshleifer/tiny-gpt2`` (<5 MB), which exercises the same
    ``from_pretrained`` → hook registration → ``run_inference`` path that the
    Qwen3.5-9B pilot load uses.  This test would fail if that path were broken,
    even though CI mocks it out.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_name = os.environ["CHELATED_INTEGRATION_MODEL"]

    def test_real_load_and_is_loaded(self):
        """load() without a model_loader must actually call from_pretrained."""
        rt = LocalModelRuntime(self.model_name, device="cpu")
        # No model_loader injected — must hit the real transformers branch.
        rt.load()
        self.assertTrue(rt.is_loaded())

    def test_real_hook_capture_shape(self):
        """Hook must capture a real tensor; raw_tensor_shape must be non-empty."""
        try:
            import torch  # type: ignore[import]
        except ImportError:
            self.skipTest("torch not available")

        # Discover first transformer block layer name so we can hook it.
        rt = LocalModelRuntime(self.model_name, device="cpu")
        rt.load()
        model = rt._model
        # Walk to first named child to find a hookable module.
        first_layer_id = None
        for name, _module in model.named_modules():
            if name:  # skip root
                first_layer_id = name
                break

        self.assertIsNotNone(first_layer_id, "Model has no named submodules")
        rt.register_hook_layers([first_layer_id])

        # Build a tiny input tensor (1 token).
        input_ids = torch.tensor([[0]])
        events = rt.run_inference(input_ids)

        self.assertGreater(len(events), 0, "No events captured; hook registration failed")
        evt = events[0]
        self.assertIsInstance(evt.raw_tensor_shape, tuple)
        self.assertGreater(len(evt.raw_tensor_shape), 0)
        # shape and raw_tensor_shape must agree.
        self.assertEqual(evt.shape, evt.raw_tensor_shape)


if __name__ == "__main__":
    unittest.main()
