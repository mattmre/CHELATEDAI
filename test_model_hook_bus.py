"""Tests for model_hook_bus.py — ModelHookBus, HookBusEvent, attach_runtime_to_bus."""

from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

from model_scope_runtime import ActivationEvent, LocalModelRuntime
from model_hook_bus import HookBusEvent, ModelHookBus, attach_runtime_to_bus


def _make_activation_event(**kwargs) -> ActivationEvent:
    defaults = {
        "schema_version": "1.0",
        "model_id": "Qwen/Qwen3.5-2B",
        "layer_id": "model.layers.15",
        "token_count": 10,
        "shape": (1, 10, 2048),
        "mean_activation": 0.02841,
        "norm_activation": 33.107,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(uuid.uuid4()),
    }
    defaults.update(kwargs)
    return ActivationEvent(**defaults)


class TestHookBusEventConstruction(unittest.TestCase):
    def test_construction(self):
        evt_id = str(uuid.uuid4())
        activation = _make_activation_event()
        bus_evt = HookBusEvent(
            event_id=evt_id,
            activation=activation,
            metadata={"source": "test", "layer_count": 3},
        )
        self.assertEqual(bus_evt.event_id, evt_id)
        self.assertIs(bus_evt.activation, activation)
        self.assertEqual(bus_evt.metadata["source"], "test")

    def test_default_metadata_is_dict(self):
        bus_evt = HookBusEvent(
            event_id=str(uuid.uuid4()),
            activation=_make_activation_event(),
        )
        self.assertIsInstance(bus_evt.metadata, dict)


class TestModelHookBusSubscribe(unittest.TestCase):
    def test_subscribe_returns_string_id(self):
        bus = ModelHookBus()
        sub_id = bus.subscribe(lambda e: None)
        self.assertIsInstance(sub_id, str)

    def test_subscribe_increments_count(self):
        bus = ModelHookBus()
        bus.subscribe(lambda e: None)
        bus.subscribe(lambda e: None)
        self.assertEqual(bus.subscriber_count(), 2)

    def test_unsubscribe_removes_subscriber(self):
        bus = ModelHookBus()
        sub_id = bus.subscribe(lambda e: None)
        result = bus.unsubscribe(sub_id)
        self.assertTrue(result)
        self.assertEqual(bus.subscriber_count(), 0)

    def test_unsubscribe_unknown_id_returns_false(self):
        bus = ModelHookBus()
        result = bus.unsubscribe("nonexistent-id")
        self.assertFalse(result)

    def test_subscriber_count_zero_initially(self):
        bus = ModelHookBus()
        self.assertEqual(bus.subscriber_count(), 0)


class TestModelHookBusEmit(unittest.TestCase):
    def test_emit_calls_subscriber(self):
        bus = ModelHookBus()
        received = []
        bus.subscribe(received.append)
        activation = _make_activation_event()
        bus.emit(activation)
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], HookBusEvent)

    def test_emit_subscriber_receives_correct_activation(self):
        bus = ModelHookBus()
        received = []
        bus.subscribe(received.append)
        activation = _make_activation_event(layer_id="model.layers.7")
        bus.emit(activation)
        self.assertEqual(received[0].activation.layer_id, "model.layers.7")

    def test_emit_with_no_subscribers_still_logs(self):
        bus = ModelHookBus()
        activation = _make_activation_event()
        bus.emit(activation)
        self.assertEqual(len(bus.get_log()), 1)

    def test_emit_returns_hook_bus_event(self):
        bus = ModelHookBus()
        activation = _make_activation_event()
        result = bus.emit(activation)
        self.assertIsInstance(result, HookBusEvent)

    def test_emit_event_id_is_uuid(self):
        bus = ModelHookBus()
        activation = _make_activation_event()
        result = bus.emit(activation)
        try:
            uuid.UUID(result.event_id)
        except ValueError:
            self.fail("event_id is not a valid UUID")

    def test_emit_with_metadata(self):
        bus = ModelHookBus()
        activation = _make_activation_event()
        result = bus.emit(activation, metadata={"run": "experiment-42"})
        self.assertEqual(result.metadata["run"], "experiment-42")

    def test_multiple_subscribers_all_called(self):
        bus = ModelHookBus()
        r1, r2, r3 = [], [], []
        bus.subscribe(r1.append)
        bus.subscribe(r2.append)
        bus.subscribe(r3.append)
        bus.emit(_make_activation_event())
        self.assertEqual(len(r1), 1)
        self.assertEqual(len(r2), 1)
        self.assertEqual(len(r3), 1)

    def test_multiple_subscribers_receive_same_event(self):
        bus = ModelHookBus()
        r1, r2 = [], []
        bus.subscribe(r1.append)
        bus.subscribe(r2.append)
        bus.emit(_make_activation_event())
        self.assertEqual(r1[0].event_id, r2[0].event_id)


class TestModelHookBusLog(unittest.TestCase):
    def test_get_log_empty_initially(self):
        bus = ModelHookBus()
        self.assertEqual(bus.get_log(), [])

    def test_get_log_accumulates(self):
        bus = ModelHookBus()
        bus.emit(_make_activation_event())
        bus.emit(_make_activation_event())
        self.assertEqual(len(bus.get_log()), 2)

    def test_clear_log_resets(self):
        bus = ModelHookBus()
        bus.emit(_make_activation_event())
        bus.clear_log()
        self.assertEqual(bus.get_log(), [])

    def test_get_log_returns_copy(self):
        bus = ModelHookBus()
        bus.emit(_make_activation_event())
        log1 = bus.get_log()
        bus.emit(_make_activation_event())
        log2 = bus.get_log()
        self.assertEqual(len(log1), 1)
        self.assertEqual(len(log2), 2)

    def test_unsubscribed_does_not_receive_future_events(self):
        bus = ModelHookBus()
        received = []
        sub_id = bus.subscribe(received.append)
        bus.emit(_make_activation_event())
        bus.unsubscribe(sub_id)
        bus.emit(_make_activation_event())
        self.assertEqual(len(received), 1)


class TestAttachRuntimeToBus(unittest.TestCase):
    def _make_runtime_with_events(self, events):
        rt = LocalModelRuntime("Qwen/Qwen3.5-2B")
        rt._model = MagicMock()

        def fake_run_inference(input_ids, *, run_id=None):
            rt._events.extend(events)
            return list(events)

        rt.run_inference = fake_run_inference
        return rt

    def test_attach_forwards_events_to_bus(self):
        activation = _make_activation_event()
        rt = self._make_runtime_with_events([activation])
        bus = ModelHookBus()
        attach_runtime_to_bus(rt, bus)
        rt.run_inference(MagicMock())
        self.assertEqual(len(bus.get_log()), 1)

    def test_attach_does_not_break_get_events(self):
        activation = _make_activation_event()
        rt = self._make_runtime_with_events([activation])
        bus = ModelHookBus()
        attach_runtime_to_bus(rt, bus)
        rt.run_inference(MagicMock())
        self.assertEqual(len(rt.get_events()), 1)

    def test_attach_multiple_events_all_emitted(self):
        activations = [_make_activation_event(layer_id=f"model.layers.{i}") for i in range(4)]
        rt = self._make_runtime_with_events(activations)
        bus = ModelHookBus()
        attach_runtime_to_bus(rt, bus)
        rt.run_inference(MagicMock())
        self.assertEqual(len(bus.get_log()), 4)

    def test_attach_bus_subscriber_receives_events(self):
        activation = _make_activation_event(layer_id="model.layers.5")
        rt = self._make_runtime_with_events([activation])
        bus = ModelHookBus()
        received = []
        bus.subscribe(received.append)
        attach_runtime_to_bus(rt, bus)
        rt.run_inference(MagicMock())
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].activation.layer_id, "model.layers.5")

    def test_attach_multiple_runs_accumulate_in_bus(self):
        rt = LocalModelRuntime("Qwen/Qwen3.5-2B")
        rt._model = MagicMock()
        call_count = [0]

        def fake_run_inference(input_ids, *, run_id=None):
            call_count[0] += 1
            evt = _make_activation_event(layer_id=f"model.layers.{call_count[0]}")
            rt._events.append(evt)
            return [evt]

        rt.run_inference = fake_run_inference
        bus = ModelHookBus()
        attach_runtime_to_bus(rt, bus)
        rt.run_inference(MagicMock())
        rt.run_inference(MagicMock())
        self.assertEqual(len(bus.get_log()), 2)


if __name__ == "__main__":
    unittest.main()
