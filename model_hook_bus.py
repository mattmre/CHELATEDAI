"""Event bus for model activation hook events."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from model_scope_runtime import ActivationEvent, LocalModelRuntime


@dataclass
class HookBusEvent:
    """Wraps an ActivationEvent with routing metadata for bus consumers."""

    event_id: str
    activation: ActivationEvent
    metadata: dict = field(default_factory=dict)


class ModelHookBus:
    """Pub/sub event bus for activation hook events."""

    def __init__(self) -> None:
        self._subscribers: dict = {}
        self._log: list = []

    def subscribe(self, callback: Callable) -> str:
        """Register a callback; returns a subscriber_id for later removal."""
        sub_id = str(uuid.uuid4())
        self._subscribers[sub_id] = callback
        return sub_id

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Remove a subscriber. Returns True if it existed, False otherwise."""
        if subscriber_id in self._subscribers:
            del self._subscribers[subscriber_id]
            return True
        return False

    def emit(
        self,
        activation: ActivationEvent,
        *,
        metadata: dict = None,
    ) -> HookBusEvent:
        """Wrap activation into a HookBusEvent, dispatch to all subscribers, log it."""
        bus_event = HookBusEvent(
            event_id=str(uuid.uuid4()),
            activation=activation,
            metadata=metadata or {},
        )
        self._log.append(bus_event)
        for callback in list(self._subscribers.values()):
            callback(bus_event)
        return bus_event

    def get_log(self) -> list:
        """Return all emitted HookBusEvents."""
        return list(self._log)

    def clear_log(self) -> None:
        """Reset the event log."""
        self._log = []

    def subscriber_count(self) -> int:
        """Return the number of active subscribers."""
        return len(self._subscribers)


def attach_runtime_to_bus(
    runtime: LocalModelRuntime,
    bus: ModelHookBus,
) -> None:
    """Wrap runtime.run_inference so events are also emitted to the bus.

    Normal get_events() behaviour is preserved; the bus receives a copy of each
    ActivationEvent produced by every subsequent run_inference call.
    """
    original_run_inference = runtime.run_inference

    def _patched_run_inference(
        input_ids: Any,
        *,
        run_id: str = None,
    ) -> list:
        events = original_run_inference(input_ids, run_id=run_id)
        for event in events:
            bus.emit(event)
        return events

    runtime.run_inference = _patched_run_inference
