"""Temperature controller for detection-triggered correction cycles."""

from __future__ import annotations


class AnnealingController:
    """Map drift signals to correction-cycle temperature and knob settings."""

    def __init__(
        self,
        initial_temperature: float = 0.0,
        cooling_rate: float = 0.7,
        trigger_threshold: float = 0.15,
        max_temperature: float = 1.0,
    ):
        if initial_temperature < 0.0:
            raise ValueError("initial_temperature must be >= 0")
        if not 0.0 < cooling_rate <= 1.0:
            raise ValueError("cooling_rate must be in (0, 1]")
        if trigger_threshold < 0.0:
            raise ValueError("trigger_threshold must be >= 0")
        if max_temperature <= 0.0:
            raise ValueError("max_temperature must be > 0")
        self.temperature = min(float(initial_temperature), float(max_temperature))
        self.cooling_rate = float(cooling_rate)
        self.trigger_threshold = float(trigger_threshold)
        self.max_temperature = float(max_temperature)
        self.epsilon = 1e-6

    def observe_drift(self, drift_magnitude: float) -> None:
        """Raise temperature proportionally when drift exceeds the trigger."""

        magnitude = float(drift_magnitude)
        if magnitude < 0.0:
            raise ValueError("drift_magnitude must be >= 0")
        if magnitude < self.trigger_threshold:
            return
        self.temperature = max(self.temperature, min(self.max_temperature, magnitude))

    def should_correct(self) -> bool:
        return self.temperature > self.epsilon

    def cycle_settings(self) -> dict:
        ratio = min(max(self.temperature / self.max_temperature, 0.0), 1.0)
        return {
            "learning_rate_scale": 0.1 + (0.9 * ratio),
            "epochs": 1 + int(round(2.0 * ratio)),
            "online_intensity": ratio,
        }

    def end_cycle(self) -> None:
        self.temperature *= self.cooling_rate
        if self.temperature <= self.epsilon:
            self.temperature = 0.0
