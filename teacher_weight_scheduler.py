"""Dynamic teacher weight scheduling for distillation training."""

import math
from chelation_logger import get_logger


class TeacherWeightScheduler:
    """Manages dynamic teacher weight during training.

    Supports 5 schedule types:
    - constant: Fixed weight throughout
    - linear_decay: Linear decrease from initial to final weight
    - cosine_annealing: Cosine decay with optional warm restarts
    - step_decay: Multiply by gamma every step_size steps
    - adaptive: Adjust based on loss trajectory
    """

    def __init__(self, schedule="constant", initial_weight=0.5,
                 total_steps=100, final_weight=0.1, gamma=0.5,
                 step_size=10, warmup_steps=0, min_weight=0.01,
                 patience=5, increase_factor=1.1, decrease_factor=0.9,
                 logger=None):
        self.schedule = schedule
        self.initial_weight = initial_weight
        self.current_weight = initial_weight
        self.total_steps = total_steps
        self.final_weight = final_weight
        self.gamma = gamma
        self.step_size = step_size
        self.warmup_steps = warmup_steps
        self.min_weight = min_weight
        self.patience = patience
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.logger = logger or get_logger()

        self._step_count = 0
        self._loss_history = []
        self._best_loss = float('inf')
        self._steps_without_improvement = 0

    def step(self, loss=None):
        """Advance one step and return current weight."""
        self._step_count += 1

        if self.warmup_steps > 0 and self._step_count <= self.warmup_steps:
            self.current_weight = (
                self.initial_weight * (self._step_count / self.warmup_steps)
            )
            return self.current_weight

        effective_step = self._step_count - self.warmup_steps
        effective_total = max(self.total_steps - self.warmup_steps, 1)

        if self.schedule == "constant":
            self.current_weight = self.initial_weight

        elif self.schedule == "linear_decay":
            progress = min(effective_step / effective_total, 1.0)
            self.current_weight = (
                self.initial_weight
                + (self.final_weight - self.initial_weight) * progress
            )

        elif self.schedule == "cosine_annealing":
            progress = min(effective_step / effective_total, 1.0)
            self.current_weight = (
                self.final_weight
                + 0.5 * (self.initial_weight - self.final_weight)
                * (1 + math.cos(math.pi * progress))
            )

        elif self.schedule == "step_decay":
            num_decays = effective_step // max(self.step_size, 1)
            self.current_weight = (
                self.initial_weight * (self.gamma ** num_decays)
            )

        elif self.schedule == "adaptive":
            if loss is not None:
                self._loss_history.append(loss)
                if loss < self._best_loss:
                    self._best_loss = loss
                    self._steps_without_improvement = 0
                    self.current_weight *= self.decrease_factor
                else:
                    self._steps_without_improvement += 1
                    if self._steps_without_improvement >= self.patience:
                        self.current_weight *= self.increase_factor
                        self._steps_without_improvement = 0

        self.current_weight = max(self.current_weight, self.min_weight)
        return self.current_weight

    def reset(self):
        """Reset scheduler state."""
        self.current_weight = self.initial_weight
        self._step_count = 0
        self._loss_history = []
        self._best_loss = float('inf')
        self._steps_without_improvement = 0

    def get_summary(self):
        """Return scheduler state summary."""
        return {
            "schedule": self.schedule,
            "current_weight": self.current_weight,
            "initial_weight": self.initial_weight,
            "step_count": self._step_count,
            "total_steps": self.total_steps,
        }


def create_weight_scheduler(schedule="constant", initial_weight=0.5, **kwargs):
    """Factory function for creating weight schedulers."""
    return TeacherWeightScheduler(
        schedule=schedule, initial_weight=initial_weight, **kwargs,
    )
