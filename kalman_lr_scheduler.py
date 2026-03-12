"""Kalman-gain-inspired adaptive learning rate for sedimentation.

Inspired by GAM-RAG (March 2026) Kalman-gain uncertainty-aware updates.
Tracks correction uncertainty and scales LR accordingly:
- High uncertainty (large variance in recent losses) -> lower LR
- Low uncertainty (stable losses) -> higher LR (corrections are reliable)
"""

from __future__ import annotations

import numpy as np
from chelation_logger import get_logger


class KalmanLRScheduler:
    """Adaptive learning rate based on correction uncertainty.

    Uses a simplified Kalman-gain analogy:
    - Process noise Q: represents expected variation in corrections
    - Measurement noise R: estimated from recent loss variance
    - Kalman gain K = Q / (Q + R): determines how much to trust new corrections
    - Effective LR = base_lr * K

    When R is high (noisy/uncertain), K is low -> conservative LR.
    When R is low (stable/confident), K is high -> aggressive LR.

    Args:
        base_lr: Base learning rate (default: 0.01)
        process_noise: Expected process noise Q (default: 0.1)
        min_lr_ratio: Minimum LR as fraction of base_lr (default: 0.1)
        max_lr_ratio: Maximum LR as fraction of base_lr (default: 2.0)
        window_size: Rolling window for variance estimation (default: 10)
    """

    def __init__(self, base_lr=0.01, process_noise=0.1,
                 min_lr_ratio=0.1, max_lr_ratio=2.0, window_size=10):
        if base_lr <= 0:
            raise ValueError("base_lr must be positive, got {}".format(base_lr))
        if process_noise <= 0:
            raise ValueError("process_noise must be positive, got {}".format(process_noise))
        if min_lr_ratio <= 0:
            raise ValueError("min_lr_ratio must be positive, got {}".format(min_lr_ratio))
        if max_lr_ratio < min_lr_ratio:
            raise ValueError(
                "max_lr_ratio ({}) must be >= min_lr_ratio ({})".format(
                    max_lr_ratio, min_lr_ratio
                )
            )
        if window_size < 2:
            raise ValueError("window_size must be >= 2, got {}".format(window_size))

        self.base_lr = base_lr
        self.process_noise = process_noise
        self.min_lr = base_lr * min_lr_ratio
        self.max_lr = base_lr * max_lr_ratio
        self.window_size = window_size
        self.logger = get_logger()

        self._loss_history = []
        self._current_lr = base_lr
        self._kalman_gain = 1.0
        self._step_count = 0

    def step(self, loss):
        """Record a loss value and update the learning rate.

        Args:
            loss: Current epoch loss value

        Returns:
            float: Updated learning rate
        """
        self._step_count += 1
        self._loss_history.append(float(loss))

        # Keep window bounded
        if len(self._loss_history) > self.window_size:
            self._loss_history = self._loss_history[-self.window_size:]

        # Need at least 2 samples for variance
        if len(self._loss_history) < 2:
            return self._current_lr

        # Estimate measurement noise R from loss variance
        R = float(np.var(self._loss_history))
        Q = self.process_noise

        # Kalman gain: K = Q / (Q + R)
        self._kalman_gain = Q / (Q + R + 1e-10)

        # Effective LR scales with Kalman gain
        self._current_lr = self.base_lr * self._kalman_gain

        # Clamp to bounds
        self._current_lr = max(self.min_lr, min(self.max_lr, self._current_lr))

        self.logger.log_event(
            "kalman_lr_update",
            "LR updated: {:.6f} (gain={:.4f}, R={:.6f})".format(
                self._current_lr, self._kalman_gain, R
            ),
            current_lr=self._current_lr,
            kalman_gain=self._kalman_gain,
            measurement_noise=R,
            step=self._step_count,
        )

        return self._current_lr

    @property
    def current_lr(self):
        """Current effective learning rate."""
        return self._current_lr

    @property
    def kalman_gain(self):
        """Current Kalman gain value (0..1)."""
        return self._kalman_gain

    def get_state(self):
        """Return a snapshot of the scheduler state.

        Returns:
            dict with current_lr, kalman_gain, base_lr, process_noise,
            step_count, loss_variance, and window_size.
        """
        return {
            "current_lr": self._current_lr,
            "kalman_gain": self._kalman_gain,
            "base_lr": self.base_lr,
            "process_noise": self.process_noise,
            "step_count": self._step_count,
            "loss_variance": float(np.var(self._loss_history)) if len(self._loss_history) >= 2 else 0.0,
            "window_size": self.window_size,
        }

    def reset(self):
        """Reset scheduler state for a new training run."""
        self._loss_history = []
        self._current_lr = self.base_lr
        self._kalman_gain = 1.0
        self._step_count = 0
