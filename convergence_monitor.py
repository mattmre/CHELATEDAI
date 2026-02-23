"""
Convergence Monitor for ChelatedAI Training Loops

Provides patience-based early stopping for sedimentation and distillation
training, inspired by standard deep learning practices.

Reference: Standard early stopping (Prechelt, 1998)
"""

import numpy as np
from chelation_logger import get_logger


class ConvergenceMonitor:
    """
    Monitors training loss to detect convergence and trigger early stopping.

    Args:
        patience: Number of epochs without improvement before stopping (default: 5)
        rel_threshold: Minimum relative improvement to count as progress (default: 0.001)
        min_epochs: Minimum epochs before early stopping can trigger (default: 3)
    """

    def __init__(self, patience=5, rel_threshold=0.001, min_epochs=3):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if rel_threshold < 0:
            raise ValueError(f"rel_threshold must be >= 0, got {rel_threshold}")
        if min_epochs < 1:
            raise ValueError(f"min_epochs must be >= 1, got {min_epochs}")

        self.patience = patience
        self.rel_threshold = rel_threshold
        self.min_epochs = min_epochs
        self.logger = get_logger()

        # State
        self._loss_history = []
        self._best_loss = float('inf')
        self._epochs_without_improvement = 0
        self._converged = False

    def record_loss(self, loss):
        """
        Record a training loss value and check for convergence.

        Args:
            loss: Current epoch's loss value (float)

        Returns:
            bool: True if training should stop (converged), False to continue
        """
        # Validate loss is finite
        if not np.isfinite(loss):
            self.logger.log_event(
                "convergence_monitor",
                f"Non-finite loss detected: {loss}"
            )
            return False  # Don't stop on NaN/inf, let training continue

        self._loss_history.append(float(loss))
        epoch = len(self._loss_history)

        # First epoch always sets baseline
        if epoch == 1:
            self._best_loss = loss
            self._epochs_without_improvement = 0
        else:
            # Check if we've made sufficient relative improvement
            if loss < self._best_loss:
                rel_improvement = (self._best_loss - loss) / (abs(self._best_loss) + 1e-12)

                if rel_improvement >= self.rel_threshold:
                    self._best_loss = loss
                    self._epochs_without_improvement = 0
                else:
                    self._epochs_without_improvement += 1
            else:
                self._epochs_without_improvement += 1

        # Check convergence criteria
        if epoch >= self.min_epochs and self._epochs_without_improvement >= self.patience:
            self._converged = True
            self.logger.log_event(
                "convergence_detected",
                f"Training converged at epoch {epoch}: "
                f"no improvement for {self.patience} epochs",
                epoch=epoch,
                best_loss=self._best_loss,
                patience=self.patience
            )
            return True

        return False

    @property
    def converged(self):
        """Whether convergence has been detected."""
        return self._converged

    @property
    def loss_history(self):
        """Copy of recorded loss history."""
        return list(self._loss_history)

    @property
    def best_loss(self):
        """Best loss observed so far."""
        return self._best_loss

    @property
    def epochs_without_improvement(self):
        """Number of consecutive epochs without sufficient improvement."""
        return self._epochs_without_improvement

    @property
    def total_epochs(self):
        """Total number of epochs recorded."""
        return len(self._loss_history)

    def reset(self):
        """Reset monitor state for a new training run."""
        self._loss_history = []
        self._best_loss = float('inf')
        self._epochs_without_improvement = 0
        self._converged = False

    def get_summary(self):
        """
        Get a summary dict of the convergence state.

        Returns:
            dict with keys: converged, total_epochs, best_loss,
                           epochs_without_improvement, loss_history
        """
        return {
            "converged": self._converged,
            "total_epochs": self.total_epochs,
            "best_loss": self._best_loss if self._loss_history else None,
            "epochs_without_improvement": self._epochs_without_improvement,
            "patience": self.patience,
            "rel_threshold": self.rel_threshold,
            "min_epochs": self.min_epochs,
            "loss_history": list(self._loss_history)
        }
