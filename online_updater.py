"""
Online Gradient Updater for ChelatedAI Inference-Time Adaptation

Provides lightweight per-query adapter updates at inference time using
contrastive learning from retrieval results.

Inspired by:
- Online-Optimized RAG (arXiv:2509.20415)
- TTARAG (arXiv:2601.11443)

Extended with pluggable loss functions (Session 22):
- TripletMarginOnlineLoss: Original triplet margin behavior
- InfoNCEOnlineLoss: NT-Xent contrastive loss with temperature
- CosineSimilarityOnlineLoss: Direct similarity optimization
- AdaptiveMargin: Dynamic margin based on retrieval quality
- OnlineLossScheduler: Composable loss weight scheduling
- OnlineUpdateDiagnostics: Per-dimension gradient stats + stability bridge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from chelation_logger import get_logger


# ============================================================
# Abstract Loss Function Base
# ============================================================

class OnlineLossFunction(ABC):
    """
    Abstract base class for online correction loss functions.

    All loss functions take adapted query, positive, and negative tensors
    and return a scalar loss. Subclasses must implement compute() and
    get_state().
    """

    @abstractmethod
    def compute(self, adapted_query, adapted_pos, adapted_neg):
        """
        Compute loss from adapted embeddings.

        Args:
            adapted_query: Adapted query tensor (1, dim)
            adapted_pos: Adapted positive tensor(s) (N, dim) or (1, dim)
            adapted_neg: Adapted negative tensor(s) (M, dim) or (1, dim)

        Returns:
            torch.Tensor: Scalar loss value
        """
        pass

    @abstractmethod
    def get_state(self):
        """
        Return dictionary of loss function state for diagnostics.

        Returns:
            dict: Loss function configuration and running state
        """
        pass


# ============================================================
# Triplet Margin Loss (refactored from original inline code)
# ============================================================

class TripletMarginOnlineLoss(OnlineLossFunction):
    """
    Triplet margin loss for online correction.

    Pulls query toward positive examples and pushes away from negatives.
    This is the original loss function refactored into the pluggable interface.

    Args:
        margin: Triplet margin distance (default: 0.1)
        aggregation: How to aggregate multiple pos/neg vectors.
            "mean" averages pos and neg before loss (original behavior).
            "per_vector" computes loss per positive-negative pair and averages.
    """

    def __init__(self, margin=0.1, aggregation="mean"):
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if aggregation not in ("mean", "per_vector"):
            raise ValueError("aggregation must be 'mean' or 'per_vector'")
        self.margin = margin
        self.aggregation = aggregation
        self._triplet_loss = nn.TripletMarginLoss(margin=self.margin)
        self._call_count = 0

    def compute(self, adapted_query, adapted_pos, adapted_neg):
        """Compute triplet margin loss."""
        self._call_count += 1

        if self.aggregation == "mean":
            # Original behavior: mean of pos, mean of neg
            pos_mean = adapted_pos.mean(dim=0, keepdim=True)
            neg_mean = adapted_neg.mean(dim=0, keepdim=True)
            return self._triplet_loss(adapted_query, pos_mean, neg_mean)
        else:
            # Per-vector: compute loss for each pos-neg pair
            n_pos = adapted_pos.shape[0]
            n_neg = adapted_neg.shape[0]
            total_loss = torch.tensor(0.0)
            count = 0
            for i in range(n_pos):
                for j in range(n_neg):
                    loss = self._triplet_loss(
                        adapted_query,
                        adapted_pos[i:i + 1],
                        adapted_neg[j:j + 1]
                    )
                    total_loss = total_loss + loss
                    count += 1
            if count > 0:
                total_loss = total_loss / count
            return total_loss

    def get_state(self):
        """Return triplet loss state."""
        return {
            "loss_type": "triplet_margin",
            "margin": self.margin,
            "aggregation": self.aggregation,
            "call_count": self._call_count,
        }


# ============================================================
# InfoNCE Loss (NT-Xent)
# ============================================================

class InfoNCEOnlineLoss(OnlineLossFunction):
    """
    InfoNCE (NT-Xent) contrastive loss for online correction.

    Treats all positives as positive pairs with the query and all negatives
    as negative pairs. Uses temperature-scaled cosine similarity with
    log-softmax for the contrastive objective.

    Args:
        temperature: Temperature scaling parameter (default: 0.07)
    """

    def __init__(self, temperature=0.07):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self._call_count = 0

    def compute(self, adapted_query, adapted_pos, adapted_neg):
        """
        Compute InfoNCE loss.

        For each positive, compute similarity with query scaled by temperature,
        then compute log-softmax over all candidates (positives + negatives).
        """
        self._call_count += 1

        # Normalize for cosine similarity
        query_norm = F.normalize(adapted_query, dim=-1)  # (1, dim)
        pos_norm = F.normalize(adapted_pos, dim=-1)  # (N, dim)
        neg_norm = F.normalize(adapted_neg, dim=-1)  # (M, dim)

        # Similarities scaled by temperature
        pos_sim = torch.mm(query_norm, pos_norm.t()) / self.temperature  # (1, N)
        neg_sim = torch.mm(query_norm, neg_norm.t()) / self.temperature  # (1, M)

        # Concatenate all similarities: positives first, then negatives
        # logits shape: (1, N+M)
        logits = torch.cat([pos_sim, neg_sim], dim=1)

        # Labels: all positive indices are targets
        n_pos = pos_sim.shape[1]

        # Average loss over all positive examples
        # For each positive i, the target is index i in logits
        log_probs = F.log_softmax(logits, dim=1)  # (1, N+M)
        loss = -log_probs[0, :n_pos].mean()

        return loss

    def get_state(self):
        """Return InfoNCE loss state."""
        return {
            "loss_type": "infonce",
            "temperature": self.temperature,
            "call_count": self._call_count,
        }


# ============================================================
# Cosine Similarity Loss
# ============================================================

class CosineSimilarityOnlineLoss(OnlineLossFunction):
    """
    Direct cosine similarity optimization loss for online correction.

    Maximizes cosine similarity between query and positives while
    minimizing similarity between query and negatives.

    Args:
        pos_weight: Weight for positive similarity term (default: 1.0)
        neg_weight: Weight for negative similarity term (default: 1.0)
    """

    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        if pos_weight < 0 or neg_weight < 0:
            raise ValueError("pos_weight and neg_weight must be non-negative")
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self._call_count = 0

    def compute(self, adapted_query, adapted_pos, adapted_neg):
        """
        Compute cosine similarity loss.

        Loss = -pos_weight * mean(cos(query, pos)) + neg_weight * mean(cos(query, neg))
        """
        self._call_count += 1

        # Normalize for cosine similarity
        query_norm = F.normalize(adapted_query, dim=-1)  # (1, dim)
        pos_norm = F.normalize(adapted_pos, dim=-1)  # (N, dim)
        neg_norm = F.normalize(adapted_neg, dim=-1)  # (M, dim)

        # Cosine similarities
        pos_sim = torch.mm(query_norm, pos_norm.t()).mean()  # scalar
        neg_sim = torch.mm(query_norm, neg_norm.t()).mean()  # scalar

        # Loss: minimize negative similarity of positives, maximize for negatives
        loss = -self.pos_weight * pos_sim + self.neg_weight * neg_sim

        return loss

    def get_state(self):
        """Return cosine similarity loss state."""
        return {
            "loss_type": "cosine_similarity",
            "pos_weight": self.pos_weight,
            "neg_weight": self.neg_weight,
            "call_count": self._call_count,
        }


# ============================================================
# Loss Factory
# ============================================================

def create_online_loss(loss_type="triplet_margin", **kwargs):
    """
    Factory function for creating online loss functions.

    Args:
        loss_type: One of "triplet_margin", "infonce", "cosine_similarity"
        **kwargs: Loss-specific parameters

    Returns:
        OnlineLossFunction instance

    Raises:
        ValueError: If loss_type is unknown
    """
    loss_map = {
        "triplet_margin": TripletMarginOnlineLoss,
        "infonce": InfoNCEOnlineLoss,
        "cosine_similarity": CosineSimilarityOnlineLoss,
    }

    if loss_type not in loss_map:
        valid = ", ".join(loss_map.keys())
        raise ValueError(f"Unknown loss_type '{loss_type}'. Valid types: {valid}")

    return loss_map[loss_type](**kwargs)


# ============================================================
# Adaptive Margin
# ============================================================

class AdaptiveMargin:
    """
    Dynamic margin adaptation based on retrieval quality signals.

    Adjusts the triplet margin based on the quality gap between positive
    and negative retrieval results. When retrieval quality is high
    (large gap), uses a tighter margin. When quality is low (small gap),
    uses a larger margin to push harder.

    Args:
        base_margin: Starting margin value (default: 0.1)
        min_margin: Minimum allowed margin (default: 0.01)
        max_margin: Maximum allowed margin (default: 0.5)
        adaptation_rate: How quickly margin adapts (default: 0.1)
        window_size: Rolling window for quality tracking (default: 50)
    """

    def __init__(self, base_margin=0.1, min_margin=0.01, max_margin=0.5,
                 adaptation_rate=0.1, window_size=50):
        if base_margin < 0:
            raise ValueError("base_margin must be non-negative")
        if min_margin < 0:
            raise ValueError("min_margin must be non-negative")
        if max_margin < min_margin:
            raise ValueError("max_margin must be >= min_margin")
        if adaptation_rate <= 0 or adaptation_rate > 1.0:
            raise ValueError("adaptation_rate must be in (0, 1]")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.base_margin = base_margin
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size

        self._current_margin = base_margin
        self._quality_history = []

    def update(self, pos_scores, neg_scores):
        """
        Update margin based on retrieval quality.

        Args:
            pos_scores: Similarity scores of positive retrievals (list/array)
            neg_scores: Similarity scores of negative retrievals (list/array)

        Returns:
            float: Updated margin value
        """
        pos_arr = np.array(pos_scores, dtype=np.float64)
        neg_arr = np.array(neg_scores, dtype=np.float64)

        if len(pos_arr) == 0 or len(neg_arr) == 0:
            return self._current_margin

        # Quality gap: mean positive score - mean negative score
        quality_gap = float(pos_arr.mean() - neg_arr.mean())
        self._quality_history.append(quality_gap)

        # Keep window bounded
        if len(self._quality_history) > self.window_size:
            self._quality_history = self._quality_history[-self.window_size:]

        # Average gap over window
        avg_gap = np.mean(self._quality_history)

        # Target margin: inversely proportional to quality gap
        # High quality gap -> smaller margin (already well-separated)
        # Low quality gap -> larger margin (need more push)
        if avg_gap > 0:
            # Scale margin inversely with gap, clamped to range
            target = self.base_margin / (1.0 + avg_gap)
        else:
            # Negative gap means negatives are closer than positives - use max
            target = self.max_margin

        # Exponential moving average toward target
        self._current_margin = (
            (1 - self.adaptation_rate) * self._current_margin
            + self.adaptation_rate * target
        )

        # Clamp to allowed range
        self._current_margin = max(self.min_margin,
                                   min(self.max_margin, self._current_margin))

        return self._current_margin

    @property
    def current_margin(self):
        """Current adaptive margin value."""
        return self._current_margin

    def get_state(self):
        """Return adaptive margin state."""
        return {
            "current_margin": self._current_margin,
            "base_margin": self.base_margin,
            "min_margin": self.min_margin,
            "max_margin": self.max_margin,
            "adaptation_rate": self.adaptation_rate,
            "window_size": self.window_size,
            "history_length": len(self._quality_history),
            "avg_quality_gap": float(np.mean(self._quality_history))
            if self._quality_history else 0.0,
        }

    def reset(self):
        """Reset to base margin."""
        self._current_margin = self.base_margin
        self._quality_history = []


# ============================================================
# Online Loss Scheduler
# ============================================================

class OnlineLossScheduler:
    """
    Schedules loss weight decay for online updates.

    Composes TeacherWeightScheduler internally to reuse the 5 scheduling
    strategies (constant, linear_decay, cosine_annealing, step_decay,
    adaptive) for controlling how aggressively the online loss drives
    updates over time.

    Args:
        schedule: Schedule type passed to TeacherWeightScheduler
        initial_weight: Starting loss weight (default: 1.0)
        **kwargs: Additional params for TeacherWeightScheduler
    """

    def __init__(self, schedule="constant", initial_weight=1.0, **kwargs):
        from teacher_weight_scheduler import TeacherWeightScheduler
        self._scheduler = TeacherWeightScheduler(
            schedule=schedule,
            initial_weight=initial_weight,
            **kwargs
        )
        self._schedule = schedule
        self._initial_weight = initial_weight

    def step(self, loss=None):
        """
        Advance one step and return current loss weight.

        Args:
            loss: Optional loss value (used by adaptive schedule)

        Returns:
            float: Current loss weight multiplier
        """
        return self._scheduler.step(loss=loss)

    @property
    def current_weight(self):
        """Current loss weight."""
        return self._scheduler.current_weight

    def reset(self):
        """Reset scheduler state."""
        self._scheduler.reset()

    def get_state(self):
        """Return scheduler state."""
        return {
            "schedule": self._schedule,
            "initial_weight": self._initial_weight,
            "current_weight": self._scheduler.current_weight,
            "step_count": self._scheduler._step_count,
        }


# ============================================================
# Online Update Diagnostics
# ============================================================

class OnlineUpdateDiagnostics:
    """
    Per-dimension gradient statistics and stability bridge for online updates.

    Tracks gradient magnitudes, per-dimension running stats, loss trends,
    and optionally bridges to StabilityTracker for structural health
    monitoring.

    Args:
        input_dim: Dimensionality of adapter (for per-dimension stats)
        stability_tracker: Optional StabilityTracker instance for bridge
        history_size: Number of recent updates to track (default: 100)
    """

    def __init__(self, input_dim, stability_tracker=None, history_size=100):
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if history_size < 1:
            raise ValueError("history_size must be >= 1")

        self.input_dim = input_dim
        self._stability_tracker = stability_tracker
        self.history_size = history_size

        # Per-dimension gradient running stats
        self._grad_sum = np.zeros(input_dim, dtype=np.float64)
        self._grad_sq_sum = np.zeros(input_dim, dtype=np.float64)
        self._grad_count = 0

        # Loss history
        self._loss_history = []

        # Gradient norm history
        self._grad_norm_history = []

    def record_gradients(self, adapter):
        """
        Record gradient statistics from adapter parameters.

        Args:
            adapter: nn.Module adapter whose gradients to record
        """
        grads = []
        with torch.no_grad():
            for p in adapter.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().cpu().flatten())

        if not grads:
            return

        grad_vec = torch.cat(grads).numpy()
        grad_norm = float(np.linalg.norm(grad_vec))
        self._grad_norm_history.append(grad_norm)

        # Keep bounded
        if len(self._grad_norm_history) > self.history_size:
            self._grad_norm_history = self._grad_norm_history[-self.history_size:]

        # Per-dimension stats (truncate or pad to input_dim)
        dim_grads = grad_vec[:self.input_dim]
        if len(dim_grads) < self.input_dim:
            padded = np.zeros(self.input_dim, dtype=np.float64)
            padded[:len(dim_grads)] = dim_grads
            dim_grads = padded

        self._grad_sum += dim_grads
        self._grad_sq_sum += dim_grads ** 2
        self._grad_count += 1

        # Bridge to StabilityTracker if available
        if self._stability_tracker is not None:
            self._stability_tracker.record_adapter_snapshot(adapter)

    def record_loss(self, loss_value):
        """
        Record a loss value.

        Args:
            loss_value: float loss from update step
        """
        self._loss_history.append(float(loss_value))
        if len(self._loss_history) > self.history_size:
            self._loss_history = self._loss_history[-self.history_size:]

        # Bridge to StabilityTracker if available
        if self._stability_tracker is not None:
            self._stability_tracker.record_loss(float(loss_value))

    def get_per_dimension_stats(self):
        """
        Get per-dimension gradient mean and standard deviation.

        Returns:
            dict with 'mean' and 'std' arrays of shape (input_dim,)
        """
        if self._grad_count == 0:
            return {
                "mean": np.zeros(self.input_dim),
                "std": np.zeros(self.input_dim),
                "count": 0,
            }

        mean = self._grad_sum / self._grad_count
        variance = (self._grad_sq_sum / self._grad_count) - mean ** 2
        # Clamp negative variance from floating point
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)

        return {
            "mean": mean,
            "std": std,
            "count": self._grad_count,
        }

    def get_gradient_health(self):
        """
        Assess gradient health based on recent norms.

        Returns:
            dict with gradient health metrics
        """
        if not self._grad_norm_history:
            return {
                "mean_norm": 0.0,
                "max_norm": 0.0,
                "min_norm": 0.0,
                "std_norm": 0.0,
                "vanishing": False,
                "exploding": False,
                "count": 0,
            }

        norms = np.array(self._grad_norm_history)
        mean_norm = float(norms.mean())
        std_norm = float(norms.std()) if len(norms) > 1 else 0.0

        return {
            "mean_norm": mean_norm,
            "max_norm": float(norms.max()),
            "min_norm": float(norms.min()),
            "std_norm": std_norm,
            "vanishing": mean_norm < 1e-7,
            "exploding": mean_norm > 100.0,
            "count": len(norms),
        }

    def get_loss_trend(self):
        """
        Compute loss trend (slope of recent losses via linear regression).

        Returns:
            dict with trend information
        """
        if len(self._loss_history) < 2:
            return {
                "slope": 0.0,
                "improving": False,
                "count": len(self._loss_history),
                "recent_mean": float(np.mean(self._loss_history))
                if self._loss_history else 0.0,
            }

        losses = np.array(self._loss_history)
        x = np.arange(len(losses), dtype=np.float64)

        # Simple linear regression slope
        x_mean = x.mean()
        y_mean = losses.mean()
        numerator = ((x - x_mean) * (losses - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()

        slope = float(numerator / denominator) if denominator > 0 else 0.0

        return {
            "slope": slope,
            "improving": slope < 0,
            "count": len(losses),
            "recent_mean": float(losses[-min(10, len(losses)):].mean()),
        }

    def get_report(self):
        """
        Get comprehensive diagnostics report.

        Returns:
            dict with all diagnostic metrics
        """
        return {
            "per_dimension": self.get_per_dimension_stats(),
            "gradient_health": self.get_gradient_health(),
            "loss_trend": self.get_loss_trend(),
        }

    def reset(self):
        """Reset all diagnostic state."""
        self._grad_sum = np.zeros(self.input_dim, dtype=np.float64)
        self._grad_sq_sum = np.zeros(self.input_dim, dtype=np.float64)
        self._grad_count = 0
        self._loss_history = []
        self._grad_norm_history = []


# ============================================================
# Online Updater (extended with pluggable loss)
# ============================================================

class OnlineUpdater:
    """
    Performs micro-gradient updates on the adapter at inference time.

    Uses a pluggable loss function between query, positive (top-k), and
    negative (bottom-k) retrieval results to adapt the model online.

    Args:
        adapter: nn.Module adapter to update (any adapter type via parameters())
        learning_rate: Step size for micro-updates (default: 0.0001)
        micro_steps: Number of gradient steps per update (default: 1)
        momentum: SGD momentum (default: 0.9)
        max_grad_norm: Gradient clipping threshold (default: 1.0)
        update_interval: Apply update every N queries (default: 1)
        margin: Triplet margin (default: 0.1) - used when loss_type="triplet_margin"
        loss_type: Loss function type (default: "triplet_margin")
            Options: "triplet_margin", "infonce", "cosine_similarity"
        loss_kwargs: Additional keyword arguments for the loss function
        adaptive_margin: Optional AdaptiveMargin instance for dynamic margins
        scheduler: Optional OnlineLossScheduler for loss weight decay
        diagnostics: Optional OnlineUpdateDiagnostics for gradient tracking
    """

    def __init__(self, adapter, learning_rate=0.0001, micro_steps=1,
                 momentum=0.9, max_grad_norm=1.0, update_interval=1,
                 margin=0.1, loss_type="triplet_margin", loss_kwargs=None,
                 adaptive_margin=None, scheduler=None, diagnostics=None):
        if not isinstance(adapter, nn.Module):
            raise TypeError("adapter must be an nn.Module instance")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if micro_steps < 1:
            raise ValueError("micro_steps must be >= 1")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")

        self.adapter = adapter
        self.learning_rate = learning_rate
        self.micro_steps = micro_steps
        self.momentum = momentum
        self.max_grad_norm = max_grad_norm
        self.update_interval = update_interval
        self.margin = margin
        self.loss_type = loss_type
        self.logger = get_logger()

        # Optimizer state (persistent across queries)
        self._optimizer = torch.optim.SGD(
            self.adapter.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

        # Create pluggable loss function
        _loss_kwargs = loss_kwargs or {}
        if loss_type == "triplet_margin" and "margin" not in _loss_kwargs:
            _loss_kwargs["margin"] = self.margin
        self._loss_fn = create_online_loss(loss_type, **_loss_kwargs)

        # Legacy triplet loss for backward compatibility of internal state
        self._triplet_loss = nn.TripletMarginLoss(margin=self.margin)

        # Optional components
        self._adaptive_margin = adaptive_margin
        self._scheduler = scheduler
        self._diagnostics = diagnostics

        # Tracking
        self._query_count = 0
        self._update_count = 0
        self._total_loss = 0.0

    def update(self, query_vec, top_k_vecs, bottom_k_vecs,
               pos_scores=None, neg_scores=None):
        """
        Perform online micro-gradient update based on retrieval results.

        Args:
            query_vec: Query embedding as numpy array (1D)
            top_k_vecs: Top-k retrieved embeddings as numpy array (2D)
            bottom_k_vecs: Bottom-k retrieved embeddings as numpy array (2D)
            pos_scores: Optional similarity scores for positive retrievals
            neg_scores: Optional similarity scores for negative retrievals

        Returns:
            dict: Update info with keys 'updated' (bool), 'loss' (float or None)
        """
        self._query_count += 1

        # Check update interval
        if self._query_count % self.update_interval != 0:
            return {"updated": False, "loss": None}

        # Need at least 1 positive and 1 negative
        if len(top_k_vecs) == 0 or len(bottom_k_vecs) == 0:
            return {"updated": False, "loss": None}

        # Update adaptive margin if present
        if self._adaptive_margin is not None and pos_scores is not None and neg_scores is not None:
            new_margin = self._adaptive_margin.update(pos_scores, neg_scores)
            # Update triplet loss margin if using triplet_margin loss
            if isinstance(self._loss_fn, TripletMarginOnlineLoss):
                self._loss_fn.margin = new_margin
                self._loss_fn._triplet_loss = nn.TripletMarginLoss(margin=new_margin)

        # Convert to tensors
        query_t = torch.tensor(query_vec, dtype=torch.float32).unsqueeze(0)
        pos_t = torch.tensor(np.array(top_k_vecs), dtype=torch.float32)
        neg_t = torch.tensor(np.array(bottom_k_vecs), dtype=torch.float32)

        was_training = self.adapter.training
        self.adapter.train()

        total_loss = 0.0
        for _ in range(self.micro_steps):
            self._optimizer.zero_grad()

            # Pass through adapter
            adapted_query = self.adapter(query_t)
            adapted_pos = self.adapter(pos_t)
            adapted_neg = self.adapter(neg_t)

            # Compute loss using pluggable loss function
            loss = self._loss_fn.compute(adapted_query, adapted_pos, adapted_neg)

            # Apply scheduler weight if present
            if self._scheduler is not None:
                weight = self._scheduler.current_weight
                loss = loss * weight

            loss.backward()

            # Record diagnostics before clipping
            if self._diagnostics is not None:
                self._diagnostics.record_gradients(self.adapter)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.adapter.parameters(), self.max_grad_norm
            )

            self._optimizer.step()
            total_loss += loss.item()

        if not was_training:
            self.adapter.eval()

        avg_loss = total_loss / self.micro_steps
        self._update_count += 1
        self._total_loss += avg_loss

        # Step scheduler if present
        if self._scheduler is not None:
            self._scheduler.step(loss=avg_loss)

        # Record loss in diagnostics
        if self._diagnostics is not None:
            self._diagnostics.record_loss(avg_loss)

        self.logger.log_event(
            "online_update",
            f"Online update #{self._update_count}: loss={avg_loss:.6f}",
            level="DEBUG",
            update_count=self._update_count,
            loss=avg_loss
        )

        return {"updated": True, "loss": avg_loss}

    @property
    def query_count(self):
        """Total queries seen."""
        return self._query_count

    @property
    def update_count(self):
        """Total updates performed."""
        return self._update_count

    @property
    def average_loss(self):
        """Average loss across all updates."""
        if self._update_count == 0:
            return 0.0
        return self._total_loss / self._update_count

    @property
    def loss_function(self):
        """The active loss function instance."""
        return self._loss_fn

    def get_stats(self):
        """Get update statistics."""
        stats = {
            "query_count": self._query_count,
            "update_count": self._update_count,
            "average_loss": self.average_loss,
            "learning_rate": self.learning_rate,
            "micro_steps": self.micro_steps,
            "update_interval": self.update_interval,
            "loss_type": self.loss_type,
        }

        # Include loss function state
        stats["loss_state"] = self._loss_fn.get_state()

        # Include adaptive margin state if present
        if self._adaptive_margin is not None:
            stats["adaptive_margin"] = self._adaptive_margin.get_state()

        # Include scheduler state if present
        if self._scheduler is not None:
            stats["scheduler"] = self._scheduler.get_state()

        # Include diagnostics report if present
        if self._diagnostics is not None:
            stats["diagnostics"] = self._diagnostics.get_report()

        return stats

    def reset_stats(self):
        """Reset statistics counters (does not reset optimizer state)."""
        self._query_count = 0
        self._update_count = 0
        self._total_loss = 0.0
