"""
Online Gradient Updater for ChelatedAI Inference-Time Adaptation

Provides lightweight per-query adapter updates at inference time using
contrastive learning from retrieval results.

Inspired by:
- Online-Optimized RAG (arXiv:2509.20415)
- TTARAG (arXiv:2601.11443)
"""

import torch
import torch.nn as nn
import numpy as np
from chelation_logger import get_logger


class OnlineUpdater:
    """
    Performs micro-gradient updates on the adapter at inference time.

    Uses triplet-margin loss between query, positive (top-k), and negative
    (bottom-k) retrieval results to adapt the model online.

    Args:
        adapter: nn.Module adapter to update (any adapter type via parameters())
        learning_rate: Step size for micro-updates (default: 0.0001)
        micro_steps: Number of gradient steps per update (default: 1)
        momentum: SGD momentum (default: 0.9)
        max_grad_norm: Gradient clipping threshold (default: 1.0)
        update_interval: Apply update every N queries (default: 1)
        margin: Triplet margin (default: 0.1)
    """

    def __init__(self, adapter, learning_rate=0.0001, micro_steps=1,
                 momentum=0.9, max_grad_norm=1.0, update_interval=1,
                 margin=0.1):
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
        self.logger = get_logger()

        # Optimizer state (persistent across queries)
        self._optimizer = torch.optim.SGD(
            self.adapter.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )
        self._triplet_loss = nn.TripletMarginLoss(margin=self.margin)

        # Tracking
        self._query_count = 0
        self._update_count = 0
        self._total_loss = 0.0

    def update(self, query_vec, top_k_vecs, bottom_k_vecs):
        """
        Perform online micro-gradient update based on retrieval results.

        Args:
            query_vec: Query embedding as numpy array (1D)
            top_k_vecs: Top-k retrieved embeddings as numpy array (2D)
            bottom_k_vecs: Bottom-k retrieved embeddings as numpy array (2D)

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

        # Convert to tensors
        query_t = torch.tensor(query_vec, dtype=torch.float32).unsqueeze(0)
        pos_t = torch.tensor(np.array(top_k_vecs), dtype=torch.float32)
        neg_t = torch.tensor(np.array(bottom_k_vecs), dtype=torch.float32)

        # Use mean of positives and negatives as anchor pairs
        pos_mean = pos_t.mean(dim=0, keepdim=True)
        neg_mean = neg_t.mean(dim=0, keepdim=True)

        was_training = self.adapter.training
        self.adapter.train()

        total_loss = 0.0
        for _ in range(self.micro_steps):
            self._optimizer.zero_grad()

            # Pass through adapter
            adapted_query = self.adapter(query_t)
            adapted_pos = self.adapter(pos_mean)
            adapted_neg = self.adapter(neg_mean)

            # Triplet loss: pull query toward positive, push away from negative
            loss = self._triplet_loss(adapted_query, adapted_pos, adapted_neg)
            loss.backward()

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

    def get_stats(self):
        """Get update statistics."""
        return {
            "query_count": self._query_count,
            "update_count": self._update_count,
            "average_loss": self.average_loss,
            "learning_rate": self.learning_rate,
            "micro_steps": self.micro_steps,
            "update_interval": self.update_interval
        }

    def reset_stats(self):
        """Reset statistics counters (does not reset optimizer state)."""
        self._query_count = 0
        self._update_count = 0
        self._total_loss = 0.0
