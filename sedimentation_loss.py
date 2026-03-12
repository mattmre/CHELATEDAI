"""Loss functions for sedimentation training.

Provides contrastive alternatives to MSE for adapter training,
following EmbedDistill (Thakur 2023) and RankDistil (AISTATS 2021).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List


class SedimentationInfoNCELoss(nn.Module):
    """InfoNCE loss for sedimentation training.

    For each sample, treats its target as positive and all other
    targets in the batch as negatives. This teaches the adapter
    to preserve retrieval structure, not just minimize pointwise error.

    Args:
        temperature: Temperature scaling for similarity scores (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            outputs: Adapter outputs (batch_size, dim)
            targets: Target vectors (batch_size, dim)

        Returns:
            Scalar loss
        """
        # Normalize for cosine similarity
        outputs_norm = F.normalize(outputs, dim=1)
        targets_norm = F.normalize(targets, dim=1)

        # Similarity matrix: each output vs all targets
        # Shape: (batch_size, batch_size)
        sim_matrix = torch.mm(outputs_norm, targets_norm.t()) / self.temperature

        # Labels: diagonal entries are positives (output[i] should match target[i])
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        # Cross-entropy loss treats this as classification:
        # for each output, classify which target it belongs to
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class SedimentationHybridLoss(nn.Module):
    """Hybrid loss combining MSE for stability with InfoNCE for retrieval quality.

    Args:
        temperature: InfoNCE temperature (default: 0.07)
        contrastive_weight: Weight for contrastive term (default: 0.5)
        mse_weight: Weight for MSE term (default: 0.5)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrastive_weight: float = 0.5,
        mse_weight: float = 0.5,
    ):
        super().__init__()
        self.infonce = SedimentationInfoNCELoss(temperature=temperature)
        self.mse = nn.MSELoss()
        self.contrastive_weight = contrastive_weight
        self.mse_weight = mse_weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute hybrid MSE + InfoNCE loss.

        Args:
            outputs: Adapter outputs (batch_size, dim)
            targets: Target vectors (batch_size, dim)

        Returns:
            Scalar loss (weighted sum of MSE and InfoNCE)
        """
        loss_mse = self.mse(outputs, targets)
        loss_infonce = self.infonce(outputs, targets)
        return self.mse_weight * loss_mse + self.contrastive_weight * loss_infonce


class HardNegativeMiner:
    """Mines hard negatives from chelation_log collision data.

    The chelation_log records embedding collisions (unrelated docs with similar
    embeddings). These are the best hard negatives for contrastive training.
    """

    def __init__(self, chelation_log: Dict[Any, List], max_negatives: int = 16):
        self.chelation_log = chelation_log
        self.max_negatives = max_negatives

    def get_hard_negative_indices(
        self,
        batch_indices: List[int],
        total_size: int,
    ) -> List[List[int]]:
        """Get indices of hard negatives for each sample in the batch.

        Args:
            batch_indices: Indices of current batch samples
            total_size: Total dataset size

        Returns:
            List of lists of hard negative indices
        """
        hard_negs: List[List[int]] = []
        # Flatten chelation_log to find collision partners
        collision_map: Dict[Any, List[int]] = {}
        for key, entries in self.chelation_log.items():
            for entry in entries:
                if isinstance(entry, dict):
                    doc_id = entry.get("doc_id", entry.get("id", None))
                    collisions = entry.get("collisions", [])
                    if doc_id is not None:
                        collision_map[doc_id] = collisions

        for idx in batch_indices:
            neg_indices = collision_map.get(idx, [])
            # Limit and filter to valid indices
            neg_indices = [n for n in neg_indices if n < total_size and n != idx]
            neg_indices = neg_indices[: self.max_negatives]
            hard_negs.append(neg_indices)

        return hard_negs


def create_sedimentation_loss(loss_type: str = "mse", **kwargs) -> nn.Module:
    """Factory for sedimentation loss functions.

    Args:
        loss_type: "mse", "infonce", or "hybrid"
        **kwargs: Loss-specific parameters

    Returns:
        nn.Module loss function

    Raises:
        ValueError: If loss_type is not recognized
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "infonce":
        temperature = kwargs.get("temperature", 0.07)
        return SedimentationInfoNCELoss(temperature=temperature)
    elif loss_type == "hybrid":
        return SedimentationHybridLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Valid: mse, infonce, hybrid"
        )
