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
        loss_type: "mse", "infonce", "hybrid", "opsd_asymmetric", "opsd_jsd_sim", "filtered_clipped"
        **kwargs: Loss-specific parameters (directive, diagnostics, chelation_log, lambda_ret, etc. for new types)

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
    elif loss_type == "opsd_asymmetric":
        # OPSD/SDPO-inspired: asymmetric privileged residual + retention (Candidate 1)
        return OPSDAymmetricPrivilegedLoss(**kwargs)
    elif loss_type == "opsd_jsd_sim":
        # JSD on in-batch similarity distributions (Candidate 2)
        return OPSDJSDSimilarityDistillLoss(**kwargs)
    elif loss_type == "filtered_clipped":
        # MIS-PO filtered + pointwise clipped (Candidate 4)
        return FilteredClippedDistillLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Valid: mse, infonce, hybrid, opsd_asymmetric, opsd_jsd_sim, filtered_clipped"
        )


# ============================================================
# OPSD / SDPO / MIS-PO Inspired Chelation Self-Distillation Losses
# Added for CHELATION_OPSD_UPGRADE (Loop 1, Agent 3)
# These close the self-healing directive → training loop with
# privileged teacher, retention, filtering, and clipping.
# ============================================================

class OPSDAymmetricPrivilegedLoss(nn.Module):
    """
    OPSD/SDPO-inspired asymmetric privileged residual distillation (Candidate 1).

    Student: normal ChelationAdapter on x.
    Teacher: privileged target from SelfEditDirective + diagnostics + chelation_log
             (reuses compute_homeostatic_target logic + synthetic examples).

    Includes explicit retention to base embeddings + adapter regularization.
    Gradients flow only through student (teacher targets detached).
    """

    def __init__(self, lambda_ret: float = 0.3, lambda_reg: float = 0.01,
                 clip_delta: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.lambda_ret = lambda_ret
        self.lambda_reg = lambda_reg
        self.clip_delta = clip_delta
        self.temperature = temperature

    def forward(self, student_embs: torch.Tensor, base_embs: torch.Tensor,
                teacher_targets: torch.Tensor, adapter=None, **extra) -> torch.Tensor:
        """
        Args:
            student_embs: [B, D] from adapter(x)  (already normalized ideally)
            base_embs: [B, D] original base model embeddings (normalized)
            teacher_targets: [B, D] privileged targets (detached, from directive/diagnostics)
            adapter: optional adapter with .regularization_loss()
        """
        student_norm = F.normalize(student_embs, p=2, dim=1)
        base_norm = F.normalize(base_embs, p=2, dim=1)
        teacher_norm = F.normalize(teacher_targets.detach(), p=2, dim=1)

        corr_loss = F.mse_loss(student_norm, teacher_norm)
        ret_loss = F.mse_loss(student_norm, base_norm)

        reg = 0.0
        if adapter is not None and hasattr(adapter, "regularization_loss"):
            reg = adapter.regularization_loss()

        delta = student_norm - base_norm
        delta_norm = torch.norm(delta, dim=1, keepdim=True)
        # Convex excess penalty: zero inside the clip_delta ball, quadratic outside,
        # so large divergences keep a nonzero restoring gradient (D-04).
        excess_delta = torch.clamp(delta_norm - self.clip_delta, min=0.0)
        delta_reg = torch.mean(excess_delta ** 2)

        total = (corr_loss +
                 self.lambda_ret * ret_loss +
                 self.lambda_reg * reg +
                 0.1 * delta_reg)
        return total


class OPSDJSDSimilarityDistillLoss(nn.Module):
    """
    JSD / KL on in-batch similarity distributions (Candidate 2).
    Full-vocabulary OPSD analog: match teacher vs student beliefs over "which items match".
    Extends InfoNCE structure for privileged teacher guidance.
    """

    def __init__(self, temperature: float = 0.07, beta: float = 0.5, lambda_ret: float = 0.2):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.temperature = temperature
        self.beta = beta
        self.lambda_ret = lambda_ret

    def forward(self, student_embs: torch.Tensor, teacher_targets: torch.Tensor,
                base_embs: torch.Tensor, **extra) -> torch.Tensor:
        """Generalized JSD between teacher and student neighbour distributions.

        P_s / P_t are softmax(S / T) over each row's off-diagonal similarities
        (temperature applied once, self-similarity excluded). The mixture is taken
        in probability space, M = beta * P_t + (1 - beta) * P_s, and the KL terms are
        weighted to match: beta * KL(P_t || M) + (1 - beta) * KL(P_s || M). The loss
        is exactly 0 when student == teacher == base (D-03).

        Note: similarity distributions are invariant to a global orthogonal rotation
        of the student, so this term alone cannot detect rotational misalignment;
        only the retention term (toward base_embs) sees absolute orientation.
        """
        student_norm = F.normalize(student_embs, p=2, dim=1)
        teacher_norm = F.normalize(teacher_targets.detach(), p=2, dim=1)
        base_norm = F.normalize(base_embs, p=2, dim=1)

        ret_loss = F.mse_loss(student_norm, base_norm)

        batch_size = student_norm.size(0)
        if batch_size < 2:
            # No off-diagonal neighbours: the similarity distribution is undefined.
            return self.lambda_ret * ret_loss

        off_diag = ~torch.eye(batch_size, dtype=torch.bool, device=student_norm.device)
        sim_s = (torch.mm(student_norm, student_norm.t()) / self.temperature)
        sim_t = (torch.mm(teacher_norm, teacher_norm.t()) / self.temperature)
        sim_s = sim_s.masked_select(off_diag).view(batch_size, batch_size - 1)
        sim_t = sim_t.masked_select(off_diag).view(batch_size, batch_size - 1)

        log_p_s = F.log_softmax(sim_s, dim=1)
        log_p_t = F.log_softmax(sim_t, dim=1)
        p_s = log_p_s.exp()
        p_t = log_p_t.exp()

        log_m = (self.beta * p_t + (1.0 - self.beta) * p_s).clamp(min=1e-30).log()

        kl_t = (p_t * (log_p_t - log_m)).sum(dim=1).mean()
        kl_s = (p_s * (log_p_s - log_m)).sum(dim=1).mean()
        jsd = self.beta * kl_t + (1.0 - self.beta) * kl_s

        return jsd + self.lambda_ret * ret_loss


class FilteredClippedDistillLoss(nn.Module):
    """
    MIS-PO-style binary filtering + OPSD pointwise clipping (Candidate 4).
    Filters on fitness_gain / delta_norm / quant_survival ratios.
    Clips per-element divergence contributions to bound "stylistic" variance.
    + small KL retention penalty.
    """

    def __init__(self, r_min: float = 0.7, r_max: float = 1.4, tau: float = 2.0,
                 kl_beta: float = 0.05):
        super().__init__()
        if r_min >= r_max:
            raise ValueError("r_min must be strictly less than r_max")
        self.r_min = r_min
        self.r_max = r_max
        self.tau = tau
        self.kl_beta = kl_beta

    def forward(self, student_embs: torch.Tensor, teacher_targets: torch.Tensor,
                fitness_gains: torch.Tensor, delta_norms: torch.Tensor,
                quant_survivals: torch.Tensor, base_embs: torch.Tensor, **extra) -> torch.Tensor:
        # Filter on the raw gain/displacement ratio so (r_min, r_max) are real bounds
        # (a sigmoid squashed it into (0, 1), making r_max = 1.4 vacuous) (D-04).
        ratios = fitness_gains / delta_norms.clamp(min=1e-6)
        mask = ((ratios > self.r_min) & (ratios < self.r_max) & quant_survivals.bool()).float()

        raw_div = (F.normalize(student_embs, dim=1) - F.normalize(teacher_targets.detach(), dim=1)) ** 2
        clipped_div = torch.minimum(raw_div, torch.full_like(raw_div, self.tau))
        # Mean over accepted samples only; an all-rejected batch contributes 0.
        per_sample_div = clipped_div.mean(dim=1)
        filtered = (per_sample_div * mask).sum() / mask.sum().clamp(min=1.0)

        # Small retention KL proxy (output space)
        base_norm = F.normalize(base_embs, dim=1)
        student_norm = F.normalize(student_embs, dim=1)
        kl_pen = F.mse_loss(student_norm, base_norm.detach()) * self.kl_beta

        return filtered + kl_pen
