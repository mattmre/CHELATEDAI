"""
Structural Stability Metrics for ChelatedAI

Diagnostic tracking of chelation mask stability, variance convergence,
persistent collapse sets, threshold oscillation, and adapter weight drift.

Provides the StabilityTracker class for comprehensive structural health
monitoring of the chelation pipeline.
"""

import numpy as np
import torch
from chelation_logger import get_logger


class StabilityTracker:
    """
    Tracks structural stability metrics across inference and training cycles.

    Records masks, variance distributions, collapse sets, thresholds,
    and adapter weight snapshots to compute stability diagnostics.
    """

    def __init__(self):
        self.logger = get_logger()

        # Mask history
        self._mask_history = []

        # Variance distribution history
        self._variance_history = []

        # Collapse set history (sets of doc IDs)
        self._collapse_history = []

        # Threshold history
        self._threshold_history = []

        # Adapter weight snapshots (flattened parameter vectors)
        self._adapter_snapshots = []

        # Loss history from training
        self._loss_history = []

        # Runtime norm diagnostics
        self._norm_history = []

    def record_mask(self, mask):
        """
        Record a chelation mask from inference.

        Args:
            mask: numpy array, binary mask of shape (input_dim,)
        """
        self._mask_history.append(np.array(mask, dtype=float))

    def record_variance_distribution(self, variances):
        """
        Record per-dimension variance distribution.

        Args:
            variances: numpy array of per-dimension variances
        """
        self._variance_history.append(np.array(variances, dtype=float))

    def record_collapse_set(self, doc_ids):
        """
        Record the set of document IDs that collapsed in this cycle.

        Args:
            doc_ids: iterable of document IDs
        """
        self._collapse_history.append(set(doc_ids))

    def record_threshold(self, threshold):
        """
        Record an active chelation threshold value.

        Args:
            threshold: float threshold value
        """
        self._threshold_history.append(float(threshold))

    def record_adapter_snapshot(self, adapter):
        """
        Record a snapshot of adapter weights.

        Args:
            adapter: nn.Module adapter instance
        """
        with torch.no_grad():
            params = []
            for p in adapter.parameters():
                params.append(p.detach().cpu().flatten())
            if params:
                snapshot = torch.cat(params).numpy()
                self._adapter_snapshots.append(snapshot)

    def record_loss(self, loss):
        """
        Record a training loss value.

        Args:
            loss: float loss value
        """
        self._loss_history.append(float(loss))

    def record_norms(self, query_norm=None, result_norms=None, adapter_input_norm=None, adapter_output_norm=None):
        """
        Record runtime vector norm diagnostics without storing raw vectors.
        """

        entry = {}
        if query_norm is not None:
            entry["query_norm"] = float(query_norm)
        if result_norms is not None:
            norms = np.array(result_norms, dtype=float).flatten()
            entry["result_norm_mean"] = float(np.mean(norms)) if norms.size else None
            entry["result_norm_std"] = float(np.std(norms)) if norms.size else None
            entry["result_count"] = int(norms.size)
        if adapter_input_norm is not None:
            entry["adapter_input_norm"] = float(adapter_input_norm)
        if adapter_output_norm is not None:
            entry["adapter_output_norm"] = float(adapter_output_norm)
        if adapter_input_norm is not None and adapter_output_norm is not None:
            denominator = max(abs(float(adapter_input_norm)), 1e-12)
            entry["adapter_norm_ratio"] = float(adapter_output_norm) / denominator
        self._norm_history.append(entry)

    # ==================== Metric Computations ====================

    def compute_mask_stability(self):
        """
        Compute Jaccard stability between consecutive masks.

        Returns:
            list of float: Jaccard similarities between consecutive mask pairs.
            Empty list if fewer than 2 masks recorded.
        """
        if len(self._mask_history) < 2:
            return []

        stabilities = []
        for i in range(1, len(self._mask_history)):
            prev = self._mask_history[i - 1]
            curr = self._mask_history[i]

            # Jaccard: |intersection| / |union|
            intersection = np.sum((prev > 0) & (curr > 0))
            union = np.sum((prev > 0) | (curr > 0))

            if union == 0:
                jaccard = 1.0  # Both empty masks
            else:
                jaccard = float(intersection) / float(union)

            stabilities.append(jaccard)

        return stabilities

    def compute_variance_convergence(self):
        """
        Compute Pearson correlation between consecutive variance distributions.

        Returns:
            list of float: Pearson correlations between consecutive distributions.
            Empty list if fewer than 2 distributions recorded.
        """
        if len(self._variance_history) < 2:
            return []

        correlations = []
        for i in range(1, len(self._variance_history)):
            prev = self._variance_history[i - 1]
            curr = self._variance_history[i]

            # Handle constant arrays (zero std)
            if np.std(prev) < 1e-12 or np.std(curr) < 1e-12:
                correlations.append(1.0 if np.allclose(prev, curr) else 0.0)
            else:
                corr = np.corrcoef(prev, curr)[0, 1]
                correlations.append(float(corr))

        return correlations

    def compute_persistent_collapse_ratio(self):
        """
        Compute the ratio of documents that persistently collapse across cycles.

        A document is "persistent" if it appears in more than half of all
        recorded collapse sets.

        Returns:
            float: Ratio of persistent collapsers to all unique collapsers.
            0.0 if no collapse history.
        """
        if len(self._collapse_history) == 0:
            return 0.0

        # Count occurrences of each doc_id
        from collections import Counter
        counts = Counter()
        for cset in self._collapse_history:
            for doc_id in cset:
                counts[doc_id] += 1

        if len(counts) == 0:
            return 0.0

        threshold = len(self._collapse_history) / 2.0
        persistent = sum(1 for c in counts.values() if c > threshold)

        return persistent / len(counts)

    def compute_threshold_oscillation(self):
        """
        Compute threshold oscillation as the standard deviation of threshold values.

        Returns:
            float: Standard deviation of recorded thresholds.
            0.0 if fewer than 2 thresholds recorded.
        """
        if len(self._threshold_history) < 2:
            return 0.0

        return float(np.std(self._threshold_history))

    def compute_adapter_drift(self):
        """
        Compute adapter weight drift as L2 distance between consecutive snapshots.

        Returns:
            list of float: L2 distances between consecutive snapshots.
            Empty list if fewer than 2 snapshots recorded.
        """
        if len(self._adapter_snapshots) < 2:
            return []

        drifts = []
        for i in range(1, len(self._adapter_snapshots)):
            prev = self._adapter_snapshots[i - 1]
            curr = self._adapter_snapshots[i]
            drift = float(np.linalg.norm(curr - prev))
            drifts.append(drift)

        return drifts

    def compute_norm_drift_report(self):
        """
        Summarize runtime norm drift from query/result/adapter norm observations.
        """

        if not self._norm_history:
            return {
                "count": 0,
                "latest": None,
                "query_norm_delta": None,
                "result_norm_mean_delta": None,
                "adapter_norm_ratio_latest": None,
                "adapter_norm_ratio_mean": None,
            }

        latest = dict(self._norm_history[-1])

        def delta_for(key):
            values = [entry[key] for entry in self._norm_history if entry.get(key) is not None]
            if len(values) < 2:
                return None
            return float(values[-1] - values[0])

        ratios = [
            entry["adapter_norm_ratio"]
            for entry in self._norm_history
            if entry.get("adapter_norm_ratio") is not None
        ]
        return {
            "count": len(self._norm_history),
            "latest": latest,
            "query_norm_delta": delta_for("query_norm"),
            "result_norm_mean_delta": delta_for("result_norm_mean"),
            "adapter_norm_ratio_latest": float(ratios[-1]) if ratios else None,
            "adapter_norm_ratio_mean": float(np.mean(ratios)) if ratios else None,
        }

    def get_stability_report(self):
        """
        Get comprehensive stability report.

        Returns:
            dict with all stability metrics
        """
        mask_stab = self.compute_mask_stability()
        var_conv = self.compute_variance_convergence()
        adapter_drift = self.compute_adapter_drift()

        report = {
            "mask_stability": {
                "values": mask_stab,
                "mean": float(np.mean(mask_stab)) if mask_stab else None,
                "min": float(np.min(mask_stab)) if mask_stab else None,
                "count": len(self._mask_history)
            },
            "variance_convergence": {
                "values": var_conv,
                "mean": float(np.mean(var_conv)) if var_conv else None,
                "count": len(self._variance_history)
            },
            "persistent_collapse_ratio": self.compute_persistent_collapse_ratio(),
            "threshold_oscillation": self.compute_threshold_oscillation(),
            "adapter_drift": {
                "values": adapter_drift,
                "mean": float(np.mean(adapter_drift)) if adapter_drift else None,
                "total": float(np.sum(adapter_drift)) if adapter_drift else 0.0,
                "count": len(self._adapter_snapshots)
            },
            "norm_drift": self.compute_norm_drift_report(),
            "loss_history": list(self._loss_history),
            "total_inferences_tracked": len(self._mask_history),
            "total_training_cycles_tracked": len(self._collapse_history)
        }

        return report

    def reset(self):
        """Reset all tracking state."""
        self._mask_history = []
        self._variance_history = []
        self._collapse_history = []
        self._threshold_history = []
        self._adapter_snapshots = []
        self._loss_history = []
        self._norm_history = []
