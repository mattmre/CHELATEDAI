"""
Learned Dimension Mask Predictor for ChelatedAI

Replaces variance-threshold masking with a neural importance predictor
that learns which dimensions to mask based on the local cluster.

Inspired by:
- MRL / Matryoshka Representation Learning (arXiv:2602.03306)
- VectorQ per-document quality scores (arXiv:2502.03771)

The MaskPreTrainer distills knowledge from variance-based masks (teacher)
into the neural predictor (student) for improved generalization.
"""

import torch
import torch.nn as nn
import numpy as np
from chelation_logger import get_logger


class DimensionMaskPredictor(nn.Module):
    """
    Neural predictor for per-dimension importance masks.

    Architecture: Linear -> ReLU -> Linear -> Sigmoid
    Input: aggregated cluster statistics (mean + variance = 2 * input_dim)
    Output: per-dimension importance scores in [0, 1]

    Args:
        input_dim: Embedding dimension
        hidden_ratio: Hidden layer size as ratio of input_dim (default: 0.25)
        threshold: Decision threshold for binary mask (default: 0.5)
    """

    def __init__(self, input_dim, hidden_ratio=0.25, threshold=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.threshold = threshold

        # Input: concatenation of mean vector and variance vector = 2 * input_dim
        feature_dim = 2 * input_dim
        hidden_dim = max(1, int(input_dim * hidden_ratio))

        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        # Initialize small weights for near-uniform predictions initially
        nn.init.normal_(self.predictor[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.predictor[0].bias)
        nn.init.normal_(self.predictor[2].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.predictor[2].bias)

    def forward(self, cluster_mean, cluster_variance):
        """
        Predict importance scores for each dimension.

        Args:
            cluster_mean: Mean vector of local cluster (1D tensor, shape [input_dim])
            cluster_variance: Variance vector of local cluster (1D tensor, shape [input_dim])

        Returns:
            Tensor of importance scores in [0, 1], shape [input_dim]
        """
        # Concatenate features
        features = torch.cat([cluster_mean, cluster_variance], dim=-1)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        scores = self.predictor(features)

        if scores.dim() == 2 and scores.shape[0] == 1:
            scores = scores.squeeze(0)

        return scores

    def predict_mask(self, cluster_np):
        """
        Predict binary mask from a numpy cluster array.

        Convenience method for integration with AntigravityEngine.

        Args:
            cluster_np: numpy array of shape (n_samples, input_dim)

        Returns:
            numpy array binary mask of shape (input_dim,)
        """
        if len(cluster_np) == 0:
            return np.ones(self.input_dim)

        cluster_mean = np.mean(cluster_np, axis=0)
        cluster_var = np.var(cluster_np, axis=0)

        with torch.no_grad():
            mean_t = torch.tensor(cluster_mean, dtype=torch.float32)
            var_t = torch.tensor(cluster_var, dtype=torch.float32)
            scores = self.forward(mean_t, var_t)
            mask = (scores >= self.threshold).float().numpy()

        return mask


class MaskPreTrainer:
    """
    Pre-trains the DimensionMaskPredictor using variance-based masks as teacher.

    Collects (cluster_stats, variance_mask) pairs during inference and
    periodically trains the predictor to mimic the variance-based masks.

    Args:
        predictor: DimensionMaskPredictor instance
        chelation_p: Percentile for variance-based teacher masks
        learning_rate: Training learning rate (default: 0.001)
        buffer_size: Maximum training examples to store (default: 1000)
    """

    def __init__(self, predictor, chelation_p=85, learning_rate=0.001, buffer_size=1000):
        if not isinstance(predictor, DimensionMaskPredictor):
            raise TypeError("predictor must be a DimensionMaskPredictor instance")

        self.predictor = predictor
        self.chelation_p = chelation_p
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.logger = get_logger()

        # Training buffer: list of (mean, variance, teacher_mask) tuples
        self._buffer = []

    def record_example(self, cluster_np):
        """
        Record a training example from a local cluster.

        Computes the variance-based teacher mask and stores the example.

        Args:
            cluster_np: numpy array of shape (n_samples, input_dim)
        """
        if len(cluster_np) == 0:
            return

        cluster_mean = np.mean(cluster_np, axis=0)
        cluster_var = np.var(cluster_np, axis=0)

        # Teacher mask: variance-based
        threshold = np.percentile(cluster_var, self.chelation_p)
        teacher_mask = (cluster_var < threshold).astype(float)

        self._buffer.append((cluster_mean, cluster_var, teacher_mask))

        # Trim buffer if needed
        if len(self._buffer) > self.buffer_size:
            self._buffer = self._buffer[-self.buffer_size:]

    def train(self, epochs=10, convergence_monitor=None):
        """
        Train the predictor on buffered examples.

        Args:
            epochs: Number of training epochs
            convergence_monitor: Optional ConvergenceMonitor for early stopping

        Returns:
            dict: Training results with 'final_loss', 'epochs_trained', 'converged'
        """
        if len(self._buffer) == 0:
            return {"final_loss": None, "epochs_trained": 0, "converged": False}

        # Build tensors
        means = torch.tensor(np.array([x[0] for x in self._buffer]), dtype=torch.float32)
        variances = torch.tensor(np.array([x[1] for x in self._buffer]), dtype=torch.float32)
        targets = torch.tensor(np.array([x[2] for x in self._buffer]), dtype=torch.float32)

        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        self.predictor.train()
        final_loss = 0.0
        epochs_trained = 0
        converged = False

        for epoch in range(epochs):
            optimizer.zero_grad()

            features = torch.cat([means, variances], dim=1)
            predictions = self.predictor.predictor(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            final_loss = loss.item()
            epochs_trained = epoch + 1

            if convergence_monitor is not None:
                if convergence_monitor.record_loss(final_loss):
                    converged = True
                    break

        self.predictor.eval()

        self.logger.log_event(
            "mask_pretraining_complete",
            f"Mask predictor trained: loss={final_loss:.6f}, epochs={epochs_trained}",
            final_loss=final_loss,
            epochs_trained=epochs_trained,
            converged=converged,
            buffer_size=len(self._buffer)
        )

        return {
            "final_loss": final_loss,
            "epochs_trained": epochs_trained,
            "converged": converged
        }

    @property
    def buffer_size_current(self):
        """Current number of examples in buffer."""
        return len(self._buffer)

    def clear_buffer(self):
        """Clear the training buffer."""
        self._buffer = []
