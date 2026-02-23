import torch
import torch.nn as nn
import os
from pathlib import Path
from config import validate_safe_path

class ChelationAdapter(nn.Module):
    """
    A lightweight residual adapter that learns to 'chelate' (correct) embeddings.
    Structure:
    Input -> [Linear -> ReLU -> Linear] -> Residual Add -> Normalize -> Output
    
    This ensures that at initialization (or with 0 weights), it acts as an identity function,
    preserving the original model's strong baseline.
    """
    def __init__(self, input_dim, hidden_dim=None):
        super(ChelationAdapter, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
            
        self.input_dim = input_dim
        
        # We use a residual structure: f(x) = x + g(x)
        # g(x) is the correction term.
        self.correction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights to be very small so we start close to Identity
        # This prevents "catastrophic forgetting" of the base model's knowledge immediately
        nn.init.normal_(self.correction_net[0].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.correction_net[0].bias)
        nn.init.normal_(self.correction_net[2].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.correction_net[2].bias)

    def forward(self, x):
        # Handle input of various ranks
        if x.dim() == 0 or x.dim() > 2:
            raise ValueError(f"ChelationAdapter expects 1D or 2D input, got {x.dim()}D tensor with shape {x.shape}")
        
        # Track if input was 1D for output reshaping
        input_was_1d = (x.dim() == 1)
        
        # Promote 1D to 2D: [dim] -> [1, dim]
        if input_was_1d:
            x = x.unsqueeze(0)
        
        # x is now [batch_size, input_dim]
        delta = self.correction_net(x)
        
        # Apply corruption/correction
        out = x + delta
        
        # Normalize to hypersphere (Cosine Similarity relies on this)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        
        # Restore original rank: [1, dim] -> [dim] if input was 1D
        if input_was_1d:
            out = out.squeeze(0)
        
        return out
    
    def save(self, path):
        # Validate path for traversal attacks
        path = validate_safe_path(Path(path))
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        # Validate path for traversal attacks
        path = validate_safe_path(Path(path))
        if os.path.exists(path):
            try:
                self.load_state_dict(torch.load(path, weights_only=True))
                return True
            except RuntimeError as e:
                print(f"Warning: Failed to load adapter weights (Dimension Mismatch?): {e}")
                return False
        return False


class OrthogonalProcrustesAdapter(nn.Module):
    """
    Orthogonal Procrustes adapter using Cayley parameterization.

    Learns an orthogonal transformation W = (I - A)(I + A)^{-1} where A is skew-symmetric.
    This ensures W^T W = I at all times, preserving vector norms.

    Inspired by Drift-Adapter (arXiv:2509.23471, EMNLP 2025).
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Skew-symmetric parameter: only upper triangle needed
        # Initialize near zero for near-identity start
        self._skew_param = nn.Parameter(torch.zeros(input_dim, input_dim) * 0.001)

    def _get_orthogonal_matrix(self):
        """Compute orthogonal matrix via Cayley transform of skew-symmetric matrix."""
        # Make skew-symmetric: A = P - P^T
        A = self._skew_param - self._skew_param.t()
        I = torch.eye(self.input_dim, device=A.device, dtype=A.dtype)  # noqa: E741
        # Cayley transform: W = (I - A)(I + A)^{-1}
        W = torch.linalg.solve(I + A, I - A)
        return W

    def forward(self, x):
        if x.dim() == 0 or x.dim() > 2:
            raise ValueError(f"OrthogonalProcrustesAdapter expects 1D or 2D input, got {x.dim()}D")
        input_was_1d = (x.dim() == 1)
        if input_was_1d:
            x = x.unsqueeze(0)

        W = self._get_orthogonal_matrix()
        out = x @ W.t()
        out = torch.nn.functional.normalize(out, p=2, dim=1)

        if input_was_1d:
            out = out.squeeze(0)
        return out

    def save(self, path):
        path = validate_safe_path(Path(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = validate_safe_path(Path(path))
        if os.path.exists(path):
            try:
                self.load_state_dict(torch.load(path, weights_only=True))
                return True
            except RuntimeError:
                return False
        return False


class LowRankAffineAdapter(nn.Module):
    """
    Low-rank affine adapter: out = x + x @ U @ V^T + b

    Rank-constrained correction with fewer parameters than full MLP.
    Inspired by adapter methods in Drift-Adapter (arXiv:2509.23471).

    Args:
        input_dim: Embedding dimension
        rank: Rank of the low-rank decomposition (default: 16)
    """
    def __init__(self, input_dim, rank=16):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        # Low-rank factors: U is (input_dim, rank), V is (input_dim, rank)
        # x @ U @ V^T gives (batch, input_dim)
        self.U = nn.Parameter(torch.randn(input_dim, rank) * 0.001)
        self.V = nn.Parameter(torch.randn(input_dim, rank) * 0.001)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        if x.dim() == 0 or x.dim() > 2:
            raise ValueError(f"LowRankAffineAdapter expects 1D or 2D input, got {x.dim()}D")
        input_was_1d = (x.dim() == 1)
        if input_was_1d:
            x = x.unsqueeze(0)

        # Low-rank correction: delta = x @ U @ V^T + b
        delta = (x @ self.U) @ self.V.t() + self.bias
        out = x + delta
        out = torch.nn.functional.normalize(out, p=2, dim=1)

        if input_was_1d:
            out = out.squeeze(0)
        return out

    def save(self, path):
        path = validate_safe_path(Path(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = validate_safe_path(Path(path))
        if os.path.exists(path):
            try:
                self.load_state_dict(torch.load(path, weights_only=True))
                return True
            except RuntimeError:
                return False
        return False


def create_adapter(adapter_type="mlp", input_dim=768, **kwargs):
    """
    Factory function to create adapter instances by type name.

    Args:
        adapter_type: One of "mlp", "procrustes", "low_rank"
        input_dim: Embedding dimension
        **kwargs: Additional args passed to adapter constructor
            - For "mlp": hidden_dim (optional)
            - For "low_rank": rank (default 16)

    Returns:
        nn.Module: Adapter instance

    Raises:
        ValueError: If adapter_type is unknown
    """
    if adapter_type == "mlp":
        kwargs.pop("rank", None)
        return ChelationAdapter(input_dim=input_dim, **kwargs)
    elif adapter_type == "procrustes":
        kwargs.pop("rank", None)
        return OrthogonalProcrustesAdapter(input_dim=input_dim)
    elif adapter_type == "low_rank":
        rank = kwargs.get("rank", 16)
        return LowRankAffineAdapter(input_dim=input_dim, rank=rank)
    else:
        valid = ["mlp", "procrustes", "low_rank"]
        raise ValueError(f"Unknown adapter_type '{adapter_type}'. Valid types: {valid}")
