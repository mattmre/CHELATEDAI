from __future__ import annotations

import os
from pathlib import Path
from config import validate_safe_path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - dependency optional for non-training environments
    _TORCH_IMPORT_ERROR = exc

    class _TorchUnavailableError(ImportError):
        """Raised when torch-backed adapters are used without torch installed."""

    def _raise_torch_unavailable():
        raise _TorchUnavailableError(
            "torch is required for chelation adapter operations; install torch for "
            "training and inference with production adapters."
        ) from _TORCH_IMPORT_ERROR

    def _missing_fn(*_args, **_kwargs):
        _raise_torch_unavailable()

    class _MissingTorchBase:
        def __init__(self, *args, **kwargs):
            _raise_torch_unavailable()

    class _MissingTorchFunctional:
        normalize = staticmethod(_missing_fn)
        softmax = staticmethod(_missing_fn)

    def _missing_nn_init(_obj: str, *args, **kwargs):  # pragma: no cover
        _raise_torch_unavailable()

    class _MissingTorchNNNamespace:
        Module = _MissingTorchBase
        Linear = _MissingTorchBase
        ReLU = _MissingTorchBase
        Sequential = _MissingTorchBase
        ModuleList = _MissingTorchBase
        init = _missing_nn_init

        class Parameter(_MissingTorchBase):
            pass

    class _TorchFallback:
        nn = _MissingTorchNNNamespace()
        nn.functional = _MissingTorchFunctional

        def __getattr__(self, _name):
            _raise_torch_unavailable()

        def __call__(self, *args, **kwargs):
            _raise_torch_unavailable()

    torch = _TorchFallback()
    nn = _MissingTorchNNNamespace()
    F = _MissingTorchFunctional()

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

    def regularization_loss(self):
        """Return 0.0 -- MLP adapter has no special regularization term."""
        return 0.0

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
        self._skew_param = nn.Parameter(torch.randn(input_dim, input_dim) * 0.001)
        # Diagonal scaling matrix (DSM): per-dimension learned scale factors.
        # Initialized to ones so the initial transform is still near-identity.
        # Drift-Adapter (EMNLP 2025) showed DSM improves recall recovery from
        # 95-97% to 98-99% by allowing dimension-wise rescaling that a pure
        # orthogonal transform cannot express.
        self._scale = nn.Parameter(torch.ones(input_dim))

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
        # Apply diagonal scaling matrix (DSM) for per-dimension rescaling
        out = out * self._scale
        out = torch.nn.functional.normalize(out, p=2, dim=1)

        if input_was_1d:
            out = out.squeeze(0)
        return out

    def regularization_loss(self):
        """Frobenius norm of the skew-symmetric matrix A = P - P^T.

        Penalizing ||A||_F^2 keeps the Cayley rotation angle small, preventing
        the optimizer from pushing _skew_param entries to large magnitudes and
        degrading NDCG over multiple sedimentation cycles.
        """
        A = self._skew_param - self._skew_param.t()
        return (A ** 2).sum()

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
        # Asymmetric init (LoRA convention, Hu et al. ICLR 2022):
        # V=zero so correction starts at exactly zero (identity behavior),
        # U=random so gradients can flow from the first step.
        self.U = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.zeros(input_dim, rank))
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

    def regularization_loss(self):
        """Return 0.0 -- LowRank adapter has no special regularization term."""
        return 0.0

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


class QuantizationAwareLowRankAdapter(LowRankAffineAdapter):
    """
    Low-rank affine adapter with built-in quantization simulation (QAT-style)
    for self-distillation of chelation corrections.

    Extends LowRankAffineAdapter (LoRA-style asymmetric U/V init) with
    differentiable fake-quantization using Straight-Through Estimator (STE).
    During training (for OPSD-style self-distillation), the output (or delta)
    is passed through a simulated INT8 scalar quantizer; gradients flow through
    as if identity (STE), so the low-rank factors learn corrections that
    survive downstream embedding quantization (Qdrant scalar INT8, packed_graph
    INT8 blocks, etc.).

    This directly addresses the gap: BoundedAdapter + QuantizationPromotionGate
    are post-hoc filters; this makes quant-robustness part of the learned
    objective (synergistic with OPSD/SDPO per-"token"/per-dim clipping and
    filtered on-policy data).

    When used in asymmetric teacher-student distillation:
    - Teacher (privileged diagnostics) can use quant_aware=True to produce
      targets that already account for quant noise.
    - Student learns low-rank delta that matches teacher even after quant sim.

    Compatible with existing BoundedAdapter wrapper (nest it) and
    simulate_int8_quantization from evolution_strategies_optimizer (non-diff
    version for fitness eval).

    Args:
        input_dim: Embedding dim
        rank: Low-rank (default 16, smaller ranks often more quant-stable)
        quant_levels: INT8 levels (127 for symmetric -127..127)
        quant_quantile: Scale estimation quantile (0.99 matches ES simulator)
        apply_quant_to: "output" (after x+delta+norm) or "delta" (before add)
        ste_scale: Whether to learn a per-dim scale inside quant (default False)
    """

    def __init__(
        self,
        input_dim: int,
        rank: int = 16,
        quant_levels: int = 127,
        quant_quantile: float = 0.99,
        apply_quant_to: str = "output",
        ste_scale: bool = False,
    ):
        super().__init__(input_dim=input_dim, rank=rank)
        self.quant_levels = int(quant_levels)
        self.quant_quantile = float(quant_quantile)
        self.apply_quant_to = apply_quant_to  # "output" or "delta"
        self.ste_scale = bool(ste_scale)
        if self.ste_scale:
            # Optional learned per-dim quant scale (multiplicative)
            self.quant_scale_param = nn.Parameter(torch.ones(input_dim))
        else:
            self.register_buffer("quant_scale_param", None)

    def _simulate_quant_ste(self, tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable fake INT8 quantization with Straight-Through Estimator."""
        if not self.training:
            # Inference: use non-diff version for exact parity with storage
            # (import here to avoid circular; in practice engine controls)
            try:
                from evolution_strategies_optimizer import simulate_int8_quantization
                return simulate_int8_quantization(
                    tensor, levels=self.quant_levels, quantile=self.quant_quantile
                )
            except Exception:
                # Fallback pure torch (no quantile, use max)
                pass

        # Training path: STE
        with torch.no_grad():
            abs_vals = tensor.detach().abs().reshape(-1)
            if abs_vals.numel() == 0:
                scale = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
            else:
                scale_ref = torch.quantile(abs_vals, self.quant_quantile)
                scale_ref = torch.clamp(scale_ref, min=1e-12)
                scale = scale_ref / float(self.quant_levels)

        # Quantize
        q = torch.clamp(torch.round(tensor / scale), -self.quant_levels, self.quant_levels)
        dequant = q * scale

        # STE: forward dequant, backward as identity
        out = (dequant - tensor).detach() + tensor

        if self.ste_scale and self.quant_scale_param is not None:
            out = out * self.quant_scale_param
        return out

    def forward(self, x):
        # Get the low-rank base output (normalized)
        base_out = super().forward(x)  # calls LowRankAffineAdapter.forward -> x + delta, normalize

        if self.apply_quant_to == "delta":
            # Recompute delta for quant application (more targeted)
            # Note: for simplicity in STE variant we quant the final; override if needed
            input_was_1d = (x.dim() == 1)
            if input_was_1d:
                x2 = x.unsqueeze(0)
            else:
                x2 = x
            x_norm = F.normalize(x2, p=2, dim=1)
            delta = base_out.unsqueeze(0) - x_norm if base_out.dim() == 1 else base_out - x_norm
            q_delta = self._simulate_quant_ste(delta)
            out = x_norm + q_delta
            out = F.normalize(out, p=2, dim=1)
            if input_was_1d:
                out = out.squeeze(0)
            return out
        else:
            # Default: quantize the final normalized output (what gets stored/retrieved)
            q_out = self._simulate_quant_ste(base_out)
            norm_dim = 0 if q_out.dim() == 1 else 1
            return F.normalize(q_out, p=2, dim=norm_dim)

    def regularization_loss(self):
        base_reg = super().regularization_loss()
        # Extra penalty on quant_scale deviation if enabled (keeps conservative)
        if self.ste_scale and self.quant_scale_param is not None:
            scale_reg = ((self.quant_scale_param - 1.0) ** 2).mean()
            return base_reg + 0.0005 * scale_reg
        return base_reg


class BoundedAdapter(nn.Module):
    """Wrapper that bounds correction magnitude and adds per-dimension scaling.

    Addresses INT8 quantization noise floor (~0.0078) by ensuring corrections
    are large enough to survive quantization, while capping them to prevent
    divergence.

    Wraps any base adapter (MLP, Procrustes, Low-rank) uniformly.

    Args:
        base_adapter: Any adapter nn.Module with .input_dim and .forward()
        min_correction: Minimum L2 norm for correction delta (default 0.01,
            above INT8 noise floor of ~0.0078)
        max_correction: Maximum L2 norm for correction delta (default 0.5,
            prevents divergence)
    """
    def __init__(self, base_adapter, min_correction=0.01, max_correction=0.5):
        super().__init__()
        self.base_adapter = base_adapter
        self.input_dim = base_adapter.input_dim
        self.min_correction = min_correction
        self.max_correction = max_correction
        # Per-dimension learned scaling (initialized to 1.0)
        self.dim_scale = nn.Parameter(torch.ones(self.input_dim))

    def forward(self, x):
        # Get base adapter output
        base_out = self.base_adapter(x)

        # Handle 1D/2D input
        input_was_1d = (x.dim() == 1)
        if input_was_1d:
            x_2d = x.unsqueeze(0)
            base_out_2d = base_out.unsqueeze(0)
        else:
            x_2d = x
            base_out_2d = base_out if base_out.dim() == 2 else base_out.unsqueeze(0)

        # Compute correction delta (what the adapter changed)
        # Base adapters normalize output, so work in normalized space
        x_norm = F.normalize(x_2d, p=2, dim=1)
        delta = base_out_2d - x_norm

        # Apply per-dimension scaling
        delta = delta * self.dim_scale

        # Compute correction magnitude per vector
        correction_norm = torch.norm(delta, dim=1, keepdim=True)

        # Scale corrections to be within [min_correction, max_correction]
        # If correction is too small, scale UP (above INT8 noise floor)
        # If correction is too large, scale DOWN (prevent divergence)
        scale = torch.ones_like(correction_norm)
        too_small = correction_norm < self.min_correction
        too_large = correction_norm > self.max_correction
        # Only rescale non-zero corrections
        nonzero = correction_norm > 1e-10

        scale = torch.where(
            too_small & nonzero,
            self.min_correction / correction_norm.clamp(min=1e-10),
            scale
        )
        scale = torch.where(
            too_large,
            self.max_correction / correction_norm.clamp(min=1e-10),
            scale
        )

        delta = delta * scale

        # Apply bounded correction
        out = x_norm + delta
        out = F.normalize(out, p=2, dim=1)

        if input_was_1d:
            out = out.squeeze(0)
        return out

    def regularization_loss(self):
        """Delegate to base adapter + L2 on dim_scale deviation from 1.0."""
        base_reg = self.base_adapter.regularization_loss()
        # Penalize dim_scale deviation from 1.0 (keeps scaling conservative)
        scale_reg = ((self.dim_scale - 1.0) ** 2).mean()
        return base_reg + 0.001 * scale_reg

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


class BlockAttnResAdapter(nn.Module):
    """
    Block Attention Residual adapter.

    Implements the Block-AttnRes design from MoonshotAI's "Attention Residuals"
    paper (2025). Instead of a single x + delta residual, runs `num_blocks`
    sequential correction blocks (standard within-block residuals) then uses
    a learned softmax cross-block attention to aggregate all block outputs.

    The attention selects which correction depth is most useful per input,
    preventing single-shot over-correction and bounding output magnitude — the
    same two pathologies AttnRes addresses in deep transformer stacks.

    Args:
        input_dim: Embedding dimension.
        num_blocks: Number of correction blocks (default 4; paper uses ~8).
        proj_dim: Key/query projection dimension (default input_dim // 4, min 32).
    """

    def __init__(self, input_dim: int, num_blocks: int = 4, proj_dim: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        proj = proj_dim if proj_dim is not None else max(input_dim // 4, 32)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim),
            )
            for _ in range(num_blocks)
        ])
        self.query_proj = nn.Linear(input_dim, proj)
        self.key_proj = nn.Linear(input_dim, proj)
        self._attn_scale = proj ** -0.5

        for block in self.blocks:
            nn.init.normal_(block[0].weight, std=0.001)
            nn.init.zeros_(block[0].bias)
            nn.init.normal_(block[2].weight, std=0.001)
            nn.init.zeros_(block[2].bias)
        nn.init.normal_(self.query_proj.weight, std=0.001)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.normal_(self.key_proj.weight, std=0.001)
        nn.init.zeros_(self.key_proj.bias)

    def forward(self, x):
        if x.dim() == 0 or x.dim() > 2:
            raise ValueError(
                f"BlockAttnResAdapter expects 1D or 2D input, got {x.dim()}D tensor with shape {x.shape}"
            )
        input_was_1d = (x.dim() == 1)
        if input_was_1d:
            x = x.unsqueeze(0)

        # Within-block residuals; collect all states including input
        block_states = [x]
        h = x
        for block in self.blocks:
            h = h + block(h)
            block_states.append(h)

        # Cross-block attention: final block state as query, all states as keys/values
        states = torch.stack(block_states, dim=1)                                   # [B, N+1, d]
        query = self.query_proj(h)                                                  # [B, proj]
        keys = self.key_proj(states)                                                # [B, N+1, proj]
        attn = (query.unsqueeze(1) @ keys.transpose(-2, -1)) * self._attn_scale    # [B, 1, N+1]
        attn = F.softmax(attn, dim=-1)
        out = (attn @ states).squeeze(1)                                            # [B, d]

        out = F.normalize(out, p=2, dim=1)
        if input_was_1d:
            out = out.squeeze(0)
        return out

    def regularization_loss(self):
        return 0.0

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


class LayerAttentionAggregator(nn.Module):
    """
    Learned cross-layer attention aggregation for transformer embeddings.

    Applies the AttnRes cross-block aggregation concept at the transformer level:
    instead of using only the last (or averaged) layer output, learns a softmax
    attention over all captured layer embeddings and returns a weighted combination.

    Intended for use with mean-pooled per-layer hidden states captured via
    ModelHookBus or equivalent layer-output extraction.

    Args:
        hidden_size: Transformer hidden dimension.
        proj_dim: Attention projection dimension (default hidden_size // 8, min 32).
    """

    def __init__(self, hidden_size: int, proj_dim: int | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        proj = proj_dim if proj_dim is not None else max(hidden_size // 8, 32)
        self.query_proj = nn.Linear(hidden_size, proj)
        self.key_proj = nn.Linear(hidden_size, proj)
        self._attn_scale = proj ** -0.5

        nn.init.normal_(self.query_proj.weight, std=0.001)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.normal_(self.key_proj.weight, std=0.001)
        nn.init.zeros_(self.key_proj.bias)

    def forward(self, layer_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate mean-pooled layer embeddings via learned attention.

        Args:
            layer_embeddings: [batch, num_layers, hidden_size] — mean-pooled
                              per-layer hidden states.

        Returns:
            [batch, hidden_size] — L2-normalized attention-weighted aggregation.
        """
        if layer_embeddings.dim() != 3 or layer_embeddings.size(-1) != self.hidden_size:
            raise ValueError(
                f"LayerAttentionAggregator expects [B, L, {self.hidden_size}] input, "
                f"got {tuple(layer_embeddings.shape)}"
            )
        # Mean of all layers as a neutral query baseline
        query = self.query_proj(layer_embeddings.mean(dim=1))       # [B, proj]
        keys = self.key_proj(layer_embeddings)                       # [B, L, proj]
        attn = (query.unsqueeze(1) @ keys.transpose(-2, -1)) * self._attn_scale  # [B, 1, L]
        attn = F.softmax(attn, dim=-1)
        out = (attn @ layer_embeddings).squeeze(1)                   # [B, d]
        return F.normalize(out, p=2, dim=1)


def create_adapter(adapter_type="mlp", input_dim=768, bounded=False,
                   min_correction=0.01, max_correction=0.5, **kwargs):
    """
    Factory function to create adapter instances by type name.

    Args:
        adapter_type: One of "mlp", "procrustes", "low_rank", "quant_low_rank", "attnres"
        input_dim: Embedding dimension
        bounded: If True, wrap in BoundedAdapter for quantization-safe corrections
                 (can be nested with quant_low_rank for double protection)
        min_correction: Minimum correction norm (BoundedAdapter only, default 0.01)
        max_correction: Maximum correction norm (BoundedAdapter only, default 0.5)
        **kwargs: Additional args passed to adapter constructor
            - For "mlp": hidden_dim (optional)
            - For "low_rank", "quant_low_rank": rank (default 16),
              quant_levels (127), quant_quantile (0.99), apply_quant_to ("output"),
              ste_scale (False)
            - For "attnres": num_blocks (default 4), proj_dim (optional)

    Returns:
        nn.Module: Adapter instance (optionally wrapped in BoundedAdapter)

    Raises:
        ValueError: If adapter_type is unknown
    """
    if adapter_type == "mlp":
        kwargs.pop("rank", None)
        kwargs.pop("num_blocks", None)
        kwargs.pop("proj_dim", None)
        kwargs.pop("quant_levels", None)
        kwargs.pop("quant_quantile", None)
        adapter = ChelationAdapter(input_dim=input_dim, **kwargs)
    elif adapter_type == "procrustes":
        kwargs.pop("rank", None)
        kwargs.pop("num_blocks", None)
        kwargs.pop("proj_dim", None)
        kwargs.pop("quant_levels", None)
        kwargs.pop("quant_quantile", None)
        adapter = OrthogonalProcrustesAdapter(input_dim=input_dim)
    elif adapter_type == "low_rank":
        kwargs.pop("num_blocks", None)
        kwargs.pop("proj_dim", None)
        kwargs.pop("quant_levels", None)
        kwargs.pop("quant_quantile", None)
        rank = kwargs.get("rank", 16)
        adapter = LowRankAffineAdapter(input_dim=input_dim, rank=rank)
    elif adapter_type == "quant_low_rank":
        # NEW for OPSD Loop 1 Agent 8: quantization-aware low-rank variant
        kwargs.pop("num_blocks", None)
        kwargs.pop("proj_dim", None)
        rank = kwargs.pop("rank", 16)
        quant_levels = kwargs.pop("quant_levels", 127)
        quant_quantile = kwargs.pop("quant_quantile", 0.99)
        apply_quant_to = kwargs.pop("apply_quant_to", "output")
        ste_scale = kwargs.pop("ste_scale", False)
        adapter = QuantizationAwareLowRankAdapter(
            input_dim=input_dim,
            rank=rank,
            quant_levels=quant_levels,
            quant_quantile=quant_quantile,
            apply_quant_to=apply_quant_to,
            ste_scale=ste_scale,
        )
    elif adapter_type == "attnres":
        num_blocks = kwargs.get("num_blocks", 4)
        proj_dim = kwargs.get("proj_dim", None)
        kwargs.pop("rank", None)
        kwargs.pop("quant_levels", None)
        kwargs.pop("quant_quantile", None)
        adapter = BlockAttnResAdapter(input_dim=input_dim, num_blocks=num_blocks, proj_dim=proj_dim)
    else:
        valid = ["mlp", "procrustes", "low_rank", "quant_low_rank", "attnres"]
        raise ValueError(f"Unknown adapter_type '{adapter_type}'. Valid types: {valid}")

    if bounded:
        adapter = BoundedAdapter(adapter, min_correction=min_correction,
                                 max_correction=max_correction)
    return adapter
