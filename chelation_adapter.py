import torch
import torch.nn as nn
import torch.optim as optim
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
