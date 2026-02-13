import torch
import torch.nn as nn
import torch.optim as optim
import os

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
        # x is [batch_size, input_dim]
        delta = self.correction_net(x)
        
        # Apply corruption/correction
        out = x + delta
        
        # Normalize to hypersphere (Cosine Similarity relies on this)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        if os.path.exists(path):
            try:
                self.load_state_dict(torch.load(path, weights_only=True))
                return True
            except RuntimeError as e:
                print(f"Warning: Failed to load adapter weights (Dimension Mismatch?): {e}")
                return False
        return False
