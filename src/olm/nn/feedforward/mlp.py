import torch
import torch.nn as nn
from .base import FeedForwardBase

class MLP(FeedForwardBase):
    """
    Standard MLP (Multi-Layer Perceptron) for Transformer blocks.
    Structure: Linear(embed_dim -> hidden_dim) -> Activation -> Dropout -> Linear(hidden_dim -> embed_dim) -> Dropout
    """
    def __init__(self, embed_dim, hidden_dim=None, activation_fn=nn.GELU(), dropout=0.0, bias=True):
        super().__init__(embed_dim)
        
        # Default hidden_dim to 4 * embed_dim if not provided
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim
            
        self.hidden_dim = hidden_dim
        
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.act = activation_fn
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x
