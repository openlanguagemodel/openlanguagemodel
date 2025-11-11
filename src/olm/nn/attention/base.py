import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AttentionBase(nn.Module, ABC):
    """
    Abstract base class for attention mechanisms.
    All attention variants (multi-head, linear, etc.)
    should inherit from this and implement `compute_attention`.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        # Shared QKV projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Shared output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @abstractmethod
    def compute_attention(self, q, k, v, mask=None):
        """Each subclass implements its own attention mechanism."""
        pass

    def forward(self, x, mask=None):
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        out = self.compute_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)