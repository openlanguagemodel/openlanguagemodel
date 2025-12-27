import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class FeedForwardBase(nn.Module, ABC):
    """
    Abstract base class for feedforward networks in a transformer block.
    All feedforward variants (MLP, GatedMLP, etc.)
    should inherit from this and implement `forward_hidden`.
    """
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the feedforward network.
        x: (batch, seq_len, embed_dim)
        Should return tensor of shape (batch, seq_len, embed_dim).
        """
        pass
