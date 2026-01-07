import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class FeedForwardBase(nn.Module, ABC):
    """
    Abstract base class for feedforward networks in a transformer block.

    Defines the interface for FFNs/MLPs. Subclasses must implement the `forward` method.

    Attributes:
        embed_dim (int): The input and output dimension.
    """
    def __init__(self, embed_dim: int, **kwargs):
        """
        Initializes the FeedForwardBase.

        Args:
            embed_dim (int): The input and output dimension.
            **kwargs: Additional arguments for subclasses.
        """
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feedforward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, embed_dim).
        """
        pass
