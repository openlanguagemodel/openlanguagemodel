import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseCombinator(nn.Module, ABC):
    """
    Abstract base class for combinator modules.

    Subclasses implement ``forward`` to define how inputs are combined.
    """
    def __init__(self):
        """Initialize the combinator base."""
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the combinator output from an input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor produced by the combinator.
        """
        pass
