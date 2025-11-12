from torch import nn
import torch
from abc import ABC, abstractmethod

class NormBase(nn.Module, ABC):
    """
    Base class for all normalization layers.
    """
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        if dtype is None: dtype = torch.float32
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to the input tensor."""
        pass
