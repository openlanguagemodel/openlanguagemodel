from torch import nn
import torch
from abc import ABC, abstractmethod

class NormBase(nn.Module, ABC):
    """
    Abstract base class for normalization layers (e.g., LayerNorm, RMSNorm).

    Standardizes the interface for normalization, ensuring all implementations
    handle model dimension, device, and dtype consistently.

    Attributes:
        d_model (int): The dimension of the input features to normalize.
        device (torch.device, optional): The device the module is on.
        dtype (torch.dtype): The data type of the module parameters.
    """
    def __init__(self, d_model: int, device=None, dtype=None):
        """
        Initializes the NormBase.

        Args:
            d_model (int): The feature dimension.
            device (torch.device, optional): Target device. Defaults to None.
            dtype (torch.dtype, optional): Target data type. Defaults to torch.float32.
        """
        super().__init__()
        if dtype is None: dtype = torch.float32
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to the input tensor."""
        pass
