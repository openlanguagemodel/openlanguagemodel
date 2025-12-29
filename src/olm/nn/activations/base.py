from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class ActivationBase(nn.Module, ABC):
    """
    Abstract base class for all activation functions.

    Ensures a consistent interface for activation layers, handling device and dtype
    initialization. Subclasses must implement the `forward` method.

    Attributes:
        device (torch.device, optional): The device the module is on.
        dtype (torch.dtype): The data type of the module parameters.
    """

    def __init__(self, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        """
        Initializes the ActivationBase.

        Args:
            device (torch.device, optional): Target device. Defaults to None.
            dtype (torch.dtype, optional): Target data type. Defaults to torch.float32.
        """
        super().__init__()
        if dtype is None:
            dtype = torch.float32
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation to ``x``."""
        raise NotImplementedError
