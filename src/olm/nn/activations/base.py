from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class ActivationBase(nn.Module, ABC):
    """Base class for all activation modules."""

    def __init__(self, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        if dtype is None:
            dtype = torch.float32
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation to ``x``."""
        raise NotImplementedError
