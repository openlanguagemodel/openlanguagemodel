# src/olm/nn/embeddings/positional/base.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class PositionalEmbeddingBase(nn.Module, ABC):
    """
    Abstract base class for all positional embedding implementations.

    Positional embeddings add information about token positions in a sequence
    to help the model understand order and relative positions. Different positional
    embedding strategies have different properties:

    - Learned (Absolute): Simple, effective, but limited to max_seq_len
    - Sinusoidal: Deterministic, can extrapolate to longer sequences
    - RoPE: Applied to Q/K directly, enables relative position modeling
    - ALiBi: Adds bias to attention scores, excellent extrapolation

    All positional embedding implementations should inherit from this base class
    and implement the forward method.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Apply positional information to input tensor(s).

        The signature and behavior of this method varies by implementation:
        - Some add to embeddings (Absolute, Sinusoidal)
        - Some rotate representations (RoPE)
        - Some return bias to add to attention scores (ALiBi)

        Returns:
            Transformed tensor(s) with positional information applied
        """
        pass

    def extra_repr(self) -> str:
        """
        String representation of the module for debugging.

        Override this in subclasses to provide useful information.
        """
        return ""
