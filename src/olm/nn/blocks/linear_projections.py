import torch
import torch.nn as nn
from olm.nn.torch_nn_wrappers import Linear
from typing import Tuple, Optional

class QKVProjection(nn.Module):
    """
    Computes Query, Key, and Value projections for attention mechanisms.

    Applies three separate linear transformations to the input to generate Q, K, and V tensors.
    Supports various weight initialization schemes.

    Attributes:
        W_q (Linear): Linear layer for Query projection.
        W_k (Linear): Linear layer for Key projection.
        W_v (Linear): Linear layer for Value projection.
    """

    def __init__(self, dim_in: int, dim_q: int, dim_k: int, dim_v: int, bias: bool = True, init: str = "xavier"):
        """
        Initializes the QKVProjection.

        Args:
            dim_in (int): Input dimension.
            dim_q (int): Output dimension for Query.
            dim_k (int): Output dimension for Key.
            dim_v (int): Output dimension for Value.
            bias (bool, optional): Whether to include bias terms. Defaults to True.
            init (str, optional): Initialization method ('xavier', 'kaiming', 'normal'). Defaults to "xavier".

        Raises:
            ValueError: If an unknown initialization method is provided.
        """
        super().__init__()

        self.W_q = Linear(dim_in, dim_q, bias=bias) # torch.nn.Linear can also be used
        self.W_k = Linear(dim_in, dim_k, bias=bias) # torch.nn.Linear can also be used
        self.W_v = Linear(dim_in, dim_v, bias=bias) # torch.nn.Linear can also be used

        layers = [self.W_q, self.W_k, self.W_v]

        # optional initialization
        for layer in layers:
            if init == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif init == "kaiming":
                nn.init.kaiming_uniform_(layer.weight)
            elif init == "normal":
                nn.init.normal_(layer.weight, std=0.02)
            else:
                raise ValueError(f"Unknown init: {init}")

            if bias:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the Q, K, V projections.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim_in).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing (Q, K, V) tensors.
        """

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        return Q, K, V
