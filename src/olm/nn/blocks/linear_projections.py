import torch
import torch.nn as nn


class QKVProjection(nn.Module):
    """
    Computes Query, Key, and Value projections for attention mechanisms.

    Applies three separate linear transformations to the input to generate Q, K, and V tensors.
    Supports various weight initialization schemes.

    Attributes:
        W_q (nn.Linear): Linear layer for Query projection.
        W_k (nn.Linear): Linear layer for Key projection.
        W_v (nn.Linear): Linear layer for Value projection.
    """

    def __init__(self, dim_in, dim_q, dim_k, dim_v, bias=True, init="xavier"):
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

        self.W_q = nn.Linear(dim_in, dim_q, bias=bias)
        self.W_k = nn.Linear(dim_in, dim_k, bias=bias)
        self.W_v = nn.Linear(dim_in, dim_v, bias=bias)

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

    def forward(self, x):
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