from olm.nn.feedforward.base import FeedForwardBase
from olm.nn.activations.geglu import GeGLU
from olm.nn.torch_nn_wrappers import Linear
import torch.nn as nn


class GeGLUFFN(FeedForwardBase):
    """
    Feed-Forward Network using GeGLU activation.

    Implements: x = DownProj(GeGLU(UpProj(x))).
    UpProj expands to 2 * hidden_dim to support splitting for the gate.

    Args:
        embed_dim (int): Input dimension.
        hidden_dim (int, optional): Hidden dimension. Defaults to 4 * embed_dim if None.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        bias (bool, optional): Whether to usage bias in linear layers. Defaults to True.
        ff_multiplier (float, optional): Expansion factor if hidden_dim is None. Defaults to 4.0.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.0,
        bias: bool = True,
        ff_multiplier: float = 4.0, 
    ):
        super().__init__(embed_dim)

        if hidden_dim is None:
            hidden_dim = int(ff_multiplier * embed_dim)

        self.hidden_dim = hidden_dim

        self.up_proj = Linear(
            embed_dim,
            2 * hidden_dim,
            bias=bias
        ) # torch.nn.Linear can also be used

        self.act = GeGLU()

        self.down_proj = Linear(
            hidden_dim,
            embed_dim,
            bias=bias
        ) # torch.nn.Linear can also be used

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x
