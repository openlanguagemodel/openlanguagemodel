from olm.nn.feedforward.base import FeedForwardBase
from olm.nn.activations import SwiGLU
from olm.nn.torch_nn_wrappers import Linear
import torch.nn as nn


class SwiGLUFFN(FeedForwardBase):
    """
    SwiGLU-based feed-forward network used in modern Transformers (e.g., LLaMA, PaLM).

    This layer implements the gated linear unit with Swish (SiLU) activation, which has been
    shown to improve performance over standard GELU/ReLU FFNs.

    Structure:
        Input
        -> Linear(embed_dim -> 2 * hidden_dim) [Splits into Gate and Value]
        -> SwiGLU(Gate * SiLU(Value))
        -> Linear(hidden_dim -> embed_dim)
        -> Dropout

    Args:
        embed_dim (int): The dimension of the input and output.
        hidden_dim (int, optional): The intermediate inner dimension. 
            If None, defaults to `int(ff_multiplier * embed_dim)`.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
        ff_multiplier (float, optional): Multiplier for default hidden dimension. Defaults to 2.5 (commonly 8/3 for SwiGLU).

    Attributes:
        up_proj (Linear): Projects and splits input into gate and value parts.
        act (SwiGLU): The activation function.
        down_proj (Linear): Projects back to embedding dimension.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.0,
        bias: bool = True,
        ff_multiplier: float = 2.5,
    ):
        super().__init__(embed_dim)

        if hidden_dim is None:
            hidden_dim = int(ff_multiplier * embed_dim)  # modern default

        self.hidden_dim = hidden_dim

        self.up_proj = Linear(
            embed_dim,
            2 * hidden_dim,   # REQUIRED for SwiGLU
            bias=bias
        ) # torch.nn.Linear can also be used

        self.act = SwiGLU()

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