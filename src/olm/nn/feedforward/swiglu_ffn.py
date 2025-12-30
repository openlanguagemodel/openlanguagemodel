from .base import FeedForwardBase
from olm.nn.activations import SwiGLU
import torch.nn as nn

class SwiGLUFFN(FeedForwardBase):
    """
    SwiGLU-based feed-forward network used in modern Transformers.

    Structure:
        Input
        → Linear(embed_dim → 2 * hidden_dim)
        → SwiGLU
        → Linear(hidden_dim → embed_dim)
        → Dropout
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim=None,
        dropout=0.0,
        bias=True,
        ff_multiplier=2.5,
    ):
        super().__init__(embed_dim)

        if hidden_dim is None:
            hidden_dim = int(ff_multiplier * embed_dim)  # modern default

        self.hidden_dim = hidden_dim

        self.up_proj = nn.Linear(
            embed_dim,
            2 * hidden_dim,   # REQUIRED for SwiGLU
            bias=bias
        )

        self.act = SwiGLU()

        self.down_proj = nn.Linear(
            hidden_dim,
            embed_dim,
            bias=bias
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x