import torch
import torch.nn as nn
from olm.nn.torch_nn_wrappers import Linear
from olm.nn.feedforward.base import FeedForwardBase
from olm.nn.activations import GELU

class ClassicFFN(FeedForwardBase):
    """
    Standard Multi-Layer Perceptron (MLP) used in Transformer blocks.

    Implements a position-wise feed-forward network consisting of two linear transformations
    with a non-linear activation function in between.

    Structure:
        Input -> Linear(embed_dim -> hidden_dim) -> Activation -> Dropout -> Linear(hidden_dim -> embed_dim) -> Dropout

    Attributes:
        hidden_dim (int): Dimension of the inner hidden layer.
        up_proj (Linear): Projection from embedding dim to hidden dim.
        act (nn.Module): Activation function.
        down_proj (Linear): Projection from hidden dim to embedding dim.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, embed_dim, hidden_dim=None, activation_fn=GELU(), dropout=0.0, bias=True):
        """
        Initializes the MLP.

        Args:
            embed_dim (int): The input and output dimension.
            hidden_dim (int, optional): The inner dimension. Defaults to 4 * embed_dim.
            activation_fn (nn.Module, optional): Activation function. Defaults to GELU().
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
        """
        super().__init__(embed_dim)
        
        # Default hidden_dim to 4 * embed_dim if not provided
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim
            
        self.hidden_dim = hidden_dim
        
        self.up_proj = Linear(embed_dim, hidden_dim, bias=bias) # torch.nn.Linear can also be used
        self.act = activation_fn
        self.down_proj = Linear(hidden_dim, embed_dim, bias=bias) # torch.nn.Linear can also be used
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x
