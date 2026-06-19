# `olm.nn.feedforward.classic_ffn`

## Classes

### `ClassicFFN(embed_dim, hidden_dim=None, activation_fn=None, dropout=0.0, bias=True)`

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

#### Methods

- `forward(self, x)`
  Forward pass of the feedforward network.
