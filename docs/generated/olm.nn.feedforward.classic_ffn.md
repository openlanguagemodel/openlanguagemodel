# `olm.nn.feedforward.classic_ffn`

Source: [`src/olm/nn/feedforward/classic_ffn.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/classic_ffn.py#L1)

## Classes

### `ClassicFFN(embed_dim, hidden_dim=None, activation_fn=None, dropout=0.0, bias=True)`

**Bases:** `olm.nn.feedforward.base.FeedForwardBase`

Source: [`src/olm/nn/feedforward/classic_ffn.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/classic_ffn.py#L7)

Standard Multi-Layer Perceptron (MLP) used in Transformer blocks.

Implements a position-wise feed-forward network consisting of two linear transformations
with a non-linear activation function in between.

**Structure**

Input -> Linear(embed_dim -> hidden_dim) -> Activation -> Dropout -> Linear(hidden_dim -> embed_dim) -> Dropout

**Attributes**

- `hidden_dim` (`int`): Dimension of the inner hidden layer.
- `up_proj` (`Linear`): Projection from embedding dim to hidden dim.
- `act` (`nn.Module`): Activation function.
- `down_proj` (`Linear`): Projection from hidden dim to embedding dim.
- `dropout` (`nn.Dropout`): Dropout layer.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/feedforward/classic_ffn.py:51`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/classic_ffn.py#L51)

Apply the position-wise feed-forward network.

**Parameters**

- `x` (`torch.Tensor`): Hidden states shaped ``[batch, seq_len, embed_dim]``.

**Returns**

- `torch.Tensor`: Hidden states shaped ``[batch, seq_len, embed_dim]``.
