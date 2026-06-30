# `olm.nn.feedforward.base`

Source: [`src/olm/nn/feedforward/base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/base.py#L1)

## Classes

### `FeedForwardBase(embed_dim: int, **kwargs)`

**Bases:** `Module`, `ABC`

Source: [`src/olm/nn/feedforward/base.py:5`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/base.py#L5)

Abstract base class for feedforward networks in a transformer block.

Defines the interface for FFNs/MLPs. Subclasses must implement the `forward` method.

**Attributes**

- `embed_dim` (`int`): The input and output dimension.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/feedforward/base.py:25`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/base.py#L25)

Forward pass of the feedforward network.

**Parameters**

- `x` (`torch.Tensor`): Input tensor of shape (batch, seq_len, embed_dim).

**Returns**

- `torch.Tensor`: Output tensor of shape (batch, seq_len, embed_dim).
