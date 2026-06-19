# `olm.nn.feedforward.base`

## Classes

### `FeedForwardBase(embed_dim: int, **kwargs)`

Abstract base class for feedforward networks in a transformer block.

Defines the interface for FFNs/MLPs. Subclasses must implement the `forward` method.

Attributes:
    embed_dim (int): The input and output dimension.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Forward pass of the feedforward network.
