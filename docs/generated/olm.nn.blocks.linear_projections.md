# `olm.nn.blocks.linear_projections`

## Classes

### `QKVProjection(dim_in: int, dim_q: int, dim_k: int, dim_v: int, bias: bool = True, init: str = 'xavier')`

Computes Query, Key, and Value projections for attention mechanisms.

Applies three separate linear transformations to the input to generate Q, K, and V tensors.
Supports various weight initialization schemes.

Attributes:
    W_q (Linear): Linear layer for Query projection.
    W_k (Linear): Linear layer for Key projection.
    W_v (Linear): Linear layer for Value projection.

#### Methods

- `forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
  Performs the Q, K, V projections.
