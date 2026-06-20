# `olm.nn.blocks.linear_projections`

Source: [`src/olm/nn/blocks/linear_projections.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/blocks/linear_projections.py#L1)

## Classes

### `QKVProjection(dim_in: int, dim_q: int, dim_k: int, dim_v: int, bias: bool = True, init: str = 'xavier')`

**Bases:** `Module`

Source: [`src/olm/nn/blocks/linear_projections.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/blocks/linear_projections.py#L6)

Computes Query, Key, and Value projections for attention mechanisms.

Applies three separate linear transformations to the input to generate Q, K, and V tensors.
Supports various weight initialization schemes.

**Forward**

Accepts ``x`` with shape ``[batch, seq_len, dim_in]`` and returns
``(q, k, v)`` with shapes ``[batch, seq_len, dim_q]``,
``[batch, seq_len, dim_k]``, and ``[batch, seq_len, dim_v]``.

**Attributes**

- `W_q` (`Linear`): Linear layer for Query projection.
- `W_k` (`Linear`): Linear layer for Key projection.
- `W_v` (`Linear`): Linear layer for Value projection.

#### Methods

##### `forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

Source: [`src/olm/nn/blocks/linear_projections.py:61`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/blocks/linear_projections.py#L61)

Performs the Q, K, V projections.

**Parameters**

- `x` (`torch.Tensor`): Input tensor of shape (batch, seq_len, dim_in).

**Returns**

tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing (Q, K, V) tensors.
