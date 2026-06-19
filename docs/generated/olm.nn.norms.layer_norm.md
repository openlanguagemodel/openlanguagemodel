# `olm.nn.norms.layer_norm`

Source: [`src/olm/nn/norms/layer_norm.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/norms/layer_norm.py#L1)

## Classes

### `LayerNorm(d_model: int, eps: float = 1e-05, elementwise_affine: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`

**Bases:** `olm.nn.norms.base.NormBase`

Source: [`src/olm/nn/norms/layer_norm.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/norms/layer_norm.py#L7)

Layer Normalization layer.

Implements Layer Normalization as described in "Layer Normalization" (https://arxiv.org/abs/1607.06450).
Normalizes the input across the features dimension.

Args:
    d_model (int): The dimension of the model to normalize.
    eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
    device (torch.device, optional): Target device.
    dtype (torch.dtype, optional): Target data type.

Attributes:
    gamma (nn.Parameter): Learnable scale parameter.
    beta (nn.Parameter): Learnable shift parameter.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/norms/layer_norm.py:36`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/norms/layer_norm.py#L36)

Forward pass of LayerNorm.

Args:
    x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

Returns:
    torch.Tensor: Normalized output tensor of the same shape.
