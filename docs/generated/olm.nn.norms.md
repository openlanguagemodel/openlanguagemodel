# `olm.nn.norms`

## Classes

### `LayerNorm(d_model: int, eps: float = 1e-05, elementwise_affine: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`

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

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Forward pass of LayerNorm.

### `RMSNorm(d_model: int, eps: float = 1e-05, device: torch.device | None = None, dtype: torch.dtype | None = None)`

RMSNorm (Root Mean Square Layer Normalization) layer.

Implements RMSNorm as described in "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467).
A simplified version of LayerNorm that scales invariance properties.

Args:
    d_model (int): The dimension of the model to normalize.
    eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
    device (torch.device, optional): Target device.
    dtype (torch.dtype, optional): Target data type.

Attributes:
    weight (nn.Parameter): Learnable scale parameter.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Forward pass of RMSNorm.
