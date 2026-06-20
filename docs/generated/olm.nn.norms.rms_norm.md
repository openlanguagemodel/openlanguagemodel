# `olm.nn.norms.rms_norm`

Source: [`src/olm/nn/norms/rms_norm.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/norms/rms_norm.py#L1)

## Classes

### `RMSNorm(d_model: int, eps: float = 1e-05, device: torch.device | None = None, dtype: torch.dtype | None = None)`

**Bases:** `olm.nn.norms.base.NormBase`

Source: [`src/olm/nn/norms/rms_norm.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/norms/rms_norm.py#L7)

RMSNorm (Root Mean Square Layer Normalization) layer.

Implements RMSNorm as described in "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467).
A simplified version of LayerNorm that scales invariance properties.

**Parameters**

- `d_model` (`int`): The dimension of the model to normalize.
- `eps` (`float, optional`): Small constant for numerical stability. Defaults to 1e-5.
- `device` (`torch.device, optional`): Target device.
- `dtype` (`torch.dtype, optional`): Target data type.

**Attributes**

- `weight` (`nn.Parameter`): Learnable scale parameter.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/norms/rms_norm.py:29`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/norms/rms_norm.py#L29)

Forward pass of RMSNorm.

**Parameters**

- `x` (`torch.Tensor`): Input tensor of shape (batch_size, sequence_length, d_model).

**Returns**

- `torch.Tensor`: Normalized output tensor of the same shape.
