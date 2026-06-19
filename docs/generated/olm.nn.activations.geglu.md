# `olm.nn.activations.geglu`

Source: [`src/olm/nn/activations/geglu.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/geglu.py#L1)

## Classes

### `GeGLU(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None`

**Bases:** `olm.nn.activations.base.ActivationBase`

Source: [`src/olm/nn/activations/geglu.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/geglu.py#L6)

GeGLU activation function.

Implements the GeGLU variant from "GLU Variants Improve Transformer".
GeGLU(x, W, V) = GELU(xW) * (xV)
Here: GeGLU(x) = GELU(gate) * value

Args:
    device (torch.device, optional): Target device.
    dtype (torch.dtype, optional): Target data type.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/activations/geglu.py:20`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/geglu.py#L20)

Forward pass of GeGLU.

Args:
    x (torch.Tensor): Input tensor.

Returns:
    torch.Tensor: Output tensor with half the last dimension.
