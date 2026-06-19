# `olm.nn.activations.reglu`

Source: [`src/olm/nn/activations/reglu.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/reglu.py#L1)

## Classes

### `ReGLU(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None`

**Bases:** `olm.nn.activations.base.ActivationBase`

Source: [`src/olm/nn/activations/reglu.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/reglu.py#L6)

ReGLU activation function.

Implements the ReGLU variant from "GLU Variants Improve Transformer".
ReGLU(x, W, V) = ReLU(xW) * (xV)
Here: ReGLU(x) = ReLU(gate) * value

Args:
    device (torch.device, optional): Target device.
    dtype (torch.dtype, optional): Target data type.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/activations/reglu.py:20`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/reglu.py#L20)

Forward pass of ReGLU.

Args:
    x (torch.Tensor): Input tensor.

Returns:
    torch.Tensor: Output tensor with half the last dimension.
