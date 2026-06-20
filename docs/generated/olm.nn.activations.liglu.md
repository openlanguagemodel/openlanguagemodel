# `olm.nn.activations.liglu`

Source: [`src/olm/nn/activations/liglu.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/liglu.py#L1)

## Classes

### `LiGLU(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None`

**Bases:** `olm.nn.activations.base.ActivationBase`

Source: [`src/olm/nn/activations/liglu.py:5`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/liglu.py#L5)

LiGLU activation function.

Implements the LiGLU variant (Linear GLU).
LiGLU(x, W, V) = (xW) * (xV)
Here: LiGLU(x) = gate * value (No activation on gate)

**Parameters**

- `device` (`torch.device, optional`): Target device.
- `dtype` (`torch.dtype, optional`): Target data type.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/activations/liglu.py:19`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/liglu.py#L19)

Forward pass of LiGLU.

**Parameters**

- `x` (`torch.Tensor`): Input tensor.

**Returns**

- `torch.Tensor`: Output tensor with half the last dimension.
