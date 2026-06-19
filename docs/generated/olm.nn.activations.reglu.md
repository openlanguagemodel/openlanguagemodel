# `olm.nn.activations.reglu`

## Classes

### `ReGLU(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None`

ReGLU activation function.

Implements the ReGLU variant from "GLU Variants Improve Transformer".
ReGLU(x, W, V) = ReLU(xW) * (xV)
Here: ReGLU(x) = ReLU(gate) * value

Args:
    device (torch.device, optional): Target device.
    dtype (torch.dtype, optional): Target data type.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Forward pass of ReGLU.
