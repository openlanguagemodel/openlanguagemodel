# `olm.nn.activations.geglu`

## Classes

### `GeGLU(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None`

GeGLU activation function.

Implements the GeGLU variant from "GLU Variants Improve Transformer".
GeGLU(x, W, V) = GELU(xW) * (xV)
Here: GeGLU(x) = GELU(gate) * value

Args:
    device (torch.device, optional): Target device.
    dtype (torch.dtype, optional): Target data type.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Forward pass of GeGLU.
