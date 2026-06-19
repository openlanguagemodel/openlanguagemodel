# `olm.nn.activations.liglu`

## Classes

### `LiGLU(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None`

LiGLU activation function.

Implements the LiGLU variant (Linear GLU).
LiGLU(x, W, V) = (xW) * (xV)
Here: LiGLU(x) = gate * value (No activation on gate)

Args:
    device (torch.device, optional): Target device.
    dtype (torch.dtype, optional): Target data type.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Forward pass of LiGLU.
