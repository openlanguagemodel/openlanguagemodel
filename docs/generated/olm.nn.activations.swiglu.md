# `olm.nn.activations.swiglu`

## Classes

### `SwiGLU(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None`

SwiGLU activation function.

Implements the SwiGLU activation as described in "GLU Variants Improve Transformer".
It applies the SiLU activation to one half of the input (the gate) and multiplies it
by the other half (the value).

Equation:
    SwiGLU(x, W, V) = Swish_1(xW) * (xV)
    Here, we assume the input `x` is already projected/concatenated such that we chunk it.
    So: SwiGLU(x) = (x_1 * SiLU(x_2)) where x = [x_1, x_2]

Args:
    device (torch.device, optional): Target device.
    dtype (torch.dtype, optional): Target data type.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Forward pass of SwiGLU.
