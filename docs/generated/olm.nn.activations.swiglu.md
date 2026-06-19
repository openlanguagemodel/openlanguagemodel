# `olm.nn.activations.swiglu`

Source: [`src/olm/nn/activations/swiglu.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/swiglu.py#L1)

## Classes

### `SwiGLU(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None`

**Bases:** `olm.nn.activations.base.ActivationBase`

Source: [`src/olm/nn/activations/swiglu.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/swiglu.py#L7)

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

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/activations/swiglu.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/swiglu.py#L26)

Forward pass of SwiGLU.

Args:
    x (torch.Tensor): Input tensor. Expected to have an even last dimension size.

Returns:
    torch.Tensor: Output tensor with half the last dimension of the input.
