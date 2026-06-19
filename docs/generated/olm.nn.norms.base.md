# `olm.nn.norms.base`

## Classes

### `NormBase(d_model: int, device=None, dtype=None)`

Abstract base class for normalization layers (e.g., LayerNorm, RMSNorm).

Standardizes the interface for normalization, ensuring all implementations
handle model dimension, device, and dtype consistently.

Attributes:
    d_model (int): The dimension of the input features to normalize.
    device (torch.device, optional): The device the module is on.
    dtype (torch.dtype): The data type of the module parameters.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Apply normalization to the input tensor.
