# `olm.nn.norms.base`

Source: [`src/olm/nn/norms/base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/norms/base.py#L1)

## Classes

### `NormBase(d_model: int, device=None, dtype=None)`

**Bases:** `Module`, `ABC`

Source: [`src/olm/nn/norms/base.py:5`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/norms/base.py#L5)

Abstract base class for normalization layers (e.g., LayerNorm, RMSNorm).

Standardizes the interface for normalization, ensuring all implementations
handle model dimension, device, and dtype consistently.

Attributes:
    d_model (int): The dimension of the input features to normalize.
    device (torch.device, optional): The device the module is on.
    dtype (torch.dtype): The data type of the module parameters.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/norms/base.py:32`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/norms/base.py#L32)

Apply normalization to the input tensor.
