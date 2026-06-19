# `olm.nn.activations.base`

## Classes

### `ActivationBase(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None`

Abstract base class for all activation functions.

Ensures a consistent interface for activation layers, handling device and dtype
initialization. Subclasses must implement the `forward` method.

Attributes:
    device (torch.device, optional): The device the module is on.
    dtype (torch.dtype): The data type of the module parameters.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Apply activation to ``x``.
