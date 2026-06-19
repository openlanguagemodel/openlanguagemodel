# `olm.nn.activations.base`

Source: [`src/olm/nn/activations/base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/base.py#L1)

## Classes

### `ActivationBase(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None`

**Bases:** `Module`, `ABC`

Source: [`src/olm/nn/activations/base.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/base.py#L8)

Abstract base class for all activation functions.

Ensures a consistent interface for activation layers, handling device and dtype
initialization. Subclasses must implement the `forward` method.

Attributes:
    device (torch.device, optional): The device the module is on.
    dtype (torch.dtype): The data type of the module parameters.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/activations/base.py:34`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/base.py#L34)

Apply activation to ``x``.
