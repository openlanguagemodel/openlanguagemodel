# `olm.nn.activations.prelu`

Source: [`src/olm/nn/activations/prelu.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/prelu.py#L1)

## Classes

### `PReLU(num_parameters: int = 1, init: float = 0.25, *, device=None, dtype=None) -> None`

**Bases:** `olm.nn.activations.base.ActivationBase`

Source: [`src/olm/nn/activations/prelu.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/prelu.py#L7)

PReLU activation wrapper.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/activations/prelu.py:15`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/prelu.py#L15)
