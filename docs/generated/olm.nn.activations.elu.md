# `olm.nn.activations.elu`

Source: [`src/olm/nn/activations/elu.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/elu.py#L1)

## Classes

### `ELU(alpha: float = 1.0, inplace: bool = False, *, device=None, dtype=None) -> None`

**Bases:** `olm.nn.activations.base.ActivationBase`

Source: [`src/olm/nn/activations/elu.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/elu.py#L7)

ELU activation wrapper.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/activations/elu.py:14`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/elu.py#L14)

Apply activation to ``x``.
