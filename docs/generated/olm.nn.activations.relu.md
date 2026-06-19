# `olm.nn.activations.relu`

Source: [`src/olm/nn/activations/relu.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/relu.py#L1)

## Classes

### `ReLU(inplace: bool = False, *, device=None, dtype=None) -> None`

**Bases:** `olm.nn.activations.base.ActivationBase`

Source: [`src/olm/nn/activations/relu.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/relu.py#L7)

ReLU activation wrapper.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/activations/relu.py:14`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/relu.py#L14)

Apply activation to ``x``.
