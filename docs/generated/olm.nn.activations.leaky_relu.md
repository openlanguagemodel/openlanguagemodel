# `olm.nn.activations.leaky_relu`

Source: [`src/olm/nn/activations/leaky_relu.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/leaky_relu.py#L1)

## Classes

### `LeakyReLU(negative_slope: float = 0.01, inplace: bool = False, *, device=None, dtype=None) -> None`

**Bases:** `olm.nn.activations.base.ActivationBase`

Source: [`src/olm/nn/activations/leaky_relu.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/leaky_relu.py#L7)

LeakyReLU activation wrapper.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/activations/leaky_relu.py:14`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/leaky_relu.py#L14)
