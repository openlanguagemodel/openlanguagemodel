# `olm.nn.activations.silu`

Source: [`src/olm/nn/activations/silu.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/silu.py#L1)

## Classes

### `SiLU(inplace: bool = False, *, device=None, dtype=None) -> None`

**Bases:** `olm.nn.activations.base.ActivationBase`

Source: [`src/olm/nn/activations/silu.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/silu.py#L7)

SiLU (Swish) activation wrapper.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/activations/silu.py:16`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/silu.py#L16)

Apply activation to ``x``.

### `Swish(inplace: bool = False, *, device=None, dtype=None) -> None`

**Bases:** `olm.nn.activations.base.ActivationBase`

Source: [`src/olm/nn/activations/silu.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/silu.py#L7)

SiLU (Swish) activation wrapper.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/activations/silu.py:16`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/activations/silu.py#L16)

Apply activation to ``x``.
