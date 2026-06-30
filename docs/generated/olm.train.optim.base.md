# `olm.train.optim.base`

Source: [`src/olm/train/optim/base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L1)

## Classes

### `OptimizerBase(params: collections.abc.Iterable[torch.Tensor] | collections.abc.Iterable[dict[str, Any]] | collections.abc.Iterable[tuple[str, torch.Tensor]], defaults: dict[str, typing.Any]) -> None`

**Bases:** `Optimizer`, `ABC`

Source: [`src/olm/train/optim/base.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L8)

Abstract base class for all optimizers in the OLM framework.

Provides a consistent interface for optimizer implementations, including
standard methods for parameter updates, gradient zeroing, and state management.
All custom optimizers should inherit from this class.

This base class extends PyTorch's Optimizer class and adds additional
functionality specific to the OLM framework.

Subclasses must implement the step() method to define the optimization logic.

#### Methods

##### `extra_repr(self) -> str`

Source: [`src/olm/train/optim/base.py:74`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L74)

String representation of the optimizer for debugging.

Override this in subclasses to provide useful information.

##### `load_state_dict(self, state_dict: Dict[str, Any])`

Source: [`src/olm/train/optim/base.py:64`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L64)

Loads the optimizer state.

**Parameters**

- `state_dict`: optimizer state. Should be an object returned from a call to state_dict().

##### `state_dict(self) -> Dict[str, Any]`

Source: [`src/olm/train/optim/base.py:48`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L48)

Returns the state of the optimizer as a dict.

It contains two entries:

- ``state``: dict holding current optimization state. Its content
  differs between optimizer classes.
- ``param_groups``: list containing all parameter groups where each
  parameter group is a dict.

**Returns**

Dictionary containing optimizer state

##### `step(self, closure: Callable[[], float] | None = None) -> float | None`

Source: [`src/olm/train/optim/base.py:22`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L22)

Performs a single optimization step.

**Parameters**

- `closure`: A closure that reevaluates the model and returns the loss. Some optimization algorithms (e.g., L-BFGS) require multiple evaluations of the loss function.

**Returns**

Optional loss value if closure is provided.

##### `zero_grad(self, set_to_none: bool = True)`

Source: [`src/olm/train/optim/base.py:37`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L37)

Sets gradients of all optimized tensors to zero or None.

**Parameters**

- `set_to_none`: Instead of setting to zero, set the grads to None. This is more memory efficient and can slightly improve performance.
- `Default`: True
