# `olm.train.optim.base`

## Classes

### `OptimizerBase(params: collections.abc.Iterable[torch.Tensor] | collections.abc.Iterable[dict[str, Any]] | collections.abc.Iterable[tuple[str, torch.Tensor]], defaults: dict[str, typing.Any]) -> None`

Abstract base class for all optimizers in the OLM framework.

Provides a consistent interface for optimizer implementations, including
standard methods for parameter updates, gradient zeroing, and state management.
All custom optimizers should inherit from this class.

This base class extends PyTorch's Optimizer class and adds additional
functionality specific to the OLM framework.

Subclasses must implement the step() method to define the optimization logic.

#### Methods

- `extra_repr(self) -> str`
  String representation of the optimizer for debugging.
- `load_state_dict(self, state_dict: Dict[str, Any])`
  Loads the optimizer state.
- `state_dict(self) -> Dict[str, Any]`
  Returns the state of the optimizer as a dict.
- `step(self, closure: Callable[[], float] | None = None) -> float | None`
  Performs a single optimization step.
- `zero_grad(self, set_to_none: bool = True)`
  Sets gradients of all optimized tensors to zero or None.
