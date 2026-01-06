# olm.train.optim.lion

### Classes

| [`Lion`](#olm.train.optim.lion.Lion)(\*args, \*\*kwargs)   | Lion optimizer (EvoLved Sign Momentum).   |
|------------------------------------------------------------|-------------------------------------------|

### *class* olm.train.optim.lion.Any(\*args, \*\*kwargs)

Bases: `object`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### *class* olm.train.optim.lion.Lion(\*args: [Any](#olm.train.optim.lion.Any), \*\*kwargs: [Any](#olm.train.optim.lion.Any))

Bases: [`OptimizerBase`](olm.train.optim.base.md#olm.train.optim.base.OptimizerBase)

Lion optimizer (EvoLved Sign Momentum).

Implements the Lion algorithm from “Symbolic Discovery of Optimization Algorithms”
(Chen et al., 2023). Lion uses only the sign of the gradient for updates,
making it more memory-efficient than Adam while often achieving better performance.

Key differences from Adam:
- Uses sign of interpolated gradient for updates (memory efficient)
- Single momentum buffer instead of two (m and v in Adam)
- Typically requires smaller learning rates (1/3 to 1/10 of AdamW)
- Larger weight decay (3-10x that of AdamW)

* **Parameters:**
  * **params** – iterable of parameters to optimize or dicts defining parameter groups
  * **lr** – learning rate (default: 1e-4, typically 3-10x smaller than AdamW)
  * **betas** – coefficients used for computing running averages (default: (0.9, 0.99))
  * **weight_decay** – weight decay coefficient (default: 0.0)
  * **use_triton** – whether to use Triton kernel for faster computation (default: False)

### Example

```pycon
>>> model = nn.Linear(10, 5)
>>> optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)
>>> optimizer.zero_grad()
>>> loss = model(input).sum()
>>> loss.backward()
>>> optimizer.step()
```

#### step(closure: Callable[[], float] | None = None) → float | None

Performs a single optimization step.

* **Parameters:**
  **closure** – A closure that reevaluates the model and returns the loss.
* **Returns:**
  Optional loss value if closure is provided.

#### zero_grad(set_to_none: bool = True)

Sets gradients of all optimized tensors to zero.

* **Parameters:**
  **set_to_none** – instead of setting to zero, set the grads to None.
  This is more memory efficient and can slightly improve performance.

### *class* olm.train.optim.lion.OptimizerBase(\*args: [Any](#olm.train.optim.lion.Any), \*\*kwargs: [Any](#olm.train.optim.lion.Any))

Bases: `Optimizer`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for all optimizers in the OLM framework.

Provides a consistent interface for optimizer implementations, including
standard methods for parameter updates, gradient zeroing, and state management.
All custom optimizers should inherit from this class.

This base class extends PyTorch’s Optimizer class and adds additional
functionality specific to the OLM framework.

Subclasses must implement the step() method to define the optimization logic.

#### extra_repr() → str

String representation of the optimizer for debugging.

Override this in subclasses to provide useful information.

#### load_state_dict(state_dict: Dict[str, [Any](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Any)])

Loads the optimizer state.

* **Parameters:**
  **state_dict** – optimizer state. Should be an object returned
  from a call to state_dict().

#### state_dict() → Dict[str, [Any](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Any)]

Returns the state of the optimizer as a dict.

It contains two entries:

- `state`: dict holding current optimization state. Its content
  differs between optimizer classes.
- `param_groups`: list containing all parameter groups where each
  parameter group is a dict.

* **Returns:**
  Dictionary containing optimizer state

#### *abstractmethod* step(closure: Callable[[], float] | None = None) → float | None

Performs a single optimization step.

* **Parameters:**
  **closure** – A closure that reevaluates the model and returns the loss.
  Some optimization algorithms (e.g., L-BFGS) require multiple
  evaluations of the loss function.
* **Returns:**
  Optional loss value if closure is provided.

#### zero_grad(set_to_none: bool = True)

Sets gradients of all optimized tensors to zero or None.

* **Parameters:**
  **set_to_none** – Instead of setting to zero, set the grads to None.
  This is more memory efficient and can slightly improve performance.
  Default: True
