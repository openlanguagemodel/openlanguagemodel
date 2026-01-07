# olm.train.optim.zero

### Classes

| [`ZeROOptimizer`](#olm.train.optim.zero.ZeROOptimizer)(\*args, \*\*kwargs)   | ZeRO (Zero Redundancy Optimizer) wrapper for distributed training.   |
|------------------------------------------------------------------------------|----------------------------------------------------------------------|

### *class* olm.train.optim.zero.Any(\*args, \*\*kwargs)

Bases: `object`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### *class* olm.train.optim.zero.OptimizerBase(\*args: [Any](#olm.train.optim.zero.Any), \*\*kwargs: [Any](#olm.train.optim.zero.Any))

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

### *class* olm.train.optim.zero.ZeROOptimizer(\*args: [Any](#olm.train.optim.zero.Any), \*\*kwargs: [Any](#olm.train.optim.zero.Any))

Bases: [`OptimizerBase`](olm.train.optim.base.md#olm.train.optim.base.OptimizerBase)

ZeRO (Zero Redundancy Optimizer) wrapper for distributed training.

Implements memory optimization techniques from “ZeRO: Memory Optimizations
Toward Training Trillion Parameter Models” (Rajbhandari et al., 2020).

ZeRO reduces memory consumption by partitioning optimizer states, gradients,
and parameters across data-parallel processes. This implementation provides
a simplified version focusing on optimizer state partitioning (ZeRO Stage 1).

For full ZeRO support with gradient and parameter partitioning, consider using
DeepSpeed or PyTorch’s FSDP (Fully Sharded Data Parallel).

* **Parameters:**
  * **optimizer** – Base optimizer to wrap (e.g., AdamW, Lion)
  * **partition_optimizer_states** – Whether to partition optimizer states (default: True)
  * **overlap_communication** – Overlap gradient communication with computation (default: True)
  * **world_size** – Number of distributed processes (default: None, auto-detected)
  * **rank** – Process rank in distributed group (default: None, auto-detected)

#### add_param_group(param_group: Dict[str, [Any](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Any)])

Add a param group to the Optimizer’s param_groups.

* **Parameters:**
  **param_group** – parameter group to add

#### *property* defaults

Access underlying optimizer’s defaults.

#### extra_repr() → str

String representation for debugging.

#### load_state_dict(state_dict: Dict[str, [Any](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Any)])

Loads the optimizer state.

* **Parameters:**
  **state_dict** – optimizer state dict

#### *property* param_groups

Access underlying optimizer’s parameter groups.

#### *property* state

Access underlying optimizer’s state.

#### state_dict() → Dict[str, [Any](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Any)]

Returns the state of the optimizer as a dict.

In distributed mode, only returns states for parameters owned by this rank.

#### step(closure: Callable[[], float] | None = None) → float | None

Performs a single optimization step.

In distributed mode, synchronizes optimizer states across ranks as needed.

* **Parameters:**
  **closure** – A closure that reevaluates the model and returns the loss.
* **Returns:**
  Optional loss value if closure is provided.

#### zero_grad(set_to_none: bool = True)

Sets gradients of all optimized parameters to zero.

* **Parameters:**
  **set_to_none** – instead of setting to zero, set the grads to None.
  Default: True (overriding base class to match modern PyTorch conventions)
