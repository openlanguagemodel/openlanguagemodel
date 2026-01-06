# olm.train.optim

### *class* olm.train.optim.AdamW(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `AdamW`

AdamW optimizer with decoupled weight decay regularization.

This is a wrapper around PyTorch’s built-in AdamW implementation from
“Decoupled Weight Decay Regularization” (Loshchilov & Hutter, 2017).
Unlike the original Adam, weight decay is applied directly to the parameters
rather than being added to the gradient.

This implementation is commonly used for training large language models and
transformers, offering better generalization than standard Adam.

Note: This class inherits from PyTorch’s AdamW which ultimately inherits from
torch.optim.Optimizer, maintaining compatibility with our OptimizerBase interface.

* **Parameters:**
  * **params** – iterable of parameters to optimize or dicts defining parameter groups
  * **lr** – learning rate (default: 1e-3)
  * **betas** – coefficients used for computing running averages of gradient and its square
    (default: (0.9, 0.999))
  * **eps** – term added to the denominator to improve numerical stability (default: 1e-8)
  * **weight_decay** – weight decay coefficient (default: 0.01)
  * **amsgrad** – whether to use the AMSGrad variant (default: False)
  * **maximize** – maximize the params based on the objective, instead of minimizing (default: False)
  * **fused** – whether to use the fused implementation (default: None, auto-detect)

### Example

```pycon
>>> model = nn.Linear(10, 5)
>>> optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
>>> optimizer.zero_grad()
>>> loss = model(input).sum()
>>> loss.backward()
>>> optimizer.step()
```

### *class* olm.train.optim.Lion(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.train.optim.OptimizerBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.train.optim.ZeROOptimizer(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### Modules

| [`adamw`](olm.train.optim.adamw.md#module-olm.train.optim.adamw)   |    |
|--------------------------------------------------------------------|----|
| [`base`](olm.train.optim.base.md#module-olm.train.optim.base)      |    |
| [`lion`](olm.train.optim.lion.md#module-olm.train.optim.lion)      |    |
| [`zero`](olm.train.optim.zero.md#module-olm.train.optim.zero)      |    |
