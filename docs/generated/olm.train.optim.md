# `olm.train.optim`

## Classes

### `AdamW(params, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, amsgrad: bool = False, maximize: bool = False, fused: bool | None = None)`

AdamW optimizer with decoupled weight decay regularization.

This is a wrapper around PyTorch's built-in AdamW implementation from
"Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017).
Unlike the original Adam, weight decay is applied directly to the parameters
rather than being added to the gradient.

This implementation is commonly used for training large language models and
transformers, offering better generalization than standard Adam.

Note: This class inherits from PyTorch's AdamW which ultimately inherits from
torch.optim.Optimizer, maintaining compatibility with our OptimizerBase interface.

Args:
    params: iterable of parameters to optimize or dicts defining parameter groups
    lr: learning rate (default: 1e-3)
    betas: coefficients used for computing running averages of gradient and its square
        (default: (0.9, 0.999))
    eps: term added to the denominator to improve numerical stability (default: 1e-8)
    weight_decay: weight decay coefficient (default: 0.01)
    amsgrad: whether to use the AMSGrad variant (default: False)
    maximize: maximize the params based on the objective, instead of minimizing (default: False)
    fused: whether to use the fused implementation (default: None, auto-detect)

Example:
    >>> model = nn.Linear(10, 5)
    >>> optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    >>> optimizer.zero_grad()
    >>> loss = model(input).sum()
    >>> loss.backward()
    >>> optimizer.step()

### `Lion(params: Iterable, lr: float = 0.0001, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0, use_triton: bool = False)`

Lion optimizer (EvoLved Sign Momentum).

Implements the Lion algorithm from "Symbolic Discovery of Optimization Algorithms"
(Chen et al., 2023). Lion uses only the sign of the gradient for updates,
making it more memory-efficient than Adam while often achieving better performance.

Key differences from Adam:
- Uses sign of interpolated gradient for updates (memory efficient)
- Single momentum buffer instead of two (m and v in Adam)
- Typically requires smaller learning rates (1/3 to 1/10 of AdamW)
- Larger weight decay (3-10x that of AdamW)

Args:
    params: iterable of parameters to optimize or dicts defining parameter groups
    lr: learning rate (default: 1e-4, typically 3-10x smaller than AdamW)
    betas: coefficients used for computing running averages (default: (0.9, 0.99))
    weight_decay: weight decay coefficient (default: 0.0)
    use_triton: whether to use Triton kernel for faster computation (default: False)

Example:
    >>> model = nn.Linear(10, 5)
    >>> optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)
    >>> optimizer.zero_grad()
    >>> loss = model(input).sum()
    >>> loss.backward()
    >>> optimizer.step()

#### Methods

- `step(self, closure: Callable[[], float] | None = None) -> float | None`
  Performs a single optimization step.
- `zero_grad(self, set_to_none: bool = True)`
  Sets gradients of all optimized tensors to zero.

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

### `ZeROOptimizer(optimizer: torch.optim.optimizer.Optimizer, partition_optimizer_states: bool = True, overlap_communication: bool = True, world_size: int | None = None, rank: int | None = None)`

ZeRO (Zero Redundancy Optimizer) wrapper for distributed training.

Implements memory optimization techniques from "ZeRO: Memory Optimizations
Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020).

ZeRO reduces memory consumption by partitioning optimizer states, gradients,
and parameters across data-parallel processes. This implementation provides
a simplified version focusing on optimizer state partitioning (ZeRO Stage 1).

For full ZeRO support with gradient and parameter partitioning, consider using
DeepSpeed or PyTorch's FSDP (Fully Sharded Data Parallel).

Args:
    optimizer: Base optimizer to wrap (e.g., AdamW, Lion)
    partition_optimizer_states: Whether to partition optimizer states (default: True)
    overlap_communication: Overlap gradient communication with computation (default: True)
    world_size: Number of distributed processes (default: None, auto-detected)
    rank: Process rank in distributed group (default: None, auto-detected)

#### Methods

- `add_param_group(self, param_group: Dict[str, Any])`
  Add a param group to the Optimizer's param_groups.
- `extra_repr(self) -> str`
  String representation for debugging.
- `load_state_dict(self, state_dict: Dict[str, Any])`
  Loads the optimizer state.
- `state_dict(self) -> Dict[str, Any]`
  Returns the state of the optimizer as a dict.
- `step(self, closure: Callable[[], float] | None = None) -> float | None`
  Performs a single optimization step.
- `zero_grad(self, set_to_none: bool = True)`
  Sets gradients of all optimized parameters to zero.
