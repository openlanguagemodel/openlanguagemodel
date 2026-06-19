# `olm.train.optim.zero`

## Classes

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
