# `olm.core.dist`

Distributed training utilities for PyTorch DDP and FSDP.

Provides wrappers and helpers for multi-GPU/multi-node training using
PyTorch's native distributed backends.

## Functions

### `all_gather(tensor: torch.Tensor) -> list[torch.Tensor]`

Gather tensors from all processes.

### `all_reduce(tensor: torch.Tensor, op: torch.distributed.distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, async_op: bool = False) -> torch.distributed.distributed_c10d.Work | None`

Reduce tensor across all processes.

Args:
    tensor: Tensor to reduce (modified in-place).
    op: Reduction operation (SUM, AVG, PRODUCT, MIN, MAX).
    async_op: If True, returns Work handle for async operation.

Example:
    >>> loss = torch.tensor([2.5])
    >>> all_reduce(loss, op=dist.ReduceOp.AVG)

### `barrier() -> None`

Synchronize all processes.

### `broadcast(tensor: torch.Tensor, src: int = 0) -> None`

Broadcast tensor from src rank to all others.

### `cleanup_distributed() -> None`

Cleanup distributed process group.

### `get_backend() -> str | None`

Get current distributed backend.

### `get_local_rank() -> int`

Get local rank on this machine.

### `get_rank() -> int`

Get rank of current process (0 if not distributed).

### `get_world_size() -> int`

Get total number of processes (1 if not distributed).

### `is_distributed() -> bool`

Check if distributed training is initialized.

### `is_main_process() -> bool`

Check if this is rank 0.

### `main_process_only(func: Callable) -> Callable`

Decorator to execute function only on rank 0.

### `print_rank_0(*args, **kwargs) -> None`

Print only on rank 0.

### `setup_distributed(backend: str | None = None, init_method: str = 'env://', timeout_minutes: int = 30) -> None`

Initialize distributed process group from environment variables.

Args:
    backend: 'nccl', 'gloo', or None (auto-detect).
    init_method: Initialization method. Defaults to 'env://'.
    timeout_minutes: Timeout for operations.

Environment variables (set by torchrun):
    RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT

Example:
    >>> # Run with: torchrun --nproc_per_node=4 train.py
    >>> setup_distributed()
