# `olm.core.dist`

Source: [`src/olm/core/dist.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L1)

Distributed training utilities for PyTorch DDP and FSDP.

Provides wrappers and helpers for single-node multi-GPU training using
PyTorch's native distributed backends. Multi-node launch recipes are planned
for a later roadmap milestone.

## Functions

### `all_gather(tensor: torch.Tensor) -> list[torch.Tensor]`

Source: [`src/olm/core/dist.py:129`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L129)

Gather tensors from all processes.

### `all_reduce(tensor: torch.Tensor, op: torch.distributed.distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, async_op: bool = False) -> torch.distributed.distributed_c10d.Work | None`

Source: [`src/olm/core/dist.py:107`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L107)

Reduce tensor across all processes.

Args:
    tensor: Tensor to reduce (modified in-place).
    op: Reduction operation (SUM, AVG, PRODUCT, MIN, MAX).
    async_op: If True, returns Work handle for async operation.

Example:
    >>> loss = torch.tensor([2.5])
    >>> all_reduce(loss, op=dist.ReduceOp.AVG)

### `barrier() -> None`

Source: [`src/olm/core/dist.py:101`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L101)

Synchronize all processes.

### `broadcast(tensor: torch.Tensor, src: int = 0) -> None`

Source: [`src/olm/core/dist.py:139`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L139)

Broadcast tensor from src rank to all others.

### `cleanup_distributed() -> None`

Source: [`src/olm/core/dist.py:95`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L95)

Cleanup distributed process group.

### `get_backend() -> str | None`

Source: [`src/olm/core/dist.py:163`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L163)

Get current distributed backend.

### `get_local_rank() -> int`

Source: [`src/olm/core/dist.py:32`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L32)

Get local rank on this machine.

### `get_rank() -> int`

Source: [`src/olm/core/dist.py:22`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L22)

Get rank of current process (0 if not distributed).

### `get_world_size() -> int`

Source: [`src/olm/core/dist.py:27`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L27)

Get total number of processes (1 if not distributed).

### `is_distributed() -> bool`

Source: [`src/olm/core/dist.py:17`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L17)

Check if distributed training is initialized.

### `is_main_process() -> bool`

Source: [`src/olm/core/dist.py:37`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L37)

Check if this is rank 0.

### `main_process_only(func: Callable) -> Callable`

Source: [`src/olm/core/dist.py:151`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L151)

Decorator to execute function only on rank 0.

### `print_rank_0(*args, **kwargs) -> None`

Source: [`src/olm/core/dist.py:145`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L145)

Print only on rank 0.

### `setup_distributed(backend: str | None = None, init_method: str = 'env://', timeout_minutes: int = 30) -> None`

Source: [`src/olm/core/dist.py:42`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/core/dist.py#L42)

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
