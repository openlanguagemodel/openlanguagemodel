"""
Distributed training utilities for PyTorch DDP and FSDP.

Provides wrappers and helpers for single-node multi-GPU training using
PyTorch's native distributed backends. Multi-node launch recipes are planned
for a later roadmap milestone.
"""

import os
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Optional, Callable, Any
import functools


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get rank of current process (0 if not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Get total number of processes (1 if not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    """Get local rank on this machine."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is rank 0."""
    return get_rank() == 0


def setup_distributed(
    backend: Optional[str] = None,
    init_method: str = "env://",
    timeout_minutes: int = 30,
) -> None:
    """
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
    """
    if dist.is_initialized():
        return

    # Auto-detect backend
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Set defaults for environment variables
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = get_local_rank()

    if world_size == 1:
        return  # No need to initialize for single process

    timeout = timedelta(minutes=timeout_minutes)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )

    # Set device for CUDA
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def cleanup_distributed() -> None:
    """Cleanup distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    async_op: bool = False,
) -> Optional[dist.Work]:
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce (modified in-place).
        op: Reduction operation (SUM, AVG, PRODUCT, MIN, MAX).
        async_op: If True, returns Work handle for async operation.

    Example:
        >>> loss = torch.tensor([2.5])
        >>> all_reduce(loss, op=dist.ReduceOp.AVG)
    """
    if not is_distributed():
        return None
    return dist.all_reduce(tensor, op=op, async_op=async_op)


def all_gather(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather tensors from all processes."""
    if not is_distributed():
        return [tensor]

    tensor_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def broadcast(tensor: torch.Tensor, src: int = 0) -> None:
    """Broadcast tensor from src rank to all others."""
    if is_distributed():
        dist.broadcast(tensor, src=src)


def print_rank_0(*args, **kwargs) -> None:
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def main_process_only(func: Callable) -> Callable:
    """Decorator to execute function only on rank 0."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        return None

    return wrapper


def get_backend() -> Optional[str]:
    """Get current distributed backend."""
    return dist.get_backend() if is_distributed() else None
