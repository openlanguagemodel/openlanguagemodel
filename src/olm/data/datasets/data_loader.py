"""
DataLoader wrapper for OLM library.

This module provides a clean wrapper around PyTorch's DataLoader with
sensible defaults for language model training and convenient helpers.
"""

from typing import Optional, Callable, Iterator, Union
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, IterableDataset, DistributedSampler, Sampler


class DataLoader(TorchDataLoader):
    """
    Wrapper around PyTorch's DataLoader with sensible defaults for LLM training.

    This class extends torch.utils.data.DataLoader with:
    - Better defaults for language model training
    - Automatic worker configuration
    - Pin memory optimization for GPU training
    - Persistent workers for efficiency
    - Distributed training support with DistributedSampler

    Args:
        dataset: Dataset to load from (can be map-style or iterable).
        batch_size: Number of samples per batch (default: 8).
        shuffle: Whether to shuffle data at every epoch (default: False for iterable datasets).
        num_workers: Number of worker processes for data loading (default: 0).
        pin_memory: If True, tensors are copied to CUDA pinned memory (default: True).
        drop_last: Drop the last incomplete batch if dataset size is not divisible by batch_size.
        persistent_workers: Keep workers alive between epochs for faster startup (default: True if num_workers > 0).
        prefetch_factor: Number of batches to prefetch per worker (default: 2).
        collate_fn: Function to merge samples into batches.
        distributed: If True, automatically creates DistributedSampler for distributed training.
        rank: Rank for distributed training (auto-detected if None).
        world_size: World size for distributed training (auto-detected if None).
        sampler: Custom sampler (overrides distributed if provided).
        **kwargs: Additional arguments passed to torch.utils.data.DataLoader.

    Example:
        >>> # Single GPU
        >>> loader = DataLoader(dataset=my_dataset, batch_size=16)
        >>>
        >>> # Distributed training (with torchrun)
        >>> loader = DataLoader(
        ...     dataset=my_dataset,
        ...     batch_size=16,
        ...     distributed=True,  # Automatically creates DistributedSampler
        ... )
        >>> for epoch in range(epochs):
        ...     loader.sampler.set_epoch(epoch)  # Important for proper shuffling
        ...     for batch in loader:
        ...         # Training loop
        ...         pass
    """

    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        batch_size: int = 8,
        shuffle: Optional[bool] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = 2,
        collate_fn: Optional[Callable] = None,
        distributed: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        **kwargs,
    ):
        # Handle distributed training
        if distributed and sampler is None and not isinstance(dataset, IterableDataset):
            # Auto-detect rank and world_size from torch.distributed
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                if rank is None:
                    rank = dist.get_rank()
                if world_size is None:
                    world_size = dist.get_world_size()

                # Create DistributedSampler
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle if shuffle is not None else True,
                    drop_last=drop_last,
                )
                # Don't shuffle in DataLoader when using DistributedSampler
                shuffle = False

        # Auto-configure shuffle based on dataset type
        if shuffle is None and sampler is None:
            shuffle = not isinstance(dataset, IterableDataset)

        # Auto-configure persistent workers
        if persistent_workers is None:
            persistent_workers = num_workers > 0

        # Only set prefetch_factor if num_workers > 0
        if num_workers == 0:
            prefetch_factor = None

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn,
            **kwargs,
        )

    def __repr__(self) -> str:
        """String representation of the DataLoader."""
        return (
            f"DataLoader("
            f"batch_size={self.batch_size}, "
            f"num_workers={self.num_workers}, "
            f"pin_memory={self.pin_memory}, "
            f"dataset={type(self.dataset).__name__}"
            f")"
        )
