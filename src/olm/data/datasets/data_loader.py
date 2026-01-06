"""
DataLoader wrapper for OLM library.

This module provides a clean wrapper around PyTorch's DataLoader with
sensible defaults for language model training and convenient helpers.
"""

from typing import Optional, Callable, Iterator, Union
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, IterableDataset


class DataLoader(TorchDataLoader):
    """
    Wrapper around PyTorch's DataLoader with sensible defaults for LLM training.

    This class extends torch.utils.data.DataLoader with:
    - Better defaults for language model training
    - Automatic worker configuration
    - Pin memory optimization for GPU training
    - Persistent workers for efficiency

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
        **kwargs: Additional arguments passed to torch.utils.data.DataLoader.

    Example:
        >>> from olm.data.datasets import DataLoader
        >>> loader = DataLoader(
        ...     dataset=my_dataset,
        ...     batch_size=16,
        ...     num_workers=4,
        ...     pin_memory=True,
        ... )
        >>> for batch in loader:
        ...     # Training loop
        ...     pass
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
        **kwargs,
    ):
        # Auto-configure shuffle based on dataset type
        if shuffle is None:
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
