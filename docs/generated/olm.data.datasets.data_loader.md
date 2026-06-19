# `olm.data.datasets.data_loader`

Source: [`src/olm/data/datasets/data_loader.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/data_loader.py#L1)

DataLoader wrapper for OLM library.

This module provides a clean wrapper around PyTorch's DataLoader with
sensible defaults for language model training and convenient helpers.

## Classes

### `DataLoader(dataset: torch.utils.data.dataset.Dataset | torch.utils.data.dataset.IterableDataset, batch_size: int = 8, shuffle: bool | None = None, num_workers: int = 0, pin_memory: bool = True, drop_last: bool = False, persistent_workers: bool | None = None, prefetch_factor: int | None = 2, collate_fn: Callable | None = None, distributed: bool = False, rank: int | None = None, world_size: int | None = None, sampler: torch.utils.data.sampler.Sampler | None = None, **kwargs)`

**Bases:** `DataLoader`

Source: [`src/olm/data/datasets/data_loader.py:13`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/data_loader.py#L13)

Wrapper around PyTorch's DataLoader with sensible defaults for LLM training.

This class extends torch.utils.data.DataLoader with:
- Better defaults for language model training
- Automatic worker configuration
- Pin memory optimization for GPU training
- Persistent workers for efficiency
- Distributed training support with DistributedSampler

For OLM text datasets, iteration usually yields batched
``(input_ids, labels)`` tensors with shape ``[batch, context_length]``.

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

#### Properties

- `multiprocessing_context`

#### Methods

##### `check_worker_number_rationality(self) -> 'None'`

Source: [`.venv/lib/python3.14/site-packages/torch/utils/data/dataloader.py:548`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/.venv/lib/python3.14/site-packages/torch/utils/data/dataloader.py#L548)
