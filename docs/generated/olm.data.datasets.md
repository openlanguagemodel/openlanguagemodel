# `olm.data.datasets`

## Classes

### `BaseTextDataset(tokenizer: Any, context_length: int, skip_batches: int = 0, shuffle: bool = False, seed: int = 42)`

Abstract base class for text-based streaming datasets.

Handles tokenization buffering and sequence generation generically.
Subclasses must implement `_get_text_iterator` to yield text chunks.

### `DataLoader(dataset: torch.utils.data.dataset.Dataset | torch.utils.data.dataset.IterableDataset, batch_size: int = 8, shuffle: bool | None = None, num_workers: int = 0, pin_memory: bool = True, drop_last: bool = False, persistent_workers: bool | None = None, prefetch_factor: int | None = 2, collate_fn: Callable | None = None, distributed: bool = False, rank: int | None = None, world_size: int | None = None, sampler: torch.utils.data.sampler.Sampler | None = None, **kwargs)`

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

#### Methods

- `check_worker_number_rationality(self) -> 'None'`

### `FineWebEduDataset(tokenizer: Any, split: str = 'train', context_length: int = 1024, subset: str = 'sample-10BT', streaming: bool = True, shuffle: bool = False, seed: int = 42, cache_dir: str | None = None, skip_batches: int = 0)`

FineWeb Edu dataset configuration.

Args:
    split: Dataset split ('train' or 'validation')
    context_length: Sequence length for training (default: 1024)
    subset: Dataset subset to use (default: 'sample-10BT')
    tokenizer: Tokenizer object (e.g. from AutoTokenizer)
    streaming: Whether to use streaming mode (default: True)
    shuffle: Whether to shuffle the dataset (default: False)
    seed: Random seed for shuffling (default: 42)
    cache_dir: Directory to cache downloaded data (default: None)
    skip_batches: Number of batches to skip

### `HuggingFaceTextDataset(dataset_name: str, split: str, context_length: int, text_fn: Callable[[Any], str], tokenizer: Any, dataset_kwargs: Dict[str, Any] | None = None, streaming: bool = True, skip_batches: int = 0, shuffle: bool = False, seed: int = 42, shuffle_buffer_size: int = 10000)`

Generic dataset loader for Hugging Face text datasets.

Inherits from BaseTextDataset to share token buffering logic and multi-worker safety.

### `LocalTextDataset(location: str | os.PathLike, tokenizer, context_length: int, skip_batches: int = 0, shuffle: bool = False, seed: int = 42)`

Dataset that streams text from local .txt files in a directory.
