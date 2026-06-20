# `olm.data.datasets`

Source: [`src/olm/data/datasets/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/__init__.py#L1)

## Classes

### `BaseTextDataset(tokenizer: Any, context_length: int, skip_batches: int = 0, shuffle: bool = False, seed: int = 42)`

**Bases:** `IterableDataset`, `ABC`

Source: [`src/olm/data/datasets/base_dataset.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/base_dataset.py#L8)

Abstract base class for text-based streaming datasets.

``BaseTextDataset`` handles tokenization, buffering, next-token target
construction, worker sharding, and distributed-rank sharding. Subclasses
only need to implement ``_get_text_iterator`` and yield raw text strings.

**Iteration**

Yields ``(input_ids, labels)`` tuples. Both tensors have shape
``[context_length]`` and dtype ``torch.long``. ``labels`` is the
one-token-shifted target sequence for causal language modeling.

**Parameters**

- `tokenizer`: Tokenizer with an ``encode`` method.
- `context_length` (`int`): Number of input tokens per sample.
- `skip_batches` (`int`): Number of yielded samples to skip, useful for coarse resume behavior.
- `shuffle` (`bool`): Whether the concrete dataset should shuffle its source stream when supported.
- `seed` (`int`): Shuffle seed.

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

**Parameters**

- `dataset`: Dataset to load from (can be map-style or iterable).
- `batch_size`: Number of samples per batch (default: 8).
- `shuffle`: Whether to shuffle data at every epoch (default: False for iterable datasets).
- `num_workers`: Number of worker processes for data loading (default: 0).
- `pin_memory`: If True, tensors are copied to CUDA pinned memory (default: True).
- `drop_last`: Drop the last incomplete batch if dataset size is not divisible by batch_size.
- `persistent_workers`: Keep workers alive between epochs for faster startup (default: True if num_workers > 0).
- `prefetch_factor`: Number of batches to prefetch per worker (default: 2).
- `collate_fn`: Function to merge samples into batches.
- `distributed`: If True, automatically creates DistributedSampler for distributed training.
- `rank`: Rank for distributed training (auto-detected if None).
- `world_size`: World size for distributed training (auto-detected if None).
- `sampler`: Custom sampler (overrides distributed if provided).
- `**kwargs`: Additional arguments passed to torch.utils.data.DataLoader.

**Example**

```python
# Single GPU
loader = DataLoader(dataset=my_dataset, batch_size=16)

# Distributed training (with torchrun)
loader = DataLoader(
    dataset=my_dataset,
    batch_size=16,
    distributed=True,  # Automatically creates DistributedSampler
)
for epoch in range(epochs):
    loader.sampler.set_epoch(epoch)  # Important for proper shuffling
    for batch in loader:
        # Training loop
        pass
```

### `FineWebEduDataset(tokenizer: Any, split: str = 'train', context_length: int = 1024, subset: str = 'sample-10BT', streaming: bool = True, shuffle: bool = False, seed: int = 42, cache_dir: str | None = None, skip_batches: int = 0)`

**Bases:** `olm.data.datasets.hf_dataset.HuggingFaceTextDataset`

Source: [`src/olm/data/datasets/hf_dataset.py:83`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/hf_dataset.py#L83)

Convenience wrapper for ``HuggingFaceFW/fineweb-edu``.

**Iteration**

Yields ``(input_ids, labels)`` tensors shaped ``[context_length]`` for
causal language-model training.

**Parameters**

- `tokenizer`: Tokenizer with an ``encode`` method.
- `split`: Dataset split ('train' or 'validation')
- `context_length`: Sequence length for training (default: 1024)
- `subset`: Dataset subset to use (default: 'sample-10BT')
- `streaming`: Whether to use streaming mode (default: True)
- `shuffle`: Whether to shuffle the dataset (default: False)
- `seed`: Random seed for shuffling (default: 42)
- `cache_dir`: Directory to cache downloaded data (default: None)
- `skip_batches`: Number of batches to skip

### `HuggingFaceTextDataset(dataset_name: str, split: str, context_length: int, text_fn: Callable[[Any], str], tokenizer: Any, dataset_kwargs: Dict[str, Any] | None = None, streaming: bool = True, skip_batches: int = 0, shuffle: bool = False, seed: int = 42, shuffle_buffer_size: int = 10000)`

**Bases:** `olm.data.datasets.base_dataset.BaseTextDataset`

Source: [`src/olm/data/datasets/hf_dataset.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/hf_dataset.py#L8)

Generic dataset loader for Hugging Face text datasets.

The Hugging Face dataset is read in streaming mode by default, then text is
extracted with ``text_fn`` and passed through ``BaseTextDataset`` for
tokenization, sharding, and next-token target construction.

**Iteration**

Yields ``(input_ids, labels)`` tensors shaped ``[context_length]``.

**Parameters**

- `dataset_name` (`str`): Hugging Face dataset name.
- `split` (`str`): Dataset split, such as ``"train"``.
- `context_length` (`int`): Number of input tokens per sample.
- `text_fn`: Function that maps a dataset example to text.
- `tokenizer`: Tokenizer with an ``encode`` method.
- `dataset_kwargs`: Extra keyword arguments passed to ``load_dataset``.
- `streaming` (`bool`): Whether to use Hugging Face streaming mode.
- `skip_batches` (`int`): Number of samples to skip before yielding.
- `shuffle` (`bool`): Whether to shuffle the dataset stream.
- `seed` (`int`): Shuffle seed.
- `shuffle_buffer_size` (`int`): Streaming shuffle buffer size.

### `LocalTextDataset(location: str | os.PathLike, tokenizer, context_length: int, skip_batches: int = 0, shuffle: bool = False, seed: int = 42)`

**Bases:** `olm.data.datasets.base_dataset.BaseTextDataset`

Source: [`src/olm/data/datasets/local_dataset.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/local_dataset.py#L8)

Dataset that streams text from local .txt files in a directory.

``LocalTextDataset`` scans ``location`` for ``.txt`` files, streams each
non-empty line, tokenizes through ``BaseTextDataset``, and yields causal
language-model samples.

**Iteration**

Yields ``(input_ids, labels)`` tensors shaped ``[context_length]``.

**Parameters**

- `location`: Directory containing ``.txt`` files.
- `tokenizer`: Tokenizer with an ``encode`` method.
- `context_length` (`int`): Number of input tokens per sample.
- `skip_batches` (`int`): Number of samples to skip before yielding.
- `shuffle` (`bool`): Whether to shuffle file order deterministically.
- `seed` (`int`): Shuffle seed.
