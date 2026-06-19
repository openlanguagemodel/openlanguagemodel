# `olm.data.datasets.hf_dataset`

Source: [`src/olm/data/datasets/hf_dataset.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/hf_dataset.py#L1)

## Classes

### `FineWebEduDataset(tokenizer: Any, split: str = 'train', context_length: int = 1024, subset: str = 'sample-10BT', streaming: bool = True, shuffle: bool = False, seed: int = 42, cache_dir: str | None = None, skip_batches: int = 0)`

**Bases:** `olm.data.datasets.hf_dataset.HuggingFaceTextDataset`

Source: [`src/olm/data/datasets/hf_dataset.py:83`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/hf_dataset.py#L83)

Convenience wrapper for ``HuggingFaceFW/fineweb-edu``.

Iteration:
    Yields ``(input_ids, labels)`` tensors shaped ``[context_length]`` for
    causal language-model training.

Args:
    tokenizer: Tokenizer with an ``encode`` method.
    split: Dataset split ('train' or 'validation')
    context_length: Sequence length for training (default: 1024)
    subset: Dataset subset to use (default: 'sample-10BT')
    streaming: Whether to use streaming mode (default: True)
    shuffle: Whether to shuffle the dataset (default: False)
    seed: Random seed for shuffling (default: 42)
    cache_dir: Directory to cache downloaded data (default: None)
    skip_batches: Number of batches to skip

### `HuggingFaceTextDataset(dataset_name: str, split: str, context_length: int, text_fn: Callable[[Any], str], tokenizer: Any, dataset_kwargs: Dict[str, Any] | None = None, streaming: bool = True, skip_batches: int = 0, shuffle: bool = False, seed: int = 42, shuffle_buffer_size: int = 10000)`

**Bases:** `olm.data.datasets.base_dataset.BaseTextDataset`

Source: [`src/olm/data/datasets/hf_dataset.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/hf_dataset.py#L8)

Generic dataset loader for Hugging Face text datasets.

The Hugging Face dataset is read in streaming mode by default, then text is
extracted with ``text_fn`` and passed through ``BaseTextDataset`` for
tokenization, sharding, and next-token target construction.

Iteration:
    Yields ``(input_ids, labels)`` tensors shaped ``[context_length]``.

Args:
    dataset_name (str): Hugging Face dataset name.
    split (str): Dataset split, such as ``"train"``.
    context_length (int): Number of input tokens per sample.
    text_fn: Function that maps a dataset example to text.
    tokenizer: Tokenizer with an ``encode`` method.
    dataset_kwargs: Extra keyword arguments passed to ``load_dataset``.
    streaming (bool): Whether to use Hugging Face streaming mode.
    skip_batches (int): Number of samples to skip before yielding.
    shuffle (bool): Whether to shuffle the dataset stream.
    seed (int): Shuffle seed.
    shuffle_buffer_size (int): Streaming shuffle buffer size.
