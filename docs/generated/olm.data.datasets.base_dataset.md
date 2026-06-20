# `olm.data.datasets.base_dataset`

Source: [`src/olm/data/datasets/base_dataset.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/base_dataset.py#L1)

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
