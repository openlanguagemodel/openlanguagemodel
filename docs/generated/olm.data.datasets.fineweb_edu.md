# `olm.data.datasets.fineweb_edu`

Source: [`src/olm/data/datasets/fineweb_edu.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/fineweb_edu.py#L1)

## Classes

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
