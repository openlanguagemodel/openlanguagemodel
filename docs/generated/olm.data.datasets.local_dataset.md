# `olm.data.datasets.local_dataset`

Source: [`src/olm/data/datasets/local_dataset.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/local_dataset.py#L1)

## Classes

### `LocalTextDataset(location: str | os.PathLike, tokenizer, context_length: int, skip_batches: int = 0, shuffle: bool = False, seed: int = 42)`

**Bases:** `olm.data.datasets.base_dataset.BaseTextDataset`

Source: [`src/olm/data/datasets/local_dataset.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/datasets/local_dataset.py#L8)

Dataset that streams text from local .txt files in a directory.

``LocalTextDataset`` scans ``location`` for ``.txt`` files, streams each
non-empty line, tokenizes through ``BaseTextDataset``, and yields causal
language-model samples.

Iteration:
    Yields ``(input_ids, labels)`` tensors shaped ``[context_length]``.

Args:
    location: Directory containing ``.txt`` files.
    tokenizer: Tokenizer with an ``encode`` method.
    context_length (int): Number of input tokens per sample.
    skip_batches (int): Number of samples to skip before yielding.
    shuffle (bool): Whether to shuffle file order deterministically.
    seed (int): Shuffle seed.
