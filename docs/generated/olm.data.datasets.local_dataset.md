# `olm.data.datasets.local_dataset`

## Classes

### `LocalTextDataset(location: str | os.PathLike, tokenizer, context_length: int, skip_batches: int = 0, shuffle: bool = False, seed: int = 42)`

Dataset that streams text from local .txt files in a directory.
