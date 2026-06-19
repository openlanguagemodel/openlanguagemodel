# `olm.data.datasets.fineweb_edu`

## Classes

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
