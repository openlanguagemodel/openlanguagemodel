# `olm.data.datasets.base_dataset`

## Classes

### `BaseTextDataset(tokenizer: Any, context_length: int, skip_batches: int = 0, shuffle: bool = False, seed: int = 42)`

Abstract base class for text-based streaming datasets.

Handles tokenization buffering and sequence generation generically.
Subclasses must implement `_get_text_iterator` to yield text chunks.
