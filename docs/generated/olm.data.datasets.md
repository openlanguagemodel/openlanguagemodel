# olm.data.datasets

### *class* olm.data.datasets.BaseTextDataset(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `IterableDataset`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for text-based streaming datasets.

Handles tokenization buffering and sequence generation generically.
Subclasses must implement \_get_text_iterator to yield text chunks.

### *class* olm.data.datasets.DataLoader(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `DataLoader`

Wrapper around PyTorch’s DataLoader with sensible defaults for LLM training.

This class extends torch.utils.data.DataLoader with:
- Better defaults for language model training
- Automatic worker configuration
- Pin memory optimization for GPU training
- Persistent workers for efficiency

* **Parameters:**
  * **dataset** – Dataset to load from (can be map-style or iterable).
  * **batch_size** – Number of samples per batch (default: 8).
  * **shuffle** – Whether to shuffle data at every epoch (default: False for iterable datasets).
  * **num_workers** – Number of worker processes for data loading (default: 0).
  * **pin_memory** – If True, tensors are copied to CUDA pinned memory (default: True).
  * **drop_last** – Drop the last incomplete batch if dataset size is not divisible by batch_size.
  * **persistent_workers** – Keep workers alive between epochs for faster startup (default: True if num_workers > 0).
  * **prefetch_factor** – Number of batches to prefetch per worker (default: 2).
  * **collate_fn** – Function to merge samples into batches.
  * **\*\*kwargs** – Additional arguments passed to torch.utils.data.DataLoader.

### Example

```pycon
>>> from olm.data.datasets import DataLoader
>>> loader = DataLoader(
...     dataset=my_dataset,
...     batch_size=16,
...     num_workers=4,
...     pin_memory=True,
... )
>>> for batch in loader:
...     # Training loop
...     pass
```

### *class* olm.data.datasets.FineWebEduDataset(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`HuggingFaceTextDataset`](olm.data.datasets.hf_dataset.md#olm.data.datasets.hf_dataset.HuggingFaceTextDataset)

FineWeb Edu dataset configuration.

* **Parameters:**
  * **split** – Dataset split (‘train’ or ‘validation’)
  * **context_length** – Sequence length for training (default: 1024)
  * **subset** – Dataset subset to use (default: ‘sample-10BT’)
  * **tokenizer** – Tokenizer object (e.g. from AutoTokenizer)
  * **streaming** – Whether to use streaming mode (default: True)
  * **shuffle** – Whether to shuffle the dataset (default: False)
  * **seed** – Random seed for shuffling (default: 42)
  * **cache_dir** – Directory to cache downloaded data (default: None)
  * **skip_batches** – Number of batches to skip

### *class* olm.data.datasets.HuggingFaceTextDataset(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`BaseTextDataset`](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.BaseTextDataset)

Generic dataset loader for Hugging Face text datasets.

Inherits from BaseTextDataset to share token buffering logic and multi-worker safety.

### *class* olm.data.datasets.LocalTextDataset(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`BaseTextDataset`](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.BaseTextDataset)

Dataset that streams text from local .txt files in a directory.

### Modules

| [`base_dataset`](olm.data.datasets.base_dataset.md#module-olm.data.datasets.base_dataset)    |                                     |
|----------------------------------------------------------------------------------------------|-------------------------------------|
| [`data_loader`](olm.data.datasets.data_loader.md#module-olm.data.datasets.data_loader)       | DataLoader wrapper for OLM library. |
| [`hf_dataset`](olm.data.datasets.hf_dataset.md#module-olm.data.datasets.hf_dataset)          |                                     |
| [`local_dataset`](olm.data.datasets.local_dataset.md#module-olm.data.datasets.local_dataset) |                                     |
