# olm.data.datasets.data_loader

DataLoader wrapper for OLM library.

This module provides a clean wrapper around PyTorch’s DataLoader with
sensible defaults for language model training and convenient helpers.

### Classes

| [`DataLoader`](#olm.data.datasets.data_loader.DataLoader)(\*args, \*\*kwargs)   | Wrapper around PyTorch's DataLoader with sensible defaults for LLM training.   |
|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------|

### *class* olm.data.datasets.data_loader.DataLoader(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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
