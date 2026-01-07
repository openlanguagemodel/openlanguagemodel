# olm.data.datasets.hf_dataset

### Classes

| [`FineWebEduDataset`](#olm.data.datasets.hf_dataset.FineWebEduDataset)(\*args, \*\*kwargs)           | FineWeb Edu dataset configuration.                     |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| [`HuggingFaceTextDataset`](#olm.data.datasets.hf_dataset.HuggingFaceTextDataset)(\*args, \*\*kwargs) | Generic dataset loader for Hugging Face text datasets. |

### *class* olm.data.datasets.hf_dataset.Any(\*args, \*\*kwargs)

Bases: `object`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### *class* olm.data.datasets.hf_dataset.BaseTextDataset(\*args: [Any](#olm.data.datasets.hf_dataset.Any), \*\*kwargs: [Any](#olm.data.datasets.hf_dataset.Any))

Bases: `IterableDataset`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for text-based streaming datasets.

Handles tokenization buffering and sequence generation generically.
Subclasses must implement \_get_text_iterator to yield text chunks.

### *class* olm.data.datasets.hf_dataset.FineWebEduDataset(\*args: [Any](#olm.data.datasets.hf_dataset.Any), \*\*kwargs: [Any](#olm.data.datasets.hf_dataset.Any))

Bases: [`HuggingFaceTextDataset`](#olm.data.datasets.hf_dataset.HuggingFaceTextDataset)

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

### *class* olm.data.datasets.hf_dataset.HuggingFaceTextDataset(\*args: [Any](#olm.data.datasets.hf_dataset.Any), \*\*kwargs: [Any](#olm.data.datasets.hf_dataset.Any))

Bases: [`BaseTextDataset`](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.BaseTextDataset)

Generic dataset loader for Hugging Face text datasets.

Inherits from BaseTextDataset to share token buffering logic and multi-worker safety.
