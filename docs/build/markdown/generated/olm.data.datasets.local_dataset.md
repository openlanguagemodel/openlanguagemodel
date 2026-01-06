# olm.data.datasets.local_dataset

### Classes

| [`LocalTextDataset`](#olm.data.datasets.local_dataset.LocalTextDataset)(\*args, \*\*kwargs)   | Dataset that streams text from local .txt files in a directory.   |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------|

### *class* olm.data.datasets.local_dataset.BaseTextDataset(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `IterableDataset`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for text-based streaming datasets.

Handles tokenization buffering and sequence generation generically.
Subclasses must implement \_get_text_iterator to yield text chunks.

### *class* olm.data.datasets.local_dataset.LocalTextDataset(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`BaseTextDataset`](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.BaseTextDataset)

Dataset that streams text from local .txt files in a directory.

### *class* olm.data.datasets.local_dataset.Union

Bases: `object`

Represent a union type

E.g. for int | str
