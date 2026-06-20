# `olm.data.tokenization.base`

Source: [`src/olm/data/tokenization/base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/base.py#L1)

## Classes

### `TokenizerBase()`

**Bases:** `ABC`

Source: [`src/olm/data/tokenization/base.py:5`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/base.py#L5)

Abstract base class for all tokenizers in OLM.

Defines the interface for converting between text strings and integer token IDs.
Subclasses must implement `encode` and `decode` methods.

#### Methods

##### `decode(self, tokens: torch.Tensor) -> str`

Source: [`src/olm/data/tokenization/base.py:32`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/base.py#L32)

Converts a sequence of token IDs back into a text string.

**Parameters**

- `tokens` (`torch.Tensor`): A 1D tensor or list of token IDs.

**Returns**

- `str`: The decoded text string.

##### `encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor`

Source: [`src/olm/data/tokenization/base.py:17`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/base.py#L17)

Converts a text string into a sequence of token IDs.

**Parameters**

- `text` (`str`): The input text to tokenize.
- `add_special_tokens` (`bool`): Whether to include tokenizer-specific special tokens such as BOS/EOS markers.

**Returns**

- `torch.Tensor`: A 1D tensor containing the token IDs.

##### `save(self, path: str) -> None`

Source: [`src/olm/data/tokenization/base.py:45`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/base.py#L45)

Saves the tokenizer to a file.

**Parameters**

- `path` (`str`): Path to save the tokenizer to.

**Returns**

None
