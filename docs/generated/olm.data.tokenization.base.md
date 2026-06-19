# `olm.data.tokenization.base`

Source: [`src/olm/data/tokenization/base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/base.py#L1)

## Classes

### `TokenizerBase()`

**Bases:** `ABC`

Source: [`src/olm/data/tokenization/base.py:3`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/base.py#L3)

Abstract base class for all tokenizers in OLM.

Defines the interface for converting between text strings and integer token IDs.
Subclasses must implement `encode` and `decode` methods.

#### Methods

##### `decode(self, tokens: torch.Tensor) -> str`

Source: [`src/olm/data/tokenization/base.py:27`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/base.py#L27)

Converts a sequence of token IDs back into a text string.

Args:
    tokens (torch.Tensor): A 1D tensor or list of token IDs.

Returns:
    str: The decoded text string.

##### `encode(self, text: str) -> torch.Tensor`

Source: [`src/olm/data/tokenization/base.py:14`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/base.py#L14)

Converts a text string into a sequence of token IDs.

Args:
    text (str): The input text to tokenize.

Returns:
    torch.Tensor: A 1D tensor containing the token IDs.

##### `save(self, path: str) -> None`

Source: [`src/olm/data/tokenization/base.py:40`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/base.py#L40)

Saves the tokenizer to a file.

Args:
    path (str): Path to save the tokenizer to.

Returns:
    None
