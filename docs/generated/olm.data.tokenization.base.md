# `olm.data.tokenization.base`

## Classes

### `TokenizerBase()`

Abstract base class for all tokenizers in OLM.

Defines the interface for converting between text strings and integer token IDs.
Subclasses must implement `encode` and `decode` methods.

#### Methods

- `decode(self, tokens: torch.Tensor) -> str`
  Converts a sequence of token IDs back into a text string.
- `encode(self, text: str) -> torch.Tensor`
  Converts a text string into a sequence of token IDs.
- `save(self, path: str) -> None`
  Saves the tokenizer to a file.
