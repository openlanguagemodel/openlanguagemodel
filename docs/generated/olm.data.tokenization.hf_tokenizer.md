# `olm.data.tokenization.hf_tokenizer`

## Classes

### `HFTokenizer(model_path: str)`

Abstract base class for all tokenizers in OLM.

Defines the interface for converting between text strings and integer token IDs.
Subclasses must implement `encode` and `decode` methods.

#### Methods

- `decode(self, tokens: torch.Tensor) -> str`
  Decodes a single 1D tensor of token IDs back into a string.
- `encode(self, text: str) -> torch.Tensor`
  Encodes a single string into a 1D PyTorch tensor of input IDs.  Padding is implicitly disabled for single inputs.
- `save(self, path: str) -> None`
  Saves tokenizer in HuggingFace format. `path` must be a directory.
