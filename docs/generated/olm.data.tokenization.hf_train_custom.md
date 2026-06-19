# `olm.data.tokenization.hf_train_custom`

## Classes

### `HFTokenizerTrainCustom(files: List[str], vocab_size: int, special_tokens: List[str], save_location: str, unk_token: str = '[UNK]')`

Abstract base class for all tokenizers in OLM.

Defines the interface for converting between text strings and integer token IDs.
Subclasses must implement `encode` and `decode` methods.

#### Methods

- `decode(self, tokens: torch.Tensor) -> str`
  Decodes a single 1D tensor of token IDs back into a string.
- `encode(self, text: str) -> torch.Tensor`
  Encodes a single string into a 1D PyTorch tensor of input IDs.  Padding is implicitly disabled for single inputs.
