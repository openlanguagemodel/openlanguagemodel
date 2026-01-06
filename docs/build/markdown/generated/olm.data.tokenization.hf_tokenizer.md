# olm.data.tokenization.hf_tokenizer

### Classes

| [`HFTokenizer`](#olm.data.tokenization.hf_tokenizer.HFTokenizer)(model_path)   |    |
|--------------------------------------------------------------------------------|----|

### *class* olm.data.tokenization.hf_tokenizer.HFTokenizer(model_path: str)

Bases: [`TokenizerBase`](olm.data.tokenization.base.md#olm.data.tokenization.base.TokenizerBase)

#### decode(tokens: torch.Tensor) → str

Decodes a single 1D tensor of token IDs back into a string.

#### encode(text: str) → torch.Tensor

Encodes a single string into a 1D PyTorch tensor of input IDs.
Padding is implicitly disabled for single inputs.

### *class* olm.data.tokenization.hf_tokenizer.TokenizerBase

Bases: [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for all tokenizers in OLM.

Defines the interface for converting between text strings and integer token IDs.
Subclasses must implement encode and decode methods.

#### *abstractmethod* decode(tokens: torch.Tensor) → str

Converts a sequence of token IDs back into a text string.

* **Parameters:**
  **tokens** (*torch.Tensor*) – A 1D tensor or list of token IDs.
* **Returns:**
  The decoded text string.
* **Return type:**
  str

#### *abstractmethod* encode(text: str) → torch.Tensor

Converts a text string into a sequence of token IDs.

* **Parameters:**
  **text** (*str*) – The input text to tokenize.
* **Returns:**
  A 1D tensor containing the token IDs.
* **Return type:**
  torch.Tensor
