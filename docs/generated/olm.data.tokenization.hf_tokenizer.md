# `olm.data.tokenization.hf_tokenizer`

Source: [`src/olm/data/tokenization/hf_tokenizer.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/hf_tokenizer.py#L1)

## Classes

### `HFTokenizer(model_path: str)`

**Bases:** `olm.data.tokenization.base.TokenizerBase`

Source: [`src/olm/data/tokenization/hf_tokenizer.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/hf_tokenizer.py#L8)

#### Methods

##### `decode(self, tokens: torch.Tensor) -> str`

Source: [`src/olm/data/tokenization/hf_tokenizer.py:28`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/hf_tokenizer.py#L28)

Decodes a single 1D tensor of token IDs back into a string.

##### `encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor`

Source: [`src/olm/data/tokenization/hf_tokenizer.py:13`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/hf_tokenizer.py#L13)

Encodes a single string into a 1D PyTorch tensor of input IDs.
Padding is implicitly disabled for single inputs.

##### `save(self, path: str) -> None`

Source: [`src/olm/data/tokenization/hf_tokenizer.py:36`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/hf_tokenizer.py#L36)

Saves tokenizer in HuggingFace format.
`path` must be a directory.
