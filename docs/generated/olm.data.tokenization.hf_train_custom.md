# `olm.data.tokenization.hf_train_custom`

Source: [`src/olm/data/tokenization/hf_train_custom.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/hf_train_custom.py#L1)

## Classes

### `HFTokenizerTrainCustom(files: List[str], vocab_size: int, special_tokens: List[str], save_location: str, unk_token: str = '[UNK]')`

**Bases:** `olm.data.tokenization.base.TokenizerBase`

Source: [`src/olm/data/tokenization/hf_train_custom.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/hf_train_custom.py#L9)

#### Methods

##### `decode(self, tokens: torch.Tensor) -> str`

Source: [`src/olm/data/tokenization/hf_train_custom.py:27`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/hf_train_custom.py#L27)

Decodes a single 1D tensor of token IDs back into a string.

##### `encode(self, text: str) -> torch.Tensor`

Source: [`src/olm/data/tokenization/hf_train_custom.py:18`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/data/tokenization/hf_train_custom.py#L18)

Encodes a single string into a 1D PyTorch tensor of input IDs.
Padding is implicitly disabled for single inputs.
