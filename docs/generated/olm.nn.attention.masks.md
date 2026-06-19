# `olm.nn.attention.masks`

## Functions

### `attention_mask_to_bool(mask: torch.Tensor, device=None) -> torch.Tensor`

Convert common attention mask conventions to a boolean keep-mask.

Supported conventions:
- bool masks: True means attend, False means block.
- binary numeric masks: 1 means attend, 0 means block.
- additive masks: 0/non-negative means attend, negative or -inf means block.
