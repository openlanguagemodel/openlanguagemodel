# `olm.nn.structure.block`

## Functions

### `load(path: str) -> ForwardRef('Block') | tuple`

### `load_block(path: str) -> ForwardRef('Block') | tuple`

### `load_model(path: str) -> ForwardRef('Block') | tuple`

## Classes

### `Block(blocks: List[torch.nn.modules.module.Module])`

Lightweight sequential container for composable submodules.

Similar to ``nn.Sequential``, but exposes the underlying list for
inspection or dynamic manipulation by higher-level builders.

Args:
    blocks: Ordered list of modules applied to the input in sequence.

Attributes:
    blocks: ModuleList storing the ordered blocks.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Apply each block to the input in sequence.
- `save(self, path: str, tokenizer: olm.data.tokenization.base.TokenizerBase = None) -> None`
