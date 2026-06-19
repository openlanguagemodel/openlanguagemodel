# `olm.nn.structure.block`

Source: [`src/olm/nn/structure/block.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L1)

## Functions

### `load(path: str) -> ForwardRef('Block') | tuple`

Source: [`src/olm/nn/structure/block.py:49`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L49)

### `load_block(path: str) -> ForwardRef('Block') | tuple`

Source: [`src/olm/nn/structure/block.py:49`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L49)

### `load_model(path: str) -> ForwardRef('Block') | tuple`

Source: [`src/olm/nn/structure/block.py:49`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L49)

## Classes

### `Block(blocks: List[torch.nn.modules.module.Module])`

**Bases:** `Module`

Source: [`src/olm/nn/structure/block.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L8)

Lightweight sequential container for composable submodules.

Similar to ``nn.Sequential``, but exposes the underlying list for
inspection or dynamic manipulation by higher-level builders.

Args:
    blocks: Ordered list of modules applied to the input in sequence.

Attributes:
    blocks: ModuleList storing the ordered blocks.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

##### `save(self, path: str, tokenizer: olm.data.tokenization.base.TokenizerBase = None) -> None`

Source: [`src/olm/nn/structure/block.py:40`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L40)
