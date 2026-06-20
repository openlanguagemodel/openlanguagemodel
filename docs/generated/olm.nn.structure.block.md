# `olm.nn.structure.block`

Source: [`src/olm/nn/structure/block.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L1)

## Functions

### `load(path: str, *, trusted: bool = True) -> ForwardRef('Block') | tuple`

Source: [`src/olm/nn/structure/block.py:56`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L56)

Load a block saved by ``Block.save``.

``Block.save`` stores the full Python module object so arbitrary custom
architectures can be restored. This uses Python pickle through PyTorch and is
only safe for trusted local artifacts.

**Parameters**

- `path`: Directory produced by ``Block.save``.
- `trusted`: Must be ``True``. Pass this explicitly in security-sensitive code to make the trusted-artifact boundary visible.

### `load_block(path: str, *, trusted: bool = True) -> ForwardRef('Block') | tuple`

Source: [`src/olm/nn/structure/block.py:56`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L56)

Load a block saved by ``Block.save``.

``Block.save`` stores the full Python module object so arbitrary custom
architectures can be restored. This uses Python pickle through PyTorch and is
only safe for trusted local artifacts.

**Parameters**

- `path`: Directory produced by ``Block.save``.
- `trusted`: Must be ``True``. Pass this explicitly in security-sensitive code to make the trusted-artifact boundary visible.

### `load_model(path: str, *, trusted: bool = True) -> ForwardRef('Block') | tuple`

Source: [`src/olm/nn/structure/block.py:56`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L56)

Load a block saved by ``Block.save``.

``Block.save`` stores the full Python module object so arbitrary custom
architectures can be restored. This uses Python pickle through PyTorch and is
only safe for trusted local artifacts.

**Parameters**

- `path`: Directory produced by ``Block.save``.
- `trusted`: Must be ``True``. Pass this explicitly in security-sensitive code to make the trusted-artifact boundary visible.

## Classes

### `Block(blocks: List[torch.nn.modules.module.Module])`

**Bases:** `Module`

Source: [`src/olm/nn/structure/block.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L8)

Lightweight sequential container for composable submodules.

Similar to ``nn.Sequential``, but exposes the underlying list for
inspection or dynamic manipulation by higher-level builders.

**Parameters**

- `blocks`: Ordered list of modules applied to the input in sequence.

**Attributes**

- `blocks`: ModuleList storing the ordered blocks.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

##### `save(self, path: str, tokenizer: olm.data.tokenization.base.TokenizerBase = None) -> None`

Source: [`src/olm/nn/structure/block.py:40`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L40)

Save this block as a trusted local artifact.

OLM preserves the complete Python module object so custom architectures can
round-trip without a separate config format. Only load artifacts you created
yourself or otherwise trust.
