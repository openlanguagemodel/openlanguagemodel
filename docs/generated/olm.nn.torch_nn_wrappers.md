# `olm.nn.torch_nn_wrappers`

Source: [`src/olm/nn/torch_nn_wrappers.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/torch_nn_wrappers.py#L1)

Thin wrappers around torch.nn modules.

**Example**

```python
Block([
    Embedding(vocab_size, embed_dim),
    AbsolutePositionalEmbedding(max_seq_len, embed_dim, dropout),
])
```

## Classes

### `Linear(*args, **kwargs)`

**Bases:** `Linear`

Source: [`src/olm/nn/torch_nn_wrappers.py:12`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/torch_nn_wrappers.py#L12)

#### Methods

##### `forward(self, x)`

Source: [`src/olm/nn/torch_nn_wrappers.py:16`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/torch_nn_wrappers.py#L16)
