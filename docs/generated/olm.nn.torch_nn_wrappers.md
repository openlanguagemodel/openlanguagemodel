# olm.nn.torch_nn_wrappers

Thin wrappers around torch.nn modules.

Example::
: Block([
  : Embedding(vocab_size, embed_dim),
    AbsolutePositionalEmbedding(max_seq_len, embed_dim, dropout),
  <br/>
  ])

### Classes

| [`Linear`](#olm.nn.torch_nn_wrappers.Linear)(\*args, \*\*kwargs)   |    |
|--------------------------------------------------------------------|----|

### *class* olm.nn.torch_nn_wrappers.Linear(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Linear`

#### forward(x)
