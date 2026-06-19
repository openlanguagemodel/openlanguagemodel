# `olm.nn.torch_nn_wrappers`

Source: [`src/olm/nn/torch_nn_wrappers.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/torch_nn_wrappers.py#L1)

Thin wrappers around torch.nn modules.

Example::
    Block([
        Embedding(vocab_size, embed_dim),
        AbsolutePositionalEmbedding(max_seq_len, embed_dim, dropout),
    ])

## Classes

### `Linear(*args, **kwargs)`

**Bases:** `Linear`

Source: [`src/olm/nn/torch_nn_wrappers.py:12`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/torch_nn_wrappers.py#L12)

Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

Args:
    in_features: size of each input sample
    out_features: size of each output sample
    bias: If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``

Shape:
    - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
      dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
    - Output: :math:`(*, H_\text{out})` where all but the last dimension
      are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

Attributes:
    weight: the learnable weights of the module of shape
        :math:`(\text{out\_features}, \text{in\_features})`. The values are
        initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
        :math:`k = \frac{1}{\text{in\_features}}`
    bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
            If :attr:`bias` is ``True``, the values are initialized from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{\text{in\_features}}`

Examples::

    >>> m = nn.Linear(20, 30)
    >>> input = torch.randn(128, 20)
    >>> output = m(input)
    >>> print(output.size())
    torch.Size([128, 30])

#### Methods

##### `extra_repr(self) -> str`

Source: [`.venv/lib/python3.14/site-packages/torch/nn/modules/linear.py:136`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/.venv/lib/python3.14/site-packages/torch/nn/modules/linear.py#L136)

Return the extra representation of the module.

##### `forward(self, x)`

Source: [`src/olm/nn/torch_nn_wrappers.py:16`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/torch_nn_wrappers.py#L16)

Runs the forward pass.

##### `reset_parameters(self) -> None`

Source: [`.venv/lib/python3.14/site-packages/torch/nn/modules/linear.py:117`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/.venv/lib/python3.14/site-packages/torch/nn/modules/linear.py#L117)

Resets parameters based on their initialization used in ``__init__``.
