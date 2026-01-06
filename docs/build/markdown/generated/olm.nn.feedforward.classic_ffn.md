# olm.nn.feedforward.classic_ffn

### Classes

| [`ClassicFFN`](#olm.nn.feedforward.classic_ffn.ClassicFFN)(\*args, \*\*kwargs)   | Standard Multi-Layer Perceptron (MLP) used in Transformer blocks.   |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------|

### *class* olm.nn.feedforward.classic_ffn.ClassicFFN(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`FeedForwardBase`](olm.nn.feedforward.base.md#olm.nn.feedforward.base.FeedForwardBase)

Standard Multi-Layer Perceptron (MLP) used in Transformer blocks.

Implements a position-wise feed-forward network consisting of two linear transformations
with a non-linear activation function in between.

Structure:
: Input -> Linear(embed_dim -> hidden_dim) -> Activation -> Dropout -> Linear(hidden_dim -> embed_dim) -> Dropout

#### hidden_dim

Dimension of the inner hidden layer.

* **Type:**
  int

#### up_proj

Projection from embedding dim to hidden dim.

* **Type:**
  [Linear](#olm.nn.feedforward.classic_ffn.Linear)

#### act

Activation function.

* **Type:**
  nn.Module

#### down_proj

Projection from hidden dim to embedding dim.

* **Type:**
  [Linear](#olm.nn.feedforward.classic_ffn.Linear)

#### dropout

Dropout layer.

* **Type:**
  nn.Dropout

#### forward(x)

Forward pass of the feedforward network.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch, seq_len, embed_dim).
* **Returns:**
  Output tensor of shape (batch, seq_len, embed_dim).
* **Return type:**
  torch.Tensor

### *class* olm.nn.feedforward.classic_ffn.FeedForwardBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for feedforward networks in a transformer block.

Defines the interface for FFNs/MLPs. Subclasses must implement the forward method.

#### embed_dim

The input and output dimension.

* **Type:**
  int

#### *abstractmethod* forward(x: torch.Tensor) → torch.Tensor

Forward pass of the feedforward network.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch, seq_len, embed_dim).
* **Returns:**
  Output tensor of shape (batch, seq_len, embed_dim).
* **Return type:**
  torch.Tensor

### *class* olm.nn.feedforward.classic_ffn.GELU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

GELU activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.feedforward.classic_ffn.Linear(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Linear`

#### forward(x)
