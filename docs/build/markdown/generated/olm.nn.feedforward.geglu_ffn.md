# olm.nn.feedforward.geglu_ffn

### Classes

| [`GeGLUFFN`](#olm.nn.feedforward.geglu_ffn.GeGLUFFN)(\*args, \*\*kwargs)   | Feed-Forward Network using GeGLU activation.   |
|----------------------------------------------------------------------------|------------------------------------------------|

### *class* olm.nn.feedforward.geglu_ffn.FeedForwardBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.feedforward.geglu_ffn.GeGLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

GeGLU activation function.

Implements the GeGLU variant from “GLU Variants Improve Transformer”.
GeGLU(x, W, V) = GELU(xW) \* (xV)
Here: GeGLU(x) = GELU(gate) \* value

* **Parameters:**
  * **device** (*torch.device* *,* *optional*) – Target device.
  * **dtype** (*torch.dtype* *,* *optional*) – Target data type.

#### forward(x: torch.Tensor) → torch.Tensor

Forward pass of GeGLU.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor.
* **Returns:**
  Output tensor with half the last dimension.
* **Return type:**
  torch.Tensor

### *class* olm.nn.feedforward.geglu_ffn.GeGLUFFN(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`FeedForwardBase`](olm.nn.feedforward.base.md#olm.nn.feedforward.base.FeedForwardBase)

Feed-Forward Network using GeGLU activation.

Implements: x = DownProj(GeGLU(UpProj(x))).
UpProj expands to 2 \* hidden_dim to support splitting for the gate.

* **Parameters:**
  * **embed_dim** (*int*) – Input dimension.
  * **hidden_dim** (*int* *,* *optional*) – Hidden dimension. Defaults to 4 \* embed_dim if None.
  * **dropout** (*float* *,* *optional*) – Dropout probability. Defaults to 0.0.
  * **bias** (*bool* *,* *optional*) – Whether to usage bias in linear layers. Defaults to True.
  * **ff_multiplier** (*float* *,* *optional*) – Expansion factor if hidden_dim is None. Defaults to 4.0.

#### forward(x)

Forward pass of the feedforward network.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch, seq_len, embed_dim).
* **Returns:**
  Output tensor of shape (batch, seq_len, embed_dim).
* **Return type:**
  torch.Tensor

### *class* olm.nn.feedforward.geglu_ffn.Linear(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Linear`

#### forward(x)
