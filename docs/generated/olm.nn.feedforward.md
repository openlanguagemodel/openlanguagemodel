# olm.nn.feedforward

### *class* olm.nn.feedforward.ClassicFFN(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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
  [Linear](olm.nn.md#olm.nn.Linear)

#### act

Activation function.

* **Type:**
  nn.Module

#### down_proj

Projection from hidden dim to embedding dim.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

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

### *class* olm.nn.feedforward.FeedForwardBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.feedforward.GeGLUFFN(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.feedforward.SwiGLUFFN(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`FeedForwardBase`](olm.nn.feedforward.base.md#olm.nn.feedforward.base.FeedForwardBase)

SwiGLU-based feed-forward network used in modern Transformers (e.g., LLaMA, PaLM).

This layer implements the gated linear unit with Swish (SiLU) activation, which has been
shown to improve performance over standard GELU/ReLU FFNs.

Structure:
: Input
  -> Linear(embed_dim -> 2 \* hidden_dim) [Splits into Gate and Value]
  -> SwiGLU(Gate \* SiLU(Value))
  -> Linear(hidden_dim -> embed_dim)
  -> Dropout

* **Parameters:**
  * **embed_dim** (*int*) – The dimension of the input and output.
  * **hidden_dim** (*int* *,* *optional*) – The intermediate inner dimension.
    If None, defaults to int(ff_multiplier \* embed_dim).
  * **dropout** (*float* *,* *optional*) – Dropout probability. Defaults to 0.0.
  * **bias** (*bool* *,* *optional*) – Whether to use bias in linear layers. Defaults to True.
  * **ff_multiplier** (*float* *,* *optional*) – Multiplier for default hidden dimension. Defaults to 2.5 (commonly 8/3 for SwiGLU).

#### up_proj

Projects and splits input into gate and value parts.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

#### act

The activation function.

* **Type:**
  [SwiGLU](olm.nn.activations.md#olm.nn.activations.SwiGLU)

#### down_proj

Projects back to embedding dimension.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

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

### Modules

| [`base`](olm.nn.feedforward.base.md#module-olm.nn.feedforward.base)                      |    |
|------------------------------------------------------------------------------------------|----|
| [`classic_ffn`](olm.nn.feedforward.classic_ffn.md#module-olm.nn.feedforward.classic_ffn) |    |
| [`geglu_ffn`](olm.nn.feedforward.geglu_ffn.md#module-olm.nn.feedforward.geglu_ffn)       |    |
| [`swiglu_ffn`](olm.nn.feedforward.swiglu_ffn.md#module-olm.nn.feedforward.swiglu_ffn)    |    |
