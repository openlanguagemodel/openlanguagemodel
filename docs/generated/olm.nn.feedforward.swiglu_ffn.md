# olm.nn.feedforward.swiglu_ffn

### Classes

| [`SwiGLUFFN`](#olm.nn.feedforward.swiglu_ffn.SwiGLUFFN)(\*args, \*\*kwargs)   | SwiGLU-based feed-forward network used in modern Transformers (e.g., LLaMA, PaLM).   |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|

### *class* olm.nn.feedforward.swiglu_ffn.FeedForwardBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.feedforward.swiglu_ffn.Linear(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Linear`

#### forward(x)

### *class* olm.nn.feedforward.swiglu_ffn.SwiGLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

SwiGLU activation function.

Implements the SwiGLU activation as described in “GLU Variants Improve Transformer”.
It applies the SiLU activation to one half of the input (the gate) and multiplies it
by the other half (the value).

Equation:
: SwiGLU(x, W, V) = Swish_1(xW) \* (xV)
  Here, we assume the input x is already projected/concatenated such that we chunk it.
  So: SwiGLU(x) = (x_1 \* SiLU(x_2)) where x = [x_1, x_2]

* **Parameters:**
  * **device** (*torch.device* *,* *optional*) – Target device.
  * **dtype** (*torch.dtype* *,* *optional*) – Target data type.

#### forward(x: torch.Tensor) → torch.Tensor

Forward pass of SwiGLU.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor. Expected to have an even last dimension size.
* **Returns:**
  Output tensor with half the last dimension of the input.
* **Return type:**
  torch.Tensor

### *class* olm.nn.feedforward.swiglu_ffn.SwiGLUFFN(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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
  [Linear](#olm.nn.feedforward.swiglu_ffn.Linear)

#### act

The activation function.

* **Type:**
  [SwiGLU](#olm.nn.feedforward.swiglu_ffn.SwiGLU)

#### down_proj

Projects back to embedding dimension.

* **Type:**
  [Linear](#olm.nn.feedforward.swiglu_ffn.Linear)

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
