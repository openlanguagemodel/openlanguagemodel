# `olm.nn.feedforward`

## Classes

### `ClassicFFN(embed_dim, hidden_dim=None, activation_fn=None, dropout=0.0, bias=True)`

Standard Multi-Layer Perceptron (MLP) used in Transformer blocks.

Implements a position-wise feed-forward network consisting of two linear transformations
with a non-linear activation function in between.

Structure:
    Input -> Linear(embed_dim -> hidden_dim) -> Activation -> Dropout -> Linear(hidden_dim -> embed_dim) -> Dropout

Attributes:
    hidden_dim (int): Dimension of the inner hidden layer.
    up_proj (Linear): Projection from embedding dim to hidden dim.
    act (nn.Module): Activation function.
    down_proj (Linear): Projection from hidden dim to embedding dim.
    dropout (nn.Dropout): Dropout layer.

#### Methods

- `forward(self, x)`
  Forward pass of the feedforward network.

### `ClassicMoEFFN(embed_dim: int, num_experts: int = 8, num_shared_experts: int = 0, top_k: int = 2, hidden_dim: int = None, activation_fn=None, dropout: float = 0.0, bias: bool = True, **kwargs)`

Mixture of Experts version of ClassicFFN.

Args:
    embed_dim (int): Input and output dimension.
    num_experts (int): Number of experts.
    num_shared_experts (int): Number of shared experts.
    top_k (int): Number of experts to route to.
    hidden_dim (int, optional): Hidden dimension of each expert.
    activation_fn (nn.Module, optional): Activation function for experts.
    dropout (float, optional): Dropout probability.
    bias (bool, optional): Whether to use bias in linear layers.

### `FeedForwardBase(embed_dim: int, **kwargs)`

Abstract base class for feedforward networks in a transformer block.

Defines the interface for FFNs/MLPs. Subclasses must implement the `forward` method.

Attributes:
    embed_dim (int): The input and output dimension.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Forward pass of the feedforward network.

### `GeGLUFFN(embed_dim: int, hidden_dim: int = None, dropout: float = 0.0, bias: bool = True, ff_multiplier: float = 4.0)`

Feed-Forward Network using GeGLU activation.

Implements: x = DownProj(GeGLU(UpProj(x))).
UpProj expands to 2 * hidden_dim to support splitting for the gate.

Args:
    embed_dim (int): Input dimension.
    hidden_dim (int, optional): Hidden dimension. Defaults to 4 * embed_dim if None.
    dropout (float, optional): Dropout probability. Defaults to 0.0.
    bias (bool, optional): Whether to usage bias in linear layers. Defaults to True.
    ff_multiplier (float, optional): Expansion factor if hidden_dim is None. Defaults to 4.0.

#### Methods

- `forward(self, x)`
  Forward pass of the feedforward network.

### `GeGLUMoEFFN(embed_dim: int, num_experts: int = 8, num_shared_experts: int = 0, top_k: int = 2, hidden_dim: int = None, dropout: float = 0.0, bias: bool = True, ff_multiplier: float = 4.0, **kwargs)`

Mixture of Experts version of GeGLUFFN.

### `SwiGLUFFN(embed_dim: int, hidden_dim: int = None, dropout: float = 0.0, bias: bool = True, ff_multiplier: float = 2.5)`

SwiGLU-based feed-forward network used in modern Transformers (e.g., LLaMA, PaLM).

This layer implements the gated linear unit with Swish (SiLU) activation, which has been
shown to improve performance over standard GELU/ReLU FFNs.

Structure:
    Input
    -> Linear(embed_dim -> 2 * hidden_dim) [Splits into Gate and Value]
    -> SwiGLU(Gate * SiLU(Value))
    -> Linear(hidden_dim -> embed_dim)
    -> Dropout

Args:
    embed_dim (int): The dimension of the input and output.
    hidden_dim (int, optional): The intermediate inner dimension.
        If None, defaults to `int(ff_multiplier * embed_dim)`.
    dropout (float, optional): Dropout probability. Defaults to 0.0.
    bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
    ff_multiplier (float, optional): Multiplier for default hidden dimension. Defaults to 2.5 (commonly 8/3 for SwiGLU).

Attributes:
    up_proj (Linear): Projects and splits input into gate and value parts.
    act (SwiGLU): The activation function.
    down_proj (Linear): Projects back to embedding dimension.
    dropout (nn.Dropout): Dropout layer.

#### Methods

- `forward(self, x)`
  Forward pass of the feedforward network.

### `SwiGLUMoEFFN(embed_dim: int, num_experts: int = 8, num_shared_experts: int = 0, top_k: int = 2, hidden_dim: int = None, dropout: float = 0.0, bias: bool = True, ff_multiplier: float = 2.5, **kwargs)`

Mixture of Experts version of SwiGLUFFN.
