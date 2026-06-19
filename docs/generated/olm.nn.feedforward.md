# `olm.nn.feedforward`

Source: [`src/olm/nn/feedforward/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/__init__.py#L1)

## Classes

### `ClassicFFN(embed_dim, hidden_dim=None, activation_fn=None, dropout=0.0, bias=True)`

**Bases:** `olm.nn.feedforward.base.FeedForwardBase`

Source: [`src/olm/nn/feedforward/classic_ffn.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/classic_ffn.py#L7)

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

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/feedforward/classic_ffn.py:51`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/classic_ffn.py#L51)

Apply the position-wise feed-forward network.

Args:
    x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

Returns:
    torch.Tensor: Hidden states shaped ``[batch, seq_len, embed_dim]``.

### `ClassicMoEFFN(embed_dim: int, num_experts: int = 8, num_shared_experts: int = 0, top_k: int = 2, hidden_dim: int = None, activation_fn=None, dropout: float = 0.0, bias: bool = True, **kwargs)`

**Bases:** `olm.nn.feedforward.moe_base.MoEFeedForwardBase`

Source: [`src/olm/nn/feedforward/classic_moe.py:4`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/classic_moe.py#L4)

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

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `MoEFeedForwardBase`)

Source: [`src/olm/nn/feedforward/moe_base.py:100`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/moe_base.py#L100)

Forward pass with MoE routing.

Args:
    x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

Returns:
    torch.Tensor: Hidden states shaped ``[batch, seq_len, embed_dim]``.

### `FeedForwardBase(embed_dim: int, **kwargs)`

**Bases:** `Module`, `ABC`

Source: [`src/olm/nn/feedforward/base.py:5`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/base.py#L5)

Abstract base class for feedforward networks in a transformer block.

Defines the interface for FFNs/MLPs. Subclasses must implement the `forward` method.

Attributes:
    embed_dim (int): The input and output dimension.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/feedforward/base.py:25`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/base.py#L25)

Forward pass of the feedforward network.

Args:
    x (torch.Tensor): Input tensor of shape (batch, seq_len, embed_dim).

Returns:
    torch.Tensor: Output tensor of shape (batch, seq_len, embed_dim).

### `GeGLUFFN(embed_dim: int, hidden_dim: int = None, dropout: float = 0.0, bias: bool = True, ff_multiplier: float = 4.0)`

**Bases:** `olm.nn.feedforward.base.FeedForwardBase`

Source: [`src/olm/nn/feedforward/geglu_ffn.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/geglu_ffn.py#L8)

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

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/feedforward/geglu_ffn.py:54`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/geglu_ffn.py#L54)

Apply GeGLU feed-forward projection.

Args:
    x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

Returns:
    torch.Tensor: Hidden states shaped ``[batch, seq_len, embed_dim]``.

### `GeGLUMoEFFN(embed_dim: int, num_experts: int = 8, num_shared_experts: int = 0, top_k: int = 2, hidden_dim: int = None, dropout: float = 0.0, bias: bool = True, ff_multiplier: float = 4.0, **kwargs)`

**Bases:** `olm.nn.feedforward.moe_base.MoEFeedForwardBase`

Source: [`src/olm/nn/feedforward/geglu_moe.py:4`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/geglu_moe.py#L4)

Mixture of Experts version of GeGLUFFN.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `MoEFeedForwardBase`)

Source: [`src/olm/nn/feedforward/moe_base.py:100`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/moe_base.py#L100)

Forward pass with MoE routing.

Args:
    x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

Returns:
    torch.Tensor: Hidden states shaped ``[batch, seq_len, embed_dim]``.

### `SwiGLUFFN(embed_dim: int, hidden_dim: int = None, dropout: float = 0.0, bias: bool = True, ff_multiplier: float = 2.5)`

**Bases:** `olm.nn.feedforward.base.FeedForwardBase`

Source: [`src/olm/nn/feedforward/swiglu_ffn.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/swiglu_ffn.py#L8)

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

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/feedforward/swiglu_ffn.py:68`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/swiglu_ffn.py#L68)

Apply SwiGLU feed-forward projection.

Args:
    x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

Returns:
    torch.Tensor: Hidden states shaped ``[batch, seq_len, embed_dim]``.

### `SwiGLUMoEFFN(embed_dim: int, num_experts: int = 8, num_shared_experts: int = 0, top_k: int = 2, hidden_dim: int = None, dropout: float = 0.0, bias: bool = True, ff_multiplier: float = 2.5, **kwargs)`

**Bases:** `olm.nn.feedforward.moe_base.MoEFeedForwardBase`

Source: [`src/olm/nn/feedforward/swiglu_moe.py:4`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/swiglu_moe.py#L4)

Mixture of Experts version of SwiGLUFFN.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `MoEFeedForwardBase`)

Source: [`src/olm/nn/feedforward/moe_base.py:100`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/moe_base.py#L100)

Forward pass with MoE routing.

Args:
    x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

Returns:
    torch.Tensor: Hidden states shaped ``[batch, seq_len, embed_dim]``.
