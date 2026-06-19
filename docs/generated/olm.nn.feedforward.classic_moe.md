# `olm.nn.feedforward.classic_moe`

## Classes

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
