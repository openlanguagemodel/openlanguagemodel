from olm.nn.feedforward.moe_base import MoEFeedForwardBase
from olm.nn.feedforward.classic_ffn import ClassicFFN

class ClassicMoEFFN(MoEFeedForwardBase):
    """
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
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_experts: int = 8, 
        num_shared_experts: int = 0,
        top_k: int = 2,
        hidden_dim: int = None,
        activation_fn = None,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        expert_kwargs = {
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "bias": bias
        }
        if activation_fn is not None:
            expert_kwargs["activation_fn"] = activation_fn
            
        super().__init__(
            embed_dim=embed_dim,
            expert_cls=ClassicFFN,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            expert_kwargs=expert_kwargs,
            **kwargs
        )
