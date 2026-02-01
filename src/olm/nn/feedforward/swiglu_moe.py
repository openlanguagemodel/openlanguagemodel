from olm.nn.feedforward.moe_base import MoEFeedForwardBase
from olm.nn.feedforward.swiglu_ffn import SwiGLUFFN

class SwiGLUMoEFFN(MoEFeedForwardBase):
    """
    Mixture of Experts version of SwiGLUFFN.
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_experts: int = 8, 
        num_shared_experts: int = 0,
        top_k: int = 2,
        hidden_dim: int = None,
        dropout: float = 0.0,
        bias: bool = True,
        ff_multiplier: float = 2.5,
        **kwargs
    ):
        expert_kwargs = {
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "bias": bias,
            "ff_multiplier": ff_multiplier
        }
            
        super().__init__(
            embed_dim=embed_dim,
            expert_cls=SwiGLUFFN,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            expert_kwargs=expert_kwargs,
            **kwargs
        )
