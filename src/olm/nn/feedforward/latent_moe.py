import torch

from olm.nn.feedforward.moe_base import MoEFeedForwardBase
from olm.nn.feedforward.classic_ffn import ClassicFFN
from olm.nn.torch_nn_wrappers import Linear


class LatentMoEFFN(MoEFeedForwardBase):
    """
    Mixture-of-Experts feed-forward with a compressed latent bottleneck.

    Tokens are projected into a low-rank latent space before expert
    computation, then the combined expert output is projected back up -- so
    every routed expert operates on ``latent_dim`` rather than the full
    ``embed_dim``, shrinking the FLOPs of a very wide expert bank. Routing
    itself is *not* compressed: the router scores the full-width hidden state,
    so expert selection sees the uncompressed representation. The shared expert
    (always active) likewise runs at full width. Used by Nemotron 3 Super.

    Structure:
        shared = SharedExperts(x)                          # full embed_dim
        weights = Router(x)                                # full embed_dim
        routed = up_proj(experts(down_proj(x), weights))    # latent_dim bottleneck
        return routed + shared

    Args:
        embed_dim: Model hidden dimension (the layer's input/output width).
        latent_dim: Bottleneck dimension routed experts operate in.
        num_experts: Total number of routable experts.
        num_shared_experts: Always-active experts, run at full ``embed_dim``.
        top_k: Number of experts routed to per token.
        hidden_dim: FFN hidden dim of each routed (latent-space) expert.
        shared_hidden_dim: FFN hidden dim of each shared (full-width) expert.
        activation_fn: Activation module shared by routed and shared experts.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
        routed_scaling_factor: Constant multiplier on the routing weights.
        router_kwargs: Routing options forwarded to ``olm.nn.moe.MoERouter``.
        expert_cls: FFN class instantiated for each expert.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int,
        num_experts: int = 8,
        num_shared_experts: int = 0,
        top_k: int = 2,
        hidden_dim: int = None,
        shared_hidden_dim: int = None,
        activation_fn=None,
        dropout: float = 0.0,
        bias: bool = False,
        routed_scaling_factor: float = 1.0,
        router_kwargs: dict = None,
        expert_cls=ClassicFFN,
    ):
        expert_kwargs = {
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "bias": bias,
        }
        if activation_fn is not None:
            expert_kwargs["activation_fn"] = activation_fn

        shared_expert_kwargs = expert_kwargs
        if shared_hidden_dim is not None:
            shared_expert_kwargs = {**expert_kwargs, "hidden_dim": shared_hidden_dim}

        # Routed experts live in the latent space; the router and the shared
        # experts stay at full width, outside the bottleneck.
        super().__init__(
            embed_dim=latent_dim,
            expert_cls=expert_cls,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            expert_kwargs=expert_kwargs,
            shared_expert_kwargs=shared_expert_kwargs,
            shared_embed_dim=embed_dim,
            router_embed_dim=embed_dim,
            router_kwargs=router_kwargs,
            routed_scaling_factor=routed_scaling_factor,
        )

        # ``FeedForwardBase.embed_dim`` documents this layer's input/output
        # width, which is the model dimension -- not the routing dimension.
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.down_proj = Linear(embed_dim, latent_dim, bias=False)
        self.up_proj = Linear(latent_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[batch, seq_len, embed_dim]``

        Returns:
            ``[batch, seq_len, embed_dim]``
        """
        # Routing is scored on the full-width hidden state; only the expert
        # branch goes through the latent bottleneck.
        routed_out = self.up_proj(
            self.compute_routed(self.down_proj(x), router_input=x)
        )
        return routed_out + self.compute_shared(x)
