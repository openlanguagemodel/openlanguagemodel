import torch
import torch.nn as nn

from olm.nn.feedforward.base import FeedForwardBase
from olm.nn.feedforward.classic_ffn import ClassicFFN
from olm.nn.feedforward.classic_moe import ClassicMoEFFN
from olm.nn.torch_nn_wrappers import Linear


class LatentMoEFFN(FeedForwardBase):
    """
    Mixture-of-Experts feed-forward with a compressed latent bottleneck.

    Tokens are projected into a low-rank latent space before routing and
    expert computation, then the combined expert output is projected back
    up -- so both the router and every routed expert operate on
    ``latent_dim`` rather than the full ``embed_dim``, shrinking the FLOPs of
    a very wide expert bank. The shared expert (always active) runs at full
    width, unaffected by the bottleneck. Used by Nemotron 3 Super.

    Structure:
        shared = SharedExpert(x)                          # full embed_dim
        routed = up_proj(MoE(down_proj(x)))                # latent_dim bottleneck
        return routed + shared

    Args:
        embed_dim: Model hidden dimension.
        latent_dim: Bottleneck dimension routed experts operate in.
        num_experts: Total number of routable experts.
        num_shared_experts: Always-active experts, run at full ``embed_dim``.
        top_k: Number of experts routed to per token.
        hidden_dim: FFN hidden dim of each routed (latent-space) expert.
        shared_hidden_dim: FFN hidden dim of each shared (full-width) expert.
        activation_fn: Activation module shared by routed and shared experts.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
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
    ):
        super().__init__(embed_dim)
        self.latent_dim = latent_dim

        self.down_proj = Linear(embed_dim, latent_dim, bias=False)
        self.up_proj = Linear(latent_dim, embed_dim, bias=False)

        self.routed = ClassicMoEFFN(
            latent_dim,
            num_experts=num_experts,
            num_shared_experts=0,
            top_k=top_k,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            dropout=dropout,
            bias=bias,
        )

        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [
                    ClassicFFN(
                        embed_dim,
                        hidden_dim=shared_hidden_dim,
                        activation_fn=activation_fn,
                        dropout=dropout,
                        bias=bias,
                    )
                    for _ in range(num_shared_experts)
                ]
            )
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[batch, seq_len, embed_dim]``

        Returns:
            ``[batch, seq_len, embed_dim]``
        """
        shared_out = torch.zeros_like(x)
        if self.shared_experts is not None:
            for expert in self.shared_experts:
                shared_out = shared_out + expert(x)

        routed_out = self.up_proj(self.routed(self.down_proj(x)))

        return routed_out + shared_out
