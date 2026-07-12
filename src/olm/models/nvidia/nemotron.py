import torch
import torch.nn as nn

from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual
from olm.nn.attention import GroupedQueryAttention, Mamba2Mixer
from olm.nn.feedforward import ClassicFFN, ClassicMoEFFN, LatentMoEFFN, FeedForwardBase
from olm.nn.feedforward.moe_base import MoERouter
from olm.nn.activations import ReLUSquared
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


class NemotronScaledRouter(MoERouter):
    """Softmax top-k router with a routed-output scaling factor.

    Identical interface to ``MoERouter`` but multiplies the normalized top-k
    weights by a fixed ``routed_scaling_factor``, matching Nemotron-H's
    ``routed_scaling_factor`` (2.5 for Nano, 5.0 for Super).

    Args:
        embed_dim (int): Dimension the router operates on (``latent_dim`` for
            LatentMoE layers, ``embed_dim`` otherwise).
        num_experts (int): Total number of routable experts.
        top_k (int): Number of experts each token is routed to.
        routed_scaling_factor (float): Multiplier applied to routing weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        top_k: int = 2,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__(embed_dim, num_experts, top_k)
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        top_k_indices, top_k_weights = super().forward(x)
        return top_k_indices, top_k_weights * self.routed_scaling_factor


class NemotronMoEFFN(FeedForwardBase):
    """Plain (non-latent) MoE with an independently-sized shared expert.

    Nemotron-H's routed and shared experts use different FFN widths
    (``moe_intermediate_size`` vs ``moe_shared_expert_intermediate_size``),
    which the generic ``ClassicMoEFFN`` can't express since it uses one
    ``hidden_dim`` for both. This composes a routed-only ``ClassicMoEFFN``
    with a separate list of full-width shared experts. Used by Nemotron Nano;
    Nemotron Super uses ``LatentMoEFFN`` instead (its MoE has a latent
    bottleneck the routed experts operate in).

    Args:
        embed_dim (int): Model dimension.
        num_experts (int): Total number of routable experts.
        num_shared_experts (int): Number of always-active shared experts.
        top_k (int): Number of experts routed to per token.
        hidden_dim (int): FFN hidden dim of each routed expert.
        shared_hidden_dim (int): FFN hidden dim of each shared expert.
        routed_scaling_factor (float): Multiplier on routed weights.
        activation_fn (nn.Module): Activation shared by routed and shared experts.
        bias (bool): Whether to use bias in linear layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        hidden_dim: int,
        shared_hidden_dim: int,
        routed_scaling_factor: float,
        activation_fn=None,
        bias: bool = False,
    ):
        super().__init__(embed_dim)
        self.routed = ClassicMoEFFN(
            embed_dim,
            num_experts=num_experts,
            num_shared_experts=0,
            top_k=top_k,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            bias=bias,
        )
        self.routed.router = NemotronScaledRouter(
            embed_dim, num_experts, top_k, routed_scaling_factor
        )

        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [
                    ClassicFFN(
                        embed_dim,
                        hidden_dim=shared_hidden_dim,
                        activation_fn=activation_fn,
                        bias=bias,
                    )
                    for _ in range(num_shared_experts)
                ]
            )
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.routed(x)
        if self.shared_experts is not None:
            for expert in self.shared_experts:
                out = out + expert(x)
        return out


class NemotronHModel(Block):
    """
    Base class for Nemotron-H / Nemotron 3 hybrid Mamba-Attention-MoE models.

    Unlike a standard Transformer, mixer and feed-forward are **not** paired
    per layer -- each layer is *either* a Mamba-2 mixer, a full-attention
    (GQA) mixer, *or* an MoE feed-forward, one component per layer, driven
    directly by ``hybrid_override_pattern`` (``'M'`` / ``'*'`` / ``'E'``
    respectively):

        x = x + Mamba2(RMSNorm(x))   # 'M'
        x = x + GQA(RMSNorm(x))      # '*'
        x = x + MoE(RMSNorm(x))      # 'E'

    Nemotron-H does not tie input/output embeddings; the named presets pass
    ``tie_weights=False``. The Multi-Token-Prediction head present in the
    Nemotron 3 Super reference checkpoint is a training/inference feature
    and is intentionally omitted here.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
        ``[batch, seq_len, vocab_size]``.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Model dimension.
        hybrid_override_pattern (str): One character per layer: ``'M'``
            (Mamba-2), ``'*'`` (full attention), or ``'E'`` (MoE feed-forward).
        num_heads (int): Number of query heads (attention layers).
        num_kv_heads (int): Number of key/value heads (attention layers, GQA).
        head_dim (int): Per-head dimension (attention layers).
        max_seq_len (int): Maximum context length.
        mamba_num_heads (int): Number of SSM heads (Mamba-2 layers).
        mamba_head_dim (int): Per-head dimension (Mamba-2 layers).
        ssm_state_size (int): Per-head recurrent state size (Mamba-2 layers).
        n_groups (int): Number of B/C groups (Mamba-2 layers).
        conv_kernel_size (int): Causal conv kernel size (Mamba-2 layers).
        num_experts (int): Total number of routable experts.
        num_shared_experts (int): Number of always-active shared experts.
        top_k (int): Number of experts routed to per token.
        moe_intermediate_size (int): FFN hidden dim of each routed expert.
        moe_shared_expert_intermediate_size (int): FFN hidden dim of each shared expert.
        routed_scaling_factor (float): Multiplier on MoE routing weights.
        moe_latent_size (int, optional): If set, MoE layers use a
            ``LatentMoEFFN`` bottleneck of this size (Nemotron Super) instead
            of the full-width ``NemotronMoEFFN`` (Nemotron Nano).
        rope_theta (float): RoPE base frequency (attention layers).
        dropout (float): Dropout probability.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
        tie_weights (bool): Whether to tie the output head to the embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hybrid_override_pattern: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        mamba_num_heads: int,
        mamba_head_dim: int,
        ssm_state_size: int,
        n_groups: int,
        conv_kernel_size: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        moe_intermediate_size: int,
        moe_shared_expert_intermediate_size: int,
        routed_scaling_factor: float,
        moe_latent_size: int = None,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-5,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)

        layer_blocks = []
        for layer_type in hybrid_override_pattern:
            if layer_type == "M":
                mixer = Mamba2Mixer(
                    embed_dim,
                    mamba_num_heads,
                    mamba_head_dim,
                    state_size=ssm_state_size,
                    n_groups=n_groups,
                    conv_kernel_size=conv_kernel_size,
                    rms_norm_eps=rms_norm_eps,
                )
            elif layer_type == "*":
                mixer = GroupedQueryAttention(
                    embed_dim,
                    num_heads,
                    num_kv_heads,
                    max_seq_len,
                    head_dim=head_dim,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    use_bias=False,
                )
            elif layer_type == "E":
                if moe_latent_size is not None:
                    mixer = LatentMoEFFN(
                        embed_dim,
                        moe_latent_size,
                        num_experts=num_experts,
                        num_shared_experts=num_shared_experts,
                        top_k=top_k,
                        hidden_dim=moe_intermediate_size,
                        shared_hidden_dim=moe_shared_expert_intermediate_size,
                        activation_fn=ReLUSquared(),
                        bias=False,
                    )
                    mixer.routed.router = NemotronScaledRouter(
                        moe_latent_size, num_experts, top_k, routed_scaling_factor
                    )
                else:
                    mixer = NemotronMoEFFN(
                        embed_dim,
                        num_experts=num_experts,
                        num_shared_experts=num_shared_experts,
                        top_k=top_k,
                        hidden_dim=moe_intermediate_size,
                        shared_hidden_dim=moe_shared_expert_intermediate_size,
                        routed_scaling_factor=routed_scaling_factor,
                        activation_fn=ReLUSquared(),
                        bias=False,
                    )
            else:
                raise ValueError(
                    f"Unknown hybrid_override_pattern layer type {layer_type!r}"
                )

            layer_blocks.append(
                Residual(Block([RMSNorm(embed_dim, eps=rms_norm_eps), mixer]))
            )

        layers = Block(layer_blocks)
        super().__init__(
            [
                embedding,
                layers,
                RMSNorm(embed_dim, eps=rms_norm_eps),
                OutputHead(
                    embed_dim,
                    vocab_size,
                    tied_embedding=embedding,
                    tie_weights=tie_weights,
                    use_norm=False,
                ),
            ]
        )


class NemotronNano30BA3B(NemotronHModel):
    """Nemotron 3 Nano (30B total, 128-expert MoE, ~3B active per token)."""

    def __init__(self):
        super().__init__(
            vocab_size=131072,
            embed_dim=2688,
            hybrid_override_pattern=(
                "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
            ),
            num_heads=32,
            num_kv_heads=2,
            head_dim=128,
            max_seq_len=262144,
            mamba_num_heads=64,
            mamba_head_dim=64,
            ssm_state_size=128,
            n_groups=8,
            conv_kernel_size=4,
            num_experts=128,
            num_shared_experts=1,
            top_k=6,
            moe_intermediate_size=1856,
            moe_shared_expert_intermediate_size=3712,
            routed_scaling_factor=2.5,
            moe_latent_size=None,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
            tie_weights=False,
        )


class NemotronSuper120BA12B(NemotronHModel):
    """Nemotron 3 Super (120B total, 512-expert Latent MoE, ~12B active per token)."""

    def __init__(self):
        super().__init__(
            vocab_size=131072,
            embed_dim=4096,
            hybrid_override_pattern=(
                "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
                "EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
            ),
            num_heads=32,
            num_kv_heads=2,
            head_dim=128,
            max_seq_len=262144,
            mamba_num_heads=128,
            mamba_head_dim=64,
            ssm_state_size=128,
            n_groups=8,
            conv_kernel_size=4,
            num_experts=512,
            num_shared_experts=1,
            top_k=22,
            moe_intermediate_size=2688,
            moe_shared_expert_intermediate_size=5376,
            routed_scaling_factor=5.0,
            moe_latent_size=1024,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
            tie_weights=False,
        )
