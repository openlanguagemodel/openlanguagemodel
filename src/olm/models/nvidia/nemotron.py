from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual
from olm.nn.attention import GroupedQueryAttention, Mamba2Mixer
from olm.nn.feedforward import ClassicMoEFFN, LatentMoEFFN
from olm.nn.activations import ReLUSquared
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


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

    MoE layers route with sigmoid scoring and an auxiliary-loss-free
    correction bias (``noaux_tc``), optionally restricted to the best
    ``topk_group`` of ``n_group`` expert groups, and scale the routed branch by
    ``routed_scaling_factor`` to keep parity with the always-active shared
    expert.

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
            of a full-width ``ClassicMoEFFN`` (Nemotron Nano).
        n_group (int, optional): Number of expert groups for group-limited
            routing. ``None`` scores every expert.
        topk_group (int, optional): Number of groups a token may draw experts
            from; required when ``n_group`` is set.
        time_step_min (float): Lower bound of the sampled initial timestep
            (Mamba-2 layers).
        time_step_max (float): Upper bound of the sampled initial timestep
            (Mamba-2 layers).
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
        n_group: int = None,
        topk_group: int = None,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-5,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)

        # Sigmoid scoring with an auxiliary-loss-free correction bias, as in
        # the reference implementation; the softmax default would change both
        # which experts a token picks and how their outputs are weighted.
        router_kwargs = {
            "scoring_func": "sigmoid",
            "routing_method": "noaux_tc",
            "norm_weights": True,
            "fp32_gate": True,
            "n_group": n_group,
            "topk_group": topk_group,
        }

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
                    time_step_min=time_step_min,
                    time_step_max=time_step_max,
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
                        routed_scaling_factor=routed_scaling_factor,
                        router_kwargs=router_kwargs,
                    )
                else:
                    mixer = ClassicMoEFFN(
                        embed_dim,
                        num_experts=num_experts,
                        num_shared_experts=num_shared_experts,
                        top_k=top_k,
                        hidden_dim=moe_intermediate_size,
                        shared_hidden_dim=moe_shared_expert_intermediate_size,
                        activation_fn=ReLUSquared(),
                        bias=False,
                        routed_scaling_factor=routed_scaling_factor,
                        router_kwargs=router_kwargs,
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
