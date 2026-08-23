from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual
from olm.nn.attention import MultiHeadLatentAttention, KimiDeltaAttention
from olm.nn.feedforward import SwiGLUFFN, SwiGLUMoEFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


class KimiLinearBlock(Block):
    """
    A single decoder block for Kimi Linear.

    Attention is Kimi Delta Attention (KDA) for most layers and Multi-head
    Latent Attention (MLA) for every ``full_attention_interval``-th layer
    (plus the final layer). The feed-forward is a dense SwiGLU for the first
    ``first_k_dense_replace`` layers and a sigmoid-routed MoE with a shared
    expert afterward.

        x = x + Attn(RMSNorm(x))   # KDA or MLA
        x = x + FFN(RMSNorm(x))    # dense SwiGLU or MoE

    Args:
        embed_dim (int): Model dimension.
        is_full_attention (bool): Whether this layer uses MLA (else KDA).
        use_moe (bool): Whether this layer uses MoE (else dense SwiGLU).
        kda_num_heads (int): Number of KDA heads.
        kda_head_dim (int): Per-head dimension for KDA.
        kda_conv_kernel_size (int): Causal conv kernel size for KDA.
        mla_num_heads (int): Number of MLA heads.
        max_seq_len (int): Maximum context length (for the RoPE cache).
        kv_lora_rank (int): Rank of the MLA compressed key/value latent.
        qk_nope_head_dim (int): Per-head non-positional query/key dimension (MLA).
        qk_rope_head_dim (int): Per-head rotary query/key dimension (MLA).
        v_head_dim (int): Per-head value dimension (MLA).
        q_lora_rank (int | None): Rank of the MLA compressed query latent.
        rope_theta (float): RoPE base frequency (MLA).
        intermediate_size (int): FFN hidden dim for dense layers.
        moe_intermediate_size (int): FFN hidden dim of each MoE expert.
        num_experts (int): Total number of routable experts.
        num_shared_experts (int): Number of always-active shared experts.
        top_k (int): Number of experts routed to per token.
        routed_scaling_factor (float): Multiplier on MoE routing weights.
        dropout (float): Dropout probability.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
    """

    def __init__(
        self,
        embed_dim: int,
        is_full_attention: bool,
        use_moe: bool,
        kda_num_heads: int,
        kda_head_dim: int,
        kda_conv_kernel_size: int,
        mla_num_heads: int,
        max_seq_len: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        rope_theta: float,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        dropout: float,
        rms_norm_eps: float,
    ):
        if is_full_attention:
            attn = MultiHeadLatentAttention(
                embed_dim,
                mla_num_heads,
                max_seq_len,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                q_lora_rank=q_lora_rank,
                dropout=dropout,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
            )
        else:
            attn = KimiDeltaAttention(
                embed_dim,
                num_heads=kda_num_heads,
                head_dim=kda_head_dim,
                conv_kernel_size=kda_conv_kernel_size,
                dropout=dropout,
            )

        if use_moe:
            # Kimi Linear scores experts with sigmoid
            # (``moe_router_activation_func='sigmoid'``) and scales the routed
            # output, rather than using the plain softmax default.
            ffn = SwiGLUMoEFFN(
                embed_dim,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                hidden_dim=moe_intermediate_size,
                bias=False,
                routed_scaling_factor=routed_scaling_factor,
                router_kwargs={"scoring_func": "sigmoid"},
            )
        else:
            ffn = SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, bias=False)

        super().__init__(
            [
                Residual(Block([RMSNorm(embed_dim, eps=rms_norm_eps), attn])),
                Residual(Block([RMSNorm(embed_dim, eps=rms_norm_eps), ffn])),
            ]
        )


class KimiLinearModel(Block):
    """
    Base class for Kimi Linear models.

    Structure:
        Embedding -> [KimiLinearBlock] x N -> RMSNorm -> OutputHead.

    Attention interleaves Kimi Delta Attention (linear) with Multi-head
    Latent Attention (full) in roughly a 3:1 ratio: every
    ``full_attention_interval``-th layer uses MLA, and the final layer
    always uses MLA regardless of the modulo (matching the reference
    checkpoint's explicit layer list). The first ``first_k_dense_replace``
    layers use a dense SwiGLU feed-forward; the rest use a sigmoid-routed
    MoE with a shared expert. Kimi Linear does not tie input/output
    embeddings; the named preset passes ``tie_weights=False``.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
        ``[batch, seq_len, vocab_size]``.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Model dimension.
        num_layers (int): Number of transformer blocks.
        kda_num_heads (int): Number of KDA heads.
        kda_head_dim (int): Per-head dimension for KDA.
        kda_conv_kernel_size (int): Causal conv kernel size for KDA.
        mla_num_heads (int): Number of MLA heads.
        max_seq_len (int): Maximum context length.
        kv_lora_rank (int): Rank of the MLA compressed key/value latent.
        qk_nope_head_dim (int): Per-head non-positional query/key dimension (MLA).
        qk_rope_head_dim (int): Per-head rotary query/key dimension (MLA).
        v_head_dim (int): Per-head value dimension (MLA).
        q_lora_rank (int | None): Rank of the MLA compressed query latent.
        intermediate_size (int): FFN hidden dim for dense layers.
        moe_intermediate_size (int): FFN hidden dim of each MoE expert.
        num_experts (int): Total number of routable experts.
        num_shared_experts (int): Number of always-active shared experts.
        top_k (int): Number of experts routed to per token.
        full_attention_interval (int): Period of the KDA:MLA pattern; every
            ``full_attention_interval``-th layer (and the final layer) uses MLA.
        first_k_dense_replace (int): Number of leading dense (non-MoE) layers.
        rope_theta (float): RoPE base frequency (MLA).
        routed_scaling_factor (float): Multiplier on MoE routing weights.
        dropout (float): Dropout probability.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
        tie_weights (bool): Whether to tie the output head to the embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        kda_num_heads: int,
        kda_head_dim: int,
        kda_conv_kernel_size: int,
        mla_num_heads: int,
        max_seq_len: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        full_attention_interval: int = 4,
        first_k_dense_replace: int = 1,
        rope_theta: float = 10000.0,
        routed_scaling_factor: float = 2.446,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-5,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        layers = Block(
            [
                KimiLinearBlock(
                    embed_dim,
                    is_full_attention=(
                        (layer_idx + 1) % full_attention_interval == 0
                        or layer_idx == num_layers - 1
                    ),
                    use_moe=layer_idx >= first_k_dense_replace,
                    kda_num_heads=kda_num_heads,
                    kda_head_dim=kda_head_dim,
                    kda_conv_kernel_size=kda_conv_kernel_size,
                    mla_num_heads=mla_num_heads,
                    max_seq_len=max_seq_len,
                    kv_lora_rank=kv_lora_rank,
                    qk_nope_head_dim=qk_nope_head_dim,
                    qk_rope_head_dim=qk_rope_head_dim,
                    v_head_dim=v_head_dim,
                    q_lora_rank=q_lora_rank,
                    rope_theta=rope_theta,
                    intermediate_size=intermediate_size,
                    moe_intermediate_size=moe_intermediate_size,
                    num_experts=num_experts,
                    num_shared_experts=num_shared_experts,
                    top_k=top_k,
                    routed_scaling_factor=routed_scaling_factor,
                    dropout=dropout,
                    rms_norm_eps=rms_norm_eps,
                )
                for layer_idx in range(num_layers)
            ]
        )
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


class KimiLinear48BA3B(KimiLinearModel):
    """Kimi Linear 48B-A3B (48B total, 256-expert MoE, ~3B active per token)."""

    def __init__(self):
        super().__init__(
            vocab_size=163840,
            embed_dim=2304,
            num_layers=27,
            kda_num_heads=32,
            kda_head_dim=128,
            kda_conv_kernel_size=4,
            mla_num_heads=32,
            max_seq_len=1048576,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=None,
            intermediate_size=9216,
            moe_intermediate_size=1024,
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            full_attention_interval=4,
            first_k_dense_replace=1,
            rope_theta=10000.0,
            routed_scaling_factor=2.446,
            rms_norm_eps=1e-5,
            tie_weights=False,
        )
