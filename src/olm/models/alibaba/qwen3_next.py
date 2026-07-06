from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual
from olm.nn.attention import GatedAttention, GatedDeltaNet
from olm.nn.feedforward import SwiGLUMoEFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


class Qwen3NextBlock(Block):
    """
    A single decoder block for Qwen3-Next.

    Hybrid attention: most layers use Gated DeltaNet (linear attention with a
    causal conv + delta-rule state); every ``full_attention_interval``-th
    layer uses full Gated Attention (softmax attention with QK-norm, partial
    RoPE, and an output gate) instead. Every layer's feed-forward is a
    sigmoid-free, softmax-routed MoE with a single shared expert.

        x = x + Attn(RMSNorm(x))   # GatedDeltaNet or GatedAttention
        x = x + MoE(RMSNorm(x))

    Args:
        embed_dim (int): Model dimension.
        num_heads (int): Number of query heads (full-attention layers).
        num_kv_heads (int): Number of key/value heads (full-attention layers, GQA).
        head_dim (int): Per-head dimension (full-attention layers).
        max_seq_len (int): Maximum context length (for the RoPE cache).
        linear_num_key_heads (int): Number of key heads (linear-attention layers).
        linear_num_value_heads (int): Number of value heads (linear-attention layers).
        linear_key_head_dim (int): Per-head key dimension (linear-attention layers).
        linear_value_head_dim (int): Per-head value dimension (linear-attention layers).
        linear_conv_kernel_size (int): Causal conv kernel size (linear-attention layers).
        moe_intermediate_size (int): FFN hidden dim of each MoE expert.
        num_experts (int): Total number of routable experts.
        num_shared_experts (int): Number of always-active shared experts.
        top_k (int): Number of experts routed to per token.
        is_full_attention (bool): Whether this layer uses Gated Attention
            (else Gated DeltaNet).
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base frequency (full-attention layers).
        partial_rotary_factor (float): Fraction of ``head_dim`` rotated by RoPE
            (full-attention layers).
        rms_norm_eps (float): Epsilon for RMSNorm layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        linear_num_key_heads: int,
        linear_num_value_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        is_full_attention: bool,
        dropout: float,
        rope_theta: float,
        partial_rotary_factor: float,
        rms_norm_eps: float,
    ):
        if is_full_attention:
            attn = GatedAttention(
                embed_dim,
                num_heads,
                num_kv_heads,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                partial_rotary_factor=partial_rotary_factor,
                use_qk_norm=True,
                rms_norm_eps=rms_norm_eps,
                dropout=dropout,
            )
        else:
            attn = GatedDeltaNet(
                embed_dim,
                num_key_heads=linear_num_key_heads,
                num_value_heads=linear_num_value_heads,
                key_head_dim=linear_key_head_dim,
                value_head_dim=linear_value_head_dim,
                conv_kernel_size=linear_conv_kernel_size,
                dropout=dropout,
            )

        moe = SwiGLUMoEFFN(
            embed_dim,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            hidden_dim=moe_intermediate_size,
            bias=False,
        )

        super().__init__(
            [
                Residual(Block([RMSNorm(embed_dim, eps=rms_norm_eps), attn])),
                Residual(Block([RMSNorm(embed_dim, eps=rms_norm_eps), moe])),
            ]
        )


class Qwen3NextModel(Block):
    """
    Base class for Qwen3-Next models.

    Structure:
        Embedding -> [Qwen3NextBlock] x N -> RMSNorm -> OutputHead.

    Attention alternates ``full_attention_interval - 1`` Gated DeltaNet
    (linear attention) layers followed by one Gated Attention (full
    attention) layer. Every layer's feed-forward is MoE (no dense layers).
    Qwen3-Next does not tie input/output embeddings; the named preset passes
    ``tie_weights=False``.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
        ``[batch, seq_len, vocab_size]``.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Model dimension.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of query heads (full-attention layers).
        num_kv_heads (int): Number of key/value heads (full-attention layers, GQA).
        head_dim (int): Per-head dimension (full-attention layers).
        max_seq_len (int): Maximum context length.
        linear_num_key_heads (int): Number of key heads (linear-attention layers).
        linear_num_value_heads (int): Number of value heads (linear-attention layers).
        linear_key_head_dim (int): Per-head key dimension (linear-attention layers).
        linear_value_head_dim (int): Per-head value dimension (linear-attention layers).
        linear_conv_kernel_size (int): Causal conv kernel size (linear-attention layers).
        moe_intermediate_size (int): FFN hidden dim of each MoE expert.
        num_experts (int): Total number of routable experts.
        num_shared_experts (int): Number of always-active shared experts.
        top_k (int): Number of experts routed to per token.
        full_attention_interval (int): Period of the linear:full pattern; every
            ``full_attention_interval``-th layer is a full-attention layer.
        rope_theta (float): RoPE base frequency.
        partial_rotary_factor (float): Fraction of ``head_dim`` rotated by RoPE.
        dropout (float): Dropout probability.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
        tie_weights (bool): Whether to tie the output head to the embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        linear_num_key_heads: int,
        linear_num_value_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        full_attention_interval: int = 4,
        rope_theta: float = 10000000.0,
        partial_rotary_factor: float = 0.25,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        layers = Block(
            [
                Qwen3NextBlock(
                    embed_dim,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    max_seq_len,
                    linear_num_key_heads,
                    linear_num_value_heads,
                    linear_key_head_dim,
                    linear_value_head_dim,
                    linear_conv_kernel_size,
                    moe_intermediate_size,
                    num_experts,
                    num_shared_experts,
                    top_k,
                    is_full_attention=(layer_idx + 1) % full_attention_interval == 0,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    partial_rotary_factor=partial_rotary_factor,
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


class Qwen3Next80BA3B(Qwen3NextModel):
    """Qwen3-Next-80B-A3B (80B total, 512-expert MoE, ~3B active per token)."""

    def __init__(self):
        super().__init__(
            vocab_size=151936,
            embed_dim=2048,
            num_layers=48,
            num_heads=16,
            num_kv_heads=2,
            head_dim=256,
            max_seq_len=262144,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_size=4,
            moe_intermediate_size=512,
            num_experts=512,
            num_shared_experts=1,
            top_k=10,
            full_attention_interval=4,
            rope_theta=10000000.0,
            partial_rotary_factor=0.25,
            rms_norm_eps=1e-6,
            tie_weights=False,
        )
