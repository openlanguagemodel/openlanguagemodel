import torch
import torch.nn as nn
from typing import List

from olm.nn.structure import Block
from olm.nn.attention import GroupedQueryAttention, SlidingWindowAttention
from olm.nn.attention.head_gated import HeadGate
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead, MultiTokenPredictionHead
from olm.nn.moe import MoEFeedForward


class Step3_5_FullAttnBlock(Block):
    """
    Full-attention block for Step 3.5 Flash.

    GQA with 64 heads, 8 KV groups, per-head sigmoid gating, QK-Norm,
    and partial RoPE (rotary_factor=0.5) at theta=5M.

    The head gate reads the normalized block input and scales each head's
    attention output *before* the output projection, as in the released
    ``Step3p5Attention``. The attention module is therefore built without
    its output projection, which the block owns as ``o_proj``.

    Structure:
        h = RMSNorm(x)
        x = x + o_proj(sigmoid(gate(h)) * GQA(h))
        x = x + FFN(RMSNorm(x))      # dense SwiGLU or MoE

    Args:
        embed_dim (int): Model dimension.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key/value heads (GQA).
        head_dim (int): Dimension per attention head.
        max_seq_len (int): Maximum context length (for the RoPE cache).
        rope_theta (float): RoPE base frequency.
        partial_rotary_factor (float): Fraction of ``head_dim`` rotated by
            RoPE.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
        ffn_module (nn.Module): Feed-forward sublayer (``SwiGLUFFN`` or
            ``MoEFeedForward``).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        partial_rotary_factor: float,
        rms_norm_eps: float,
        ffn_module: nn.Module,
    ):
        super().__init__([])
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = GroupedQueryAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            max_seq_len,
            head_dim=head_dim,
            rope_theta=rope_theta,
            use_bias=False,
            use_qk_norm=True,
            rms_norm_eps=rms_norm_eps,
            partial_rotary_factor=partial_rotary_factor,
        )
        # The head gate sits between attention and its output projection.
        self.o_proj = self.self_attn.out_proj
        self.self_attn.out_proj = nn.Identity()
        self.head_gate = HeadGate(embed_dim, num_heads)
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.ffn = ffn_module
        self.last_router_logits = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return _step3_5_block_forward(self, x, **kwargs)


class Step3_5_SlidingAttnBlock(Block):
    """
    Sliding-window attention block for Step 3.5 Flash.

    GQA with 96 heads, 8 KV groups, window_size=512, per-head sigmoid
    gating, QK-Norm, and full RoPE (rotary_factor=1.0) at theta=10K.
    The head gate is applied before the output projection, as in
    ``Step3_5_FullAttnBlock``.

    Structure:
        h = RMSNorm(x)
        x = x + o_proj(sigmoid(gate(h)) * SWA(h))
        x = x + FFN(RMSNorm(x))      # dense SwiGLU or MoE

    Args:
        embed_dim (int): Model dimension.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key/value heads (GQA).
        head_dim (int): Dimension per attention head.
        max_seq_len (int): Maximum context length (for the RoPE cache).
        window_size (int): Sliding attention window, in tokens.
        rope_theta (float): RoPE base frequency.
        partial_rotary_factor (float): Fraction of ``head_dim`` rotated by
            RoPE.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
        ffn_module (nn.Module): Feed-forward sublayer (``SwiGLUFFN`` or
            ``MoEFeedForward``).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        window_size: int,
        rope_theta: float,
        partial_rotary_factor: float,
        rms_norm_eps: float,
        ffn_module: nn.Module,
    ):
        super().__init__([])
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = SlidingWindowAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            max_seq_len,
            window_size=window_size,
            head_dim=head_dim,
            rope_theta=rope_theta,
            use_qk_norm=True,
            rms_norm_eps=rms_norm_eps,
            partial_rotary_factor=partial_rotary_factor,
        )
        # The head gate sits between attention and its output projection.
        self.o_proj = self.self_attn.out_proj
        self.self_attn.out_proj = nn.Identity()
        self.head_gate = HeadGate(embed_dim, num_heads)
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.ffn = ffn_module
        self.last_router_logits = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return _step3_5_block_forward(self, x, **kwargs)


def _step3_5_block_forward(block, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Shared forward of the two Step 3.5 block types."""
    h = block.attn_norm(x)
    attn_out = block.self_attn(h, **{k: v for k, v in kwargs.items() if k == "mask"})
    B, N, _ = attn_out.shape
    # HeadGate reads the head axis from dim 1 when it matches num_heads, so
    # use the [B, heads, N, head_dim] layout, which stays unambiguous when
    # seq_len == num_heads.
    attn_out = attn_out.view(B, N, block.num_heads, block.head_dim).transpose(1, 2)
    attn_out = block.head_gate(h, attn_out).transpose(1, 2).reshape(B, N, -1)
    x = x + block.o_proj(attn_out)

    h = block.ffn_norm(x)
    if isinstance(block.ffn, MoEFeedForward):
        h, router_logits = block.ffn(h)
        block.last_router_logits = router_logits
    else:
        h = block.ffn(h)
    return x + h


# Per-layer schedules from the released config.json. They have 48 entries:
# 45 decoder layers plus the 3 multi-token-prediction layers stored after them.
STEP3_5_LAYER_TYPES = [
    "full",
    "sliding",
    "sliding",
    "sliding",
] * 12


STEP3_5_MOE_LAYERS = set(range(3, 45))

STEP3_5_ROPE_THETA = [
    5000000.0,
    10000.0,
    10000.0,
    10000.0,
] * 12

STEP3_5_PARTIAL_ROTARY = [
    0.5,
    1.0,
    1.0,
    1.0,
] * 12


class Step3_5_Flash_Model(Block):
    """
    Step 3.5 Flash -- throughput-oriented 196B sparse MoE.

    45 layers in a repeating full / sliding / sliding / sliding pattern.
    Full attention layers use 64 heads with partial RoPE (0.5) at theta=5M;
    sliding-window layers use 96 heads with full RoPE at theta=10K.
    All layers have per-head sigmoid gating and QK-Norm.

    Layers 0--2 use a dense SwiGLU; the other 42 (indices 3--44) use a
    288-expert sigmoid-routed MoE with top-8 selection, one shared expert,
    router bias, FP32 gate, weight normalization, and 3x routed scaling.
    Three MTP heads for speculative decoding.

    Structure:
        Embedding -> [FullAttn | SlidingAttn] x 45 -> RMSNorm -> OutputHead
        + MultiTokenPredictionHead (3 heads)

    Implementation note:
        The released checkpoint applies Llama-3-style RoPE scaling (factor 2)
        on the full-attention layers only and clamps the SwiGLU activations of
        the last two MoE layers (``swiglu_limits``); neither is modeled.
        Multi-token prediction uses the lightweight
        ``MultiTokenPredictionHead``, whereas the released MTP modules
        (stored as ``model.layers.45``--``47``) are full decoder layers, so
        they are not part of the 45-layer stack.

    Forward:
        Token IDs ``[batch, seq_len]`` -> logits ``[batch, seq_len, vocab_size]``.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension of the dense layers.
        moe_intermediate_size (int): Hidden dimension of each routed expert
            and of the shared expert.
        num_layers (int): Number of decoder layers.
        full_attn_heads (int): Query heads in full-attention layers.
        sliding_attn_heads (int): Query heads in sliding-window layers.
        num_kv_heads (int): Number of key/value heads (GQA).
        head_dim (int): Dimension per attention head.
        max_seq_len (int): Maximum context length.
        window_size (int): Sliding attention window, in tokens.
        num_experts (int): Total number of routable experts.
        num_experts_per_tok (int): Number of experts routed to per token.
        num_mtp_heads (int): Number of multi-token-prediction heads.
        routed_scaling_factor (float): Multiplier on the routed-expert output.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
        tie_weights (bool): Whether to tie the output head to the embedding.
        layer_types (list[str] | None): ``"full"`` or ``"sliding"`` per layer.
            Defaults to ``STEP3_5_LAYER_TYPES``.
        moe_layers (set[int] | None): Indices of MoE layers. Defaults to
            ``STEP3_5_MOE_LAYERS``.
        rope_thetas (list[float] | None): RoPE base per layer. Defaults to
            ``STEP3_5_ROPE_THETA``.
        partial_rotary_factors (list[float] | None): Rotated fraction of each
            head, per layer. Defaults to ``STEP3_5_PARTIAL_ROTARY``.
    """

    def __init__(
        self,
        vocab_size: int = 128896,
        embed_dim: int = 4096,
        intermediate_size: int = 11264,
        moe_intermediate_size: int = 1280,
        num_layers: int = 45,
        full_attn_heads: int = 64,
        sliding_attn_heads: int = 96,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 262144,
        window_size: int = 512,
        num_experts: int = 288,
        num_experts_per_tok: int = 8,
        num_mtp_heads: int = 3,
        routed_scaling_factor: float = 3.0,
        rms_norm_eps: float = 1e-5,
        tie_weights: bool = False,
        layer_types: List[str] = None,
        moe_layers: set = None,
        rope_thetas: List[float] = None,
        partial_rotary_factors: List[float] = None,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        layer_types = layer_types or STEP3_5_LAYER_TYPES[:num_layers]
        moe_layers = moe_layers if moe_layers is not None else STEP3_5_MOE_LAYERS
        rope_thetas = rope_thetas or STEP3_5_ROPE_THETA[:num_layers]
        partial_rotary_factors = (
            partial_rotary_factors or STEP3_5_PARTIAL_ROTARY[:num_layers]
        )

        layers = nn.ModuleList()
        for i in range(num_layers):
            if i in moe_layers:
                ffn = MoEFeedForward(
                    embed_dim=embed_dim,
                    expert_cls=SwiGLUFFN,
                    num_experts=num_experts,
                    num_shared_experts=1,
                    top_k=num_experts_per_tok,
                    expert_kwargs={"hidden_dim": moe_intermediate_size, "bias": False},
                    scoring_func="sigmoid",
                    use_router_bias=True,
                    norm_weights=True,
                    fp32_gate=True,
                    routed_scaling_factor=routed_scaling_factor,
                )
            else:
                ffn = SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, bias=False)

            if layer_types[i] == "full":
                layers.append(
                    Step3_5_FullAttnBlock(
                        embed_dim,
                        full_attn_heads,
                        num_kv_heads,
                        head_dim,
                        max_seq_len,
                        rope_thetas[i],
                        partial_rotary_factors[i],
                        rms_norm_eps,
                        ffn,
                    )
                )
            else:
                layers.append(
                    Step3_5_SlidingAttnBlock(
                        embed_dim,
                        sliding_attn_heads,
                        num_kv_heads,
                        head_dim,
                        max_seq_len,
                        window_size,
                        rope_thetas[i],
                        partial_rotary_factors[i],
                        rms_norm_eps,
                        ffn,
                    )
                )

        final_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        output_head = OutputHead(
            embed_dim,
            vocab_size,
            tied_embedding=embedding if tie_weights else None,
            tie_weights=tie_weights,
            use_norm=False,
        )

        super().__init__([embedding, final_norm, output_head])
        self.transformer_blocks = layers
        self.mtp = MultiTokenPredictionHead(
            embed_dim,
            vocab_size,
            num_heads=num_mtp_heads,
            rms_norm_eps=rms_norm_eps,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        token_embeddings = self.blocks[0](x)

        hidden = token_embeddings
        for block in self.transformer_blocks:
            hidden = block(hidden, **kwargs)

        hidden = self.blocks[1](hidden)
        return self.blocks[2](hidden)

    def forward_with_mtp(self, x: torch.Tensor, **kwargs):
        """Return main logits and MTP auxiliary logits for training."""
        token_embeddings = self.blocks[0](x)

        hidden = token_embeddings
        for block in self.transformer_blocks:
            hidden = block(hidden, **kwargs)

        hidden = self.blocks[1](hidden)
        logits = self.blocks[2](hidden)
        mtp_logits = self.mtp(hidden, token_embeddings)
        return logits, mtp_logits


class Step3_5_Flash_196B(Step3_5_Flash_Model):
    """Step 3.5 Flash (196B total, ~11B active)."""

    def __init__(self):
        super().__init__(
            vocab_size=128896,
            embed_dim=4096,
            intermediate_size=11264,
            moe_intermediate_size=1280,
            num_layers=45,
            full_attn_heads=64,
            sliding_attn_heads=96,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=262144,
            window_size=512,
            num_experts=288,
            num_experts_per_tok=8,
            num_mtp_heads=3,
            routed_scaling_factor=3.0,
            rms_norm_eps=1e-5,
            tie_weights=False,
            layer_types=STEP3_5_LAYER_TYPES[:45],
            moe_layers=STEP3_5_MOE_LAYERS,
            rope_thetas=STEP3_5_ROPE_THETA[:45],
            partial_rotary_factors=STEP3_5_PARTIAL_ROTARY[:45],
        )
