import torch
import torch.nn as nn

from olm.nn.structure import Block
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead, MultiTokenPredictionHead
from olm.nn.moe import MoEFeedForward


class MiniMaxM2_5_Block(Block):
    """
    Transformer block for MiniMax M2.5.

    GQA with per-layer QK-Norm, partial RoPE (rotary_dim=64 of 128),
    and a 256-expert MoE with sigmoid top-8 routing plus router bias.

    Structure:
        x = x + GQA(RMSNorm(x))
        x = x + MoE(RMSNorm(x))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__([])

        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads, max_seq_len,
            head_dim=head_dim,
            rope_theta=rope_theta,
            use_bias=False,
            use_qk_norm=True,
            rms_norm_eps=rms_norm_eps,
        )

        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.moe = MoEFeedForward(
            embed_dim=embed_dim,
            expert_cls=SwiGLUFFN,
            num_experts=num_experts,
            num_shared_experts=0,
            top_k=num_experts_per_tok,
            expert_kwargs={"hidden_dim": moe_intermediate_size, "bias": False},
            scoring_func="sigmoid",
            use_router_bias=True,
            norm_weights=True,
        )
        self.last_router_logits = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.attn_norm(x)
        x = self.self_attn(x, **{k: v for k, v in kwargs.items() if k == "mask"})
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x, router_logits = self.moe(x)
        self.last_router_logits = router_logits
        return residual + x


class MiniMaxM2_5_Model(Block):
    """
    MiniMax M2.5 -- 230B sparse MoE with 10B active parameters.

    All 62 layers use full GQA with per-layer QK-Norm.  Partial RoPE
    (rotary_dim=64 of head_dim=128) with theta=5M.  Three MTP heads
    for speculative decoding support.

    Structure:
        Embedding -> [MiniMaxM2_5_Block] x 62 -> RMSNorm -> OutputHead
        + MultiTokenPredictionHead (3 heads)

    Forward:
        Token IDs ``[batch, seq_len]`` -> logits ``[batch, seq_len, vocab_size]``.
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
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        num_mtp_modules: int = 3,
        rope_theta: float = 5000000.0,
        rms_norm_eps: float = 1e-6,
        tie_weights: bool = False,
    ):
        embedding = Embedding(vocab_size, embed_dim)

        transformer_blocks = nn.ModuleList([
            MiniMaxM2_5_Block(
                embed_dim, num_heads, num_kv_heads, head_dim, max_seq_len,
                moe_intermediate_size, num_experts, num_experts_per_tok,
                rope_theta, rms_norm_eps,
            )
            for _ in range(num_layers)
        ])

        final_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        output_head = OutputHead(
            embed_dim, vocab_size,
            tied_embedding=embedding if tie_weights else None,
            tie_weights=tie_weights,
            use_norm=False,
        )

        super().__init__([embedding, final_norm, output_head])

        self.transformer_blocks = transformer_blocks
        self.mtp = MultiTokenPredictionHead(
            embed_dim, vocab_size, num_heads=num_mtp_modules,
            rms_norm_eps=rms_norm_eps,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        token_embeddings = self.blocks[0](x)

        hidden = token_embeddings
        for block in self.transformer_blocks:
            hidden = block(hidden, **kwargs)

        hidden = self.blocks[1](hidden)
        logits = self.blocks[2](hidden)
        return logits

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


class MiniMaxM2_5_230B(MiniMaxM2_5_Model):
    """MiniMax M2.5 230B -- 256 experts, top-8, 10B active."""

    def __init__(self):
        super().__init__(
            vocab_size=200064,
            embed_dim=3072,
            num_layers=62,
            num_heads=48,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=196608,
            moe_intermediate_size=1536,
            num_experts=256,
            num_experts_per_tok=8,
            num_mtp_modules=3,
            rope_theta=5000000.0,
            rms_norm_eps=1e-6,
            tie_weights=False,
        )
