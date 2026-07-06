import torch
import torch.nn as nn

from olm.nn.structure import Block
from olm.nn.attention import GroupedQueryAttention, MultiHeadLatentAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead
from olm.nn.moe import MoEFeedForward


# ---------------------------------------------------------------------------
# Sarvam 30B -- GQA + MoE
# ---------------------------------------------------------------------------


class Sarvam30B_DenseBlock(Block):
    """Dense transformer block for the first layer of Sarvam 30B."""

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
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
        self.ffn = SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.attn_norm(x), **{k: v for k, v in kwargs.items() if k == "mask"})
        x = residual + x

        residual = x
        x = self.ffn(self.ffn_norm(x))
        return residual + x


class Sarvam30B_MoEBlock(Block):
    """
    MoE transformer block for Sarvam 30B.

    GQA with QK-Norm and a 128-expert sigmoid-routed MoE with top-6
    selection, router bias, one shared expert, and 2.5x routed scaling.
    """

    def __init__(
        self,
        embed_dim: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        num_experts: int,
        num_experts_per_tok: int,
        rope_theta: float,
        rms_norm_eps: float,
        routed_scaling_factor: float,
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
            num_shared_experts=1,
            top_k=num_experts_per_tok,
            expert_kwargs={
                "hidden_dim": moe_intermediate_size, "bias": False,
            },
            scoring_func="sigmoid",
            use_router_bias=True,
            norm_weights=True,
            fp32_gate=True,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.last_router_logits = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.attn_norm(x), **{k: v for k, v in kwargs.items() if k == "mask"})
        x = residual + x

        residual = x
        x, router_logits = self.moe(self.ffn_norm(x))
        self.last_router_logits = router_logits
        return residual + x


class Sarvam30B_Model(Block):
    """
    Sarvam 30B -- reasoning-oriented sparse MoE for Indian languages.

    19 layers of GQA with QK-Norm.  First layer uses a dense FFN; the
    remaining 18 use 128-expert sigmoid-routed MoE with top-6, one
    shared expert, router bias, and 2.5x routed scaling.

    Structure:
        Embedding -> [DenseBlock, MoEBlock x 18] -> RMSNorm -> OutputHead

    Forward:
        Token IDs ``[batch, seq_len]`` -> logits ``[batch, seq_len, vocab_size]``.
    """

    def __init__(
        self,
        vocab_size: int = 262144,
        embed_dim: int = 4096,
        intermediate_size: int = 8192,
        moe_intermediate_size: int = 1024,
        shared_expert_intermediate_size: int = 1024,
        num_layers: int = 19,
        num_heads: int = 64,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 131072,
        num_experts: int = 128,
        num_experts_per_tok: int = 6,
        first_k_dense: int = 1,
        rope_theta: float = 8000000.0,
        rms_norm_eps: float = 1e-6,
        routed_scaling_factor: float = 2.5,
        tie_weights: bool = False,
    ):
        embedding = Embedding(vocab_size, embed_dim)

        layers = nn.ModuleList()
        for i in range(num_layers):
            if i < first_k_dense:
                layers.append(Sarvam30B_DenseBlock(
                    embed_dim, intermediate_size, num_heads, num_kv_heads,
                    head_dim, max_seq_len, rope_theta, rms_norm_eps,
                ))
            else:
                layers.append(Sarvam30B_MoEBlock(
                    embed_dim, moe_intermediate_size,
                    shared_expert_intermediate_size,
                    num_heads, num_kv_heads, head_dim, max_seq_len,
                    num_experts, num_experts_per_tok,
                    rope_theta, rms_norm_eps, routed_scaling_factor,
                ))

        final_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        output_head = OutputHead(
            embed_dim, vocab_size,
            tied_embedding=embedding if tie_weights else None,
            tie_weights=tie_weights, use_norm=False,
        )

        super().__init__([embedding, final_norm, output_head])
        self.transformer_blocks = layers

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.blocks[0](x)
        for block in self.transformer_blocks:
            x = block(x, **kwargs)
        x = self.blocks[1](x)
        return self.blocks[2](x)


class Sarvam30B(Sarvam30B_Model):
    """Sarvam 30B with default config."""

    def __init__(self):
        super().__init__()


# ---------------------------------------------------------------------------
# Sarvam 105B -- MLA + MoE
# ---------------------------------------------------------------------------


class Sarvam105B_DenseBlock(Block):
    """Dense MLA block for the first layer of Sarvam 105B."""

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__([])
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = MultiHeadLatentAttention(
            embed_dim, num_heads, max_seq_len,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            rope_theta=rope_theta, rms_norm_eps=rms_norm_eps,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.ffn = SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.attn_norm(x), **{k: v for k, v in kwargs.items() if k == "mask"})
        x = residual + x

        residual = x
        x = self.ffn(self.ffn_norm(x))
        return residual + x


class Sarvam105B_MoEBlock(Block):
    """
    MoE transformer block for Sarvam 105B.

    MLA with KV compression (kv_lora_rank=512) for minimal KV cache,
    paired with 128-expert top-8 sigmoid-routed MoE.
    """

    def __init__(
        self,
        embed_dim: int,
        moe_intermediate_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_seq_len: int,
        num_experts: int,
        num_experts_per_tok: int,
        rope_theta: float,
        rms_norm_eps: float,
        routed_scaling_factor: float,
    ):
        super().__init__([])
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = MultiHeadLatentAttention(
            embed_dim, num_heads, max_seq_len,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            rope_theta=rope_theta, rms_norm_eps=rms_norm_eps,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.moe = MoEFeedForward(
            embed_dim=embed_dim,
            expert_cls=SwiGLUFFN,
            num_experts=num_experts,
            num_shared_experts=1,
            top_k=num_experts_per_tok,
            expert_kwargs={"hidden_dim": moe_intermediate_size, "bias": False},
            scoring_func="sigmoid",
            use_router_bias=True,
            norm_weights=True,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.last_router_logits = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.attn_norm(x), **{k: v for k, v in kwargs.items() if k == "mask"})
        x = residual + x

        residual = x
        x, router_logits = self.moe(self.ffn_norm(x))
        self.last_router_logits = router_logits
        return residual + x


class Sarvam105B_Model(Block):
    """
    Sarvam 105B -- MLA-based sparse MoE for Indian languages.

    32 layers of Multi-Head Latent Attention with KV compression
    (kv_lora_rank=512, head_dim split into 128 nope + 64 rope).
    First layer is dense FFN; remaining 31 use 128-expert top-8 MoE
    with DeepSeek-YaRN RoPE scaling for 131K context.

    Structure:
        Embedding -> [DenseBlock, MoEBlock x 31] -> RMSNorm -> OutputHead

    Forward:
        Token IDs ``[batch, seq_len]`` -> logits ``[batch, seq_len, vocab_size]``.
    """

    def __init__(
        self,
        vocab_size: int = 262144,
        embed_dim: int = 4096,
        intermediate_size: int = 16384,
        moe_intermediate_size: int = 2048,
        num_layers: int = 32,
        num_heads: int = 64,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        max_seq_len: int = 131072,
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
        first_k_dense: int = 1,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        routed_scaling_factor: float = 2.5,
        tie_weights: bool = False,
    ):
        embedding = Embedding(vocab_size, embed_dim)

        layers = nn.ModuleList()
        for i in range(num_layers):
            if i < first_k_dense:
                layers.append(Sarvam105B_DenseBlock(
                    embed_dim, intermediate_size, num_heads, kv_lora_rank,
                    qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                    max_seq_len, rope_theta, rms_norm_eps,
                ))
            else:
                layers.append(Sarvam105B_MoEBlock(
                    embed_dim, moe_intermediate_size, num_heads,
                    kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
                    v_head_dim, max_seq_len, num_experts, num_experts_per_tok,
                    rope_theta, rms_norm_eps, routed_scaling_factor,
                ))

        final_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        output_head = OutputHead(
            embed_dim, vocab_size,
            tied_embedding=embedding if tie_weights else None,
            tie_weights=tie_weights, use_norm=False,
        )

        super().__init__([embedding, final_norm, output_head])
        self.transformer_blocks = layers

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.blocks[0](x)
        for block in self.transformer_blocks:
            x = block(x, **kwargs)
        x = self.blocks[1](x)
        return self.blocks[2](x)


class Sarvam105B(Sarvam105B_Model):
    """Sarvam 105B with default config."""

    def __init__(self):
        super().__init__()
