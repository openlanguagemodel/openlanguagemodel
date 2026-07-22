import torch
import torch.nn as nn

from olm.nn.structure import Block
from olm.nn.attention import MultiHeadLatentAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead
from olm.nn.moe import MoEFeedForward


class KimiK2Block(Block):
    """
    A single decoder block for Kimi K2.

    Multi-head Latent Attention throughout, paired with either a dense
    SwiGLU feed-forward (the first layer) or a fine-grained MoE block with a
    shared expert and auxiliary-loss-free (sigmoid, noaux_tc) routing --
    the same DeepSeek-V3-derived recipe Kimi K2 is built on, at a wider
    expert count:

        x = x + MLA(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Args:
        embed_dim (int): Model dimension.
        num_heads (int): Number of attention heads.
        max_seq_len (int): Maximum context length (for the RoPE cache).
        kv_lora_rank (int): Rank of the compressed key/value latent.
        qk_nope_head_dim (int): Per-head dimension of the non-positional query/key part.
        qk_rope_head_dim (int): Per-head dimension of the rotary query/key part.
        v_head_dim (int): Per-head dimension of the value.
        q_lora_rank (int): Rank of the compressed query latent.
        dense_intermediate_size (int): FFN hidden dim for dense layers.
        moe_intermediate_size (int): FFN hidden dim of each MoE expert.
        num_experts (int): Total number of routable experts.
        num_shared_experts (int): Number of always-active shared experts.
        top_k (int): Number of experts routed to per token.
        use_moe (bool): Whether this layer is an MoE layer (else dense SwiGLU).
        routed_scaling_factor (float): Multiplicative factor applied to the
            combined routed-expert output.
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base frequency.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        dense_intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        use_moe: bool,
        routed_scaling_factor: float,
        dropout: float,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__([])
        self.use_moe = use_moe
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = MultiHeadLatentAttention(
            embed_dim,
            num_heads,
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
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.last_router_logits = None

        if use_moe:
            self.ffn = MoEFeedForward(
                embed_dim=embed_dim,
                expert_cls=SwiGLUFFN,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                expert_kwargs={"hidden_dim": moe_intermediate_size, "bias": False},
                scoring_func="sigmoid",
                routing_method="noaux_tc",
                use_router_bias=True,
                norm_weights=True,
                fp32_gate=True,
                routed_scaling_factor=routed_scaling_factor,
            )
        else:
            self.ffn = SwiGLUFFN(
                embed_dim, hidden_dim=dense_intermediate_size, bias=False
            )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.self_attn(
            self.attn_norm(x), **{k: v for k, v in kwargs.items() if k == "mask"}
        )
        x = residual + x

        residual = x
        if self.use_moe:
            x, router_logits = self.ffn(self.ffn_norm(x))
            self.last_router_logits = router_logits
        else:
            x = self.ffn(self.ffn_norm(x))
        return residual + x


class KimiK2Model(Block):
    """
    Base class for the Kimi K2 language model.

    Structure:
        Embedding -> [KimiK2Block] x N -> RMSNorm -> OutputHead.

    The first ``first_k_dense_replace`` layer(s) use a dense SwiGLU
    feed-forward; the remaining layers use a fine-grained MoE with a shared
    expert and sigmoid, auxiliary-loss-free (``noaux_tc``) routing. Attention
    is Multi-head Latent Attention throughout. Kimi K2 does not tie
    input/output embeddings (the named ``KimiK2_1T`` preset passes
    ``tie_weights=False``) and does not use Multi-Token Prediction (unlike
    its DeepSeek-V3 architectural relative).

    Notes:
        - This models the language-model component only.
        - Long-context RoPE scaling is approximated with plain RoPE, the
          same simplification used for the other MLA models in this repo.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits
        shaped ``[batch, seq_len, vocab_size]``.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Model dimension.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        max_seq_len (int): Maximum context length.
        kv_lora_rank (int): Rank of the compressed key/value latent.
        qk_nope_head_dim (int): Per-head dimension of the non-positional query/key part.
        qk_rope_head_dim (int): Per-head dimension of the rotary query/key part.
        v_head_dim (int): Per-head dimension of the value.
        q_lora_rank (int): Rank of the compressed query latent.
        dense_intermediate_size (int): FFN hidden dim for dense layers.
        moe_intermediate_size (int): FFN hidden dim of each MoE expert.
        num_experts (int): Total number of routable experts.
        num_shared_experts (int): Number of always-active shared experts.
        top_k (int): Number of experts routed to per token.
        first_k_dense_replace (int): Number of leading dense (non-MoE) layers.
        routed_scaling_factor (float): Multiplicative factor applied to the
            combined routed-expert output.
        rope_theta (float): RoPE base frequency.
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
        max_seq_len: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        dense_intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        first_k_dense_replace: int = 1,
        routed_scaling_factor: float = 2.827,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)

        layers = nn.ModuleList(
            [
                KimiK2Block(
                    embed_dim,
                    num_heads,
                    max_seq_len,
                    kv_lora_rank,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                    v_head_dim,
                    q_lora_rank,
                    dense_intermediate_size,
                    moe_intermediate_size,
                    num_experts,
                    num_shared_experts,
                    top_k,
                    use_moe=layer_idx >= first_k_dense_replace,
                    routed_scaling_factor=routed_scaling_factor,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    rms_norm_eps=rms_norm_eps,
                )
                for layer_idx in range(num_layers)
            ]
        )
        final_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        output_head = OutputHead(
            embed_dim,
            vocab_size,
            tied_embedding=embedding,
            tie_weights=tie_weights,
            use_norm=False,
        )

        super().__init__([embedding, final_norm, output_head])
        self.transformer_blocks = layers

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.blocks[0](x)
        for block in self.transformer_blocks:
            x = block(x, **kwargs)
        x = self.blocks[1](x)
        return self.blocks[2](x)


class KimiK2_1T(KimiK2Model):
    """Kimi K2 (1T total, ~32B active; language-model component)."""

    def __init__(self):
        super().__init__(
            vocab_size=163840,
            embed_dim=7168,
            num_layers=61,
            num_heads=64,
            max_seq_len=131072,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=1536,
            dense_intermediate_size=18432,
            moe_intermediate_size=2048,
            num_experts=384,
            num_shared_experts=1,
            top_k=8,
            first_k_dense_replace=1,
            routed_scaling_factor=2.827,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            tie_weights=False,
        )
