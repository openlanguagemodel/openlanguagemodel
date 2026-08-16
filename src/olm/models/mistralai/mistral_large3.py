from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual
from olm.nn.attention import MultiHeadLatentAttention
from olm.nn.feedforward import SwiGLUFFN, SwiGLUMoEFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


class MistralLarge3Block(Block):
    """
    A single decoder block for Mistral Large 3.

    Pre-norm structure with Multi-head Latent Attention and either a dense
    SwiGLU feed-forward (the first few layers) or a fine-grained MoE block with
    a shared expert (the remaining layers):

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
        dropout: float,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        attn = MultiHeadLatentAttention(
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

        if use_moe:
            ffn = SwiGLUMoEFFN(
                embed_dim,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                hidden_dim=moe_intermediate_size,
                bias=False,
            )
        else:
            ffn = SwiGLUFFN(
                embed_dim, hidden_dim=dense_intermediate_size, bias=False
            )

        super().__init__(
            [
                Residual(Block([RMSNorm(embed_dim, eps=rms_norm_eps), attn])),
                Residual(Block([RMSNorm(embed_dim, eps=rms_norm_eps), ffn])),
            ]
        )


class MistralLarge3Model(Block):
    """
    Base class for the Mistral Large 3 language model.

    Structure:
        Embedding -> [MistralLarge3Block] x N -> RMSNorm -> OutputHead.

    The first ``first_k_dense_replace`` layers use a dense SwiGLU feed-forward;
    the remaining layers use a fine-grained MoE with a shared expert. Attention
    is Multi-head Latent Attention throughout. Mistral Large 3 does not tie
    input/output embeddings; the named preset passes ``tie_weights=False``.

    Notes:
        - This models the language-model component only; the reference checkpoint
          also includes a vision encoder, which is omitted here.
        - The published config does not specify the router gating function, so the
          library default (softmax top-k routing) is used.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
        ``[batch, seq_len, vocab_size]``.

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
        first_k_dense_replace: int = 3,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        layers = Block(
            [
                MistralLarge3Block(
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
                    dropout=dropout,
                    rope_theta=rope_theta,
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


class MistralLarge3(MistralLarge3Model):
    """Mistral Large 3 (675B total, ~41B active; language-model component)."""

    def __init__(self):
        super().__init__(
            vocab_size=131072,
            embed_dim=7168,
            num_layers=61,
            num_heads=128,
            max_seq_len=262144,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=1536,
            dense_intermediate_size=16384,
            moe_intermediate_size=4096,
            num_experts=128,
            num_shared_experts=1,
            top_k=4,
            first_k_dense_replace=3,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            tie_weights=False,
        )
