from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


class Nanbeige4_1_Block(Block):
    """
    Transformer block for Nanbeige 4.1.

    Nanbeige 4.1 follows the Llama 3.2 recipe closely -- pre-norm GQA
    with SwiGLU, but with untied embeddings and a high RoPE base of 70M
    for 262K context.

    Structure:
        x = x + GQA(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))
    """

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int = 128,
        dropout: float = 0.0,
        rope_theta: float = 70000000.0,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__([
            Residual(Block([
                RMSNorm(embed_dim, eps=rms_norm_eps),
                GroupedQueryAttention(
                    embed_dim, num_heads, num_kv_heads, max_seq_len,
                    head_dim=head_dim,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    use_bias=False,
                ),
            ])),
            Residual(Block([
                RMSNorm(embed_dim, eps=rms_norm_eps),
                SwiGLUFFN(
                    embed_dim, hidden_dim=intermediate_size,
                    dropout=dropout, bias=False,
                ),
            ])),
        ])


class Nanbeige4_1_Model(Block):
    """
    Base class for Nanbeige 4.1 models.

    Dense decoder-only transformer with a Llama-like architecture.
    Untied input/output embeddings and a 166K vocabulary targeting
    multilingual CJK coverage.

    Structure:
        Embedding -> [Nanbeige4_1_Block] x N -> RMSNorm -> OutputHead

    Forward:
        Token IDs ``[batch, seq_len]`` -> logits ``[batch, seq_len, vocab_size]``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int = 128,
        rope_theta: float = 70000000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-5,
        tie_weights: bool = False,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        super().__init__([
            embedding,
            Repeat(
                lambda: Nanbeige4_1_Block(
                    embed_dim, intermediate_size, num_heads, num_kv_heads,
                    max_seq_len, head_dim, dropout, rope_theta, rms_norm_eps,
                ),
                num_layers,
            ),
            RMSNorm(embed_dim, eps=rms_norm_eps),
            OutputHead(
                embed_dim, vocab_size,
                tied_embedding=embedding if tie_weights else None,
                tie_weights=tie_weights,
                use_norm=False,
            ),
        ])


class Nanbeige4_1_3B(Nanbeige4_1_Model):
    """Nanbeige 4.1 3B -- small on-device model with 262K context."""

    def __init__(self):
        super().__init__(
            vocab_size=166144,
            embed_dim=2560,
            intermediate_size=10496,
            num_layers=32,
            num_heads=20,
            num_kv_heads=4,
            max_seq_len=262144,
            head_dim=128,
            rope_theta=70000000.0,
            rms_norm_eps=1e-5,
            tie_weights=False,
        )
