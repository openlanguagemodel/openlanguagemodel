
import torch
import torch.nn as nn
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward.geglu_ffn import GeGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding

class GemmaBlock(nn.Module):
    """
    A single Transformer block for the Gemma architecture.

    Uniquely employs a 'Post-Norm-ish' residual structure as described in technical 
    reports: `x = RMSNorm(x + f(x))`. This ensures the input to the next layer is 
    unit variance, without the strict Pre-Norm `x + f(Norm(x))` path.

    Structure:
        residual = x
        x = Attention(x)  [No Norm before Attn]
        x = RMSNorm(residual + x)
        
        residual = x
        x = GeGLU(x)      [No Norm before MLP]
        x = RMSNorm(residual + x)

    Args:
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        head_dim (int): Explicit head dimension (Gemma uses massive heads).
        max_seq_len (int): Max context length.
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base.
        logit_softcapping (float, optional): Attention logit soft-capping value (e.g. 50.0).
    """
    def __init__(self, embed_dim, intermediate_size, num_heads, num_kv_heads, head_dim, max_seq_len, dropout, rope_theta, logit_softcapping=None):
        super().__init__()
        # Attention Sub-layer
        self.attn = GroupedQueryAttention(
            embed_dim, 
            num_heads, 
            num_kv_heads, 
            max_seq_len, 
            head_dim=head_dim, 
            dropout=dropout, 
            rope_theta=rope_theta,
            use_bias=False,
            logit_softcapping=logit_softcapping
        )
        self.post_attn_norm = RMSNorm(embed_dim, eps=1e-6)
        
        # MLP Sub-layer
        self.mlp = GeGLUFFN(embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False)
        self.post_mlp_norm = RMSNorm(embed_dim, eps=1e-6)
        
    def forward(self, x):
        # Attention
        residual = x
        x = self.attn(x)
        x = self.post_attn_norm(x + residual)
        
        # MLP
        residual = x
        x = self.mlp(x)
        x = self.post_mlp_norm(x + residual)
        
        return x

class GemmaModel(nn.Module):
    """
    Base class for the Gemma 2 architecture.
    
    Features:
    - Post-Norm Residual structure.
    - GeGLU Activations.
    - 256 Head Dimension (larger than standard 128).
    - Large Vocabulary (~256k).
    - Logit Soft-capping (Attention=50.0, Final=30.0) standard for Gemma 2.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        intermediate_size (int): FFN dimension.
        num_layers (int): Number of layers.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of KV heads.
        head_dim (int): Explicit head dimension size.
        max_seq_len (int, optional): Max context. Defaults to 8192.
        rope_theta (float, optional): RoPE base. Defaults to 10000.0.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        attn_logit_softcapping (float, optional): Soft-capping for attention logits. Defaults to 50.0.
        final_logit_softcapping (float, optional): Soft-capping for final logits. Defaults to 30.0.
    """
    def __init__(self, vocab_size, embed_dim, intermediate_size, num_layers, num_heads, num_kv_heads, head_dim, max_seq_len=8192, rope_theta=10000.0, dropout=0.0, attn_logit_softcapping=50.0, final_logit_softcapping=30.0):
        super().__init__()
        
        self.embedding = Embedding(vocab_size, embed_dim)
        
        self.transformer = Repeat(lambda: GemmaBlock(
            embed_dim, intermediate_size, num_heads, num_kv_heads, head_dim, max_seq_len, dropout, rope_theta, logit_softcapping=attn_logit_softcapping
        ), num_layers)
        
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.final_logit_softcapping = final_logit_softcapping
        
        # Note: We can't usage Block directly if we need to intercept the output of self.head for capping.
        # But Block calls sequential. 
        # I need a custom forward to apply final capping.
        # So I will not usage Block for the top level, or I will wrap the head.
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        logits = self.head(x)
        
        if self.final_logit_softcapping is not None:
             logits = self.final_logit_softcapping * torch.tanh(logits / self.final_logit_softcapping)
             
        return logits

class Gemma2_27B(GemmaModel):
    """Gemma 2 27B Implementation."""
    def __init__(self):
        super().__init__(
            vocab_size=256000,
            embed_dim=4608,
            intermediate_size=36864,
            num_layers=46,
            num_heads=32,
            num_kv_heads=16,
            head_dim=128, 
            max_seq_len=8192,
            rope_theta=10000.0,
            attn_logit_softcapping=50.0,
            final_logit_softcapping=30.0
        )

class Gemma2_9B(GemmaModel):
    """Gemma 2 9B Implementation."""
    def __init__(self):
        super().__init__(
            vocab_size=256000,
            embed_dim=3584,
            intermediate_size=14336,
            num_layers=42,
            num_heads=16,
            num_kv_heads=8,
            head_dim=256,
            max_seq_len=8192,
            rope_theta=10000.0,
            attn_logit_softcapping=50.0,
            final_logit_softcapping=30.0
        )

class Gemma2_2B(GemmaModel):
    """Gemma 2 2B Implementation."""
    def __init__(self):
        super().__init__(
            vocab_size=256000,
            embed_dim=2304,
            intermediate_size=9216,
            num_layers=26,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            max_seq_len=8192,
            rope_theta=10000.0,
            attn_logit_softcapping=50.0,
            final_logit_softcapping=30.0
        )
