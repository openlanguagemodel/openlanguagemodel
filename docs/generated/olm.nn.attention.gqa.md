# `olm.nn.attention.gqa`

## Classes

### `GroupedQueryAttention(embed_dim: int, num_heads: int, num_kv_heads: int, max_seq_len: int, head_dim: int | None = None, dropout: float = 0.0, rope_theta: float = 10000.0, use_bias: bool = False, qkv_bias: bool = False, use_qk_norm: bool = False, rms_norm_eps: float = 1e-06, attention_scale: float | None = None, attn_logit_softcap: float | None = None)`

Grouped Query Attention (GQA) with Rotary Positional Embeddings.

GQA is a distinct attention mechanism where the number of Key/Value heads is smaller
than the number of Query heads. This reduces memory bandwidth usage during inference
(smaller KV cache) while maintaining performance close to Multi-Head Attention (MHA).

If num_kv_heads == num_heads, this is equivalent to MHA.
If num_kv_heads == 1, this is equivalent to Multi-Query Attention (MQA).

Args:
    embed_dim (int): Total dimension of the model.
    num_heads (int): Number of Query heads.
    num_kv_heads (int): Number of Key/Value heads. Must divide num_heads.
    max_seq_len (int): Maximum sequence length for RoPE.
    dropout (float, optional): Dropout probability. Defaults to 0.0.
    rope_theta (float, optional): Base frequency for RoPE. Defaults to 10000.0.
    use_bias (bool, optional): Whether to use bias in linear projections. Defaults to False.

#### Methods

- `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Forward pass of Grouped Query Attention.
