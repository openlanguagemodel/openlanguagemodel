# `olm.nn.attention.gqa`

Source: [`src/olm/nn/attention/gqa.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/gqa.py#L1)

## Classes

### `GroupedQueryAttention(embed_dim: int, num_heads: int, num_kv_heads: int, max_seq_len: int, head_dim: int | None = None, dropout: float = 0.0, rope_theta: float = 10000.0, use_bias: bool = False, qkv_bias: bool = False, use_qk_norm: bool = False, rms_norm_eps: float = 1e-06, attention_scale: float | None = None, attn_logit_softcap: float | None = None)`

**Bases:** `Module`

Source: [`src/olm/nn/attention/gqa.py:11`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/gqa.py#L11)

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

##### `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/attention/gqa.py:104`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/gqa.py#L104)

Forward pass of Grouped Query Attention.

Args:
    x (torch.Tensor): Input tensor of shape [batch, seq_len, embed_dim].
    mask (torch.Tensor, optional): Attention mask of shape [batch, 1, seq_len, seq_len]
        or [batch, seq_len, seq_len]. Defaults to None.

Returns:
    torch.Tensor: Output tensor of shape [batch, seq_len, embed_dim].
