"""Independent closed-form parameter-count formulas for each OLM model family.

Each function takes the same kwargs as the corresponding base-class __init__
and returns the exact unique parameter count (de-duplicated, tied weights
counted once). All formulas were derived from the source architecture and
validated by comparison to ``sum(p.numel() for p in model.parameters())``
on reduced models.

Architectural assumptions encoded here:
  GPT-2
    - Token embedding + learned positional embedding (both trainable)
    - MHA with bias (q/k/v/out each d×d + bias)
    - ClassicFFN 4× hidden, bias in both projections
    - 2 LayerNorms per block (gamma + beta = 2d each)
    - OutputHead with LayerNorm (use_norm=True default)
    - Tied LM head (counts 0 extra params)

  Llama 2 / Llama 3 / Phi-3 / Phi-4  (SwiGLU/GeGLU, GQA or MHA, no bias)
    - Token embedding only (RoPE has no params)
    - GQA: q_proj(d,d), k_proj(d,kv_dim), v_proj(d,kv_dim), out_proj(d,d) — no bias
      (Same formula covers MHA via kv_dim = num_kv_heads*(d//num_heads))
    - SwiGLU / GeGLU: up_proj(d, 2×intermediate) + down_proj(intermediate, d) — no bias
    - 2 RMSNorms per block (weight only = d each)
    - 1 final RMSNorm
    - Tied LM head (0 extra)

  Qwen 2.5  (like Llama but qkv_bias=True, out_proj no bias)
    - Same as Llama formula plus q_bias(d) + k_bias(kv_dim) + v_bias(kv_dim)

  Gemma 2   (GeGLU, GQA with explicit head_dim, 4 RMSNorms per block)
    - q_dim = num_heads * head_dim  (may differ from embed_dim)
    - Attn: d*q_dim + 2*d*kv_dim + q_dim*d = 2*d*q_dim + 2*d*kv_dim (no bias)
    - GeGLU: same structure/count as SwiGLU
    - 4 RMSNorms per block (sandwich norm)
    - Gemma2FinalLogitSoftcap: no trainable parameters

  OLMo   (LayerNorm elementwise_affine=False → 0 norm params, SwiGLU, MHA no bias)
    - Token embedding only
    - 4×d² attn params (FlashAttentionwithRoPE, bias=False)
    - 3×d×intermediate FFN params
    - No norm params anywhere

  OPT    (MHA + bias, ClassicFFN + bias, learned positional embedding, LayerNorm)
    - Token embedding + learned positional embedding (max_seq_len=2048 hardcoded)
    - MHA with bias: 4*(d²+d)
    - ClassicFFN (intermediate, bias): 2*d*intermediate + intermediate + d
    - 2 LayerNorms per block (2d each)
    - 1 final LayerNorm
    - OutputHead use_norm=False, tied (0 extra)
"""

from __future__ import annotations


def count_params_unique(model) -> int:
    """De-duplicated parameter count (tied params counted once)."""
    seen: set = set()
    total = 0
    for p in model.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            total += p.numel()
    return total


# ---------------------------------------------------------------------------
# GPT-2 style
# ---------------------------------------------------------------------------

def gpt2_params(
    vocab_size: int,
    embed_dim: int,
    num_layers: int,
    max_seq_len: int,
    **_kw,
) -> int:
    """
    Exact unique param count for GPT2Model.

    Components:
      token_embedding:  vocab_size * d
      pos_embedding:    max_seq_len * d
      per block (L):
        attn (MHA, bias):   4*(d² + d)
        ffn  (4×d, bias):   2*(4d*d + 4d) + (d)  →  8d² + 5d
          actually: up(d,4d+b) = 4d²+4d, down(4d,d+b) = 4d²+d  → total 8d²+5d
        2 LayerNorms:        4d
      output_head LayerNorm: 2d  (use_norm=True, default)
      tied lm_head:           0
    """
    d = embed_dim
    attn_per_block = 4 * d * d + 4 * d
    ffn_per_block = 8 * d * d + 5 * d
    ln_per_block = 4 * d          # 2 LayerNorms × (gamma + beta)
    block_params = attn_per_block + ffn_per_block + ln_per_block
    output_head_ln = 2 * d
    return (
        vocab_size * d
        + max_seq_len * d
        + num_layers * block_params
        + output_head_ln
    )


# ---------------------------------------------------------------------------
# Llama 2 / Llama 3 / Phi-3 / Phi-4  (SwiGLU/GeGLU, GQA, no bias)
# ---------------------------------------------------------------------------

def llama_swiglu_gqa_params(
    vocab_size: int,
    embed_dim: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    **_kw,
) -> int:
    """
    Exact unique param count for Llama2Model, Llama3Model, Phi3Model, Phi4Model.

    Works for both MHA (num_kv_heads == num_heads) and GQA, because the formula
    collapses to 4d² when kv_dim == d.

    Components:
      token_embedding:  vocab_size * d
      per block (L):
        GQA (no bias):  2d² + 2*d*kv_dim   (q+out = 2d²; k+v = 2*d*kv_dim)
        SwiGLU/GeGLU:   3*d*intermediate    (up 2× + down 1×, no bias)
        2 RMSNorms:     2d
      final RMSNorm:    d
      tied lm_head:     0
    """
    d = embed_dim
    head_dim = d // num_heads
    kv_dim = num_kv_heads * head_dim
    attn_per_block = 2 * d * d + 2 * d * kv_dim
    ffn_per_block = 3 * d * intermediate_size
    rms_per_block = 2 * d
    block_params = attn_per_block + ffn_per_block + rms_per_block
    final_rms = d
    return vocab_size * d + num_layers * block_params + final_rms


# ---------------------------------------------------------------------------
# Qwen 2.5  (like Llama but adds qkv_bias)
# ---------------------------------------------------------------------------

def qwen2_params(
    vocab_size: int,
    embed_dim: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    **_kw,
) -> int:
    """
    Exact unique param count for Qwen2Model.

    Differs from llama_swiglu_gqa_params by qkv_bias=True:
      q_bias:  d
      k_bias:  kv_dim
      v_bias:  kv_dim
    (out_proj has no bias.)
    """
    d = embed_dim
    head_dim = d // num_heads
    kv_dim = num_kv_heads * head_dim
    # weight part same as Llama GQA
    attn_weights = 2 * d * d + 2 * d * kv_dim
    # bias part
    attn_bias = d + 2 * kv_dim
    ffn_per_block = 3 * d * intermediate_size
    rms_per_block = 2 * d
    block_params = attn_weights + attn_bias + ffn_per_block + rms_per_block
    final_rms = d
    return vocab_size * d + num_layers * block_params + final_rms


# ---------------------------------------------------------------------------
# Gemma 2  (GeGLU, explicit head_dim, 4 RMSNorms per block)
# ---------------------------------------------------------------------------

def gemma2_params(
    vocab_size: int,
    embed_dim: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    **_kw,
) -> int:
    """
    Exact unique param count for Gemma2Model.

    head_dim is specified explicitly and may differ from embed_dim // num_heads.
    q_dim = num_heads * head_dim  (can exceed embed_dim for some configs).

    Components:
      token_embedding: vocab_size * d  (Gemma2Embedding adds no extra params)
      per block (L):
        GQA (no bias): 2*d*q_dim + 2*d*kv_dim
        GeGLU:         3*d*intermediate   (same structure as SwiGLU)
        4 RMSNorms:    4d
      final RMSNorm:   d
      Gemma2FinalLogitSoftcap: 0 trainable params
      tied lm_head:    0
    """
    d = embed_dim
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    attn_per_block = 2 * d * q_dim + 2 * d * kv_dim
    ffn_per_block = 3 * d * intermediate_size
    rms_per_block = 4 * d
    block_params = attn_per_block + ffn_per_block + rms_per_block
    final_rms = d
    return vocab_size * d + num_layers * block_params + final_rms


# ---------------------------------------------------------------------------
# OLMo  (no norm params, MHA no bias, SwiGLU no bias)
# ---------------------------------------------------------------------------

def olmo_params(
    vocab_size: int,
    embed_dim: int,
    intermediate_size: int,
    num_layers: int,
    **_kw,
) -> int:
    """
    Exact unique param count for OLMoModel.

    LayerNorm(elementwise_affine=False) contributes 0 params throughout.

    Components:
      token_embedding: vocab_size * d
      per block (L):
        MHA (FlashAttentionwithRoPE, bias=False): 4*d²
        SwiGLU (bias=False): 3*d*intermediate
        2 LN (no affine):    0
      final LN (no affine):  0
      tied lm_head:           0
    """
    d = embed_dim
    attn_per_block = 4 * d * d
    ffn_per_block = 3 * d * intermediate_size
    block_params = attn_per_block + ffn_per_block
    return vocab_size * d + num_layers * block_params


# ---------------------------------------------------------------------------
# OPT  (MHA + bias, ClassicFFN + bias, learned pos embed, LayerNorms)
# ---------------------------------------------------------------------------

def opt_params(
    vocab_size: int,
    embed_dim: int,
    intermediate_size: int,
    num_layers: int,
    **_kw,
) -> int:
    """
    Exact unique param count for OPTModel.

    OPTModel hardcodes max_seq_len=2048 for the positional embedding.

    Components:
      token_embedding:  vocab_size * d
      pos_embedding:    2048 * d
      per block (L):
        MHA (bias):      4*(d² + d)
        ClassicFFN(bias): 2*d*intermediate + intermediate + d
        2 LayerNorms:    4d
      final LayerNorm:   2d
      OutputHead use_norm=False, tied: 0
    """
    d = embed_dim
    pos_embed = 2048 * d
    attn_per_block = 4 * d * d + 4 * d
    ffn_per_block = 2 * d * intermediate_size + intermediate_size + d
    ln_per_block = 4 * d
    block_params = attn_per_block + ffn_per_block + ln_per_block
    final_ln = 2 * d
    return (
        vocab_size * d
        + pos_embed
        + num_layers * block_params
        + final_ln
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FORMULA_REGISTRY = {
    "gpt2_params": gpt2_params,
    "llama_swiglu_gqa_params": llama_swiglu_gqa_params,
    "qwen2_params": qwen2_params,
    "gemma2_params": gemma2_params,
    "olmo_params": olmo_params,
    "opt_params": opt_params,
}
