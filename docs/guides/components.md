# Building Blocks

This guide is a conceptual tour of the components in `olm.nn` — the layers you compose into models. For exact signatures and parameters, follow the links into the [API reference](../api/nn.md); this page focuses on what each component is and when to choose it.

Every component here is a plain `torch.nn.Module` and follows a base-class-per-family pattern: an abstract base defines the interface, and concrete classes implement variants. To add your own variant, subclass the relevant base.

## Embeddings

Embeddings map token ids to vectors and inject position information. How position is handled determines which attention layer you pair them with (see [Attention](#attention) below).

**Token embedding** — [`Embedding(vocab_size, embedding_dim)`](../api/nn.md#embedding) is the lookup table from token ids to vectors. Every model starts with one.

**Positional schemes:**

| Component | Where position enters | Notable use |
|-----------|----------------------|-------------|
| [`AbsolutePositionalEmbedding`](../api/nn.md#absolutepositionalembedding) | Learned vector added to token embeddings | GPT-2 |
| [`SinusoidalPositionalEmbedding`](../api/nn.md#sinusoidalpositionalembedding) | Fixed sin/cos added to token embeddings | Original Transformer |
| [`RotaryPositionalEmbedding`](../api/nn.md#rotarypositionalembedding) (RoPE) | Rotation applied to queries/keys inside attention | Llama, Qwen, most modern LLMs |
| [`ALiBiPositionalBias`](../api/nn.md#alibipositionalbias) | Distance-based bias added to attention scores | Length extrapolation |

For long-context work, RoPE has scaled variants — [`PartialRotaryPositionalEmbedding`](../api/nn.md#partialrotarypositionalembedding) (rotate only a fraction of dimensions, as in some Llama configs) and [`ScaledRotaryPositionalEmbedding`](../api/nn.md#scaledrotarypositionalembedding) (linear, NTK, dynamic-NTK, YaRN, and XPos scaling for extending context beyond the trained length).

> [!NOTE]
> Absolute and sinusoidal embeddings are *added to token embeddings* at the input. RoPE and ALiBi act *inside attention*. This is why the choice of positional scheme is tied to the choice of attention layer.

## Attention

OLM provides a family of attention mechanisms that share a common base ([`AttentionBase`](../api/nn.md#attentionbase), or [`AttentionwithRoPEBase`](../api/nn.md#attentionwithropebase) for the rotary variants).

| Layer | Position handling | Backend | Use when |
|-------|-------------------|---------|----------|
| [`MultiHeadAttention`](../api/nn.md#multiheadattention) | none (add a positional embedding at the input) | explicit softmax | Teaching, full control, GPT-2-style models |
| [`MultiHeadAttentionwithRoPE`](../api/nn.md#multiheadattentionwithrope) | RoPE, internal | explicit softmax | Modern decoder with readable internals |
| [`FlashAttention`](../api/nn.md#flashattention) | none | `scaled_dot_product_attention` | Speed/memory, with an input positional embedding |
| [`FlashAttentionwithRoPE`](../api/nn.md#flashattentionwithrope) | RoPE, internal | `scaled_dot_product_attention` | Fast modern decoder (recommended default) |
| [`GroupedQueryAttention`](../api/nn.md#groupedqueryattention) | RoPE, internal | `scaled_dot_product_attention` | Large models / inference efficiency |
| [`MultiHeadAttentionwithALiBi`](../api/nn.md#multiheadattentionwithalibi) | ALiBi, internal | explicit softmax | Training short, testing long |

A few practical notes:

- The `Flash*` and `GroupedQueryAttention` layers call PyTorch's `scaled_dot_product_attention`, which automatically dispatches to fused Flash-Attention kernels when the hardware and inputs allow. The plain `MultiHeadAttention*` layers compute attention explicitly (matmul → softmax → matmul), which is slower but easy to read and modify.
- **Grouped-query attention** uses fewer key/value heads than query heads. Setting `num_kv_heads == num_heads` recovers standard MHA; `num_kv_heads == 1` gives multi-query attention. It also offers optional QK-normalization (a Qwen-2 feature).
- For causal language modeling, pass `causal=True` (the layer applies a causal mask) — or let the SDPA backend handle it.

```python
from olm.nn.attention import FlashAttentionwithRoPE, GroupedQueryAttention

# A fast, modern self-attention layer
attn = FlashAttentionwithRoPE(embed_dim=768, num_heads=12, max_seq_len=2048, causal=True)

# Grouped-query attention: 32 query heads, 8 KV heads
gqa = GroupedQueryAttention(embed_dim=4096, num_heads=32, num_kv_heads=8, max_seq_len=8192)
```

## Normalization

| Layer | Formula | Used by |
|-------|---------|---------|
| [`LayerNorm`](../api/nn.md#layernorm) | normalize by mean and variance, then scale and shift | GPT-2, BERT |
| [`RMSNorm`](../api/nn.md#rmsnorm) | normalize by root-mean-square only, then scale | Llama, Qwen, Gemma |

Both compute in float32 internally for numerical stability and cast back to the input dtype, which matters under mixed precision. RMSNorm drops the mean-centering and bias of LayerNorm, making it slightly cheaper; it is the common choice in recent LLMs.

## Feed-forward networks

The position-wise feed-forward network (FFN) is applied to every token independently.

| Layer | Structure | Used by |
|-------|-----------|---------|
| [`ClassicFFN`](../api/nn.md#classicffn) | `Linear → activation → Linear` (default hidden = 4×) | GPT-2 (GELU) |
| [`SwiGLUFFN`](../api/nn.md#swigluffn) | gated: `(SiLU(W₁x) ⊙ W₂x) → Linear` | Llama, PaLM |
| [`GeGLUFFN`](../api/nn.md#gegluffn) | gated, GELU variant of the above | Gemma |

Gated FFNs (SwiGLU, GeGLU) split the up-projection into a *gate* and a *value*, multiply them elementwise, and tend to outperform a plain MLP at equal parameter count — which is why modern models use them. OLM's gated FFNs default their hidden dimension via an `ff_multiplier` so the gated and ungated variants land at comparable parameter counts.

### Mixture-of-Experts

For sparse scaling, each FFN has a Mixture-of-Experts counterpart — [`ClassicMoEFFN`](../api/nn.md#classicmoeffn), `SwiGLUMoEFFN`, and `GeGLUMoEFFN` — built on [`MoEFeedForwardBase`](../api/nn.md#moefeedforwardbase). A [`MoERouter`](../api/nn.md#moerouter) performs top-k softmax gating over `num_experts` experts, optionally with a number of always-on `num_shared_experts`. Only the selected experts run per token, so capacity grows without a proportional increase in compute.

> [!NOTE]
> The current MoE layers focus on readable top-k routing and expert composition.
> Load-balancing auxiliary losses and expert-parallel dispatch are roadmap items,
> so add your own auxiliary term in a custom training loop if your experiment
> depends on balanced expert usage.

```python
from olm.nn.feedforward import SwiGLUFFN

ffn = SwiGLUFFN(embed_dim=768)         # dense gated FFN

from olm.nn.feedforward.swiglu_moe import SwiGLUMoEFFN
moe = SwiGLUMoEFFN(embed_dim=768, num_experts=8, top_k=2)   # sparse MoE FFN
```

## Activations

All activations subclass [`ActivationBase`](../api/nn.md#activationbase) and are registered in the `ACTIVATIONS` registry, so they can be selected by name in config-driven workflows.

- **Pointwise:** `ReLU`, `LeakyReLU`, `ELU`, `SELU`, `PReLU`, `GELU`, `SiLU`, `Mish`, `Softplus`, `Tanh`, `Sigmoid`, `Softmax`, `Identity`.
- **Gated linear units** (used inside gated FFNs): `SwiGLU`, `GeGLU`, `ReGLU`, `LiGLU`, `GLU`. These halve their input along the last dimension into two parts and gate one with the other.

```python
from olm.core.registry import ACTIVATIONS

act = ACTIVATIONS.get("gelu")()   # look up by name
```

## Putting it together

These components are designed to be combined with the [Block system](architecture.md). A single modern transformer layer pairs a normalization, an attention, and a gated FFN inside residual connections:

```python
from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual
from olm.nn.norms import RMSNorm
from olm.nn.attention import FlashAttentionwithRoPE
from olm.nn.feedforward import SwiGLUFFN

def layer(d_model, n_heads, max_seq_len):
    return Block([
        Residual(Block([RMSNorm(d_model),
                        FlashAttentionwithRoPE(d_model, n_heads, max_seq_len, causal=True)])),
        Residual(Block([RMSNorm(d_model), SwiGLUFFN(d_model)])),
    ])
```

## Next steps

- [The Block System](architecture.md) — how to wire these components into a model.
- [Tutorial: Custom Architectures](../tutorials/custom-architecture.md) — build and train a model from these parts.
- [`olm.nn` API reference](../api/nn.md) — full signatures for every component above.
