# Modern Language Modelling

GPT-2 (2019) is still a perfectly good place to start, and it's the model you build
in [Your First Language Model](first-model.md). But the field has refined the recipe
quite a bit since. The striking thing is how little of it is genuinely new: a modern
model is mostly GPT-2 with a handful of its components swapped for better ones.

Because OLM is built from interchangeable parts, "modernising" a model is literally
swapping parts. This page is a quick tour of what changed from GPT-2 to a current
model (the Llama 3 / Qwen 2.5 generation), what people reach for now, and the OLM
component for each — so you can read a new model's description and rebuild it yourself.

> This is a practical tour, not a theory lesson. It assumes you know the basic pieces
> — embeddings, attention, feed-forward, normalization, blocks. If those are fuzzy,
> start with [Learn From Scratch](../learn/index.md).

## The baseline: GPT-2

A GPT-2 block, in the parts you already know: a **learned absolute** positional
embedding, **LayerNorm** (pre-norm), standard **multi-head attention** with bias
terms, and a **GELU** feed-forward. In OLM:

```python
from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual
from olm.nn.attention import MultiHeadAttention
from olm.nn.feedforward import ClassicFFN
from olm.nn.norms import LayerNorm

gpt2_block = Block([
    Residual(Block([LayerNorm(embed_dim), MultiHeadAttention(embed_dim, num_heads, causal=True)])),
    Residual(Block([LayerNorm(embed_dim), ClassicFFN(embed_dim)])),
])
```

Now let's modernise it, one swap at a time.

## Position: learned → RoPE

GPT-2 adds a learned vector for each position slot. It works, but it doesn't extend
past the length it was trained on, and it bakes position into the token vectors. Almost
every recent model — Llama, Qwen, Mistral, and most others — instead uses **RoPE**
(rotary positional embeddings), which encodes position *inside* attention by rotating
the queries and keys. (A few models, like the older MPT, use **ALiBi** instead.)

In OLM, RoPE lives inside the attention component, so you drop the separate positional
embedding entirely and use a RoPE-aware attention:

```python
from olm.nn.attention import MultiHeadAttentionwithRoPE

# replaces a separate AbsolutePositionalEmbedding + plain attention
MultiHeadAttentionwithRoPE(embed_dim, num_heads, max_seq_len, causal=True)
```

## Normalization: LayerNorm → RMSNorm

**RMSNorm** is a lighter normalization that skips the mean-centring LayerNorm does. In
practice it trains just as well and is a little faster, so Llama, Qwen, Mistral, and
Gemma all use it.

```python
from olm.nn.norms import RMSNorm

RMSNorm(embed_dim)   # in place of LayerNorm(embed_dim)
```

## Feed-forward: GELU MLP → SwiGLU

The plain GELU MLP gave way to **gated** feed-forwards, most commonly **SwiGLU**. For
the same compute budget it tends to model a touch better, and it's now standard in
Llama, Qwen, and Mistral. (Gemma uses the closely related **GeGLU**.)

```python
from olm.nn.feedforward import SwiGLUFFN

SwiGLUFFN(embed_dim, bias=False)   # in place of ClassicFFN(embed_dim)
```

## Attention efficiency: MHA → GQA (and Flash)

Two refinements, both about doing the *same* attention more cheaply:

- **Grouped-Query Attention (GQA).** Several query heads share a single set of
  key/value heads, which dramatically cuts the memory the model needs while generating
  text. Llama 2 (larger sizes), Llama 3 (all sizes), Qwen 2, and Mistral all use it.
  (The extreme case, one shared key/value head, is Multi-Query Attention.)
- **FlashAttention.** Exactly the same maths as standard attention, computed by a
  faster, more memory-efficient algorithm — a drop-in speed-up.

```python
from olm.nn.attention import GroupedQueryAttention

# 16 query heads sharing 4 key/value heads; RoPE is built in
GroupedQueryAttention(embed_dim, num_heads=16, num_kv_heads=4, max_seq_len=max_seq_len)
```

OLM also provides `FlashAttention` and `FlashAttentionwithRoPE` if you want the speed-up
with the same result.

## Dropping the biases

Modern models usually remove the bias terms from their linear layers (in both attention
and the feed-forward). It slightly improves stability and trims a few parameters. You've
already seen `bias=False` and `use_bias=False` on the components above — that's all there
is to it.

## At the frontier: Mixture of Experts

The largest recent models — Mixtral, Qwen-MoE, DeepSeek — replace the single
feed-forward with a **Mixture of Experts (MoE)**: many expert feed-forwards, but each
token is routed to only a couple of them. The model gains huge capacity while doing only
a little work per token.

```python
from olm.nn.feedforward.swiglu_moe import SwiGLUMoEFFN

# 8 expert feed-forwards; each token uses the top 2
SwiGLUMoEFFN(embed_dim, num_experts=8, top_k=2, bias=False)
```

## Putting it together: a modern block

Stack those choices up — RoPE (inside GQA), RMSNorm, SwiGLU, no biases — and you have a
Llama-3-style model, assembled from the same `Block`/`Residual`/`Repeat` you already use:

```python
from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual, Repeat
from olm.nn.embeddings import Embedding
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.blocks import OutputHead

vocab_size, embed_dim, num_layers, max_seq_len = 32000, 2048, 16, 4096
num_heads, num_kv_heads = 16, 4

modern = Block([
    Embedding(vocab_size, embed_dim),                       # no separate positional embedding —
    Repeat(lambda: Block([                                  # RoPE lives inside the attention
        Residual(Block([
            RMSNorm(embed_dim),
            GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, max_seq_len),
        ])),
        Residual(Block([
            RMSNorm(embed_dim),
            SwiGLUFFN(embed_dim, bias=False),
        ])),
    ]), num_layers),
    OutputHead(embed_dim, vocab_size),
])
```

Compare it with the GPT-2 block at the top: same skeleton, four components swapped.

## A quick timeline

The broad strokes of how the recipe evolved (sizes and details vary within each family):

| Model (year) | Notable choices |
|---|---|
| **GPT-2** (2019) | learned absolute positions, LayerNorm, multi-head attention, GELU MLP, biases |
| **GPT-3** (2020) | same recipe, scaled up enormously |
| **GPT-J / NeoX** (2021–22) | RoPE positions adopted |
| **Llama** (2023) | RoPE + RMSNorm + SwiGLU, biases dropped |
| **Llama 2** (2023) | adds GQA on the larger sizes |
| **Mistral / Mixtral** (2023) | GQA + sliding-window attention; Mixtral adds sparse MoE |
| **Llama 3** (2024) | GQA across all sizes, much larger (128k) vocabulary |
| **Qwen 2.5** (2024) | RoPE + RMSNorm + SwiGLU + GQA |
| **Gemma 2** (2024) | RMSNorm + GeGLU + GQA |

The "modern default" most teams reach for today: **RoPE + RMSNorm + SwiGLU + GQA, no
biases** — with **MoE** when scaling to the frontier.

## You don't have to assemble it by hand

OLM ships these architectures as ready-made reference models, each built from exactly the
components above:

```python
from olm.models import Llama3_1_8B, Qwen2_5_7B

model = Llama3_1_8B()
```

See them all in the [`olm.models` reference](../api/models.md). When you want to change
one — swap GQA for plain attention, try GeGLU instead of SwiGLU, drop in an MoE — you now
know it's a single-component edit. The full menu of parts is in
[Building Blocks](../guides/components.md), and [Custom Architectures](custom-architecture.md)
walks through assembling your own.

## What you learned

- A modern LM is mostly GPT-2 with better-chosen components, not a different design.
- The common modern recipe is **RoPE + RMSNorm + SwiGLU + GQA + no biases**, with **MoE**
  at the frontier.
- In OLM each choice is a one-component swap, and the reference models in `olm.models`
  bundle them for you.
