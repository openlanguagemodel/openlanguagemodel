# How Transformer Blocks Work: Attention, MLPs, Norms, and Residuals

A transformer language model is mostly repeated blocks. Each block takes token
representations in, lets them communicate, transforms them independently, and
returns the same shape so the next block can continue.

In OpenLanguageModel, that structure is written directly:

```python
from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm

block = Block([
    Residual(Block([
        RMSNorm(d_model),
        GroupedQueryAttention(d_model, num_heads, num_kv_heads, context_length),
    ])),
    Residual(Block([
        RMSNorm(d_model),
        SwiGLUFFN(d_model),
    ])),
])
```

## Attention

Attention lets each token read other tokens. In a causal language model, token
10 can read tokens 0 through 10, but not token 11. Modern models often use RoPE,
grouped-query attention, or Flash/SDPA kernels for speed and memory behavior.

OLM exposes those as normal PyTorch modules, so attention is a replaceable
component instead of a hidden method inside a giant model class.

## Feed-Forward Layers

After attention mixes information across positions, the feed-forward layer
transforms each token representation independently. Recent models often use
gated MLPs like SwiGLU or GeGLU because they work well at scale.

## Norms and Residuals

Normalization keeps activations numerically stable. Residual connections let a
block add useful changes without destroying the representation it received.
Those two details are why the same `[batch, seq, hidden]` shape can pass through
many layers.

## Why Blocks Help

The [OLM Block System](/docs/guides/architecture/) separates what a component
does from how components are wired. That makes it easier to teach transformer
architecture, debug model changes, and run ablations without rewriting a full
forward pass.

For a deeper lesson-oriented explanation, start with
[A Whole Transformer Block](/docs/learn/a-transformer-block/).
