# Custom Architectures

The prebuilt [`LM`](../api/nn.md#lm) is convenient, but the reason OLM exists is to let you build your own architectures. In this tutorial you will assemble a GPT-style model by hand from blocks and combinators, then package it as a reusable class — the same way the models in [`olm.models`](../api/models.md) are written.

This tutorial builds on [The Block System](../guides/architecture.md) and [Building Blocks](../guides/components.md). Skim those first if `Block`, `Residual`, and `Repeat` are unfamiliar.

## The plan

We will build a classic GPT-style decoder:

```text
input ids
  -> token embedding + learned positional embedding
  -> N x [ x + Attn(LN(x)) ;  x + FFN(LN(x)) ]
  -> LayerNorm + Linear head
```

Each arrow is a block, each `+` is a `Residual`, and the `N x` is a `Repeat`.

## 1. One transformer layer

Start with a single decoder layer — two residual sub-blocks. We use causal [`FlashAttention`](../api/nn.md#flashattention) and a classic GELU MLP.

```python
from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual
from olm.nn.attention import FlashAttention
from olm.nn.feedforward import ClassicFFN
from olm.nn.norms import LayerNorm

def gpt_layer(d_model, n_heads, dropout=0.1):
    return Block([
        Residual(Block([
            LayerNorm(d_model),
            FlashAttention(d_model, n_heads, dropout=dropout, causal=True),
        ])),
        Residual(Block([
            LayerNorm(d_model),
            ClassicFFN(d_model, dropout=dropout),
        ])),
    ])
```

## 2. Stack it into a model

Now wrap embeddings, a `Repeat` of layers, and an output head into one `Block`. We subclass `Block` so the result is a reusable class with a clean constructor.

```python
from olm.nn.structure.combinators import Repeat
from olm.nn.embeddings import Embedding, AbsolutePositionalEmbedding
from olm.nn.blocks import OutputHead

class MiniGPT(Block):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 max_seq_len=256, dropout=0.1):
        super().__init__([
            Block([
                Embedding(vocab_size, d_model),
                AbsolutePositionalEmbedding(max_seq_len, d_model, dropout),
            ]),
            Repeat(lambda: gpt_layer(d_model, n_heads, dropout), n_layers),
            OutputHead(d_model, vocab_size),
        ])
```

That is a complete, trainable GPT — no `forward()` required, because `Block` already defines it, and the architecture reads top to bottom.

> [!TIP]
> GPT-2 ties its output projection to its input embedding to save parameters and improve quality. Because blocks expose their children, you can do this in `__init__` after `super().__init__(...)`:
>
> ```python
> # OutputHead is [LayerNorm, Linear]; the Embedding wrapper holds .embedding
> self.blocks[2].blocks[1].weight = self.blocks[0].blocks[0].embedding.weight
> ```

## 3. Train it

Your custom class is an ordinary `nn.Module`, so it plugs straight into the [`Trainer`](../guides/datasets-and-training.md):

```python
import os, tempfile, urllib.request, torch
from olm.train import Trainer
from olm.data.tokenization import HFTokenizer
from olm.data.datasets import LocalTextDataset, DataLoader

CTX = 256
tok = HFTokenizer("gpt2")
model = MiniGPT(tok.vocab_size, d_model=256, n_heads=8, n_layers=6, max_seq_len=CTX)

data_dir = tempfile.mkdtemp()
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    os.path.join(data_dir, "input.txt"),
)
loader = DataLoader(LocalTextDataset(data_dir, tok, CTX, shuffle=True), batch_size=16)

device = "cuda" if torch.cuda.is_available() else "cpu"
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
Trainer(model, opt, loader, device, context_length=CTX, use_amp=False).train(
    epochs=1, max_steps=300, log_interval=20
)
```

## 4. Experiment with the structure

This is where the design pays off. Swapping a layer's components changes the architecture with no other code changes — the model class, training loop, and data pipeline are untouched. Here are three drop-in replacements for `gpt_layer`.

### Modern stack (Llama-style)

RMSNorm, RoPE attention, and a SwiGLU feed-forward:

```python
from olm.nn.attention import MultiHeadAttentionwithRoPE
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm

def modern_layer(d_model, n_heads, max_seq_len):
    return Block([
        Residual(Block([
            RMSNorm(d_model),
            MultiHeadAttentionwithRoPE(d_model, n_heads, max_seq_len, causal=True),
        ])),
        Residual(Block([RMSNorm(d_model), SwiGLUFFN(d_model)])),
    ])
```

### Grouped-query attention

Fewer key/value heads for inference efficiency:

```python
from olm.nn.attention import GroupedQueryAttention

def gqa_layer(d_model, n_heads, n_kv_heads, max_seq_len):
    return Block([
        Residual(Block([
            RMSNorm(d_model),
            GroupedQueryAttention(d_model, n_heads, n_kv_heads, max_seq_len),
        ])),
        Residual(Block([RMSNorm(d_model), SwiGLUFFN(d_model)])),
    ])
```

### ALiBi (length extrapolation)

A distance bias instead of positional embeddings, for testing on longer sequences than seen in training:

```python
from olm.nn.attention import MultiHeadAttentionwithALiBi

def alibi_layer(d_model, n_heads, max_seq_len):
    return Block([
        Residual(Block([
            LayerNorm(d_model),
            MultiHeadAttentionwithALiBi(d_model, n_heads, max_seq_len=max_seq_len),
        ])),
        Residual(Block([LayerNorm(d_model), SwiGLUFFN(d_model)])),
    ])
```

Because structure and components are decoupled, an ablation that would normally mean forking a model file is a one-line swap.

## Next steps

- Read the reference implementations in [`olm.models`](../api/models.md) — they use exactly these patterns.
- Add a [custom combinator](../guides/architecture.md#defining-a-custom-combinator) to express a structure the built-ins do not cover.
- Scale your model up with [Distributed Training](distributed-training.md).
