# Your First Language Model

In this tutorial you will build a small GPT-style language model, train it on Shakespeare, and generate text from it — in a single script. By the end you will understand the four parts of every OLM project: tokenizer, model, data, and trainer.

This tutorial runs on CPU in a few minutes, and faster on a GPU. It assumes you have [installed OLM](../getting-started.md#installation).

> [!NOTE]
> New to terms like *token*, *RoPE*, *logits*, or *perplexity*? You do not need
> them all to follow along — but [Key Concepts](../concepts.md) explains every one
> in plain English. Keep it open in a tab.

## 1. The tokenizer

A tokenizer converts text to integer ids. We reuse GPT-2's BPE tokenizer through [`HFTokenizer`](../api/data.md#hftokenizer), which wraps any Hugging Face tokenizer:

```python
from olm.data.tokenization import HFTokenizer

tok = HFTokenizer("gpt2")
print(tok.vocab_size)          # 50257
print(tok.encode("Hello!"))    # tensor([15496,    0])
```

## 2. The model

[`LM`](../api/nn.md#lm) is a ready-made GPT-style architecture: a token embedding, a stack of pre-norm transformer blocks (with [RoPE and SwiGLU](../concepts.md#positional-information-rope-alibi-and-friends)), and an output head. We keep it small.

```python
from olm.nn.blocks import LM

CTX = 128   # context length

model = LM(
    vocab_size=tok.vocab_size,
    embed_dim=192,
    num_heads=6,
    num_layers=6,
    max_seq_len=CTX,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params/1e6:.1f}M parameters")
```

> [!TIP]
> Keep `max_seq_len` greater than or equal to your context length. `LM` precomputes Rotary Positional Embeddings up to `max_seq_len`; sequences longer than that raise an error, so size it to your longest expected sequence.

## 3. The data

We stream from a folder of `.txt` files with [`LocalTextDataset`](../api/data.md#localtextdataset), downloading Tiny Shakespeare into a temporary folder first.

```python
import os, tempfile, urllib.request
from olm.data.datasets import LocalTextDataset, DataLoader

data_dir = tempfile.mkdtemp()
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    os.path.join(data_dir, "input.txt"),
)

dataset = LocalTextDataset(data_dir, tok, context_length=CTX, shuffle=True)
loader = DataLoader(dataset, batch_size=16)
```

The dataset yields `(input_ids, labels)` pairs where `labels` is `input_ids` shifted by one position — the supervision signal for next-token prediction.

## 4. Training

The [`Trainer`](../api/train.md#trainer) connects the pieces. We disable AMP so the example runs on CPU.

```python
import torch
from olm.train import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

trainer = Trainer(model, optimizer, loader, device, context_length=CTX, use_amp=False)
trainer.train(epochs=1, max_steps=500, log_interval=20)
```

The loss and perplexity fall as training proceeds:

```text
Epoch  |   Step   |    Loss    | Perplexity  |  Tokens/s  |     LR
--------------------------------------------------------------------------------
  1    |    20    |   6.8521   |    946.10   |    18233   |  3.00e-04
  1    |   500    |   4.1207   |     61.66   |    19011   |  2.78e-04
```

## 5. Generating text

Let us sample from the model. OLM includes a small `generate` helper for ordinary autoregressive sampling: it encodes a prompt, repeatedly feeds the model, applies temperature/top-k sampling, and crops the context window so RoPE never overflows.

```python
from olm.nn import generate

print(generate(model, tok, "ROMEO:", context_length=CTX, device=device))
```

After only 500 steps on a tiny model the output will not be Shakespeare, but it will have learned the *shape* of the text — character names, line breaks, and dialogue rhythm:

```text
ROMEO:
And the see the sward and the heart speak the heart,
That the hath the manst the sentle the heart...
```

Train longer, enlarge the model, or feed it more data and the samples sharpen quickly.

## 6. Saving your model

Persist the model and tokenizer so you can reload them later without redefining the architecture:

```python
model.save("./shakespeare-mini", tokenizer=tok)

# later, in a fresh process:
from olm.nn.structure import load_model
model, tok = load_model("./shakespeare-mini")
```

## The complete script

<details>
<summary>first_model.py</summary>

```python
import os, tempfile, urllib.request
import torch

from olm.nn.blocks import LM
from olm.nn import generate
from olm.train import Trainer
from olm.data.tokenization import HFTokenizer
from olm.data.datasets import LocalTextDataset, DataLoader

CTX = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer + model
tok = HFTokenizer("gpt2")
model = LM(tok.vocab_size, embed_dim=192, num_heads=6, num_layers=6, max_seq_len=CTX)

# Data
data_dir = tempfile.mkdtemp()
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    os.path.join(data_dir, "input.txt"),
)
loader = DataLoader(LocalTextDataset(data_dir, tok, CTX, shuffle=True), batch_size=16)

# Train
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
Trainer(model, opt, loader, device, context_length=CTX, use_amp=False).train(
    epochs=1, max_steps=500, log_interval=20
)

# Generate
print(generate(model, tok, "ROMEO:", context_length=CTX, device=device))
```

</details>

## Next steps

- [Custom Architectures](custom-architecture.md) — replace the prebuilt `LM` with a model you design from blocks.
- [Datasets & Training](../guides/datasets-and-training.md) — callbacks, schedulers, checkpoints, and scaling up.
