# Build a GPT-2 Style Language Model from Scratch in PyTorch

If you want to learn language modelling, the best path is still to train a
small model yourself. OpenLanguageModel (OLM) is designed for that path: you get
plain PyTorch modules, readable transformer blocks, and a trainer that keeps the
loop understandable.

The point is not to hide GPT-style models behind a large framework. The point is
to make each piece visible: token IDs, embeddings, attention, feed-forward
layers, residual connections, loss, optimizer, and sampling.

## The Short Version

```python
import torch

from olm.nn.blocks import LM
from olm.train import Trainer
from olm.data.tokenization import HFTokenizer
from olm.data.datasets import FineWebEduDataset, DataLoader

tok = HFTokenizer("gpt2")
model = LM(
    tok.vocab_size,
    embed_dim=640,
    num_heads=10,
    num_layers=12,
    max_seq_len=1024,
    ff_multiplier=2.75,
)

dataset = FineWebEduDataset(tok, context_length=1024)
loader = DataLoader(dataset, batch_size=8, num_workers=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"

Trainer(
    model,
    optimizer,
    loader,
    device,
    context_length=1024,
    use_amp=device == "cuda",
).train(epochs=1, max_steps=20_000)
```

This is a roughly GPT-2-small-scale training shape, but the code is still short
enough for a student to read line by line. The model is an ordinary
`torch.nn.Module`, so you can inspect it, replace layers, or move it into your
own loop.

## What OLM Gives You

OLM handles the repetitive parts of a language-model run: streaming text,
tokenization, batches of input/target token IDs, mixed precision, gradient
accumulation, schedules, callbacks, metrics, and checkpointing.

The model definition remains visible. If you want to understand the architecture
instead of only calling `from_pretrained`, start with the
[Your First Language Model](/docs/tutorials/first-model/) tutorial and then read
the [Block System guide](/docs/guides/architecture/).

## Why This Matters

Training from scratch teaches details that inference-only libraries hide. You
see why context length matters, why attention needs a causal mask, why the loss
is next-token cross entropy, and how sampling changes when the model improves.

That is why OLM treats GPT-style training as both a practical workflow and a
learning tool. The first run should be small enough to finish. The code should
still look like the architecture.
