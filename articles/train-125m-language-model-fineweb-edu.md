# How to Train a 125M Language Model on FineWeb-Edu

FineWeb-Edu is a useful dataset for learning real language-model training
without immediately jumping to billion-parameter infrastructure. OLM can stream
FineWeb-Edu, tokenize with GPT-2's tokenizer, and train a roughly 125M parameter
model with readable PyTorch code.

This page is a practical training report template: the exact loss curve depends
on hardware, batch size, token budget, and optimizer settings, but the model
shape and code are intended to be reproducible.

## Model Shape

```python
model = LM(
    tok.vocab_size,
    embed_dim=640,
    num_heads=10,
    num_layers=12,
    max_seq_len=1024,
    ff_multiplier=2.75,
)
```

With GPT-2 vocabulary size, OLM's current untied output head, and this hidden
size/depth, the model is about 125M parameters. That puts it near the scale
where training is meaningful but still understandable.

## Training Code

```python
import torch

from olm.nn.blocks import LM
from olm.train import Trainer
from olm.data.tokenization import HFTokenizer
from olm.data.datasets import FineWebEduDataset, DataLoader

tok = HFTokenizer("gpt2")
dataset = FineWebEduDataset(tok, context_length=1024)
loader = DataLoader(dataset, batch_size=8, num_workers=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"

losses = Trainer(
    model,
    optimizer,
    loader,
    device,
    context_length=1024,
    use_amp=device == "cuda",
).train(epochs=1, max_steps=20_000)
```

For lower-cost cloud GPUs, a two-hour H100 run can land around a small-dollar
experiment rather than a lab-scale training bill. Treat the exact price as a
provider-dependent estimate, not a fixed promise.

## What To Record

For a useful report, log the token budget, loss curve, GPU type, wall-clock
time, batch size, gradient accumulation, optimizer, and any checkpoint used for
sampling. The [Datasets & Training](/docs/guides/datasets-and-training/) guide
covers the trainer features behind this run: AMP, schedules, callbacks,
checkpointing, DDP, and FSDP.

## Why FineWeb-Edu

Tiny datasets are good for smoke tests. FineWeb-Edu is better for showing
language-model behavior because the model sees diverse, cleaner web text. That
makes it a better benchmark page for students, instructors, and researchers who
want to see whether OLM can run a real training path.
