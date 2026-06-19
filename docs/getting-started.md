# Getting Started

This page is the shortest path from a fresh clone to a model that trains. OLM keeps the model as normal PyTorch while giving you ready-made pieces for tokenization, text streaming, training loops, AMP, checkpointing, DDP, FSDP, and automatic trainer selection.

## Install

```bash
git clone https://github.com/openlanguagemodel/openlanguagemodel.git
cd openlanguagemodel
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest tests
```

## Train On Local Text

Create a directory with one or more `.txt` files, then run:

```python
import torch

from olm.data.datasets import DataLoader, LocalTextDataset
from olm.data.tokenization import HFTokenizer
from olm.nn.blocks import LM
from olm.train import Trainer
from olm.train.optim import AdamW

tokenizer = HFTokenizer("gpt2")
context_length = 128

dataset = LocalTextDataset(
    "./data",
    tokenizer,
    context_length=context_length,
    shuffle=True,
)
loader = DataLoader(dataset, batch_size=8, num_workers=0, pin_memory=False)

model = LM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    max_seq_len=context_length,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = Trainer(
    model,
    AdamW,
    loader,
    device=device,
    context_length=context_length,
    learning_rate=3e-4,
    use_amp=device.startswith("cuda"),
)

losses = trainer.train(epochs=1, max_steps=100)
```

`LM` ties the output projection to the input token embedding by default. Pass
`tie_embeddings=False` when you want a separate output head matrix.

## Train From FineWeb-Edu

For real language-model data, use the built-in FineWeb-Edu wrapper:

```python
from olm.data.datasets import DataLoader, FineWebEduDataset
from olm.data.tokenization import HFTokenizer

tokenizer = HFTokenizer("gpt2")
dataset = FineWebEduDataset(
    tokenizer=tokenizer,
    subset="sample-10BT",
    context_length=1024,
    streaming=True,
    shuffle=True,
)
loader = DataLoader(dataset, batch_size=8, num_workers=4)
```

## Let OLM Pick The Trainer

`AutoTrainer` inspects the available hardware and chooses a single-device, DDP, or FSDP trainer path:

```python
from olm.train import AutoTrainer
from olm.train.optim import AdamW

trainer = AutoTrainer(
    model,
    AdamW,
    loader,
    device="auto",
    context_length=1024,
    learning_rate=3e-4,
    grad_accum_steps=8,
)
trainer.train(epochs=1)
```

Use `Trainer`, `DDPTrainer`, or `FSDPTrainer` directly when you want explicit control.

## Next Steps

- Read [`datasets-and-training.md`](datasets-and-training.md) for data streaming, callbacks, checkpointing, DDP, FSDP, and AutoTrainer.
- Read [`architecture.md`](architecture.md) for blocks, residuals, and custom architectures.
- Use [`api.md`](api.md) for exact signatures and method docs.
- Explore runnable scripts in [`../examples`](../examples).
