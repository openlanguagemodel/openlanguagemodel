---
description: Seed OLM runs, make them deterministic, and understand why a fresh model starts at a loss of about ln(vocab_size).
---
# Reproducibility

Two runs of the same script should produce the same model. That makes experiments comparable, lets you tell a real improvement from noise, and makes a bug you saw once show up again. This page covers seeding, deterministic mode, what still varies, and how to read the very first loss of a run.

## Seed everything

Every random number in a training run comes from one of four generators: Python's `random`, NumPy's global generator, the torch CPU generator, and one torch generator per GPU. `seed_everything` seeds all of them:

```python
from olm.core import seed_everything
from olm.nn.blocks import LM

seed_everything(42)  # before the model is built
model = LM(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=32)
```

Call it **before you build the model**. Weight initialization draws from the torch generator, so seeding afterwards makes dropout and shuffling repeatable but not the starting weights.

Seeding is enough to repeat a run exactly on CPU. This trains the same model twice:

```python
import torch
from torch.utils.data import TensorDataset

from olm.data.datasets import DataLoader
from olm.train import AdamW, Trainer


def train_once(seed):
    seed_everything(seed)
    model = LM(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=32, dropout=0.1)
    tokens = torch.randint(0, 256, (32, 33))
    loader = DataLoader(TensorDataset(tokens[:, :-1], tokens[:, 1:]), batch_size=8, pin_memory=False)
    trainer = Trainer(model, AdamW, loader, device="cpu", context_length=32, use_amp=False)
    return trainer.train(epochs=2, log_interval=100)


print(train_once(0) == train_once(0))  # True
print(train_once(0) == train_once(1))  # False
```

The random data, the initial weights, dropout and the shuffle order all come from the seeded generators, so the losses match exactly.

## Seed DataLoader shuffling and workers

When an epoch starts, a shuffling `DataLoader` draws the seed of its permutation from the global torch generator. With `num_workers > 0` it also draws a base seed, from which PyTorch seeds each worker's torch, Python and NumPy generators (once, when persistent workers start). After `seed_everything` all of that is repeatable, but it shifts whenever the rest of your program draws a different number of random values first, for example after a change to the model changes how many numbers its initialization draws. `make_generator` gives the loader a stream of its own, so the data order stays fixed across such changes:

```python
from olm.core import make_generator, seed_worker

loader = DataLoader(
    TensorDataset(torch.randint(0, 256, (32, 33))),
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    generator=make_generator(42),
    worker_init_fn=seed_worker,
)
```

`seed_worker` is optional. It is the recipe from PyTorch's reproducibility notes: it reseeds each worker's Python and NumPy generators from the worker's torch seed, so all three follow from one number. OLM's `DataLoader` passes both arguments through to PyTorch's, and checkpoints save and restore the loader's generator.

## Deterministic mode

Seeding fixes the random numbers; it does not make every GPU kernel produce the same bits every time. Some CUDA kernels accumulate floating-point sums in whatever order threads finish, so two runs from the same seed can drift apart by rounding errors that grow over training. `deterministic=True` asks PyTorch for deterministic kernels:

```python
seed_everything(42, deterministic=True)
```

It calls `torch.use_deterministic_algorithms(True, warn_only=True)`, sets `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`, and sets the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8` (unless you set it already), which cuBLAS needs for deterministic matrix multiplies.

- Call it at the start of your script. cuBLAS reads `CUBLAS_WORKSPACE_CONFIG` when CUDA first multiplies matrices, so setting it later has no effect.
- Deterministic kernels can be slower. Use the mode to debug and to compare runs, not necessarily for production training.
- With `warn_only=True` (the default), an operation that has no deterministic implementation runs anyway and PyTorch warns. Pass `warn_only=False` to make it raise instead, which tells you exactly which operation it is. PyTorch's documentation of [`torch.use_deterministic_algorithms`](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html) lists those operations.
- These are process-wide PyTorch settings. `deterministic=False` (the default) leaves them as they are; it does not turn a previous `True` off.

Even in deterministic mode, results can differ between GPU models, driver and library versions, and numbers of GPUs.

## Resuming is part of reproducibility

A run that is interrupted and resumed from a checkpoint should be the same run. OLM's checkpoints store every random generator's state, for every rank, and the position in the data, so a resumed run continues with the same batches, dropout masks and learning rate. On CPU the resumed losses equal the uninterrupted ones exactly. See [Checkpointing & Resuming](checkpointing.md) for what is saved and when a resume is exact.

## Why a fresh model starts at a loss of about ln(vocab_size)

The first loss a run prints tells you whether the model and data are wired up correctly, before any learning has happened.

A model that knows nothing should spread its probability evenly over the `V` tokens of the vocabulary, giving each one probability `1/V`. Cross-entropy is the negative log-probability of the correct next token, so its value is

```text
loss = -ln(1/V) = ln V
```

For `V = 50257` (GPT-2's tokenizer) that is 10.82; for `V = 1000` it is 6.91.

A freshly initialized network is not exactly uniform: its logits `z_1 ... z_V` are small random numbers. Cross-entropy is `logsumexp(z) - z_label`. If the logits are roughly independent with mean 0 and standard deviation `s`, then `logsumexp(z) = ln V + ln(mean of e^z_i)`, and the mean of `e^z` over many Gaussian logits is `e^(s²/2)`. The correct token's logit is 0 on average, so

```text
expected first loss ≈ ln V + s²/2
```

A logit spread of 0.16 adds 0.01; a spread of 0.58 adds 0.17. So **a first loss slightly above ln V is healthy**.

Initialization can break this. OLM's `Embedding` keeps PyTorch's `nn.Embedding` default, a standard normal N(0, 1), and most OLM models reuse the embedding matrix as the output projection (a *tied* head). The last hidden state of token `x` is then close to its own embedding row, normalized, so the logit of `x` itself is roughly the squared length of a random N(0, 1) vector of size `embed_dim`: about `embed_dim`. That one logit dominates the softmax. The model confidently predicts that the next token repeats the current one, and the loss is about `embed_dim` instead of ln V.

This measures it for `V = 1000` and `embed_dim = 64`, on CPU:

```python
import math

import torch.nn.functional as F

vocab_size, embed_dim = 1000, 64
seed_everything(0)
input_ids = torch.randint(0, vocab_size, (8, 128))
labels = torch.randint(0, vocab_size, (8, 128))


def first_loss(embedding_std=None, tie_embeddings=True, targets=labels):
    seed_everything(1)
    model = LM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        max_seq_len=128,
        tie_embeddings=tie_embeddings,
    )
    with torch.no_grad():
        if embedding_std is not None:
            model.blocks[0].embedding.weight.normal_(0.0, embedding_std)
        logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    return loss.item(), logits.std().item()


print(f"ln(vocab_size)                 {math.log(vocab_size):6.2f}")
for name, kwargs in [
    ("tied head, embeddings N(0, 1)", {"embedding_std": 1.0}),
    ("tied head, embeddings std 0.02", {"embedding_std": 0.02}),
    ("untied head", {"tie_embeddings": False}),
]:
    loss, logit_std = first_loss(**kwargs)
    print(f"{name:30} {loss:6.2f}   logit std {logit_std:.2f}")
print(f"{'tied, N(0, 1), labels = inputs':30} {first_loss(1.0, targets=input_ids)[0]:6.2f}")
```

```text
ln(vocab_size)                   6.91
tied head, embeddings N(0, 1)   61.61   logit std 8.27
tied head, embeddings std 0.02   6.91   logit std 0.16
untied head                      7.05   logit std 0.58
tied, N(0, 1), labels = inputs   0.00
```

- With N(0, 1) embeddings and a tied head, the first loss is about `embed_dim` (it is about 250 for `embed_dim = 256`).
- With embeddings drawn at standard deviation 0.02, the value that the GPT-2, Llama and Qwen reference configurations use, the first loss is ln V.
- An untied head does not have the problem: `OutputHead` normalizes the hidden state, and PyTorch's default `nn.Linear` initialization gives logits with a spread of about 1/√3 ≈ 0.58, so the loss is about ln V + 1/6.
- The last line is a diagnostic: when labels equal the inputs (a dataset that forgot to shift its labels by one position), this model scores almost 0 before training.

The model does recover from a large first loss, but it spends its first steps undoing the initialization. That matters for short runs and for comparisons: [issue #85](https://github.com/openlanguagemodel/openlanguagemodel/issues/85) shows that the ranking printed by the architecture-ablation notebook comes from how each block amplifies unit-scale embeddings, and flips with the seed once embeddings start at 0.02. [PR #84](https://github.com/openlanguagemodel/openlanguagemodel/pull/84) proposes making 0.02 the default for `Embedding`. Until your OLM version includes that change, re-initialize the embedding as in the snippet if you need a first loss of ln V.

How to read your first logged loss:

| First loss | Likely meaning |
|---|---|
| a little above ln V | healthy |
| far above ln V, near `embed_dim` | unit-scale embeddings with a tied head (see above) |
| far below ln V | the labels leak the answer, for example labels equal to the inputs |

> [!NOTE]
> The Trainer prints perplexity as `exp(min(loss, 20))`, so every loss above 20 shows the same perplexity, 485165195.41.

## See also

- [Checkpointing & Resuming](checkpointing.md)
- [Defaults Reference](../reference/defaults.md): every default, including the ones no signature shows.
