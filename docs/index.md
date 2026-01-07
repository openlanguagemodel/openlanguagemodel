# Home

<div class="olm-hero">

<div class="olm-title">OpenLanguageModel (OLM)</div>

<div class="olm-subtitle">
Modular, transparent framework for building, training, and experimenting with transformer-based language models
</div>

<div class="olm-badge">Open-source • PyTorch-first • Structure-first modeling</div>

<!-- <div class="olm-authors">
OpenLanguageModel Team
</div> -->

<!-- <div class="olm-affil">
openlanguagemodel/openlanguagemodel
</div> -->

<div class="olm-links">
  <a class="olm-btn" href="https://github.com/openlanguagemodel/openlanguagemodel">
    <span class="olm-icon"><svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg></span> Code
  </a>
  <a class="olm-btn" href="https://github.com/openlanguagemodel/openlanguagemodel/issues">
    <span class="olm-icon"><svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zM4 8a.5.5 0 011 0v3a.5.5 0 01-1 0V8zm4 0a.5.5 0 011 0v3a.5.5 0 01-1 0V8zm4 0a.5.5 0 011 0v3a.5.5 0 01-1 0V8zM8 5.5a.5.5 0 110-1 .5.5 0 010 1z"/></svg></span> Issues
  </a>
  <a class="olm-btn" href="https://github.com/openlanguagemodel/openlanguagemodel#installation">
    <span class="olm-icon"><svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path d="M8.5 1.5a.5.5 0 00-1 0v6.793L5.354 6.146a.5.5 0 10-.708.708l3 3a.5.5 0 00.708 0l3-3a.5.5 0 00-.708-.708L8.5 8.293V1.5z"/><path d="M3 10a1 1 0 011-1h8a1 1 0 011 1v3a1 1 0 01-1 1H4a1 1 0 01-1-1v-3zm1 0v3h8v-3H4z"/></svg></span> Install
  </a>
  <a class="olm-btn" href="https://github.com/openlanguagemodel/openlanguagemodel/tree/main/docs">
    <span class="olm-icon"><svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path d="M2 2a2 2 0 012-2h8a2 2 0 012 2v12a2 2 0 01-2 2H4a2 2 0 01-2-2V2zm2-1a1 1 0 00-1 1v12a1 1 0 001 1h8a1 1 0 001-1V2a1 1 0 00-1-1H4z"/><path d="M6 5h4v1H6V5zm0 2h4v1H6V7zm0 2h4v1H6V9z"/></svg></span> Docs
  </a>
</div>

<div class="olm-lead">
OLM is designed to make <strong style="color: #249def;">sandboxing ideas and prototyping new architectures easy</strong>, while still exposing the full complexity required for serious research and large-scale training.
</div>

</div>

<hr/>

## Abstract

OpenLanguageModel (OLM) is a modular, transparent framework for building, training, and experimenting with transformer-based language models.

OLM deliberately avoids black-box abstractions: every major component is explicit, inspectable, and replaceable. You can start training quickly, then progressively peel back layers as you explore, modify, or reimplement parts of the system.

<hr/>

## Why OLM?

Most ML systems force a trade-off:

-   High-level frameworks → easy to use, hard to extend
-   Low-level code → flexible, slow to iterate

**OLM sits in the middle.** You can:

-   Get a model training with minimal setup
-   Swap architectural components without rewriting everything
-   Introduce new wiring patterns / structures
-   Drop down to raw PyTorch whenever needed

<hr/>

## Minimal Training Example

A simple example of training a simple language model on the [TinyShakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) dataset locally.

![Minimal training example visualization](https://raw.githubusercontent.com/openlanguagemodel/openlanguagemodel/dev/image.png)

```python
import sys, os, torch, urllib.request
from torch.utils.data import DataLoader
from tempfile import TemporaryDirectory

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from olm.data.datasets import Dataset
from olm.data.tokenization.hf_tokenizer import HFTokenizer
from olm.train.trainer import Trainer
from olm.nn.blocks import LM

with TemporaryDirectory() as tmp:
  urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    os.path.join(tmp, "i.txt")
  )

  tokenizer, device = HFTokenizer("gpt2"), "cuda" if torch.cuda.is_available() else "cpu"
  model = LM(tokenizer.vocab_size, 64, 4, 2, 33)
  optimizer = torch.optim.AdamW(model.parameters(), 3e-4)
  dataset = Dataset(tmp, tokenizer, 32)
  dataloader = DataLoader(dataset, 4)
  trainer = Trainer(model, optimizer, dataloader, device, 32, use_amp=False)
  losses = trainer.train(1, 10, 100)
  print(f"S:{losses[0]:.4f} E:{losses[-1]:.4f} OK:{losses[-1]<losses[0]}")
```

<hr/>

## Design Philosophy

-   **Accessible by default** – training and experimentation should be easy to start
-   **Transparent by construction** – no implicit behavior, no magic helpers
-   **Structure as a first-class concept** – composition matters as much as blocks

Rather than hiding complexity, OLM **organizes it** into clear, navigable layers.

<hr/>

## Repository Structure

```text
openlanguagemodel/
├── configs/            # YAML experiment configurations
├── docs/               # Design notes and guides
├── examples/           # End-to-end training examples
├── src/olm/            # Core library code
│ ├── data/             # Datasets, tokenization, loaders
│ ├── models/           # High-level model definitions
│ ├── nn/               # Neural building blocks and structure
│ ├── train/            # Training loop and orchestration
│ └── utils/            # Shared helpers
├── tests/
└── verify_imports.py
```

<hr/>

## Installation

```bash
git clone https://github.com/openlanguagemodel/openlanguagemodel.git
cd openlanguagemodel
pip install -e .
```

An editable install is recommended so you can inspect, modify, and extend components easily.

<hr/>

## PyTorch as the Foundation

OLM is built directly on top of **PyTorch**.

-   All models are standard `torch.nn.Module`s
-   Autograd, optimizers, and AMP come directly from torch
-   No custom execution engines or hidden graph layers

This means you can drop into raw PyTorch at any moment, and the code will accept that change readily. Debugging, error handling, and pipeline management behave exactly as expected. Knowledge of PyTorch is encouraged although not completely necessary.

**OLM extends PyTorch — it does not replace it.**

<hr/>

## Configuration & Experiment Setup

Models in OLM can be described using simple YAML configuration files:

```yaml
model:
    name: gpt
    vocab_size: 50257
    n_layers: 12
    n_heads: 12
    d_model: 768

training:
    batch_size: 64
    max_steps: 100000
```

Configurations describe **what** to run, not **how** it runs. All execution logic lives in Python and is fully editable. This separation keeps experiments reproducible without turning configuration files into code.

<hr/>

## Core Architecture: `olm.nn`

At the heart of OLM is the `olm.nn` package. This is where **all neural logic lives**. Conceptually, everything in OLM resolves to components defined here.

```
olm.nn/
├── attention/      # Multi-head attention, masking, projections
├── activations/    # GELU, SwiGLU, custom activations
├── norms/          # LayerNorm and variants
├── embeddings/     # Token and positional embeddings
├── blocks/         # Frequently used transformer blocks
├── feedforward/    # Feedforward layers
├── moe/            # Mixture of experts
└── structure/      # Residuals, combinators, block wiring
```

Each component is:

-   A plain `torch.nn.Module`
-   Independently testable
-   Safe to extend, replace, or rewrite

You can use these building blocks directly, subclass them, or bypass them entirely.

<hr/>

## Structural Composition: `olm.nn.structure`

A distinguishing feature of OLM is its explicit treatment of **structure**.

Instead of hard-coding how layers are connected, OLM separates _what a block does_ from _how blocks are wired together_.

The `olm.nn.structure` module provides:

-   Residual combinators
-   Block wrappers
-   Explicit composition utilities

This makes it easy to:

-   Experiment with alternative residual paths
-   Implement pre-norm, post-norm, or custom normalization schemes
-   Build non-standard transformer variants
-   Reuse the same core layers across multiple architectures

Custom structures are not special cases — they are first-class citizens. Entirely new wiring patterns can be implemented without modifying existing layers.

<hr/>

## Models: `olm.models`

Models in OLM are intentionally lightweight. They:

-   Assemble components from `olm.nn`
-   Define forward passes clearly
-   Contain no training or optimization logic

This separation allows you to:

-   Reuse the same architecture across different training setups
-   Modify internal blocks without touching the trainer
-   Prototype new architectures quickly

<hr/>

## Data Pipeline: `olm.data`

The `olm.data` module handles everything related to **input text and batching**, while remaining flexible enough for different research workflows.

It provides:

-   Dataset abstractions
-   Tokenization hooks
-   Iterable and streaming datasets
-   Collation utilities for language modeling

<hr/>

## Training Setup: `olm.train`

OLM is designed so that **setting up training is simple**, even though nothing is hidden.

A typical training setup involves:

```python
model = build_model(cfg)
dataloader = build_dataloader(cfg)
trainer = Trainer(model, dataloader, ...)
trainer.train()
```

The trainer exists to connect components, not to dictate behavior. If you want to modify the training loop — logging, accumulation, precision, or checkpointing — you can do so directly.

<hr/>

## Who OLM Is For

OLM works well for:

-   Students learning how transformers are built
-   Researchers prototyping new architectures
-   Engineers who want control without unnecessary boilerplate
