# OpenLanguageModel (OLM)

OpenLanguageModel is a PyTorch-native library for building, training, teaching, and researching transformer language models. It is designed for people who want the model architecture to stay visible while the training stack stays manageable.

OLM gives you:

- readable transformer components in `olm.nn`
- implemented model families in `olm.models`
- local and Hugging Face dataset streams in `olm.data`
- single-device, single-node multi-GPU DDP/FSDP, AMP, checkpointing, callbacks, and automatic trainer selection in `olm.train`

[Website](https://openlanguagemodel.github.io/openlanguagemodel/) · [Docs](docs/) · [Install](docs/installation.md) · [Colab Notebooks](docs/colab-notebooks.md) · [API Reference](docs/api.md) · [Examples](examples/) · [Issues](https://github.com/openlanguagemodel/openlanguagemodel/issues)

## Why OLM

Most language-model libraries either hide the architecture behind configuration, or make you rebuild the whole training path from scratch. OLM sits in the middle: every block is an ordinary `torch.nn.Module`, but data loading, optimization, mixed precision, single-node multi-GPU training, checkpointing, and logging are already wired into a clean path.

That makes it useful for:

- students learning how language models are assembled and trained
- researchers running ablations on attention, norms, feed-forward layers, and residual structure
- practitioners who want existing PyTorch workflows without a hidden runtime

## Llama 3 Block In OLM

Model code in OLM is meant to read like the architecture it represents. For example, the Llama 3 block is built from RMSNorm, grouped-query attention, SwiGLU, and explicit residual structure:

```python
from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm


class Llama3Block(Block):
    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        dropout: float,
        rope_theta: float,
    ):
        super().__init__([
            Residual(Block([
                RMSNorm(embed_dim, eps=1e-5),
                GroupedQueryAttention(
                    embed_dim,
                    num_heads,
                    num_kv_heads,
                    max_seq_len,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    use_bias=False,
                ),
            ])),
            Residual(Block([
                RMSNorm(embed_dim, eps=1e-5),
                SwiGLUFFN(
                    embed_dim,
                    hidden_dim=intermediate_size,
                    dropout=dropout,
                    bias=False,
                ),
            ])),
        ])
```

Source: [`src/olm/models/meta/llama3.py`](src/olm/models/meta/llama3.py)

## Train With The Stack Connected

You can keep the model and optimizer as normal PyTorch objects while OLM handles the training loop details:

```python
import torch

from olm.data.datasets import DataLoader, FineWebEduDataset
from olm.data.tokenization import HFTokenizer
from olm.models.openai import GPT2Model
from olm.train import AutoTrainer
from olm.train.optim import AdamW

tokenizer = HFTokenizer("gpt2")
dataset = FineWebEduDataset(tokenizer, context_length=1024, streaming=True)
loader = DataLoader(dataset, batch_size=8, num_workers=4)

model = GPT2Model(
    vocab_size=tokenizer.vocab_size,
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    max_seq_len=1024,
)

trainer = AutoTrainer(
    model,
    AdamW,
    loader,
    device="auto",
    context_length=1024,
    learning_rate=3e-4,
    grad_accum_steps=8,
)
trainer.train(epochs=1, max_steps=1000)
```

`AutoTrainer` chooses between CPU, single-GPU, and single-node multi-GPU DDP/FSDP paths based on the hardware and model. You can still use `Trainer`, `DDPTrainer`, or `FSDPTrainer` directly when you want explicit control.

## Implemented Model Families

OLM includes named presets and configurable base classes for common transformer families:

| Family | Source |
|---|---|
| GPT-2 | [`src/olm/models/openai/gpt2.py`](src/olm/models/openai/gpt2.py) |
| Llama 2 | [`src/olm/models/meta/llama2.py`](src/olm/models/meta/llama2.py) |
| Llama 3 / 3.1 / 3.2 | [`src/olm/models/meta/llama3.py`](src/olm/models/meta/llama3.py) |
| Qwen 2.5 | [`src/olm/models/alibaba/qwen2.py`](src/olm/models/alibaba/qwen2.py) |
| Phi-3 / Phi-3.5 | [`src/olm/models/microsoft/phi3.py`](src/olm/models/microsoft/phi3.py) |
| Phi-4 | [`src/olm/models/microsoft/phi4.py`](src/olm/models/microsoft/phi4.py) |
| Gemma 2 | [`src/olm/models/google/gemma2.py`](src/olm/models/google/gemma2.py) |
| OLMo | [`src/olm/models/allenai/olmo.py`](src/olm/models/allenai/olmo.py) |
| OPT | [`src/olm/models/facebook/opt.py`](src/olm/models/facebook/opt.py) |

See [`docs/api.md`](docs/api.md) for the generated API reference and [`examples/`](examples/) for training scripts.

## Installation

Use Python 3.10, 3.11, or 3.12.

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

Optional extras:

```bash
pip install -e ".[wandb]"  # Weights & Biases logging
pip install -e ".[docs]"   # documentation tooling
```

See [`docs/installation.md`](docs/installation.md) for dependency and release-build details.

## Documentation Flow

- Install from [`docs/installation.md`](docs/installation.md)
- Start with [`docs/getting-started.md`](docs/getting-started.md)
- Review architecture concepts in [`docs/architecture.md`](docs/architecture.md)
- Run guided Colabs from [`docs/colab-notebooks.md`](docs/colab-notebooks.md)
- Train from examples in [`examples/`](examples/)
- Use [`docs/datasets-and-training.md`](docs/datasets-and-training.md) for data, trainer, AutoTrainer, callbacks, and checkpointing
- Use [`docs/api.md`](docs/api.md) when you need exact signatures and source-defined methods
- Read [`docs/release-v2.2.0.md`](docs/release-v2.2.0.md) for the v2.2 release notes

## Project Status

OLM v2.2 is the stabilization and release-readiness pass: tied output embeddings by default, model-family smoke coverage, AutoTrainer, streaming datasets, AMP, checkpointing, single-node DDP/FSDP paths, clearer installation docs, and a stronger generated API reference. Multi-node training remains a v4 roadmap item.

## Citation

```bibtex
@software{openlanguagemodel2026,
  title = {OpenLanguageModel},
  author = {Tavish Mankash and Vardhaman Kalloli and Keshava Prasad},
  year = {2026},
  url = {https://github.com/openlanguagemodel/openlanguagemodel}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
