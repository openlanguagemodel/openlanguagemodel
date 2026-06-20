# Installation

OLM is a Python package built on PyTorch. For v2.2, use Python 3.10, 3.11, or 3.12.

## From Source

```bash
git clone https://github.com/openlanguagemodel/openlanguagemodel.git
cd openlanguagemodel
pip install -e .
```

## Development Install

```bash
pip install -e ".[dev]"
pytest tests
```

The development extra installs the test and documentation tools used by the repository.

## Optional Extras

```bash
pip install -e ".[wandb]"
pip install -e ".[docs]"
```

- `wandb` installs Weights & Biases logging support.
- `docs` installs the Sphinx dependencies used by the generated API reference.

## Core Dependencies

OLM v2.2 requires:

- `torch>=2.1.0`
- `transformers>=4.57.1`
- `datasets>=2.0.0`
- `numpy>=1.20.0`
- `tqdm>=4.60.0`
- `pyyaml>=6.0`

## Verify The Install

```bash
python - <<'PY'
import olm
from olm.nn.blocks import LM

print("olm", olm.__version__)
model = LM(vocab_size=128, embed_dim=32, num_heads=4, num_layers=1, max_seq_len=16)
print(type(model).__name__)
PY
```

## Building Distributions

Release builds use standard Python packaging:

```bash
pip install -e ".[dev]"
python -m build
twine check dist/*
```

Publish only after tests pass and the GitHub release notes are ready.
