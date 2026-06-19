# OpenLanguageModel Docs

OpenLanguageModel (OLM) is a PyTorch-native library for building, training, teaching, and researching transformer language models. The docs are organized around the path most users take: start a run, understand the concepts, change the architecture, then look up exact APIs when needed.

## Start Here

- [`getting-started.md`](getting-started.md): install OLM and run a small local training loop.
- [`installation.md`](installation.md): supported Python versions, extras, dependency ranges, and release-build commands.
- [`datasets-and-training.md`](datasets-and-training.md): use local text, FineWeb-Edu, `Trainer`, `AutoTrainer`, callbacks, checkpointing, and single-node DDP/FSDP.
- [`architecture.md`](architecture.md): understand `Block`, `Residual`, `Repeat`, `Parallel`, and how OLM assembles custom models.
- [`api.md`](api.md): generated API reference with signatures, docstrings, and source-defined methods.

## Common Tasks

| Task | Where to go |
|---|---|
| Install OLM or choose extras | [`installation.md`](installation.md) |
| Train from local `.txt` files | [`getting-started.md`](getting-started.md) |
| Stream FineWeb-Edu | [`datasets-and-training.md`](datasets-and-training.md) |
| Let OLM pick CPU/GPU/single-node DDP/FSDP | [`datasets-and-training.md#6-automatic-trainer-selection-autotrainer`](datasets-and-training.md#6-automatic-trainer-selection-autotrainer) |
| Build a custom architecture | [`architecture.md`](architecture.md) |
| Look up model constructors | [`api.md`](api.md) |
| Inspect runnable scripts | [`../examples`](../examples) |

## Implemented Models

Each model family is implemented as source code, not hidden configuration:

- GPT-2: [`../src/olm/models/openai/gpt2.py`](../src/olm/models/openai/gpt2.py)
- Llama 2: [`../src/olm/models/meta/llama2.py`](../src/olm/models/meta/llama2.py)
- Llama 3 / 3.1 / 3.2: [`../src/olm/models/meta/llama3.py`](../src/olm/models/meta/llama3.py)
- Qwen 2.5: [`../src/olm/models/alibaba/qwen2.py`](../src/olm/models/alibaba/qwen2.py)
- Phi-3 / Phi-3.5: [`../src/olm/models/microsoft/phi3.py`](../src/olm/models/microsoft/phi3.py)
- Phi-4: [`../src/olm/models/microsoft/phi4.py`](../src/olm/models/microsoft/phi4.py)
- Gemma 2: [`../src/olm/models/google/gemma2.py`](../src/olm/models/google/gemma2.py)
- OLMo: [`../src/olm/models/allenai/olmo.py`](../src/olm/models/allenai/olmo.py)
- OPT: [`../src/olm/models/facebook/opt.py`](../src/olm/models/facebook/opt.py)

## Project Direction

v2.2 is focused on stabilization and polish: bug fixes, model-configuration checks, API-reference quality, docs/website integration, SEO foundations, roadmap cleanup, and release readiness. New research features are intentionally out of scope for v2.2.
