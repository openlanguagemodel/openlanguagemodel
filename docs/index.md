# OpenLanguageModel Docs

OpenLanguageModel (OLM) is a PyTorch-native library for building, training, teaching, and researching transformer language models. The docs are organized around the path most users take: start a run, understand the concepts, change the architecture, then look up exact APIs when needed.

## Start Here

- [`getting-started.md`](getting-started.md): install OLM and run a small local training loop.
- [`installation.md`](installation.md): supported Python versions, extras, dependency ranges, and release-build commands.
- [`release-v2.2.0.md`](release-v2.2.0.md): public release notes for the v2.2 stabilization release.
- [`releasing.md`](releasing.md): maintainer checklist for GitHub and PyPI releases.
- [`datasets-and-training.md`](datasets-and-training.md): use local text, FineWeb-Edu, `Trainer`, `AutoTrainer`, callbacks, checkpointing, and single-node DDP/FSDP.
- [`architecture.md`](architecture.md): understand `Block`, `Residual`, `Repeat`, `Parallel`, and how OLM assembles custom models.
- [`colab-notebooks.md`](colab-notebooks.md): runnable Colab notebooks for the main learning path.
- [`api.md`](api.md): generated API reference with signatures, docstrings, and source-defined methods.

## Common Tasks

| Task | Where to go |
|---|---|
| Install OLM or choose extras | [`installation.md`](installation.md) |
| Publish a release | [`releasing.md`](releasing.md) |
| Train from local `.txt` files | [`getting-started.md`](getting-started.md) |
| Stream FineWeb-Edu | [`datasets-and-training.md`](datasets-and-training.md) |
| Let OLM pick CPU/GPU/single-node DDP/FSDP | [`datasets-and-training.md#6-automatic-trainer-selection-autotrainer`](datasets-and-training.md#6-automatic-trainer-selection-autotrainer) |
| Build a custom architecture | [`architecture.md`](architecture.md) |
| Run the guided Colabs | [`colab-notebooks.md`](colab-notebooks.md) |
| Find a model family or preset | [`models/index.md`](models/index.md) |
| Look up model constructors | [`api.md`](api.md) |
| Inspect runnable scripts | [`../examples`](../examples) |

## Implemented Models

OLM implements model families from GPT-2 to recent frontier models such as DeepSeek-V3, Kimi K2 and Qwen3.5, each as a readable source file that assembles the architecture from public OLM blocks. Presets build a released model's architecture with randomly initialised weights; OLM does not load pretrained checkpoints.

- [`models/index.md`](models/index.md): the Model Zoo, generated from the code. It lists every family and preset with parameter counts measured on the `meta` device, layer counts, context lengths and the attention and feed-forward layers each preset builds.
- Annotated architecture guides that walk through the source section by section: [GPT-2](models/gpt-2.md), [Llama 3](models/llama-3.md), [Qwen3](models/qwen-3.md), [Gemma 3](models/gemma-3.md), [DeepSeek-V3](models/deepseek-v3.md), and [Kimi K2 and Kimi Linear](models/kimi.md).

## Project Direction

v2.2 is focused on stabilization and polish: bug fixes, model-configuration checks, API-reference quality, docs/website integration, SEO foundations, roadmap cleanup, and release readiness. See [`release-v2.2.0.md`](release-v2.2.0.md) for the public release notes.
