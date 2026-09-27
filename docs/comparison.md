# OLM vs Other Libraries

This page compares OLM with the PyTorch projects people most often weigh it against: nanoGPT and its successor nanochat, LitGPT, torchtune, and Hugging Face `transformers` with its `Trainer`. It tries to be specific and fair, and it says plainly where the other projects are the better choice.

> [!NOTE]
> The OLM maintainers wrote this page. Facts about other projects come from their own READMEs, docs and package indexes, listed under [Sources](#sources), and were checked on **27 September 2026**. These projects move quickly. If a row is out of date, please [open an issue](https://github.com/openlanguagemodel/openlanguagemodel/issues).

## The short version

- **OLM** is for building and changing language-model *architectures* from readable, shared parts, and for pretraining them from scratch on one machine.
- **nanoGPT** is the smallest readable GPT-2 training script. Its README now marks it as deprecated and points to **nanochat**, a minimal end-to-end pipeline (tokenizer, pretraining, SFT, RL, evaluation, chat) for one GPT model on one GPU node.
- **LitGPT** gives you 20+ pretrained open models with recipes to pretrain, fine-tune, evaluate and serve them at scale.
- **torchtune** was PyTorch's native fine-tuning and post-training library. Active development stopped in July 2025.
- **`transformers` + `Trainer`** is the ecosystem default for using and fine-tuning pretrained models, with over a million checkpoints on the Hugging Face Hub.

If you need a model that already knows language, OLM is the wrong starting point today; see [What OLM Is (and Isn't)](why-olm.md).

## At a glance

### Scope and models

| Library | Main job | How models are written | Models | Pretrained weights |
|---|---|---|---|---|
| **OLM** | Build, ablate and pretrain LM architectures; teach how they work | One file per family, composed from shared `olm.nn` components with `Block`, `Residual` and `Repeat` | 27 decoder-only text model classes, 47 named presets, from GPT-2 to DeepSeek-V3, Kimi K2 and Qwen3.5 | None. Presets are randomly initialized |
| **nanoGPT** | Smallest readable GPT-2 training code | One `model.py` of about 300 lines (its README's figure) | GPT-2 | Loads OpenAI's four GPT-2 checkpoints |
| **nanochat** | Minimal full pipeline for a small chat model | One GPT model sized by a single `--depth` setting | One architecture | None to import. You train your own model |
| **LitGPT** | Pretrain, fine-tune, evaluate and deploy open LLMs | One config-driven `GPT` class in `litgpt/model.py` serves every family | 20+ families, including Llama 3.x, Gemma 3, Qwen3 and Qwen3 MoE, Phi-4 and Mixtral | Downloads from the Hugging Face Hub |
| **torchtune** | Fine-tune and post-train open LLMs | Builder functions that compose shared modules (`TransformerDecoder`, `MultiHeadAttention`, `FeedForward`) | Llama 3.1 to 4, Mistral, Gemma 2, Phi-3/4, Qwen 2 to 3 | Downloads from the Hugging Face Hub |
| **transformers + Trainer** | Define, use and fine-tune pretrained models | One self-contained modeling file per model ("One Model, One File"); many are generated from short `modular_*.py` files that reuse other models' code | Hundreds of architectures across text, vision, audio and multimodal | 1M+ checkpoints on the Hub |

### Training

| Library | Pretraining from scratch | Fine-tuning and post-training | Distributed training |
|---|---|---|---|
| **OLM** | Yes, the core use case: streaming datasets, AMP, gradient accumulation, callbacks, checkpoints | None built in. LoRA and SFT are coming in 2.3; DPO and RL are on the v3 roadmap | One node: DDP and FSDP, with `AutoTrainer` choosing. No tensor, pipeline or expert parallelism. Multi-node planned for v4 |
| **nanoGPT** | Yes: reproduces GPT-2 124M on OpenWebText (about 4 days on 8×A100, per its README) | Full fine-tuning from GPT-2 weights | DDP, one or several nodes via `torchrun` |
| **nanochat** | Yes: a GPT-2-capability model for about $48 (about 2 hours on 8×H100, per its README) | SFT and RL stages in the pipeline | One node via `torchrun`, or a single GPU |
| **LitGPT** | Yes (`litgpt pretrain`) | Full, LoRA, QLoRA and Adapter fine-tuning | FSDP through Lightning Fabric; advertises 1 to 1000+ GPUs or TPUs |
| **torchtune** | No; it targets fine-tuning | SFT (full, LoRA, QLoRA), knowledge distillation, DPO, PPO, GRPO, quantization-aware training | Multi-GPU with FSDP2; multi-node since February 2025 |
| **transformers + Trainer** | Possible (build from a config and train), but the library centers on pretrained models | `Trainer`, plus [PEFT](https://huggingface.co/docs/peft/index) for LoRA and [TRL](https://huggingface.co/docs/trl/index) for SFT, DPO, GRPO and reward modeling | DDP, FSDP and DeepSpeed through Accelerate, including multi-node |

### Inference, status and license

| Library | Inference | Status (Sept 2026) | License |
|---|---|---|---|
| **OLM** | No built-in `generate()` in 2.2; an example sampler without a KV cache. A KV-cache `generate()` is coming in 2.3. Not a serving engine | Alpha, actively developed; 2.2.1 on PyPI (Aug 2026) | MIT |
| **nanoGPT** | `sample.py`; recomputes the full context per token (no KV cache) | README marks it deprecated in favor of nanochat | MIT |
| **nanochat** | Inference engine with a KV cache, and a chat CLI | Actively developed | MIT |
| **LitGPT** | `litgpt generate`, `litgpt chat`, `litgpt serve`; quantization | Actively developed; 0.5.13 (Jul 2026) | Apache-2.0 |
| **torchtune** | A generation recipe for checking models | No longer actively developed (announced Jul 2025); last release 0.6.1 (Apr 2025) | BSD-3-Clause |
| **transformers + Trainer** | `pipeline()` and `generate()` with many decoding strategies; for production serving, engines such as vLLM and SGLang build on its model definitions | Actively developed; 5.17.0 (Sep 2026) | Apache-2.0 |

## Project by project

### OLM and nanoGPT / nanochat

nanoGPT and OLM share a goal: code you can read end to end. nanoGPT does it with a single GPT-2 in about 300 lines of model code, and that minimalism is its strength. There is nothing to learn except the model and the loop. Its README now marks it as deprecated and sends readers to nanochat. nanochat keeps the minimal style but covers the whole path to a small chat model on one GPU node: tokenizer training, pretraining, supervised fine-tuning, reinforcement learning, evaluation and KV-cache inference.

**Pick nanoGPT or nanochat when** you want the smallest possible codebase for one model and are happy to edit it directly, or, with nanochat, when you want to see every stage from raw text to a chatbot. nanochat covers post-training and KV-cache inference, which OLM does not do yet.

**Pick OLM when** you want to compare *architectures*. Changing nanoGPT's attention or norm means editing its one model class, and keeping several variants side by side means keeping several copies of it. In OLM, GPT-2, Llama 3, Gemma 3, DeepSeek-V3 and 23 other model classes are built from the same component library. Swapping grouped-query attention for multi-head latent attention, or a dense feed-forward layer for a Mixture-of-Experts layer, is a local edit, and the training loop stays the same.

### OLM and LitGPT

LitGPT is a mature, production-minded toolkit: 20+ model families with downloadable weights, command-line recipes for pretraining, fine-tuning (including LoRA and QLoRA), evaluation and serving, quantization, and FSDP scaling through Lightning Fabric to many GPUs, nodes or TPUs. Its model code is readable, and it keeps every family in one config-driven `GPT` class.

**Pick LitGPT when** you want to fine-tune, evaluate or serve an existing open model, need quantization, or need to scale past one machine. In all of these areas LitGPT is ahead of OLM today.

**Pick OLM when** the architecture itself is what you are changing. In LitGPT a new attention or feed-forward variant is added to the shared model file and selected by a config field. In OLM it is a new `nn.Module` placed in a `Block`, and it can be mixed with the existing parts of any preset. Each OLM family, including hybrids such as Qwen3-Next and Kimi Linear (linear plus full attention) and DeepSeek-V3 (latent attention plus sigmoid-routed MoE), is its own file that names the parts it uses. OLM also has a teaching path built on the same code.

### OLM and torchtune

torchtune's design principles are close to OLM's: native PyTorch, composition over inheritance, and small reusable modules assembled by builder functions. It was built for post-training, with recipes for SFT, LoRA, QLoRA, knowledge distillation, DPO, PPO, GRPO and quantization-aware training. In July 2025 its maintainers stopped active development: no new features, with critical bug fixes and security patches through the end of 2025.

**Pick torchtune when** you already depend on one of its recipes and it works for you. For new post-training work, a maintained library such as TRL is the safer choice.

**Pick OLM when** you want to pretrain from scratch or study architectures. torchtune never targeted pretraining, and OLM does not target post-training yet.

### OLM and Hugging Face transformers + Trainer

`transformers` is the standard way to use pretrained models. It has hundreds of architectures across modalities and more than a million checkpoints on the Hub. `Trainer`, PEFT and TRL cover fine-tuning and post-training, Accelerate covers DDP, FSDP, DeepSpeed and multi-node, and inference engines build on its model definitions. Each model file is self-contained on purpose ("One Model, One File"), so you can read one model top to bottom without jumping between files. The same file also carries the machinery the rest of the library relies on, such as KV-cache handling, pluggable attention backends and generation support. The README says the library is not meant as a modular toolbox of building blocks for neural nets.

**Pick `transformers` when** you need a specific pretrained model, want to fine-tune or align it, or need a mature, widely supported stack. For most applied work this is the right default, and OLM does not try to replace it.

**Pick OLM when** you want exactly that modular toolbox: a model assembled from interchangeable parts that you can swap, rewire and train from scratch, with a training stack that stays small enough to read.

## When to pick which

| If you want to… | Start with |
|---|---|
| Understand how a GPT works, in the fewest lines | nanoGPT (deprecated but still readable), or the OLM [Learn From Scratch](learn/index.md) course |
| See the whole path from raw text to a chat model | nanochat |
| Compare attention, norm, positional or MoE choices under one training loop | **OLM** |
| Implement a new layer and test it inside known architectures (Llama, Qwen, Gemma, DeepSeek) | **OLM** |
| Pretrain a small model from scratch on one GPU or one multi-GPU node | **OLM**, nanochat or LitGPT |
| Pretrain across many nodes | [torchtitan](https://github.com/pytorch/torchtitan), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) or LitGPT |
| Fine-tune or align an existing pretrained model (LoRA, SFT, DPO, GRPO) | transformers with PEFT and TRL, or LitGPT |
| Serve a model to users | [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang) |

## Where OLM falls short today

Being honest about the gaps matters more than the table. Compared with the projects above, OLM 2.2 lacks:

- **Pretrained weights and checkpoint loading.** Every other project here except nanochat can start from someone else's weights.
- **Text generation.** There is no built-in `generate()` and no KV cache yet (coming in 2.3).
- **Fine-tuning and post-training.** There is no LoRA, SFT, DPO or RL (LoRA and SFT are coming in 2.3).
- **Scale beyond one machine.** There is no multi-node support, and no tensor, pipeline or expert parallelism.
- **Memory-saving training features.** There is no activation checkpointing and no quantized training. On CUDA, `use_amp=True` autocasts to fp16 even on GPUs that support bf16.
- **Maturity.** OLM is alpha software with a smaller community. The API may still change.

The [FAQ](faq.md) and the [roadmap](roadmap.md) track which of these are being worked on.

## Sources

Checked on 27 September 2026.

- nanoGPT: [README](https://github.com/karpathy/nanoGPT) (deprecation note, model and training script sizes, GPT-2 loading, multi-node DDP) and [`model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) (sampling without a KV cache).
- nanochat: [README and scripts](https://github.com/karpathy/nanochat) (pipeline stages, 8×H100 speedrun cost, single-node design, KV-cache engine).
- LitGPT: [README](https://github.com/Lightning-AI/litgpt), [supported models](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/download_model_weights.md), [`litgpt/model.py`](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py) and [PyPI release history](https://pypi.org/project/litgpt/).
- torchtune: [README](https://github.com/meta-pytorch/torchtune), [The future of torchtune (#2883)](https://github.com/meta-pytorch/torchtune/issues/2883), [overview and design principles](https://meta-pytorch.org/torchtune/stable/overview.html), [`torchtune.modules`](https://meta-pytorch.org/torchtune/stable/api_ref_modules.html) and [PyPI release history](https://pypi.org/project/torchtune/).
- Hugging Face: [`transformers` README](https://github.com/huggingface/transformers), [docs home](https://huggingface.co/docs/transformers/index), [philosophy](https://huggingface.co/docs/transformers/philosophy), [Trainer and Accelerate](https://huggingface.co/docs/transformers/accelerate), [TRL](https://huggingface.co/docs/trl/index), [PEFT](https://huggingface.co/docs/peft/index), [PyPI release history](https://pypi.org/project/transformers/), and the [TGI maintenance notice](https://github.com/huggingface/text-generation-inference), which recommends vLLM and SGLang.
- Licenses: each repository's license as reported by GitHub.
- OLM: this repository at the 2.2.1 release. The model counts come from `olm.models.__all__`.
