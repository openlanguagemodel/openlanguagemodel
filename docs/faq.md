# FAQ

Short answers to the questions people ask most about OLM, with links to the page that has the details. Everything here describes OLM 2.2.1, the current release on PyPI.

If your question is not here, search the [GitHub issues](https://github.com/openlanguagemodel/openlanguagemodel/issues) or open a new one.

## About OLM

### What is OLM, in one paragraph?

OLM is a PyTorch library for building language-model architectures from readable, swappable parts and pretraining them on one machine, from one CPU up to all the GPUs of a single node. It is not a serving engine, a multi-node platform, a pretrained-weights hub or an RLHF framework. [What OLM Is (and Isn't)](why-olm.md) explains the scope, and [OLM vs other libraries](comparison.md) compares it with nanoGPT, LitGPT, torchtune and Hugging Face `transformers`.

### How do I install it, and which Python and PyTorch versions work?

```bash
pip install openlanguagemodel
```

The package installs as `olm` (`import olm`). It supports Python 3.10, 3.11 and 3.12.

Use PyTorch 2.4 or newer. `pyproject.toml` declares `torch>=2.1.0`, but that floor is too low ([#76](https://github.com/openlanguagemodel/openlanguagemodel/issues/76)):

- `olm.train` imports `torch.amp.GradScaler`, which needs PyTorch 2.3.
- pip installs a `transformers` 5.x release by default, and recent 5.x releases turn off their PyTorch support below PyTorch 2.4. `HFTokenizer.encode` asks `transformers` for PyTorch tensors, so it fails there.

To work on OLM itself, install from source. [Installation](installation.md) covers both.

### How do I cite OLM?

Use the BibTeX entry in the [README](https://github.com/openlanguagemodel/openlanguagemodel#citation). The design is described in the paper [OpenLanguageModel: Readable and Composable Small-Language-Model Pretraining for Education and Research](https://arxiv.org/abs/2607.16669) (arXiv:2607.16669).

## Models and weights

### Do the presets come with pretrained weights?

No. Every class in `olm.models`, such as `GPT2`, `Llama3_1_8B` or `DeepSeekV3_671B`, builds a published model's configuration with **randomly initialized** weights. The presets are for training from scratch, for studying an architecture, and for ablations. They are not for chatting with or fine-tuning an existing model.

### Can I load Hugging Face (or Meta, Qwen, Mistral) checkpoints into an OLM model?

Not in 2.2. OLM has no `from_pretrained()` and no checkpoint converter, and parameter names do not match Hugging Face's. `HFTokenizer` loads Hugging Face *tokenizers*, not models. Export and conversion tooling is a long-term item on the [roadmap](roadmap.md).

If you need pretrained weights, use Hugging Face [`transformers`](https://huggingface.co/docs/transformers/index) or [LitGPT](https://github.com/Lightning-AI/litgpt).

### Which models are included, and what is the difference between `LM`, `GPT2Model` and `GPT2`?

OLM 2.2.1 has 27 model classes and 47 named presets in `olm.models`, from GPT-2 to DeepSeek-V3, Kimi K2 and Qwen3.5. Each family lives in [`src/olm/models/<org>/`](https://github.com/openlanguagemodel/openlanguagemodel/tree/main/src/olm/models). The three kinds of class are:

- `olm.nn.blocks.LM` is a small, configurable GPT-style model: pre-norm LayerNorm, attention with RoPE and a SwiGLU feed-forward. The tutorials use it.
- `GPT2Model`, `Llama3Model`, `Qwen3Model` and the other `...Model` classes build one family with every hyperparameter as an argument. Use them for your own sizes.
- `GPT2`, `Llama3_2_1B`, `Qwen2_5_7B` and the other named presets take no arguments and hard-code a released configuration.

### How do I look at a large preset without running out of memory?

Build it on PyTorch's `meta` device. Parameters then have shapes but no storage, so you can count them or print the module tree for any size:

```python
import torch

from olm.models import Llama3_1_8B

with torch.device("meta"):
    model = Llama3_1_8B()

print(f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
print(model.blocks[0])  # the token embedding
```

This prints 7.50B, not the 8.0B of the released Llama 3.1 8B. The Llama 3.1 presets currently tie the output head to the token embedding, while the release has a separate output matrix ([#83](https://github.com/openlanguagemodel/openlanguagemodel/issues/83)).

Building `Llama3_1_8B()` without `meta` would allocate about 30 GB of fp32 random weights.

## Training

### Can I use OLM blocks in my own training loop?

Yes. Every OLM component and model is a plain `torch.nn.Module`, and the `Trainer` is optional. [What OLM Is (and Isn't)](why-olm.md#plain-pytorch-no-runtime) has a complete, runnable example: a hand-written `nn.Module` built from OLM blocks, trained by a plain PyTorch loop. [Datasets & Training](guides/datasets-and-training.md#bring-your-own-training-loop) shows the same loop around an OLM `DataLoader`.

### What loss should a freshly initialized model start at?

A model that knows nothing should spread its guesses evenly over the vocabulary, which gives a cross-entropy loss of ln(vocab_size): about 10.8 for GPT-2's 50,257 tokens.

In OLM 2.2 you will see that only with an untied output head. With the default **tied** embeddings, the first loss is roughly `embed_dim`, far above ln(vocab_size). The token embedding uses PyTorch's default N(0, 1) initialization, and the tied head reuses that matrix. So a fresh model gives the *current* input token a logit of about `embed_dim` and predicts "the same token again" with high confidence. Training recovers, but it first has to undo the initialization: in the small GPT-2 run measured in [#84](https://github.com/openlanguagemodel/openlanguagemodel/pull/84), the loss was 122 at step 1 and only back to chance level at step 50.

```python
import math

import torch
import torch.nn.functional as F

from olm.nn.blocks import LM

torch.manual_seed(0)
vocab_size = 50257
tokens = torch.randint(0, vocab_size, (4, 65))
input_ids, labels = tokens[:, :-1], tokens[:, 1:]

for tie in (True, False):
    model = LM(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_len=64,
        tie_embeddings=tie,
    )
    with torch.no_grad():
        logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))
    print(f"tie_embeddings={tie}: first loss {loss.item():.1f}")

print(f"ln(vocab_size) = {math.log(vocab_size):.1f}")
```

With `embed_dim=128`, the tied model starts near 120 and the untied one near 11.

A change that initializes embeddings with standard deviation 0.02, so that a fresh model starts at ln(vocab_size), is under review in [#84](https://github.com/openlanguagemodel/openlanguagemodel/pull/84).

### How do I train on my own text, or with my own tokenizer?

- **Your own text.** Point `LocalTextDataset` at a folder of `.txt` files. [Getting Started](getting-started.md#train-on-local-text) has a complete example.
- **Any Hugging Face dataset.** Use `HuggingFaceTextDataset`, or `FineWebEduDataset` for FineWeb-Edu. See [Datasets & Training](guides/datasets-and-training.md#datasets).
- **Your own tokenizer.** `HFTokenizer` accepts a Hugging Face Hub name (`HFTokenizer("gpt2")`) or a local directory saved with `save_pretrained`. For anything else, subclass `olm.data.tokenization.TokenizerBase` and implement `encode` (text to a 1-D tensor of token ids) and `decode`. Override `save` too if you want `model.save(..., tokenizer=tok)` to store it. `load_model` only restores an `HFTokenizer`, so reload a custom tokenizer yourself.

Size the model's `vocab_size` as `len(tok.tokenizer)`, the full size of the underlying Hugging Face tokenizer. In 2.2, `tok.vocab_size` is only the base vocabulary and leaves out added special tokens, whose ids sit at or above it. Qwen tokenizers, for example, report a `vocab_size` of 151,643 but encode `<|im_start|>` as id 151,644, which would index past an embedding of that size. For GPT-2 the two numbers are equal (50,257).

### How do I save and load a trained model?

`model.save(directory, tokenizer=tok)` writes the whole module and, optionally, the tokenizer. `load_model(directory)` from `olm.nn.structure` returns the model, or a `(model, tokenizer)` pair when a tokenizer was saved, so you never rebuild the architecture by hand. The format pickles Python objects, so only load directories you created or trust. See [Saving and loading](guides/datasets-and-training.md#saving-and-loading).

### How do I resume an interrupted training run?

OLM 2.2 has no one-call resume. `CheckpointCallback` writes `step_N.pt` files that hold the model, optimizer, gradient-scaler and learning-rate-scheduler state, the step, the epoch and the loss history. You reload them by hand, as shown in [Resuming trainer checkpoints](guides/datasets-and-training.md#resuming-trainer-checkpoints). That recipe has three gaps:

- The scheduler state is saved but not restored. The default warmup-cosine scheduler is only created inside `train()`, so `trainer.scheduler` is still `None` when the recipe runs, the recipe skips it, and the schedule restarts from warmup. Pass your own `scheduler=` to the `Trainer` so that it exists before `train()` and can be restored.
- Random-number-generator state and callback state are not saved.
- The data position is approximate. The recipe's `skip_batches = step * batch_size` ignores gradient accumulation, and datasets apply `skip_batches` separately to each data-loader worker and each rank.

A one-call resume that restores all of this is coming in 2.3.

### How do I add a load-balancing loss to an MoE model?

The `Trainer` adds any auxiliary losses a model returns in an `LMOutput` (`olm.train.LMOutput`) to the language-model loss. The built-in MoE presets return plain logits, so in 2.2 **they train without a load-balancing loss** unless you add one.

For the families built on `olm.nn.moe.MoERouter` (DeepSeek-V3, Kimi K2, Llama 4 Maverick, Qwen3.5, Ling 2.5, MiniMax M2.5, Sarvam and Step 3.5 Flash), each router keeps its last routing decision in `router.last_stats`. A small wrapper turns that into a loss per MoE layer:

```python
import torch
import torch.distributed as dist
import torch.nn as nn

from olm.models import DeepSeekV3Model
from olm.nn.moe import MoERouter
from olm.train import LMOutput, TrainerCallback
from olm.train.losses import SequenceLoadBalanceLoss


class WithRouterLoss(nn.Module):
    """Return an LMOutput whose aux_losses hold one balance loss per MoE layer."""

    def __init__(self, model, coefficient=1e-3):
        super().__init__()
        self.model = model
        self.routers = [m for m in model.modules() if isinstance(m, MoERouter)]
        self.balance = [
            SequenceLoadBalanceLoss(r.num_experts, r.top_k, coefficient, r.scoring_func)
            for r in self.routers
        ]

    def forward(self, input_ids):
        logits = self.model(input_ids)
        aux_losses = [
            loss(r.last_stats.router_logits, r.last_stats.top_k_indices)
            for r, loss in zip(self.routers, self.balance)
            if r.last_stats is not None  # a router that has never run has no stats
        ]
        return LMOutput(logits=logits, aux_losses=aux_losses)


class UpdateExpertBias(TrainerCallback):
    """Loss-free balancing: nudge each router's expert bias after every step."""

    def on_step_end(self, trainer, step, loss):
        for m in trainer.model.modules():
            if isinstance(m, MoERouter) and m.expert_bias is not None:
                fraction = m.last_stats.expert_fraction.clone()
                if dist.is_available() and dist.is_initialized():
                    # Use the load over the global batch so every rank applies
                    # the same update and the biases stay identical.
                    dist.all_reduce(fraction)
                    fraction /= dist.get_world_size()
                m.update_expert_bias_(fraction)


tiny = DeepSeekV3Model(
    vocab_size=128,
    embed_dim=32,
    num_layers=4,
    num_heads=4,
    max_seq_len=16,
    kv_lora_rank=8,
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
    q_lora_rank=8,
    dense_intermediate_size=64,
    moe_intermediate_size=16,
    num_experts=4,
    num_shared_experts=1,
    top_k=2,
    first_k_dense_replace=1,
)
model = WithRouterLoss(tiny)
out = model(torch.randint(0, 128, (2, 16)))
print(len(out.aux_losses), "balance losses")  # one per MoE layer: 3
```

Pass `model` to the `Trainer` as usual. Add `callbacks=[UpdateExpertBias()]` for routers built with `routing_method="noaux_tc"` (DeepSeek-V3, Kimi K2 and Ling 2.5). In 2.2 nothing else calls `update_expert_bias_()`.

The expert bias is a parameter that never receives a gradient, so DDP does not keep it in sync across GPUs. That is why the callback averages each router's expert load over all ranks before the update. Without the `all_reduce`, every GPU would drift to its own biases and route tokens differently, and `CheckpointCallback` would save only rank 0's copy. In a single process the callback skips the `all_reduce`.

The older MoE layers in `olm.nn.feedforward` (used by Qwen3, Qwen3-Next, Mistral Large 3, Kimi Linear and MiniMax M2) discard their router logits, so this wrapper finds no routers there ([#82](https://github.com/openlanguagemodel/openlanguagemodel/issues/82)).

### Why does DDP fail on my MoE model with "Expected to have finished reduction in the prior iteration"?

Some parameters get no gradient in a step. That happens to an expert that receives no tokens, to the expert bias of a `noaux_tc` router, and on every step to a multi-token-prediction (MTP) head. DeepSeek-V3, Qwen3.5, MiniMax M2.5 and Step 3.5 Flash build an MTP head by default, but only `forward_with_mtp()` runs it, not the plain `forward()`. DDP's default `find_unused_parameters=False` treats unused parameters as an error on the next step. Pass `find_unused_parameters=True` to `DDPTrainer`, or `ddp_find_unused_parameters=True` to `AutoTrainer`. Dense models do not need it.

### Why do I get "OutputHead ties weights by default"?

`OutputHead` reuses the token-embedding matrix as its output projection unless told otherwise, so it needs that embedding:

```python
from olm.nn.blocks import OutputHead
from olm.nn.embeddings import Embedding

embedding = Embedding(1000, 64)
head = OutputHead(64, 1000, tied_embedding=embedding)  # tied (the default)
untied = OutputHead(64, 1000, tie_weights=False)  # separate output matrix
```

Code written for OLM before 2.2 may call `OutputHead(embed_dim, vocab_size)` without either argument. That form raises this error in 2.2.

### Why do I get an `IndexError` when my input is longer than `max_seq_len`?

Rotary (RoPE) position tables are built for `max_seq_len` positions when the model is created, so a longer input indexes past the end of the table and raises an `IndexError`. On a GPU the same mistake can surface as a CUDA device-side assert. Models with learned absolute positions, such as `GPT2Model`, check the length first and raise a `ValueError` that names `max_seq_len`. Keep every input, including a prompt plus the tokens generated after it, at or below `max_seq_len`, or build the model with a larger `max_seq_len`.

## Hardware

### Which GPU do I need?

It depends on the model size. Full training with AdamW in fp32 needs about 16 bytes per parameter before activations: 4 for the weights, 4 for the gradients and 8 for AdamW's two moment estimates. `olm.train.estimate_model_size` applies this rule, and it works on `meta` models:

```python
import torch

from olm.models import GPT2, Llama3_2_1B
from olm.train import estimate_model_size

for preset in (GPT2, Llama3_2_1B):
    with torch.device("meta"):
        model = preset()
    est = estimate_model_size(model)
    print(
        f"{preset.__name__}: {est['num_params'] / 1e6:,.0f}M params, "
        f"about {est['total_gb']:.1f} GB before activations"
    )
```

A GPT-2-small-sized model (124M parameters) needs under 2 GB for this, so any recent CUDA GPU can train it at a modest batch size and context length. A 1.2B model like `Llama3_2_1B` needs about 18 GB before activations, which is more than a 16 GB card has. Activation memory grows with batch size, context length and depth. Lower the batch size and use `grad_accum_steps` to keep the effective batch size.

### Can I train on a free Colab T4?

Yes, for small models. On a single T4, `AutoTrainer(device="auto")` picks the single-GPU `Trainer`. With `use_amp=True`, training runs in fp16 with a gradient scaler, which suits the T4 because it has no bf16 support. The [Colab notebooks](colab-notebooks.md) run their small demos on any runtime and tell you to switch to a GPU runtime for the 125M FineWeb-Edu run. OLM has not published T4 throughput numbers yet.

### Does OLM support multiple GPUs, or multiple machines?

- **Multiple GPUs in one machine: yes.** `DDPTrainer` replicates the model on every GPU. `FSDPTrainer` shards it when it does not fit on one GPU. `AutoTrainer` chooses between them, and you launch with `torchrun --nproc_per_node=<gpus>`. See [Distributed Training](tutorials/distributed-training.md).
- **Multiple machines: not supported yet.** The trainers use standard `torch.distributed`, but multi-node runs are not tested. Multi-node training is planned for v4 (see the [roadmap](roadmap.md)).
- **Model parallelism: no.** There is no tensor, pipeline or expert parallelism.

### Can I train on a CPU or a Mac?

On a CPU, yes. The test suite runs on CPU, and tiny models train there in seconds, which makes the CPU a good place to learn and debug. For anything larger than a toy model you want a CUDA GPU. Apple-silicon GPUs (the `mps` device) are not tested.

## Generation, fine-tuning and extending OLM

### Can I generate text? Is there a KV cache?

OLM 2.2 has no built-in `generate()` and no KV cache. [`examples/inference/chat.py`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/examples/inference/chat.py) has a small temperature and top-k sampler that you can copy. It recomputes the whole context for every new token, which is fine for small models and short outputs. A `generate()` with a KV cache is coming in 2.3. For serving a model to users, use an inference engine such as [vLLM](https://github.com/vllm-project/vllm).

### Can I fine-tune with LoRA, or do SFT, DPO or RLHF?

Not with built-in tools in 2.2.

- You can keep training a model you pretrained yourself with the same `Trainer`.
- LoRA and SFT utilities are coming in 2.3, ahead of the v3 [roadmap](roadmap.md) that lists them.
- DPO, PPO and GRPO remain on the v3 roadmap.
- To post-train a pretrained model today, use Hugging Face [TRL](https://huggingface.co/docs/trl/index) with [PEFT](https://huggingface.co/docs/peft/index).

### How do I add a new model architecture?

1. Copy [`src/olm/models/meta/llama3.py`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py). A family file holds a `XxxBlock`, a `XxxModel` that takes every hyperparameter as an argument, and named presets with the released values.
2. Build the block from existing `olm.nn` components with `Block`, `Residual` and `Repeat`. Write a custom `forward` only when the combinators cannot express the data flow.
3. Put the file in `src/olm/models/<org>/` and export the model and presets from that package's `__init__.py` and from `olm/models/__init__.py`.
4. Add a tiny-config smoke test to `tests/test_model_smoke.py`. Never build a full-size preset in a test: use `torch.device("meta")`.

[The Block System](guides/architecture.md) and [Custom Architectures](tutorials/custom-architecture.md) explain the building blocks. To contribute the model, open a pull request against `dev` as described in [CONTRIBUTING.md](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/CONTRIBUTING.md).

### How do I report a bug or contribute?

Open an issue on [GitHub](https://github.com/openlanguagemodel/openlanguagemodel/issues) with a minimal script that reproduces the problem, plus your OLM, PyTorch and Python versions. For code or docs changes, see [CONTRIBUTING.md](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/CONTRIBUTING.md). Pull requests go to the `dev` branch.
