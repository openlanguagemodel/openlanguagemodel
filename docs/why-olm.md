# What OLM Is (and Isn't)

OLM is a PyTorch library for building language models you can read, changing their architecture one part at a time, and pretraining them on one machine.

This page says what OLM is for, who it is for, and what it deliberately does not try to be, so you can decide quickly whether it fits your project. For a side-by-side look at nanoGPT, LitGPT, torchtune and Hugging Face `transformers`, see [OLM vs other libraries](comparison.md). For short answers to common questions, see the [FAQ](faq.md).

> [!NOTE]
> This page describes OLM **2.2.1**, the current release on PyPI (`pip install openlanguagemodel`). Where a feature is being built for the next release, the page says "coming in 2.3".

## What OLM is for

### Architectures you can read

An OLM model is written as the architecture it implements. A Llama-style block reads exactly like its description: RMSNorm and grouped-query attention inside a residual connection, then RMSNorm and a SwiGLU feed-forward inside another one.

```python
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual

embed_dim, num_heads, num_kv_heads, max_seq_len = 256, 8, 2, 512

llama_style_block = Block(
    [
        Residual(
            Block(
                [
                    RMSNorm(embed_dim),
                    GroupedQueryAttention(
                        embed_dim, num_heads, num_kv_heads, max_seq_len
                    ),
                ]
            )
        ),
        Residual(Block([RMSNorm(embed_dim), SwiGLUFFN(embed_dim)])),
    ]
)
```

The structure is expressed with a few combinators (`Block`, `Residual`, `Repeat`, `Parallel`), and the parts come from one shared library, `olm.nn`. The `GroupedQueryAttention`, `RMSNorm` and `SwiGLUFFN` classes in this block are reused by most of the presets, from Llama 2 to Mistral Small 3.1. GPT-2, DeepSeek-V3 and Kimi K2 are built from the same library with the parts they need, such as learned positions, LayerNorm, latent attention and Mixture-of-Experts layers. The real Llama 3 block in [`llama3.py`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py) is the same shape with its released hyperparameters. See [The Block System](guides/architecture.md) for how the combinators work.

OLM ships 27 model classes with 47 named presets. They run from GPT-2 (2019), OPT, Llama 2/3, Qwen 2.5, Phi-3/4, Gemma 2 and OLMo to recent designs: Qwen3, Qwen3-Next and Qwen3.5, Gemma 3, OLMo 3, Llama 4 Maverick, DeepSeek-V3, Kimi K2 and Kimi Linear, Mistral Small 3.1 and Mistral Large 3, MiniMax M2 and M2.5, Ling 2.5, Step 3.5 Flash, Sarvam and Nanbeige 4.1. The component library underneath includes multi-head latent attention (MLA), sliding-window and gated attention, linear attention (Gated DeltaNet, Kimi Delta Attention, Lightning Attention), block-sparse attention, Mixture-of-Experts routing with load-balancing losses, and multi-token prediction heads. Every model is source code in [`src/olm/models/`](https://github.com/openlanguagemodel/openlanguagemodel/tree/main/src/olm/models), not a configuration file.

### Changing one part at a time

Because the wiring and the parts are separate, an ablation is usually a local edit, often one line. Swap the attention layer for `SlidingWindowAttention`, `RMSNorm` for `LayerNorm`, or `SwiGLUFFN` for a Mixture-of-Experts layer, and the rest of the model and the training loop stay the same. The [Custom Architectures](tutorials/custom-architecture.md) tutorial and the ablation notebook in [Colab Notebooks](colab-notebooks.md) walk through this.

### One path from learning to research to pretraining

The same components serve three stages, so nothing you learn early is thrown away later:

1. **Learning.** The [Learn From Scratch](learn/index.md) course and the [Colab notebooks](colab-notebooks.md) build working models with OLM's blocks from the first lesson, starting from a little Python.
2. **Research.** You change the architecture, keep everything else fixed, and compare runs. Every component is a plain `torch.nn.Module`, so a module you write yourself drops into a `Block` next to OLM's own.
3. **Real pretraining on one machine.** `olm.data` streams Hugging Face datasets (with a FineWeb-Edu shortcut) or reads local text files. `olm.train` gives you a `Trainer` with mixed precision, gradient accumulation, gradient clipping, a warmup-cosine schedule and callbacks for checkpoints, validation, metrics and early stopping. `DDPTrainer` and `FSDPTrainer` scale that loop to every GPU in one node, and `AutoTrainer` picks between CPU, one GPU, DDP and FSDP for you. The [GPT-2 124M on FineWeb-Edu example](https://github.com/openlanguagemodel/openlanguagemodel/tree/main/examples/gpt2-fineweb-edu-10b) is a complete script with a config file and a `torchrun` launch.

### Plain PyTorch, no runtime

OLM is a library you call, not a framework that owns your program. The `Trainer` is optional. The model below is your own `nn.Module`: PyTorch's embedding and output layers wrapped around two OLM transformer layers, trained by a loop that has nothing to do with OLM.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.structure import Block
from olm.nn.structure.combinators import Residual


class TinyLM(nn.Module):
    """A hand-written model that uses OLM blocks for its transformer layers."""

    def __init__(self, vocab_size=64, embed_dim=64, num_layers=2, max_seq_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            Block(
                [
                    Residual(
                        Block(
                            [
                                RMSNorm(embed_dim),
                                GroupedQueryAttention(embed_dim, 4, 2, max_seq_len),
                            ]
                        )
                    ),
                    Residual(Block([RMSNorm(embed_dim), SwiGLUFFN(embed_dim)])),
                ]
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


torch.manual_seed(0)
model = TinyLM()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

for step in range(200):
    # Toy data: every sequence counts upward from a random start (mod 64),
    # so each next token is predictable from the one before it.
    start = torch.randint(0, 64, (16, 1))
    tokens = (start + torch.arange(33)) % 64
    input_ids, labels = tokens[:, :-1], tokens[:, 1:]

    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 50 == 0 or step == 199:
        print(f"step {step:3d}  loss {loss.item():.3f}")
```

It runs on a CPU in a few seconds. The loss starts near ln 64 ≈ 4.2 (a uniform guess over 64 tokens) and falls close to zero as the model learns the counting rule. Anything that works on an `nn.Module` works here: `torch.compile`, DDP or FSDP from `torch.distributed`, your own optimizer, or another library's trainer.

## Who OLM is for

- **Students and teachers** who want to see how a language model is put together and train one, rather than call a hosted model.
- **Researchers** running controlled ablations on attention, positional encoding, normalization, feed-forward and MoE choices, or implementing a new component and testing it inside known architectures.
- **Small teams pretraining small models** on one GPU or one multi-GPU machine, who want the training stack wired up without hiding the model.

## What OLM is not

OLM has a narrow scope on purpose. If your project is mainly one of the following, another tool will serve you better today.

### Not a serving engine

OLM does not serve models. It has no continuous batching, paged KV cache, quantized kernels or HTTP server, and it is not trying to compete with engines built for that.

- To serve a model in production, use an inference engine such as [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang). For local inference, use [llama.cpp](https://github.com/ggml-org/llama.cpp).
- OLM 2.2 has no built-in `generate()`. The [`examples/inference/chat.py`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/examples/inference/chat.py) script has a simple temperature and top-k sampler that recomputes the whole context for every new token.
- A `generate()` with a KV cache is coming in 2.3. It is meant for sampling and evaluating the models you train, not for serving traffic.

### Not a multi-node training platform

OLM trains on **one machine**: one CPU, one GPU, or all the GPUs of a single node with DDP or FSDP.

- There is no tensor, pipeline, context or expert parallelism.
- There are no cluster launchers, and multi-node runs are not tested. The trainers use standard `torch.distributed`, so a multi-node `torchrun` launch may work, but nothing in OLM is built or tested for it.
- Multi-node training is on the [roadmap](roadmap.md) for v4.
- For pretraining across many nodes today, look at [torchtitan](https://github.com/pytorch/torchtitan) or NVIDIA's [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

### Not a pretrained-weights hub

The presets in `olm.models` are **architectures with randomly initialized weights**, not checkpoints. `Llama3_1_8B()` builds a model from the Llama 3.1 8B configuration, with random weights. (A few older presets, including the Llama 3.1 ones, still tie the output head where the release does not, so their parameter counts come out low by one embedding matrix: [#83](https://github.com/openlanguagemodel/openlanguagemodel/issues/83).) OLM 2.2 has no `from_pretrained()` and cannot load Hugging Face, Meta or Qwen checkpoints.

To inspect a large preset without allocating memory, build it on PyTorch's `meta` device:

```python
import torch

from olm.models import Llama3_2_1B

with torch.device("meta"):
    model = Llama3_2_1B()

num_params = sum(p.numel() for p in model.parameters())
print(f"{num_params / 1e9:.2f}B parameters, no memory allocated")
```

If you need a model that already knows language, start from [Hugging Face `transformers`](https://huggingface.co/docs/transformers/index) or [LitGPT](https://github.com/Lightning-AI/litgpt), which both download pretrained weights.

### Not a fine-tuning or RLHF framework

OLM 2.2 has no LoRA, instruction-tuning (SFT) data utilities, DPO, PPO or GRPO.

- You can keep training any OLM model you trained yourself with the same `Trainer`.
- LoRA and SFT utilities are coming in 2.3, ahead of the v3 [roadmap](roadmap.md) that lists them.
- Preference optimization and RL methods remain on the v3 roadmap.
- For post-training a pretrained model today, use Hugging Face [TRL](https://huggingface.co/docs/trl/index) with [PEFT](https://huggingface.co/docs/peft/index), or LitGPT. For a small end-to-end example that includes SFT and RL, read Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat).

## Where OLM is today

OLM is alpha software (the package is marked `Development Status :: 3 - Alpha`), and its API may still change. Known gaps are tracked in [GitHub issues](https://github.com/openlanguagemodel/openlanguagemodel/issues). The main ones for 2.2 are:

- There is no one-call training resume, no built-in text generation, and no evaluation helper beyond the validation-loss callback. All three are coming in 2.3.
- There is no activation (gradient) checkpointing.
- With `use_amp=True` on a CUDA GPU, the trainers autocast to fp16 with a gradient scaler, even on GPUs that support bf16.
- With tied embeddings (the default), a freshly initialized model starts at a much higher loss than ln(vocab_size). The [FAQ](faq.md#what-loss-should-a-freshly-initialized-model-start-at) explains why.
- The built-in MoE presets return plain logits, so they train without a load-balancing loss unless you add one. The [FAQ](faq.md#how-do-i-add-a-load-balancing-loss-to-an-moe-model) shows how.

The design of OLM and its first experiments are described in the paper [OpenLanguageModel: Readable and Composable Small-Language-Model Pretraining for Education and Research](https://arxiv.org/abs/2607.16669) (arXiv:2607.16669).

## Next steps

- [OLM vs other libraries](comparison.md): when to pick OLM, nanoGPT, nanochat, LitGPT, torchtune or Hugging Face `transformers`.
- [FAQ](faq.md): short answers to common questions.
- [Getting Started](getting-started.md): install OLM and train a small model.
- [The Block System](guides/architecture.md): how models are assembled.
