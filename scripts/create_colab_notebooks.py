"""Generate the tracked Colab notebooks for OLM learning.

The notebooks are authored here as plain Markdown/code cells so they can be
reviewed and regenerated without hand-editing large JSON blobs.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def _source(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def notebook(name: str, cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": name,
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


INSTALL_CELL = """
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("olm") is None:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "git+https://github.com/openlanguagemodel/openlanguagemodel.git",
    ])
"""


def first_language_model() -> dict:
    return notebook(
        "01_first_language_model_colab.ipynb",
        [
            md(
                """
                # First Language Model With OLM

                <a href="https://colab.research.google.com/github/openlanguagemodel/openlanguagemodel/blob/main/notebooks/01_first_language_model_colab.ipynb" target="_blank">
                  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
                </a>

                This notebook trains a tiny causal language model on a local text file,
                samples from it, saves it, and loads it back.

                It is intentionally small. The point is to see the whole loop:
                data -> tokenizer -> model -> trainer -> generation -> save/load.
                """
            ),
            md(
                """
                ## Install OLM

                In Colab, this installs the latest GitHub version. If you are running
                inside a local checkout with OLM already installed, this cell does
                nothing.
                """
            ),
            code(INSTALL_CELL),
            md("## Imports And Reproducibility"),
            code(
                """
                import math
                import os
                import random
                import shutil
                from pathlib import Path

                import torch

                from olm.data.datasets import DataLoader, LocalTextDataset
                from olm.data.tokenization import HFTokenizer
                from olm.nn.blocks import LM
                from olm.nn.structure import load_model
                from olm.train import Trainer
                from olm.train.optim import AdamW

                seed = 42
                random.seed(seed)
                torch.manual_seed(seed)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                print("device:", device)
                """
            ),
            md(
                """
                ## Make A Tiny Local Dataset

                `LocalTextDataset` reads `.txt` files from a directory and turns the
                stream into `(input_ids, labels)` pairs for next-token prediction.
                """
            ),
            code(
                """
                data_dir = Path("tiny_olm_data")
                data_dir.mkdir(exist_ok=True)

                seed_text = '''
                Language models learn by predicting the next token.
                A transformer reads a sequence, mixes information with attention,
                and writes new hidden states through feed-forward layers.
                OLM keeps these pieces visible. Embeddings, attention, norms,
                residual paths, output heads, and training loops are all ordinary
                PyTorch modules.
                '''

                repeated = "\\n".join(seed_text.strip() for _ in range(500))
                (data_dir / "tiny.txt").write_text(repeated, encoding="utf-8")

                print((data_dir / "tiny.txt").read_text(encoding="utf-8")[:500])
                """
            ),
            md("## Tokenizer, Dataset, And Loader"),
            code(
                """
                tokenizer = HFTokenizer("gpt2")
                context_length = 128

                dataset = LocalTextDataset(
                    data_dir,
                    tokenizer,
                    context_length=context_length,
                    shuffle=True,
                    seed=seed,
                )

                loader = DataLoader(
                    dataset,
                    batch_size=8,
                    num_workers=0,
                    pin_memory=device.startswith("cuda"),
                )

                x, y = next(iter(loader))
                print("input batch:", tuple(x.shape), x.dtype)
                print("label batch:", tuple(y.shape), y.dtype)
                print("decoded example:")
                print(tokenizer.decode(x[0]))
                """
            ),
            md(
                """
                ## Build A Small LM

                `LM` is a compact GPT-style model assembled from OLM blocks. Its
                output head ties weights to the input token embedding by default.
                """
            ),
            code(
                """
                model = LM(
                    vocab_size=tokenizer.vocab_size,
                    embed_dim=128,
                    num_heads=4,
                    num_layers=4,
                    max_seq_len=context_length,
                    dropout=0.0,
                )

                params = sum(p.numel() for p in model.parameters())
                print(f"parameters: {params:,}")

                # The output projection reuses the token embedding matrix.
                print("tied output head:", model.blocks[-1].weight is model.blocks[0].embedding.weight)
                """
            ),
            md("## A Tiny Generation Helper"),
            code(
                """
                @torch.no_grad()
                def generate(model, tokenizer, prompt, max_new_tokens=80, temperature=0.8, top_k=50):
                    model.eval()
                    input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)

                    for _ in range(max_new_tokens):
                        idx = input_ids[:, -context_length:]
                        logits = model(idx)[:, -1, :]
                        logits = logits / max(temperature, 1e-6)

                        if top_k is not None:
                            k = min(top_k, logits.size(-1))
                            values, _ = torch.topk(logits, k)
                            logits[logits < values[:, [-1]]] = -float("inf")

                        probs = torch.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1)
                        input_ids = torch.cat([input_ids, next_id], dim=1)

                    return tokenizer.decode(input_ids[0].cpu())

                model = model.to(device)
                print(generate(model, tokenizer, "Language models", max_new_tokens=40))
                """
            ),
            md(
                """
                ## Train

                This is deliberately short. Increase `max_steps` if you want the
                generated text to become less chaotic.
                """
            ),
            code(
                """
                trainer = Trainer(
                    model,
                    AdamW,
                    loader,
                    device=device,
                    context_length=context_length,
                    learning_rate=3e-4,
                    weight_decay=0.1,
                    grad_accum_steps=1,
                    use_amp=device.startswith("cuda"),
                    grad_clip_norm=1.0,
                    use_warmup_cosine=True,
                )

                losses = trainer.train(epochs=1, max_steps=50, log_interval=10)
                print("first loss:", losses[0])
                print("last loss:", losses[-1])
                """
            ),
            md("## Generate After Training"),
            code(
                """
                print(generate(model, tokenizer, "Language models", max_new_tokens=100))
                """
            ),
            md("## Save And Load"),
            code(
                """
                save_dir = Path("tiny_olm_model")
                if save_dir.exists():
                    shutil.rmtree(save_dir)

                model.cpu().save(str(save_dir), tokenizer=tokenizer)
                loaded_model, loaded_tokenizer = load_model(str(save_dir))
                loaded_model = loaded_model.to(device)

                print("saved files:", sorted(p.name for p in save_dir.iterdir()))
                print(generate(loaded_model, loaded_tokenizer, "A transformer", max_new_tokens=80))
                """
            ),
            md(
                """
                ## What To Try Next

                - Change `embed_dim`, `num_layers`, or `num_heads`.
                - Increase `max_steps`.
                - Replace the local text file with your own notes or essays.
                - Move to the FineWeb-Edu notebook for a real pretraining dataset.
                """
            ),
        ],
    )


def fineweb_125m() -> dict:
    return notebook(
        "02_train_125m_fineweb_edu_colab.ipynb",
        [
            md(
                """
                # Train A 125M GPT-Style Model On FineWeb-Edu

                <a href="https://colab.research.google.com/github/openlanguagemodel/openlanguagemodel/blob/main/notebooks/02_train_125m_fineweb_edu_colab.ipynb" target="_blank">
                  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
                </a>

                This notebook builds a GPT-2-small-size model, streams FineWeb-Edu
                from Hugging Face, and runs a budget-aware training smoke run.

                The default step count is intentionally small. Treat it as a
                preflight check, then increase the steps on rented GPU hardware.
                """
            ),
            md("## Install OLM"),
            code(INSTALL_CELL),
            md("## Imports"),
            code(
                """
                import math
                import random
                import time

                import torch

                from olm.data.datasets import DataLoader, FineWebEduDataset
                from olm.data.tokenization import HFTokenizer
                from olm.models.openai import GPT2Model
                from olm.train import AutoTrainer
                from olm.train.optim import AdamW

                seed = 42
                random.seed(seed)
                torch.manual_seed(seed)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                print("device:", device)
                if device == "cpu":
                    print("For real 125M training, switch Colab to a GPU runtime.")
                """
            ),
            md(
                """
                ## Budget Knobs

                Keep `MAX_STEPS` low for a smoke run. For a real run, increase it
                after the first batches are stable. Cost depends on the GPU provider,
                so the useful number to track here is tokens processed.
                """
            ),
            code(
                """
                CONTEXT_LENGTH = 1024
                MICRO_BATCH_SIZE = 1
                GRAD_ACCUM_STEPS = 32
                MAX_STEPS = 20

                effective_batch = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
                tokens_per_step = effective_batch * CONTEXT_LENGTH
                planned_tokens = tokens_per_step * MAX_STEPS

                print("effective batch:", effective_batch)
                print("tokens per optimizer step:", f"{tokens_per_step:,}")
                print("planned tokens in this notebook run:", f"{planned_tokens:,}")
                """
            ),
            md("## Tokenizer And FineWeb-Edu Stream"),
            code(
                """
                tokenizer = HFTokenizer("gpt2")

                dataset = FineWebEduDataset(
                    tokenizer=tokenizer,
                    subset="sample-10BT",
                    split="train",
                    context_length=CONTEXT_LENGTH,
                    streaming=True,
                    shuffle=True,
                    seed=seed,
                )

                loader = DataLoader(
                    dataset,
                    batch_size=MICRO_BATCH_SIZE,
                    num_workers=0,
                    pin_memory=device.startswith("cuda"),
                )

                x, y = next(iter(loader))
                print("batch:", tuple(x.shape), tuple(y.shape))
                print(tokenizer.decode(x[0][:200]))
                """
            ),
            md("## Build The 125M GPT-Style Model"),
            code(
                """
                model = GPT2Model(
                    vocab_size=tokenizer.vocab_size,
                    embed_dim=768,
                    num_layers=12,
                    num_heads=12,
                    max_seq_len=CONTEXT_LENGTH,
                    dropout=0.1,
                )

                params = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"parameters: {params:,}")
                print(f"trainable: {trainable:,}")
                print("approx size fp32:", f"{params * 4 / 1e9:.2f} GB")
                """
            ),
            md("## Train A Short Preflight Run"),
            code(
                """
                trainer = None
                losses = []

                if device == "cpu":
                    print("Skipping 125M training on CPU. In Colab, choose Runtime -> Change runtime type -> GPU.")
                else:
                    trainer = AutoTrainer(
                        model,
                        AdamW,
                        loader,
                        device="auto",
                        context_length=CONTEXT_LENGTH,
                        learning_rate=3e-4,
                        weight_decay=0.1,
                        grad_accum_steps=GRAD_ACCUM_STEPS,
                        use_amp=True,
                        grad_clip_norm=1.0,
                        preset="balanced",
                        verbose=True,
                    )

                    start = time.time()
                    losses = trainer.train(epochs=1, max_steps=MAX_STEPS, log_interval=5)
                    elapsed = time.time() - start

                    print("final loss:", losses[-1])
                    print("elapsed minutes:", elapsed / 60)
                    print("tokens processed:", f"{trainer.total_tokens_processed:,}")
                """
            ),
            md(
                """
                ## Scale The Run

                After the preflight works:

                - Increase `MAX_STEPS`.
                - Keep `MICRO_BATCH_SIZE` small if memory is tight.
                - Increase `GRAD_ACCUM_STEPS` to raise the effective batch size.
                - Use `AutoTrainer(..., preset="memory_efficient")` if the model is
                  close to GPU memory limits.
                - Save checkpoints periodically for longer rented-GPU runs.
                """
            ),
            code(
                """
                if trainer is None:
                    print("No checkpoint saved because training was skipped.")
                else:
                    checkpoint = {
                        "model_state_dict": trainer.model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "losses": losses,
                        "context_length": CONTEXT_LENGTH,
                        "max_steps": MAX_STEPS,
                    }
                    torch.save(checkpoint, "gpt2_125m_fineweb_edu_preflight.pt")
                    print("saved gpt2_125m_fineweb_edu_preflight.pt")
                """
            ),
        ],
    )


def custom_architecture_ablation() -> dict:
    return notebook(
        "03_custom_architecture_ablation_colab.ipynb",
        [
            md(
                """
                # Custom Architecture Ablation

                <a href="https://colab.research.google.com/github/openlanguagemodel/openlanguagemodel/blob/main/notebooks/03_custom_architecture_ablation_colab.ipynb" target="_blank">
                  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
                </a>

                This notebook builds two tiny language models that differ only in
                their block choices, then trains each for one optimizer step.

                The goal is not to get a good model. The goal is to see how local an
                ablation can be in OLM.
                """
            ),
            md("## Install OLM"),
            code(INSTALL_CELL),
            md("## Imports And Tiny Data"),
            code(
                """
                import random
                from pathlib import Path

                import torch

                from olm.data.datasets import DataLoader, LocalTextDataset
                from olm.data.tokenization import HFTokenizer
                from olm.nn.attention import FlashAttention, GroupedQueryAttention
                from olm.nn.blocks import OutputHead
                from olm.nn.embeddings import Embedding
                from olm.nn.feedforward import ClassicFFN, SwiGLUFFN
                from olm.nn.norms import LayerNorm, RMSNorm
                from olm.nn.structure import Block
                from olm.nn.structure.combinators import Repeat, Residual
                from olm.train import Trainer
                from olm.train.optim import AdamW

                seed = 7
                random.seed(seed)
                torch.manual_seed(seed)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                context_length = 64
                data_dir = Path("ablation_data")
                data_dir.mkdir(exist_ok=True)
                (data_dir / "notes.txt").write_text(
                    "\\n".join(
                        "Attention moves information across tokens. Norms stabilize hidden states. Feed-forward layers transform features."
                        for _ in range(300)
                    ),
                    encoding="utf-8",
                )

                tokenizer = HFTokenizer("gpt2")
                dataset = LocalTextDataset(data_dir, tokenizer, context_length=context_length, shuffle=True, seed=seed)
                loader = DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=device.startswith("cuda"))
                fixed_batch = next(iter(loader))
                print(tuple(fixed_batch[0].shape), tuple(fixed_batch[1].shape))
                """
            ),
            md("## Define A Swappable Block"),
            code(
                """
                def make_attention(kind, embed_dim, num_heads, max_seq_len, dropout):
                    if kind == "mha":
                        return FlashAttention(embed_dim, num_heads, dropout=dropout, causal=True)
                    if kind == "gqa":
                        return GroupedQueryAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            num_kv_heads=max(1, num_heads // 2),
                            max_seq_len=max_seq_len,
                            dropout=dropout,
                            use_bias=False,
                        )
                    raise ValueError(f"unknown attention kind: {kind}")


                def make_ffn(kind, embed_dim, hidden_dim, dropout):
                    if kind == "classic":
                        return ClassicFFN(embed_dim, hidden_dim=hidden_dim, dropout=dropout)
                    if kind == "swiglu":
                        return SwiGLUFFN(embed_dim, hidden_dim=hidden_dim, dropout=dropout, bias=False)
                    raise ValueError(f"unknown ffn kind: {kind}")


                def make_norm(kind, embed_dim):
                    if kind == "layernorm":
                        return LayerNorm(embed_dim)
                    if kind == "rmsnorm":
                        return RMSNorm(embed_dim)
                    raise ValueError(f"unknown norm kind: {kind}")


                class AblationBlock(Block):
                    def __init__(
                        self,
                        embed_dim,
                        num_heads,
                        max_seq_len,
                        attention="mha",
                        norm="layernorm",
                        ffn="classic",
                        hidden_dim=None,
                        dropout=0.0,
                    ):
                        hidden_dim = hidden_dim or 4 * embed_dim
                        super().__init__([
                            Residual(Block([
                                make_norm(norm, embed_dim),
                                make_attention(attention, embed_dim, num_heads, max_seq_len, dropout),
                            ])),
                            Residual(Block([
                                make_norm(norm, embed_dim),
                                make_ffn(ffn, embed_dim, hidden_dim, dropout),
                            ])),
                        ])


                class TinyAblationLM(Block):
                    def __init__(
                        self,
                        vocab_size,
                        embed_dim=128,
                        num_heads=4,
                        num_layers=2,
                        max_seq_len=64,
                        attention="mha",
                        norm="layernorm",
                        ffn="classic",
                    ):
                        embedding = Embedding(vocab_size, embed_dim)
                        super().__init__([
                            embedding,
                            Repeat(
                                lambda: AblationBlock(
                                    embed_dim,
                                    num_heads,
                                    max_seq_len,
                                    attention=attention,
                                    norm=norm,
                                    ffn=ffn,
                                    hidden_dim=4 * embed_dim,
                                ),
                                num_layers,
                            ),
                            make_norm(norm, embed_dim),
                            OutputHead(embed_dim, vocab_size, tied_embedding=embedding),
                        ])
                """
            ),
            md("## One-Step Ablation Runner"),
            code(
                """
                class OneBatchLoader:
                    def __iter__(self):
                        yield fixed_batch

                    def __len__(self):
                        return 1


                def run_one_step(name, **model_kwargs):
                    torch.manual_seed(seed)
                    model = TinyAblationLM(tokenizer.vocab_size, max_seq_len=context_length, **model_kwargs)
                    params = sum(p.numel() for p in model.parameters())

                    trainer = Trainer(
                        model,
                        AdamW,
                        OneBatchLoader(),
                        device=device,
                        context_length=context_length,
                        learning_rate=1e-3,
                        weight_decay=0.1,
                        use_amp=device.startswith("cuda"),
                        use_warmup_cosine=False,
                    )
                    losses = trainer.train(epochs=1, max_steps=1, log_interval=1)
                    return {"name": name, "params": params, "loss": losses[-1]}
                """
            ),
            md("## Compare Two Blocks"),
            code(
                """
                results = [
                    run_one_step(
                        "MHA + LayerNorm + ClassicFFN",
                        attention="mha",
                        norm="layernorm",
                        ffn="classic",
                    ),
                    run_one_step(
                        "GQA + RMSNorm + SwiGLU",
                        attention="gqa",
                        norm="rmsnorm",
                        ffn="swiglu",
                    ),
                ]

                for result in results:
                    print(f"{result['name']}: params={result['params']:,}, one-step loss={result['loss']:.4f}")
                """
            ),
            md(
                """
                ## What To Try Next

                - Keep the same training loop and change only `make_attention`.
                - Add QK normalization to `GroupedQueryAttention`.
                - Increase `num_layers` and run more than one step.
                - Move the same custom block into a reusable model file.
                """
            ),
        ],
    )


def autotrainer_distributed() -> dict:
    return notebook(
        "04_autotrainer_distributed_colab.ipynb",
        [
            md(
                """
                # AutoTrainer And Single-Node Multi-GPU Training

                <a href="https://colab.research.google.com/github/openlanguagemodel/openlanguagemodel/blob/main/notebooks/04_autotrainer_distributed_colab.ipynb" target="_blank">
                  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
                </a>

                This notebook shows how `AutoTrainer` chooses CPU, single-GPU, DDP,
                or FSDP paths. It runs safely on a normal Colab runtime, while also
                showing the exact single-node multi-GPU launch pattern.

                Multi-node training is a v4 roadmap item; this notebook is about v2
                single-machine scaling.
                """
            ),
            md("## Install OLM"),
            code(INSTALL_CELL),
            md("## Imports And A Tiny Training Setup"),
            code(
                """
                import random
                from pathlib import Path

                import torch

                from olm.data.datasets import DataLoader, LocalTextDataset
                from olm.data.tokenization import HFTokenizer
                from olm.nn.blocks import LM
                from olm.train import AutoTrainer
                from olm.train.device import (
                    TrainerStrategy,
                    determine_strategy,
                    detect_devices,
                    parse_device_string,
                    print_strategy_summary,
                )
                from olm.train.optim import AdamW

                seed = 123
                random.seed(seed)
                torch.manual_seed(seed)

                context_length = 128
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print("torch cuda available:", torch.cuda.is_available())
                print("torch cuda device count:", torch.cuda.device_count())

                data_dir = Path("autotrainer_data")
                data_dir.mkdir(exist_ok=True)
                (data_dir / "train.txt").write_text(
                    "\\n".join(
                        "AutoTrainer keeps the model readable while choosing the training path from the hardware."
                        for _ in range(500)
                    ),
                    encoding="utf-8",
                )

                tokenizer = HFTokenizer("gpt2")
                dataset = LocalTextDataset(data_dir, tokenizer, context_length=context_length, shuffle=True, seed=seed)
                loader = DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=device.startswith("cuda"))

                model = LM(
                    tokenizer.vocab_size,
                    embed_dim=128,
                    num_heads=4,
                    num_layers=2,
                    max_seq_len=context_length,
                    dropout=0.0,
                )
                """
            ),
            md("## Inspect Hardware And Strategy"),
            code(
                """
                hardware = detect_devices()
                print(hardware)

                config = parse_device_string("auto", model=model)
                config = determine_strategy(config, model=model, preset="balanced")
                print_strategy_summary(config)
                """
            ),
            md("## Let AutoTrainer Configure The Run"),
            code(
                """
                trainer = AutoTrainer(
                    model,
                    AdamW,
                    loader,
                    device="auto",
                    context_length=context_length,
                    learning_rate=3e-4,
                    weight_decay=0.1,
                    grad_accum_steps=2,
                    use_amp=device.startswith("cuda"),
                    preset="balanced",
                    verbose=True,
                )

                print("selected trainer:", type(trainer).__name__)
                losses = trainer.train(epochs=1, max_steps=3, log_interval=1)
                print("losses:", losses)
                """
            ),
            md(
                """
                ## Single-Node Multi-GPU Pattern

                In a notebook you normally have one Python process, so this section is
                explanatory. For multiple GPUs on one machine, put the training code
                in a script and launch it with `torchrun`.
                """
            ),
            code(
                """
                print("Single machine, 4 GPUs:")
                print("torchrun --nproc_per_node=4 train.py")
                print()
                print("Inside train.py, keep the same AutoTrainer call:")
                print('trainer = AutoTrainer(model, AdamW, loader, device="auto", context_length=1024)')
                """
            ),
            md("## Forcing A Strategy"),
            code(
                """
                # Do not run this block unless the notebook was launched with torchrun
                # on a machine that has multiple GPUs.
                RUN_FORCED_MULTI_GPU_EXAMPLE = False

                if RUN_FORCED_MULTI_GPU_EXAMPLE:
                    trainer = AutoTrainer(
                        model,
                        AdamW,
                        loader,
                        device="auto",
                        context_length=context_length,
                        force_strategy=TrainerStrategy.MULTI_GPU_DDP,
                        verbose=True,
                    )
                    trainer.train(epochs=1, max_steps=3)
                else:
                    print("Skipped forced DDP example. Use torchrun on a multi-GPU machine to run it.")
                """
            ),
            md(
                """
                ## Mental Model

                - CPU or one GPU: `AutoTrainer` returns the base `Trainer`.
                - Multiple GPUs on one machine: it can select `DDPTrainer` or
                  `FSDPTrainer`.
                - DDP is usually simpler and faster when the model fits on each GPU.
                - FSDP helps when model states need to be sharded across GPUs.
                - Multi-node launch/configuration is intentionally reserved for v4.
                """
            ),
        ],
    )


NOTEBOOKS = {
    "01_first_language_model_colab.ipynb": first_language_model(),
    "02_train_125m_fineweb_edu_colab.ipynb": fineweb_125m(),
    "03_custom_architecture_ablation_colab.ipynb": custom_architecture_ablation(),
    "04_autotrainer_distributed_colab.ipynb": autotrainer_distributed(),
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    for filename, nb in NOTEBOOKS.items():
        path = NOTEBOOK_DIR / filename
        path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
