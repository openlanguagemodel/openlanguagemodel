# Tutorials

Hands-on, end-to-end walkthroughs that take you from a blank file to a working result. Each tutorial is self-contained and can be copied and run as-is. If you are new to OLM, work through them in order.

1. **[Your First Language Model](first-model.md)** — build, train, and sample from a small GPT-style model on Tiny Shakespeare. The fundamentals of the library, in one script. *(~10 minutes, runs on CPU.)*

2. **[Modern Language Modelling](modern-language-modelling.md)** — see how the recipe moved from GPT-2 to Llama/Qwen-style models: RoPE, RMSNorm, SwiGLU, GQA, and MoE. *(~10 minutes.)*

3. **[Custom Architectures](custom-architecture.md)** — use blocks and combinators to design your own architecture and package it as a reusable model class. *(~15 minutes, runs on CPU.)*

4. **[Distributed Training](distributed-training.md)** — scale a real training run across multiple GPUs with DDP and FSDP using `torchrun`. *(~15 minutes, requires multiple GPUs.)*

5. **[Experiment Tracking](experiment-tracking.md)** — log metrics, gradients, and checkpoints, and run hyperparameter sweeps with Weights & Biases. *(~10 minutes.)*

## Prerequisites

All tutorials assume you have [installed OLM](../getting-started.md#installation) and have a working PyTorch. The first two run on a laptop CPU; the distributed tutorial requires more than one GPU.

For complete, runnable projects — including a full GPT-2 pretraining run on FineWeb-Edu — see the [`examples/`](https://github.com/openlanguagemodel/openlanguagemodel/tree/main/examples) directory in the repository.
