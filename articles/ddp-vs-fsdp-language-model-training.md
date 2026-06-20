# DDP vs FSDP for Language Model Training

PyTorch gives you several ways to scale training. The two most common starting
points are Distributed Data Parallel (DDP) and Fully Sharded Data Parallel
(FSDP). OLM exposes both through trainer classes so you can scale a run without
changing the model definition.

## DDP

DDP replicates the model on each GPU and synchronizes gradients. It is usually
the simplest choice when the full model and optimizer state fit on one GPU.

Use DDP when:

- the model fits comfortably on each GPU
- you mainly want more throughput
- you want the simplest distributed mental model

OLM's `DDPTrainer` handles distributed wrapping, metric aggregation, rank-aware
logging, and checkpoint behavior.

## FSDP

FSDP shards parameters, gradients, and optimizer states across GPUs. It is the
better choice when model memory becomes the bottleneck.

Use FSDP when:

- the model is too large to replicate on every GPU
- optimizer state memory is limiting your batch size
- you want to train larger models on the same hardware pool

OLM's `FSDPTrainer` keeps the API close to the single-GPU trainer while using
PyTorch's native sharding support.

## What OLM Manages

Distributed training is more than model wrapping. OLM's training stack also
covers distributed sampling, AMP, gradient accumulation, schedules, callbacks,
metrics, checkpointing, and rank-aware logging.

Read the [Distributed Training](/docs/tutorials/distributed-training/) tutorial
for runnable `torchrun` examples, or the
[Datasets & Training](/docs/guides/datasets-and-training/) guide for the full
training stack.
