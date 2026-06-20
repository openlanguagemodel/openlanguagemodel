# Distributed Training

When one GPU is not enough — for speed, or because the model does not fit — OLM scales out with PyTorch's native DDP and FSDP backends. This tutorial walks through a complete multi-GPU script and explains the moving parts.

It requires more than one GPU (on one or several machines) and a working NCCL/CUDA setup. The same concepts work on CPU with the `gloo` backend for testing.

## DDP vs FSDP

- **DDP** ([`DDPTrainer`](../api/train.md#ddptrainer)) replicates the full model on every GPU and averages gradients each step. Use it when the model fits on a single GPU.
- **FSDP** ([`FSDPTrainer`](../api/train.md#fsdptrainer)) shards parameters, gradients, and optimizer state across GPUs. Use it when the model is too large for one GPU.

Both are drop-in subclasses of [`Trainer`](../api/train.md#trainer) — the same constructor, plus a few backend options.

## A complete DDP script

Save this as `train_ddp.py`:

```python
import torch
from olm.core.dist import setup_distributed, cleanup_distributed, get_local_rank
from olm.train.trainer import DDPTrainer
from olm.data.tokenization import HFTokenizer
from olm.data.datasets import FineWebEduDataset, DataLoader
from olm.models import GPT2

def main():
    # 1. Initialize the process group (reads env vars set by torchrun)
    setup_distributed()
    device = f"cuda:{get_local_rank()}"

    # 2. Model and data
    tok = HFTokenizer("gpt2")
    model = GPT2()

    dataset = FineWebEduDataset(tok, context_length=1024, subset="sample-10BT")
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
        distributed=True,     # each rank receives a different data shard
    )

    # 3. Trainer — same API as single-GPU, just DDPTrainer
    trainer = DDPTrainer(
        model,
        torch.optim.AdamW,
        loader,
        device=device,
        context_length=1024,
        grad_accum_steps=4,
        learning_rate=6e-4,
        weight_decay=0.1,
        grad_clip_norm=1.0,
    )
    trainer.train(epochs=1, max_steps=10_000, log_interval=50)

    cleanup_distributed()

if __name__ == "__main__":
    main()
```

Launch it with `torchrun`:

```bash
# Single machine, 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py
```

`DDPTrainer` wraps the model in `DistributedDataParallel`, uses `no_sync()` during gradient accumulation to avoid unnecessary communication, aggregates metrics across ranks, and logs and checkpoints only on rank 0.

### Multi-node

Run the same script on each machine, changing only `--node_rank` and pointing every node at the rank-0 address:

```bash
# Node 0 (the master)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=10.0.0.1 --master_port=29500 train_ddp.py

# Node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr=10.0.0.1 --master_port=29500 train_ddp.py
```

> [!TIP]
> With DDP, the global batch is `batch_size × num_gpus × grad_accum_steps`. The example above is `16 × 4 × 4 = 256` sequences, roughly 262K tokens per step. Scale your learning rate and warmup with the global batch, not the per-GPU batch.

## Switching to FSDP

For models too large to replicate, change two things: use `FSDPTrainer` and add sharding options. Everything else is identical.

```python
from olm.train.trainer import FSDPTrainer

trainer = FSDPTrainer(
    model,
    torch.optim.AdamW,
    loader,
    device=device,
    context_length=2048,
    learning_rate=3e-4,
    sharding_strategy="FULL_SHARD",     # shard params, grads, and optimizer state
    auto_wrap_policy="size",            # auto-wrap layers above the threshold
    min_num_params=int(1e8),            # 100M-parameter wrap threshold
    mixed_precision_policy="bf16",      # bf16 compute on Ampere or newer
    cpu_offload=False,                  # set True to trade speed for memory
)
trainer.train(epochs=1, max_steps=10_000)
```

Choosing options:

| Goal | Setting |
|------|---------|
| Maximum memory savings | `sharding_strategy="FULL_SHARD"`, `cpu_offload=True` |
| Faster, less aggressive sharding | `sharding_strategy="SHARD_GRAD_OP"` |
| Multi-node large models | `sharding_strategy="HYBRID_SHARD"` |
| Wrap by layer type | `auto_wrap_policy="transformer"`, `transformer_layer_cls=...` |

### Saving FSDP checkpoints

Gather the full (unsharded) model onto rank 0 for a portable checkpoint:

```python
trainer.save_checkpoint("checkpoints/model.pt", state_dict_type="FULL_STATE_DICT")
```

Or save a sharded checkpoint — each rank writes its own shard — for faster, memory-light saves with very large models:

```python
trainer.save_checkpoint("checkpoints/model_sharded", state_dict_type="SHARDED_STATE_DICT")
```

## Common pitfalls

> [!WARNING]
> Always pass `distributed=True` to the `DataLoader`. Without it, every GPU trains on the same data, wasting the cluster. The flag installs a `DistributedSampler` so each rank sees a disjoint shard.

When comparing a distributed run to a single-GPU baseline, compare *global* batch sizes: logged throughput is summed across ranks, and loss is averaged across ranks.

## Next steps

- Add [experiment tracking](experiment-tracking.md) — `WandBCallback` logs from rank 0 only, automatically.
- Review the full [`DDPTrainer`](../api/train.md#trainers) and [`FSDPTrainer`](../api/train.md#trainers) options.
