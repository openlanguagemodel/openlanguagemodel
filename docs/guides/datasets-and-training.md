# Datasets & Training

This guide covers OLM's training stack end to end: where data comes from, how it is batched, how the `Trainer` runs the loop, and how to customize every part with callbacks, schedulers, checkpoints, and distributed backends.

OLM is built to train on corpora larger than memory. Datasets **stream** text — reading and tokenizing it on demand — so the working set never has to fit in RAM.

## Datasets

Datasets are streaming `IterableDataset`s. Each tokenizes text, buffers the tokens, and yields fixed-length `(input_ids, labels)` pairs ready for next-token prediction, with multi-worker sharding handled automatically. There are three built-in sources:

| Dataset | Source |
|---------|--------|
| [`LocalTextDataset`](../api/data.md#localtextdataset) | A folder of your own `.txt` files |
| [`HuggingFaceTextDataset`](../api/data.md#huggingfacetextdataset) | Any text dataset on the Hugging Face Hub |
| [`FineWebEduDataset`](../api/data.md#finewebedudataset) | The FineWeb-Edu corpus, preconfigured |

```python
from olm.data.tokenization import HFTokenizer
from olm.data.datasets import LocalTextDataset, HuggingFaceTextDataset, FineWebEduDataset

tok = HFTokenizer("gpt2")

# A folder of .txt files
dataset = LocalTextDataset("./my_text_folder", tokenizer=tok, context_length=1024, shuffle=True)

# Any Hugging Face dataset — text_fn selects the text field from each row
dataset = HuggingFaceTextDataset(
    dataset_name="wikitext",
    split="train",
    context_length=1024,
    text_fn=lambda example: example["text"],
    tokenizer=tok,
    dataset_kwargs={"name": "wikitext-103-raw-v1"},
)

# The FineWeb-Edu shortcut
dataset = FineWebEduDataset(tok, subset="sample-10BT", context_length=2048)
```

> [!TIP]
> Set `shuffle=True` to randomize ordering. For local files OLM shuffles the file order; for streaming datasets it shuffles a rolling buffer (`shuffle_buffer_size`, default 10,000). With multiple workers or GPUs, each worker is assigned a disjoint slice of the stream, so no example is seen twice.

## The DataLoader

The [`DataLoader`](../api/data.md#dataloader) is a thin wrapper over PyTorch's, with defaults tuned for language-model training: pinned memory, persistent workers, and one-flag distributed sampling.

```python
from olm.data.datasets import DataLoader

loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

It sets `pin_memory=True` (faster host-to-GPU copies) and `persistent_workers=True` when `num_workers > 0` (no per-epoch worker restart). Pass `distributed=True` to install a `DistributedSampler` for multi-GPU runs.

## The Trainer

The [`Trainer`](../api/train.md#trainer) runs the loop. At minimum it needs a model, an optimizer, a dataloader, a device, and a context length:

```python
import torch
from olm.train import Trainer

trainer = Trainer(
    model,
    torch.optim.AdamW(model.parameters(), lr=3e-4),
    loader,
    device="cuda",
    context_length=1024,
)
losses = trainer.train(epochs=1, log_interval=10)
```

### Letting the Trainer build the optimizer

Pass an optimizer **class** instead of an instance and the Trainer groups parameters for you — applying weight decay to weight matrices (2-D and larger tensors) but not to biases or normalization parameters, a standard and beneficial default.

```python
from olm.train.optim import AdamW

trainer = Trainer(
    model, AdamW, loader,                # the class, not an instance
    device="cuda", context_length=1024,
    learning_rate=3e-4,
    weight_decay=0.1,                    # applied only to 2-D+ parameters
)
```

### Scheduling, warmup, and decay

By default the Trainer constructs a **warmup + cosine** schedule: the learning rate ramps up linearly, then decays along a cosine curve. Warmup defaults to roughly 10% of total steps. Override any of it, or supply your own scheduler via `scheduler=`, or disable the default with `use_warmup_cosine=False`.

```python
trainer = Trainer(
    model, AdamW, loader, device="cuda", context_length=1024,
    warmup_steps=2000,
    total_steps=100_000,
    min_lr=1e-5,
)
```

The built-in schedulers (`WarmupCosineScheduler`, `CosineAnnealingLR`, `WarmupLR`, `LinearLR`, `LinearDecayLR`) are documented in the [API reference](../api/train.md#schedulers).

### Performance features

These are enabled by default (except gradient clipping) and fully overridable:

| Feature | Argument | Effect |
|---------|----------|--------|
| Mixed precision (AMP) | `use_amp=True` | bf16/fp16 compute for speedups on modern GPUs |
| Gradient accumulation | `grad_accum_steps=8` | Simulate a larger batch than fits in memory |
| Gradient clipping | `grad_clip_norm=1.0` | Cap the gradient norm to prevent divergence |

```python
trainer = Trainer(
    model, AdamW, loader, device="cuda", context_length=1024,
    grad_accum_steps=8,
    use_amp=True,
    grad_clip_norm=1.0,
)
```

The effective batch size is `batch_size × grad_accum_steps × num_gpus`. Tune `grad_accum_steps` to reach a target batch size when GPU memory is the limit.

## Bring your own training loop

The `Trainer` is a convenience, not a runtime requirement. OLM models and
components are plain `torch.nn.Module`s, so a hand-written loop works exactly as
you would expect:

```python
import torch
import torch.nn.functional as F

model.to(device)
model.train()

for input_ids, labels in loader:
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    logits = model(input_ids)               # [batch, seq, vocab]
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

This is useful for research code where you want full control over loss terms,
gradient surgery, custom distributed logic, or evaluation inside the step.

## Callbacks

Callbacks inject behavior at defined points in the loop — `on_train_begin`, `on_step_end`, `on_epoch_end`, and others. Pass them with `callbacks=[...]`.

```python
from olm.train import Trainer, CheckpointCallback, ValidationCallback, EarlyStoppingCallback

trainer = Trainer(
    model, AdamW, loader, device="cuda", context_length=1024,
    callbacks=[
        CheckpointCallback("checkpoints", save_every=1000, keep_last_n=5),
        ValidationCallback(val_loader, eval_every=500, device="cuda"),
        EarlyStoppingCallback(patience=5),
    ],
)
```

Built-in callbacks include [`CheckpointCallback`](../api/train.md#checkpointcallback), [`ValidationCallback`](../api/train.md#validationcallback), [`EarlyStoppingCallback`](../api/train.md#earlystoppingcallback), `MetricsLoggerCallback`, `ThroughputCallback`, and `LRMonitorCallback`.

To write your own, subclass [`TrainerCallback`](../api/train.md#trainercallback) and override the hooks you need:

```python
from olm.train import TrainerCallback

class StepLogger(TrainerCallback):
    def __init__(self, every=500):
        self.every = every

    def on_step_end(self, trainer, step, loss):
        if step % self.every == 0:
            print(f"step {step}: loss={loss:.4f}")

trainer = Trainer(..., callbacks=[StepLogger(500)])
```

The full hook surface — `on_train_begin/end`, `on_epoch_begin/end`, `on_batch_begin/end`, `on_step_begin/end` — is documented on [`TrainerCallback`](../api/train.md#trainers).

## Saving and loading

Models built on the `Block` system have a `.save()` method that serializes the entire model, so you never reconstruct the architecture by hand when loading.

```python
model.save("./checkpoints/final_model", tokenizer=tok)
```

```python
from olm.nn.structure import load_model

# If a tokenizer was saved alongside the model, both are returned
model, tok = load_model("./checkpoints/final_model")

# Otherwise just the model
model = load_model("./checkpoints/model_only")
```

> [!NOTE]
> Because `.save()` stores the whole module, loading reconstructs the exact architecture — there is no need to remember `vocab_size`, `num_layers`, or which attention variant you used.
> Re-running `model.save("./checkpoints/final_model")` updates the saved model in
> that directory instead of failing just because the directory already exists.

### Resuming trainer checkpoints

`CheckpointCallback` writes training-state checkpoints: model weights, optimizer
state, AMP scaler state, scheduler state, epoch, step, and logged losses. These
`.pt` files are different from a `model.save()` directory, so load them with
`torch.load` and restore each state explicitly:

```python
import torch

checkpoint = torch.load("checkpoints/step_5000.pt", map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

if "scheduler_state_dict" in checkpoint and trainer.scheduler is not None:
    trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

if "scaler_state_dict" in checkpoint:
    trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])

trainer.global_step = checkpoint.get("step", 0)
trainer.current_epoch = checkpoint.get("epoch", 0)
trainer.losses = checkpoint.get("losses", [])
```

For streaming datasets, resume the data position too. OLM datasets accept
`skip_batches`, which currently skips yielded training sequences before the
`DataLoader` groups them into batches. A simple approximation is therefore:

```python
resume_step = checkpoint.get("step", 0)
dataset = FineWebEduDataset(tok, context_length=1024, skip_batches=resume_step * batch_size)
loader = DataLoader(dataset, batch_size=batch_size)
```

The full FineWeb-Edu example includes a complete resume script:
[`examples/gpt2-fineweb-edu-10b/train.py`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/examples/gpt2-fineweb-edu-10b/train.py).

## Distributed training

For large models or higher throughput, scale across GPUs with PyTorch's native backends. OLM provides two drop-in trainers that subclass `Trainer`:

- **`DDPTrainer`** — replicates the model on each GPU and synchronizes gradients. Best when the model fits on one GPU.
- **`FSDPTrainer`** — fully sharded data parallel; splits parameters across GPUs to train models larger than a single GPU's memory.

Both launch with `torchrun`:

```bash
# Single machine, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Two machines, 4 GPUs each (run on each node, changing --node_rank)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 train.py
```

### DDP

```python
from olm.core.dist import setup_distributed, get_local_rank
from olm.train.trainer import DDPTrainer
from olm.data.datasets import DataLoader

setup_distributed()    # reads RANK / WORLD_SIZE / LOCAL_RANK from torchrun

loader = DataLoader(dataset, batch_size=16, num_workers=4, distributed=True)

trainer = DDPTrainer(
    model, torch.optim.AdamW, loader,
    device=f"cuda:{get_local_rank()}",
    context_length=1024,
    learning_rate=3e-4,
    grad_accum_steps=4,
)
trainer.train(epochs=10, log_interval=100)   # metrics are aggregated across ranks
```

With DDP, use `distributed=True` on the `DataLoader` so each GPU sees different data; only rank 0 logs and writes checkpoints (handled automatically).

### FSDP

```python
from olm.train.trainer import FSDPTrainer
from olm.core.dist import setup_distributed, get_local_rank

setup_distributed()

trainer = FSDPTrainer(
    model, torch.optim.AdamW,
    DataLoader(dataset, batch_size=8, distributed=True),
    device=f"cuda:{get_local_rank()}",
    context_length=2048,
    learning_rate=3e-4,
    sharding_strategy="FULL_SHARD",      # most memory-efficient
    auto_wrap_policy="size",             # wrap layers above min_num_params
    min_num_params=int(1e8),
    mixed_precision_policy="bf16",       # Ampere or newer
    cpu_offload=False,
)
trainer.train(epochs=10)
```

Choosing between them:

| Scenario | Recommendation |
|----------|----------------|
| Model fits on one GPU | DDP — simpler and faster |
| Model exceeds one GPU | FSDP with `FULL_SHARD` |
| Multi-node | FSDP with `HYBRID_SHARD` |
| Maximum throughput | DDP, or FSDP with `SHARD_GRAD_OP` |
| Maximum model size | FSDP with `cpu_offload=True` |

Save an FSDP checkpoint by gathering the full model on rank 0:

```python
trainer.save_checkpoint("./checkpoints/model.pt", state_dict_type="FULL_STATE_DICT")
```

### AutoTrainer

Use `AutoTrainer` when you want OLM to pick the training path from the hardware
in front of it. It detects CPU/GPU availability, GPU count, approximate model
size, distributed environment variables, and then returns the appropriate
trainer: plain `Trainer`, `DDPTrainer`, or `FSDPTrainer`.

```python
from olm.train import AutoTrainer
from olm.train.optim import AdamW

trainer = AutoTrainer(
    model,
    AdamW,
    loader,
    device="auto",
    context_length=2048,
    learning_rate=3e-4,
)
trainer.train(epochs=1)
```

Presets tune the automatic choice:

| Preset | Bias |
|--------|------|
| `balanced` | Default hardware-aware selection |
| `memory_efficient` | Prefer FSDP and memory-saving settings |
| `speed` | Prefer throughput when the model fits comfortably |

For explicit control, pass a `DeviceConfig` or force a strategy:

```python
from olm.train import TrainerStrategy

trainer = AutoTrainer(
    model,
    AdamW,
    loader,
    device="auto",
    force_strategy=TrainerStrategy.MULTI_GPU_DDP,
    context_length=2048,
)
```

`AutoTrainer` also works under `torchrun`; `device="auto"` will use the
distributed environment that `torchrun` provides.

## Experiment tracking

OLM includes Weights & Biases integration via a single callback that logs loss, perplexity, learning rate, throughput, and system metrics:

```python
from olm.logging import WandBCallback

trainer = Trainer(..., callbacks=[WandBCallback(project="my-llm", name="run-1")])
```

See the [Experiment Tracking tutorial](../tutorials/experiment-tracking.md) for gradient logging, checkpoint artifacts, alerts, and hyperparameter sweeps.

## Next steps

- [Tutorial: Your First Language Model](../tutorials/first-model.md) — a guided run with text generation.
- [Tutorial: Distributed Training](../tutorials/distributed-training.md) — a complete multi-GPU script.
- [`olm.train` API reference](../api/train.md) — every trainer, callback, optimizer, scheduler, and loss.
