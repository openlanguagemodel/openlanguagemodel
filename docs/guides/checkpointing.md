---
description: Save training checkpoints, resume interrupted runs exactly, keep the best model and stop early, with OLM's Trainer.
---
# Checkpointing & Resuming

A long training run gets interrupted: a preempted cloud instance, a Colab session that times out, a crash at step 40,000. OLM's `Trainer` writes checkpoints that hold the complete training state, and resuming from one continues the run where it stopped. On CPU a resumed run reproduces the uninterrupted run's losses and weights bit for bit; the tests check this for single-process and two-process DDP training.

OLM saves two kinds of files, for two jobs:

| File | Written by | Holds | Use it to |
|---|---|---|---|
| Training checkpoint (`.pt`) | `CheckpointCallback`, `trainer.save_checkpoint()` | model and optimizer state, schedule, step counters, random state, data position, callback state | resume training, or load weights |
| Model export (directory) | `model.save()` | the whole pickled model and optionally its tokenizer | use a finished model; see [Saving and loading](datasets-and-training.md#saving-and-loading) |

## Save checkpoints while training

Add a `CheckpointCallback`. The examples on this page train a small model on random tokens so they run on a laptop CPU in seconds; swap in your own model and data.

```python
import torch
from torch.utils.data import TensorDataset

from olm.core import seed_everything
from olm.data.datasets import DataLoader
from olm.nn.blocks import LM
from olm.train import AdamW, CheckpointCallback, Trainer


def make_loader(seed=0):
    tokens = torch.randint(0, 256, (64, 33), generator=torch.Generator().manual_seed(seed))
    dataset = TensorDataset(tokens[:, :-1], tokens[:, 1:])  # (input_ids, labels)
    return DataLoader(dataset, batch_size=8, pin_memory=False)


def make_trainer(resume_from=None, callbacks=None):
    seed_everything(0)
    model = LM(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=32)
    return Trainer(
        model,
        AdamW,
        make_loader(),
        device="cpu",
        context_length=32,
        grad_accum_steps=2,
        use_amp=False,
        callbacks=callbacks or [CheckpointCallback("checkpoints", save_every=2)],
        resume_from=resume_from,
    )


trainer = make_trainer()
trainer.train(epochs=2, log_interval=100)
# checkpoints/ now holds step_2.pt, step_4.pt, step_6.pt and step_8.pt
```

`CheckpointCallback` writes `step_<N>.pt` every `save_every` optimizer steps and once more when training ends, keeps the newest `keep_last_n` (5 by default), and writes `best_model.pt` when validation improves (see [below](#keep-the-best-model-and-stop-early)). Each file is written to a temporary name and renamed into place, so a crash during a save never leaves a truncated checkpoint behind.

You can also save at any point yourself:

```python
trainer.save_checkpoint("checkpoints/manual.pt")
```

## Resume a run

Pass the checkpoint to the new `Trainer` and call `train()` with **the same arguments as the interrupted run**. `epochs` and `max_steps` count from the start of the run, not from the resume point.

```python
resumed = make_trainer(resume_from="checkpoints/step_4.pt")
resumed.train(epochs=2, log_interval=100)

print(resumed.global_step)                # 8
print(resumed.losses == trainer.losses)   # True: the same eight losses
```

The resumed trainer starts in the saved epoch, skips the batches that epoch had already used, and continues with the same learning rate, optimizer moments and random numbers.

For a job that can be restarted at any time, use `resume_from="latest"`. It loads the newest `step_<N>.pt` in the `CheckpointCallback` directory, and starts a new run when there is none yet, so the same script works for the first launch and every restart:

```python
trainer = make_trainer(resume_from="latest")
trainer.train(epochs=2, log_interval=100)
```

`resume_from` also accepts a directory (the newest `step_<N>.pt` in it) and works the same way on `DDPTrainer` and `AutoTrainer`. To resume an existing `Trainer` object, call `trainer.load_checkpoint(path)` before `train()`.

### What is restored

| State | Restored | Notes |
|---|---|---|
| Model weights | yes | Loaded into the unwrapped model, so DDP and `torch.compile` prefixes never matter. |
| Optimizer state | yes | AdamW moments, step counts and every parameter group's learning rate. |
| Learning-rate schedule | yes | Including the default warmup + cosine schedule, which is rebuilt with its saved length before the optimizer state is loaded. |
| AMP `GradScaler` | yes | When scaling was on (fp16 on CUDA). |
| Step counters | yes | `global_step`, `current_epoch`, steps and batches used in the epoch, tokens processed. |
| Loss history | yes | `trainer.losses` holds every step; `train()` returns only the new ones. |
| Random state | yes | Python, NumPy, torch CPU and CUDA generators, for every rank, and the dataloader's own `generator`. |
| Data position | yes | The saved epoch restarts its iterator with the random state it had and skips the batches already used. |
| Callback state | yes | Best validation loss, early-stopping patience, the best checkpoint's value. Custom callbacks opt in with `state_dict()` / `load_state_dict()`. |
| An early stop | yes | A run that early stopping ended stays ended. |
| `trainer.training_state` metrics | no | `val_loss` and similar values reappear at the next validation. |
| Weights & Biases run | no | Use `WandBCallback(resume=...)` for that. |

`on_epoch_begin` callbacks run again for the resumed epoch, and `MetricsLoggerCallback` starts a new file.

### When a resume is exact

A resumed run is the same run, step for step, when:

- the data, batch size, `grad_accum_steps`, number of GPUs and `num_workers` are unchanged (the Trainer warns when gradient accumulation or the number of processes differs);
- the dataset yields the same order when its random state is the same. Shuffled map-style datasets, iterable datasets that read in a fixed order, and OLM's text datasets with a fixed `seed` all do;
- the checkpoint was saved at the end of an optimizer step, which is where `CheckpointCallback` saves;
- no callback draws random numbers in `on_epoch_begin` or `on_epoch_end` (the built-in callbacks draw none). A checkpoint from the last step of an epoch restores the random state of that step, so such draws would not be repeated.

Validation does not affect exactness: `ValidationCallback` restores every random generator after each validation, whatever its `DataLoader` settings.

On CPU that gives bit-identical losses and weights. On GPUs, some kernels are not deterministic, so a resumed run matches the uninterrupted one only as closely as two uninterrupted runs match each other; see [Reproducibility](reproducibility.md).

With persistent `DataLoader` workers (the default when `num_workers > 0`), random numbers that a dataset draws *inside* its worker processes are not restored, because the workers of the new process are seeded afresh. OLM's text datasets draw none there.

### How the data position is restored

The Trainer counts the batches each epoch has used. On resume it creates the epoch's iterator with the random state it had when the interrupted run created it, so a shuffled dataset produces the same order, and reads past the batches that were already used before training continues.

Skipping means reading: the skipped batches are loaded and thrown away. For a map-style dataset in memory that is quick. For a streaming dataset, the skipped text is downloaded and tokenized again, so resuming late in a long epoch can take minutes to hours before the first step. The Trainer prints how many batches it skips.

Do not also set `skip_batches` on an OLM dataset when you resume through the Trainer; the data would be skipped twice. `skip_batches` is for hand-written training loops. It counts **samples in each DataLoader worker's shard**, and every optimizer step uses `batch_size × grad_accum_steps` samples on each process, taken in turn from its `W = max(num_workers, 1)` workers. After `N` optimizer steps, skip `N × batch_size × grad_accum_steps / W` samples in each shard; this is exact when `N × grad_accum_steps` is a multiple of `W`. The formula `N × batch_size` that earlier versions of these docs used ignores gradient accumulation and workers, and replays data the model has already seen.

## What a checkpoint contains

A checkpoint is one `torch.save` file that loads with `torch.load(weights_only=True)`, which cannot run code stored in the file.

| Key | Contents |
|---|---|
| `format_version` | `1` |
| `olm_version`, `torch_version` | versions that wrote the file |
| `model` | model state dict, without `module.` or `_orig_mod.` prefixes |
| `optimizer` | optimizer state dict (`None` in FSDP checkpoints) |
| `scheduler`, `scheduler_class` | scheduler state dict and class name, or `None` |
| `scaler` | `GradScaler` state (empty when scaling is off) |
| `trainer` | `global_step`, `epoch`, `epoch_step`, `batches_in_epoch`, `total_tokens_processed`, `grad_accum_steps`, `world_size`, `total_steps`, `warmup_steps` and the flags the data position needs |
| `rng` | one entry per rank: the random state at save time and at the start of the epoch |
| `callbacks` | state of each callback that has some, keyed by class name |
| `losses` | the loss of every step |

The keys of earlier OLM versions, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `step` and `epoch`, are kept as aliases of the same objects (they take no extra space), so scripts that read them keep working.

## Load weights without resuming

To evaluate a checkpoint, or to start a new run (with a fresh optimizer and schedule) from its weights, load only the model:

```python
from olm.train import read_checkpoint

checkpoint = read_checkpoint("checkpoints/step_8.pt")
model = LM(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=32)
model.load_state_dict(checkpoint["model"])

print(checkpoint["format_version"], checkpoint["trainer"]["global_step"])  # 1 8
```

`read_checkpoint` returns every file in the current layout. It also reads checkpoints written by earlier OLM versions: `CheckpointCallback` and `WandBCallback` files with `model_state_dict`, and `FSDPTrainer` files with `model` and `global_step`. It strips `module.` and `_orig_mod.` prefixes from their model keys.

> [!WARNING]
> `read_checkpoint` and `load_checkpoint` pass `weights_only=True` to `torch.load`. Pass `weights_only=False` only for files you trust: it lets the file run arbitrary code while loading. You would need it only for a custom scheduler or callback that stores Python objects other than numbers, strings, lists, dicts and tensors.

## Keep the best model and stop early

`ValidationCallback` measures the validation loss every `eval_every` steps. `CheckpointCallback` saves `best_model.pt` whenever it improves, and `EarlyStoppingCallback` stops training after `patience` validations in a row without an improvement larger than `min_delta`:

```python
from olm.train import EarlyStoppingCallback, ValidationCallback

trainer = make_trainer(
    callbacks=[
        CheckpointCallback("best_run", save_every=100),
        ValidationCallback(make_loader(seed=1), eval_every=2),
        EarlyStoppingCallback(patience=3, min_delta=0.01),
    ]
)
trainer.train(epochs=4, log_interval=100)

best = read_checkpoint("best_run/best_model.pt")
print("best validation at step", best["trainer"]["global_step"])
```

- **Patience counts validations**, not optimizer steps. With `eval_every=500` and `patience=5`, training stops after 2,500 steps without improvement.
- Callbacks run in a fixed order whatever their position in the list: `ValidationCallback` first, then early stopping and your callbacks, then `CheckpointCallback`. So `best_model.pt` always holds the weights of the step that was validated.
- `monitor` and `mode` choose the metric: `EarlyStoppingCallback(monitor="val_accuracy", mode="max")` for a metric where higher is better. Both callbacks read it from `trainer.training_state`.
- A custom validation callback must write the step it ran at as `trainer.training_state["val_step"]` next to its metric. That is how the other callbacks tell a new validation from an old value.
- The best value and the patience count are part of the checkpoint, so a resumed run does not overwrite a better `best_model.pt` with its first validation.

## Distributed training

With `DDPTrainer`, every rank must take part in a save because the checkpoint collects each rank's random state; `CheckpointCallback` does this for you, and only rank 0 writes the file. To resume, give every rank the same `resume_from`. This script runs as one process too, where `DDPTrainer` trains like `Trainer`:

```python
from olm.core.dist import get_local_rank, setup_distributed
from olm.train import DDPTrainer

setup_distributed()  # reads RANK and WORLD_SIZE from torchrun; does nothing without them

seed_everything(0)
model = LM(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=32)
tokens = torch.randint(0, 256, (64, 33), generator=torch.Generator().manual_seed(0))
loader = DataLoader(
    TensorDataset(tokens[:, :-1], tokens[:, 1:]),
    batch_size=8,
    pin_memory=False,
    distributed=True,  # a DistributedSampler when a process group exists
)
device = f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu"

trainer = DDPTrainer(
    model,
    AdamW,
    loader,
    device=device,
    context_length=32,
    use_amp=False,
    callbacks=[CheckpointCallback("ddp_checkpoints", save_every=2)],
    resume_from="latest",
)
trainer.train(epochs=2, log_interval=100)
```

Launch it with `torchrun --nproc_per_node=2 train.py`, and relaunch the same command after an interruption. The number of processes must stay the same for the data position to be exact.

Early stopping is decided on every rank together: validation losses are averaged over ranks, and when any rank's callbacks ask to stop, all ranks stop, so no rank waits forever in a collective the others have left.

### FSDP

`FSDPTrainer` does not resume a sharded run yet:

- `save_checkpoint(path)` (and `CheckpointCallback`) gathers the full model on rank 0 and writes a version 1 checkpoint **without optimizer state**, because the sharded optimizer state is not gathered.
- Creating an `FSDPTrainer` with `resume_from` pointing at a checkpoint, or calling `load_checkpoint`, raises `NotImplementedError`. `resume_from="latest"` with no checkpoint yet still starts a new run.
- To continue from saved weights with a fresh optimizer and schedule, load them into the model before building the trainer: `model.load_state_dict(read_checkpoint(path)["model"])`.
- `save_checkpoint(path, state_dict_type="LOCAL_STATE_DICT")` or `"SHARDED_STATE_DICT"` writes one file per rank in FSDP's own sharded layout, which `read_checkpoint` does not read.

## Checkpoints from earlier OLM versions

`load_checkpoint` and `resume_from` accept `CheckpointCallback` files written before format version 1. They hold no random state or data position, so the Trainer warns and resumes as closely as it can: in the first epoch it assumes every step used `grad_accum_steps` batches and skips that many; in later epochs it restarts the epoch from its first batch. The loss curve continues, but not bit for bit.

## See also

- [Reproducibility](reproducibility.md): seeding, deterministic mode, and why a fresh model's first loss is what it is.
- [Defaults Reference](../reference/defaults.md): every default of the Trainer and its callbacks.
- [`examples/gpt2-fineweb-edu-10b/train.py`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/examples/gpt2-fineweb-edu-10b/train.py): a full training script that resumes with `--resume`.
