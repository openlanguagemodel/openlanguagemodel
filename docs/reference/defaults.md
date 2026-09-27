---
description: Every default of OLM's training stack in one place, taken from the code and checked by a test.
---
# Defaults Reference

This page lists every default of OLM's training stack: the trainers, callbacks, optimizers, schedulers, losses and datasets, plus the choices OLM makes that no signature shows. The values are copied from the source. `tests/test_defaults_reference.py` compares every argument table (the sections named after an import path, such as `olm.train.Trainer`) with the real signatures, so the test fails when a default changes and this page does not. The first two tables, which describe behaviour rather than arguments, are maintained by hand.

`required` means the argument has no default.

## Choices no signature shows

These are the defaults that surprise people, because you cannot see them by reading a constructor.

| What | Default | Details |
|---|---|---|
| Mixed-precision dtype | fp16 on CUDA, bf16 on CPU | `use_amp=True` wraps the forward pass in `torch.autocast` without a `dtype`, so PyTorch's per-device default applies. That is fp16 on every CUDA GPU, including ones that support bf16. |
| GradScaler | on when `use_amp=True` and the device is CUDA | PyTorch's defaults: initial scale 65536, growth factor 2.0, backoff factor 0.5, growth interval 2000 steps. It is not configurable. |
| Learning-rate schedule | `WarmupCosineScheduler` | Built inside `train()` when you pass no `scheduler` and `use_warmup_cosine=True`. Linear warmup to `learning_rate`, then a cosine decay to `min_lr`. |
| Schedule length (`total_steps`) | optimizer steps `train()` will take | `epochs × ceil(len(dataloader) / grad_accum_steps)`, where each epoch is capped at `steps_per_epoch` and the total at `max_steps`. A dataloader without a length (an iterable dataset without `__len__`) uses the smaller of `max_steps` and `epochs × steps_per_epoch`, or whichever of the two is given, and otherwise warns and assumes `epochs × 10000`. |
| Warmup length (`warmup_steps`) | about 10% of the schedule | `min(max(int(0.1 × total_steps), 1), 5000, total_steps)`. |
| Optimizer built from a class | `optimizer_cls(groups, lr=learning_rate)` | Parameters with 2 or more dimensions get `weight_decay`; biases and norm weights get 0.0; frozen parameters are left out. Every other hyperparameter comes from the class: AdamW's `betas=(0.9, 0.999)` and `eps=1e-8`. |
| Weight decay | 0.0 | The Trainer's `weight_decay=0.0` replaces the optimizer class's own default, such as AdamW's 0.01. It has no effect when you pass an optimizer instance. |
| Gradient clipping | off | `grad_clip_norm=None`. |
| Loss of a step | mean of the micro-batch losses | Each micro-batch loss is divided by the number of micro-batches in the step. The last group of an epoch may be smaller; its gradients are rescaled to a full group's average. |
| `context_length` | not applied | The Trainer stores it but does not truncate batches. Your dataset decides the sequence length. |
| Token count | `input_ids.numel()` per batch | It includes padding and positions labelled -100. |
| Displayed perplexity | `exp(min(loss, 20))` | A loss above 20 prints as 485165195.41. |
| Loss history | `trainer.losses` keeps every step | It survives a resume. `train()` returns only the losses of that call. |
| Callback order | ascending `order`, then list order | `ThroughputCallback` -20, `ValidationCallback` -10, other callbacks 0, `CheckpointCallback` 10. |
| Validation loss | `trainer.loss` averaged over batches | Averaged over ranks under DDP and FSDP. Auxiliary and MTP losses are not included. fp16 autocast on CUDA only. |
| Checkpoints | format version 1, `step_<N>.pt` and `best_model.pt` | See [Checkpointing & Resuming](../guides/checkpointing.md). |
| Seeding | none | Nothing is seeded unless you call [`seed_everything`](../guides/reproducibility.md). |
| Initial weights | PyTorch's defaults | `Embedding` keeps `nn.Embedding`'s N(0, 1), so a model with a tied output head starts at a loss of the order of `embed_dim` instead of ln(vocab_size). See [Reproducibility](../guides/reproducibility.md#why-a-fresh-model-starts-at-a-loss-of-about-lnvocab_size). |

### How `AutoTrainer` chooses a trainer

With `device="auto"` (or `"cuda:auto"`), `AutoTrainer` counts GPUs and estimates memory as `4.2 × parameters × 4 bytes` (fp32 weights, gradients and two Adam states, plus 20% for activations).

| GPUs | Preset | Result |
|---|---|---|
| 0 | any | `Trainer` on CPU, with `use_amp` forced to `False` |
| 1 | any | `Trainer` on `cuda`; warns when the estimate exceeds 80% of GPU memory |
| 2 or more | `balanced`, `conservative` | `DDPTrainer` when estimate ÷ GPUs < 60% of one GPU's memory, otherwise `FSDPTrainer` with `HYBRID_SHARD` (up to 4 GPUs) or `FULL_SHARD` |
| 2 or more | `memory_efficient` | `FSDPTrainer` with `HYBRID_SHARD` (up to 4 GPUs) or `FULL_SHARD`, and CPU offload |
| 2 or more | `speed` | `DDPTrainer` when the estimate is below 60% of one GPU's memory, otherwise `FSDPTrainer` with `HYBRID_SHARD` |

On GPUs the chosen mixed-precision setting is `bf16` for compute capability 8.0 or newer and `fp16` otherwise, but only `FSDPTrainer` reads it (as `mixed_precision_policy`). A multi-GPU strategy calls `setup_distributed()`, which does nothing when `WORLD_SIZE` is 1, so launch multi-GPU runs with `torchrun`.

## Trainers

### `olm.train.Trainer`

| Argument | Default | Notes |
|---|---|---|
| `model` | required | Moved to `device`. |
| `optimizer` | required | An optimizer instance, or a class that the Trainer builds (see above). |
| `dataloader` | required | Must yield `(input_ids, labels)`. |
| `device` | required | `"cpu"`, `"cuda"` or `"cuda:N"`. |
| `context_length` | required | Stored, not applied. |
| `grad_accum_steps` | `1` | Micro-batches per optimizer step. |
| `use_amp` | `True` | See mixed precision above. |
| `loss` | `CrossEntropyLoss` | A loss class or instance, called as `loss(logits, labels)`. |
| `mtp_loss` | `None` | Added when the model returns `mtp_logits`. |
| `callbacks` | `None` | No callbacks. |
| `scheduler` | `None` | Builds the default warmup + cosine schedule when `use_warmup_cosine=True`. |
| `grad_clip_norm` | `None` | No clipping. |
| `warmup_steps` | `None` | Derived from `total_steps`. |
| `total_steps` | `None` | Derived from the arguments of `train()`. |
| `min_lr` | `0.0` | End of the cosine decay. |
| `learning_rate` | `3e-4` | Used only when `optimizer` is a class. |
| `weight_decay` | `0.0` | Used only when `optimizer` is a class. |
| `use_warmup_cosine` | `True` | |
| `resume_from` | `None` | A checkpoint file, a directory, or `"latest"`. |

### `olm.train.Trainer.train`

| Argument | Default | Notes |
|---|---|---|
| `epochs` | required | Counted from the start of the run, also after a resume. |
| `log_interval` | `10` | Print every N optimizer steps. |
| `max_steps` | `None` | Counts optimizer steps, not micro-batches. |
| `steps_per_epoch` | `None` | Counts optimizer steps. |

### `olm.train.Trainer.load_checkpoint`

| Argument | Default | Notes |
|---|---|---|
| `path` | required | |
| `strict` | `True` | Passed to the model's `load_state_dict`. |
| `weights_only` | `True` | Passed to `torch.load`; keep it unless you trust the file. |

### `olm.train.DDPTrainer`

<!-- defaults-inherit: olm.train.Trainer -->
Takes every `Trainer` argument with the same default, plus:

| Argument | Default | Notes |
|---|---|---|
| `ddp_backend` | `None` | Informational; the process group is created by `setup_distributed`. |
| `find_unused_parameters` | `False` | Mixture-of-experts models whose experts can receive no tokens need `True`. |
| `broadcast_buffers` | `True` | |
| `bucket_cap_mb` | `25` | |
| `gradient_as_bucket_view` | `True` | |
| `static_graph` | `False` | |

The model is wrapped in `DistributedDataParallel` only when a process group exists; otherwise it trains like `Trainer`. Gradient all-reduce is skipped on all but the last micro-batch of a step when the dataloader has a length.

### `olm.train.FSDPTrainer`

<!-- defaults-inherit: olm.train.Trainer -->
Takes every `Trainer` argument with the same default, plus:

| Argument | Default | Notes |
|---|---|---|
| `sharding_strategy` | `"FULL_SHARD"` | |
| `auto_wrap_policy` | `"size"` | `"transformer"` needs `transformer_layer_cls`; `None` wraps nothing. |
| `min_num_params` | `100_000_000` | For the `"size"` policy. |
| `transformer_layer_cls` | `None` | |
| `cpu_offload` | `False` | |
| `backward_prefetch` | `"BACKWARD_PRE"` | |
| `mixed_precision_policy` | `None` | `"bf16"` or `"fp16"` sets parameter, reduction and buffer dtypes. |
| `limit_all_gathers` | `True` | |
| `use_orig_params` | `True` | |

### `olm.train.FSDPTrainer.save_checkpoint`

| Argument | Default | Notes |
|---|---|---|
| `path` | required | |
| `state_dict_type` | `"FULL_STATE_DICT"` | Full weights on rank 0, no optimizer state. `"LOCAL_STATE_DICT"` and `"SHARDED_STATE_DICT"` write one file per rank. |

### `olm.train.AutoTrainer`

<!-- defaults-inherit: olm.train.Trainer -->
Takes every `Trainer` argument with the same default, except the two below, plus the options that follow.

| Argument | Default | Notes |
|---|---|---|
| `device` | `"auto"` | Also `"cuda:auto"`, `"cpu:auto"`, `"cpu"`, `"cuda"`, `"cuda:N"` or a `DeviceConfig`. |
| `context_length` | `1024` | |
| `preset` | `"balanced"` | `"memory_efficient"`, `"speed"`, `"conservative"` (same as `"balanced"`). |
| `force_strategy` | `None` | A `TrainerStrategy` that skips the automatic choice. |
| `verbose` | `True` | Prints the chosen strategy for `"auto"` or a `DeviceConfig`. |
| `ddp_find_unused_parameters` | `False` | |
| `ddp_broadcast_buffers` | `True` | |
| `ddp_bucket_cap_mb` | `25` | |
| `fsdp_min_num_params` | `100_000_000` | |
| `fsdp_transformer_layer_cls` | `None` | |
| `fsdp_backward_prefetch` | `"BACKWARD_PRE"` | |
| `fsdp_limit_all_gathers` | `True` | |
| `fsdp_use_orig_params` | `True` | |

## Callbacks

### `olm.train.CheckpointCallback`

| Argument | Default | Notes |
|---|---|---|
| `checkpoint_dir` | `"checkpoints"` | Created when the callback is built. |
| `save_every` | `1000` | Optimizer steps between `step_<N>.pt` files. |
| `keep_last_n` | `5` | Older step files are deleted; `best_model.pt` never is. |
| `save_best` | `True` | Writes `best_model.pt` when a validation improves `monitor`. |
| `monitor` | `"val_loss"` | |
| `mode` | `"min"` | |
| `save_on_train_end` | `True` | Saves the last step unless it was just saved. |

### `olm.train.EarlyStoppingCallback`

| Argument | Default | Notes |
|---|---|---|
| `patience` | `5` | Validations, not optimizer steps. |
| `min_delta` | `0.0` | |
| `monitor` | `"val_loss"` | |
| `mode` | `"min"` | |

### `olm.train.ValidationCallback`

| Argument | Default | Notes |
|---|---|---|
| `val_dataloader` | required | |
| `eval_every` | `500` | Optimizer steps. |
| `device` | `None` | The trainer's device. |
| `use_amp` | `True` | fp16 autocast, on CUDA only. |

### `olm.train.LRMonitorCallback`

| Argument | Default | Notes |
|---|---|---|
| `log_every` | `100` | Prints the first parameter group's learning rate. |

### `olm.train.MetricsLoggerCallback`

| Argument | Default | Notes |
|---|---|---|
| `log_dir` | `"logs"` | Writes `metrics_<timestamp>.jsonl` with `step`, `epoch`, `train_loss`, `learning_rate` and `val_loss` when present. |
| `log_every` | `10` | |

### `olm.train.ThroughputCallback`

| Argument | Default | Notes |
|---|---|---|
| `log_every` | `100` | |
| `context_length` | `None` | The trainer's `context_length`, used for samples per second. |
| `batch_size` | `None` | Deprecated and ignored. |

### `olm.logging.WandBCallback`

| Argument | Default | Notes |
|---|---|---|
| `project` | required | |
| `entity` | `None` | |
| `name` | `None` | |
| `tags` | `None` | |
| `notes` | `None` | |
| `config` | `None` | |
| `log_frequency` | `1` | Optimizer steps between logs. System metrics are logged every `10 × log_frequency` steps. |
| `log_gradients` | `False` | Only used with `watch_model=True`. |
| `log_model` | `False` | Uploads a version 1 checkpoint as an artifact at the end of every epoch. |
| `watch_model` | `False` | |
| `watch_freq` | `1000` | |
| `log_predictions` | `False` | |
| `log_system_metrics` | `True` | |
| `alert_thresholds` | `None` | |
| `offline` | `False` | |
| `resume` | `None` | Weights & Biases run resumption, separate from `Trainer(resume_from=...)`. |
| `group` | `None` | |
| `job_type` | `"train"` | |
| `save_code` | `True` | |
| `reinit` | `True` | |

Only rank 0 logs.

## Optimizers

### `olm.train.AdamW`

| Argument | Default | Notes |
|---|---|---|
| `params` | required | |
| `lr` | `1e-3` | |
| `betas` | `(0.9, 0.999)` | Many LLM recipes use `(0.9, 0.95)`. |
| `eps` | `1e-8` | |
| `weight_decay` | `0.01` | Replaced by the Trainer's `weight_decay` when the Trainer builds the optimizer. |
| `amsgrad` | `False` | |
| `maximize` | `False` | |
| `fused` | `None` | PyTorch chooses. |

### `olm.train.Lion`

| Argument | Default | Notes |
|---|---|---|
| `params` | required | |
| `lr` | `1e-4` | |
| `betas` | `(0.9, 0.99)` | |
| `weight_decay` | `0.0` | |
| `use_triton` | `False` | Stored; there is no Triton path. |

### `olm.train.optim.zero.ZeROOptimizer`

| Argument | Default | Notes |
|---|---|---|
| `optimizer` | required | |
| `partition_optimizer_states` | `True` | |
| `overlap_communication` | `True` | Not used. |
| `world_size` | `None` | From the process group. |
| `rank` | `None` | From the process group. |

## Schedulers

All schedulers step once per optimizer step.

### `olm.train.WarmupCosineScheduler`

| Argument | Default | Notes |
|---|---|---|
| `optimizer` | required | |
| `warmup_steps` | required | The learning rate at step `s < warmup_steps` is `lr × (s + 1) / warmup_steps`. |
| `total_steps` | required | |
| `min_lr` | `0` | |
| `last_epoch` | `-1` | |

### `olm.train.WarmupLR`

| Argument | Default | Notes |
|---|---|---|
| `optimizer` | required | |
| `warmup_steps` | required | Constant after warmup. |
| `start_lr` | `0` | |
| `last_epoch` | `-1` | |

### `olm.train.CosineAnnealingLR`

| Argument | Default | Notes |
|---|---|---|
| `optimizer` | required | |
| `T_max` | required | |
| `eta_min` | `0` | |
| `last_epoch` | `-1` | |

### `olm.train.LinearLR`

| Argument | Default | Notes |
|---|---|---|
| `optimizer` | required | |
| `total_steps` | required | |
| `end_lr` | `0` | |
| `start_factor` | `1.0` | |
| `last_epoch` | `-1` | |

### `olm.train.LinearDecayLR`

| Argument | Default | Notes |
|---|---|---|
| `optimizer` | required | |
| `total_steps` | required | Decays to 0. |
| `last_epoch` | `-1` | |

## Losses

### `olm.train.CrossEntropyLoss`

| Argument | Default | Notes |
|---|---|---|
| `reduction` | `"mean"` | Stored but not used: the loss is always the mean over positions whose label is not -100. |

### `olm.train.MaskedCELoss`

| Argument | Default | Notes |
|---|---|---|
| `ignore_index` | `-100` | |
| `reduction` | `"mean"` | `"sum"` or `"none"` also work. The mask argument of `forward` is optional. |

### `olm.train.KLLoss`

| Argument | Default | Notes |
|---|---|---|
| `kl_coeff` | `1.0` | |
| `from_logits` | `True` | |
| `**kwargs` | — | Passed to `LossBase`: `reduction="mean"`. |

### `olm.train.ZLoss`

| Argument | Default | Notes |
|---|---|---|
| `z_coeff` | `1e-4` | |
| `**kwargs` | — | Passed to `LossBase`: `reduction="mean"`. |

### `olm.train.losses.load_balance.LoadBalanceLoss`

| Argument | Default | Notes |
|---|---|---|
| `num_experts` | required | |
| `top_k` | `2` | |
| `coefficient` | `0.01` | |

### `olm.train.losses.sequence_load_balance.SequenceLoadBalanceLoss`

| Argument | Default | Notes |
|---|---|---|
| `num_experts` | required | |
| `top_k` | `2` | |
| `coefficient` | `0.001` | |
| `scoring_func` | `"softmax"` | |

### `olm.train.losses.mtp.MTPLoss`

| Argument | Default | Notes |
|---|---|---|
| `num_heads` | `3` | |
| `head_weights` | `None` | `1 / num_heads` for every head. |
| `coefficient` | `1.0` | |

## Data

### `olm.data.datasets.DataLoader`

| Argument | Default | Notes |
|---|---|---|
| `dataset` | required | |
| `batch_size` | `8` | |
| `shuffle` | `None` | `True` for map-style datasets, `False` for iterable ones. |
| `num_workers` | `0` | |
| `pin_memory` | `True` | |
| `drop_last` | `False` | |
| `persistent_workers` | `None` | `True` when `num_workers > 0`. |
| `prefetch_factor` | `2` | Ignored when `num_workers` is 0. |
| `collate_fn` | `None` | |
| `distributed` | `False` | Adds a `DistributedSampler` for map-style datasets when a process group exists. |
| `rank` | `None` | From the process group. |
| `world_size` | `None` | From the process group. |
| `sampler` | `None` | |
| `**kwargs` | — | Passed to `torch.utils.data.DataLoader`, for example `generator` and `worker_init_fn`. |

### `olm.data.datasets.BaseTextDataset`

| Argument | Default | Notes |
|---|---|---|
| `tokenizer` | required | |
| `context_length` | required | |
| `skip_batches` | `0` | Counts samples, separately for each worker and rank shard. The Trainer does not need it to resume. |
| `shuffle` | `False` | |
| `seed` | `42` | |

### `olm.data.datasets.LocalTextDataset`

| Argument | Default | Notes |
|---|---|---|
| `location` | required | |
| `tokenizer` | required | |
| `context_length` | required | |
| `skip_batches` | `0` | |
| `shuffle` | `False` | |
| `seed` | `42` | |

### `olm.data.datasets.HuggingFaceTextDataset`

| Argument | Default | Notes |
|---|---|---|
| `dataset_name` | required | |
| `split` | required | |
| `context_length` | required | |
| `text_fn` | required | |
| `tokenizer` | required | |
| `dataset_kwargs` | `None` | |
| `streaming` | `True` | |
| `skip_batches` | `0` | |
| `shuffle` | `False` | |
| `seed` | `42` | |
| `shuffle_buffer_size` | `10_000` | |

### `olm.data.datasets.FineWebEduDataset`

| Argument | Default | Notes |
|---|---|---|
| `tokenizer` | required | |
| `split` | `"train"` | |
| `context_length` | `1024` | |
| `subset` | `"sample-10BT"` | |
| `streaming` | `True` | |
| `shuffle` | `False` | |
| `seed` | `42` | |
| `cache_dir` | `None` | |
| `skip_batches` | `0` | |

## Reproducibility, checkpoints and processes

### `olm.core.seed_everything`

| Argument | Default | Notes |
|---|---|---|
| `seed` | required | |
| `deterministic` | `False` | `True` also turns on deterministic algorithms and the cuDNN and cuBLAS settings. |
| `warn_only` | `True` | Operations without a deterministic version warn instead of raising. |

### `olm.train.read_checkpoint`

| Argument | Default | Notes |
|---|---|---|
| `path` | required | |
| `map_location` | `"cpu"` | |
| `weights_only` | `True` | |

### `olm.core.dist.setup_distributed`

| Argument | Default | Notes |
|---|---|---|
| `backend` | `None` | `"nccl"` when CUDA is available, else `"gloo"`. |
| `init_method` | `"env://"` | Sets `MASTER_ADDR=localhost` and `MASTER_PORT=29500` when they are unset. |
| `timeout_minutes` | `30` | |

It does nothing when `WORLD_SIZE` is 1 or a process group already exists.
