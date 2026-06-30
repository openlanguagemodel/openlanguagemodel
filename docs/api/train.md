# Training API

Trainers, callbacks, optimizers, schedules, and device selection.

## Modules

| Module | Public API |
|---|---|
| [`olm.train`](../generated/olm.train.md) | `AdamW`, `CheckpointCallback`, `CosineAnnealingLR`, `CrossEntropyLoss`, `DDPTrainer`, `DeviceConfig`, `EarlyStoppingCallback`, `FSDPTrainer`, +25 more |
| [`olm.train.callbacks`](../generated/olm.train.callbacks.md) | `CheckpointCallback`, `EarlyStoppingCallback`, `LRMonitorCallback`, `MetricsLoggerCallback`, `ThroughputCallback`, `ValidationCallback` |
| [`olm.train.callbacks.checkpoint_cb`](../generated/olm.train.callbacks.checkpoint_cb.md) | `CheckpointCallback` |
| [`olm.train.callbacks.early_stopping_cb`](../generated/olm.train.callbacks.early_stopping_cb.md) | `EarlyStoppingCallback` |
| [`olm.train.callbacks.lr_monitor_cb`](../generated/olm.train.callbacks.lr_monitor_cb.md) | `LRMonitorCallback` |
| [`olm.train.callbacks.metrics_logger_cb`](../generated/olm.train.callbacks.metrics_logger_cb.md) | `MetricsLoggerCallback` |
| [`olm.train.callbacks.throughput_cb`](../generated/olm.train.callbacks.throughput_cb.md) | `ThroughputCallback` |
| [`olm.train.callbacks.validation_cb`](../generated/olm.train.callbacks.validation_cb.md) | `ValidationCallback` |
| [`olm.train.device`](../generated/olm.train.device.md) | `DeviceConfig`, `TrainerStrategy`, `detect_devices`, `determine_strategy`, `estimate_model_size`, `parse_device_string`, `print_strategy_summary` |
| [`olm.train.losses`](../generated/olm.train.losses.md) | `CrossEntropyLoss`, `KLLoss`, `LossBase`, `MaskedCELoss`, `ZLoss` |
| [`olm.train.losses.base`](../generated/olm.train.losses.base.md) | `LossBase` |
| [`olm.train.losses.cross_entropy`](../generated/olm.train.losses.cross_entropy.md) | `CrossEntropyLoss` |
| [`olm.train.losses.kllloss`](../generated/olm.train.losses.kllloss.md) | `KLLoss` |
| [`olm.train.losses.mce`](../generated/olm.train.losses.mce.md) | `MaskedCELoss` |
| [`olm.train.losses.zloss`](../generated/olm.train.losses.zloss.md) | `ZLoss` |
| [`olm.train.optim`](../generated/olm.train.optim.md) | `AdamW`, `Lion`, `OptimizerBase`, `ZeROOptimizer` |
| [`olm.train.optim.adamw`](../generated/olm.train.optim.adamw.md) | `AdamW` |
| [`olm.train.optim.base`](../generated/olm.train.optim.base.md) | `OptimizerBase` |
| [`olm.train.optim.lion`](../generated/olm.train.optim.lion.md) | `Lion` |
| [`olm.train.optim.zero`](../generated/olm.train.optim.zero.md) | `ZeROOptimizer` |
| [`olm.train.schedulers`](../generated/olm.train.schedulers.md) | `CosineAnnealingLR`, `LinearDecayLR`, `LinearLR`, `SchedulerBase`, `WarmupCosineScheduler`, `WarmupLR` |
| [`olm.train.schedulers.base`](../generated/olm.train.schedulers.base.md) | `SchedulerBase` |
| [`olm.train.schedulers.cosine`](../generated/olm.train.schedulers.cosine.md) | `CosineAnnealingLR` |
| [`olm.train.schedulers.linear`](../generated/olm.train.schedulers.linear.md) | `LinearDecayLR`, `LinearLR` |
| [`olm.train.schedulers.warmup`](../generated/olm.train.schedulers.warmup.md) | `WarmupCosineScheduler`, `WarmupLR` |
| [`olm.train.trainer`](../generated/olm.train.trainer.md) | `CheckpointCallback`, `DDPTrainer`, `EarlyStoppingCallback`, `FSDPTrainer`, `LRMonitorCallback`, `MetricsLoggerCallback`, `ThroughputCallback`, `Trainer`, +4 more |
| [`olm.train.trainer.auto_trainer`](../generated/olm.train.trainer.auto_trainer.md) | `AutoTrainer`, `auto_trainer` |
| [`olm.train.trainer.ddp_trainer`](../generated/olm.train.trainer.ddp_trainer.md) | `DDPTrainer` |
| [`olm.train.trainer.fsdp_trainer`](../generated/olm.train.trainer.fsdp_trainer.md) | `FSDPTrainer` |
| [`olm.train.trainer.trainer`](../generated/olm.train.trainer.trainer.md) | `Trainer`, `TrainerCallback` |
